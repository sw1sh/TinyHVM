// metal/jit.m — JIT capture/replay for training loops
// Records GPU dispatch sequence on step 0, replays for steps 1+.
// JITState, JITCmd, JITSlot defined in init.m.
int jit_debug_replay = 0;
static u32 jit_orig_buf_ids[JIT_MAX_SLOTS]; // capture-time buf_ids (for tensor update)
static u64 jit_slot_offsets[JIT_MAX_SLOTS]; // slot → byte offset in unified buffer
static u64 jit_slot_sizes[JIT_MAX_SLOTS];   // slot → region size
static u32 jit_n_plan;                       // number of plan indices
static u64 jit_plan_sizes[JIT_MAX_SLOTS];    // plan_idx → size (for default path)
static u32 jit_plan_bufs[JIT_MAX_SLOTS];     // plan_idx → buf_id (for zeroing)
static u32 jit_unified_buf;                  // unified buffer's buf_id

static u32 jit_slot_for_buf(u32 buf_id) {
    for (u32 i = 0; i < jit.n_slots; i++)
        if (jit.slots[i].buf_id == buf_id) return i;
    assert(jit.n_slots < JIT_MAX_SLOTS);
    u32 s = jit.n_slots++;
    jit.slots[s].buf_id = buf_id;
    jit.slots[s].alloc_size = metal_pool.sizes[buf_id];
    jit.slots[s].persistent = 0;
    return s;
}

void jit_begin_capture(u32 n_persistent_hint) {
    (void)n_persistent_hint;
    memset(&jit, 0, sizeof(jit));
    jit.state = JIT_CAPTURE;
    u32 n_bufs = metal_pool.count;
    jit.persistent_count = n_bufs;
    for (u32 i = 0; i < n_bufs && i < JIT_MAX_SLOTS; i++) {
        jit.slots[i].buf_id = i;
        jit.slots[i].alloc_size = metal_pool.sizes[i];
        jit.slots[i].persistent = 1;
    }
    jit.n_slots = n_bufs;
}

// Record dispatch using buf_ids directly (avoids MTLBuffer pointer aliasing)
static void jit_record_dispatch_ids(id<MTLComputePipelineState> pipe,
                                      u32 *buf_ids, u32 n_bufs,
                                      const void **params, u64 *param_sizes, u32 n_params,
                                      u32 gw, u32 gh, u32 gd, u32 tw, u32 th, u32 td) {
    if (jit.n_cmds >= JIT_MAX_CMDS) return;
    JITCmd *cmd = &jit.cmds[jit.n_cmds++];
    memset(cmd, 0, sizeof(*cmd));
    cmd->pipe = pipe;
    cmd->n_bufs = n_bufs > 24 ? 24 : n_bufs;
    cmd->is_mps = 0;
    for (u32 i = 0; i < cmd->n_bufs; i++)
        cmd->buf_slots[i] = jit_slot_for_buf(buf_ids[i]);
    u32 offset = 0;
    cmd->n_params = n_params;
    for (u32 i = 0; i < n_params && params; i++) {
        u32 sz = (u32)param_sizes[i];
        assert(offset + sz <= JIT_MAX_PARAMS);
        memcpy(cmd->params + offset, params[i], sz);
        cmd->param_sizes[i] = sz;
        offset += sz;
    }
    cmd->grid[0] = gw; cmd->grid[1] = gh; cmd->grid[2] = gd;
    cmd->tg[0] = tw; cmd->tg[1] = th; cmd->tg[2] = td;
}

// Legacy: record using MTLBuffer pointers (searches pool backwards for buf_ids)
static void jit_record_dispatch_1d(id<MTLComputePipelineState> pipe,
                                     id<MTLBuffer> *bufs, u32 n_bufs,
                                     const void **params, u64 *param_sizes, u32 n_params,
                                     u32 gw, u32 gh, u32 gd, u32 tw, u32 th, u32 td) {
    u32 ids[24];
    u32 nb = n_bufs > 24 ? 24 : n_bufs;
    for (u32 i = 0; i < nb; i++) {
        ids[i] = 0;
        for (u32 b = metal_pool.count; b > 0; b--) {
            if (metal_pool.bufs[b-1] == bufs[i]) { ids[i] = b-1; break; }
        }
    }
    jit_record_dispatch_ids(pipe, ids, n_bufs, params, param_sizes, n_params,
                            gw, gh, gd, tw, th, td);
}

static void jit_record_mps(u32 dst_buf, u32 a_buf, u32 b_buf, u32 M, u32 K, u32 N,
                            BOOL trans_a, BOOL trans_b,
                            u32 phys_a_rows, u32 phys_a_cols, u32 phys_b_rows, u32 phys_b_cols) {
    if (jit.n_cmds >= JIT_MAX_CMDS) return;
    JITCmd *cmd = &jit.cmds[jit.n_cmds++];
    memset(cmd, 0, sizeof(*cmd));
    cmd->is_mps = 1;
    cmd->mps_m = M; cmd->mps_k = K; cmd->mps_n = N;
    cmd->mps_dst_slot = jit_slot_for_buf(dst_buf);
    cmd->mps_a_slot = jit_slot_for_buf(a_buf);
    cmd->mps_b_slot = jit_slot_for_buf(b_buf);
    cmd->mps_trans_a = trans_a; cmd->mps_trans_b = trans_b;
    cmd->mps_phys_a_rows = phys_a_rows; cmd->mps_phys_a_cols = phys_a_cols;
    cmd->mps_phys_b_rows = phys_b_rows; cmd->mps_phys_b_cols = phys_b_cols;
}

void jit_end_capture(void) {
    jit.state = JIT_OFF;
    if (batch_dirty) metal_flush();

    // Detect GPU-written slots and snapshot CPU-written constants
    u8 gpu_written[JIT_MAX_SLOTS]; memset(gpu_written, 0, jit.n_slots);
    for (u32 ci = 0; ci < jit.n_cmds; ci++) {
        JITCmd *cmd = &jit.cmds[ci];
        if (cmd->is_blit) {
            gpu_written[cmd->blit_dst_slot] = 1;
        } else if (cmd->is_mps) {
            gpu_written[cmd->mps_dst_slot] = 1;
        } else if (cmd->n_bufs > 0) {
            gpu_written[cmd->buf_slots[0]] = 1;
        }
    }

    // Snapshot CPU-written ephemeral buffer contents (GPU idle after flush).
    jit.n_consts = 0;
    for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
        if (gpu_written[s]) continue;
        u64 sz = jit.slots[s].alloc_size;
        if (sz > 0 && sz <= JIT_CONST_MAX_BYTES && jit.n_consts < JIT_MAX_CONST) {
            u32 bid = jit.slots[s].buf_id;
            JITConst *c = &jit.consts[jit.n_consts++];
            c->slot = s;
            c->size = (u32)sz;
            memcpy(c->data, BUF_CONTENTS(bid), sz);
        }
    }

    // ── Memory Planner ─────────────────────────────────────────
    u32 n_eph = jit.n_slots - jit.persistent_count;
    if (n_eph > 0) {
        u32 first_use[JIT_MAX_SLOTS], last_use[JIT_MAX_SLOTS];
        // last_use_is_read: 1 if the slot's last use is a READ (input), not a WRITE (output).
        // Such slots can be released at last_use (not last_use+1) since the GPU
        // reads the buffer completely before writing the output in the same dispatch.
        u8 last_use_is_read[JIT_MAX_SLOTS];
        for (u32 s = 0; s < jit.n_slots; s++) { first_use[s] = UINT32_MAX; last_use[s] = 0; last_use_is_read[s] = 1; }

        for (u32 ci = 0; ci < jit.n_cmds; ci++) {
            JITCmd *cmd = &jit.cmds[ci];
            if (cmd->is_blit) {
                u32 slots[] = { cmd->blit_dst_slot, cmd->blit_src_slot };
                for (int j = 0; j < 2; j++) {
                    if (first_use[slots[j]] == UINT32_MAX) first_use[slots[j]] = ci;
                    if (ci >= last_use[slots[j]]) {
                        last_use[slots[j]] = ci;
                        last_use_is_read[slots[j]] = (j > 0); // j=0 is dst (write), j>0 is src (read)
                    }
                }
            } else if (cmd->is_mps) {
                u32 slots[] = { cmd->mps_dst_slot, cmd->mps_a_slot, cmd->mps_b_slot };
                for (int j = 0; j < 3; j++) {
                    if (first_use[slots[j]] == UINT32_MAX) first_use[slots[j]] = ci;
                    if (ci >= last_use[slots[j]]) {
                        last_use[slots[j]] = ci;
                        last_use_is_read[slots[j]] = (j > 0);
                    }
                }
            } else {
                for (u32 j = 0; j < cmd->n_bufs; j++) {
                    u32 s = cmd->buf_slots[j];
                    if (first_use[s] == UINT32_MAX) first_use[s] = ci;
                    if (ci >= last_use[s]) {
                        last_use[s] = ci;
                        last_use_is_read[s] = (j > 0); // j=0 is output (write), j>0 is input (read)
                    }
                }
            }
        }

        // Const slots: alive for entire replay (GPU reads at any point)
        for (u32 ci = 0; ci < jit.n_consts; ci++) {
            first_use[jit.consts[ci].slot] = 0;
            if (last_use[jit.consts[ci].slot] < jit.n_cmds)
                last_use[jit.consts[ci].slot] = jit.n_cmds; // keep alive through all cmds
        }
        // CPU-written non-const slots: also alive for entire replay
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
            if (!gpu_written[s] && first_use[s] != UINT32_MAX) {
                first_use[s] = 0;
                if (last_use[s] < jit.n_cmds)
                    last_use[s] = jit.n_cmds;
            }
        }


        // Greedy planner: assign plan buffer indices to ephemeral slots.
        // Slots with non-overlapping lifetimes share the same physical buffer.
        // No size constraint — plan buffer sized to MAX slot in each group.
        // Optimization: slots whose last use is a READ can be released at last_use
        // (not last_use+1), enabling same-command reuse by the output slot.
        struct { u64 size; u32 plan_idx; } fpool[2048];
        u32 fpool_n = 0;
        u32 next_plan = 0;
        u32 plan_assignment[JIT_MAX_SLOTS]; // current state (may be released)
        u32 slot_plan[JIT_MAX_SLOTS];       // saved plan_idx (never overwritten)
        u64 plan_buf_size[JIT_MAX_SLOTS];
        memset(plan_assignment, 0xFF, sizeof(plan_assignment));
        memset(slot_plan, 0xFF, sizeof(slot_plan));

        u64 old_mem = 0;
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++)
            old_mem += jit.slots[s].alloc_size;

        for (u32 ci = 0; ci <= jit.n_cmds; ci++) {
            // Release slots: standard release at last_use+1, but read-only slots
            // release at last_use (read completes before write in same dispatch).
            for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
                u32 threshold = last_use[s] + 1; // always release AFTER last use
                if (plan_assignment[s] != 0xFFFFFFFF && plan_assignment[s] != 0xFFFFFFFE && ci >= threshold) {
                    if (fpool_n < 2048) {
                        fpool[fpool_n].size = plan_buf_size[plan_assignment[s]];
                        fpool[fpool_n].plan_idx = plan_assignment[s];
                        fpool_n++;
                    }
                    plan_assignment[s] = 0xFFFFFFFE;
                }
            }
            if (ci == jit.n_cmds) break;

            // Assign ALL slots with first_use == ci (including forced first_use=0)
            for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
                if (plan_assignment[s] != 0xFFFFFFFF) continue;
                if (first_use[s] != ci) continue;
                u64 sz = jit.slots[s].alloc_size;
                if (sz == 0) sz = 4;

                // Find best-fit free buffer (no size constraint, just >= sz)
                int best = -1;
                u64 best_size = UINT64_MAX;
                for (u32 f = 0; f < fpool_n; f++) {
                    if (fpool[f].size >= sz && fpool[f].size < best_size) {
                        best = (int)f;
                        best_size = fpool[f].size;
                    }
                }
                if (best >= 0) {
                    plan_assignment[s] = fpool[best].plan_idx;
                    slot_plan[s] = fpool[best].plan_idx;
                    if (sz > plan_buf_size[fpool[best].plan_idx])
                        plan_buf_size[fpool[best].plan_idx] = sz;
                    fpool[best] = fpool[--fpool_n];
                } else {
                    plan_assignment[s] = next_plan;
                    slot_plan[s] = next_plan;
                    plan_buf_size[next_plan] = sz;
                    next_plan++;
                }
            }
        }

        u64 new_mem = 0;
        for (u32 p = 0; p < next_plan; p++) new_mem += plan_buf_size[p];

        // Save original buf_ids before overwriting with plan_assignment
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++)
            jit_orig_buf_ids[s] = jit.slots[s].buf_id;
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
            u32 p = slot_plan[s];
            if (p < next_plan) {
                jit.slots[s].buf_id = p;
            } else {
                // Unreferenced slot — assign a fresh plan entry
                jit.slots[s].buf_id = next_plan;
                u64 sz = jit.slots[s].alloc_size;
                if (sz == 0) sz = 4;
                plan_buf_size[next_plan] = sz;
                next_plan++;
            }
        }

        // Compute per-plan-idx offsets (contiguously packed)
        jit_n_plan = next_plan;
        u64 plan_offsets[JIT_MAX_SLOTS];
        u64 heap_off = 0;
        for (u32 p = 0; p < next_plan; p++) {
            plan_offsets[p] = heap_off;
            heap_off += (plan_buf_size[p] + 4095ULL) & ~4095ULL;
        }
        fprintf(stderr, "  Plan offsets: %u plans, heap_off=%.2f MB\n",
            next_plan, (double)heap_off/1048576.0);
        // Store per-SLOT offset and size (for unified path)
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
            u32 p = slot_plan[s]; // saved plan_idx (never overwritten by release)
            if (p < next_plan) {
                jit_slot_offsets[s] = plan_offsets[p];
                jit_slot_sizes[s] = plan_buf_size[p];
            } else {
                jit_slot_offsets[s] = 0;
                jit_slot_sizes[s] = jit.slots[s].alloc_size;
            }
        }
        // Debug
        u64 max_slot_off = 0;
        u32 n_assigned = 0, n_released = 0;
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
            if (jit_slot_offsets[s] > max_slot_off) max_slot_off = jit_slot_offsets[s];
            u32 p2 = plan_assignment[s];
            if (p2 < next_plan) n_assigned++;
            else n_released++;
        }
        fprintf(stderr, "  Slots: %u assigned, %u released. Max offset=%.2f MB\n",
            n_assigned, n_released, (double)max_slot_off/1048576.0);
        // Store per-plan-idx sizes (for default path)
        for (u32 p = 0; p < next_plan; p++)
            jit_plan_sizes[p] = plan_buf_size[p];

        fprintf(stderr, "JIT captured %u cmds, %u slots (%u persistent), %u consts\n",
                jit.n_cmds, jit.n_slots, jit.persistent_count, jit.n_consts);
        fprintf(stderr, "JIT memory: %u slots -> %u bufs, %.2f MB (was %.2f MB)\n",
                n_eph, next_plan, (double)new_mem/1048576.0, (double)old_mem/1048576.0);
    } else {
        fprintf(stderr, "JIT captured %u cmds, %u slots (%u persistent), %u consts\n",
                jit.n_cmds, jit.n_slots, jit.persistent_count, jit.n_consts);
    }


    // ── JIT Dispatch Composition Diagnostic ───────────────────────
    {
        u32 n_compute = 0, n_mps = 0, n_blit = 0, n_barrier = 0;
        for (u32 ci = 0; ci < jit.n_cmds; ci++) {
            JITCmd *cmd = &jit.cmds[ci];
            if (cmd->is_blit) { n_blit++; }
            else if (cmd->is_mps) { n_mps++; }
            else if (cmd->n_bufs == 0) { n_barrier++; }
            else { n_compute++; }
        }
        fprintf(stderr, "\n=== JIT Dispatch Composition (%u total) ===\n", jit.n_cmds);
        fprintf(stderr, "  Compute dispatches: %u\n", n_compute);
        fprintf(stderr, "  MPS matmul:         %u\n", n_mps);
        fprintf(stderr, "  Blit copies:        %u\n", n_blit);
        fprintf(stderr, "  Barriers:           %u\n", n_barrier);

        // Grid size histogram for compute dispatches
        // Buckets: <256, <1K, <4K, <16K, <64K, <256K, <1M, <4M, <16M, >=16M
        u32 grid_buckets[10] = {0};
        const char *grid_labels[] = {"<256","<1K","<4K","<16K","<64K","<256K","<1M","<4M","<16M",">=16M"};
        u64 grid_thresholds[] = {256,1024,4096,16384,65536,262144,1048576,4194304,16777216,UINT64_MAX};
        for (u32 ci = 0; ci < jit.n_cmds; ci++) {
            JITCmd *cmd = &jit.cmds[ci];
            if (cmd->is_blit || cmd->is_mps || cmd->n_bufs == 0) continue;
            u64 total = (u64)cmd->grid[0] * cmd->grid[1] * cmd->grid[2];
            for (int b = 0; b < 10; b++) {
                if (total < grid_thresholds[b]) { grid_buckets[b]++; break; }
            }
        }
        fprintf(stderr, "\n  Compute grid size histogram (total threads):\n");
        for (int b = 0; b < 10; b++) {
            if (grid_buckets[b] > 0)
                fprintf(stderr, "    %-8s: %u\n", grid_labels[b], grid_buckets[b]);
        }

        // Top 5 most common pipeline states by pointer
        struct { void *pipe; u32 count; } pipe_counts[512];
        u32 n_pipes = 0;
        for (u32 ci = 0; ci < jit.n_cmds; ci++) {
            JITCmd *cmd = &jit.cmds[ci];
            if (cmd->is_blit || cmd->is_mps || cmd->n_bufs == 0) continue;
            void *p = (__bridge void *)cmd->pipe;
            int found = -1;
            for (u32 pi = 0; pi < n_pipes; pi++) {
                if (pipe_counts[pi].pipe == p) { found = (int)pi; break; }
            }
            if (found >= 0) { pipe_counts[found].count++; }
            else if (n_pipes < 512) {
                pipe_counts[n_pipes].pipe = p;
                pipe_counts[n_pipes].count = 1;
                n_pipes++;
            }
        }
        // Sort top 5 by count (simple selection sort for top entries)
        fprintf(stderr, "\n  Top pipeline states (compute only):\n");
        for (int rank = 0; rank < 5 && rank < (int)n_pipes; rank++) {
            u32 best = rank;
            for (u32 j = rank + 1; j < n_pipes; j++) {
                if (pipe_counts[j].count > pipe_counts[best].count) best = j;
            }
            if (best != (u32)rank) {
                void *tmp_p = pipe_counts[rank].pipe; u32 tmp_c = pipe_counts[rank].count;
                pipe_counts[rank].pipe = pipe_counts[best].pipe; pipe_counts[rank].count = pipe_counts[best].count;
                pipe_counts[best].pipe = tmp_p; pipe_counts[best].count = tmp_c;
            }
            // Try to get pipeline label
            id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipe_counts[rank].pipe;
            NSString *label = pso.label;
            if (label && label.length > 0)
                fprintf(stderr, "    #%d: %3u dispatches  pipe=%p  label=%s\n",
                    rank+1, pipe_counts[rank].count, pipe_counts[rank].pipe, label.UTF8String);
            else
                fprintf(stderr, "    #%d: %3u dispatches  pipe=%p\n",
                    rank+1, pipe_counts[rank].count, pipe_counts[rank].pipe);
        }
        fprintf(stderr, "  Unique compute pipelines: %u\n", n_pipes);
        fprintf(stderr, "===========================================\n\n");
    }
}

// Resolve capture-time buf_id to replay-time buf_id.
// Call AFTER jit_replay (or after first replay allocates ephemeral buffers).
u32 jit_resolve_buf(u32 capture_buf_id) {
    if (!jit.ephemeral_ready) return capture_buf_id;
    for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
        if (jit_orig_buf_ids[s] == capture_buf_id)
            return jit.slots[s].buf_id;
    }
    return capture_buf_id; // persistent or not found
}

// Update ALL tensor buf_ids for ephemeral slots (call once after first replay)
void jit_update_tensor_bufs(TinyHVM *ctx) {
    if (!jit.ephemeral_ready) return;
    // Loop over ALL tensors (including ephemeral ones beyond tensor_count,
    // which were created during capture but freed by thvm_reset)
    u32 max_tid = ctx->tensor_count;
    for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
        u32 old_bid = jit_orig_buf_ids[s];
        u32 new_bid = jit.slots[s].buf_id;
        if (old_bid == new_bid) continue;
        u32 scan_end = (max_tid + 2000 < MAX_TENSORS) ? max_tid + 2000 : MAX_TENSORS;
        for (u32 t = 0; t < scan_end; t++) {
            if (ctx->tensors[t].buf_id == old_bid)
                ctx->tensors[t].buf_id = new_bid;
        }
    }
}

// Flush previous replay's GPU work — call BEFORE overwriting shared buffers.
void jit_flush(void) {
    if (batch_dirty) metal_flush();
}

// Allocate ephemeral buffers via offset-based suballocation.
// One unified MTLBuffer, each slot at a different offset.
void jit_alloc_ephemeral(void) {
    if (jit.ephemeral_ready) return;
    jit.ephemeral_ready = 1;

    // Compute total unified buffer size from slot offsets
    u64 heap_total = 0;
    for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
        u64 end = jit_slot_offsets[s] + ((jit_slot_sizes[s] + 4095ULL) & ~4095ULL);
        if (end > heap_total) heap_total = end;
    }
    if (heap_total == 0) heap_total = 4;

    // Drain pending_free and release free-list to reclaim capture-time memory
    for (u32 i = 0; i < pending_free_count; i++)
        metal_buf_free(pending_free[i]);
    pending_free_count = 0;
    for (u32 i = 0; i < free_count; i++) {
        if (free_list[i].buf) {
            metal_bytes_total -= free_list[i].size;
            free_list[i].buf = nil;
        }
    }
    free_count = 0;

    if (getenv("UNIFIED")) {
        // Offset suballocation: ONE buffer, per-slot offsets
        fprintf(stderr, "  UNIFIED: allocating %.2f MB heap (%u slots)\n",
            (double)heap_total/1048576.0, jit.n_slots - jit.persistent_count);
        jit_unified_buf = metal_buf_alloc(heap_total);
        id<MTLBuffer> unified_mtl = metal_pool.bufs[jit_unified_buf];
        fprintf(stderr, "  UNIFIED: buf_id=%u mtl=%p len=%llu\n",
            jit_unified_buf, (void*)unified_mtl, (u64)unified_mtl.length);
        fprintf(stderr, "  Creating %u alias buf_ids (pool.count=%u, MAX=%u, pers=%u, n_slots=%u)\n",
            jit.n_slots - jit.persistent_count, metal_pool.count, (u32)MAX_BUFS, jit.persistent_count, jit.n_slots);
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
            u64 sz = jit_slot_sizes[s];
            if (sz == 0) sz = 4;
            assert(metal_pool.count < MAX_BUFS);
            u32 bid = metal_pool.count++;
            metal_pool.bufs[bid] = unified_mtl;
            metal_pool.sizes[bid] = sz;
            buf_offset[bid] = jit_slot_offsets[s];
            buf_refcount[bid] = 0;
            jit.slots[s].buf_id = bid;
        }
        fprintf(stderr, "  Aliases: pool.count=%u\n", metal_pool.count);
        // Verify no offset exceeds buffer length
        u32 n_oob = 0;
        for (u32 s2 = jit.persistent_count; s2 < jit.n_slots; s2++) {
            u32 b2 = jit.slots[s2].buf_id;
            if (b2 && metal_pool.bufs[b2] == unified_mtl) {
                if (buf_offset[b2] + metal_pool.sizes[b2] > unified_mtl.length && n_oob++ < 3)
                    fprintf(stderr, "  OOB: slot=%u bid=%u off=%llu sz=%llu len=%llu\n",
                        s2, b2, (u64)buf_offset[b2], metal_pool.sizes[b2], (u64)unified_mtl.length);
            }
        }
        fprintf(stderr, "  UNIFIED: aliases done, pool.count=%u, oob=%u\n", metal_pool.count, n_oob);
        return;
    }

    // Default: plan-idx based, one buffer per plan_idx
    {
        u32 plan_bufs[JIT_MAX_SLOTS];
        for (u32 p = 0; p < jit_n_plan; p++) {
            plan_bufs[p] = metal_buf_alloc(jit_plan_sizes[p]);
            jit_plan_bufs[p] = plan_bufs[p];
        }

        for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
            u32 p = jit.slots[s].buf_id;
            if (p < jit_n_plan)
                jit.slots[s].buf_id = plan_bufs[p];
            else if (jit.slots[s].alloc_size > 0)
                jit.slots[s].buf_id = metal_buf_alloc(jit.slots[s].alloc_size);
            else
                jit.slots[s].buf_id = 0;
        }
        return;
    }

    // Allocate ONE unified buffer
    jit_unified_buf = metal_buf_alloc(heap_total);
    id<MTLBuffer> unified_mtl = metal_pool.bufs[jit_unified_buf];
    fprintf(stderr, "  Unified: requested=%.2f MB, got=%.2f MB (pool.sizes=%.2f MB)\n",
        (double)heap_total/1048576.0, (double)unified_mtl.length/1048576.0,
        (double)metal_pool.sizes[jit_unified_buf]/1048576.0);

    // Each ephemeral slot gets its own buf_id aliasing the unified buffer
    for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
        u64 sz = jit_slot_sizes[s];
        if (sz == 0) sz = 4;
        assert(metal_pool.count < MAX_BUFS);
        u32 bid = metal_pool.count++;
        metal_pool.bufs[bid] = unified_mtl;
        metal_pool.sizes[bid] = sz;
        buf_offset[bid] = jit_slot_offsets[s];
        buf_refcount[bid] = 0;
        jit.slots[s].buf_id = bid;
    }
}

// Restore JIT constant data (call before jit_replay_commands)
void jit_restore_consts(void) {
    if (!jit.ephemeral_ready) jit_alloc_ephemeral();
    // NOTE: ephemeral buffer zeroing is UNNECESSARY because JIT replay executes
    // the same commands as capture — every buffer is written before read.
    // Only const data (CPU-written scalars) needs restoration.
    // Restore captured const data (overwrites zeros for const slots)
    for (u32 ci = 0; ci < jit.n_consts; ci++) {
        u32 bid = jit.slots[jit.consts[ci].slot].buf_id;
        if (bid && bid < MAX_BUFS && metal_pool.bufs[bid])
            memcpy(BUF_CONTENTS(bid), jit.consts[ci].data, jit.consts[ci].size);
    }
}

// Encode all JIT commands (call after jit_restore_consts + any buffer overrides)
void jit_replay_commands(void) {
    for (u32 ci = 0; ci < jit.n_cmds; ci++) {
        JITCmd *cmd = &jit.cmds[ci];
        // Barrier: flush GPU and start new command buffer (preserves capture ordering)
        if (!cmd->is_blit && !cmd->is_mps && cmd->n_bufs == 0) {
            if (batch_dirty) metal_flush();
            continue;
        }
        if (cmd->is_blit) {
            if (batch_encoder) { [batch_encoder endEncoding]; batch_encoder = nil; }
            if (!batch_cmd) batch_cmd = [mtl_queue commandBuffer];
            u32 dst = jit.slots[cmd->blit_dst_slot].buf_id;
            u32 src = jit.slots[cmd->blit_src_slot].buf_id;
            id<MTLBlitCommandEncoder> blit = [batch_cmd blitCommandEncoder];
            [blit copyFromBuffer:metal_pool.bufs[src] sourceOffset:BUF_OFFSET(src)
                        toBuffer:metal_pool.bufs[dst] destinationOffset:BUF_OFFSET(dst)
                            size:cmd->blit_size];
            [blit endEncoding];
            batch_dirty = 1;
        } else if (cmd->is_mps) {
            if (batch_encoder) { [batch_encoder endEncoding]; batch_encoder = nil; }
            if (!batch_cmd) batch_cmd = [mtl_queue commandBuffer];
            u32 dst = jit.slots[cmd->mps_dst_slot].buf_id;
            u32 a = jit.slots[cmd->mps_a_slot].buf_id;
            u32 b = jit.slots[cmd->mps_b_slot].buf_id;
            MPSMatrixDescriptor *dA = [MPSMatrixDescriptor matrixDescriptorWithRows:cmd->mps_phys_a_rows columns:cmd->mps_phys_a_cols rowBytes:cmd->mps_phys_a_cols*4 dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dB = [MPSMatrixDescriptor matrixDescriptorWithRows:cmd->mps_phys_b_rows columns:cmd->mps_phys_b_cols rowBytes:cmd->mps_phys_b_cols*4 dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dC = [MPSMatrixDescriptor matrixDescriptorWithRows:cmd->mps_m columns:cmd->mps_n rowBytes:cmd->mps_n*4 dataType:MPSDataTypeFloat32];
            MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[a] offset:BUF_OFFSET(a) descriptor:dA];
            MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[b] offset:BUF_OFFSET(b) descriptor:dB];
            MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[dst] offset:BUF_OFFSET(dst) descriptor:dC];
            MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:mtl_dev transposeLeft:cmd->mps_trans_a transposeRight:cmd->mps_trans_b resultRows:cmd->mps_m resultColumns:cmd->mps_n interiorColumns:cmd->mps_k alpha:1.0 beta:0.0];
            [mm encodeToCommandBuffer:batch_cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
            batch_dirty = 1;
        } else {
            // Verify all buffers exist before encoding
            int skip = 0;
            for (u32 i = 0; i < cmd->n_bufs; i++) {
                u32 _bid = jit.slots[cmd->buf_slots[i]].buf_id;
                if (!_bid || _bid >= MAX_BUFS || !metal_pool.bufs[_bid]) {
                    fprintf(stderr, "JIT_REPLAY: nil buf cmd=%u slot=%u bid=%u\n", ci, cmd->buf_slots[i], _bid);
                    skip = 1; break;
                }
            }
            if (skip) continue;
            id<MTLComputeCommandEncoder> enc = get_encoder();
            [enc setComputePipelineState:cmd->pipe];
            for (u32 i = 0; i < cmd->n_bufs; i++) {
                u32 _bid = jit.slots[cmd->buf_slots[i]].buf_id;
                [enc setBuffer:metal_pool.bufs[_bid] offset:BUF_OFFSET(_bid) atIndex:i];
            }
            u32 poff = 0;
            for (u32 i = 0; i < cmd->n_params; i++) {
                [enc setBytes:cmd->params + poff length:cmd->param_sizes[i] atIndex:cmd->n_bufs + i];
                poff += cmd->param_sizes[i];
            }
            if (cmd->grid[1] == 0) {
                // GROUP_REDUCE: dispatchThreadgroups (grid[1]==0 sentinel)
                [enc dispatchThreadgroups:MTLSizeMake(cmd->grid[0], 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(cmd->tg[0], cmd->tg[1], cmd->tg[2])];
            } else {
                [enc dispatchThreads:MTLSizeMake(cmd->grid[0], cmd->grid[1], cmd->grid[2])
                   threadsPerThreadgroup:MTLSizeMake(cmd->tg[0], cmd->tg[1], cmd->tg[2])];
            }
            batch_dirty = 1;
        }
    }
    total_dispatches += jit.n_cmds;
}

// Full replay: restore consts + encode commands
void jit_replay(void) {
    jit_restore_consts();
    jit_replay_commands();
}
