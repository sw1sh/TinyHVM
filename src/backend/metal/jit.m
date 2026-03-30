// metal/jit.m — JIT capture/replay for training loops
// Records GPU dispatch sequence on step 0, replays for steps 1+.
// JITState, JITCmd, JITSlot defined in init.m.
static u32 jit_orig_buf_ids[JIT_MAX_SLOTS]; // capture-time buf_ids (for tensor update)

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

    // Map buf_ids to slots for constants saved during capture.
    u32 n_valid = 0;
    for (u32 ci = 0; ci < jit.n_consts; ci++) {
        u32 bid = jit.consts[ci].slot;
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
            if (jit.slots[s].buf_id == bid) {
                jit.consts[n_valid] = jit.consts[ci];
                jit.consts[n_valid].slot = s;
                n_valid++;
                break;
            }
        }
    }
    jit.n_consts = n_valid;

    // ── Memory Planner ─────────────────────────────────────────
    // Scan the captured dispatch sequence to find first/last use of each
    // ephemeral slot. Slots with non-overlapping lifetimes share buffers.
    // Same algorithm as tinygrad's _internal_memory_planner.
    u32 n_eph = jit.n_slots - jit.persistent_count;
    if (n_eph > 0) {
        u32 first_use[JIT_MAX_SLOTS], last_use[JIT_MAX_SLOTS];
        for (u32 s = 0; s < jit.n_slots; s++) { first_use[s] = UINT32_MAX; last_use[s] = 0; }

        // Scan commands for slot usage
        for (u32 ci = 0; ci < jit.n_cmds; ci++) {
            JITCmd *cmd = &jit.cmds[ci];
            if (cmd->is_blit) {
                u32 slots[] = { cmd->blit_dst_slot, cmd->blit_src_slot };
                for (int j = 0; j < 2; j++) {
                    if (first_use[slots[j]] == UINT32_MAX) first_use[slots[j]] = ci;
                    last_use[slots[j]] = ci;
                }
            } else if (cmd->is_mps) {
                u32 slots[] = { cmd->mps_dst_slot, cmd->mps_a_slot, cmd->mps_b_slot };
                for (int j = 0; j < 3; j++) {
                    if (first_use[slots[j]] == UINT32_MAX) first_use[slots[j]] = ci;
                    last_use[slots[j]] = ci;
                }
            } else {
                for (u32 j = 0; j < cmd->n_bufs; j++) {
                    u32 s = cmd->buf_slots[j];
                    if (first_use[s] == UINT32_MAX) first_use[s] = ci;
                    last_use[s] = ci;
                }
            }
        }

        // Mark const-written slots as first_use=0 (alive from start)
        for (u32 ci = 0; ci < jit.n_consts; ci++) {
            u32 s = jit.consts[ci].slot;
            first_use[s] = 0;
        }

        // Detect "input" slots: read by a command but never written (output)
        // by any command. These are externally written (CPU memset/memcpy
        // between replays) and must be first_use=0 to prevent sharing.
        // Convention: buf_slots[0] is the output, rest are inputs.
        u8 gpu_written[JIT_MAX_SLOTS]; memset(gpu_written, 0, jit.n_slots);
        for (u32 ci = 0; ci < jit.n_cmds; ci++) {
            JITCmd *cmd = &jit.cmds[ci];
            if (cmd->is_blit) {
                gpu_written[cmd->blit_dst_slot] = 1;
            } else if (cmd->is_mps) {
                gpu_written[cmd->mps_dst_slot] = 1;
            } else if (cmd->n_bufs > 0) {
                gpu_written[cmd->buf_slots[0]] = 1; // first buf = output
            }
        }
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
            if (!gpu_written[s] && first_use[s] != UINT32_MAX)
                first_use[s] = 0; // externally written → alive from start
        }

        // Greedy planner: assign physical buffer indices to ephemeral slots.
        // Slots with non-overlapping [first_use, last_use] share the same buffer.
        // jit.slots[s].plan_buf = index into a compacted buffer array.
        // Free pool: (size, plan_idx) entries for reuse.
        struct { u64 size; u32 plan_idx; } fpool[256];
        u32 fpool_n = 0;
        u32 next_plan = 0;  // next physical buffer index
        // Store plan assignment per slot
        u32 plan_assignment[JIT_MAX_SLOTS]; // slot → plan buffer index
        u64 plan_buf_size[JIT_MAX_SLOTS];   // plan buffer → size
        memset(plan_assignment, 0xFF, sizeof(plan_assignment));

        // Process ephemeral slots in first_use order
        // Simple: iterate commands, for each command release dead slots, then assign
        for (u32 ci = 0; ci <= jit.n_cmds; ci++) {
            // Release slots whose last_use < ci
            for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
                if (plan_assignment[s] != 0xFFFFFFFF && last_use[s] < ci) {
                    // Return to free pool
                    if (fpool_n < 256) {
                        fpool[fpool_n].size = jit.slots[s].alloc_size;
                        fpool[fpool_n].plan_idx = plan_assignment[s];
                        fpool_n++;
                    }
                    plan_assignment[s] = 0xFFFFFFFE; // mark as released
                }
            }
            if (ci == jit.n_cmds) break;

            // Assign slots first used at this command
            JITCmd *cmd = &jit.cmds[ci];
            u32 cmd_slots[24]; u32 cmd_ns = 0;
            if (cmd->is_blit) {
                cmd_slots[cmd_ns++] = cmd->blit_dst_slot;
                cmd_slots[cmd_ns++] = cmd->blit_src_slot;
            } else if (cmd->is_mps) {
                cmd_slots[cmd_ns++] = cmd->mps_dst_slot;
                cmd_slots[cmd_ns++] = cmd->mps_a_slot;
                cmd_slots[cmd_ns++] = cmd->mps_b_slot;
            } else {
                for (u32 j = 0; j < cmd->n_bufs; j++)
                    cmd_slots[cmd_ns++] = cmd->buf_slots[j];
            }
            for (u32 j = 0; j < cmd_ns; j++) {
                u32 s = cmd_slots[j];
                if (s < jit.persistent_count) continue; // persistent — skip
                if (plan_assignment[s] != 0xFFFFFFFF) continue; // already assigned
                if (first_use[s] != ci) continue; // not born yet
                u64 sz = jit.slots[s].alloc_size;

                // Find compatible free buffer
                int found = -1;
                for (u32 f = 0; f < fpool_n; f++) {
                    if (fpool[f].size >= sz && fpool[f].size <= sz * 2) {
                        found = (int)f;
                        break;
                    }
                }
                if (found >= 0) {
                    plan_assignment[s] = fpool[found].plan_idx;
                    fpool[found] = fpool[--fpool_n];
                } else {
                    plan_assignment[s] = next_plan;
                    plan_buf_size[next_plan] = sz;
                    next_plan++;
                }
            }
        }

        // Store plan in JIT slots (overwrite buf_id temporarily — resolved at replay)
        u32 n_reused = n_eph - next_plan;
        u64 old_mem = 0, new_mem = 0;
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++)
            old_mem += jit.slots[s].alloc_size;
        for (u32 p = 0; p < next_plan; p++)
            new_mem += plan_buf_size[p];

        // Save original buf_ids before overwriting with plan_assignment
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++)
            jit_orig_buf_ids[s] = jit.slots[s].buf_id;
        // Overload buf_id with plan_assignment for replay
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++)
            jit.slots[s].buf_id = plan_assignment[s];

        fprintf(stderr, "JIT captured %u cmds, %u slots (%u persistent), %u consts\n",
                jit.n_cmds, jit.n_slots, jit.persistent_count, jit.n_consts);
        fprintf(stderr, "JIT memory reduced from %.2f MB -> %.2f MB, %u -> %u bufs\n",
                (double)old_mem/1e6, (double)new_mem/1e6, n_eph, next_plan);
    } else {
        fprintf(stderr, "JIT captured %u cmds, %u slots (%u persistent), %u consts\n",
                jit.n_cmds, jit.n_slots, jit.persistent_count, jit.n_consts);
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

// Allocate ephemeral buffers without executing any commands.
// Call once after jit_end_capture, before any replay.
void jit_alloc_ephemeral(void) {
    if (jit.ephemeral_ready) return;
    jit.ephemeral_ready = 1; // set BEFORE alloc to prevent re-entry
        // Find max plan_idx to know how many physical buffers needed
        u32 max_plan = 0;
        for (u32 i = jit.persistent_count; i < jit.n_slots; i++) {
            u32 p = jit.slots[i].buf_id; // plan_idx from memory planner
            if (p != 0xFFFFFFFF && p != 0xFFFFFFFE && p > max_plan) max_plan = p;
        }
        // Allocate physical buffers for each plan_idx
        u32 plan_bufs[JIT_MAX_SLOTS];
        for (u32 p = 0; p <= max_plan; p++) {
            // Find the largest alloc_size among slots assigned to this plan_idx
            u64 max_sz = 0;
            for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
                if (jit.slots[s].buf_id == p && jit.slots[s].alloc_size > max_sz)
                    max_sz = jit.slots[s].alloc_size;
            }
            if (max_sz == 0) max_sz = 4; // avoid zero-size alloc
            plan_bufs[p] = metal_buf_alloc(max_sz);
        }
        // Resolve slot buf_ids: each slot gets the buf_id of its plan_idx
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
            u32 p = jit.slots[s].buf_id;
            if (p <= max_plan)
                jit.slots[s].buf_id = plan_bufs[p];
            else
                jit.slots[s].buf_id = metal_buf_alloc(jit.slots[s].alloc_size); // fallback
        }
}

void jit_replay(void) {
    if (!jit.ephemeral_ready) jit_alloc_ephemeral();

    // Restore constant data
    for (u32 ci = 0; ci < jit.n_consts; ci++) {
        u32 bid = jit.slots[jit.consts[ci].slot].buf_id;
        memcpy(BUF_CONTENTS(bid), jit.consts[ci].data, jit.consts[ci].size);
    }

    // Encode all commands
    for (u32 ci = 0; ci < jit.n_cmds; ci++) {
        JITCmd *cmd = &jit.cmds[ci];
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
            [enc dispatchThreads:MTLSizeMake(cmd->grid[0], cmd->grid[1], cmd->grid[2])
               threadsPerThreadgroup:MTLSizeMake(cmd->tg[0], cmd->tg[1], cmd->tg[2])];
            batch_dirty = 1;
        }
    }
    total_dispatches += jit.n_cmds;
}
