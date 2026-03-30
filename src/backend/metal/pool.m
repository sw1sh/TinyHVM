// metal/pool.m — Metal buffer pool functions (state in init.m)

// Tracks total Metal buffer bytes currently allocated (pool + free list).
// Only changes on newBufferWithLength (up) and actual buffer release (down).
// Free list reuse doesn't change this — the Metal memory is still committed.
static u64 metal_bytes_total = 0;

// Tracks in-use bytes (excludes free list). This is what the budget guards.
static u64 metal_bytes_inuse = 0;

// Buffer lifetime tracking for mid-step reuse.
// Buffers whose last_use + 2 < dispatch_counter can be safely reused
// (Metal guarantees sequential execution within command buffer).
static u32 buf_last_use[MAX_BUFS];
static u32 dispatch_counter = 0;
// Remaining dispatch uses for each buffer (set by pre-scan, decremented after dispatch).
// When remaining_uses reaches 0, the buffer is truly dead and can be stolen.
static u16 buf_remaining_uses[MAX_BUFS];

// ── Step-Level Memory Planner ──────────────────────────────────
// Learn buffer lifetimes from step 0, plan reuse for step 1+.
// Neural net training repeats the same graph every step.
// Similar to JIT: record on first pass, optimize on replay.
#define PLAN_MIN_BYTES (512 * 1024)  // plan buffers >= 512KB
#define MAX_PLAN_ENTRIES 2048

// Memory plan: maps alloc sequence N → earlier alloc M whose buffer to reuse.
// -1 = no reuse (allocate fresh). Values are indices into the alloc sequence.
static i32 mem_plan_reuse[MAX_PLAN_ENTRIES]; // reuse[N] = M or -1
static u32 mem_plan_count = 0;
static u32 mem_plan_cursor = 0;
static int mem_plan_active = 0;

// Recording: track large alloc sequence + lifetimes during step 0
static u32 plan_alloc_ids[MAX_PLAN_ENTRIES]; // buf_ids in alloc order
static u32 plan_alloc_birth[MAX_PLAN_ENTRIES]; // dispatch_counter at alloc time
static u32 plan_alloc_count = 0;
// Step 1+: buf_ids assigned to each plan entry (for reuse lookup)
static u32 plan_step_bufs[MAX_PLAN_ENTRIES];

#define METAL_MEM_BUDGET (4ULL * 1024 * 1024 * 1024) // 4GB hard limit

static u32 metal_buf_alloc(u64 bytes) {
    bytes = MAX(bytes, 4);

    u32 id = metal_pool.count++;
    if (id >= MAX_BUFS) {
        fprintf(stderr, "FATAL: Metal buffer pool exhausted (%u buffers). Aborting safely.\n", id);
        exit(1);
    }

    // 0. Memory plan: suballocation from a single large MTLBuffer.
    // Each alloc gets a unique offset region — no memory overlap, no GPU conflicts.
    if (0 && mem_plan_active && bytes >= PLAN_MIN_BYTES && mem_plan_cursor < mem_plan_count) {
        u32 c = mem_plan_cursor++;
        i32 reuse = mem_plan_reuse[c];
        if (reuse >= 0 && (u32)reuse < c) {
            // Reuse: take the SAME offset as the dead alloc (different time, same region)
            u32 src_id = plan_step_bufs[reuse];
            if (src_id && metal_pool.bufs[src_id]) {
                metal_pool.bufs[id] = metal_pool.bufs[src_id];
                metal_pool.sizes[id] = metal_pool.sizes[src_id];
                buf_offset[id] = buf_offset[src_id]; // same region, different time
                plan_step_bufs[c] = id;
                goto done;
            }
        }
        plan_step_bufs[c] = 0; // set after normal alloc below
    }

    // 1. Free list (from pool_reset)
    {
        u32 best_idx = UINT32_MAX;
        u64 best_size = UINT64_MAX;
        for (u32 i = 0; i < free_count; i++) {
            u64 sz = free_list[i].size;
            if (sz >= bytes && sz <= bytes * 2 && sz < best_size) {
                best_idx = i;
                best_size = sz;
            }
        }
        if (best_idx != UINT32_MAX) {
            metal_pool.bufs[id] = free_list[best_idx].buf;
            metal_pool.sizes[id] = free_list[best_idx].size;
            free_list[best_idx] = free_list[--free_count];
            metal_bytes_inuse += metal_pool.sizes[id];
            goto done;
        }
    }

    // 2. Fresh allocation
    metal_pool.bufs[id] = [mtl_dev newBufferWithLength:bytes
                                               options:MTLResourceStorageModeShared];
    metal_pool.sizes[id] = bytes;
    metal_bytes_total += bytes;
    metal_bytes_inuse += bytes;

done:
    // Record large allocs for planning
    if (bytes >= PLAN_MIN_BYTES && plan_alloc_count < MAX_PLAN_ENTRIES) {
        u32 pc = plan_alloc_count;
        if (!mem_plan_active) {
            plan_alloc_ids[pc] = id;
            plan_alloc_birth[pc] = dispatch_counter;
        }
        if (mem_plan_active && mem_plan_cursor > 0 && plan_step_bufs[mem_plan_cursor-1] == 0)
            plan_step_bufs[mem_plan_cursor-1] = id;
        plan_alloc_count++;
    }

    // Budget guard — check after allocation so we report accurate numbers
    if (metal_bytes_inuse > METAL_MEM_BUDGET) {
        fprintf(stderr, "FATAL: GPU memory budget exceeded (%.0fMB in-use / %.0fMB budget). Aborting safely.\n",
            (double)metal_bytes_inuse / 1e6, (double)METAL_MEM_BUDGET / 1e6);
        fprintf(stderr, "  Allocation: %.1fMB. Total Metal: %.0fMB. Buffers: %u. Free list: %u.\n",
            (double)bytes / 1e6, (double)metal_bytes_total / 1e6, metal_pool.count, free_count);
        exit(1);
    }

    buf_refcount[id] = 1;
    buf_cpu_only[id] = 0;
    buf_offset[id] = 0; // default: no offset (full buffer)
    thvm_prof_buf_alloc(bytes);
    return id;
}

static void metal_buf_incref(u32 id) {
    if (id == 0) return;
    buf_refcount[id]++;
}

static void metal_buf_decref(u32 id) {
    if (id == 0) return;
    if (buf_refcount[id] == 0) return; // guard against double-decref
    if (--buf_refcount[id] == 0) {
        if (pending_free_count < PENDING_FREE_CAP)
            pending_free[pending_free_count++] = id;
    }
}

static void metal_buf_free(u32 id) {
    u64 sz = metal_pool.sizes[id];
    if (sz <= metal_bytes_inuse) metal_bytes_inuse -= sz;
    else metal_bytes_inuse = 0;
    if (metal_pool.bufs[id] && free_count < MAX_FREE_BUFS) {
        free_list[free_count].buf = metal_pool.bufs[id];
        free_list[free_count].size = sz;
        free_count++;
        // Metal memory stays allocated (in free list) — metal_bytes_total unchanged
    } else if (metal_pool.bufs[id]) {
        // Free list full — actually release Metal buffer
        [metal_pool.bufs[id] release];
        if (sz <= metal_bytes_total) metal_bytes_total -= sz;
        else metal_bytes_total = 0;
    }
    metal_pool.bufs[id] = nil;
    metal_pool.sizes[id] = 0;
}

static void metal_buf_write(u32 id, const void *data, u64 bytes) {
    buf_cpu_only[id] = 1; // CPU wrote this — no GPU sync needed on read
    memcpy(BUF_CONTENTS(id), data, bytes);
    thvm_prof_buf_write(bytes);
    // JIT: save CPU-written constant data for ephemeral buffers.
    // Save by buf_id (slot mapping done later at jit_end_capture).
    if (jit.state == JIT_CAPTURE && bytes <= JIT_CONST_MAX_BYTES &&
        id >= jit.persistent_count && jit.n_consts < JIT_MAX_CONST) {
        // Save FIRST write only (keyed by buf_id via slot field temporarily)
        int already = 0;
        for (u32 ci = 0; ci < jit.n_consts; ci++)
            if (jit.consts[ci].slot == id) { already = 1; break; } // slot = buf_id temporarily
        if (!already) {
            JITConst *c = &jit.consts[jit.n_consts++];
            c->slot = id; // TEMPORARY: buf_id, mapped to slot at end_capture
            c->size = (u32)bytes;
            memcpy(c->data, data, bytes);
        }
    }
}

// Read without flushing — safe for CPU-written metadata that GPU hasn't touched.
void metal_buf_read_nosync(u32 id, void *out, u64 bytes) {
    u64 actual = metal_pool.sizes[id];
    u64 n = bytes < actual ? bytes : actual;
    memcpy(out, BUF_CONTENTS(id), n);
    if (n < bytes) memset((char*)out + n, 0, bytes - n);
}

static u8 buf_cpu_only[MAX_BUFS]; // 1 = only written by CPU, never GPU output

static void metal_buf_read(u32 id, void *out, u64 bytes) {
    if (batch_dirty) {
        if (buf_cpu_only[id]) {
            // CPU-written buffer (axes, shapes, scalars from thvm_tensor).
            // Never modified by GPU dispatches — safe to read without sync.
            metal_buf_read_nosync(id, out, bytes);
            return;
        }
        metal_flush();
    }
    u64 actual = metal_pool.sizes[id];
    if (bytes > actual) {
        // Read what we can, zero the rest (view strides may exceed buffer for broadcasts)
        memcpy(out, BUF_CONTENTS(id), actual);
        memset((char*)out + actual, 0, bytes - actual);
    } else {
        memcpy(out, BUF_CONTENTS(id), bytes);
    }
    thvm_prof_buf_read(bytes);
}

static void metal_pool_reset(u32 keep) {
    if (batch_dirty) metal_flush();
    u32 buf_keep = keep + 1;
    u32 n_free = metal_pool.count - buf_keep;
    if (n_free == 0) goto reset_counters;

    // ── Build index-based reuse plan from this step's profile ──
    // No MTLBuffer pointers stored — just indices. On step 1+, reuse
    // is resolved by looking up the earlier alloc's buf_id at runtime.
    if (plan_alloc_count > 0 && !mem_plan_active) {
        mem_plan_count = plan_alloc_count;
        // For each alloc, find the earliest dead alloc with compatible size
        // "Dead" = buf_last_use is set and < all buf_last_use of allocs between them
        // Track which allocs are "in use" by a later reuse (can't be re-added to pool)
        u8 alloc_in_use[MAX_PLAN_ENTRIES]; memset(alloc_in_use, 0, plan_alloc_count);
        i32 fpool[64]; u32 fpool_n = 0;

        for (u32 a = 0; a < plan_alloc_count; a++) {
            u32 bid = plan_alloc_ids[a];
            u64 bsz = metal_pool.sizes[bid];

            // Release dead allocs: p is dead if its buf_last_use < alloc a's birth
            u32 a_birth = plan_alloc_birth[a];
            for (u32 p = 0; p < a; p++) {
                if (!alloc_in_use[p]) continue;
                u32 pbid = plan_alloc_ids[p];
                u32 pd = buf_last_use[pbid];
                if (pd == 0) continue;         // never used — unknown lifetime
                if (pd >= a_birth) continue;   // p still alive when a is born
                alloc_in_use[p] = 0;
                if (fpool_n < 64) fpool[fpool_n++] = (i32)p;
            }

            // Find size-compatible match in free pool
            int found = -1;
            for (u32 f = 0; f < fpool_n; f++) {
                u32 fpbid = plan_alloc_ids[fpool[f]];
                u64 fpsz = metal_pool.sizes[fpbid];
                if (fpsz >= bsz && fpsz <= bsz * 2) {
                    found = fpool[f];
                    alloc_in_use[found] = 1; // mark as in use
                    fpool[f] = fpool[--fpool_n]; // remove from pool
                    break;
                }
            }
            mem_plan_reuse[a] = (found >= 0) ? (i32)found : -1;
            // Mark alloc a itself as in-use (its buffer, whether fresh or reused,
            // can't be reused until a dies)
            alloc_in_use[a] = 1;
        }

        u32 reuse_count = 0;
        for (u32 i = 0; i < mem_plan_count; i++)
            if (mem_plan_reuse[i] >= 0) reuse_count++;
        if (reuse_count > 0)
            fprintf(stderr, "MEM_PLAN: %u allocs, %u reused\n", mem_plan_count, reuse_count);
    }

    // Move ephemeral buffers to free list. Dedup: shared MTLBuffers (from plan
    // reuse) must only be added once. Skip duplicates.
    for (u32 i = buf_keep; i < metal_pool.count; i++) {
        u64 sz = metal_pool.sizes[i];
        if (sz <= metal_bytes_inuse) metal_bytes_inuse -= sz;
        else metal_bytes_inuse = 0;
        if (metal_pool.bufs[i]) {
            // Check if this MTLBuffer is already in the free list (shared slot)
            int dup = 0;
            for (u32 f = 0; f < free_count && !dup; f++)
                if (free_list[f].buf == metal_pool.bufs[i]) dup = 1;
            if (!dup && free_count < MAX_FREE_BUFS) {
                free_list[free_count].buf = metal_pool.bufs[i];
                free_list[free_count].size = sz;
                free_count++;
            }
            // Don't [release] — ARC handles it when slot is nil'd
        }
        metal_pool.bufs[i] = nil;
        metal_pool.sizes[i] = 0;
        buf_refcount[i] = 0;
    }

reset_counters:
    pending_free_count = 0;
    dispatch_counter = 0;
    memset(buf_last_use, 0, sizeof(buf_last_use));
    memset(buf_remaining_uses, 0, sizeof(buf_remaining_uses));
    metal_pool.count = buf_keep;

    // Activate plan for next step
    mem_plan_active = (mem_plan_count > 0);
    mem_plan_cursor = 0;
    plan_alloc_count = 0;
}

static void metal_pool_set_persistent(u32 max_persistent_buf) {
    (void)max_persistent_buf; // now handled by buf_cpu_only tracking
}

// Memory checkpoint: try to recycle consumed buffers when memory is high.
// Called after fused kernel dispatch with leaf buf_ids that may be reclaimable.
// Flushes GPU if there are pending_free buffers above a memory threshold.
// Pre-scan: mark buffer as having one more future dispatch use.
static void metal_buf_mark_use(u32 id) {
    if (id && id < MAX_BUFS) buf_remaining_uses[id]++;
}

static void metal_mem_checkpoint(u32 *leaf_buf_ids, u32 n_leaves) {
    dispatch_counter++;
    for (u32 i = 0; i < n_leaves; i++) {
        u32 bid = leaf_buf_ids[i];
        if (bid && bid < MAX_BUFS) {
            buf_last_use[bid] = dispatch_counter;
            if (buf_remaining_uses[bid] > 0) buf_remaining_uses[bid]--;
        }
    }
    // Flush if memory pressure is high and there are buffers to reclaim
    if (pending_free_count > 0 && metal_bytes_inuse > 6ULL * 1024 * 1024 * 1024)
        metal_flush();
}
