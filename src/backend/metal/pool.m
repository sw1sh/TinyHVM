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

static struct {
    id<MTLBuffer> buf;  // retained MTLBuffer for reuse
    u64 size;
} mem_plan[MAX_PLAN_ENTRIES];
static u32 mem_plan_count = 0;
static u32 mem_plan_cursor = 0;  // advances during step 1+
static int mem_plan_active = 0;  // 1 = plan ready for this step

// Recording: track large alloc sequence + lifetimes during step 0
static u32 plan_alloc_ids[MAX_PLAN_ENTRIES]; // buf_ids in alloc order
static u32 plan_alloc_count = 0;

#define METAL_MEM_BUDGET (4ULL * 1024 * 1024 * 1024) // 4GB hard limit

static u32 metal_buf_alloc(u64 bytes) {
    bytes = MAX(bytes, 4);

    u32 id = metal_pool.count++;
    if (id >= MAX_BUFS) {
        fprintf(stderr, "FATAL: Metal buffer pool exhausted (%u buffers). Aborting safely.\n", id);
        exit(1);
    }

    // 0. Memory plan reuse (step 1+): use pre-planned buffer
    if (0 && mem_plan_active && bytes >= PLAN_MIN_BYTES && mem_plan_cursor < mem_plan_count) {
        u32 c = mem_plan_cursor++;
        if (mem_plan[c].buf && mem_plan[c].size >= bytes && mem_plan[c].size <= bytes * 2) {
            metal_pool.bufs[id] = mem_plan[c].buf;
            metal_pool.sizes[id] = mem_plan[c].size;
            static u32 plan_use_n = 0;
            if (plan_use_n < 3) { plan_use_n++;
                fprintf(stderr, "  PLAN_USE[%u]: buf=%u %.1fMB (plan %.1fMB)\n",
                    c, id, (double)bytes/1e6, (double)mem_plan[c].size/1e6);
            }
            goto done;
        }
        // Size mismatch — fall through to normal alloc
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
    if (bytes >= PLAN_MIN_BYTES && !mem_plan_active && plan_alloc_count < MAX_PLAN_ENTRIES)
        plan_alloc_ids[plan_alloc_count++] = id;

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
    memcpy(metal_pool.bufs[id].contents, data, bytes);
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
    memcpy(out, metal_pool.bufs[id].contents, n);
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
        memcpy(out, metal_pool.bufs[id].contents, actual);
        memset((char*)out + actual, 0, bytes - actual);
    } else {
        memcpy(out, metal_pool.bufs[id].contents, bytes);
    }
    thvm_prof_buf_read(bytes);
}

static void metal_pool_reset(u32 keep) {
    if (batch_dirty) metal_flush();
    u32 buf_keep = keep + 1;
    u32 n_free = metal_pool.count - buf_keep;
    if (n_free == 0) goto reset_counters;

    // ── Build memory plan from this step's allocation profile ──
    // For each recorded large alloc, check if any EARLIER alloc's buffer
    // can be reused (earlier alloc dead before this alloc happens).
    // "Dead" = buf_last_use[earlier] is set AND < buf_last_use of intervening allocs.
    if (plan_alloc_count > 0 && !mem_plan_active) {
        // Greedy reuse: maintain a pool of free'd MTLBuffers keyed by size.
        // Process allocs in order. After each alloc, check if its buffer dies
        // before the next alloc — if so, add to the free pool.
        struct { id<MTLBuffer> buf; u64 size; } fpool[64];
        u32 fpool_n = 0;

        mem_plan_count = 0;
        for (u32 a = 0; a < plan_alloc_count && mem_plan_count < MAX_PLAN_ENTRIES; a++) {
            u32 bid = plan_alloc_ids[a];
            u64 bsz = metal_pool.sizes[bid];

            // Release dead buffers from earlier allocs into the free pool
            for (u32 p = 0; p < a; p++) {
                u32 pbid = plan_alloc_ids[p];
                if (!pbid || !metal_pool.bufs[pbid]) continue;
                u32 pdeath = buf_last_use[pbid];
                if (pdeath == 0) continue; // never used as leaf — unclear lifetime
                // Buffer p is dead if no alloc between p+1 and a reads from it
                // (i.e., pdeath < the smallest buf_last_use of allocs p+1..a-1 OR
                //  pdeath < dispatch_counter at alloc a's time).
                // Simplification: if pdeath <= buf_last_use[bid], p died before a was used.
                // But we don't know a's "birth dispatch". Use: pdeath < dispatch of a.
                // Since allocs are ordered, a reasonable check: pdeath is "old".
                // Actually, just check if it's already in fpool:
                int in_fp = 0;
                for (u32 f = 0; f < fpool_n; f++)
                    if (fpool[f].buf == metal_pool.bufs[pbid]) { in_fp = 1; break; }
                if (in_fp) continue;
                // Add if death was before current alloc's index (conservative)
                // Use dispatch order: if pdeath < alloc-a-related dispatches
                // Simplest: just add ALL dead buffers from before a
                if (fpool_n < 64) {
                    fpool[fpool_n].buf = metal_pool.bufs[pbid];
                    fpool[fpool_n].size = metal_pool.sizes[pbid];
                    fpool_n++;
                }
            }

            // Find match in free pool
            int found = -1;
            for (u32 f = 0; f < fpool_n; f++) {
                if (fpool[f].buf && fpool[f].size >= bsz && fpool[f].size <= bsz * 4) {
                    found = (int)f; break;
                }
            }
            if (found >= 0) {
                mem_plan[mem_plan_count].buf = fpool[found].buf;
                mem_plan[mem_plan_count].size = fpool[found].size;
                fpool[found] = fpool[--fpool_n]; // remove from pool
            } else {
                mem_plan[mem_plan_count].buf = nil; // no reuse — allocate fresh
                mem_plan[mem_plan_count].size = 0;
            }
            mem_plan_count++;
        }

        u32 reuse_count = 0;
        for (u32 i = 0; i < mem_plan_count; i++)
            if (mem_plan[i].buf) reuse_count++;
        u64 plan_bytes = 0;
        for (u32 i = 0; i < mem_plan_count; i++)
            if (mem_plan[i].buf) plan_bytes += mem_plan[i].size;
        if (reuse_count > 0)
            fprintf(stderr, "MEM_PLAN: %u allocs, %u reused (%.0fMB retained)\n",
                    mem_plan_count, reuse_count, (double)plan_bytes/1e6);
    }

    // Move ephemeral buffers to free list, but RETAIN plan-reuse buffers
    for (u32 i = buf_keep; i < metal_pool.count; i++) {
        u64 sz = metal_pool.sizes[i];
        if (sz <= metal_bytes_inuse) metal_bytes_inuse -= sz;
        else metal_bytes_inuse = 0;
        // Check if retained by memory plan
        int in_plan = 0;
        for (u32 p = 0; p < mem_plan_count && !in_plan; p++)
            if (mem_plan[p].buf == metal_pool.bufs[i]) in_plan = 1;
        if (in_plan) {
            metal_pool.bufs[i] = nil;
            metal_pool.sizes[i] = 0;
            buf_refcount[i] = 0;
            continue; // plan retains the MTLBuffer
        }
        if (metal_pool.bufs[i] && free_count < MAX_FREE_BUFS) {
            free_list[free_count].buf = metal_pool.bufs[i];
            free_list[free_count].size = sz;
            free_count++;
        } else if (metal_pool.bufs[i]) {
            [metal_pool.bufs[i] release];
            if (sz <= metal_bytes_total) metal_bytes_total -= sz;
            else metal_bytes_total = 0;
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
