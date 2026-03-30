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

// Memory plan: maps alloc sequence N → earlier alloc M whose OFFSET to reuse.
static i32 mem_plan_reuse[MAX_PLAN_ENTRIES]; // reuse[N] = M or -1
static u64 mem_plan_offset[MAX_PLAN_ENTRIES]; // byte offset in the big buffer
static u64 mem_plan_sizes[MAX_PLAN_ENTRIES]; // alloc size
static u32 mem_plan_count = 0;
static u32 mem_plan_cursor = 0;
static int mem_plan_active = 0;
static id<MTLHeap> mem_plan_heap = nil; // Metal heap for suballocation
static u64 mem_plan_heap_size = 0;

// Recording: track large alloc sequence + lifetimes during step 0
static u32 plan_alloc_ids[MAX_PLAN_ENTRIES];
static u32 plan_alloc_birth[MAX_PLAN_ENTRIES];
static u32 plan_alloc_count = 0;

#define METAL_MEM_BUDGET (8ULL * 1024 * 1024 * 1024) // 8GB hard limit

static u32 metal_buf_alloc(u64 bytes) {
    bytes = MAX(bytes, 4);

    u32 id = metal_pool.count++;
    if (id >= MAX_BUFS) {
        fprintf(stderr, "FATAL: Metal buffer pool exhausted (%u buffers). Aborting safely.\n", id);
        exit(1);
    }

    // 0. Heap alloc: ALL large buffers from MTLHeap on step 1+
    if (mem_plan_active && bytes >= PLAN_MIN_BYTES && mem_plan_heap) {
        if (mem_plan_cursor < mem_plan_count) mem_plan_cursor++;
        metal_pool.bufs[id] = [mem_plan_heap newBufferWithLength:bytes
                                                         options:MTLResourceStorageModeShared];
        if (metal_pool.bufs[id]) {
            metal_pool.sizes[id] = bytes;
            buf_offset[id] = 0;
            goto done;
        }
    }

    // 1. Free list (from pool_reset) — skip large buffers if plan is active
    if (!(mem_plan_active && bytes >= PLAN_MIN_BYTES)) {
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

    // 2. Mid-step steal (step 0, before plan exists)
    if (!mem_plan_active && bytes >= 1024*1024 && dispatch_counter > 2) {
        u32 reuse_id = 0;
        u64 reuse_size = UINT64_MAX;
        for (u32 i = 1; i < id; i++) {
            if (!metal_pool.bufs[i]) continue;
            u64 sz = metal_pool.sizes[i];
            if (sz < bytes || sz > bytes * 2) continue;
            if (buf_remaining_uses[i] > 0) continue;
            if (buf_last_use[i] > 0 && buf_last_use[i] + 1 >= dispatch_counter) continue;
            if (buf_last_use[i] == 0 && i + 200 > id) continue;
            if (sz < reuse_size) { reuse_id = i; reuse_size = sz; }
        }
        if (reuse_id) {
            metal_pool.bufs[id] = metal_pool.bufs[reuse_id];
            metal_pool.sizes[id] = metal_pool.sizes[reuse_id];
            metal_pool.bufs[reuse_id] = nil;
            metal_pool.sizes[reuse_id] = 0;
            buf_refcount[reuse_id] = 0;
            buf_last_use[reuse_id] = 0;
            goto done;
        }
    }

    // 3. Fresh allocation
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
        // (step 1+: plan cursor already advanced in alloc path above)
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
                // If last_use==0 (consumed via view alias, not tracked),
                // assume dead if old enough (50+ allocs ago in step 0)
                if (pd == 0 && a - p < 10) continue;
                if (pd > 0 && pd >= a_birth) continue;
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
        // Assign offsets: each alloc gets a unique region. Reused allocs share
        // the same offset as their dead predecessor (different time, same region).
        u64 next_offset = 0;
        #define ALIGN_4K(x) (((x) + 0xFFF) & ~0xFFFULL)
        for (u32 a = 0; a < plan_alloc_count; a++) {
            u32 bid = plan_alloc_ids[a];
            u64 bsz = metal_pool.sizes[bid];
            i32 r = mem_plan_reuse[a];
            if (r >= 0) {
                // Reuse: same offset as the dead alloc
                mem_plan_offset[a] = mem_plan_offset[r];
                mem_plan_sizes[a] = mem_plan_sizes[r]; // same region size
            } else {
                // Fresh: allocate new region
                mem_plan_offset[a] = next_offset;
                mem_plan_sizes[a] = bsz;
                next_offset += ALIGN_4K(bsz);
            }
        }

        // Compute total alloc bytes (not just peak simultaneous)
        u64 total_alloc = 0;
        for (u32 a = 0; a < plan_alloc_count; a++)
            total_alloc += ALIGN_4K(metal_pool.sizes[plan_alloc_ids[a]]);

        // Size heap for total allocation (Metal auto-reclaims released regions)
        u64 heap_size = total_alloc > next_offset * 2 ? total_alloc : next_offset * 2;
        if (heap_size > mem_plan_heap_size) {
            MTLHeapDescriptor *desc = [[MTLHeapDescriptor alloc] init];
            desc.size = heap_size;
            desc.storageMode = MTLStorageModeShared;
            desc.hazardTrackingMode = MTLHazardTrackingModeTracked;
            mem_plan_heap = [mtl_dev newHeapWithDescriptor:desc];
            mem_plan_heap_size = heap_size;
        }

        if (reuse_count > 0)
            fprintf(stderr, "MEM_PLAN: %u allocs, %u reused, peak=%.0fMB total=%.0fMB heap=%.0fMB\n",
                mem_plan_count, reuse_count, (double)next_offset/1e6,
                (double)total_alloc/1e6, (double)heap_size/1e6);
    }

    // If plan is active, release large free list entries (heap replaces them)
    if (mem_plan_count > 0) {
        u32 new_fc = 0;
        for (u32 f = 0; f < free_count; f++) {
            if (free_list[f].size >= PLAN_MIN_BYTES) {
                // Large buffer — release, heap will handle it
                free_list[f].buf = nil; // ARC releases
            } else {
                free_list[new_fc++] = free_list[f]; // keep small buffers
            }
        }
        free_count = new_fc;
    }

    // Move ephemeral buffers to free list.
    // Heap-backed buffers just get nil'd (heap manages their memory).
    for (u32 i = buf_keep; i < metal_pool.count; i++) {
        u64 sz = metal_pool.sizes[i];
        // Heap-backed: just nil (heap reclaims). Don't subtract from inuse
        // (heap allocs never incremented it).
        if (mem_plan_heap && metal_pool.bufs[i] &&
            metal_pool.bufs[i].heap == mem_plan_heap) {
            metal_pool.bufs[i] = nil; metal_pool.sizes[i] = 0;
            buf_refcount[i] = 0; buf_offset[i] = 0; continue;
        }
        if (sz <= metal_bytes_inuse) metal_bytes_inuse -= sz;
        else metal_bytes_inuse = 0;
        if (metal_pool.bufs[i]) {
            // When plan is active, release large buffers (heap replaces them)
            int skip_free = (mem_plan_count > 0 && sz >= PLAN_MIN_BYTES);
            if (!skip_free && free_count < MAX_FREE_BUFS) {
                free_list[free_count].buf = metal_pool.bufs[i];
                free_list[free_count].size = sz;
                free_count++;
            } else {
                // Buffer released (ARC handles Metal dealloc on nil)
                if (sz <= metal_bytes_total) metal_bytes_total -= sz;
                else metal_bytes_total = 0;
            }
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
            // Heap-backed: release consumed buffers back to heap immediately.
            // Metal's heap auto-reclaims the region for future allocs.
            // Safe: Metal hazard tracking handles GPU read ordering.
            if (mem_plan_heap && buf_remaining_uses[bid] == 0 &&
                metal_pool.bufs[bid] &&
                metal_pool.bufs[bid].heap == mem_plan_heap) {
                metal_pool.bufs[bid] = nil;
                metal_pool.sizes[bid] = 0;
            }
        }
    }
    if (pending_free_count > 0 && metal_bytes_inuse > 6ULL * 1024 * 1024 * 1024)
        metal_flush();
}
