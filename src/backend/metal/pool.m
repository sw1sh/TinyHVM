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

#define METAL_MEM_BUDGET (8ULL * 1024 * 1024 * 1024) // 8GB hard limit

static u32 metal_buf_alloc(u64 bytes) {
    bytes = MAX(bytes, 4);

    u32 id = metal_pool.count++;
    if (id >= MAX_BUFS) {
        fprintf(stderr, "FATAL: Metal buffer pool exhausted (%u buffers). Aborting safely.\n", id);
        exit(1);
    }

    // 1. Check free list first (from pool_reset)
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
    } else if (bytes >= 1024*1024 && dispatch_counter > 2) {
        // 2. For large allocs: try to reclaim a "dead" buffer from the pool.
        // A buffer is reclaimable if it was last used as a dispatch leaf
        // at least 2 dispatches ago (GPU guarantees sequential execution,
        // so the reader has completed). Size must be compatible.
        u32 reuse_id = 0;
        u64 reuse_size = UINT64_MAX;
        u32 keep = 1; // scan all live buffers for reuse
        for (u32 i = keep; i < id; i++) {
            if (!metal_pool.bufs[i]) continue;
            u64 sz = metal_pool.sizes[i];
            if (sz < bytes || sz > bytes * 2) continue;
            if (buf_last_use[i] == 0) continue; // never used as leaf
            if (buf_last_use[i] + 1 >= dispatch_counter) continue; // too recent
            if (sz < reuse_size) { reuse_id = i; reuse_size = sz; }
        }
        if (reuse_id) {
            // Steal the Metal buffer from the old slot
            metal_pool.bufs[id] = metal_pool.bufs[reuse_id];
            metal_pool.sizes[id] = metal_pool.sizes[reuse_id];
            metal_pool.bufs[reuse_id] = nil;
            u64 old_size = metal_pool.sizes[reuse_id];
            metal_pool.sizes[reuse_id] = 0;
            buf_refcount[reuse_id] = 0;
            buf_last_use[reuse_id] = 0;
            // inuse stays same (stolen, not new)
        } else {
            metal_pool.bufs[id] = [mtl_dev newBufferWithLength:bytes
                                                       options:MTLResourceStorageModeShared];
            metal_pool.sizes[id] = bytes;
            metal_bytes_total += bytes;
            metal_bytes_inuse += bytes;
        }
    } else {
        metal_pool.bufs[id] = [mtl_dev newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
        metal_pool.sizes[id] = bytes;
        metal_bytes_total += bytes;
        metal_bytes_inuse += bytes;
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
    if (batch_dirty) metal_flush(); // needed: can't free GPU-active buffers
    u32 buf_keep = keep + 1;
    // (buf_cpu_only tracking replaces persistent_buf_count approach)
    u32 n_free = metal_pool.count - buf_keep;
    if (n_free == 0) return;

    // Move to free list for reuse (no deallocation)
    for (u32 i = buf_keep; i < metal_pool.count; i++) {
        u64 sz = metal_pool.sizes[i];
        if (sz <= metal_bytes_inuse) metal_bytes_inuse -= sz;
        else metal_bytes_inuse = 0;
        // Metal memory stays allocated — metal_bytes_total unchanged
        if (metal_pool.bufs[i] && free_count < MAX_FREE_BUFS) {
            free_list[free_count].buf = metal_pool.bufs[i];
            free_list[free_count].size = sz;
            free_count++;
        } else if (metal_pool.bufs[i]) {
            // Free list full — actually release
            [metal_pool.bufs[i] release];
            if (sz <= metal_bytes_total) metal_bytes_total -= sz;
            else metal_bytes_total = 0;
        }
        metal_pool.bufs[i] = nil;
        metal_pool.sizes[i] = 0;
        buf_refcount[i] = 0;
    }
    // Clear pending_free — all ephemeral buffers already moved to free_list above
    pending_free_count = 0;
    dispatch_counter = 0;
    memset(buf_last_use, 0, sizeof(buf_last_use));
    metal_pool.count = buf_keep;
}

static void metal_pool_set_persistent(u32 max_persistent_buf) {
    (void)max_persistent_buf; // now handled by buf_cpu_only tracking
}

// Memory checkpoint: try to recycle consumed buffers when memory is high.
// Called after fused kernel dispatch with leaf buf_ids that may be reclaimable.
// Flushes GPU if there are pending_free buffers above a memory threshold.
static void metal_mem_checkpoint(u32 *leaf_buf_ids, u32 n_leaves) {
    dispatch_counter++;
    // Mark leaf buffers with their last-use dispatch number
    for (u32 i = 0; i < n_leaves; i++) {
        u32 bid = leaf_buf_ids[i];
        if (bid && bid < MAX_BUFS)
            buf_last_use[bid] = dispatch_counter;
    }
    // Flush if memory pressure is high and there are buffers to reclaim
    if (pending_free_count > 0 && metal_bytes_inuse > 2ULL * 1024 * 1024 * 1024)
        metal_flush();
}
