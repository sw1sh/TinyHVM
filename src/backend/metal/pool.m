// metal/pool.m — Metal buffer pool functions (state in init.m)

static u32 metal_buf_alloc(u64 bytes) {
    bytes = MAX(bytes, 4);
    u32 id = metal_pool.count++;
    if (id >= MAX_BUFS - 64) {
        fprintf(stderr, "METAL: buf pool near limit (%u/%u), flushing pending free\n", id, MAX_BUFS);
        // Try to recover by flushing pending free buffers
        if (batch_dirty) metal_flush();
        for (u32 i = 0; i < pending_free_count; i++) metal_buf_free(pending_free[i]);
        pending_free_count = 0;
    }
    if (id >= MAX_BUFS) {
        fprintf(stderr, "FATAL: Metal buffer pool exhausted (%u buffers). Aborting safely.\n", id);
        fprintf(stderr, "  This happens with large models (beautiful_mnist 5x5 conv backward).\n");
        fprintf(stderr, "  The decomposed MM creates too many intermediate buffers.\n");
        exit(1);  // Clean exit instead of GPU page fault
    }

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
    } else {
        metal_pool.bufs[id] = [mtl_dev newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
        metal_pool.sizes[id] = bytes;
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
    assert(buf_refcount[id] > 0 && "buf_decref on zero refcount");
    if (--buf_refcount[id] == 0) {
        if (pending_free_count < PENDING_FREE_CAP)
            pending_free[pending_free_count++] = id;
    }
}

static void metal_buf_free(u32 id) {
    if (metal_pool.bufs[id] && free_count < MAX_FREE_BUFS) {
        free_list[free_count].buf = metal_pool.bufs[id];
        free_list[free_count].size = metal_pool.sizes[id];
        free_count++;
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
        if (metal_pool.bufs[i] && free_count < MAX_FREE_BUFS) {
            free_list[free_count].buf = metal_pool.bufs[i];
            free_list[free_count].size = metal_pool.sizes[i];
            free_count++;
        }
        metal_pool.bufs[i] = nil;
        metal_pool.sizes[i] = 0;
        buf_refcount[i] = 0;
    }
    // Clear pending_free — all ephemeral buffers already moved to free_list above
    pending_free_count = 0;
    metal_pool.count = buf_keep;
}

static void metal_pool_set_persistent(u32 max_persistent_buf) {
    (void)max_persistent_buf; // now handled by buf_cpu_only tracking
}
