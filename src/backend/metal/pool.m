// metal/pool.m — Metal buffer pool functions (state in init.m)

static u32 metal_buf_alloc(u64 bytes) {
    bytes = MAX(bytes, 4);
    u32 id = metal_pool.count++;
    assert(id < MAX_BUFS);

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
    thvm_prof_buf_alloc(bytes);
    return id;
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
    // Shared memory: CPU writes are visible to GPU.
    // Only need to flush if writing to a buffer that GPU is currently reading.
    // For newly allocated buffers (most common case during backward),
    // no flush needed — GPU hasn't seen the buffer yet.
    // TODO: track per-buffer GPU usage for precise flushing.
    // For now, skip flush entirely — Metal shared memory is coherent on Apple Silicon.
    memcpy(metal_pool.bufs[id].contents, data, bytes);
    thvm_prof_buf_write(bytes);
}

// Read without flushing — safe for CPU-written metadata that GPU hasn't touched.
void metal_buf_read_nosync(u32 id, void *out, u64 bytes) {
    u64 actual = metal_pool.sizes[id];
    u64 n = bytes < actual ? bytes : actual;
    memcpy(out, metal_pool.bufs[id].contents, n);
    if (n < bytes) memset((char*)out + n, 0, bytes - n);
}

static void metal_buf_read(u32 id, void *out, u64 bytes) {
    if (batch_dirty) metal_flush();
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
    }
    metal_pool.count = buf_keep;
}
