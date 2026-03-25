// metal/batch.m — Command buffer batching

static u32 flush_count = 0;

static void metal_flush(void) {
    if (batch_encoder) {
        [batch_encoder endEncoding];
        batch_encoder = nil;
    }
    if (batch_cmd) {
        [batch_cmd commit];
        [batch_cmd waitUntilCompleted];
        batch_cmd = nil;
        flush_count++;
    }
    batch_dirty = 0;
}

static id<MTLComputeCommandEncoder> get_encoder(void) {
    if (!batch_cmd) {
        batch_cmd = [mtl_queue commandBuffer];
    }
    if (!batch_encoder) {
        batch_encoder = [batch_cmd computeCommandEncoder];
    }
    return batch_encoder;
}

static void metal_begin_batch(void) {
    batch_active = 1;
}

static void metal_end_batch(void) {
    if (batch_dirty) metal_flush();
    batch_active = 0;
}

u32 metal_get_flush_count(void) { return flush_count; }
void metal_reset_flush_count(void) { flush_count = 0; }
