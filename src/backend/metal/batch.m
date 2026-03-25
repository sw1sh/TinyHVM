// metal/batch.m — Command buffer batching

#include <time.h>
static double flush_total_ms = 0;
static u32 flush_count_total = 0;

static void metal_flush(void) {
    if (batch_encoder) {
        [batch_encoder endEncoding];
        batch_encoder = nil;
    }
    if (batch_cmd) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        [batch_cmd commit];
        [batch_cmd waitUntilCompleted];
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ms = (double)(t1.tv_sec-t0.tv_sec)*1000.0 + (double)(t1.tv_nsec-t0.tv_nsec)/1e6;
        flush_total_ms += ms;
        flush_count_total++;
        batch_cmd = nil;
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
