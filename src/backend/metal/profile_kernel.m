// metal/profile_kernel.m — Per-kernel GPU profiling (tinygrad-compatible)
// When THVM_DEBUG >= 2, each kernel gets its own command buffer for
// accurate GPU-side timing via GPUStartTime/GPUEndTime.
// Output format matches tinygrad's DEBUG=2 exactly.

static int _thvm_debug = -1;

typedef struct {
    u64 global_ops;    // total FLOPs across all kernels
    u64 global_mem;    // total bytes accessed
    double time_sum_s; // cumulative GPU time in seconds
    u32 kernel_count;  // number of kernels executed
    u64 mem_used;      // current GPU memory allocated (bytes)
} ThvmGlobalCounters;

static ThvmGlobalCounters gc = {0};

static void gc_reset(void) {
    gc.global_ops = 0; gc.global_mem = 0; gc.time_sum_s = 0; gc.kernel_count = 0;
}

// Estimate FLOPs and memory for a kernel dispatch
typedef struct {
    u64 ops;    // estimated FLOPs
    u64 mem;    // estimated bytes accessed (read + write)
    u32 numel;  // output elements
    const char *name; // kernel description
} KernelEstimate;

// Per-kernel profiling: commit individual command buffer and read GPU timestamps
static double profile_single_dispatch(void) {
    if (!batch_cmd) return 0;
    if (batch_encoder) { [batch_encoder endEncoding]; batch_encoder = nil; }
    [batch_cmd commit];
    [batch_cmd waitUntilCompleted];
    // Metal GPU timestamps (same as tinygrad's cmdbuf_st_time/cmdbuf_en_time)
    double st = [batch_cmd GPUStartTime];
    double en = [batch_cmd GPUEndTime];
    batch_cmd = nil;
    batch_dirty = 0;
    return (en - st); // seconds
}

// Print kernel profile line (tinygrad DEBUG=2 format)
static void profile_print_kernel(KernelEstimate est, double gpu_seconds) {
    gc.kernel_count++;
    gc.global_ops += est.ops;
    gc.global_mem += est.mem;
    gc.time_sum_s += gpu_seconds;

    double gflops = est.ops / (gpu_seconds > 0 ? gpu_seconds : 1e-20) * 1e-9;
    double gbps = est.mem / (gpu_seconds > 0 ? gpu_seconds : 1e-20) * 1e-9;

    fprintf(stderr, "*** METAL %4u %-40s  n=%7u  tm %8.2fus/%8.2fms (%9.2f GFLOPS %6.1f GB/s)\n",
        gc.kernel_count,
        est.name ? est.name : "?",
        est.numel,
        gpu_seconds * 1e6,
        gc.time_sum_s * 1e3,
        gflops,
        gbps);
}

// Profiled dispatch wrapper: if THVM_DEBUG>=2, profile each kernel individually
static void profiled_dispatch(u32 numel, u64 ops_est, u64 mem_est, const char *name) {
    if (_thvm_debug < 0) {
        const char *env = getenv("THVM_DEBUG");
        _thvm_debug = env ? atoi(env) : 0;
    }
    if (_thvm_debug >= 2) {
        double et = profile_single_dispatch();
        KernelEstimate est = {ops_est, mem_est, numel, name};
        profile_print_kernel(est, et);
    }
}

// Print summary (call at end of step or program)
void thvm_profile_kernels_summary(void) {
    if (gc.kernel_count == 0) return;
    fprintf(stderr, "\n=== TinyHVM Kernel Profile ===\n");
    fprintf(stderr, "  Kernels:    %u\n", gc.kernel_count);
    fprintf(stderr, "  GPU time:   %.2fms\n", gc.time_sum_s * 1e3);
    fprintf(stderr, "  Total ops:  %.2f GFLOPS\n", gc.global_ops * 1e-9);
    fprintf(stderr, "  Total mem:  %.2f GB\n", gc.global_mem * 1e-9);
    fprintf(stderr, "  Throughput: %.2f GFLOPS\n",
        gc.global_ops / (gc.time_sum_s > 0 ? gc.time_sum_s : 1e-20) * 1e-9);
    fprintf(stderr, "  Bandwidth:  %.2f GB/s\n",
        gc.global_mem / (gc.time_sum_s > 0 ? gc.time_sum_s : 1e-20) * 1e-9);
    fprintf(stderr, "==============================\n");
}
