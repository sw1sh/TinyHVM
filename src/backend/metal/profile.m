// metal/profile.m — Graph-level profile report (tinygrad-style kernel breakdown)

static void metal_profile_report(void) {
    if (!thvm_prof_global.enabled) return;
    thvm_prof_phase_end();

    printf("\n  ╔═══════════════════════════════════════════════════════════╗\n");
    printf("  ║            TinyHVM Step Profile (THVM_PROFILE=1)         ║\n");
    printf("  ╠═══════════════════════════════════════════════════════════╣\n");

    printf("  ║  Phase Timing:                                           ║\n");
    u64 total_phase_ns = 0;
    for (u32 i = 0; i < PHASE_COUNT; i++) total_phase_ns += thvm_prof_global.phase_ns[i];
    for (u32 i = 0; i < PHASE_COUNT; i++) {
        if (thvm_prof_global.phase_ns[i] == 0) continue;
        f32 ms = (f32)thvm_prof_global.phase_ns[i] / 1e6f;
        f32 pct = total_phase_ns ? 100.0f * (f32)thvm_prof_global.phase_ns[i] / (f32)total_phase_ns : 0;
        printf("  ║    %-10s %8.1fms  (%5.1f%%)                        ║\n",
               phase_names[i], ms, pct);
    }
    printf("  ║    %-10s %8.1fms                                  ║\n",
           "TOTAL", (f32)total_phase_ns / 1e6f);

    printf("  ╠═══════════════════════════════════════════════════════════╣\n");
    printf("  ║  UOp Dispatch:                                           ║\n");
    printf("  ║  %-10s %6s %8s %8s %6s                  ║\n",
           "Op", "Count", "Total", "Avg", "Tens");
    printf("  ╠═══════════════════════════════════════════════════════════╣\n");
    u64 total_uop_ns = 0;
    u32 total_uop_cnt = 0;
    u32 total_uop_tens = 0;
    const char *ext_uop_names[] = {
    };
    for (u32 i = 0; i < PROF_UOP_MAX; i++) {
        if (thvm_prof_global.uop_cnt[i] == 0 && thvm_prof_global.uop_tensors[i] == 0) continue;
        const char *name = "?";
        if (i < UOP_COUNT && i < sizeof(uop_names)/sizeof(uop_names[0])) name = uop_names[i];
        f32 total_ms = (f32)thvm_prof_global.uop_ns[i] / 1e6f;
        f32 avg_us = thvm_prof_global.uop_cnt[i] ?
            (f32)thvm_prof_global.uop_ns[i] / (f32)thvm_prof_global.uop_cnt[i] / 1e3f : 0;
        printf("  ║  %-10s %6u %6.1fms %6.0fμs %6u                  ║\n",
               name, thvm_prof_global.uop_cnt[i], total_ms, avg_us,
               thvm_prof_global.uop_tensors[i]);
        total_uop_ns += thvm_prof_global.uop_ns[i];
        total_uop_cnt += thvm_prof_global.uop_cnt[i];
        total_uop_tens += thvm_prof_global.uop_tensors[i];
    }
    printf("  ║  %-10s %6u %6.1fms          %6u                  ║\n",
           "TOTAL", total_uop_cnt, (f32)total_uop_ns / 1e6f, total_uop_tens);

    printf("  ╠═══════════════════════════════════════════════════════════╣\n");
    printf("  ║  Memory:                                                 ║\n");
    printf("  ║    buf_alloc:  %6u calls, %8.1f MB this step        ║\n",
           thvm_prof_global.buf_alloc_cnt,
           (f32)thvm_prof_global.buf_bytes_alloc / (1024.0f * 1024.0f));
    printf("  ║    live bufs:  %8.1f MB current, %8.1f MB peak      ║\n",
           (f32)thvm_prof_global.buf_bytes_current / (1024.0f * 1024.0f),
           (f32)thvm_prof_global.buf_bytes_peak / (1024.0f * 1024.0f));

    printf("  ║  Tensors:                                                ║\n");
    printf("  ║    created: %5u   freed: %5u   peak: %5u             ║\n",
           thvm_prof_global.tensor_created,
           thvm_prof_global.tensor_freed,
           thvm_prof_global.tensor_peak);

    printf("  ║  Heap:                                                   ║\n");
    printf("  ║    peak: %8llu words (%5.1f MB)   at_reset: %8llu   ║\n",
           (unsigned long long)thvm_prof_global.heap_peak,
           (f32)thvm_prof_global.heap_peak * 8.0f / (1024.0f * 1024.0f),
           (unsigned long long)thvm_prof_global.heap_at_reset);

    printf("  ║  CPU↔GPU:                                                ║\n");
    printf("  ║    read:  %6u calls, %8.1f MB                      ║\n",
           thvm_prof_global.cpu_read_cnt,
           (f32)thvm_prof_global.cpu_read_bytes / (1024.0f * 1024.0f));
    printf("  ║    write: %6u calls, %8.1f MB                      ║\n",
           thvm_prof_global.cpu_write_cnt,
           (f32)thvm_prof_global.cpu_write_bytes / (1024.0f * 1024.0f));

    printf("  ╚═══════════════════════════════════════════════════════════╝\n");
    (void)ext_uop_names;
}

static void metal_profile_reset(void) {
    thvm_prof_step_reset();
}
