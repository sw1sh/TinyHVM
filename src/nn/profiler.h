// nn/profiler.h — Lightweight per-section profiling
// Usage: PROF_INIT(); PROF_START(); ... PROF_ACC(section); ... PROF_REPORT();
//
// Uses mach_absolute_time on macOS for nanosecond precision.

#ifndef NN_PROFILER_H
#define NN_PROFILER_H

#ifdef __APPLE__
#include <mach/mach_time.h>

#define PROF_MAX_SECTIONS 16

typedef struct {
    const char *name;
    u64 total_ticks;
} ProfEntry;

typedef struct {
    ProfEntry entries[PROF_MAX_SECTIONS];
    u32 n_entries;
    u32 n_steps;
    mach_timebase_info_data_t tbi;
    u64 _start;
} Profiler;

static Profiler prof_global;

static void prof_init(void) {
    memset(&prof_global, 0, sizeof(prof_global));
    mach_timebase_info(&prof_global.tbi);
}

static u32 prof_section(const char *name) {
    for (u32 i = 0; i < prof_global.n_entries; i++)
        if (prof_global.entries[i].name == name) return i;
    u32 idx = prof_global.n_entries++;
    prof_global.entries[idx].name = name;
    return idx;
}

#define PROF_START() prof_global._start = mach_absolute_time()
#define PROF_ACC(name) do { \
    u64 _now = mach_absolute_time(); \
    u32 _idx = prof_section(name); \
    prof_global.entries[_idx].total_ticks += _now - prof_global._start; \
    prof_global._start = _now; \
} while(0)
#define PROF_STEP() prof_global.n_steps++

static void prof_report(void) {
    if (prof_global.n_steps == 0) return;
    f32 steps = (f32)prof_global.n_steps;
    f32 ns_per_tick = (f32)prof_global.tbi.numer / (f32)prof_global.tbi.denom;
    u64 total_ticks = 0;
    printf("\n  === Profile (avg per step, %u steps) ===\n", prof_global.n_steps);
    for (u32 i = 0; i < prof_global.n_entries; i++) {
        f32 ms = (f32)prof_global.entries[i].total_ticks * ns_per_tick / 1e6f / steps;
        total_ticks += prof_global.entries[i].total_ticks;
        printf("  %-14s %6.1f ms\n", prof_global.entries[i].name, ms);
    }
    printf("  %-14s %6.1f ms\n", "TOTAL", (f32)total_ticks * ns_per_tick / 1e6f / steps);
}

#else
// No-op profiler for non-Apple platforms
#define PROF_START()
#define PROF_ACC(name)
#define PROF_STEP()
static void prof_init(void) {}
static void prof_report(void) {}
#endif

#endif // NN_PROFILER_H
