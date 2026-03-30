# SUP-Based Beam Search for Kernel Optimization

## Motivation

tinygrad's BEAM search tests multiple kernel configurations (LOCAL_SIZE, UPCAST, UNROLL, GROUP_REDUCE) and keeps the fastest. Currently TinyHVM uses hardcoded LOCAL_SIZE=256 for GROUP_REDUCE. The IC model's SUP mechanism maps naturally to optimization search: each configuration is a superposition branch, collapse selects the winner.

## Architecture

```
Kernel K needs optimization
→ &L{K(local=32), K(local=64), K(local=128), K(local=256)}
→ Time each branch via Metal GPU timestamps
→ Collapse to fastest
→ Cache winner by kernel signature hash
```

## Search Space (per kernel)

| Parameter | Options | Description |
|-----------|---------|-------------|
| LOCAL_SIZE | 32, 64, 128, 256, 512, 1024 | Threads per threadgroup |
| UPCAST | 1, 2, 4, 8 | Elements per thread (inner loop unroll) |
| UNROLL | 1, 2, 4 | Outer loop unrolling factor |
| GROUP_REDUCE | on/off | Parallel vs sequential reduce |
| FLOAT4 | on/off | Vectorized loads/stores |

Total: ~200 configurations. BEAM prunes to top-N per iteration.

## IC-Native Implementation

### Phase 1: Explicit Search (no SUP)

Simple loop in `cg_get_pipe_rs`:
```c
if (getenv("THVM_BEAM")) {
    UOpKernel variants[6];
    double times[6];
    u32 local_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int v = 0; v < 6; v++) {
        variants[v] = build_kernel(ops, ..., local_sizes[v]);
        times[v] = benchmark_kernel(variants[v], leaf_bufs, out_buf);
    }
    best = argmin(times);
    cache_winner(hash, variants[best], local_sizes[best]);
}
```

### Phase 2: SUP-Based Search

Represent the search as an IC superposition:
```c
// Create labeled SUP with each configuration as a branch
Term configs = thvm_sup(ctx, label,
    kernel_with_config(ops, local=32),
    kernel_with_config(ops, local=64));

// Collapse by benchmarking
CollapseResult cr = thvm_collapse_timed(ctx, configs);
// cr.best = fastest branch
```

The SUP labels encode the configuration space. DUP projections extract individual variants. Collapse with timing gives the optimal configuration.

### Phase 3: Cost Model

Train a predictor: `(shape, n_ops, n_leaves, reduce_numel, rank) → optimal_config`

Data source: benchmark results from Phase 1/2 (already collected via `thvm_profile_kernels_summary`).

Model: small decision tree or lookup table indexed by `(reduce_numel_bucket, out_numel_bucket)`.

## Integration Points

| File | Change |
|------|--------|
| `src/backend/metal/uop.m` | `uop_from_fused` accepts config params, generates different MSL |
| `src/backend/metal/codegen.m` | `cg_get_pipe_rs` calls beam search on cache miss |
| `src/backend/metal/profile_kernel.m` | `benchmark_kernel()` function using GPUStartTime/GPUEndTime |
| `src/tinyhvm.h` | Add `THVM_BEAM` env var check |
| Cache | Extend `cg_cache` to store optimal config per hash |

## Expected Impact

- Reduce kernel: 31us → 18-20us (match tinygrad) on M3
- Elementwise kernel: 14us → 10-12us (match tinygrad) with optimal UPCAST
- Overall: ~10-15% step time reduction from better kernel configs
- Long-term: cost model eliminates search overhead entirely

## Implementation Sequence

1. Add `benchmark_kernel()` to profile_kernel.m (time single kernel N times, return median)
2. Add LOCAL_SIZE search in `cg_get_pipe_rs` on cache miss (THVM_BEAM=1)
3. Add UPCAST parameter to UOp IR (process N elements per thread)
4. Implement full BEAM with pruning (top-4 per iteration, 3 iterations)
5. Build cost model from collected data
6. (Optional) SUP-based search using labeled superpositions

## Benchmarking Protocol

Same as tinygrad's `_time_program`:
- 3 runs per configuration
- Take median (not mean — avoids outliers)
- Clear L2 cache between runs (dispatch a large memset)
- Timeout: 10s per configuration
- Skip configurations that produce wrong results (validation check)
