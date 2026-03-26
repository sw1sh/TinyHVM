# Metal Performance Reference for Apple Silicon ML

Everything needed to make TinyHVM's Metal backend fast. Researched 2026-03-25.

---

## 1. Apple GPU Architecture

### Pipeline layout (M1/M2 — Apple Family 7/8)

Each GPU core has **128 FP32 ALUs** across 4 schedulers. Each scheduler dispatches one instruction per cycle from one SIMD (32 threads).

```
Core = 4 schedulers × 32 ALUs = 128 ALUs
     + 32 SFUs (RECIP, RSQRT, SIN, EXP2, LOG2)
```

SIMD width is always 32 threads on Apple GPUs.

### Instruction latencies

| Op | FP16 | FP32 | INT32 |
|----|------|------|-------|
| ADD | 1 cyc | 2 cyc | 1 cyc |
| MUL | 1 cyc | 2 cyc | 4 cyc |
| FMA | 1 cyc | 2 cyc | 4 cyc |
| RECIP | 6 cyc | 6 cyc | — |
| RSQRT | 8 cyc | 8 cyc | — |
| SIN | ~10 cyc | ~10 cyc | — |

Register dependencies add 0.56–0.84 cycles (16-bit) or 0.84 cycles (32-bit) in dependent chains.

### Cache hierarchy (per core)

| Level | Size | Bandwidth | Notes |
|-------|------|-----------|-------|
| Register file | ~208 KB | — | Largest of any GPU architecture |
| Instruction cache | 12 KB | — | Small |
| L1 data cache | **8 KB** | 64 B/cycle | **Extremely small** — much smaller than NVIDIA/AMD |
| Texture cache | 24 KB | — | Separate from data L1 |
| Threadgroup memory | ~60 KB physical, **32 KB API limit** | — | Shared within threadgroup |
| SIMD shuffle | — | **256 B/cycle** | Industry-leading — 4x L1 bandwidth |

**GPU-wide caches:**

| Chip | L2 | SLC (L3) |
|------|-----|----------|
| M1 | 768 KB | 8 MB |
| M1 Pro | 256 KB | 24 MB |
| M1 Max | 512 KB | 48 MB |
| M1 Ultra | — | 96 MB |
| M2 | ~1.5 MB | — |
| M2 Pro | 3 MB | 24 MB |
| M2 Max | — | 48 MB |

L2 bandwidth: >1 TB/s (M2 Pro). SLC latency: 234 ns. DRAM latency: 342 ns.

**Key insight:** Apple deliberately invested in SIMD shuffle bandwidth (256 B/cycle) over L1 cache size (8 KB). Design for intra-SIMD communication, not cache-based sharing.

### `simdgroup_matrix` — Apple's "tensor cores"

8×8 matrix multiply-accumulate using existing FP32 pipelines. Not separate fixed-function hardware like NVIDIA tensor cores — they use existing ALUs more efficiently by reducing register pressure.

- Throughput: ~101.7 FP32 FMA ops/core/cycle, ~102.5 FP16 FMA ops/core/cycle
- `MATMUL<8x8xF16>`: ~17 cycles throughput
- `MATMUL<8x8xF32>`: ~18 cycles throughput
- Supported since A14 (Apple Family 7)

### Occupancy model

| Limit | Value |
|-------|-------|
| SIMD width | 32 threads |
| Max threads/threadgroup | 1,024 |
| Max threadgroup memory | 32 KB |
| Max SIMDs/core | 88 |
| Design target min SIMDs/core | 24 |
| GPRs per SIMD-group | 128 × 32-bit |

Fewer registers → more SIMDs fit in register file → higher occupancy. Threadgroup memory >2 KB starts reducing occupancy.

At low occupancy, FP16/INT16 is significantly faster than FP32: latencies 3.9 vs 6.6 cycles (widening to 3.9 vs 11.3 for FMA at very low occupancy).

### Memory bandwidth per chip

| Chip | Cores | BW (GB/s) | Max RAM | FP32 TFLOPS |
|------|-------|-----------|---------|-------------|
| M1 | 7–8 | 68 | 16 GB | 2.6 |
| M1 Pro | 14–16 | 200 | 32 GB | ~5.3 |
| M1 Max | 24–32 | 400 | 64 GB | ~10.4 |
| M1 Ultra | 48–64 | 800 | 128 GB | ~21 |
| M2 | 8–10 | 100 | 24 GB | 3.6 |
| M2 Pro | 16–19 | 200 | 32 GB | ~6.5 |
| M2 Max | 30–38 | 400 | 96 GB | ~13.6 |
| M3 | 8–10 | 100 | 24 GB | 3.5 |
| M3 Pro | 14–18 | **150** | 36 GB | — |
| M3 Max | 30–40 | 300–400 | 128 GB | — |
| M4 | 8–10 | 120 | 32 GB | 4.3 |
| M4 Pro | 16–20 | 273 | 64 GB | — |
| M4 Max | 32–40 | 546 | 128 GB | ~18.4 |
| M5 | 8–10 | 154 | 32 GB | 4.2 |
| M5 Pro | 16–20 | 307 | 64 GB | ~8.3 |

**M3 Pro bandwidth regression:** 150 GB/s (192-bit bus) vs 200 GB/s on M1/M2 Pro (256-bit).

Measured GPU bandwidth (STREAM): M1=60, M2=91, M3=92, M4=100 GB/s (base chips).

### Unified memory

Single LPDDR pool shared by CPU, GPU, and Neural Engine. Zero-copy between CPU and GPU — no PCIe bottleneck. Apple Silicon's biggest advantage for ML: models that exceed discrete GPU VRAM can run in-place.

Drawback: bandwidth shared across all consumers.

### M3/M4 improvements (Apple Family 9)

1. **Dynamic caching:** Registers dynamically allocated/deallocated over shader lifetime. Register file acts as cache. Significantly increases occupancy without programmer effort.

2. **Flexible on-chip memory:** All on-chip memory (registers, threadgroup, tile, stack, buffer data) shares a unified cache. GPU dynamically adjusts occupancy to prevent spilling.

3. **Parallel ALU pipelines:** FP16, FP32, and INT operations can execute in parallel to a greater degree. Up to 2x ALU utilization when mixing types.

**Implication for compute shaders:** On Family 9+, consider reading directly from device/constant buffers instead of copying to threadgroup memory if working set fits in cache. Hardware handles occupancy automatically.

### M5 neural accelerators (Apple Family 10)

- Dedicated neural accelerator in each GPU core
- FP16: ~1024 FLOPS/cycle/core → estimated ~70 TFLOPS on M5 Max
- INT8: ~2048 OPS/cycle/core → estimated ~130 TOPS on M5 Max
- Optimal matrix size: 32×32 minimum
- Accessed via Metal 4 Tensor APIs and Metal Performance Primitives (MPP)
- Up to 4x speedup over M4 for LLM time-to-first-token

---

## 2. Metal Compute Best Practices

### Threadgroup sizing

- Get SIMD width: `pipelineState.threadExecutionWidth` (always 32)
- Get max: `pipelineState.maxTotalThreadsPerThreadgroup`
- Must be a multiple of 32
- **256 threads/threadgroup** is a good default for 1D workloads
- Max: 1,024 threads/threadgroup

### Memory access patterns

- **Coalescing:** Contiguous threads must access contiguous addresses. 128-byte cache line — align to it.
- **Bank conflicts:** Threadgroup memory has banking. Multiple threads hitting same bank serialize.
- **L1 is only 8 KB:** Working set must be tiny or you miss to L2/DRAM.
- **Family 9+:** On-chip memory is flexible — threadgroup memory acts more like a cache.

### Address spaces

| Space | Use case | Properties |
|-------|----------|-----------|
| `device` | Read-write buffers | Uses L1 data cache |
| `constant` | Read-only uniform data | Cached and pre-fetched, optimized for broadcast |
| `threadgroup` | Shared within threadgroup | 32 KB limit, fastest for cooperative work |

### SIMD operations — prefer these over threadgroup memory

- `simd_shuffle`: Near-zero latency within 32-thread SIMD. 256 B/cycle.
- `simd_sum`, `simd_min`, `simd_max`: Efficient SIMD-wide reductions.
- `simdgroup_matrix`: 8×8 matrix multiply-accumulate.
- `simd_shuffle_and_fill` (A15+): Optimized for sliding-window ops.

**Key principle:** Apple GPU was designed for fast intra-SIMD communication but slower inter-SIMD (threadgroup) communication. Prefer SIMD-scoped operations.

### Atomics — avoid when possible

- **FP32 atomics are emulated in software** on Apple GPUs — terrible performance.
- Device memory atomics: ~59 ns latency (M2 Pro).
- Strategy: use SIMD reductions then single atomic per SIMD, or restructure entirely.
- metal-flash-attention split backward into two kernels to avoid FP32 atomics.

### Avoiding GPU stalls

- Triple-buffer dynamic data (CPU writes N+1 while GPU reads N).
- Submit command buffers before GPU finishes previous work.
- Use `MTLEvent` for fine-grained sync instead of `waitUntilCompleted`.
- Avoid synchronous GPU readback — pipeline the work.
- Use indirect command buffers for GPU-driven dispatch (Metal 3+).
- Cache pipeline state objects — don't recreate per dispatch.

---

## 3. MPS for ML

### What it provides

- `MPSMatrixMultiplication`: Highly optimized GEMM (C = αAB + βC)
- `MPSCNNConvolution`, `MPSCNNPooling`, `MPSCNNBatchNormalization`
- `MPSCNNNeuron`: ReLU, sigmoid, tanh, etc.
- `MPSCNNSoftMax`, `MPSMatrixSoftMax`

### MPS Graph vs individual MPS ops

**Individual:** One op per dispatch. Fine-grained control but high dispatch overhead.

**MPSGraph:** Chains operations, applies **kernel stitching** — recognizes adjacent operations and fuses them into hand-tuned kernels via Metal compiler. GeLU fusion: 10–50x speedup vs unfused.

Use MPSGraph for complex graphs. Use individual MPS kernels for single-op fine-grained control.

### Performance

| Chip | MPS SGEMM (TFLOPS) | Theoretical peak |
|------|-------------------|-----------------|
| M1 | 1.36 | 2.6 |
| M2 | 2.24 | 3.6 |
| M3 | 2.47 | 3.5 |
| M4 | 2.9 | 4.3 |
| M1 Max | ~7 | ~10.4 |

Hand-written Metal kernels typically reach ~50% of MPS for matmul. metal-flash-attention reaches ~83% of theoretical peak (custom GEMM, not MPS).

### When to use MPS vs custom kernels

**Use MPS:** Standard ops (matmul, conv, pooling, norm). You want Apple's ongoing optimization for new hardware.

**Use custom kernels:** Non-standard ops, custom fusion patterns, quantized ops with custom formats, thin matrices where MPS overhead dominates.

### Limitations

- Limited to standard layer types
- 256 MB per-MTLBuffer limit for MPSMatrixMultiplication
- Cannot change weights after kernel creation
- Some ops still fall back to CPU in PyTorch MPS

---

## 4. GEMM on Metal

### Best approaches

1. **simdgroup_matrix:** 8×8 multiply-accumulate. Essential for competitive performance.
2. **Multi-level tiling:** Grid → threadgroup → SIMD (registers via simdgroup_matrix).
3. **Threadgroup memory as explicit L1:** Load A/B tiles into 32 KB threadgroup memory, compute partial products.

### Block sizes that work

| Project | Strategy |
|---------|----------|
| metal-flash-attention | 16–32 along parallelization, 80–128 along traversal (heavily warped) |
| Burn framework | (8,8,8) tiles aligned with simdgroup_matrix |
| ggml | 32–64 threads for quantized matmul |

General: tile size must balance register pressure, threadgroup memory (32 KB), and parallelism.

### Measured GEMM performance

| Implementation | Chip | GFLOPS | % of peak |
|---------------|------|--------|-----------|
| MPS GEMM | M1 Max | ~7,000 | 67% |
| Custom (metal-flash-attention) | M1 Max | ~8,800 | **83%** |
| Custom (amx-benchmarks) | M1 Max | 9,258 | **87%** |
| MPS GEMM | M4 | 2,900 | 67% |
| Accelerate/AMX (CPU) | M1 Max | 2,746 | — |
| OpenBLAS/NEON (CPU) | M1 Max | 362 | — |

### Register pressure management

- 128 GPRs per SIMD-group at 32 bits each.
- At D>128: neither operand blocks nor accumulator fit in registers.
- metal-flash-attention: embrace intentional register spilling with asymmetric block shapes.
- Use FP16 to halve register usage.
- Family 9+: dynamic register allocation mitigates this automatically.

---

## 5. Kernel Fusion on Metal

### Approaches

1. **Single-kernel fusion:** Write one compute kernel for matmul+bias+ReLU. Eliminates intermediate memory traffic.

2. **MPSGraph stitching:** MPSGraph recognizes adjacent operations and fuses into hand-tuned kernels.

3. **Runtime codegen:** Generate MSL source at runtime for specific operation chains, compile with Metal API. Used by metal-flash-attention, tinygrad, MLX, and TinyHVM.

### Metal function constants

Declare `constant bool`/`constant int` in MSL. Set at pipeline creation via `MTLFunctionConstantValues`. Metal compiler folds constants and eliminates dead code paths.

```metal
constant bool use_relu [[function_constant(0)]];
constant uint op_type [[function_constant(1)]];

kernel void fused_op(...) {
    float val = /* compute */;
    if (use_relu) val = max(val, 0.0f);  // compiled away if false
}
```

Advantages over macros: one uber-shader, specialize at pipeline creation. Reduces compile time.

### Indirect compute commands (Metal 3+)

Encode dispatches into indirect command buffers on the GPU. Build once, reuse. GPU-driven dispatch eliminates CPU-GPU sync overhead.

---

## 6. Key Open Source Projects

### metal-flash-attention (philipturner, 594 stars)

FlashAttention on Metal. 83% ALU utilization on M1 Max.

**GEMM kernel design:**
- Dynamic block sizes based on head dimension D
- Three-dimensional blocking for register pressure management
- Heavily warped aspect ratios: 16–32 × 80–128
- At D=256: intentional register spilling (still 61–71% utilization)

**Backward pass:** Split into two kernels (dQ and dK/dV) to avoid FP32 atomics. 7 GEMMs instead of 5 but 100% parallelization efficiency.

**BF16 emulation** for pre-Family-9 devices. Runtime code generation for compiler compatibility.

### MLX (Apple, ml-explore/mlx)

Apple's own ML framework for Metal.

- Lazy evaluation: arrays materialized on demand
- Dynamic computation graphs (like PyTorch)
- Unified memory: no explicit CPU-GPU transfers
- `mx.compile`: fuses multiple kernel launches into one
- `mx.fast`: tuned RMSNorm, attention, etc.
- Quantized matmul kernels:
  - `qvm_split_k`: large K, splits reduction across threadgroups
  - `qmv_quad`: small K, Metal quadgroup shuffle ops
  - Template specialization for 2/4/8-bit widths

### ggml Metal backend

- All MSL kernels in single `ggml-metal.metal` file
- One kernel per quantization type (`kernel_mul_mat_q4_0`, etc.)
- `ggml_metal_graph_compute()` distributes nodes across command buffers
- Concurrent kernel execution for independent ops
- Q4_K most popular: ~4.5 bits/weight, super-block size QK_K=256
- Function constants for kernel specialization
- Runtime compilation with pipeline cache

### tinygrad Metal backend

- `MetalLanguage` renderer: UOp IR → MSL source strings
- Compiles custom kernel for every operation (extreme shape specialization)
- Fusion during scheduling (ReduceOps with elementwise children)
- Supports `simdgroup_matrix` (wmma) in Metal kernels

### Other notable projects

| Project | What |
|---------|------|
| **metalQwen3** | Complete Qwen3 transformer in custom Metal kernels (RMSNorm, QuantMatMul, Softmax, SwiGLU, RoPE, MHA). No CPU fallbacks. |
| **Burn** (Tracel AI) | Rust ML framework, CubeCL compiles to CUDA/ROCm/Metal/Vulkan/WebGPU. (8,8,8) matmul tiles. |
| **Draw Things** | Image generation using Metal FlashAttention. 4.6x M5-over-M4 improvement. |
| **flash_attn_metal_cpp** | FlashAttention via metal-cpp headers. |

---

## 7. Profiling and Optimization

### Xcode Metal Debugger

- **Shader Cost Graph:** Flame graph with per-line cost annotations and GPU instruction counts.
- **Performance Heat Maps:** Execution cost, thread divergence, instruction count.
- **Shader Execution History:** Timeline of SIMD group execution. Loop detection.

### GPU counters on Apple Silicon

**Occupancy:**
- Total Occupancy (SIMDs concurrently running per core)
- Occupancy Manager Target (GPU's dynamic goal — <100% means GPU restricts to prevent cache thrashing)

**ALU:**
- FP32 Limiter/Utilization
- FP16 Limiter/Utilization

**Memory:**
- L1 Eviction Rate (spills from register/threadgroup/tile/stack)
- L1 Load/Store Bandwidth (by memory type)
- GPU Last Level Cache Utilization/Limiter
- MMU Limiter (TLB misses)

**Key concept:** Utilization = work done / peak rate. Limiter = (work + stalls) / peak rate. If limiter >> utilization, stalls are the bottleneck.

### Occupancy triaging workflow

1. ALU and memory limiters low but occupancy low? → occupancy is limiting.
2. Shader Launch Limiter <1%? → workload too small.
3. Check threadgroup memory usage and Occupancy Manager Target.
4. Occupancy Manager Target <100%? → check L1 Eviction Rate.
5. Identify problem via L1 Load/Store Bandwidth breakdown.
6. Fix: smaller types, less on-chip memory, better locality.

### Runtime counters

`MTLCounterSampleBuffer` for programmatic GPU counter access. Stage boundary timings for precise start/end of compute passes.

---

## 8. Quantization on Metal

### FP16

- **M1 through M4: FP16 has SAME throughput as FP32** (1:1 ratio). Same ALU pipelines.
- FP16-to-FP32 conversion is free (zero cost).
- FP16 advantage: half register pressure, half memory bandwidth, half threadgroup memory.
- **M5: FP16 can be 2x FP32** — two FP16 instructions issued concurrently.

### BF16

- Available on **all Apple Silicon** (Family 6+, M1 onwards) in Metal.
- MSL `bfloat` type in Metal 3.1+.
- Supported in MPS and MPSGraph (macOS Sonoma+).

### INT8

- A17 Pro and M4+ add explicit INT8 op support on GPU.
- Neural Engine supports INT8 across all M-series.
- M5 neural accelerators: ~130 TOPS INT8 (M5 Max estimate).

### MLX quantization approach

- `quantized_matmul`: fused dequantize + matmul.
- Loads packed uint8, unpacks with bit-shifting, applies `w_hat = scale * val + bias`.
- Large-K: `qvm_split_k` (split reduction across 8/32 threadgroups).
- Small-K: `qmv_quad` (quadgroup shuffle).
- Template specialization for 2/4/8-bit widths.

---

## 9. Roofline Analysis

### Compute-to-bandwidth ratio (base chips)

| Chip | FP32 TFLOPS | Measured BW (GB/s) | FLOP/byte |
|------|------------|-------------------|-----------|
| M1 | 2.6 | 60 | ~43 |
| M2 | 3.6 | 91 | ~40 |
| M3 | 3.5 | 92 | ~38 |
| M4 | 4.3 | 100 | ~43 |

**Arithmetic intensity crossover: ~19–22 FLOP/byte** (practical roofline using theoretical compute, measured bandwidth).

### Where ML operations fall

**Compute-bound** (>19 FLOP/byte):
- Large GEMM / batch matmul
- Convolution with large channels
- LLM prefill (time-to-first-token)

**Bandwidth-bound** (<19 FLOP/byte):
- Elementwise ops (ReLU, GELU, add, mul, scale)
- Softmax, layer norm, batch norm
- Reductions (sum, mean)
- LLM token generation (weight loading per token)
- Small matrix operations

### Practical implications for TinyHVM

1. **Most inference is bandwidth-bound.** Quantization (FP32 → INT4) reduces memory traffic 8x → direct throughput gain.

2. **Kernel fusion matters most for bandwidth-bound ops.** Fusing ReLU into matmul saves one read+write of the intermediate. For compute-bound GEMM, fusion barely matters.

3. **FP16 on M1–M4 saves bandwidth, not compute.** Since throughput is 1:1, benefit is purely from halving data movement. On M5, FP16 also doubles compute.

4. **L2/SLC hierarchy matters.** M2 Pro L2 >1 TB/s. If tile fits in L2 (3 MB), effective bandwidth is 10x DRAM. Structure tiling to fit.

5. **Unified memory is double-edged.** No copy overhead (great), but bandwidth shared with CPU. M4 Max at 546 GB/s provides good headroom.

---

## 10. Common Pitfalls

### Performance killers

1. **`waitUntilCompleted` after every dispatch** — stalls CPU pipeline. Use callbacks/events.
2. **Too many small dispatches** — each has overhead. Fuse or batch.
3. **FP32 atomics** — emulated in software. Use SIMD reductions instead.
4. **16-byte alignment violations** — `float3` is 16-byte aligned in Metal. Packed 12-byte layout from other frameworks causes silent corruption.
5. **Threadgroup memory as cache on Family 9+** — hardware already caches flexibly. Profile before assuming benefit.
6. **Register pressure (pre-Family 9)** — excessive registers directly reduce occupancy.
7. **Threadgroup memory over SIMD shuffles** — shuffle is 4x faster than threadgroup memory on Apple GPUs.
8. **Large threadgroup memory** — >2 KB starts affecting occupancy.
9. **Not caching pipeline state objects** — pipeline creation includes shader compilation.

### CUDA → Metal gotchas

| CUDA | Metal | Pitfall |
|------|-------|---------|
| `__syncthreads()` | `threadgroup_barrier()` | Same concept, different name |
| `__shfl_sync()` | `simd_shuffle()` | Metal shuffle is faster (256 B/cycle) |
| `atomicAdd(float)` | **Not native** | Emulated, avoid at all costs |
| `printf` in kernel | **Not available** | Use shader validation or capture |
| `kernel<<<B,T>>>()` | Manual command encoding | More verbose host code |
| Warp = 32 | SIMD = 32 | Same width, different name |
| Shared memory 48–164 KB | Threadgroup memory 32 KB | Much less shared memory available |
| L1 cache 128 KB | L1 cache 8 KB | **16x less** — don't rely on caching |
| Independent address space | Unified memory | No memcpy needed, but bandwidth shared |

### Apple-specific gotchas

- **FP16 ≠ 2x FP32 on M1–M4** — same throughput. Only M5 gets 2x.
- **L1 is only 8 KB** — don't assume data will be cached.
- **M3 Pro bandwidth regression** — 150 GB/s vs 200 GB/s on M1/M2 Pro.
- **Runtime shader compilation** — first dispatch includes compilation. Cache pipeline state objects.

---

## 11. Sources

### Microarchitecture
- [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
- [Apple G13 GPU reference (dougallj)](https://dougallj.github.io/applegpu/docs.html)
- [Chips and Cheese: M2 Pro iGPU](https://chipsandcheese.com/p/a-brief-look-at-apples-m2-pro-igpu)
- [Apple vs Oranges HPC paper (arXiv:2502.05317)](https://arxiv.org/html/2502.05317v1)

### Apple tech talks / WWDC
- [Explore GPU advancements in M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111375/)
- [Learn performance best practices for Metal shaders](https://developer.apple.com/videos/play/tech-talks/111373/)
- [Discover new Metal profiling tools](https://developer.apple.com/videos/play/tech-talks/111374/)
- [Optimize Metal apps with GPU counters (WWDC20)](https://developer.apple.com/videos/play/wwdc2020/10603/)
- [Accelerate ML with MPSGraph (WWDC21)](https://developer.apple.com/videos/play/wwdc2021/10152/)

### Apple documentation
- [Calculating threadgroup and grid sizes](https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes)
- [Reducing shader bottlenecks](https://developer.apple.com/documentation/metal/performance_tuning/reducing_shader_bottlenecks)
- [Using function specialization](https://developer.apple.com/documentation/metal/using-function-specialization-to-build-pipeline-variants)
- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)

### Projects
- [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
- [philipturner/amx-benchmarks](https://github.com/philipturner/amx-benchmarks)
- [philipturner/applegpuinfo](https://github.com/philipturner/applegpuinfo)
- [ml-explore/mlx](https://github.com/ml-explore/mlx)
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- [BoltzmannEntropy/metalQwen3](https://github.com/BoltzmannEntropy/metalQwen3)

### Benchmarks / analysis
- [M5 GPU roofline analysis](https://www.michaelstinkerings.org/apple-m5-gpu-roofline-analysis/)
- [M5 neural accelerator benchmarks](https://tzakharko.github.io/apple-neural-accelerators-benchmark/)
- [Apple M5 ML research (Apple)](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Flopper Apple Silicon guide](https://flopper.io/docs/apple-silicon-explained)
- [Burn SOTA multiplatform matmul](https://burn.dev/blog/sota-multiplatform-matmul/)
- [Gimlet Labs: AI-generated Metal kernels](https://gimletlabs.ai/blog/ai-generated-metal-kernels)
