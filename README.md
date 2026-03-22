# TinyHVM

Interaction-net-based runtime for tensor computation. Draws architectural inspiration from [HVM4](https://github.com/HigherOrderCO/HVM4) and operational ideas from [tinygrad](https://github.com/tinygrad/tinygrad).

## What

A minimal C runtime where:
- Computation graphs are interaction nets (confluent graph rewriting)
- Tensors are first-class opaque nodes with GPU-backed buffers
- Reduction dispatches tensor ops to GPU kernels via a pluggable backend
- No dependencies beyond system libc and Accelerate/Metal

## Build

```bash
make          # CPU backend (Accelerate BLAS)
make test     # run tests
make metal    # Metal GPU backend (macOS)
```

## Status

Phase 1 POC — forward pass of `relu(matmul(x, w) + b)`.

## See Also

- `resources/report.md` — feasibility report
- `resources/taelin_gists_index.md` — Taelin's HVM/IC gists
