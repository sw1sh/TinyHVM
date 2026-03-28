# Refactor: Pure IC Gradient (delete backward_local)

## Problem

Two parallel gradient implementations exist:
1. **IC-native** (interact/_.c UOP_GRAD handler) — lazy TAG_TOP terms, driven by trampoline
2. **Eager tape** (grad/_.c backward_local) — 500 lines of imperative reverse walk, calls metal_mul_reduce_sum directly, has its own gradient accumulator array

The eager tape is a hack. It bypasses the IC, prevents fusion, and has known provenance bugs with fused reduces. It duplicates every gradient rule from the interact handler.

## Plan

### Step 1: Delete backward_local and thvm_backward

- Remove lines 44-542 from `src/grad/_.c` (keep thvm_grad + thvm_grad_multi)
- Remove `thvm_backward` from `src/tinyhvm.h`
- Update old test files that call `thvm_backward` to use `thvm_grad_multi`
- Remove the `reduce_id` and `grad_accum` helpers

### Step 2: Fix RELU backward to use output not input

In `src/interact/_.c`, RELU backward currently uses `at` (input):
```c
Term mask = thvm_op(ctx, UOP_CMP, at, thvm_tensor(ctx, &z, SHAPE(1)));
```
Change to use `y` (output): `y > 0 iff at > 0` for RELU.
```c
Term mask = thvm_op(ctx, UOP_CMP, y, thvm_tensor(ctx, &z, SHAPE(1)));
```
This allows RELU in fused reduce chains — the virtual intermediate for RELU input
has no data, but `y` (the real output tensor) IS available.

Similarly check: LOG backward uses `at` → needs `exp(y)` alternative. DIV uses both
inputs. SQRT uses `at`. These need case-by-case fixes or must stay outside fused reduces.

### Step 3: Remove/relax safety check

With RELU using `y` instead of `at`, remove RELU from the safety check list.
Keep LOG/DIV/SQRT/CMP in the check (they still need `at`).

### Step 4: Verify

- All gradient parity tests pass (MLP, 2-conv CE, conv+pool+CE)
- Training converges (CNN pool → 80%+)
- Dispatch count drops (the forward loss SUM chain fuses more)

## Expected dispatch reduction

The loss chain `SUM(MUL(one_hot, LOG(MAX(DIV(e, expand(SUM(e))), eps))))` currently
dispatches ~8 kernels. With RELU safe for fusion, the outer SUM can fuse with more
of the chain. Expected: ~4-6 dispatches for loss, saving 2-4 per step.

The backward chain's lazy gy (already implemented) creates longer fusable chains
when RELU is safe. Expected: additional 4-8 dispatch savings per step.

Total expected: 106 → ~90-96 dispatches for 2-conv+pool.
