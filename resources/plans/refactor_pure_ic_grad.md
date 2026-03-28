# Refactor: Pure IC Gradient

**Status: DONE.** `backward_local` and `thvm_backward` deleted. All gradients flow
through the IC-native UOP_GRAD interaction handler.

## What Changed

- `src/grad/_.c`: 542 → 42 lines. Only `thvm_grad` + `thvm_grad_multi` remain.
- `reduce_id` moved to `src/reduce/_.c`
- RELU backward uses `y` (output) not `at` (input) — safe in fused reduce chains
- LOG backward uses `DIV(gy, at)` — `at` still needed, stays in safety check
- EXP backward uses `MUL(gy, y)` — already safe
- SQRT backward uses `DIV(gy, 2*y)` — already safe

## GRAD Handler Optimizations

### Lazy ENSURE
The GRAD handler no longer ENSUREs all inputs at the top. Only ops that read data
(MM, RMAX, LOG, DIV, MAX) call ENSURE. This lets backward ops stay deferred,
forming longer fusable chains via tensor_materialize.

### Dead-branch skip
BIN_GRAD checks `requires_grad` on both inputs. When only one branch is live,
it uses UN_GRAD (tail call) instead of creating ADD(GRAD3_left, GRAD3_right)
and forcing reduction.

### Lazy gy (trampoline change)
For UOP_GRAD terms, the trampoline fires the interaction WITHOUT reducing arg1 (gy).
Chain rule formulas wrap gy in new lazy ops, creating fusable chains. Base case
and deposit explicitly reduce gy when needed.

## Safety Check (fused reduces)
Fused reduce chains reject ops whose backward reads intermediate data:
- **Blocked**: DIV, MAX, LOG (backward needs `at`)
- **Safe**: ADD, SUB, NEG, MUL, RELU, EXP, SQRT (backward uses `y` or leaves only)
