# Training Loop: Recursion Research

Status: **active research** — the correct encoding of training loops within TinyHVM's
interaction calculus is an open design problem.

## Current Approach

Training loops use **pure functional recursion** via standard HVM combinators:

```
train(counter)(w)(b) =
    IFZ counter
        → CTR(w, b)                              // base case: return params
        → λm.
            pred  = x @ w + b                     // forward
            loss  = mean((pred - y)²)
            grads = thvm_grad_multi_keep([w, b])   // backward → CTR(gw, gb)
            MAT grads λgw. λgb.                    // destructure bundle
                w' = w - lr * gw                   // SGD update
                b' = b - lr * gb
                train(m)(DETACH w')(DETACH b')      // recurse
```

Key combinators: `REF` (recursive reference), `LAM`/`APP` (lambda), `MAT`/`CTR` (pattern match),
`IFZ` (conditional on natural number).

### HVM4 Alignment

Interaction rules follow HVM4 semantics (`HVM4/docs/hvm/interactions`):
- **APP-MAT-CTR** for bundle destructuring (not APP-CTR shortcut)
- **CTR** for multi-value bundling (tuples)
- Standard beta reduction for APP-LAM

## Design Constraints

These are firm decisions — do not revisit:

1. **No explicit loop/state combinator** — recursion uses standard HVM REF/LAM/APP/MAT
2. **Phase 1 is an initializer** — runs once, exposes the compute frontier
3. **Phase 3 re-fires same prescheduled kernels** across loop iterations
4. **Scheduler walks the whole graph** (even behind combinators), not just root terms
5. **Running eval in a loop is the wrong model** — single reduce should complete the loop
6. **Only single iteration is planned and JIT'd** — the loop body reuses the same kernels

## How It Should Work

```
Phase 1 (once):
    Reduce the recursive program
    → Unrolls single loop iteration naturally
    → REF halts because UOPs block reductions
    → Exposes: forward compute + backward GRAD + loop combinators

Phase 2 (once):
    Schedule the exposed compute frontier
    → Walk through combinators, fuse kernels
    → Plan single iteration's worth of kernels

Phase 3 (loops):
    Reduce with dispatch enabled
    → Fire prescheduled kernels
    → Combinators sequence loop iterations
    → Same kernel IDs re-fired each iteration
    → No return to phase 1 or 2
```

## Open Questions

### DETACH
Currently used to cut backward graph between iterations. Should DETACH be necessary
if scheduling is correct? If the scheduler walks behind combinators and plans correctly,
updated parameters should naturally be fresh tensors without backward links.

### Parameter Updates
Current test uses `next_w = w - lr * gw` as functional updates with DETACH.
The alternative: a single param tensor term with `ASSIGN` update in the loop body.
But ASSIGN is phase-3-only, so it cannot sequence through phase-1 CTR/APP.

### Scheduler Scope
Should phase 2 schedule only root-reachable compute, or walk the entire heap including
terms behind REF/LAM/APP? Walking everything would schedule forward+backward in one pass
but may over-schedule unreachable branches.

### Memory Planning in Loop
The memory planner must handle kernel re-firing across iterations. Buffer reuse within
an iteration is standard; reuse across iterations (same kernel, different data) may need
the planner to recognize the loop boundary.

## Test

`test/test_tiny_linear_sgd_loop.m` — functional recursive linear SGD with:
- 4x3 weight matrix, 4-element bias
- 2 training steps (configurable via `THVM_TRAIN_STEPS`)
- CTR bundle for gradient returns
- MAT destructuring for gradient extraction

## Files

- `test/test_tiny_linear_sgd_loop.m` — functional loop test
- `src/interact/combinators.c` — APP/LAM/MAT/CTR/IFZ rules
- `src/schedule/_.c` — thvm_eval 3-phase orchestration
- HVM4 reference: `/Users/swish/src/HVM4/docs/hvm/interactions`
