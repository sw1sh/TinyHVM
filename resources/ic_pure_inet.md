# TinyHVM: Pure Inet Training Model

## Status

| Component | Status |
|-----------|--------|
| `enter/apply` AP reducer (`thvm_reduce`) | ✅ **Implemented** (HVM4 `wnf/_.c` style) |
| TAG_TOP frame stack — arg0→arg1 staged | ✅ Implemented |
| `MEMO_RETURN` uses `thvm_reduce` | ✅ Fixed (was raw `thvm_interact` loop) |
| `TAG_LAM` / `TAG_APP` beta-reduce | ✅ Handled in apply phase |
| `TAG_DP0` / `TAG_DP1` / `TAG_SUP` | ✅ Handled in apply phase |
| `UOP_ASSIGN`, `UOP_ITE`, `UOP_LOAD_BATCH` | ✅ Implemented |
| Always-on provenance / no `recording` flag | ✅ `recording` kept as inference `no_grad` hint |
| Full training loop as one inet term | 🔲 Future (§Phase 3) |
| Fixpoint term / `TAG_REF` global defs | 🔲 Future |
| Eraser propagation frees activations | 🔲 Future |

---

## How the AP Reducer Works (Implemented)

`thvm_reduce` mimics HVM4's `wnf/_.c` **enter/apply trampoline**. Two phases,
one explicit frame stack, no C recursion for tensor ops.

### ENTER phase

Walk the head, pushing eliminators as frames, descend into the strict arg.

```
  enter(TAG_TEN / ERA / NUM / LAM / SUP)  →  goto apply         // WNF, done
  enter(TAG_TOP(uop, loc))  →  push frame(loc); enter(heap[loc+0])
  enter(TAG_APP(loc))       →  push frame(loc); enter(heap[loc+0]) // fun
  enter(TAG_DP0/1(loc))     →  push frame(loc); enter(heap[loc+0]) // sup
  enter(other)              →  fire thvm_interact inline; enter(result)
```

For `TAG_TOP`: enter descends into `heap[loc+0]` (the strict/left argument).

### APPLY phase

Pop frames, dispatch on the WHNF result in hand.

```
  TAG_TOP frame  +  TEN  whnf →  write heap[loc+0]; check heap[loc+1]:
                                  if ready (any WNF)  → fire thvm_interact
                                  else push TAG_TOP1 sentinel; enter arg1
  TAG_TOP1 frame +  any  whnf →  write heap[loc+1]; reconstruct TAG_TOP; fire
  TAG_APP  frame +  LAM  whnf →  beta-reduce; enter(body)
  TAG_APP  frame +  SUP  whnf →  APP-SUP rule; continue apply
  TAG_DP0/1     +  SUP  whnf →  DUP-SUP annihilate; enter result
  TAG_DP0/1     +  LAM  whnf →  DUP-LAM; enter result
  stuck                       →  whnf = frame; continue
```

**Analogue to HVM OP2**: `OP2(x, y)` in HVM enters `x`; when `x=NUM`, pushes
`F_OP2_NUM` and enters `y`; when `y=NUM`, fires. In TinyHVM: `TAG_TOP(uop, loc)`
enters `heap[loc+0]`; when `arg0=TEN`, pushes `TAG_TOP1` and enters `heap[loc+1]`;
when `arg1=WNF`, fires `thvm_interact`.

### `MEMO_RETURN`

```c
#define MEMO_RETURN(result) do {               \
    Term _r = (result);                        \
    if (term_tag(_r) == TAG_TOP)               \
        _r = thvm_reduce(ctx, _r);            \  // ← enter/apply, not raw interact
    ctx->reduce_memo[memo_loc] = _r;           \
    return _r;                                 \
} while(0)
```

The old `while(TAG_TOP) thvm_interact(...)` broke out early when the sub-term
was not immediately ready. The result was left un-memoized, causing infinite
re-entry of the same `TAG_TOP`. Fixed.

---

## The Current Training Pattern

```c
// Each training step (imperative C shell around the lazy graph):
thvm_start_recording(ctx);           // sets recording = 1
Term loss = thvm_reduce(ctx, loss_expr);
thvm_stop_recording(ctx);

Term dW = thvm_reduce(ctx, thvm_grad(ctx, loss, W));

// Optimizer (still imperative, one memcpy per weight):
W_data -= lr * dW_data;
thvm_reset(ctx, n_weights);          // recycle activation slots
```

This works. The AP reducer means every `thvm_reduce` correctly stages args
via the frame stack — no recursive arg-resolution, no stack overflow on deep
graphs.

---

## The Problem with the Current Architecture

The training loop is an **imperative C shell** around what should be a single
closed term. This misses three things:

1. **Sharing**: weights used in forward and backward should be one node,
   shared via DUP-SUP. Currently copied/duplicated at the C level.
2. **Composition**: optimizer, grad, forward, data load are separate C phases.
   In a true inet they are one term.
3. **One-pass execution**: iteration is a fixpoint term inside the net, not a
   C `for` loop.

---

## Pure Inet Model (Future)

The training loop as a **single closed inet term**:

```
train = FIX(λself. λW. λb. λdata. λn.
  if n == 0
    then (W, b)
    else
      let (X, Y, data') = next(data)          // UOP_LOAD_BATCH: lazy
      let H    = RELU(X @ W + b)              // forward
      let loss = MSE(H, Y)
      let dW   = GRAD(loss, W)                // lazy GRAD node
      let db   = GRAD(loss, b)
      let W'   = ASSIGN(W, W - lr * dW)       // in-place via UOP_ASSIGN
      let b'   = ASSIGN(b, b - lr * db)
      self(W', b', data', n-1))               // tail-recursive
```

`thvm_reduce(program)` runs every step. Activations erased by ERA propagation
as lambdas close. Weights are the only persistent tensors.

### Phase 1 — Always-on provenance

Remove `ctx->recording` guard. Provenance written whenever `requires_grad`
propagates from inputs. Keep field as `no_grad` inference hint.

**Risk**: negligible overhead for `requires_grad=0` tensors.

### Phase 2 — `TAG_REF` definition table

```c
// Register named body in ctx->defs[]
thvm_define(ctx, "train", build_train_body(ctx));

// Build recursive term
Term prog = thvm_app(ctx, thvm_ref(ctx, "train"), W_initial);
Term result = thvm_reduce(ctx, prog);
```

`TAG_REF(name)` unfolds to the body via one AP rule in the apply phase:
```
enter(TAG_REF(name))  →  enter(ctx->defs[name])
```

### Phase 3 — Full MLP Training Loop as One Inet

Target test: `test/test_inet_mlp.m`

```c
Term program = build_train_loop(ctx, W, b, data_stream, n_steps);
Term (W_final, b_final) = thvm_reduce(ctx, program);
f32 acc = eval(ctx, W_final, b_final, test_data);
assert(acc > 0.90f);
```

---

## Invariants of the Pure Inet Model

1. **Single reduction** — `thvm_reduce(program)` called once.
2. **No explicit backward** — `GRAD(y, gy, x)` is a lazy TAG_TOP alongside
   the forward graph. Reduces on demand.
3. **No `recording` flag** — provenance always-on via `requires_grad`.
4. **`UOP_ASSIGN` for weight mutation** — the only controlled side effect.
   Reduces `src`, blits into `dst`'s existing buf, returns same `TAG_TEN`.
   Analogue: tinygrad's `UOP_STORE` / `Tensor.assign()`.
5. **ERA frees activations** — intermediate tensors erased when their lambda
   closes (future; currently `thvm_reset` handles this manually).

---

## §Side Effects: `UOP_ASSIGN`

```
UOP_ASSIGN(dst_term, src_term)
  → reduce src_term to TAG_TEN(src_id)
  → blit src buffer into dst buffer  (GPU memcpy, no new allocation)
  → return dst_term unchanged          (same TAG_TEN, same buf_id)
```

Optimizer step with ASSIGN:

```
let dW  = GRAD(loss, W)
let W'  = ASSIGN(W, W - lr * dW)   // in-place; W' identifies same buf as W
self(W', b', data', n-1)
```

`UOP_ASSIGN` is a **performance annotation**, not a correctness requirement.
The fully functional version (new buffers each step) is correct and simpler
to implement first.
