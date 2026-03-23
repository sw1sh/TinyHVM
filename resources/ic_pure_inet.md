# TinyHVM: Pure Inet Training Model

## The Problem with the Current Architecture

Training currently works like this:

```c
// Step 1: build + forward (with recording)
thvm_start_recording(ctx);
Term loss = thvm_reduce(ctx, loss_expr);
thvm_stop_recording(ctx);

// Step 2: build + backward
Term dW = thvm_reduce(ctx, thvm_grad(ctx, loss_term, W));

// Step 3: optimizer step (imperative C)
W_data -= lr * dW_data;

// Step 4: destroy the graph
thvm_reset(ctx, n_weights);
// Loop, rebuild from scratch
```

This is an **imperative wrapper around a lazy graph evaluator**. The inet is not the
program — it is a repeated scratch buffer used by a C for-loop. This misses three
properties that make inets powerful:

1. **Sharing**: weights used in both forward and backward should be the same
   node, deduplicated by DUP-SUP interaction. Currently duplicated / copied.
2. **Composition**: the optimizer step, gradient, forward, and data load are
   separate imperative phases. In a true inet they are one closed term.
3. **One-pass execution**: the full program should reduce to WNF once. Iteration
   is a fixpoint term inside the net, not a C `for`.

---

## The Pure Inet Model

The training loop is a **single closed inet term** that reduces completely:

```
program = train(W, b, DataStream, n_steps)

train = FIX(λself. λW. λb. λdata. λn.
  if n == 0
    then (W, b)
    else
      let (X, Y, data') = next(data)          // lazy batch load
      let H   = RELU(X @ W + b)               // forward
      let loss = MSE(H, Y)                     // loss
      let dW   = GRAD(loss, W)                 // backward (lazy)
      let db   = GRAD(loss, b)
      let W'   = W - lr * dW                  // optimizer (lazy)
      let b'   = b - lr * db
      self(W', b', data', n-1))               // tail-recursive
```

`reduce(program)` runs all steps. Intermediate activations (`H`, `loss`) are
erased by `TAG_ERA` propagation as lambdas close. Weights are the only persistent
tensors.

---

## What Changes vs Current Architecture

### 1. `recording` flag → removed

`recording = 1` currently gates all provenance writes (`creator_op`, `src_ids`).
In the pure inet model, provenance is **always written** — at construction time,
the reducer knows whether a tensor is on a grad path because `requires_grad`
propagates from the leaf parameters.

**Change**: replace every `if (ctx->recording)` guard with
`if (md->requires_grad_output)` (propagated from inputs), or simply always write
provenance. Overhead is zero for tensors with `requires_grad == 0`.

### 2. `thvm_reset` → eraser propagation

`thvm_reset` manually free tensors above a watermark. In the pure inet, activations
are freed when their lambda closes and the term becomes unreachable (erased by a
`TAG_ERA` propagation).

**Short-term PoC approach**: keep `thvm_reset` as a GC hint but don't require it
for correctness. The net produces updated weight tensors as its WNF; reset just
recycles intermediate slots.

### 3. Training loop → `TAG_REF` / `TAG_LAM` / `TAG_APP`

The C `for` loop becomes a recursive term using:

| Tag | Role |
|-----|------|
| `TAG_LAM(var, body)` | Lambda abstraction |
| `TAG_APP(fun, arg)` | Lambda application |
| `TAG_REF(name)` | Named global (for fixpoint) |
| `TAG_SUP` / `TAG_DP` | Duplication when a value is used twice |

Interaction rules needed in `thvm_reduce`:
```
APP(LAM(x, body), arg) → body[x := arg]    // beta reduction
DP0(SUP(a, b))         → a                  // left projection
DP1(SUP(a, b))         → b                  // right projection
APP(ERA, arg)          → ERA                // erase
REF(name)              → defs[name]         // unfold (for fixpoint)
```

These are the standard HVM interaction rules. `TAG_LAM/APP` values are heap entries
with 2 slots each (same structure as binary tensor ops).

### 4. `UOP_LOAD(stream, idx)` — lazy data batches

```
TAG_TOP(UOP_LOAD, loc)  where heap[loc] = {data_ptr, idx_term}
```

When reduced: materializes batch `idx` from the data source. `idx` can itself be
a lazy term (the step counter). The net pulls batches on demand.

### 5. `UOP_ITE` — conditional (for loop termination)

```
UOP_ITE(cond, then_term, else_term)
```

When reduced: reduces `cond` → if non-zero, reduces `then_term`; else reduces
`else_term`. Bottom of the recursion returns `(W, b)` as a final value.

---

## PoC Implementation Plan

### Phase 0 — Already Works (Demonstrate)

Show that **one training step** expressed as a single lazy graph reduces correctly
today, with no `start_recording`/`stop_recording` separation:

```c
ctx->recording = 1;  // always on

// Build the full computation graph (all lazy TAG_TOP terms)
Term H    = thvm_op(ctx, UOP_RELU, thvm_op(ctx, UOP_MM, x_t, W_t), term_era());
Term loss = mse_lazy(ctx, H, y_t);    // UOP_SUM of squared diff
Term dW   = thvm_grad(ctx, loss, W_t);      // lazy GRAD node
Term W_new = thvm_op(ctx, UOP_SUB, W_t,
               thvm_op(ctx, UOP_MUL, lr_t, dW));

// ONE reduction runs everything: forward → grad → update
Term result = thvm_reduce(ctx, W_new);
// result is TAG_TEN(new_W_id) — updated weights
```

**Target**: `test/test_inet_step.m` passes, demonstrating the entire
forward+backward+update as a single reduction.

---

### Phase 1 — Always-On Provenance (remove `recording` flag)

**Goal**: remove the `ctx->recording` guard everywhere. Provenance is always
written when `requires_grad` propagates.

**Changes**:

1. `tinyhvm.h`: remove `recording` field from `TinyHVM` struct (or keep as a
   `no_grad` hint for inference speed).
2. `tinyhvm.c`: replace every `if (ctx->recording)` provenance block with
   `if (ma->requires_grad || mb->requires_grad)` — already computed to set
   `md->requires_grad`.
3. Remove `thvm_start_recording` / `thvm_stop_recording` from the API.
4. Update all tests and `beautiful_mnist.m`.

**Risk**: ~5% overhead for tensors with `requires_grad=0` (they now write 0 to
`creator_op` unconditionally). Acceptable.

---

### Phase 2 — LAM / APP / REF Interaction Rules

**Goal**: add the 5 interaction rules to `thvm_reduce`. This gives the reducer
the ability to reduce lambda applications — the building blocks of the training loop.

**New tags** (already in the tag enum):
```c
#define TAG_LAM  0  // lambda abstraction: heap[loc] = {var_placeholder, body}
#define TAG_APP  1  // application: heap[loc] = {function, argument}
```

**New interaction cases** in `thvm_reduce → case TAG_TOP`:
```c
case UOP_LAM_APP: {  // or: special-case TAG_APP in the tag dispatch
    // APP(LAM(x, body), arg) → body with x substituted by arg
    Term fun = thvm_reduce(ctx, heap_read(ctx, loc));
    if (term_tag(fun) == TAG_LAM) {
        u64 lam_loc = term_val(fun);
        Term var     = heap_read(ctx, lam_loc);     // TAG_VAR placeholder
        Term body    = heap_read(ctx, lam_loc + 1);
        // Substitute: write arg at var's heap location
        heap_set(ctx, term_val(var), heap_read(ctx, loc + 1));
        MEMO_RETURN(thvm_reduce(ctx, body));
    }
    ...
}
```

**TAG_REF** (global definition lookup):
```c
// ctx->defs: hash map from u64 name → Term (heap loc of body)
case TAG_REF:
    MEMO_RETURN(thvm_reduce(ctx, ctx->defs[ext/*name*/]));
```

---

### Phase 3 — UOP_ITE and UOP_LOAD

**UOP_ITE**:
```c
case UOP_ITE: {
    Term cond = thvm_reduce(ctx, heap_read(ctx, loc));
    // cond should be TAG_NUM
    if (term_as_u32(cond) != 0)
        MEMO_RETURN(thvm_reduce(ctx, heap_read(ctx, loc + 1)));
    else
        MEMO_RETURN(thvm_reduce(ctx, heap_read(ctx, loc + 2)));
}
```

**UOP_LOAD**:
```c
case UOP_LOAD: {
    Term src = heap_read(ctx, loc);     // data source (pointer to f32 array + shape)
    Term idx = thvm_reduce(ctx, heap_read(ctx, loc + 1));
    u32 batch_idx = term_as_u32(idx);
    // Materialize batch at batch_idx
    MEMO_RETURN(thvm_tensor(ctx, data_src + batch_idx * batch_stride, batch_shape));
}
```

---

### Phase 4 — PoC MLP Training Loop as One Inet

**Target test**: `test/test_inet_mlp.m`

```c
// Build the training inet once
Term program = build_train_loop(ctx, W, b, data_stream, n_steps);

// Reduce to WNF — runs all n_steps
Term (W_final, b_final) = thvm_reduce(ctx, program);

// Eval on test set
f32 acc = eval(ctx, W_final, b_final, test_data);
assert(acc > 0.9f);
```

`build_train_loop` builds the recursive term using LAM/APP/REF:
```c
// Define 'train' in the global def table
thvm_define(ctx, "train", build_train_body(ctx));
// Seed with initial weights and data
Term program = thvm_app(ctx,
    thvm_app(ctx, thvm_ref(ctx, "train"), W),
    thvm_app(ctx, b, thvm_app(ctx, data_stream, n_steps_term)));
```

---

## Invariants of the Pure Inet Model

1. **Single reduction**: `thvm_reduce(program)` is called once. It may internally
   call `thvm_reduce` recursively (REF unfolding), but from the caller's perspective
   it is one call.
2. **No explicit backward pass**: `GRAD(y, gy, x)` is just a lazy TOP node built
   alongside the forward graph. It reduces on demand when the result is needed.
3. **No recording flag**: provenance is written whenever `requires_grad` is true.
4. **Weights persist via sharing**: W and b appear in both numerator and denominator
   of the `GRAD` node. The DUP-SUP mechanism (TAG_SUP / TAG_DP) ensures they are
   reduced once and shared correctly.
5. **Erasure frees activations**: intermediate tensors (H, loss, etc.) are erased
   by TAG_ERA propagation when the lambda that produced them closes.
