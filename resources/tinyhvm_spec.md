# TinyHVM Spec — Interaction Calculus for Tensor Programs

## Overview

TinyHVM is an ML runtime built on the **Interaction Calculus** (IC).
All computation — tensor algebra, autograd, scheduling, training loops,
data loading — is expressed as a single IC term that reduces to the result.

It combines two ideas:
1. **HVM**: general-purpose computation via interaction net reduction (LAM/APP, DUP/SUP, ERA)
2. **tinygrad**: tensor programs as a minimal UOp DAG, scheduled and lowered to GPU kernels

The key insight: tinygrad's UOps are **sugar nodes** in the IC, just like HVM's
natural numbers and lists are sugar on top of LAM/APP. Every UOp has interaction
rules defining how it reduces (when applied to concrete tensors) and how it
interacts with DUP (cloning → differentiation) and ERA (erasure → memory freeing).

---

## 1. Term Grammar

A TinyHVM term extends the IC with tensor and scheduling primitives:

```
Term ::=
  ─── IC Core ───
  | VAR: Name                          -- variable (affine, global scope)
  | LAM: "λ" Name "." Term            -- lambda
  | APP: "(" Term " " Term ")"        -- application
  | SUP: "&" Label "{" Term "," Term "}"  -- superposition
  | DP0: Name "₀"                     -- left projection of thunk
  | DP1: Name "₁"                     -- right projection of thunk
  | ERA: "*"                           -- eraser
  | REF: "@" Name                      -- global definition reference
  | INC: "↑" Term                      -- increment (delay)
  | DRY: "^" Term                      -- stuck application
  | MAT: "λ{" cases "}"               -- pattern match
  | USE: "λ{" body "}"                -- unit eliminator

  ─── Tensors ───
  | BUF: size, dtype, device           -- allocated buffer (leaf)
  | TEN: tensor_id                     -- realized tensor handle (WNF)

  ─── Movement Ops (no compute, modify View only) ───
  | RESHAPE:  (T, shape')
  | EXPAND:   (T, shape')             -- broadcast size-1 axes
  | PERMUTE:  (T, π)                  -- reorder axes
  | PAD:      (T, before, after)      -- zero-pad
  | SHRINK:   (T, begin, end)         -- slice/crop
  | FLIP:     (T, axes)               -- reverse along axes
  | CAT:      (T₀, T₁, ..., axis)    -- concatenate
  | INDEX:    (T, i₀, i₁, ...)       -- advanced indexing

  ─── Elementwise Ops (unary/binary/ternary, like HVM's OP2 on NUM) ───
  | OP1: op, (A,)                      -- unary:  RECIP, CAST, BITCAST, COPY
  | OP2: op, (A, B)                    -- binary: ADD, MUL, MAX, MOD, IDIV,
  |                                    --         CMPL, CMPN, XOR, OR, AND,
  |                                    --         SHR, SHL
  | OP3: op, (P, A, B)                -- ternary: WHERE

  ─── Reduce Ops ───
  | REDUCE: (T, op, axes)             -- op ∈ {Add, Max, Mul}

  ─── Function Ops ───
  | CALL:     (body, a₀, a₁, ...)    -- substitute Params in body
  | PARAM:    slot, dtype, device      -- placeholder
  | TUPLE:    (v₀, v₁, ...)          -- pack multiple returns
  | GETTUPLE: (T, idx)                -- extract from tuple

  ─── Store/Ordering ───
  | STORE:  (buf, val)                -- write val → buf
  | AFTER:  (buf, deps...)            -- ordering dependency
  | SINK:   (stores...)               -- collect side effects

  ─── Scheduling/Codegen ───
  | RANGE: (bound, type)              -- iterator 0..bound
  | END:   (body, range)              -- close a range loop
  | LOAD:  (idx,)                     -- read element from buffer
  | SPECIAL: (bound, name)            -- GPU thread index
  | BARRIER: (deps...)                -- workgroup sync
  | WMMA:  (A, B, acc, config)        -- tensor core matmul

  ─── Autograd ───
  | GRAD: (y, gy, x)                  -- ∂y/∂x weighted by gy
  | DETACH: (T,)                      -- stop gradient propagation
```

Scalars are 0-dimensional tensors. There is no separate scalar type.
`CONST(3.14, f32)` is sugar for `BUF(1, f32, cpu)` with `STORE(buf, 3.14)`.

### Derived (decomposed) ops

These are NOT primitive — they desugar into the primitives above:

| Derived | Decomposition |
|---------|--------------|
| `NEG(A)` | `OP2(MUL, A, CONST(-1))` |
| `SUB(A,B)` | `OP2(ADD, A, NEG(B))` |
| `DIV(A,B)` | `OP2(MUL, A, OP1(RECIP, B))` |
| `EXP2(A)` | polynomial approx via `MUL`, `ADD` |
| `LOG2(A)` | exponent extract + polynomial |
| `SQRT(A)` | `EXP2(OP2(MUL, CONST(0.5), LOG2(A)))` |
| `MM(A,B)` | `REDUCE(OP2(MUL, RESHAPE(A,[M,K,1]), RESHAPE(B,[1,K,N])), Add, [1])` |
| `SOFTMAX(x)` | max → sub → exp → sum → div composition |

---

## 2. Interaction Rules

### 2.1 IC Core (inherited from HVM4)

Full inventory of HVM4 interactions, all inherited by TinyHVM:

#### Lambda / Application
| Active Pair | Rule | Effect |
|---|---|---|
| APP-LAM | `(λx.f a) → x←a; f` | β-reduce |
| APP-SUP | `(&L{f,g} a) → !A &L=a; &L{(f A₀),(g A₁)}` | overlap |
| APP-ERA | `(* a) → *` | erase |
| APP-INC | `(↑f x) → ↑(f x)` | delay |
| APP-DRY | `(^(f x) a) → ^(^(f x) a)` | stuck |
| APP-CTR | `(ctr a) → ^(ctr a)` | stuck |
| APP-NAM | `(name a) → ^(name a)` | stuck |

#### Pattern Match (MAT)
| Active Pair | Rule | Effect |
|---|---|---|
| APP-MAT-NUM | `(λ{#a:h;m} #a) → h` | match |
| APP-MAT-NUM | `(λ{#a:h;m} #b) → (m #b)` | mismatch, try next |
| APP-MAT-CTR | `(λ{#K:h;m} #K{a,b}) → (h a b)` | match constructor |
| APP-MAT-CTR | `(λ{#K:h;m} #L{a,b}) → (m #L{a,b})` | mismatch |
| APP-MAT-SUP | `(λ{..} &L{a,b}) → !H &L=h; !M &L=m; &L{..}` | distribute |
| MAT-INC | `(λ{..} ↑x) → ↑(λ{..} x)` | delay |

#### DUP (Superposition / Collapse)
| Active Pair | Rule | Effect |
|---|---|---|
| DUP-LAM | `!F &L=λx.f → F₀←λx0.G₀; F₁←λx1.G₁; x←&L{x0,x1}; !G &L=f` | entangle |
| DUP-SUP (same) | `!X &L=&L{a,b} → X₀←a; X₁←b` | annihilate |
| DUP-SUP (diff) | `!X &L=&R{a,b} → !A&L=a; !B&L=b; X₀←&R{A₀,B₀}; X₁←&R{A₁,B₁}` | commute |
| DUP-NOD | `!X &L=T{a,b,..} → !A&L=a; !B&L=b; ..; X₀←T{A₀,B₀,..}; X₁←T{A₁,B₁,..}` | clone node |
| DUP-NAM | `!X &L=name → X₀←name; X₁←name` | clone name |
| DUP-DRY | `!X &L=^(f x) → !F&L=f; !A&L=x; X₀←^(F₀ A₀); X₁←^(F₁ A₁)` | clone stuck |

#### ERA (Eraser)
| Active Pair | Rule |
|---|---|
| ERA any node | erase recursively — each child gets ERA |

#### OP2 (Binary Numeric — HVM4 native)
| Active Pair | Rule | Effect |
|---|---|---|
| OP2-NUM-NUM | `(#a op #b) → #(a op b)` | compute |
| OP2-ERA | `(* op y) → *` | erase |
| OP2-NUM-ERA | `(#n op *) → *` | erase |
| OP2-SUP | `(&L{a,b} op y) → !Y&L=y; &L{(a op Y₀),(b op Y₁)}` | distribute |
| OP2-NUM-SUP | `(#n op &L{a,b}) → !X&L=#n; &L{(X₀ op a),(X₁ op b)}` | distribute |
| OP2-INC | `(↑x op y) → ↑(x op y)` | delay |

#### AND, OR (Boolean / Short-circuit)
| Active Pair | Rule |
|---|---|
| AND-NUM | `(#0 .&. b) → #0`, `(#n .&. b) → b` when n≠0 |
| OR-NUM | `(#0 .\|. b) → b`, `(#n .\|. b) → #1` when n≠0 |
| AND/OR-ERA | → ERA |
| AND/OR-SUP | distribute through SUP |

#### EQL (Structural Equality)
| Active Pair | Rule |
|---|---|
| EQL-NUM | `(#a === #b) → #(a==b)` |
| EQL-CTR | same tag → recursive; diff tag → #0 |
| EQL-LAM | unify vars, compare bodies |
| EQL-ERA | → ERA |
| EQL-SUP | distribute |
| EQL-ANY | `(* === b) → 1` (wildcard) |

#### USE (Unit Eliminator)
| Active Pair | Rule |
|---|---|
| USE-VAL | `(λ{f} x) → (f x)` |
| USE-ERA | `(λ{f} *) → *` |
| USE-SUP | distribute |

#### ALO (Allocation / Substitution Environment)
| Active Pair | Rule |
|---|---|
| ALO-VAR | `@{s} n → s[n]` |
| ALO-LAM | `@{s} λx.f → λx'.@{x',s}f` |
| ALO-NOD | `@{s} T{a,b,...} → T{@{s}a, @{s}b, ...}` |
| ALO-DUP | `@{s} !x &L=v;t → !x'&L=@{s}v; @{x',s}t` |

### 2.2 Tensor Interactions

Tensor ops follow the **same pattern as HVM4's OP2 on NUM**, but operating on
tensors instead of scalars. The memory layout is identical: an OP node stores
`{op_code, arg0, arg1}` in consecutive heap slots.

#### OP1-TEN (Unary on realized tensor)
```
OP1(RECIP, TEN_a) → dispatch backend.op_unary; produce TEN_result
OP1(CAST, TEN_a)  → dispatch dtype conversion kernel; produce TEN_result
OP1(COPY, TEN_a)  → DMA to target device; produce TEN_result
```

#### OP2-TEN (Binary on realized tensors) — same shape as OP2-NUM-NUM
```
OP2(ADD, TEN_a, TEN_b) → dispatch backend.op_binary; produce TEN_result
OP2(MUL, TEN_a, TEN_b) → dispatch backend.op_binary; produce TEN_result
... (all binary ops)
```

#### OP3-TEN (Ternary)
```
OP3(WHERE, TEN_p, TEN_a, TEN_b) → dispatch ternary kernel; produce TEN_result
```

#### REDUCE-TEN
```
REDUCE(TEN_a, Add, axes) → dispatch backend.op_reduce; produce TEN_result
REDUCE(TEN_a, Max, axes) → dispatch backend.op_reduce; produce TEN_result
```

#### Movement-TEN (View transforms — no compute)
```
RESHAPE(TEN_a, shape')  → new TEN sharing buffer, modified View
EXPAND(TEN_a, shape')   → set strides to 0 on broadcast dims
PERMUTE(TEN_a, π)       → permute strides
PAD(TEN_a, b, e)        → adjust offset and shape
SHRINK(TEN_a, b, e)     → adjust offset and shape
```

#### Tensor ops with ERA — same pattern as OP2-ERA
```
OP2(ADD, *, x) → x       -- additive identity
OP2(ADD, x, *) → x       -- additive identity
OP2(MUL, *, x) → *       -- multiplicative annihilation
OP2(MUL, x, *) → *       -- multiplicative annihilation
OP1(any, *)    → *        -- propagate erasure
REDUCE(*, ..)  → *        -- nothing to reduce
RESHAPE(*, ..) → *        -- nothing to reshape
```

#### Tensor ops with SUP — same pattern as OP2-SUP
```
OP2(op, &L{a,b}, y)  →  !Y &L = y; &L{OP2(op,a,Y₀), OP2(op,b,Y₁)}
OP2(op, TEN_x, &L{a,b}) → !X &L = TEN_x; &L{OP2(op,X₀,a), OP2(op,X₁,b)}
REDUCE(&L{a,b}, op, ax) → &L{REDUCE(a,op,ax), REDUCE(b,op,ax)}
```

#### DUP-TEN — clone a tensor node
```
! X &L = OP2(op, a, b)
─────────────────────── DUP-OP2 (same as DUP-NOD)
! A &L = a
! B &L = b
X₀ ← OP2(op, A₀, B₀)
X₁ ← OP2(op, A₁, B₁)
```

---

## 3. Autograd (GRAD node — Option A)

`GRAD(y, gy, x)` is a lazy term. Reducing it:
1. Reduce `y` to `TEN(y_id)`
2. Read `y`'s provenance: `creator_op`, `src_ids`
3. Apply chain rule for `creator_op` (emit new lazy GRAD terms)
4. Recurse until base case: `y_id == x_id → return gy`

| Forward Op | ∂/∂a | ∂/∂b |
|---|---|---|
| `ADD(a,b)` | `gy` | `gy` |
| `MUL(a,b)` | `gy * b` | `gy * a` |
| `RECIP(a)` | `-gy * RECIP(a)²` | — |
| `REDUCE(a, Add, axes)` | `EXPAND(gy, a.shape)` | — |
| `REDUCE(a, Max, axes)` | `WHERE(a == y, gy, 0)` | — |
| `CAST(a, dtype)` | `CAST(gy, a.dtype)` | — |
| `RESHAPE(a, s')` | `RESHAPE(gy, a.shape)` | — |
| `EXPAND(a, s')` | `REDUCE(gy, Add, expanded_axes)` | — |
| `PERMUTE(a, π)` | `PERMUTE(gy, π⁻¹)` | — |

Higher-order gradients: `GRAD(GRAD(y, gy, x), gz, w)` — just GRADs of GRADs.
The GRAD handler emits new lazy tensor ops, which are themselves differentiable.

`DETACH(T)` stops gradient flow: `GRAD(DETACH(y), gy, x) → ERA`.

---

## 4. Dtypes

All tensors carry a dtype. Scalars are 0-dimensional tensors with the same dtype system.

| dtype | bits | description |
|---|---|---|
| `f64` | 64 | IEEE 754 double |
| `f32` | 32 | IEEE 754 single |
| `f16` | 16 | IEEE 754 half |
| `bf16` | 16 | bfloat16 |
| `f8_e4m3` | 8 | FP8 (E4M3) |
| `f8_e5m2` | 8 | FP8 (E5M2) |
| `c64` | 64 | complex (2×f32) |
| `c128` | 128 | complex (2×f64) |
| `i64` | 64 | signed integer |
| `i32` | 32 | signed integer |
| `i16` | 16 | signed integer |
| `i8` | 8 | signed integer |
| `u64` | 64 | unsigned integer |
| `u32` | 32 | unsigned integer |
| `u16` | 16 | unsigned integer |
| `u8` | 8 | unsigned integer |
| `bool` | 1 | boolean |
| `q4_0` | 4 | 4-bit quantized (GGML block: 32 values + 1 f16 scale) |
| `q4_1` | 4 | 4-bit quantized with min (GGML) |
| `q8_0` | 8 | 8-bit quantized (GGML block: 32 values + 1 f16 scale) |
| `q5_0` | 5 | 5-bit quantized (GGML) |
| `q5_1` | 5 | 5-bit quantized with min |

`CAST` converts between dtypes. `BITCAST` reinterprets bits (same size required).
Quantized types use block layout for dequantization in compute kernels.

---

## 5. View (Shape + Strides)

Every tensor has a `View`:

```
View = {
  shape:   [d₀, d₁, ..., dₙ]   -- dimensions
  strides: [s₀, s₁, ..., sₙ]   -- elements between consecutive indices
  offset:  integer               -- into backing buffer
  contiguous: bool               -- strides == row-major strides
}
```

`numel` is derived: `product(shape)`. Not stored.

Movement ops produce a new View sharing the same buffer.
This is the tinygrad lazy view algebra — zero-copy metadata transforms.

---

## 6. Backend Interface

```
Backend = {
  init      : () → status
  shutdown  : () → ()
  buf_alloc : bytes → buf_id
  buf_free  : buf_id → ()
  buf_write : buf_id × data × bytes → ()
  buf_read  : buf_id × bytes → data

  -- Compute dispatch (strided, view-aware)
  dispatch  : kernel × buf_ids × views × params → ()

  -- Device identity
  device_id : () → device
}
```

Backends: `cpu` (always), `metal` (Apple Silicon), `cuda` (future), `webgpu` (future).

Each backend has a kernel compiler that lowers `RANGE/END/LOAD/STORE` sequences
(the scheduled form) into device-specific code (MSL, PTX, etc).

---

## 7. Scheduling as IC

The scheduler transforms high-level tensor ops into `RANGE/END/LOAD/STORE` loops.
In TinyHVM, the scheduler is itself an IC program: a set of `@`-functions that
pattern-match on tensor op nodes and produce scheduled loop nests.

```
-- Pseudocode: scheduling ADD as an IC function
@sched_add : λop. λout_buf. λin_bufs.
  let n = shape_numel(op) in
  let r = RANGE(n, GLOBAL) in
  let a = LOAD(in_bufs[0], r) in
  let b = LOAD(in_bufs[1], r) in
  let v = OP2(ADD, a, b) in
  END(STORE(out_buf, v), r)
```

This means:
- Scheduling rules are data, not hardcoded C
- Fusion = rewrite rules that merge adjacent schedules
- Optimization (tiling, vectorization) = further rewrite passes
- All expressible as IC term transformations

---

## 8. Kernel Optimizations (OptOps)

Tinygrad's kernel optimization search space, expressed as IC transformations:

| OptOp | Effect |
|---|---|
| `SPLIT(axis, k, target_type)` | Split axis by factor k into (outer, inner) |
| `PADTO(axis, multiple)` | Pad axis to next multiple |
| `SWAP(axis_i, axis_j)` | Swap two iteration axes |
| `NOLOCALS` | Disable shared memory |
| `TC(reduce_idx, config)` | Apply tensor core WMMA |

Axis types after scheduling:

| AxisType | Semantics |
|---|---|
| GLOBAL | GPU workgroup dimension |
| LOCAL | Workgroup local (shared memory) |
| WARP | Warp-level lanes (tensor cores) |
| THREAD | CPU thread parallelism |
| LOOP | Sequential loop |
| REDUCE | Reduction axis |
| GROUP_REDUCE | Shared-memory group reduce |
| UPCAST | Register vectorization |
| UNROLL | Fully unrolled loop |

---

## 9. Multi-Device

IC is inherently parallel: independent interactions can fire concurrently.
Multi-device is natural — DUP a tensor to replicate it across devices.

```
Device = single_device | (device₀, device₁, ..., deviceₙ)
```

A tensor on an n-tuple device is **sharded**: each device holds `1/n` of the data.
`axis` tracks which dimension is sharded. Collectives decompose into DUP + REDUCE:

| Collective | Decomposition |
|---|---|
| broadcast | DUP tensor to all devices |
| scatter | shard tensor across devices (RESHAPE → per-device SHRINK) |
| gather | collect shards (CAT from all devices) |
| allreduce | gather + reduce + broadcast |

Cross-device data movement happens when an interaction requires operands
that live on different devices — the runtime inserts a DMA transfer.

---

## 10. Program Lifecycle (Pure Inet)

The entire training program is one IC term:

```
@main =
  let data   = load_dataset("mnist") in
  let model  = init_weights(seed) in
  let trained = train_loop(model, data, epochs=10) in
  eval(trained, data.test)
```

Where `train_loop` is a recursive IC function:

```
@train_loop : λmodel. λdata. λepochs.
  IF(epochs == 0,
    model,
    let batch   = next_batch(data) in
    let logits  = forward(model, batch.x) in
    let loss    = cross_entropy(logits, batch.y) in
    let grads   = GRAD(loss, ones_like(loss), model.params) in
    let model'  = adam_step(model, grads) in
    train_loop(model', data, epochs - 1)
  )
```

Reducing `@main` to normal form:
1. Unfolds `@main` → `@train_loop` → recursive reduction
2. Each iteration builds a lazy computation graph
3. `loss` term requires `logits` → forces `forward` → tensor ops fire
4. `grads` = GRAD nodes → forces backward pass via chain rule
5. `adam_step` updates weight buffers
6. Loop continues until `epochs == 0`
7. Final result: trained model tensors

No C training loop. No `thvm_reset`. Just reduce one term.

---

## 11. Common Ops as Compositions

```python
# gemm: C[M,N] = A[M,K] @ B[K,N]
def gemm(A, B):
  M,K = A.shape; _,N = B.shape
  return REDUCE(OP2(MUL, RESHAPE(A,[M,K,1]), RESHAPE(B,[1,K,N])), Add, [1])

# conv2d via im2col
def conv2d(x, w, stride, padding):
  patches = unfold(PAD(x, padding), w.shape, stride)  # im2col
  return REDUCE(OP2(MUL, patches, w), Add, reduce_axes)

# softmax
def softmax(x, axis=-1):
  m = REDUCE(x, Max, [axis])
  e = EXP2(OP2(MUL, OP2(ADD, x, NEG(EXPAND(m))), LOG2E))
  return OP2(MUL, e, OP1(RECIP, EXPAND(REDUCE(e, Add, [axis]))))

# cross_entropy
def cross_entropy(logits, labels):
  return NEG(REDUCE(OP2(MUL, one_hot(labels), LOG2(softmax(logits))), Add, [-1]))

# layer_norm
def layer_norm(x, gamma, beta, eps=1e-5):
  mean = OP2(MUL, REDUCE(x, Add, [-1]), OP1(RECIP, CONST(x.shape[-1])))
  var  = OP2(MUL, REDUCE(OP2(MUL, OP2(ADD, x, NEG(mean)),
                                   OP2(ADD, x, NEG(mean))), Add, [-1]),
             OP1(RECIP, CONST(x.shape[-1])))
  return OP2(ADD, OP2(MUL, OP2(MUL, OP2(ADD, x, NEG(mean)),
             OP1(RECIP, SQRT(OP2(ADD, var, CONST(eps))))), gamma), beta)
```

---

## 12. Source Architecture

```
src/
├── tinyhvm.h          -- types, UOp codes, API
├── tinyhvm.c          -- hub (includes all sub-files)
├── term/              -- term constructors, tag helpers
├── heap/              -- bump allocator, read/write
├── tensor/            -- tensor creation, view algebra
├── interact/          -- interaction rules (the core)
├── reduce/            -- enter/apply reducer (HVM4-style)
├── grad/              -- autograd (GRAD handler)
├── ops/               -- tensor op dispatch
├── ctx/               -- context management
├── nn/                -- high-level layers (softmax, loss, etc.)
├── backend/
│   ├── cpu/           -- CPU backend
│   └── metal/         -- Metal backend
└── shaders/           -- GPU shader source
```

---

## References

1. Lafont (1997). "Interaction combinators." *Information and Computation* 137(1).
2. Ehrhard & Regnier (2003). "The differential lambda-calculus." *TCS* 309(1-3).
3. tinygrad (2024). tinyspec: UOp dialect from tensor programs to command buffers.
4. HVM4 (2025). Interaction Calculus runtime. github.com/HigherOrderCO/HVM.
