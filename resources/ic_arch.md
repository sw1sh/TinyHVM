# TinyHVM Architecture

TinyHVM is a lazy tensor computation engine built on an interaction-calculus (IC)
reduction graph. Computation is expressed as a lazy heap of `Term` nodes; reduction
collapses the graph into realized tensors pushed to GPU/CPU buffers.

---

## 1. Term Layout (64 bits)

```
63      62..56    55..38     37..0
[SUB]   [TAG:7]   [EXT:18]   [VAL:38]
```

| Field | Bits | Meaning |
|-------|------|---------|
| SUB   | 1    | Substitution flag (IC dup/sup)  |
| TAG   | 7    | Term type (see below) |
| EXT   | 18   | Extension / label |
| VAL   | 38   | Payload (pointer or inline value) |

### Tags in use

| Tag | Name | VAL meaning |
|-----|------|-------------|
| `TAG_ERA` (6) | Eraser | — |
| `TAG_NUM` (7) | Inline number | u32 or f32 (EXT[0] = dtype) |
| `TAG_TEN` (10) | Tensor handle | `tensor_id` (index into `ctx->tensors[]`) |
| `TAG_TOP` (11) | Lazy tensor op | `heap_loc` of first arg slot; EXT = UOP code |
| `TAG_VAR/SUP/DP0/DP1` | IC core | used for dup/sup combinator rewrites |

A `TAG_TOP(UOP_SUM, loc)` node is lazy: its operands live at `heap[loc]` and
`heap[loc+1]`. Reducing it dispatches the real computation.

---

## 2. Heap

```c
Term *heap;   // flat array of Term, HEAP_CAP = 256M
u64   heap_pos; // bump allocator top
```

All lazy expressions live here. Movement ops (`RESHAPE`, `EXPAND`, …) allocate a
2-slot node `{child_term, extra_arg}`. Binary ops allocate `{a, b}`.

---

## 3. Reducer — `thvm_reduce`

Single entry point for all computation:

```
thvm_reduce(t):
  if TAG_TEN → return t (already realized)
  if TAG_ERA/NUM → return t
  if TAG_TOP:
    loc = term_val(t)
    if reduce_memo[loc] → return memo  // fast path
    ... dispatch on uop ...
    MEMO_RETURN(result)               // set reduce_memo[loc] = result
```

`reduce_memo` is a flat array indexed by heap loc. It makes the reducer
idempotent: re-reducing the same lazy term always returns the same realized
tensor.

### Reduction order

For a `TAG_TOP(uop, loc)`:
1. Movement ops (`RESHAPE`, `EXPAND`, `PERMUTE`, `SHRINK`, `PAD`): modify `View`
   metadata only, share the source buffer. No compute.
2. `UOP_GRAD`: autograd interaction (see §5).
3. Pattern match for `UOP_FUSING` (see §6).
4. Elementwise binary / unary: reduce both args → call backend `op_binary` /
   `op_unary`.
5. `UOP_SUM` / `UOP_RMAX`: reduce args → call backend `op_reduce` (strided).
6. `UOP_MM`: call backend `op_mm` (BLAS/MPS).

---

## 4. Tensor Metadata — `TensorMeta`

```c
typedef struct {
    u32   buf_id;        // GPU/CPU buffer handle
    u32   dtype;         // DTYPE_F32, DTYPE_F16, …
    u32   refcount;
    View  view;          // shape + strides + offset + numel
    void *host_ptr;      // cached host-side copy

    // Autograd provenance
    u8    requires_grad;
    u32   creator_op;    // UOP that produced this tensor
    u32   src_ids[2];    // input tensor ids (for REACHES traversal)

    // Set when creator_op == UOP_FUSING
    u64   fusing_loc;    // heap loc of the original TAG_TOP subnet root
    u32   fusing_uop;    // UOP of that root (e.g. UOP_SUM)
} TensorMeta;
```

`View` carries shape, strides, offset. Movement ops return a new `TensorMeta`
sharing the same `buf_id` with a modified `View` — no data copy.

---

## 5. Autograd

### Recording

`ctx->recording = 1` during the forward pass. When a realized tensor is created
while recording, its `creator_op` and `src_ids` are set for the GRAD handler.

### GRAD node

```
UOP_GRAD(y, gy, x)  →  ∂y/∂x, weighted by incoming gradient gy
```

`thvm_grad(ctx, y, x)` allocates `GRAD(y, ones_like(y), x)` on the heap.
Reducing it dispatches:

```c
case UOP_GRAD:
  reduce y → TAG_TEN(y_id)
  dispatch on ctx->tensors[y_id].creator_op:
    UOP_ADD  → da + db
    UOP_MUL  → gy * b, gy * a
    UOP_SUM  → expand(gy, input.shape)
    UOP_FUSING → (see §6)
    UOP_MM   → ...
    ...
```

### REACHES

`REACHES(y_id, x_id)` is a stack-based DFS over the provenance graph (`creator_op`,
`src_ids`). Returns 1 if `x` appears anywhere in `y`'s computation graph. Used to
prune branches where `∂y/∂x = 0`.

### `thvm_backward`

Calls `thvm_grad(ctx, loss, param)` for each parameter, then reduces all gradient
terms.

---

## 6. UOP_FUSING — Fused Kernels

`UOP_FUSING` is a realized tensor that was produced by a fused kernel but retains
a reference to the original unfused subnet so backward can walk through it.

### Forward (pattern match in reducer)

When the reducer sees `SUM(MUL(a, b))` (and `!ctx->no_fuse`):

1. Reduce `a`, `b` to realized tensors.
2. Compute broadcast shape, determine reduce axes.
3. Dispatch one Metal/CPU `mul_reduce_sum` kernel (no MUL intermediate buffer).
4. If `ctx->recording`, record provenance:

```c
md->creator_op  = UOP_FUSING;
md->src_ids[0]  = ma_id;     // a — for REACHES traversal
md->src_ids[1]  = mb_id;     // b — for REACHES traversal
md->fusing_loc  = loc;       // heap loc of original SUM TAG_TOP
md->fusing_uop  = UOP_SUM;   // outermost fused op
```

### Backward (GRAD dispatch)

```c
case UOP_FUSING:
  saved = reduce_memo[fusing_loc];
  reduce_memo[fusing_loc] = 0;   // clear forward memo
  no_fuse = 1;                   // prevent re-fusing
  recording = 1;                 // record provenance on unfused path

  unfused = thvm_reduce(TAG_TOP(fusing_uop, fusing_loc));
  // → runs SUM(MUL) the slow way: creates MUL tensor (src_ids={a,b}),
  //   creates SUM tensor (src_ids={mul_id})

  reduce_memo[fusing_loc] = saved;  // restore fast path
  no_fuse = recording = saved_values;

  GRAD(unfused, gy, x)              // standard backward through unfused graph
```

**Key invariant**: forward always returns the fast fused result. Backward runs the
unfused subnet ONCE to get provenance, then delegates to the standard GRAD handler.
No pattern-specific gradient logic in `case UOP_FUSING`.

### Adding a new fusion rule

1. Detect pattern in `thvm_reduce` (before the generic elementwise path).
2. Dispatch one fused kernel.
3. Store `fusing_loc` (heap loc of the subnet root) and `fusing_uop` (root UOP).
4. Store leaf tensor ids in `src_ids` so `REACHES` can traverse.
5. Backward comes for free from the generic `case UOP_FUSING` handler.

---

## 7. Backend Interface

```c
typedef struct {
    int  (*init)(void);
    u32  (*buf_alloc)(u64 bytes);
    void (*buf_free)(u32 id);
    void (*buf_write)(u32 id, const void *data, u64 bytes);
    void (*buf_read)(u32 id, void *out, u64 bytes);
    void (*op_unary)(u32 uop, u32 dst, const View *dv, u32 src, const View *sv);
    void (*op_binary)(u32 uop, u32 dst, const View *dv,
                      u32 a, const View *av, u32 b, const View *bv);
    void (*op_mm)(u32 dst, u32 a, const View *av, u32 b, const View *bv,
                  u32 M, u32 K, u32 N);
    void (*op_reduce)(u32 uop, u32 dst, u32 dst_numel,
                      u32 src, u32 src_numel, u32 reduce_dim);
    ...
} Backend;
```

Two backends: `cpu_backend` (always available) and `metal_backend` (Apple Silicon).
Metal dispatches compute to the GPU via Metal shaders (`shaders.metal`).

All ops receive full `View` structs so strides, offsets, and broadcast dimensions
are handled correctly.

---

## 8. Source Map

| File | What it does |
|------|-------------|
| `src/tinyhvm.h` | All types, UOp codes, API declarations |
| `src/tinyhvm.c` | Reducer, autograd, movement ops, high-level API |
| `src/cpu.c` | CPU backend: elementwise, reduce, matmul loops |
| `src/metal.m` | Metal backend: GPU kernels, `metal_mul_reduce_sum` |
| `src/shaders.metal` | MSL compute shaders |
| `src/layers.c` | Conv2d, maxpool, sequential model helpers |
| `resources/ic_autograd.md` | Autograd design notes |
| `resources/ic_fusion.md` | Fusion design and FUSING roadmap |

---

## 9. Lifecycle

```
thvm_init(backend)           → allocate heap, tensor table, memo
thvm_start_recording(ctx)    → ctx->recording = 1
[ build lazy graph with thvm_op, thvm_reshape, ... ]
thvm_stop_recording(ctx)     → ctx->recording = 0
thvm_reduce / thvm_realize   → collapses lazy graph, fills buffers
thvm_backward(ctx, loss, params, grads, n)  → reduce grad terms
[ SGD / Adam steps ]
thvm_reset(ctx, n_weights)   → free all tensors above n_weights, clear heap/memo
```

`thvm_reset` is the training-loop boundary — it frees activation buffers while
keeping weight tensors intact.
