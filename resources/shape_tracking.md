# Shape Tracking Design

How TinyHVM handles tensor shapes, strides, views, and broadcasting. Inspired by tinygrad's ShapeTracker but implemented in C with the interaction net model.

## Tinygrad's Approach (What We're Learning From)

Tinygrad separates **shape** from **data**. A tensor's shape is tracked by a `ShapeTracker` — a stack of `View` objects. Each View has:

```
View {
    shape:    (3, 4, 5)      # logical dimensions
    strides:  (20, 5, 1)     # step size in each dimension (row-major default)
    offset:   0              # starting position in buffer
    mask:     None | ((lo,hi), ...)  # valid range per dimension (for padding)
    contiguous: bool         # is this just a plain dense tensor?
}
```

The crucial trick: **movement ops don't move data.** Reshape, transpose, broadcast, slice — these just push new Views onto the stack or modify strides. The actual data stays put in its buffer. A kernel only gets generated when you do a compute op.

### Movement ops in tinygrad (6 total):

| Op | What it does to the View |
|---|---|
| `reshape(new_shape)` | Merge or split dims. May need a new View if non-contiguous |
| `permute(axes)` | Reorder strides: `strides = [strides[a] for a in axes]` |
| `expand(new_shape)` | Set stride=0 where size=1→N (broadcast, no copy) |
| `shrink((lo,hi), ...)` | Adjust offset, narrow each dim |
| `pad((lo,hi), ...)` | Add mask for out-of-bounds regions |
| `flip(axes)` | Negate strides, adjust offset |

Broadcasting is really just `expand` — set stride to 0 and repeat the dimension.

---

## TinyHVM's Shape Tracking

We want this same power but in C, fitting within our 64-bit term / heap-based system.

### The View Struct

```c
typedef struct {
    u32 ndim;
    u32 shape[MAX_DIM];    // logical shape
    i32 strides[MAX_DIM];  // can be negative (for flip) or zero (for broadcast)
    i32 offset;            // starting element in buffer
    u32 numel;             // total logical elements
    u8  contiguous;        // 1 if standard row-major layout
} View;
```

Note: strides are `i32` not `u32` because flipping negates them.

### Where Views Live

Currently a `TensorMeta` stores `shape[8]` and `numel`. We replace that with a `View`:

```c
typedef struct {
    u32  buf_id;       // GPU buffer this is a view of
    u32  dtype;
    u32  refcount;
    View view;         // shape + strides + offset
    void *host_ptr;    // host shadow (for readback)
} TensorMeta;
```

### Movement UOps

These are new UOps that create a new `TensorMeta` with modified `View` but **same `buf_id`**:

```c
#define UOP_RESHAPE    16
#define UOP_PERMUTE    17
#define UOP_EXPAND     18  // = broadcast
#define UOP_SHRINK     19  // = slice
#define UOP_PAD        20
#define UOP_FLIP       21
```

Movement ops are free — no GPU dispatch, no data copy. They just allocate a new TensorMeta pointing at the same buffer.

### How Broadcasting Works

When `add([2,3], [1,3])` is called, the reducer needs to match shapes. Steps:

1. Check if shapes match directly → dispatch add
2. If not, try broadcasting: for each dim, if one tensor has size=1 and the other has size=N, expand size=1 to size=N by setting stride=0
3. After expansion, shapes match → dispatch add

The GPU kernel doesn't know about broadcasting. It just indexes `buf[i * stride]`. When stride=0, it reads the same element for every index in that dimension. The kernel loops over the output shape and uses strides to index into each input.

### Stride-Based Kernel Indexing

The generic GPU kernel for a binary op becomes:

```c
// For output position i (flat index):
// Convert i to multi-dim coords using output shape
// Index into each input using that input's strides
for (u32 i = 0; i < out_numel; i++) {
    u32 idx_a = 0, idx_b = 0, rem = i;
    for (int d = ndim-1; d >= 0; d--) {
        u32 coord = rem % out_shape[d];
        rem /= out_shape[d];
        idx_a += coord * stride_a[d];
        idx_b += coord * stride_b[d];
    }
    dst[i] = a[idx_a + off_a] OP b[idx_b + off_b];
}
```

This handles broadcasting (stride=0), transposition (permuted strides), slicing (offset), and views — all in one loop without copying data.

### Transpose

Transpose of a 2D tensor is just permuting strides:

```c
// Before: shape=(M,N), strides=(N,1)
// After:  shape=(N,M), strides=(1,N)
View transpose_2d(View v) {
    swap(v.shape[0], v.shape[1]);
    swap(v.strides[0], v.strides[1]);
    v.contiguous = 0;
    return v;
}
```

No data movement. For matmul backward, `mm(grad, Bᵀ)` uses B's buffer with permuted strides.

### Contiguity

A View is contiguous when:
- offset = 0
- strides = row-major strides (i.e., `strides[i] = product(shape[i+1:])`)
- no mask

Some ops need contiguous inputs (e.g., BLAS matmul wants dense row-major). Before dispatching, the reducer checks contiguity and inserts a `UOP_COPY` (physical copy with stride resolution) if needed.

---

## What This Buys Us

1. `Tensor.reshape(...)` → free (metadata only)
2. `Tensor.transpose()` → free (swap strides)
3. `Tensor.broadcast_to(...)` → free (set stride=0)
4. `Tensor[2:5, :]` → free (adjust offset + shape)
5. Matmul backward (`mm(grad, Bᵀ)`) → no copy of B, just permuted strides
6. Bias add (`[M,N] + [N]`) → expand bias with stride=0 in dim 0

Only compute ops and explicit copies touch GPU buffers. Everything else is just View arithmetic.

---

## Implementation Order

1. Replace `shape[]/numel` in TensorMeta with `View` struct
2. Add `view_create()`, `view_permute()`, `view_expand()` helpers
3. Update GPU backend ops to use strided indexing
4. Add broadcasting logic in the reducer (auto-expand before binary ops)
5. Update matmul dispatch: check contiguity, copy if needed
6. Test: `add([2,3], [1,3])`, `transpose`, bias broadcast in forward pass
