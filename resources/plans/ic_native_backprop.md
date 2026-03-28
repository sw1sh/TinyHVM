# IC-Native Backpropagation

## Design

Everything is lazy reduction. `thvm_grad_multi(ctx, loss, params, grad_slots, n)` creates
a single TAG_TOP(UOP_GRAD) term. `thvm_reduce` drives the entire backward pass through
interaction rules — no graph walks, no eager tape, no separate backward implementation.

### GRAD Interaction (src/interact/_.c)

```
GRAD(y, gy, x):
  if y == x:       base case → deposit gy via ASSIGN
  if y == CTR:     multi-target → match params, deposit via ASSIGN
  else:            chain rule → look up y.creator_op, produce new GRAD terms
```

Each backward rule creates lazy ops and new GRAD3 terms:
- `UN_GRAD(da)`: unary → GRAD3(input, da, x) via GRAD_STEP (tail call)
- `BIN_GRAD(da, db)`: binary → ADD(GRAD3(a, da, x), GRAD3(b, db, x))

### Key Properties

1. **No backward_local.** Deleted. ONE gradient implementation via IC interaction rules.

2. **Lazy gy.** The trampoline fires GRAD without reducing arg1 (gy). Chain rule
   formulas wrap gy in new lazy ops, creating fusable chains. Only base case and
   deposit explicitly reduce gy.

3. **Lazy ENSURE.** The GRAD handler does NOT ENSURE inputs at the top. Only ops
   that read data (MM, RMAX, LOG, DIV, MAX) call ENSURE. Others create lazy ops
   that reference deferred tensors — the materialize path handles them.

4. **Dead-branch skip.** BIN_GRAD checks `requires_grad` on both inputs. Single-live
   branch uses UN_GRAD (tail call) instead of ADD + full reduction.

5. **RELU backward uses y.** `CMP(y, 0)` instead of `CMP(at, 0)`. Since `relu(x) > 0
   iff x > 0`, the mask is identical. This allows RELU in fused reduce chains (virtual
   intermediates don't have input data, but y is the real output tensor).

### Integration with Fusion

Deferred elementwise dispatch creates backward tensors with `buf_id=0`. These form
chains that `tensor_materialize` fuses at boundaries:

```
GRAD(MUL): MUL(gy, bt) → deferred
GRAD(ADD): ADD(da, db)  → deferred
ENSURE at SUM → materialize_walk → fused 2-op kernel for MUL+ADD
```

Shared intermediates (defer_consumers > 0) get multi-output side buffers.

### SUM Provenance

Fused SUM(elementwise) records proper provenance for backward:
- Virtual intermediates (lower IDs) with pre-reduce shapes, buf_id from dst
- dst has creator_op=UOP_SUM, src_ids[0]=virtual ew_last
- Backward: SUM expand → elementwise chain → leaves

### Gradient Correctness

Verified at float32 precision:
- MLP + CE: 3.5e-8 max error (numpy parity)
- 2-conv + CE: 7.2e-7 max error (numpy parity)
- conv+relu+pool+CE: <0.05 relative error (finite differences)
- All fused reduce axes (non-trailing, multi-axis): exact (0 error)
