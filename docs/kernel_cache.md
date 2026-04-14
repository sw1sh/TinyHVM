# Kernel Result Caching

## Structural vs Runtime Identity

The public IR now exposes structural `UOP_KERNEL` nodes on the heap.
Dispatch/cache state is private runtime metadata built lazily from those nodes.

The runtime still uses a compact numeric kernel index (`kid`) internally, but it
is discovered from the structural kernel location:

```text
heap KERNEL loc -> runtime kid -> KernelEntry + cached TAG_TEN result
```

This keeps the visible IR faithful while still reusing the existing runtime
tables.

## Current Runtime State

| Table | Meaning |
|------|---------|
| `sched_kernel_locs[kid]` | heap location of the structural `KERNEL` |
| `sched_kernels[kid]` | lowered `KernelEntry` |
| `kid_results[kid]` | cached dispatched result |
| `kid_input_bufs[kid][]` | input buffers read by that kernel |
| `kid_input_epochs[kid][]` | epochs observed at last dispatch |

The `UOP_KERNEL` handler looks up the runtime entry from the structural heap
location, building one on demand if needed.

## Why Epoch Checks Still Matter

In a training loop:

```text
SEQ(ASSIGN(w, ...), KERNEL(...reads w...))
```

the structural `KERNEL` node is stable across iterations, but the buffers it
reads can change after `ASSIGN`.

So cache validity is not:

```text
same heap kernel loc => safe reuse
```

It is:

```text
same heap kernel loc AND same input buffer epochs => safe reuse
```

## Current Policy

Per-buffer epoch tracking is still the right policy:

1. `ASSIGN` increments `buf_epoch[dst_buf_id]`
2. `UOP_KERNEL` records input buffer IDs and epochs at dispatch time
3. cache hit only if all current epochs still match
4. otherwise the cached result is invalidated and the kernel is re-dispatched

This gives:

- O(1) invalidation work in `ASSIGN`
- O(n_inputs) cache validation in `KERNEL`
- precise reuse across loop iterations

## Relation to Structural KERNEL

Because the cache is keyed from the structural kernel location:

- step graphs can show the real `KERNEL` node
- `DUP` can clone a kernel structurally and get a fresh runtime instance
- runtime metadata can change without altering the visible IR contract

## Files

- `src/interact/tensor_ops.c` — `UOP_KERNEL` cache check and registration
- `src/schedule/_.c` — runtime tables and backend dispatch
- `src/tinyhvm.h` — runtime table sizes and `KernelEntry`
