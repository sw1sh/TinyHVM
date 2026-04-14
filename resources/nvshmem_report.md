# NVSHMEM: Relevance to IC-Based Multi-GPU NN Frameworks

## What It Is

NVSHMEM is NVIDIA's GPU-native implementation of the OpenSHMEM PGAS (Partitioned Global Address Space) standard. It creates a shared address space across multiple GPUs where **any GPU thread can directly read/write memory on any other GPU** via one-sided put/get operations — no CPU mediation required.

Source: https://github.com/NVIDIA/nvshmem

## Why This Matters for TinyHVM

### The tinygrad constraint

tinygrad's architecture is a **linear schedule**: lazy graph → linearize → schedule → dispatch. Each kernel is a self-contained unit fused from a contiguous subgraph. Multi-GPU in tinygrad means NCCL collectives inserted between scheduled kernels — the CPU orchestrates all inter-device transfers as explicit copy operations between device buffers.

This means tinygrad **cannot**:
- Have one kernel's thread read from another GPU's buffer mid-execution
- Dynamically route data between devices based on runtime values
- Overlap compute and inter-device communication within a single kernel
- Express irregular/sparse communication patterns (e.g., MoE routing) without materializing full buffers

### The IC opportunity

TinyHVM's interaction combinator net is fundamentally **non-linear** — terms connect to terms via ports, and reduction proceeds by local interaction rules, not a global linear schedule. This maps naturally onto NVSHMEM's model:

| NVSHMEM concept | IC analogue |
|---|---|
| Symmetric heap (global address space) | IC heap spanning multiple GPUs |
| PE (processing element per GPU) | Reducer thread group per device |
| `nvshmem_ptr()` — direct remote pointer | Port wire crossing device boundary |
| `nvshmem_put_signal` — write + notify | Interaction rule producing result on remote device + signaling readiness |
| `nvshmem_atomic_compare_swap` | Atomic redex claiming (parallel reduction) |
| Weak ordering + explicit fence | Interaction net's natural causal ordering |

**Key insight**: In an IC net, when a wire crosses a device boundary, the reducer on GPU-A that needs to interact with a node on GPU-B can simply `nvshmem_g()` (get) the remote node's data, perform the interaction locally, and `nvshmem_p()` (put) the result back — all from within the reduction kernel. No CPU round-trip. No collective shape constraints.

### Concrete advantages over NCCL for IC reduction

1. **Fine-grained, irregular access**: IC reduction touches nodes in graph-determined order, not bulk-transfer order. NVSHMEM's per-thread put/get matches this exactly. NCCL would require materializing and transferring entire buffers even when only scattered nodes are needed.

2. **GPU-initiated communication**: The reduction kernel itself decides what to fetch and where to write. No need to return to CPU between interaction steps that cross device boundaries.

3. **Natural compute-comm overlap**: GPU warp scheduling hides remote access latency — while warps waiting on remote data are parked, other warps continue reducing local interactions. This is free overlap without stream-level pipelining.

4. **Atomic redex claiming**: `nvshmem_atomic_compare_swap` enables multiple GPUs to race for redexes on a shared heap without locks — exactly the pattern HVM4 uses for parallel IC reduction, extended across devices.

5. **Constant latency at scale**: NVSHMEM latency stays ~16μs regardless of GPU count (within NVLink: sub-microsecond via direct load/store). NCCL latency grows linearly.

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  Application                     │
├──────────────────┬──────────────────────────────┤
│   Host API       │   Device API                  │
│   (libnvshmem_   │   (libnvshmem_device.a)       │
│    host.so)      │   - linked into CUDA kernels  │
│   - init/malloc  │   - nvshmem_p/g/put/get       │
│   - collectives  │   - atomics, signals, waits   │
│   - teams        │   - fence/quiet               │
├──────────────────┴──────────────────────────────┤
│              Transport Layer                      │
│   IBRC (InfiniBand RC) | IBGDA | UCX | P2P      │
│   NVLink direct | GDRCopy | Proxy fallback       │
└─────────────────────────────────────────────────┘
```

**Memory model**: `nvshmem_malloc(size)` is collective — all PEs allocate the same size. Returns a "symmetric address" valid on all GPUs. Within NVLink, `nvshmem_ptr(addr, peer)` returns a raw device pointer that's directly dereferenceable — zero-copy, zero-overhead.

**Ordering**: Weak by default (writes may reorder). `nvshmem_fence()` orders prior writes before subsequent ones. `nvshmem_quiet()` waits for all pending ops to complete.

## Key APIs

### Remote Memory Access
```c
// Single-element (from GPU kernel thread)
nvshmem_float_p(dest, value, peer_pe);      // put one float
float v = nvshmem_float_g(source, peer_pe); // get one float

// Bulk
nvshmem_float_put(dest, src, nelems, peer);    // blocking bulk put
nvshmem_float_get(dest, src, nelems, peer);    // blocking bulk get
nvshmem_float_put_nbi(dest, src, n, peer);     // non-blocking

// Direct pointer (NVLink only — zero overhead)
float *remote = (float*)nvshmem_ptr(sym_addr, peer);
remote[i] = local_val;  // direct store to remote GPU memory
```

### Signaling (producer-consumer)
```c
// Writer: put data + signal atomically
nvshmem_float_put_signal(dest, src, nelems, sig_addr, sig_val, NVSHMEM_SIGNAL_SET, peer);

// Reader: wait for signal, then consume data
nvshmem_signal_wait_until(sig_addr, NVSHMEM_CMP_EQ, expected);
```

### Atomics
```c
// Compare-and-swap (for redex claiming across devices)
old = nvshmem_int_atomic_compare_swap(dest, expected, desired, peer);

// Fetch-add (for distributed counters)
old = nvshmem_int_atomic_fetch_add(dest, increment, peer);
```

### Collectives (when you do need bulk ops)
```c
nvshmem_float_sum_reduce(team, dest, src, nelems);  // allreduce
nvshmem_float_broadcast(team, dest, src, nelems, root);
nvshmemx_float_alltoall_on_stream(team, dest, src, nelems, stream);
```

## NVSHMEM vs NCCL Comparison

| Dimension | NCCL | NVSHMEM |
|---|---|---|
| Communication model | Two-sided bulk collectives | One-sided put/get, any pattern |
| Initiation | CPU launches on CUDA stream | GPU threads inside kernels |
| Granularity | Entire buffer | Single element to bulk |
| Access pattern | Fixed (allreduce, allgather, etc.) | Arbitrary (any thread → any address → any GPU) |
| Routing | Multi-hop ring/tree algorithms | Direct P2P or NIC; no multi-hop |
| Latency scaling | Linear with GPU count | Near-constant (~16μs IBRC) |
| Compute-comm overlap | Stream-level pipelining from CPU | Warp-level scheduling in kernel |
| Complexity | Simple (few API calls) | Manual sync, PE management |
| Maturity for ML | Dominant (PyTorch, tinygrad, etc.) | Niche (LBANN, GNN, custom) |

## Relevant Prior Art

- **LBANN** (spatial-parallel convolution): NVSHMEM halo exchange achieved ~2x speedup over MPI at 32 GPUs for large-image convolution. Relevant because conv halo exchange is exactly the kind of irregular boundary communication that IC wires across devices would produce.

- **MGG** (multi-GPU GNN): Stored entire graph node embeddings in NVSHMEM symmetric heap. All GPUs access any node with ~equal latency. This is the closest existing analogue to "IC heap spanning GPUs."

- **Jacobi stencil benchmark**: 3.3x faster than MPI on 384 GPUs. NVSHMEM continued scaling past the point where MPI hit a wall at 16 nodes.

## Design Sketch: IC Heap over NVSHMEM

```
GPU 0                          GPU 1
┌──────────────────┐          ┌──────────────────┐
│ IC Heap (local)  │          │ IC Heap (local)  │
│ ┌──────────────┐ │          │ ┌──────────────┐ │
│ │ nodes[0..N]  │◄──NVSHMEM──►│ nodes[0..M]  │ │
│ └──────────────┘ │  put/get  │ └──────────────┘ │
│                  │          │                  │
│ Reducer warps:   │          │ Reducer warps:   │
│ - claim redex    │          │ - claim redex    │
│ - if remote port:│          │ - if remote port:│
│   nvshmem_g()    │          │   nvshmem_g()    │
│ - interact       │          │ - interact       │
│ - nvshmem_p()    │          │ - nvshmem_p()    │
│   result back    │          │   result back    │
└──────────────────┘          └──────────────────┘

Symmetric heap: nvshmem_malloc(MAX_NODES * sizeof(Node))
Redex claiming: nvshmem_atomic_compare_swap on node tag
Wire crossing:  port address encodes (pe_id, local_offset)
```

**Address encoding**: A 64-bit port address could pack `[pe_id:8][local_offset:56]`, letting any reducer decode whether an interaction is local (direct memory) or remote (nvshmem_g/p). The IC's wire abstraction makes this transparent — a wire is a wire regardless of which GPU the target node lives on.

**Migration vs. remote access**: For frequently interacting cross-device node pairs, you could migrate nodes to co-locate them (nvshmem_get + local insert + redirect wires). The IC's garbage collection naturally handles orphaned remote references. This is impossible in tinygrad's model where buffer placement is decided at schedule time and fixed.

## Tradeoffs and Risks

1. **No multi-hop routing**: NVSHMEM requires direct connectivity (NVLink or NIC). For >8 GPU topologies without full mesh, some PE pairs may fall back to CPU proxy (slow). NCCL's ring/tree algorithms handle arbitrary topologies transparently.

2. **Manual synchronization**: The weak ordering model means you must carefully fence/quiet to avoid reading stale data. IC reduction's causal structure helps here (you only read a port after the interaction that wrote it), but getting this right in CUDA is non-trivial.

3. **Symmetric allocation is collective**: All GPUs must allocate the same heap size. Dynamic heap growth requires collective reallocation. This constrains IC heap management.

4. **NVIDIA-only**: NVSHMEM is CUDA-only. No Metal, no AMD. For TinyHVM's current Metal backend, this is irrelevant until CUDA support exists. But it's worth knowing the model for when multi-GPU becomes relevant.

5. **Ecosystem immaturity for ML**: Most ML frameworks use NCCL. NVSHMEM tooling, debugging, and community knowledge for ML workloads is thin.

## Addendum: Normalization, Cache Keys, and SUP Search

The tinygrad blog's "schedule cache normalization" is orthogonal to NVSHMEM,
but it is highly relevant to TinyHVM because it cleanly separates three
questions that we should not blur together:

1. What is the kernel structurally?
2. How is that kernel lowered for a backend?
3. Which tuned variant wins on a specific device?

### What tinygrad is normalizing

Tinygrad first rewrites the graph into a form that forgets concrete per-run
identity and keeps only the structure that matters for later rewrites and
caching. The blog's examples are exactly the right instinct:

- concrete buffer-bearing ops become abstract params
- globally unique IDs become local per-invocation IDs
- cache keys depend on graph structure, not incidental object identity

That improves cache hit rate because two equivalent kernels hash the same way
even if they were built from different tensors, buffer objects, or process
lifetimes.

### TinyHVM analogue

For TinyHVM, the normalized unit should be the explicit structural `KERNEL`
subtree plus the metadata that actually changes generated code:

- fused op structure
- normalized views / shapes / dtypes
- backend and device class
- reduction shape and other tuning-relevant features

It should *not* include heap locations, `kid`, runtime buffer slots, tensor
IDs, or any temporary handles introduced during reduction. Those are runtime
artifacts, not kernel semantics. If we hash on them, we destroy cache reuse.

This is one of the strongest lessons from tinygrad for the current structural
`KERNEL` direction: make the kernel boundary explicit in the IR, then
normalize that explicit structure before lowering or tuning.

### Relation to BEAM search

Tinygrad's BEAM search is not deciding the semantic graph anymore. By the time
BEAM runs, the compiler already knows what kernel it is tuning. BEAM is a
per-kernel autotuner over intra-kernel decisions such as workgroup size,
upcast, unroll, and grouped reduction strategy.

That suggests a clean TinyHVM pipeline:

1. `reduce(FUSE(program))` discovers explicit `KERNEL` nodes.
2. Normalize each `KERNEL` into an address-free canonical signature.
3. Look up compiled code and tuning results by that signature.
4. Only on cache miss, run heuristic search / BEAM / profiling.
5. Cache the winning config under the normalized signature.

This keeps fusion discovery, cache identity, and autotuning as separate
layers. Tinygrad is strong here. We should copy that separation.

### Where SUP search actually fits

SUP-based search is most promising where the search space is discrete,
symbolic, and richly shared:

- fusion cut placement
- axis role assignment
- legal schedule filtering
- template family selection
- symbolic cost-model evaluation

This is where optimal sharing can compress a combinatorial search space that
BEAM would otherwise sample greedily.

Where SUP search is *less* compelling is the final "which Metal/CUDA kernel is
actually fastest on this hardware?" decision. That cost function is not pure.
It depends on bank conflicts, occupancy, cache behavior, register pressure,
and other hardware effects that only appear when you really compile and run
the candidate. Once candidates cross the dispatch boundary, the sharing
advantage mostly disappears and you still owe real measurements.

So the best fit looks hybrid:

- use SUPs to encode structural alternatives compactly
- use a symbolic or learned model to prune obviously bad branches
- hand the surviving small set to ordinary autotuning / narrow BEAM
- cache the winner by normalized `KERNEL` signature

In other words: SUPs are closer to "search over program structure" than to
"replace the last mile of hardware autotuning."

### Multi-GPU note

If TinyHVM grows a multi-device scheduler later, placement and topology should
sit *beside* the normalized kernel key, not inside the kernel identity itself.
The same normalized `KERNEL` may need different winners on:

- single GPU execution
- NCCL-style staged copies
- NVSHMEM-style remote-access execution

If those concerns get mixed into one undifferentiated cache key, cache reuse
will be noisy and tuning results will bleed across incompatible execution
regimes.

## Bottom Line

NVSHMEM's PGAS model is a natural fit for extending IC reduction across multiple GPUs. The key architectural alignment is that **IC wires are already one-sided** — a port points to a node, and reduction proceeds by reading/writing through ports. NVSHMEM simply extends the reach of those ports across device boundaries without changing the reduction model.

tinygrad can't do this because its linear schedule requires all inter-device transfers to be explicit copy kernels inserted by the scheduler. TinyHVM's non-linear IC structure means device boundaries can be transparent at the wire level — which is exactly what PGAS provides.

This is a "be aware of" resource, not an "implement now" item. The prerequisite is multi-GPU CUDA support for TinyHVM, which doesn't exist yet. But when that time comes, NVSHMEM > NCCL for IC-based reduction because the communication pattern is fundamentally irregular and fine-grained, not bulk-collective.
