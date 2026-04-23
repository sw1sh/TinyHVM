# Memory Planning

> **Status: proposal.** This document describes a design direction
> informed by the tinygrad memory planner.  TLSF is not yet
> implemented in TinyHVM; the current allocator is the per-step slot
> reuse logic in `src/schedule/_.c` plus the array pool in
> `src/backend/cpu/pool.c`.  Treat the "TLSF in TinyHVM" section
> below as a roadmap, not a spec of existing code.

## Problem

The scheduler currently allocates physical tensor storage (`tensor_create()`) at schedule time
(phase 2). This is wrong — boundary tensors should be **logical planned values** until dispatch.

Reasons:
- Tensors allocated early cannot share buffers (no reuse analysis yet)
- Tinygrad achieves 0.82GB at BS=512 via 58→1 buffer reuse planner
- Physical allocation during scheduling mixes two abstraction layers

## Design

### Separation of Concerns

| Layer | Responsibility | When |
|-------|---------------|------|
| **Fusion** (UOP_FUSE) | Choose compute boundaries, create kernel nodes | Phase 2 |
| **Planning** | Assign logical slot IDs, compute use counts, map buffer reuse | Phase 2/3 boundary |
| **Dispatch** (UOP_KERNEL) | Allocate physical buffers, execute kernels | Phase 3 |

### Planning Phase

After fusion produces the kernel DAG:
1. Walk the kernel DAG (acyclic dependency graph of UOP_KERNEL nodes)
2. Assign logical value IDs to each kernel output
3. Compute use counts (how many consumers per value)
4. Plan buffer reuse: values with non-overlapping lifetimes share physical slots
5. No physical allocation — output is annotated kernel DAG with slot assignments

### Logical vs Physical Values

```
Phase 2 output:  K1[slot=0] → K2[slot=1] → K3[slot=0]  (slot 0 reused)
Phase 3 input:   Allocate physical buffers for unique active slots only
```

### Boundary Tensors

Current: `sched_prepare_boundary_output()` calls `tensor_create()` per kernel output.
Target: boundary outputs are logical slot IDs. Physical buffers allocated lazily at dispatch.

## TLSF Allocator

### What It Is

TLSF (Two-Level Segregated Fit) is an O(1) worst-case dynamic memory allocator. It is
what tinygrad uses for GPU buffer management. The core idea: use two levels of bitmaps
to index free blocks by size class, so finding a good-fit block requires only a couple
of hardware `ffs`/`fls` bit-scan instructions instead of list traversal.

### How It Works

**First level (FL)**: Divides block sizes by most significant bit. FL index = `fls(size)`.
Each FL class covers `[2^k, 2^(k+1))`.

**Second level (SL)**: Each FL class is subdivided linearly into `2^SL_LOG2` sub-classes
(typically 16 or 32). For size `s` in FL class `k`:
```
sl = (s >> (k - SL_LOG2)) XOR (1 << SL_LOG2)
```

Total free lists: `FL_COUNT * SL_COUNT` (e.g., 32 × 16 = 512).

**Bitmaps** track which free lists are non-empty:
- `fl_bitmap`: single 32-bit word, bit `i` set → FL class `i` has free blocks
- `sl_bitmap[FL_COUNT]`: one 32-bit word per FL class, bit `j` set → sub-class `j` non-empty

**Block header** (minimal):
- `size` with two stolen low bits: bit 0 = free/used, bit 1 = prev block free/used
- `prev_phys_block` — pointer to physically previous block
- `next_free`, `prev_free` — doubly-linked free list pointers (only when free)

**Allocation (O(1))**:
1. Map requested size → (FL, SL), rounding up to next SL boundary
2. Check `sl_bitmap[fl]` from bit `sl`. If empty, check `fl_bitmap` from bit `fl+1` via `ffs`
3. Pop head of `blocks[fl][sl]`, update bitmaps if list emptied
4. Split block if remainder is large enough; insert remainder into its free list
5. Return pointer

**Deallocation (O(1))**:
1. Mark block free
2. Coalesce with physically adjacent free blocks (at most one left, one right)
3. Compute (FL, SL) for merged block, insert into free list, set bitmap bits

Every step is bounded: bit shifts + hardware bit-scan + linked-list head ops. No loops
that scale with free block count or total memory.

### How Tinygrad Uses TLSF

Tinygrad runs TLSF **at compile time on CPU**, not at runtime on GPU. The schedule is
fully determined before execution, so the planner simulates all alloc/free events and
precomputes a memory layout. This is described in the tinyblog pipeline as a step
downstream of kernel structure — memory planning happens after schedule creation,
treating buffer slots as an execution detail, not an IR contract.

Two uses in tinygrad:

**1. Memory planner** (`tinygrad/engine/memory.py`):
- Compute each buffer's lifetime as `[first_appearance, last_appearance]` across the schedule
- Create one `TLSFAllocator` with `total_memory = 2× sum of all buffer sizes` (~15% frag headroom)
- Simulate alloc at first appearance, free at last appearance + 1
- Each alloc returns an **offset** into a single large backing buffer
- Allocate ONE GPU buffer of size = max offset seen; all intermediates are sub-regions
- For devices without offset support: fall back to pool matching by `(device, dtype, options, nbytes)`

**2. GPU virtual memory manager** (`tinygrad/runtime/support/memory.py`):
- AMD/NVIDIA drivers use `TLSFAllocator` for physical and virtual address space management
- `MemoryManager` creates `boot_allocator` and `pa_allocator` (both TLSF) for physical memory
- Class-level `va_allocator` (TLSF) manages virtual address space
- `valloc()` allocates VA via TLSF, then maps physical pages through page tables

Result: 58 unique buffers → 1 shared buffer at BS=512. 0.82GB instead of multi-GB.

### Alternatives and Why TLSF Wins for GPU Buffers

| Allocator | Alloc | Free | Internal Frag | External Frag | Arbitrary Sizes |
|-----------|-------|------|---------------|---------------|-----------------|
| **TLSF** | O(1) | O(1) | Low (~15%) | Low | Yes |
| **Buddy** | O(log N) | O(log N) | High (~25%) | Low | Yes (rounded to 2^k) |
| **Slab** | O(1) | O(1) | Zero | Medium | No (fixed sizes) |
| **Best-fit tree** | O(log N) | O(log N) | Minimal | Medium | Yes |
| **Pool** | O(1) | O(1) | Zero | N/A | No (fixed sizes) |

**Buddy allocator**: Splits memory in half recursively. Every block is a power-of-two
size. A 33KB request wastes 31KB (rounded to 64KB). Average ~25% internal fragmentation.
Used by Linux page allocator, Vulkan Memory Allocator (legacy mode). Too wasteful for
tensor buffers where shapes produce arbitrary byte counts.

**Slab allocator**: Pre-allocates fixed-size object pools. O(1) and zero internal frag
for known sizes, but requires predefined size classes. Cannot handle arbitrary tensor
shapes. Used by Linux kmem_cache, GPU descriptor pools.

**Best-fit with balanced tree**: Maintains a red-black tree of free blocks sorted by size.
Finds smallest block ≥ request. Good fragmentation but O(log N) per operation. Used by
dlmalloc/glibc. Not bounded — unacceptable if the allocator is on the critical path.

**Pool allocator**: tinygrad's fallback for devices without offset support (Qualcomm,
OpenCL). Matches by exact `(device, dtype, options, nbytes)` tuple. O(1) but only reuses
buffers of identical size — no splitting, no coalescing.

**Why TLSF**: GPU tensor buffers are arbitrary-sized (determined by tensor shapes), need
fast allocation (critical path), and fragmentation = wasted VRAM. TLSF is the only
allocator providing O(1) for arbitrary sizes with low fragmentation. The ~15% overhead is
acceptable; buddy's ~25% is not. Best-fit's O(log N) is not bounded.

### TLSF in TinyHVM

TinyHVM's current memory management:
- **Heap**: bump-pointer allocator for IC terms (`src/heap/alloc.c`), per-thread banking
- **Buffers**: simple array pool (`src/backend/cpu/pool.c`), `buf_id → (pointer, size)`
- **Scheduler**: in-step slot reuse (`src/schedule/_.c`), tracks kernel execution order
  and reuses slots where `slot_end[sid] < current_pos` and compatible backend/size
- **GC**: bulk `pool_reset()` between steps; no mid-step buffer freeing due to GRAD
  traversal constraints (tensor metadata still accessed after decref)

The gap: TinyHVM uses 4.3GB at BS=64 vs tinygrad's 0.82GB at BS=512.

**Implementation path for TLSF in TinyHVM**:

The key insight from the tinyblog is that memory planning is downstream of kernel
structure. TLSF doesn't change fusion or scheduling — it replaces the physical allocation
strategy inside dispatch.

**Step 1 — TLSF allocator in C** (~150 lines):
```c
typedef struct {
    uint32_t fl_bitmap;
    uint32_t sl_bitmap[FL_COUNT];
    Block*   blocks[FL_COUNT][SL_COUNT];  // free list heads
    // block headers stored inline in the managed region
} TLSFAlloc;

size_t tlsf_alloc(TLSFAlloc *a, size_t size);   // returns offset
void   tlsf_free(TLSFAlloc *a, size_t offset);
```

This is a straightforward port. The mattconte/tlsf reference implementation (public
domain C) is ~600 lines including all edge cases. TinyHVM only needs the core: alloc,
free, coalesce. No alignment complexity (Metal buffers already page-aligned).

**Step 2 — Compile-time simulation** (in `sched_all`):
After the kernel DAG is built and execution order determined:
1. Walk kernels in execution order
2. For each kernel output: `offset = tlsf_alloc(&sim, nbytes)`
3. For each kernel input whose last consumer is this kernel: `tlsf_free(&sim, offset)`
4. Record `max_offset` seen
5. Allocate one backing buffer of `max_offset` bytes
6. Each kernel's output buffer = `backing + offset`

This runs on CPU before any GPU dispatch. The TLSF instance is ephemeral — created per
step, used for simulation, discarded after offsets are assigned.

**Step 3 — Single backing buffer per step**:
Replace per-kernel `buf_alloc()` in dispatch with sub-region assignment:
```c
// Before (current):
buf_id = pool_alloc(backend, nbytes);

// After (TLSF planned):
buf_id = pool_get_subregion(backing_buf, offset, nbytes);
```

This requires the backend pool to support offset-based sub-allocation. For Metal:
`[buffer newRemoteBufferViewForDevice:... offset:... length:...]` or pointer arithmetic
on shared memory buffers.

**Step 4 — Pinned buffers**:
Weights, optimizer state, and cross-step tensors must be excluded from the TLSF pool.
These are "pinned" — allocated separately, never freed by the per-step planner. The
scheduler already tracks this via `SchedSlot.pinned`.

**What changes, what doesn't**:
- Fusion: unchanged. TLSF is downstream of kernel structure.
- Scheduling: unchanged. Execution order and slot reuse logic stay the same.
- Dispatch: changes from `pool_alloc` to `backing + offset`.
- Buffer GC: simplified. One `pool_free(backing_buf)` per step instead of per-kernel frees.
- `tensor_decref` constraints: relaxed. GRAD traversal accesses tensor metadata (buf_id,
  shape), not buffer contents. Sub-region offsets stored in metadata are still valid even
  after the backing buffer is freed — the metadata outlives the physical memory.

### IC-Specific Considerations

Unlike tinygrad's linear schedule, TinyHVM's execution is IC reduction — kernels fire
when their input wires become available, not in a predetermined order. This means:

1. **Execution order is not statically known at planning time**. The planner must use the
   DAG dependency order (which is static) as a conservative proxy for actual firing order.
   Any topological sort of the kernel DAG is valid for lifetime analysis.

2. **Loop iterations reuse the same plan**. The planner computes offsets for one iteration.
   The IC loop combinator re-fires kernels into the same backing buffer. Buffer contents
   are overwritten, which is correct — each iteration produces fresh data.

3. **Dynamic IC structure** (conditionals, data-dependent paths) may produce different
   kernel sets per step. The planner must handle the worst case or re-plan when the kernel
   DAG changes. For training loops the DAG is stable, so re-planning is rare.

4. **Parallel reduction**: if multiple reducers fire kernels concurrently, the backing
   buffer must be large enough for all concurrently live sub-regions. The TLSF simulation
   already handles this — concurrent kernels' outputs have overlapping lifetimes and get
   distinct offsets.

## Tinygrad Reference

Tinygrad's memory planner (`tinygrad/engine/memory.py`):
- Walks scheduled kernel DAG
- Assigns buffer numbers based on lifetime analysis
- Achieves massive reduction: 58 unique buffers → 1 via reuse at BS=512
- Key insight: non-overlapping lifetimes of intermediate values enable sharing
- TLSF allocator at `tinygrad/runtime/support/memory.py` (~98 lines Python)
- Original algorithm: Masmano et al., "TLSF: a New Dynamic Memory Allocator for
  Real-Time Systems", ECRTS 2004
- Reference C implementation: mattconte/tlsf (public domain, ~600 lines)

## Phase 3 Execution Net

The phase 3 runtime net is a **general IC graph** (not necessarily a DAG):
- Kernel dependencies form a DAG
- But the surrounding combinator structure (loop, conditionals) may have cycles
- The memory planner dependency graph must still be acyclic
- Plan the DAG portion; let combinators drive iteration

## Integration with Loop

For training loops with kernel re-firing:
- Single iteration's kernels are planned once
- Buffer slots are reused across iterations (same kernel, fresh data)
- The planner doesn't need to see the loop — it plans one iteration's worth
- Phase 3 reduction naturally re-fires planned kernels via combinator sequencing

## Files

- `src/schedule/_.c` — current `sched_all()`, `sched_prepare_boundary_output()`
- `src/interact/tensor_ops.c` — UOP_KERNEL dispatch, `thvm_sched_dispatch_kernel()`
- `src/tinyhvm.h` — KernelEntry, sched_slot_*, buffer management
- `src/heap/alloc.c` — bump-pointer IC heap allocator
- `src/backend/cpu/pool.c` — current buffer pool (to be replaced by TLSF sub-regions)
- Reference: `resources/plans/memory_planner.md`
