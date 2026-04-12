# Kernel Dependency Model

## Problem

When a training loop is unrolled lazily, multiple iterations' kernels and ASSIGNs
coexist in the same graph. The scheduler must respect causal order:

```
Iteration 1:  K1(w) → grad     →  ASSIGN(w, w - lr*grad)
Iteration 2:  K2(w) → grad'    →  ASSIGN(w, w - lr*grad')
                ↑
                must read AFTER iteration 1's ASSIGN writes
```

Without causal ordering, K2 reads the original buffer — wrong result.

## Two Approaches

### A. Dependency in the scheduler (tinygrad approach)

The scheduler analyzes buffer read/write sets and builds a kernel dependency DAG:

```
For each kernel K and each ASSIGN A:
  if K reads buffer B and A writes buffer B:
    K depends on A  (read-after-write)
  if A reads kernel K's output:
    A depends on K  (data flow)
```

Execution follows topological order of this DAG.

**Pros**: works for any graph shape, centralized analysis
**Cons**: requires global buffer tracking, not IC-faithful (external scheduler logic)

### B. Dependency in the IC graph (interaction-native)

Express ordering constraints as IC terms. The graph itself encodes "A must fire
before B" using combinators.

**Option B1: Strict sequencing via AND/OR**

HVM4 has `AND(a, b)`: strict on `a`, then returns `b` if `a` is non-zero.
This is a sequencing primitive:

```
AND(ASSIGN(w, expr), next_computation)
  Phase 1: AND tries to reduce left → ASSIGN is WNF → AND blocks
  Phase 3: ASSIGN fires → returns w (TAG_TEN, non-zero) → AND returns right side
```

Generalized: `SEQ(effect, continuation)` — reduce effect, discard result, return
continuation. This is `AND` without the zero-check.

**Option B2: Threading ASSIGN result as parameter**

Already explored: `train(m)(ASSIGN(w, w-lr*gw))` threads the ASSIGN result
as the next iteration's w parameter. The data dependency is explicit in the graph.

Problem: the fuser resolves ASSIGN to the underlying tensor, losing the dependency.

**Option B3: ASSIGN as a graph edge, not a transparent leaf**

When the fuser encounters ASSIGN as an input to a kernel, it should:
1. Schedule the ASSIGN's source computation as a dependency kernel
2. Mark the ASSIGN as a causal barrier
3. Only THEN schedule the downstream kernel

This means ASSIGN creates a **scheduling dependency edge**, not just a leaf.

## Proposed: Hybrid B2 + B3

The program encodes dependencies via parameter threading (B2):
```
train(m)(ASSIGN(w, w-lr*gw))(ASSIGN(b, b-lr*gb))
```

The scheduler recognizes ASSIGN chains as dependency barriers (B3):
- When building kernel K2 and encountering `ASSIGN(w, ...)` as an input:
  - Schedule the ASSIGN's source expression first (as a dependency kernel)
  - Create a dependency edge: K2 depends on the ASSIGN kernel
  - K2 reads from w's buffer AFTER ASSIGN writes to it

The execution order emerges from the kernel DAG:
```
K1_fwd(w) → K1_grad → K_update_w(w, grad) → ASSIGN(w) → K2_fwd(w) → ...
```

## How ASSIGN Becomes a Dependency Edge

Currently the fuser treats ASSIGN as a transparent leaf (resolves to dst tensor).
Change: ASSIGN should be an **opaque boundary** like UOP_KERNEL:

1. **During fusion walk**: when encountering ASSIGN as input, stop walking.
   ASSIGN is a leaf boundary. Its "output" is the dst tensor.

2. **During scheduling**: ASSIGN becomes a scheduled node in the kernel DAG.
   It has dependencies (its source expression kernels) and dependents
   (any kernel reading the same buffer after it).

3. **During dispatch**: ASSIGN fires after its dependencies complete.
   Downstream kernels fire after ASSIGN completes.

## Relation to UOP_SCHED

UOP_SCHED should build the full dependency DAG:

```
UOP_SCHED fires:
  1. Walk the post-fusion graph
  2. For each UOP_KERNEL and ASSIGN:
     - Record buffer read set and write set
  3. Build dependency edges:
     - RAW (read-after-write): K reads buffer that ASSIGN writes → K depends on ASSIGN
     - WAR (write-after-read): ASSIGN writes buffer that K reads → ASSIGN depends on K
     - WAW (write-after-write): two ASSIGNs on same buffer → ordered
  4. Topological sort → execution plan
  5. Assign buffer slots (memory planning)
```

This is the same analysis a CPU scheduler does (RAW/WAR/WAW hazards).

## IC Faithfulness

The dependency DAG is NOT an IC concept — it's a scheduling optimization.
In a pure IC model, interactions fire in any order (confluence).

But ASSIGN is inherently non-confluent: `ASSIGN(w, 2); read(w)` vs
`read(w); ASSIGN(w, 2)` give different results. ASSIGN breaks confluence.

So: **ASSIGN is the single point where IC ordering matters.** The dependency
model is specifically about ordering ASSIGN relative to reads/writes on the
same buffer. Everything else remains confluent.

The program encodes the intended order via data flow (parameter threading).
The scheduler enforces it via dependency analysis. Both are needed:
- Data flow: tells the scheduler WHAT the intended order is
- Dependency analysis: ensures the scheduler RESPECTS that order

## HVM4 Mechanisms (Not Directly Applicable)

- **INC**: priority hint for collapse enumeration, not execution ordering
- **AND/OR**: short-circuit boolean — could serve as sequencing primitive
  but only for numeric values (zero-check)
- **Strict fields**: OP2, EQL, DSU, DDU force left-to-right evaluation
  in WNF reduction — relevant for eager sub-expressions but not for
  cross-kernel scheduling

## Files

- `src/schedule/_.c` — sched_all, UOP_FUSE/UOP_SCHED handlers
- `src/interact/tensor_ops.c` — ASSIGN interaction, UOP_KERNEL dispatch
- `src/fuse/_.c` — fusion boundary walker (needs ASSIGN awareness)
- `docs/memory.md` — buffer planning (needs dependency DAG integration)
