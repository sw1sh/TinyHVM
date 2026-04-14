# Tinyblog Tinygrad Reference

Archived reference derived from [A Tinyblog about Tinygrad](https://tinyblog-phi.vercel.app/tinygrad), fetched on 2026-04-14.

This is a working reference for TinyHVM design work, not a verbatim mirror. It
captures the parts of the article that matter for fusion, scheduling, lowering,
and runtime structure.

## Core Thesis

- Tinygrad uses one graph IR, `UOp`, across the whole stack.
- Compilation is lazy-first: nothing substantial happens until `realize()`.
- The biggest architectural split is:
  - scheduling / rangeify decides kernel boundaries and execution order
  - lowering decides what each kernel looks like on a target backend
  - runtime executes the resulting kernel schedule

## Frontend to Runtime Pipeline

The article describes the execution path as:

```text
Tensor API
-> lazy UOp graph
-> big SINK root
-> schedule-cache normalization
-> rangeify / realize-map / buffer removal
-> split kernels
-> execution schedule
-> memory planning
-> lowering / rendering
-> runtime dispatch
```

Important takeaway: the "what kernels exist?" decision is made before backend
lowering and before runtime dispatch.

## Architectural Points Relevant to TinyHVM

### 1. One IR, progressively rewritten

- Tinygrad does not lower to a separate low-level IR.
- The same UOp graph is rewritten in place through increasingly concrete stages.
- Kernel boundaries are explicit in the graph, not reconstructed later from
  provenance.

### 2. Single sink root

- Realization starts by inserting one large sink node.
- That root guarantees the compiler sees every demanded output.
- This is valuable for TinyHVM because explicit sink structure lines up well with
  demand-driven IC reduction and visible step graphs.

### 3. Fusion boundary selection is inter-kernel work

- Rangeify / realize-map decides where buffers must exist.
- Buffer removal then decides how aggressively kernels fuse.
- This is not a backend concern; backend lowering only handles the inside of a
  chosen kernel.

### 4. Kernel calls are explicit graph nodes

- After rangeify and split-kernel passes, the graph contains explicit kernel
  call-like structure.
- Runtime consumes those explicit kernel boundaries instead of inventing them on
  the side.

### 5. Memory planning is downstream of kernel structure

- Tinygrad does memory planning after schedule creation.
- Buffer slots are an execution detail attached to the schedule, not the IR
  contract that decides fusion.

## Rangeify Notes

The article emphasizes a few specific phases:

- tagging ops from the sink backward
- building a realize map of materialization boundaries
- propagating ranges backward through the graph
- removing unnecessary buffers to fuse more work
- splitting the graph into kernels
- inserting WAR dependencies
- building the final execution schedule

For TinyHVM, the important analogy is:

- `FUSE` propagation is the local mechanism that discovers fuseable structure
- explicit `KERNEL` nodes should represent the chosen inter-kernel structure
- lowering / dispatch should be subsequent steps, not the visible fusion IR

## Runtime / Observability

The article highlights two things that are particularly relevant to TinyHVM:

- Tinygrad keeps observability high via simple logging and visualization.
- The runtime executes an explicit schedule of kernels rather than hidden
  side-effects buried inside graph rewrites.

That reinforces the TinyHVM design goal that step graphs should show:

```text
FUSE propagation
-> KERNEL node visible on heap
-> KERNEL dispatch
-> consumer interaction
```

## TinyHVM-Oriented Summary

The blog's strongest lesson is not any one pass name. It is the separation of
concerns:

- graph rewriting decides kernel boundaries
- kernel nodes are explicit and inspectable
- lowering is a later concern
- runtime caching is not the user-visible IR

That is exactly the direction taken by the structural `KERNEL` redesign in this
repo.
