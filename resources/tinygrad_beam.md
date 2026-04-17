# Tinygrad BEAM and JITBEAM

Reference for TinyHVM development — how tinygrad's beam-search kernel optimizer
works, how JITBEAM layers on top, what the action space looks like, and how
results are cached. Based on tinygrad source at `/Users/swish/src/tinygrad`
as of April 2026.

---

## 1. What BEAM Is

BEAM is tinygrad's **beam-search-based kernel autotuner**. After the scheduler
emits a kernel AST and heuristic optimizations produce a baseline, BEAM searches
the space of structural optimizations (upcast, unroll, local, group, padto,
tensor-core, swap) by **actually compiling and timing candidates on the target
device**, keeping the top `amt` performers each round, and iterating until no
further progress is made.

Unlike a pure analytical cost model, BEAM measures real kernel wall time on real
hardware, so it adapts to cache effects, register pressure, and tensor-core
availability without a hand-written performance model.

**Core loop** lives in [`codegen/opt/search.py:137-200`](../../../tinygrad/tinygrad/codegen/opt/search.py):

1. Start from the heuristic-optimized kernel.
2. Generate all valid `Opt` actions via `get_kernel_actions()`.
3. Compile each child in a multiprocessing pool, dedup by binary.
4. Time surviving children 3× on-device; keep `min(tms)`.
5. Prune children whose compute-op count is >1000× the best so far.
6. Keep top `amt` by time — this is the new beam.
7. Exit when best-time improvement < `BEAM_MIN_PROGRESS`.

Typical convergence: 3–10 iterations, 10–300 s per kernel, parallelized across
`PARALLEL` workers.

---

## 2. BEAM vs JITBEAM

| Variable | Scope | Effect |
|----------|-------|--------|
| `BEAM`   | Per-kernel | Beam width used whenever a kernel is lowered |
| `JITBEAM`| Inside `TinyJit` capture | Overrides `BEAM` for kernels realized under a `TinyJit` |

`JITBEAM` exists because JIT-captured graphs amortize compile-time work across
many calls: it's worth spending a deeper beam search (e.g. `JITBEAM=4`) during
capture, even if eager code runs with `BEAM=1` or `BEAM=0`. The override is a
simple `ContextVar` swap in [`engine/jit.py:285`](../../../tinygrad/tinygrad/engine/jit.py).

Common pattern:

```python
with Context(BEAM=1, JITBEAM=4):
    jit_fn(x)     # first call: deep beam search, cached
jit_fn(x)         # subsequent: replays cached opts
```

`IGNORE_JIT_FIRST_BEAM=1` skips search on the first capture call when you
need immediate execution and will accept a cold optimization.

---

## 3. Action Space

Defined as `actions` in [`codegen/opt/search.py:15-26`](../../../tinygrad/tinygrad/codegen/opt/search.py).
Validated at application time by `Kernel.apply_opt` in [`codegen/opt/kernel.py:252-342`](../../../tinygrad/tinygrad/codegen/opt/kernel.py).

| OptOp    | Amounts                        | Axes | Role |
|----------|--------------------------------|------|------|
| UPCAST   | 0, 2, 3, 4, 5, 7               | 8    | Register-level unrolling / per-thread work |
| UNROLL   | 0, 4, 7                        | 5    | Unroll reduce dims for ILP |
| LOCAL    | 2, 3, 4, 8, 13, 16, 29         | 6    | Move an axis to threadgroup / shared memory |
| GROUP    | 0, 4, 8, 16                    | 3    | Group a reduce dim |
| GROUPTOP | 0..N                            | 3    | Top-variant grouping |
| PADTO    | 32                             | 7    | Pad an axis up to a power of two |
| TC       | —                              | —    | Enable tensor cores on matmul-shaped kernels |
| SWAP     | —                              | pairs| Permute axes |
| NOLOCALS | —                              | —    | Disable local-memory allocations |

~400 candidates per iteration per kernel in practice. `apply_opt` rejects
combinations that would blow shared memory, exceed register budgets, or
violate axis-kind invariants.

---

## 4. Constraints & Pruning

Two pruning layers:

**Hard limits** (reject before timing):
- `BEAM_UPCAST_MAX=256` — product of all UPCAST amounts
- `BEAM_LOCAL_MAX=1024` — product of local-memory dims
- `BEAM_UOPS_MAX=3000` — lowered uop count per kernel
- `BEAM_TIMEOUT_SEC=10` — per-candidate compile timeout

**Soft heuristic** (during search):
- Track `least_compute_ops`; drop candidates doing >1000× that arithmetic.
- Abort timing runs that exceed 3× the current best time.

Baseline heuristics (applied before BEAM whenever `NOOPT=0`) live in
[`codegen/opt/heuristic.py`](../../../tinygrad/tinygrad/codegen/opt/heuristic.py):
matvec patterns, reduction grouping, image-float4 loads, small-dim upcast,
small-reduce unroll. BEAM starts from this baseline rather than raw AST.

---

## 5. Cache

Results persist in a SQLite database (WAL mode) at:

```
${XDG_CACHE_HOME:-~/.cache}/tinygrad/cache.db
```

Table names embed a `VERSION` constant (currently 22) so a tinygrad upgrade
auto-invalidates stale entries. The beam cache stores the winning
`applied_opts` list keyed by:

```python
{"ast": lin.ast.key,
 "amt": amt,
 "allow_test_size": bool,
 "device": lin.opts.device,
 "suffix": lin.opts.suffix}
```

Hit path: load the opts list, replay onto a fresh `Kernel`, skip search
entirely (~1–10 ms). Miss path: run the full search and store the result.
Storage helpers: [`helpers.py:226-272`](../../../tinygrad/tinygrad/helpers.py).

Knobs:
- `CACHELEVEL=0` disables all disk caching.
- `IGNORE_BEAM_CACHE=1` forces a miss for this run (still writes result).
- `CACHEDB=/path` overrides the database location.

---

## 6. Integration Point

When a `ScheduleItem` becomes a program:

```
schedule.py  →  realize.get_program()
              →  codegen/opt/__init__.get_optimized_ast()
                    ├── heuristic.py              (baseline)
                    └── if BEAM ≥ 1: search.beam_search()
              →  codegen.full_rewrite()           (linearize UOps)
              →  renderer.render()                (source code)
              →  compiler.compile()               (binary)
              →  CompiledRunner                   (execute)
```

BEAM runs **before linearization**, so it's choosing the high-level loop
structure (tiling, axis placement, reduction shape), not the final codegen.
Entry point: [`codegen/opt/__init__.py:10-36`](../../../tinygrad/tinygrad/codegen/opt/__init__.py).

---

## 7. Environment Variables

**Search width**
- `BEAM` — beam width; 0 disables, 1+ enables. Higher = deeper, slower.
- `JITBEAM` — BEAM override inside `TinyJit` captures.
- `BEAM_MIN_PROGRESS=0.01` — convergence threshold (µs improvement).

**Limits**
- `BEAM_UPCAST_MAX=256`, `BEAM_LOCAL_MAX=1024`, `BEAM_UOPS_MAX=3000`,
  `BEAM_TIMEOUT_SEC=10`.

**Parallelism**
- `PARALLEL=N` — compile workers (default: cpu_count on GPU, 0 on CPU).
- `BEAM_MAX_TASKS_PER_CHILD=16` — recycle workers to bound memory growth.

**Action-space toggles**
- `BEAM_PADTO=1` — enable PADTO actions.
- `NOLOCALS=1` — disable all local-memory actions.
- `NOOPT=1` — skip even the baseline heuristics.

**Diagnostics**
- `BEAM_DEBUG=1|2` — iteration / per-kernel logs.
- `BEAM_LOG_SURPASS_MAX=1` — log constraint rejections.
- `BEAM_STRICT_MODE=1` — raise on compile errors instead of skipping.

**JIT**
- `IGNORE_JIT_FIRST_BEAM=1` — skip BEAM on first JIT capture call.

**Cache**
- `CACHELEVEL=0|1|2`, `IGNORE_BEAM_CACHE=1`, `CACHEDB=/path`.

---

## 8. Relevance to TinyHVM

TinyHVM today does no beam-style autotuning — fusion is done via inet
interaction rules and scheduling is deterministic. Points where BEAM's
design is worth borrowing from or contrasting against:

- **Action = opt, not rewrite.** Tinygrad's `Opt` is a typed edit (axis,
  amount, kind) applied to a kernel; the search is over sequences of these
  edits, not over free-form code. TinyHVM's inet rules are the analogue of
  the baseline heuristic; a future autotuner would need a comparable typed
  action space before beam search makes sense.
- **Timing-based cost.** BEAM sidesteps analytical modeling entirely. For
  Metal / CUDA backends this is practical; for an inet runtime the unit of
  "timeable kernel" is less obvious — the natural unit is the compiled
  kernel that a fused reduce lowers to, not an individual interaction.
- **Cache key = AST hash + device.** Persistent disk cache keyed on a
  stable AST hash is what makes BEAM tolerable in practice. Any TinyHVM
  autotuner would need an analogous canonical key over the fused
  sub-interaction being compiled.
- **JITBEAM split.** Separating per-kernel and per-graph search budgets is
  a clean idea: capture-time work amortizes, eager-time work doesn't.
  Mirrors the distinction between a one-shot reduce and a captured
  training step.

---

## 9. Key Source File Map

| Concern | File |
|---------|------|
| Beam search loop | `tinygrad/codegen/opt/search.py:137-200` |
| Action space | `tinygrad/codegen/opt/search.py:15-26` |
| Candidate validation | `tinygrad/codegen/opt/kernel.py:252-342` |
| Baseline heuristics | `tinygrad/codegen/opt/heuristic.py` |
| BEAM integration | `tinygrad/codegen/opt/__init__.py:10-36` |
| JITBEAM override | `tinygrad/engine/jit.py:285` |
| Disk cache | `tinygrad/helpers.py:226-272` |
| Tests | `tinygrad/test/test_search.py` |
| User docs | `tinygrad/docs/env_vars.md`, `docs/developer/speed.md` |
