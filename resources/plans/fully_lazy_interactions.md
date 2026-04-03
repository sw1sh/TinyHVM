# Fully Lazy Interactions

## Problem

`thvm_interact` is supposed to be a pure rewrite step: take WNF args, produce a new term.
Instead, it has 20+ scattered `thvm_reduce` calls that eagerly force sub-terms mid-interaction.
This is fundamentally wrong — the trampoline (`thvm_reduce`) should be the **only** driver
of reduction. Interaction handlers should never recurse into the reducer.

## Current violations

### interact/_.c — GRAD handler
| Line | Call | Why it exists |
|------|------|---------------|
| 18   | `RETURN_REDUCED(result)` macro | Forces TAG_TOP results to WNF before returning |
| 50   | `GRAD_RETURN(r)` macro | Same pattern for GRAD results |
| 61   | `x = thvm_reduce(ctx, x)` | Trampoline only reduces arg0/arg1; slot 2 (x) stays raw |
| 104  | `GRAD_RETURN(thvm_reduce(ctx, gy))` | Base case forces gy before returning |
| 271  | `thvm_reduce(ctx, orig)` | UOP_FUSING: unfuses by reducing original |
| 276  | `thvm_reduce(ctx, gy)` | UOP_SUM backward: reduces gy before reshape |
| 378  | `thvm_reduce(ctx, gy)` | Conv backward: reduces gy before permute |
| 546  | `thvm_reduce(ctx, y)` | y still lazy → reduce then retry |
| 548  | `thvm_reduce(ctx, t)` | Re-enters reducer with updated y |

### interact/_.c — Phase 3 ops (ASSIGN, TODEVICE, WHERE, IFZ, LOG_PRINT)
| Line | Call | Why it exists |
|------|------|---------------|
| 561  | `thvm_reduce(ctx, src_raw)` | ASSIGN: reduces src |
| 562  | `thvm_reduce(ctx, dst_t)` | ASSIGN: reduces dst |
| 611  | `thvm_reduce(ctx, ...)` | TODEVICE: reduces tensor |
| 612  | `thvm_reduce(ctx, ...)` | TODEVICE: reduces device scalar |
| 644–646 | 3× `thvm_reduce` | WHERE: reduces cond, then, else |
| 675  | `thvm_reduce(ctx, ...)` | IFZ: reduces counter |
| 693  | `thvm_reduce(ctx, ...)` | LOG_PRINT: reduces tensor |

### interact/_.c — reduce ops (SUM/RMAX axes)
| Line | Call | Why it exists |
|------|------|---------------|
| 1025 | `thvm_reduce(ctx, sum_arg)` | Reduces axes tensor for SUM |
| 1052 | `thvm_reduce(ctx, sum_arg)` | Second path for explicit axes |

### fuse/_.c
| Line | Call | Why it exists |
|------|------|---------------|
| 410  | `thvm_reduce(ctx, lt)` | Resolves lazy leaves before re-walking fuse tree |

### inet/_.c (collapse)
| Line | Call | Why it exists |
|------|------|---------------|
| 211  | collapse_dfs | Reduces each node during DFS |
| 239  | grouped_dfs | Same |
| 348  | collapse_ordered | Same |
| 395  | ws_process | Same |

Collapse is a legitimate top-level entry point — it drives reduction. These are fine.

## Root causes

1. **Trampoline only reduces 2 args.** TAG_TOP ops with 3 args (GRAD, WHERE, IFZ)
   must reduce their extra args manually. The trampoline should handle N-ary args.

2. **No "demand" protocol.** When an interaction handler needs a WNF arg that isn't
   ready, it calls `thvm_reduce` instead of returning a term that says "I need this
   reduced first, then re-fire me." The trampoline should handle this.

3. **RETURN_REDUCED / GRAD_RETURN force results.** Interaction results that are TAG_TOP
   get eagerly reduced before returning. But the trampoline already handles re-entering
   `enter:` on returned TAG_TOP — these macros are redundant with the trampoline.

4. **GRAD handler mixes rewriting with reduction.** It reads provenance metadata,
   produces chain-rule formulas (pure rewriting), but also forces sub-terms at
   specific points (gy for SUM backward, y when lazy, x always).

## Design: make `thvm_interact` fully pure

### Principle
`thvm_interact(ctx, t)` is a **single rewrite step**. It reads WNF args from the heap
and returns a new term. It **never** calls `thvm_reduce`. The trampoline is the sole
reducer.

### Phase 1: Extend trampoline to N-ary arg reduction

Currently the trampoline does:
```
enter: push TAG_TOP frame, reduce arg0
apply(TAG_TOP): arg0 done, check arg1 → push TAG_TOP1, reduce arg1
apply(TAG_TOP1): arg1 done, fire interact
```

Extend to 3 args for ops that need it (GRAD x, WHERE cond/then/else, IFZ counter/zero/succ):
```
apply(TAG_TOP1): arg1 done, check if op needs arg2
  → if arity==3: push TAG_TOP2, reduce arg2
  → else: fire interact
apply(TAG_TOP2): arg2 done, fire interact
```

Add a static `u32 uop_arity(u32 uop)` that returns 2 or 3. GRAD, WHERE, IFZ return 3.
Everything else returns 2. This replaces all manual arg reduction inside the handler.

**Exception: GRAD gy (arg1) must stay lazy.** The trampoline already special-cases
`UOP_GRAD` at the TAG_TOP apply phase (line 140) — it fires the handler without reducing
arg1. This is correct and stays. But arg2 (x) DOES need reduction, so the special case
becomes: for GRAD, skip arg1, reduce arg2, then fire.

New trampoline flow for GRAD:
```
enter: push TAG_TOP frame, reduce arg0 (y)
apply(TAG_TOP): arg0 done → GRAD special case → skip arg1 (gy stays lazy)
  → push TAG_TOP2 frame, reduce arg2 (x)
apply(TAG_TOP2): arg2 done, fire interact
```

### Phase 2: Remove RETURN_REDUCED / GRAD_RETURN forcing

These macros exist because the handler wants to guarantee WNF output. But the
trampoline already handles TAG_TOP results: after `thvm_interact` returns, the
trampoline does `next = r; goto enter;` which re-enters the reduce loop. A TAG_TOP
result will naturally be reduced by the trampoline.

Replace:
- `RETURN_REDUCED(result)` → `return (result);`
- `GRAD_RETURN(r)` → just restore flags and `return (r);`

The trampoline's `enter:` loop handles re-reduction. No need to force in the handler.

### Phase 3: Remove remaining `thvm_reduce` inside GRAD handler

After Phase 1, the GRAD handler receives:
- `y` = WNF (reduced by trampoline arg0)
- `gy` = lazy (intentionally, for chain composition)
- `x` = WNF (reduced by trampoline arg2, via Phase 1)

Remaining `thvm_reduce` calls to eliminate:

**Line 104 — base case `thvm_reduce(gy)`:**
The base case `y == x` returns `gy`. Currently forces gy first. Instead: return gy
as-is. The trampoline will reduce it when the consumer demands it.

**Lines 276, 378 — SUM/conv backward `thvm_reduce(gy)`:**
These reduce gy because `thvm_reshape` needs a TAG_TEN input to read shape. Instead:
use `st_get(term_val(gy))` to read shape from the ShapeTracker when gy is TAG_TOP.
The reshape/permute constructors (`thvm_reshape`, `thvm_permute`) should accept lazy
inputs — they create new TAG_TOP nodes wrapping the lazy gy; no reduction needed.
If the constructors already handle TAG_TOP args (they do — they just wrap in new
TAG_TOP), these reduces are unnecessary.

**Lines 546–548 — y still lazy:**
If y is TAG_TOP after arg0 reduction, it means the trampoline couldn't reduce it (stuck).
With Phase 1, arg0 is always reduced by the trampoline before the handler fires.
This dead path can be deleted.

**Line 271 — UOP_FUSING unfuse:**
`thvm_reduce(ctx, orig)` reconstructs the un-fused op and reduces it. This is an
ENSURE-like materialization, not a rewrite. Convert to: return `GRAD3(unfused, gy, x)`
where unfused is the TAG_TOP original (not reduced). The GRAD handler will fire again
on the un-fused term, which the trampoline reduces normally.

### Phase 4: Remove `thvm_reduce` from Phase 3 ops

After Phase 1, all args are WNF when the handler fires. The explicit reduces in
ASSIGN, TODEVICE, WHERE, IFZ, LOG_PRINT are all redundant — delete them and read
args directly from the heap (already WNF).

**ASSIGN special case:** ASSIGN needs both args materialized (buf_id != 0) to blit.
Currently it calls `thvm_reduce` + `ENSURE`. With fully-lazy, ASSIGN should:
1. Receive WNF args from trampoline (TAG_TEN with possibly buf_id=0)
2. Call `ENSURE` on src and dst (ENSURE is not `thvm_reduce` — it's a materialize-to-GPU
   call, which is legitimate for a side-effecting op)

### Phase 5: Remove `thvm_reduce` from fuse/_.c lazy leaf resolution

Line 410: `thvm_reduce(ctx, lt)` resolves lazy fuse-tree leaves. This happens inside
`fuse_or_reduce` which is called from the rewrite layer, not from the interact handler.
The fuser should not reduce inside its walk. Instead:
- If a leaf is lazy (TAG_TOP), reject the fusion attempt (return the original term)
- The trampoline will reduce the leaf, then the fuser can try again on the next pass

This is already partially implemented (the `has_lazy` check exists). Remove the
eager-resolve path and just bail out when `has_lazy == 1`.

### Phase 6: Remove SUM/RMAX axes reduction (lines 1025, 1052)

The axes tensor is arg1 of SUM/RMAX. After Phase 1, the trampoline already reduces
arg1 for non-GRAD ops. These explicit reduces are redundant — delete them.

### Phase 7: Combinator arg reduction via trampoline

The trampoline currently fires `thvm_interact` for combinator tags (APP, DP0/DP1,
OP2, VAR, DSU, DDU, EQL, AND, OR, UDP) without reducing args first. The handlers
reduce args internally via `thvm_reduce`. Move arg reduction to the trampoline.

**TAG_APP**: Trampoline already pushes APP frame at enter, reduces fun (arg0).
In apply phase, `heap_set(loc+0, whnf)` stores reduced fun. But the handler
re-reads and re-reduces it. After trampoline reduces fun:
- If fun is LAM → beta-reduce (already in trampoline apply)
- If fun is SUP → APP-SUP distribution (already in trampoline apply)
- If fun is MAT → need arg1 reduced. Push frame, reduce arg1, then fire.

**TAG_DP0/DP1**: Same pattern — trampoline reduces val (arg0), handler re-reduces.
Remove redundant reduce from handler.

**TAG_OP2**: 2-arg numeric op. Trampoline needs to reduce both args before firing.
Similar to TAG_TOP: push frame, reduce arg0, then check arg1, push OP2_1 frame,
reduce arg1, then fire.

**TAG_VAR**: Substitute and re-reduce. The `thvm_reduce(ctx, sub)` at the end
is a legitimate tail call — return the substitution and let the trampoline reduce it.
Replace with `return sub;` (trampoline handles TAG_TOP/combinator results).

**TAG_DSU/DDU**: Reduce label_expr (arg0). Already handled by trampoline if we
push a frame. For now: just return label_expr and let trampoline reduce.

**TAG_EQL/AND/OR**: 2-arg ops. Same as OP2 — need trampoline frame for both args.

**TAG_UDP**: Like DP0/DP1 — reduce shared value.

**Approach**: For combinator tags that are 1-arg (APP fun, DP0/DP1 val, VAR sub,
DSU/DDU label), the trampoline already enters arg0 at `enter:` via frame push.
The handler should just read the already-reduced arg from the heap. For 2-arg
combinators (OP2, EQL, AND, OR), add a second-arg frame similar to TAG_TOP1.

### Phase 8: WHERE arg2 via extended trampoline

WHERE uses 3 args but TAG_TOP2 only fires for GRAD. Extend TAG_TOP2 to also
fire for WHERE/IFZ (any op that heap_allocs 3+ slots).

## Status

| Phase | Status | thvm_reduce calls |
|-------|--------|-------------------|
| 1 | ✅ DONE | 35 → 34 (GRAD arg2 via TAG_TOP2) |
| 2 | ✅ DONE | 34 → 33 (RETURN_REDUCED/GRAD_RETURN) |
| 4 | ✅ DONE | 33 → 25 (ASSIGN/TODEVICE/IFZ/LOG_PRINT) |
| 6 | ✅ DONE | 25 → 19 (SUM/RMAX axes) |
| 3 | ✅ DONE | 19 → 14 (GRAD handler fully lazy: 0 calls) |
| 5 | TODO   | fuse/_.c lazy leaf (1 call) |
| 7 | TODO   | Combinator args (13 calls in combinators.c) |
| 8 | TODO   | WHERE arg2 (1 call in tensor_ops.c) |

## Ordering

Phase 1 → 2 → 4 → 6 → 3 → 5 → 7 → 8

## Validation

After each phase:
1. `bin/test_full_arch` — gradient correctness (CPU=Metal)
2. Count `thvm_reduce` calls: `grep -rn thvm_reduce src/interact/ | grep -v // | wc -l`
3. Target: **zero** calls in interact/ (grad.c + tensor_ops.c + combinators.c)

## Non-goals

- Collapse functions (`collapse_dfs`, `grouped_dfs`, etc.) legitimately drive reduction
  as top-level entry points. They stay.
- `reduce/_.c` is the trampoline itself — obviously keeps `thvm_reduce`.
- `ENSURE` calls are materialization (GPU dispatch), not reduction. They stay.
