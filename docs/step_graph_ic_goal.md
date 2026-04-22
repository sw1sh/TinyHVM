# Step-graph target: IC-native VJP interaction trace

Goal for the wnf VJP refactor.  Each step is one IC interaction.  The
redex (principal-port edge of the two interacting terms) is drawn red.
All mutations are persistent: a consumed node is ERA'd on the heap, a
new node gets a fresh cell.  No frames, no restore-after-dump hacks.

Example: `vjp_sum_of_square`

```c
f32 xd[] = {1, 2, 3};
Term t1 = thvm_tensor(ctx, xd, SHAPE(3));
Term sq = thvm_op(ctx, UOP_MUL, t1, t1);
Term y  = thvm_sum_axes(ctx, sq, (u32[]){0}, 1);
thvm_eval(ctx, thvm_grad(ctx, y, t1));  // expect [2, 4, 6]
```

## GRAD term layout (IC)

A `GRAD` node has 3 principal ports:
- `y`  — incoming: the forward term being differentiated.
- `tgt` — incoming: the target tensor (for leaf-annihilation match).
- `gy` — incoming: the cotangent accumulator.  The outermost GRAD is
  seeded with `ones(y.shape)`.
Plus 1 free output port: the contributed gradient.

The **redex edge** is always the `y→GRAD` edge: the y-port of the GRAD
connects to the principal port (output) of some other term.  When that
other term reduces to a WHNF head, the VJP rule fires and rewrites the
local neighbourhood.

## Per-rule rewrites

**GRAD ⊳ TEN**.  If `TEN_y == TEN_tgt`, GRAD annihilates and its `gy`
input becomes the output.  Otherwise GRAD annihilates to ERA.

**GRAD ⊳ SUM**.  SUM is consumed.  A new GRAD appears in SUM's position,
with `y = SUM.in` (the pre-sum tensor), `tgt` unchanged, and
`gy = EXPAND(old_gy, shape(SUM.in))`.  The old gy is wrapped in
EXPAND to match the pre-sum shape.

**GRAD ⊳ MUL**.  MUL is consumed.  TWO new GRAD nodes appear, one per
operand of MUL.  Their `y` ports connect to `MUL.a` and `MUL.b`.  Each
gets its own cotangent expression: `MUL.a`'s GRAD has
`gy = MUL(old_gy, MUL.b)` and `MUL.b`'s GRAD has `gy = MUL(old_gy, MUL.a)`.
The old_gy is shared via a DUP cell.  An ADD node wires the two sub-GRADs'
outputs back to the original GRAD's output.

**GRAD ⊳ ADD**.  ADD is consumed.  Two sub-GRADs, one per operand.  Each
gets the SAME gy (via DUP).  Outputs merge via ADD (which itself is
the new output).

**GRAD ⊳ <unary elementwise>** (NEG/EXP/LOG/SQRT/RELU).  Operand's GRAD
appears in the old GRAD's position.  `gy` is wrapped with the Jacobian
factor (e.g. `MUL(old_gy, NEG_ONE)` for NEG; `MUL(old_gy, EXP(a))` for
EXP; `DIV(old_gy, a)` for LOG; …).

## Interaction trace for `vjp_sum_of_square`

Forward is `sum(t1 * t1)` with `t1 = [1,2,3]`.  Redex in each step is
the edge marked `⚡`.  Nodes that just fired are `(consumed)`.  New
nodes introduced at this step are **bold**.

### step_000 — initial state

```
  t1 ──a──┐
          ├──> MUL@1 ──in──> SUM@3 ⚡──y──> GRAD@5 ──out──> (free)
  t1 ──b──┘              ▲                ▲
                      t2.axes      ones@seed ──gy──
                                       t1  ──tgt──
```

- Redex: `SUM ──y──> GRAD` (highlighted)
- Heap cells: `MUL@1, SUM@3, ones@4, GRAD@5 {y, tgt, gy}`

### step_001 — `GRAD ⊳ SUM` fires

SUM and the old GRAD are consumed.  A fresh GRAD slides to MUL; the
old `ones` cotangent is wrapped in a new EXPAND.

```
  t1 ──a──┐
          ├──> MUL@1 ⚡──y──> GRAD@new ──out──> (free)
  t1 ──b──┘                      ▲
                                 │
                 EXPAND@new ──gy──
                ▲        ▲
                │        │
            ones@seed  shape_meta
                      t1  ──tgt──> GRAD@new
```

- Redex: `MUL ──y──> GRAD@new`
- Consumed: SUM@3, GRAD@5.  Their heap cells are now ERA.
- New: `EXPAND@new` (cotangent), `GRAD@new` (sub-GRAD).

### step_002 — `GRAD ⊳ MUL` fires

MUL is consumed.  Two sub-GRADs appear, one per MUL operand.  Their
cotangents cross-multiply with the other operand.  The incoming
EXPAND cotangent is DUP'd because it's used by both sub-GRADs.  An
ADD wires the two sub-GRADs back to the original output port.

```
              EXPAND@prev
                  │
                 DUP@new
                 ╱     ╲
             gy0         gy1
              │           │
              ▼           ▼
  gy0 ──a──┐           ┌──a── gy1
           ├─>MUL_a@new ├─>MUL_b@new
  t1  ──b──┘           └──b── t1

  MUL_a@new ──gy──> GRAD_a@new ⚡──y──> t1
  t1 ──tgt─────────> GRAD_a@new

  MUL_b@new ──gy──> GRAD_b@new ─────y──> t1
  t1 ──tgt─────────> GRAD_b@new

  GRAD_a ──a──┐
              ├─> ADD@new ──out──> (free)
  GRAD_b ──b──┘
```

- Redex: `t1 ──y──> GRAD_a@new` (wnf picks the left arm first)
- Consumed: MUL@1, previous GRAD@new.
- New: `DUP` cell for EXPAND, `MUL_a`, `MUL_b`, `GRAD_a`, `GRAD_b`, `ADD`.

### step_003 — `GRAD ⊳ TEN` fires (left arm)

`t1 == tgt` match.  GRAD_a annihilates; its gy (MUL_a) becomes the
contribution at its output port.  The t1 reference on the y-port is
released.

```
              EXPAND@prev
                  │
                 DUP
                 ╱     ╲
             gy0         gy1
              │           │
  gy0 ──a──┐           ┌──a── gy1
           ├─>MUL_a@new ├─>MUL_b@new
  t1  ──b──┘           └──b── t1

  MUL_a ──a──┐
             ├─> ADD ──out──> (free)
             │
             │          MUL_b ──gy──> GRAD_b ⚡──y──> t1
             │          t1 ──tgt────> GRAD_b
             │
             └── GRAD_b ──b──> ADD
```

- Redex: `t1 ──y──> GRAD_b` (right arm now)
- Consumed: GRAD_a@new (annihilated).
- No new nodes — just a local rewire.

### step_004 — `GRAD ⊳ TEN` fires (right arm)

Same rule for GRAD_b.  Its position in ADD is replaced by MUL_b.

```
              EXPAND
                 │
                DUP
                ╱   ╲
            gy0       gy1
             │         │
  gy0 ──a──┐          ┌──a── gy1
           ├─>MUL_a   ├─>MUL_b
  t1  ──b──┘          └──b── t1

  MUL_a ──a──┐
             ├─> ADD ──out──> (free)
  MUL_b ──b──┘
```

- No GRAD remaining.  Pure backward compute tree.
- All redexes involving GRAD are resolved.

### step_005_final — FUSE phase compiles the bwd tree to a kernel

```
  t1 ──┐
       ├──> KERNEL[ADD(MUL(EXPAND, t1), MUL(EXPAND, t1))] ──out──> gradient
  ones ┘
```

- Single kernel absorbs ADD + both MULs + shared EXPAND.
- Output tensor is `[2, 4, 6]` — the gradient.

## Summary counts

| step | node counts                                 | redex           |
|------|---------------------------------------------|-----------------|
| 000  | MUL=1, SUM=1, GRAD=1                        | SUM → GRAD      |
| 001  | MUL=1, EXPAND=1, GRAD=1                     | MUL → GRAD      |
| 002  | DUP=1, MUL=3, EXPAND=1, GRAD=2, ADD=1       | t1 → GRAD_a     |
| 003  | DUP=1, MUL=3 (→2 cotangents + 0), GRAD=1, ADD=1 | t1 → GRAD_b |
| 004  | DUP=1, MUL=2, EXPAND=1, ADD=1               | (none — done)   |
| 005f | KERNEL=1                                    | (none)          |

## Properties the refactor must preserve

1. **Persistent slide.**  Once `GRAD ⊳ SUM` fires, the new GRAD stays
   on MUL for subsequent steps.  No restore, no flicker.  The heap
   reflects the IC state at all times.

2. **Fresh cells for new GRADs.**  Each sub-GRAD (from binary rules)
   gets its own heap cell.  The original GRAD cell is ERA'd.

3. **Consumed nodes are ERA'd.**  After `GRAD ⊳ SUM` fires, the SUM
   heap cell is released.  The dumper sees no stale SUM node.

4. **DUP only when sharing a non-atom.**  `gy` is a compute TOP →
   DUP'd for binary rules.  `t1` is a TEN atom → aliased without DUP.

5. **Leaf annihilation.**  `GRAD ⊳ TEN` with matching target: the GRAD
   erases and its gy propagates as the output contribution.  No frame,
   no stack, just a local rewrite.

6. **Value correctness.**  Final gradient is `[2, 4, 6]` for this
   example.  All current numeric tests (143/143) continue to pass.

## Implementation sketch

- `thvm_grad` allocates 3 slots: `y, tgt, gy`.  `gy` starts as
  `term_era()`.
- `wnf`'s GRAD enter: if `gy` is ERA, seed with `ones(y.shape)` and
  write to slot 2; then push a `WNF_F_VJP_IC` frame; descend into y
  for WHNF.
- Apply-phase for `WNF_F_VJP_IC`: dispatch on `whnf`'s head.
  - `TEN`: match → `whnf = gy`; else `whnf = term_era()`.  ERA the
    GRAD cell.
  - Unary TOP (`SUM`/`NEG`/…): compute new_gy, mutate GRAD cell in
    place (`y = operand, gy = new_gy`), re-enter GRAD (re-push frame,
    `next = grad_term`, goto enter).
  - Binary TOP (`MUL`/`ADD`/…): allocate two fresh GRAD cells + one
    ADD cell, wire up, ERA old GRAD cell, `whnf = add_term`.
- Delete old `WNF_F_VJP`, `WNF_F_VJP_BIN1`, `WNF_F_VJP_BIN2` frame
  kinds and the `VJP_RECURSE_INTO` / `VJP_BINARY` macros.
- The step-graph hook becomes trivial: just highlight the redex edge
  (always at `grad_slot+0`) — no slide hack, no restore.
