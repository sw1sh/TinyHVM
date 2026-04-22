// wnf/_.c — HVM4-style eval/apply stack machine for IC reduction.
//
// See resources/plans/eval_apply_stack_machine.md.
//
// Stage 2: GRAD recursion via stack-based continuation frames.  When a
// GRAD rule needs the gradient of an operand (Leibniz recursion), we
// push a continuation frame encoding "halfway through building the
// output" and descend into the operand.  The operand's own GRAD fires
// (via another frame), returns its result, and the continuation pops
// to assemble the final output.
//
// This eliminates the nested-TOP issue from the old rule where
// recursion built fresh GRAD TOPs that required outer-layer sweeps.
//
// Supported in stage 2:
//   - GRAD × TEN leaf (match → ones, mismatch → ERA)
//   - GRAD × NUM → ERA
//   - GRAD × ERA → ERA
//   - GRAD × TAG_TOP UOP_ADD/UOP_SUB (Leibniz-phase frames)
//   - GRAD × TAG_TOP UOP_NEG (single-phase frame)
//   - GRAD × TAG_TOP UOP_MUL (Leibniz with forward-value cross-term)
// All combinator and GRAD rules are now handled natively; no fallback.

// ──────────────────────────────────────────────────────────────────────
// Frame kinds.  Stored on wnf stack.  Not heap terms.
// ──────────────────────────────────────────────────────────────────────
enum {
    WNF_F_NONE = 0,
    // Descend into y of GRAD(y, tgt).  On apply, dispatch on whnf.
    WNF_F_GRAD,            // t0=tgt
    // ADD/SUB Leibniz, 2-phase
    WNF_F_GRAD_AB_PHASE1,  // t0=b, t1=tgt, t2=op
    WNF_F_GRAD_AB_PHASE2,  // t0=da, t1=op
    // NEG: wrap in NEG.
    WNF_F_GRAD_NEG,
    // MUL Leibniz: b*da + a*db
    WNF_F_GRAD_MUL_PHASE1, // t0=a, t1=b, t2=tgt
    WNF_F_GRAD_MUL_PHASE2, // t0=l=MUL(b,da), t1=a
    // DIV quotient rule: (da*b - a*db) / b²
    WNF_F_GRAD_DIV_PHASE1, // t0=a, t1=b, t2=tgt
    WNF_F_GRAD_DIV_PHASE2, // t0=l=MUL(da,b), t1=a, t2=b
    // Unary nonlinear: da scaled by a function of the operand forward value.
    WNF_F_GRAD_EXP_POST,   // t0=y_whnf (= exp(a) already computed)
    WNF_F_GRAD_LOG_POST,   // t0=a
    WNF_F_GRAD_SQRT_POST,  // t0=y_whnf (= sqrt(a))
    WNF_F_GRAD_RELU_POST,  // t0=a (for mask = a>0)
    // SUM post: fwd → SUM(da, axes); rev → EXPAND(da, a_shape).
    WNF_F_GRAD_SUM_POST,   // t0=a, t1=axes
    // RESHAPE/PERMUTE post: fwd applies forward; rev applies inverse,
    // guarded by numel match (fall back to da as pass-through when
    // target's numel doesn't match the operand).
    WNF_F_GRAD_VIEW_POST,  // t0=a, t1=shape_meta, t2=yloc_as_Term (for y_shape)
                           // flags: is_fwd | (uop << 1)  where uop ∈ {RESHAPE,PERMUTE}
    // SHRINK/PAD fwd post: apply same view to tangent.
    WNF_F_GRAD_SHRINKPAD_FWD_POST, // t0=shape_meta; flags: (uop << 1)
    // MAX Leibniz: d(max(a,b))/dt = (a>=b)*da + (a<b)*db.
    WNF_F_GRAD_MAX_PHASE1, // t0=a, t1=b, t2=tgt — need da
    WNF_F_GRAD_MAX_PHASE2, // t0=a, t1=b, t2=da — need db
    // WHERE: gradient distributes through both branches, masked by cond.
    WNF_F_GRAD_WHERE_PHASE1, // t0=cond, t1=a, t2=b, t3=tgt — need da
    WNF_F_GRAD_WHERE_PHASE2, // t0=cond, t1=b, t2=da, t3=tgt — need db
    // MM forward-mode (JVP): Leibniz JVP(a)@b + a@JVP(b).
    WNF_F_GRAD_MM_FWD_PHASE1, // t0=a, t1=b, t2=tgt — need da
    WNF_F_GRAD_MM_FWD_PHASE2, // t0=l=MM(da,b), t1=a — need db
    // RMAX post: dA = da * mask where mask = 1 - (expand(rmax(a)) > a).
    WNF_F_GRAD_RMAX_POST, // t0 = pre-built mask term
    // APP frame — pushed in enter, popped in apply.
    // t0 = original APP term (for rebuild on unhandled whnf).
    // t1 = arg (APP.slot 1).
    // t2 = APP heap loc.
    WNF_F_APP,
    // SEQ frame — sequential evaluation: reduce slot 0, discard, return slot 1.
    // t0 = original SEQ term (for rebuild when blocked).
    // t1 = b (slot 1).
    // t2 = SEQ heap loc.
    WNF_F_SEQ,
    // DUP aux (DP0/DP1): reduce body, then dispatch.
    // t0 = original DP term (for rebuild on unhandled whnf).
    // flags bit 0 = side (0 for DP0, 1 for DP1).
    WNF_F_DUP,
    // OP2 phase 1: x being reduced (slot 0).  After x is WHNF, need y.
    // t0 = original OP2 term, t2 = OP2 loc, flags = opr.
    WNF_F_OP2_X,
    // OP2 phase 2: x is NUM (stored in t1), now reducing y.  After y is
    // WHNF, compute op.
    // t0 = original OP2, t1 = x (NUM), t2 = OP2 loc, flags = opr.
    WNF_F_OP2_Y,
    // APP ⊳ MAT second phase: whnf of APP.fun was MAT; now drive arg to WHNF.
    // t0 = original APP term, t1 = MAT term (fun), t2 = APP heap loc.
    WNF_F_APP_MAT,
    // DSU: dynamic SUP — reduce slot 0 (label) to WHNF, then build SUP.
    // t0 = original DSU term, t2 = DSU heap loc.
    WNF_F_DSU,
    // DDU: dynamic DUP — reduce slot 0 (label) to WHNF, then DUP+APP.
    // t0 = original DDU term, t2 = DDU heap loc.
    WNF_F_DDU,
    // UDP: unordered DUP consumer — reduce body to WHNF, then dispatch.
    // t0 = original UDP term, t2 = UDP heap loc.
    WNF_F_UDP,
    // EQL phase 1 — x (slot 0) being reduced.
    // t0 = original EQL term, t2 = EQL loc.
    WNF_F_EQL_X,
    // EQL phase 2 — x is an atom (t1), reducing y (slot 1).
    // t0 = original EQL, t1 = x (atom), t2 = EQL loc.
    WNF_F_EQL_Y,
    // AND / OR single-phase — reduce a (slot 0), then dispatch.
    // t0 = original term, t2 = loc, flags bit 0: 0=AND, 1=OR.
    WNF_F_AND_OR,

    // ─── VJP (reverse-mode) frames ────────────────────────────────────
    // Proper reverse-mode auto-diff: seed cotangent g = ones(y.shape) at
    // outermost entry; each op's rule consumes upstream g and produces
    // downstream cotangents for operands.  Leaf rule returns g on match.
    // Kept separate from WNF_F_GRAD (JVP) so the two modes don't entangle.
    //
    // t0 = tgt, t1 = cotangent g (0 sentinel = seed on first apply).
    WNF_F_VJP,
    // Binary-op accumulator phase 1: got grad_a from inner recursion on a,
    // now descend into b with its cotangent g_b.
    // t0 = tgt, t1 = b (operand), t2 = g_b (cotangent for b).
    WNF_F_VJP_BIN1,
    // Binary-op accumulator phase 2: got grad_b.  Return grad_a + grad_b
    // (with ERA peephole).
    // t0 = grad_a (saved from phase 1).
    WNF_F_VJP_BIN2,
};

typedef struct {
    u8   kind;
    u8   flags;     // is_fwd in low bit, op (ADD/SUB) in next
    Term t0, t1, t2, t3;
} WnfFrame;

#define WNF_STACK_INIT_CAP 4096

static WnfFrame *g_wnf_stack_buf = NULL;
static u32       g_wnf_stack_cap = 0;
static u32       g_wnf_stack_pos = 0;

// Per-rule hook fired at each interaction (each ctx->itrs++).  Default
// NULL (no overhead).  Callers register a hook to snapshot heap state
// after each rule — used by the step-graph dumper.
typedef void (*WnfStepHook)(TinyHVM *ctx);
static WnfStepHook g_wnf_step_hook = NULL;

void thvm_wnf_set_step_hook(WnfStepHook hook) { g_wnf_step_hook = hook; }
void thvm_wnf_clear_step_hook(void)          { g_wnf_step_hook = NULL; }

// Per-rule budget.  0 = unbounded.  Set via thvm_reduce_budget() before a
// single call; reset to 0 on entry.  When exceeded, the apply loop stops
// firing new rules and drains the frame stack into a Term representing
// the current partial reduction state.
static u32 g_wnf_budget        = 0;
static u32 g_wnf_budget_fired  = 0;
static int g_wnf_budget_hit    = 0;

#define WNF_FIRED() do { \
    ctx->itrs++; \
    if (g_wnf_step_hook) g_wnf_step_hook(ctx); \
    if (g_wnf_budget > 0 && ++g_wnf_budget_fired >= g_wnf_budget) \
        g_wnf_budget_hit = 1; \
} while (0)

#define WNF_BUDGET_HIT() (g_wnf_budget_hit)

static void wnf_stack_push(WnfFrame f) {
    if (!g_wnf_stack_buf) {
        g_wnf_stack_buf = (WnfFrame *)malloc(sizeof(WnfFrame) * WNF_STACK_INIT_CAP);
        g_wnf_stack_cap = WNF_STACK_INIT_CAP;
        g_wnf_stack_pos = 0;
    }
    if (g_wnf_stack_pos + 1 >= g_wnf_stack_cap) {
        g_wnf_stack_cap *= 2;
        g_wnf_stack_buf = (WnfFrame *)realloc(g_wnf_stack_buf, sizeof(WnfFrame) * g_wnf_stack_cap);
    }
    g_wnf_stack_buf[g_wnf_stack_pos++] = f;
}

// All combinator and GRAD rules are handled natively in this file.

// ──────────────────────────────────────────────────────────────────────
// Shape / cotangent helpers (used by VJP rules).
// ──────────────────────────────────────────────────────────────────────

static Shape wnf_shape_of(TinyHVM *ctx, Term t) {
    Shape s = SHAPE(1);
    // Follow DP / VAR / ANN chains to the underlying shape-carrying term.
    for (int hops = 0; hops < 32; hops++) {
        u8 tag = term_tag(t);
        if (tag == TAG_TEN) {
            u32 id = (u32)term_val(t);
            if (id < ctx->tensor_count) s = ctx->tensors[id].view.shape;
            return s;
        }
        if (tag == TAG_TOP) {
            u64 loc = term_val(t);
            const View *v = st_get(loc);
            if (v) { s = v->shape; return s; }
            // No registered ShapeTracker — derive from operands.
            u32 ext = term_ext(t);
            if (loc == 0) return s;
            u64 shape_slot = 0;
            if (ext == UOP_WHERE) shape_slot = 1;
            t = heap_read(ctx, loc + shape_slot);
            continue;
        }
        if (tag == TAG_DP0 || tag == TAG_DP1) {
            u64 dl = term_val(t);
            if (dl == 0 || dl >= ctx->heap_pos) return s;
            Term cell = heap_read(ctx, dl);
            if (term_is_sub(cell)) cell = term_strip_sub(cell);
            t = cell;
            continue;
        }
        if (tag == TAG_VAR) {
            u64 vl = term_val(t);
            if (vl == 0 || vl >= ctx->heap_pos) return s;
            Term sub = heap_read(ctx, vl);
            if (term_is_sub(sub)) return s;
            t = sub;
            continue;
        }
        if (tag == TAG_ANN) {
            u64 al = term_val(t);
            if (al == 0 || al >= ctx->heap_pos) return s;
            t = heap_read(ctx, al);
            continue;
        }
        return s;
    }
    return s;
}

static Term wnf_ones_of_shape(TinyHVM *ctx, Shape s) {
    // Build a concrete ones TEN of shape s.  We allocate a host buffer so
    // the result is a materialised TEN (rather than EXPAND TOP), which
    // keeps downstream elementwise interactions (WHERE, MUL, ...) from
    // stalling on unmaterialised compute TOPs.
    u32 n = 1;
    for (u32 i = 0; i < s.rank; i++) n *= s.dims[i];
    if (n == 0) n = 1;
    Shape out = s;
    if (out.rank == 0) { out.rank = 1; out.dims[0] = 1; }
    f32 *buf = (f32 *)malloc((size_t)n * sizeof(f32));
    for (u32 i = 0; i < n; i++) buf[i] = 1.0f;
    Term t = thvm_tensor(ctx, buf, out);
    free(buf);
    return t;
}

static Term wnf_zeros_of_shape(TinyHVM *ctx, Shape s) {
    u32 n = 1;
    for (u32 i = 0; i < s.rank; i++) n *= s.dims[i];
    if (n == 0) n = 1;
    Shape out = s;
    if (out.rank == 0) { out.rank = 1; out.dims[0] = 1; }
    f32 *buf = (f32 *)calloc((size_t)n, sizeof(f32));
    Term t = thvm_tensor(ctx, buf, out);
    free(buf);
    return t;
}

// ──────────────────────────────────────────────────────────────────────
// Leaf-rule helpers (legacy JVP path).
// ──────────────────────────────────────────────────────────────────────

static Term wnf_grad_ten_leaf(TinyHVM *ctx, Term tgt, Term y_whnf, int is_fwd) {
    if (term_tag(tgt) != TAG_TEN) {
        // Target not yet resolved — rebuild GRAD(y_whnf, tgt) stuck.
        // Trampoline / caller will retry when tgt reduces further.
        u64 loc = heap_alloc(ctx, 2);
        heap_set(ctx, loc + 0, y_whnf);
        heap_set(ctx, loc + 1, tgt);
        u32 uop = is_fwd ? UOP_GRAD_FWD : UOP_GRAD;
        return term_new(TAG_TOP, uop, loc);
    }
    u32 ytid = (u32)term_val(y_whnf);
    u32 ttid = (u32)term_val(tgt);
    if (ytid != ttid) { WNF_FIRED(); return term_era(); }
    Shape osh = SHAPE(1);
    if (ytid < ctx->tensor_count) osh = ctx->tensors[ytid].view.shape;
    if (is_fwd && osh.rank > 0) {
        u32 n = 1; for (u32 i = 0; i < osh.rank; i++) n *= osh.dims[i];
        f32 *buf = (f32 *)malloc(n * sizeof(f32));
        for (u32 i = 0; i < n; i++) buf[i] = 1.0f;
        Term out = thvm_tensor(ctx, buf, osh);
        free(buf);
        WNF_FIRED(); return out;
    }
    Shape one_shape = {.rank = osh.rank};
    for (u32 i = 0; i < osh.rank; i++) one_shape.dims[i] = 1;
    if (one_shape.rank == 0) { one_shape.rank = 1; one_shape.dims[0] = 1; }
    f32 v = 1.0f;
    Term scalar = thvm_tensor(ctx, &v, one_shape);
    int is_scalar_shape = (osh.rank == 0) ||
                          (osh.rank == 1 && osh.dims[0] == 1);
    WNF_FIRED();
    return is_scalar_shape ? scalar : thvm_expand(ctx, scalar, osh);
}

// Rebuild a stuck GRAD(y_whnf, tgt) term as-is (no further reduction).
// Used when the GRAD apply frame encounters an unhandled whnf shape
// (unknown TOP uop, DP-stuck, VAR, etc.) — matches legacy grad.c's
// "return unchanged" fall-through path (grad.c:540-541).
static Term wnf_grad_rebuild(TinyHVM *ctx, Term tgt, Term y_whnf, int is_fwd) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc + 0, y_whnf);
    heap_set(ctx, loc + 1, tgt);
    u32 uop = is_fwd ? UOP_GRAD_FWD : UOP_GRAD;
    return term_new(TAG_TOP, uop, loc);
}

// ERA peephole helpers (ADD(x, ERA) → x, MUL(x, ERA) → ERA, etc.)
// matching the helpers in src/interact/grad.c.
static inline int wnf_is_era(Term t) { return term_tag(t) == TAG_ERA; }

static Term wnf_mk_add(TinyHVM *ctx, Term a, Term b) {
    if (wnf_is_era(a)) return b;
    if (wnf_is_era(b)) return a;
    return thvm_op_raw(ctx, UOP_ADD, a, b);
}
static Term wnf_mk_sub(TinyHVM *ctx, Term a, Term b) {
    if (wnf_is_era(a)) return thvm_op_raw(ctx, UOP_NEG, b, term_era());
    if (wnf_is_era(b)) return a;
    return thvm_op_raw(ctx, UOP_SUB, a, b);
}
static Term wnf_mk_mul(TinyHVM *ctx, Term a, Term b) {
    if (wnf_is_era(a) || wnf_is_era(b)) return term_era();
    return thvm_op_raw(ctx, UOP_MUL, a, b);
}
static Term wnf_mk_neg(TinyHVM *ctx, Term a) {
    if (wnf_is_era(a)) return term_era();
    return thvm_op_raw(ctx, UOP_NEG, a, term_era());
}

// ──────────────────────────────────────────────────────────────────────
// Main loop.
// ──────────────────────────────────────────────────────────────────────

static inline int wnf_is_atom(u8 tag) {
    // TAG_ERA intentionally excluded: inert ERA(0) is WNF but active
    // ERA(val != 0) must run the walker (handled in enter-phase).
    return tag == TAG_TEN || tag == TAG_NUM ||
           tag == TAG_LAM || tag == TAG_SUP || tag == TAG_ANY;
}

// Stack-drain helper: walk remaining wnf frames and reconstruct a parent
// Term for each, installing whnf at the appropriate slot.  Used when a
// budget-bound reduce exhausts its budget mid-rule; the drain produces a
// valid Term representing the current partial reduction state so the
// caller can resume later.
static Term wnf_drain_stack(TinyHVM *ctx, Term whnf, u32 base) {
    while (g_wnf_stack_pos > base) {
        WnfFrame f = g_wnf_stack_buf[--g_wnf_stack_pos];
        switch (f.kind) {
            case WNF_F_APP: {
                u64 loc = (u64)f.t2;
                if (loc + 1 < ctx->heap_pos) heap_set(ctx, loc + 0, whnf);
                whnf = f.t0;
                break;
            }
            case WNF_F_SEQ: {
                u64 loc = (u64)f.t2;
                if (loc + 1 < ctx->heap_pos) heap_set(ctx, loc + 0, whnf);
                whnf = f.t0;
                break;
            }
            case WNF_F_DUP: {
                u64 dup_loc = term_val(f.t0);
                if (dup_loc < ctx->heap_pos) heap_set(ctx, dup_loc, whnf);
                whnf = f.t0;
                break;
            }
            case WNF_F_OP2_X: {
                u64 loc = (u64)f.t2;
                if (loc + 1 < ctx->heap_pos) heap_set(ctx, loc + 0, whnf);
                whnf = f.t0;
                break;
            }
            case WNF_F_OP2_Y: {
                u64 loc = (u64)f.t2;
                if (loc + 1 < ctx->heap_pos) {
                    heap_set(ctx, loc + 0, f.t1);
                    heap_set(ctx, loc + 1, whnf);
                }
                whnf = f.t0;
                break;
            }
            case WNF_F_APP_MAT: {
                u64 loc = (u64)f.t2;
                if (loc + 1 < ctx->heap_pos) {
                    heap_set(ctx, loc + 0, f.t1);
                    heap_set(ctx, loc + 1, whnf);
                }
                whnf = f.t0;
                break;
            }
            case WNF_F_DSU:
            case WNF_F_DDU: {
                u64 loc = (u64)f.t2;
                if (loc + 2 < ctx->heap_pos) heap_set(ctx, loc + 0, whnf);
                whnf = f.t0;
                break;
            }
            case WNF_F_UDP: {
                u64 udp_loc = (u64)f.t2;
                if (udp_loc < ctx->heap_pos) heap_set(ctx, udp_loc, whnf);
                whnf = f.t0;
                break;
            }
            case WNF_F_EQL_X: {
                u64 loc = (u64)f.t2;
                if (loc + 1 < ctx->heap_pos) heap_set(ctx, loc + 0, whnf);
                whnf = f.t0;
                break;
            }
            case WNF_F_EQL_Y: {
                u64 loc = (u64)f.t2;
                if (loc + 1 < ctx->heap_pos) {
                    heap_set(ctx, loc + 0, f.t1);
                    heap_set(ctx, loc + 1, whnf);
                }
                whnf = f.t0;
                break;
            }
            case WNF_F_AND_OR: {
                u64 loc = (u64)f.t2;
                if (loc + 1 < ctx->heap_pos) heap_set(ctx, loc + 0, whnf);
                whnf = f.t0;
                break;
            }
            // GRAD / VJP frames: rebuild GRAD(whnf, tgt) so the outer caller
            // can resume the reduction later.
            case WNF_F_GRAD:
            case WNF_F_VJP: {
                whnf = wnf_grad_rebuild(ctx, f.t0, whnf, f.kind == WNF_F_GRAD);
                break;
            }
            // VJP binary / grad post frames: these represent mid-rule state.
            // Rebuild a coarse GRAD placeholder; the VJP rule will re-fire
            // on resume.  Loses precision but keeps the Term valid.
            default: {
                // No clean rebuild for mid-rule post frames; leave whnf as-is
                // (best-effort — usually the outer GRAD frame above on the
                // stack will rebuild properly).
                break;
            }
        }
    }
    return whnf;
}

// Public budget-bounded reduce: fire at most `budget` rules, then drain
// the frame stack into a resumable Term.  budget == 0 means unbounded
// (equivalent to thvm_reduce).
Term thvm_reduce_budget(TinyHVM *ctx, Term term, u32 budget);

Term thvm_reduce(TinyHVM *ctx, Term term) {
    return thvm_reduce_budget(ctx, term, 0);
}

Term thvm_reduce_budget(TinyHVM *ctx, Term term, u32 budget) {
    if (wnf_is_atom(term_tag(term))) return term;

    u32 saved_budget       = g_wnf_budget;
    u32 saved_budget_fired = g_wnf_budget_fired;
    int saved_budget_hit   = g_wnf_budget_hit;
    g_wnf_budget           = budget;
    g_wnf_budget_fired     = 0;
    g_wnf_budget_hit       = 0;

    // Outermost-GRAD book-keeping: if the top-level term is a GRAD with a
    // TEN target, remember target.shape so we can materialize ERA to
    // zeros(target.shape) at return.  Inner GRADs flowing through
    // recursion continue to emit ERA (which participates in peephole
    // collapses above them); only the *outermost* result materializes.
    Shape outermost_tgt_shape = {.rank = 0};
    int outermost_is_grad = 0;
    if (g_wnf_stack_pos == 0 && term_tag(term) == TAG_TOP) {
        u32 uop = term_ext(term);
        if (uop == UOP_GRAD || uop == UOP_GRAD_FWD) {
            u64 loc = term_val(term);
            if (loc != 0 && loc + 1 < ctx->heap_pos) {
                Term tgt = heap_read(ctx, loc + 1);
                if (term_tag(tgt) == TAG_TEN) {
                    u32 ttid = (u32)term_val(tgt);
                    if (ttid < ctx->tensor_count) {
                        outermost_tgt_shape = ctx->tensors[ttid].view.shape;
                        outermost_is_grad = 1;
                    }
                }
            }
        }
    }

    u32 base = g_wnf_stack_pos;
    Term next = term;
    Term whnf;

enter: {
    // Budget exhausted in an earlier rule fire — don't start another one.
    // Fall through to apply so pending frames get drained.
    if (WNF_BUDGET_HIT()) { whnf = next; goto apply; }

    u8 tag = term_tag(next);

    if (wnf_is_atom(tag)) { whnf = next; goto apply; }

    // VAR: look up substitution.  If sub-flagged → unbound, return as-is.
    // ERA payload (nonzero val) → active-ERA clone.  DETACH TOP → force.
    // Otherwise → take the sub and re-enter.
    if (tag == TAG_VAR) {
        u64 loc = term_val(next);
        if (loc >= ctx->heap_pos) { whnf = next; goto apply; }
        Term sub = heap_read(ctx, loc);
        if (term_is_sub(sub)) { whnf = next; goto apply; }
        if (term_tag(sub) == TAG_ERA && term_val(sub) != 0) {
            WNF_FIRED();
            whnf = thvm_make_active_era(ctx, sub);
            goto apply;
        }
        if (term_tag(sub) == TAG_TOP && term_ext(sub) == UOP_DETACH) {
            Term forced = thvm_force_tensor_term(ctx, sub);
            if (term_tag(forced) == TAG_TEN) {
                heap_set(ctx, loc, forced);
                WNF_FIRED();
                whnf = forced;
                goto apply;
            }
            whnf = forced;
            goto apply;
        }
        next = sub;
        goto enter;
    }

    // ANN: annotation — strip and return inner (slot 0).
    if (tag == TAG_ANN) {
        u64 loc = term_val(next);
        if (loc >= ctx->heap_pos) { whnf = next; goto apply; }
        WNF_FIRED();
        next = heap_read(ctx, loc);
        goto enter;
    }

    // BRI: bridge — WNF until applied (like LAM).
    if (tag == TAG_BRI) { whnf = next; goto apply; }

    // CTR: unary unwrap / ERA-head shrink; otherwise WNF.
    //   CTR#1{x} → x
    //   CTR#N{ERA, xs...} → CTR#(N-1){xs...}
    if (tag == TAG_CTR) {
        u32 ar = term_ext(next);
        u64 loc = term_val(next);
        if (ar == 0 || loc == 0 || loc >= ctx->heap_pos) {
            whnf = next; goto apply;
        }
        Term head = heap_read(ctx, loc);
        if (ar == 1) {
            WNF_FIRED();
            next = head;
            goto enter;
        }
        if (term_tag(head) == TAG_ERA) {
            WNF_FIRED();
            next = term_new(TAG_CTR, (u8)(ar - 1), loc + 1);
            goto enter;
        }
        // Multi-field CTR with non-ERA head: WNF.
        whnf = next;
        goto apply;
    }

    // REF: unfold named definition into lazy allocation frontier.
    if (tag == TAG_REF) {
        u32 name = (u32)term_ext(next);
        if (name >= ctx->def_count) { whnf = next; goto apply; }
        if (ctx->def_books[name] == 0)
            ctx->def_books[name] = thvm_book_from_dynamic(ctx, ctx->defs[name]);
        WNF_FIRED();
        next = thvm_alo_realize(ctx, ctx->def_books[name], 0);
        goto enter;
    }

    // ALO: force exactly one static/book layer into dynamic net.
    if (tag == TAG_ALO) {
        WNF_FIRED();
        next = thvm_alo_force(ctx, next);
        goto enter;
    }

    // DP0 / DP1: check SUB-bit shortcut, else push frame and descend
    // into body (principal).  Transparent projection for pure compute
    // TOP is inline (no frame push) — HVM4 SUB-bit pattern.
    if (tag == TAG_DP0 || tag == TAG_DP1) {
        u64 dup_loc = term_val(next);
        if (dup_loc == 0 || dup_loc >= ctx->heap_pos) {
            whnf = next; goto apply;
        }
        Term cell = heap_read(ctx, dup_loc);
        if (term_is_sub(cell)) {
            // Sibling already fired (or DUP collapsed via ERA) — take stored.
            Term v = term_strip_sub(cell);
            if (term_tag(v) == TAG_TEN)
                tensor_incref(ctx, (u32)term_val(v));
            WNF_FIRED();
            whnf = v;
            goto apply;
        }
        // Transparent projection for pure compute TOPs: both auxes can
        // share the same TAG_TOP handle without firing the DUP.
        if (term_tag(cell) == TAG_TOP) {
            u32 cu = term_ext(cell);
            if (cu != UOP_DETACH && cu != UOP_ASSIGN &&
                cu != UOP_KERNEL && cu != UOP_EXEC &&
                cu != UOP_GRAD && cu != UOP_GRAD_FWD) {
                whnf = cell;
                goto apply;
            }
        }
        // Push frame, drive body to WHNF.
        WnfFrame f = {
            .kind = WNF_F_DUP,
            .flags = (u8)(tag == TAG_DP1 ? 1 : 0),
            .t0 = next, .t1 = 0, .t2 = 0, .t3 = 0
        };
        wnf_stack_push(f);
        next = cell;
        goto enter;
    }

    // OP2: numeric binary op.  Reduce x first.
    if (tag == TAG_OP2) {
        u64 loc = term_val(next);
        if (loc == 0 || loc + 1 >= ctx->heap_pos) {
            whnf = next; goto apply;
        }
        u32 opr = term_ext(next);
        Term x = heap_read(ctx, loc + 0);
        WnfFrame f = {
            .kind = WNF_F_OP2_X, .flags = (u8)opr,
            .t0 = next, .t1 = 0, .t2 = (Term)loc, .t3 = 0
        };
        wnf_stack_push(f);
        next = x;
        goto enter;
    }

    // SEQ: push continuation frame, descend into slot 0 (strict arg).
    if (tag == TAG_SEQ) {
        u64 loc = term_val(next);
        if (loc == 0 || loc + 1 >= ctx->heap_pos) {
            whnf = next; goto apply;
        }
        Term a = heap_read(ctx, loc + 0);
        Term b = heap_read(ctx, loc + 1);
        WnfFrame f = {
            .kind = WNF_F_SEQ, .flags = 0,
            .t0 = next, .t1 = b, .t2 = (Term)loc, .t3 = 0
        };
        wnf_stack_push(f);
        next = a;
        goto enter;
    }

    // APP: push continuation frame, descend into fun (principal).
    if (tag == TAG_APP) {
        u64 loc = term_val(next);
        if (loc == 0 || loc + 1 >= ctx->heap_pos) {
            whnf = next; goto apply;
        }
        Term fun = heap_read(ctx, loc + 0);
        Term arg = heap_read(ctx, loc + 1);
        WnfFrame f = {
            .kind = WNF_F_APP, .flags = 0,
            .t0 = next, .t1 = arg, .t2 = (Term)loc, .t3 = 0
        };
        wnf_stack_push(f);
        next = fun;
        goto enter;
    }

    if (tag == TAG_TOP) {
        u32 ext = term_ext(next);
        if (ext == UOP_GRAD || ext == UOP_GRAD_FWD) {
            u64 loc = term_val(next);
            if (loc == 0 || loc + 1 >= ctx->heap_pos) {
                whnf = next; goto apply;
            }
            Term y = heap_read(ctx, loc + 0);
            Term tgt = heap_read(ctx, loc + 1);
            // Resolve target to WHNF so DP-wrapped leaves compare as TEN.
            if (term_tag(tgt) != TAG_TEN) {
                Term tgt_r = thvm_reduce(ctx, tgt);
                if (tgt_r != tgt) {
                    heap_set(ctx, loc + 1, tgt_r);
                    tgt = tgt_r;
                }
            }
            if (ext == UOP_GRAD_FWD) {
                // JVP path (unchanged).
                WnfFrame f = {
                    .kind = WNF_F_GRAD, .flags = 1,
                    .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                };
                wnf_stack_push(f);
            } else {
                // VJP (reverse) path — cotangent g seeded on first apply.
                WnfFrame f = {
                    .kind = WNF_F_VJP, .flags = 0,
                    .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                };
                wnf_stack_push(f);
            }
            next = y;
            goto enter;
        }
        // Non-GRAD TOPs (ADD/MUL/KERNEL/ASSIGN/etc.): compute TOPs are
        // treated as WHNF from IC's perspective; dispatch logic lives
        // in enclosing GRAD frames (if any) via apply-phase dispatch.
        // At root level, direct uops (ASSIGN/IFZ/LOG_PRINT/TODEVICE/
        // CAST/DETACH/WHERE/EXEC/KERNEL/FUSE) and certain peephole
        // triggers (ERA-arg, ADD-zero) need the scheduler/tensor
        // materialization path in the legacy trampoline.  Pure compute
        // TOPs (ADD/MUL/SUM/MM/etc.) are already WNF under IC; return
        // them as-is without touching the fallback.
        // Direct uops (ASSIGN/IFZ/LOG_PRINT/TODEVICE/CAST/DETACH/WHERE/
        // EXEC/KERNEL/FUSE) and peephole triggers (ERA-arg, ADD-zero)
        // fire eagerly regardless of frame depth — they're effectful
        // and should run wherever they appear (inside SEQ, at root, ...).
        // Pure compute TOPs (ADD/MUL/SUM/MM/...) remain WNF so that
        // enclosing GRAD frames see the unmaterialized TOP.
        // Exception: WHERE under an immediate GRAD/VJP parent stays lazy
        // so the GRAD/VJP rule can pattern-match on WHERE(c,a,b).
        u8 parent_kind = (g_wnf_stack_pos > base)
            ? g_wnf_stack_buf[g_wnf_stack_pos - 1].kind : 0;
        int under_grad = (parent_kind == WNF_F_GRAD ||
                          parent_kind == WNF_F_VJP);
        if (ext == UOP_WHERE && under_grad) {
            whnf = next;
            goto apply;
        }
        if (reduce_top_direct_uop_ctx(ctx, ext) ||
            reduce_top_has_era_arg(ctx, next) ||
            reduce_top_has_add_zero_arg(ctx, next)) {
            // Drive arg0 (and for ASSIGN also src) to WHNF first —
            // matches the legacy TOP trampoline's "reduce arg0 then fire"
            // scheme so IFZ sees a TEN counter, ASSIGN sees TEN dst, etc.
            u64 tloc = term_val(next);
            if (tloc != 0 && tloc < ctx->heap_pos) {
                Term a0 = heap_read(ctx, tloc + 0);
                Term a0r = thvm_reduce(ctx, a0);
                if (a0r != a0) heap_set(ctx, tloc + 0, a0r);
                if (ext == UOP_ASSIGN && tloc + 1 < ctx->heap_pos) {
                    Term a1 = heap_read(ctx, tloc + 1);
                    Term a1r = thvm_reduce(ctx, a1);
                    // ASSIGN fires only when src is TEN/ERA.  If src is
                    // still a compute TOP, force materialisation via the
                    // scheduler so the blit has a buffer to copy.
                    if (term_tag(a1r) == TAG_TOP) a1r = thvm_eval(ctx, a1r);
                    if (a1r != a1) heap_set(ctx, tloc + 1, a1r);
                }
                // WHERE needs cond + then + else all in WHNF (and materialized
                // to TEN) for the interact rule to fire.
                if (ext == UOP_WHERE && tloc + 2 < ctx->heap_pos) {
                    for (u32 i = 1; i <= 2; i++) {
                        Term ai = heap_read(ctx, tloc + i);
                        Term air = thvm_reduce(ctx, ai);
                        if (term_tag(air) == TAG_TOP) air = thvm_eval(ctx, air);
                        if (air != ai) heap_set(ctx, tloc + i, air);
                    }
                }
                // IFZ needs its counter (arg 0) as a TEN — already covered
                // by the a0 reduction above.
            }
            Term r = thvm_interact(ctx, next);
            if (r != next) {
                WNF_FIRED();
                next = r;
                goto enter;
            }
        }
        whnf = next;
        goto apply;
    }

    // USP: unordered superposition — WNF (like SUP/LAM).
    if (tag == TAG_USP) { whnf = next; goto apply; }

    // MAT: pattern matcher — WNF until applied.
    if (tag == TAG_MAT) { whnf = next; goto apply; }

    // INC: priority wrapper — transparent, tail-call into slot 0.
    if (tag == TAG_INC) {
        u64 loc = term_val(next);
        if (loc >= ctx->heap_pos) { whnf = next; goto apply; }
        WNF_FIRED();
        next = heap_read(ctx, loc);
        goto enter;
    }

    // DSU: dynamic SUP — reduce slot 0 (label) to WHNF, then build SUP.
    if (tag == TAG_DSU) {
        u64 loc = term_val(next);
        if (loc == 0 || loc + 2 >= ctx->heap_pos) { whnf = next; goto apply; }
        Term label_expr = heap_read(ctx, loc + 0);
        WnfFrame f = {
            .kind = WNF_F_DSU, .flags = 0,
            .t0 = next, .t1 = 0, .t2 = (Term)loc, .t3 = 0
        };
        wnf_stack_push(f);
        next = label_expr;
        goto enter;
    }

    // DDU: dynamic DUP — reduce slot 0 (label) to WHNF, then DUP(val)+APP(bod).
    if (tag == TAG_DDU) {
        u64 loc = term_val(next);
        if (loc == 0 || loc + 2 >= ctx->heap_pos) { whnf = next; goto apply; }
        Term label_expr = heap_read(ctx, loc + 0);
        WnfFrame f = {
            .kind = WNF_F_DDU, .flags = 0,
            .t0 = next, .t1 = 0, .t2 = (Term)loc, .t3 = 0
        };
        wnf_stack_push(f);
        next = label_expr;
        goto enter;
    }

    // ERA: active eraser — iterative walker, no frame needed.  Drives the
    // payload term at era_loc through one layer at a time, erasing children
    // and routing one child as the continuation (others become detached
    // ERAs).  Terminates when the payload becomes an atom (TEN/NUM/LAM/ANY/
    // SUP/USP/ERA(0)/CTR-empty) or hits a DP/VAR boundary.
    if (tag == TAG_ERA) {
        u64 era_loc = term_val(next);
        if (era_loc == 0 || era_loc >= ctx->heap_pos) { whnf = next; goto apply; }
        #define WNF_ERA_ARITY(_tg, _ext) ({ \
            u32 _ar = 0; \
            switch ((_tg)) { \
                case TAG_TOP: _ar = thvm_uop_storage_arity((_ext)); break; \
                case TAG_APP: case TAG_LAM: case TAG_BRI: case TAG_SEQ: \
                case TAG_SUP: case TAG_USP: case TAG_OP2: case TAG_EQL: \
                case TAG_AND: case TAG_OR:  case TAG_MAT: case TAG_ANN: \
                case TAG_ALO: _ar = 2; break; \
                case TAG_DSU: case TAG_DDU: _ar = 3; break; \
                case TAG_DP0: case TAG_DP1: case TAG_UDP: case TAG_ERA: \
                case TAG_VAR: case TAG_INC: _ar = 1; break; \
                default: _ar = 0; break; \
            } \
            _ar; \
        })
        u32 ext_bit = term_ext(next) & 1u;
        while (1) {
            Term cur = thvm_era_payload(ctx, heap_read(ctx, era_loc));
            u8 vtag = term_tag(cur);
            u32 vext = term_ext(cur);
            u64 vval = term_val(cur);
            Term cont = term_era();
            int have_cont = 0;
            int done = 0;

            if (vtag == TAG_DP0 || vtag == TAG_DP1) {
                u64 dl = vval;
                if (dl < ctx->heap_pos) {
                    Term cell = heap_read(ctx, dl);
                    if (term_is_sub(cell)) {
                        Term orphan = term_strip_sub(cell);
                        if (term_tag(orphan) != TAG_ERA)
                            thvm_spawn_detached_era(ctx, orphan);
                        heap_set(ctx, dl, term_era());
                    } else {
                        heap_set(ctx, dl, term_set_sub(cell));
                    }
                }
                done = 1;
            } else if (vtag == TAG_VAR) {
                if (vval >= ctx->heap_pos) { done = 1; }
                else {
                    Term sub = heap_read(ctx, vval);
                    if (term_is_sub(sub)) {
                        u64 el2 = heap_alloc(ctx, 1);
                        heap_set(ctx, el2, term_era());
                        heap_set(ctx, vval, term_era_at(el2));
                        done = 1;
                    } else {
                        Term payload = thvm_era_payload(ctx, sub);
                        if (term_tag(payload) == TAG_ERA && term_val(payload) == 0) {
                            done = 1;
                        } else {
                            cont = payload;
                            have_cont = 1;
                        }
                    }
                }
            } else if (vtag == TAG_TEN) {
                tensor_release(ctx, (u32)vval);
                done = 1;
            } else if (vtag == TAG_CTR) {
                u32 ar = vext;
                if (vval < ctx->heap_pos) {
                    for (u32 i = 0; i < ar; i++) {
                        Term child = thvm_era_payload(ctx, heap_read(ctx, vval + i));
                        heap_set(ctx, vval + i, term_era());
                        if (term_tag(child) == TAG_ERA && term_val(child) == 0) continue;
                        if (!have_cont) {
                            cont = child;
                            have_cont = 1;
                        } else {
                            thvm_spawn_detached_era(ctx, child);
                        }
                    }
                }
                if (!have_cont) done = 1;
            } else {
                u32 ar = WNF_ERA_ARITY(vtag, vext);
                if (ar > 0 && vval < ctx->heap_pos) {
                    for (u32 i = 0; i < ar; i++) {
                        Term child = thvm_era_payload(ctx, heap_read(ctx, vval + i));
                        heap_set(ctx, vval + i, term_era());
                        if (term_tag(child) == TAG_ERA && term_val(child) == 0) continue;
                        if (!have_cont) {
                            cont = child;
                            have_cont = 1;
                        } else {
                            thvm_spawn_detached_era(ctx, child);
                        }
                    }
                }
                if (!have_cont) done = 1;
            }

            if (done) {
                heap_set(ctx, era_loc, term_era());
                WNF_FIRED();
                whnf = term_era();
                #undef WNF_ERA_ARITY
                goto apply;
            }
            // Continue walking: install cont at era_loc, flip ext bit, loop.
            heap_set(ctx, era_loc, cont);
            WNF_FIRED();
            ext_bit ^= 1u;
            // Continue iterating — next = ERA(ext_bit, era_loc).
            (void)ext_bit;
        }
    }

    // AND / OR: short-circuit boolean — reduce left, dispatch.
    if (tag == TAG_AND || tag == TAG_OR) {
        u64 loc = term_val(next);
        if (loc == 0 || loc + 1 >= ctx->heap_pos) { whnf = next; goto apply; }
        Term a = heap_read(ctx, loc + 0);
        WnfFrame f = {
            .kind = WNF_F_AND_OR, .flags = (u8)(tag == TAG_OR ? 1 : 0),
            .t0 = next, .t1 = 0, .t2 = (Term)loc, .t3 = 0
        };
        wnf_stack_push(f);
        next = a;
        goto enter;
    }

    // EQL: structural equality — 2-phase CPS reduce x then y.
    if (tag == TAG_EQL) {
        u64 loc = term_val(next);
        if (loc == 0 || loc + 1 >= ctx->heap_pos) { whnf = next; goto apply; }
        Term x = heap_read(ctx, loc + 0);
        WnfFrame f = {
            .kind = WNF_F_EQL_X, .flags = 0,
            .t0 = next, .t1 = 0, .t2 = (Term)loc, .t3 = 0
        };
        wnf_stack_push(f);
        next = x;
        goto enter;
    }

    // UDP: unordered DUP consumer — drive body to WHNF, then dispatch.
    if (tag == TAG_UDP) {
        u64 udp_loc = term_val(next);
        if (udp_loc == 0 || udp_loc >= ctx->heap_pos) { whnf = next; goto apply; }
        Term body = heap_read(ctx, udp_loc);
        WnfFrame f = {
            .kind = WNF_F_UDP, .flags = 0,
            .t0 = next, .t1 = 0, .t2 = (Term)udp_loc, .t3 = 0
        };
        wnf_stack_push(f);
        next = body;
        goto enter;
    }

    // Unknown tag: all 27 TAG_* values are dispatched explicitly above,
    // so this is unreachable in practice.  Leave the term as-is if a
    // new tag is introduced later — an explicit rule is required.
    whnf = next;
    goto apply;
}

apply: {
    while (g_wnf_stack_pos > base) {
        // Budget exhausted — stop popping new frames; drain logic at
        // apply_done will reconstruct the partial state into a Term.
        if (WNF_BUDGET_HIT()) goto apply_done;
        WnfFrame f = g_wnf_stack_buf[--g_wnf_stack_pos];
        u8 wtag = term_tag(whnf);

        switch (f.kind) {
        case WNF_F_GRAD: {
            int is_fwd = f.flags & 1;
            Term tgt = f.t0;

            // Leaf rules
            if (wtag == TAG_TEN) {
                whnf = wnf_grad_ten_leaf(ctx, tgt, whnf, is_fwd);
                continue;
            }
            if (wtag == TAG_NUM || wtag == TAG_ERA) {
                whnf = term_era();
                WNF_FIRED();
                continue;
            }

            if (wtag == TAG_TOP) {
                u32 wuop = term_ext(whnf);
                u64 wloc = term_val(whnf);

                // ADD / SUB: Leibniz, phase 1 (compute da, then db).
                if (wuop == UOP_ADD || wuop == UOP_SUB) {
                    Term a = heap_read(ctx, wloc + 0);
                    Term b = heap_read(ctx, wloc + 1);
                    WnfFrame ph1 = {
                        .kind = WNF_F_GRAD_AB_PHASE1, .flags = (u8)is_fwd,
                        .t0 = b, .t1 = tgt,
                        .t2 = (Term)(u64)wuop,   // encode op number
                        .t3 = 0
                    };
                    wnf_stack_push(ph1);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // NEG: d(-a)/dt = -da
                if (wuop == UOP_NEG) {
                    Term a = heap_read(ctx, wloc + 0);
                    WnfFrame neg = {
                        .kind = WNF_F_GRAD_NEG, .flags = 0,
                        .t0 = 0, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(neg);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // MUL Leibniz: d(a*b)/dt = b*da + a*db — phase 1 (compute da).
                if (wuop == UOP_MUL) {
                    Term a = heap_read(ctx, wloc + 0);
                    Term b = heap_read(ctx, wloc + 1);
                    WnfFrame ph1 = {
                        .kind = WNF_F_GRAD_MUL_PHASE1, .flags = (u8)is_fwd,
                        .t0 = a, .t1 = b, .t2 = tgt, .t3 = 0
                    };
                    wnf_stack_push(ph1);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // DIV quotient rule: d(a/b)/dt = (da*b - a*db) / b²
                if (wuop == UOP_DIV) {
                    Term a = heap_read(ctx, wloc + 0);
                    Term b = heap_read(ctx, wloc + 1);
                    WnfFrame ph1 = {
                        .kind = WNF_F_GRAD_DIV_PHASE1, .flags = (u8)is_fwd,
                        .t0 = a, .t1 = b, .t2 = tgt, .t3 = 0
                    };
                    wnf_stack_push(ph1);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // EXP: d(exp a)/dt = exp(a) * da — whnf IS exp(a).
                if (wuop == UOP_EXP) {
                    Term a = heap_read(ctx, wloc + 0);
                    WnfFrame post = {
                        .kind = WNF_F_GRAD_EXP_POST, .flags = 0,
                        .t0 = whnf, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(post);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // LOG: d(log a)/dt = da / a.
                if (wuop == UOP_LOG) {
                    Term a = heap_read(ctx, wloc + 0);
                    WnfFrame post = {
                        .kind = WNF_F_GRAD_LOG_POST, .flags = 0,
                        .t0 = a, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(post);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // SQRT: d(sqrt a)/dt = da / (2 * sqrt(a)) — whnf IS sqrt(a).
                if (wuop == UOP_SQRT) {
                    Term a = heap_read(ctx, wloc + 0);
                    WnfFrame post = {
                        .kind = WNF_F_GRAD_SQRT_POST, .flags = 0,
                        .t0 = whnf, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(post);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // RELU: d(relu a)/dt = (a>0) * da.
                if (wuop == UOP_RELU) {
                    Term a = heap_read(ctx, wloc + 0);
                    WnfFrame post = {
                        .kind = WNF_F_GRAD_RELU_POST, .flags = 0,
                        .t0 = a, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(post);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // CMP: non-differentiable → ERA.
                if (wuop == UOP_CMP) {
                    whnf = term_era();
                    WNF_FIRED();
                    continue;
                }

                // EXPAND: direct-emit (no recursion on compute-op operand).
                // Leaf operand matching target → sum_to_shape(ones(y_shape)).
                // Leaf operand non-matching → zeros(tgt.shape).
                // Fwd mode: expand recursive da to y_shape.
                if (wuop == UOP_EXPAND) {
                    Term a = heap_read(ctx, wloc + 0);
                    Shape y_shape = SHAPE(1);
                    const View *yv = st_get(wloc); if (yv) y_shape = yv->shape;
                    if (is_fwd) {
                        // Recurse on operand, then expand tangent to y_shape.
                        // Use a post frame.
                        WnfFrame post = {
                            .kind = WNF_F_GRAD_VIEW_POST, .flags = (u8)(is_fwd | (UOP_EXPAND << 1)),
                            .t0 = a, .t1 = 0, .t2 = whnf, .t3 = 0
                        };
                        wnf_stack_push(post);
                        WnfFrame inner = {
                            .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                            .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                        };
                        wnf_stack_push(inner);
                        next = a;
                        goto enter;
                    }
                    // Reverse mode — direct emit, no recursion.
                    u32 ttid = (u32)term_val(tgt);
                    Shape tsh = (ttid < ctx->tensor_count)
                        ? ctx->tensors[ttid].view.shape : SHAPE(1);
                    if (term_tag(a) == TAG_TEN && term_tag(tgt) == TAG_TEN) {
                        u32 a_tid = (u32)term_val(a);
                        if (a_tid != ttid) {
                            // Leaf mismatch — zeros.
                            Shape one_shape = {.rank = tsh.rank};
                            for (u32 i = 0; i < tsh.rank; i++) one_shape.dims[i] = 1;
                            if (one_shape.rank == 0) { one_shape.rank = 1; one_shape.dims[0] = 1; }
                            f32 v = 0.0f;
                            Term scalar = thvm_tensor(ctx, &v, one_shape);
                            int is_sc = (tsh.rank == 0) ||
                                        (tsh.rank == 1 && tsh.dims[0] == 1);
                            whnf = is_sc ? scalar : thvm_expand(ctx, scalar, tsh);
                            WNF_FIRED();
                            continue;
                        }
                    }
                    // ones(y_shape) sum-reduced to tsh.
                    Shape one_shape = {.rank = y_shape.rank};
                    for (u32 i = 0; i < y_shape.rank; i++) one_shape.dims[i] = 1;
                    if (one_shape.rank == 0) { one_shape.rank = 1; one_shape.dims[0] = 1; }
                    f32 v = 1.0f;
                    Term scalar = thvm_tensor(ctx, &v, one_shape);
                    Term y_ones = (y_shape.rank == 0)
                        ? scalar : thvm_expand(ctx, scalar, y_shape);
                    whnf = (y_shape.rank != 0 && tsh.rank != 0)
                        ? sum_to_shape(ctx, y_ones, y_shape, tsh) : y_ones;
                    WNF_FIRED();
                    continue;
                }

                // RESHAPE / PERMUTE: recurse on operand then apply
                // forward (fwd) or inverse (rev) view transform.
                if (wuop == UOP_RESHAPE || wuop == UOP_PERMUTE) {
                    Term a = heap_read(ctx, wloc + 0);
                    Term shape = heap_read(ctx, wloc + 1);
                    WnfFrame post = {
                        .kind = WNF_F_GRAD_VIEW_POST,
                        .flags = (u8)(is_fwd | (wuop << 1)),
                        .t0 = a, .t1 = shape, .t2 = whnf, .t3 = 0
                    };
                    wnf_stack_push(post);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // SHRINK / PAD fwd: apply same op to tangent.
                if ((wuop == UOP_SHRINK || wuop == UOP_PAD) && is_fwd) {
                    Term a = heap_read(ctx, wloc + 0);
                    Term shape = heap_read(ctx, wloc + 1);
                    WnfFrame post = {
                        .kind = WNF_F_GRAD_SHRINKPAD_FWD_POST,
                        .flags = (u8)(wuop << 1),
                        .t0 = shape, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(post);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = 1,  // is_fwd=1
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }
                // SHRINK / PAD rev: direct-emit from shape metadata.
                //   SHRINK bwd: PAD(ones(y_shape), complement_pairs)
                //   PAD    bwd: SHRINK(ones(y_shape), unpad_pairs)
                // If shape metadata is missing / rank 0, recurse on a and
                // drop the view (matches legacy grad.c:344 / grad.c:370).
                if (wuop == UOP_SHRINK || wuop == UOP_PAD) {
                    Term a = heap_read(ctx, wloc + 0);
                    Term shape_t = heap_read(ctx, wloc + 1);
                    Shape a_shape = SHAPE(1);
                    if (term_tag(a) == TAG_TEN) {
                        u32 aid = (u32)term_val(a);
                        if (aid < ctx->tensor_count)
                            a_shape = ctx->tensors[aid].view.shape;
                    } else if (term_tag(a) == TAG_TOP) {
                        const View *av = st_get(term_val(a));
                        if (av) a_shape = av->shape;
                    }
                    if (term_tag(shape_t) == TAG_TEN && a_shape.rank > 0) {
                        u32 sid = (u32)term_val(shape_t);
                        u32 nd = a_shape.rank;
                        u32 sf[MAX_DIM * 2];
                        tensor_meta_read_u32(ctx, sid, sf, MAX_DIM * 2);
                        if (wuop == UOP_SHRINK) {
                            Shape ys = {.rank = nd};
                            for (u32 j = 0; j < nd; j++)
                                ys.dims[j] = sf[j*2+1] - sf[j*2];
                            Shape one_shape = {.rank = ys.rank};
                            for (u32 i = 0; i < ys.rank; i++) one_shape.dims[i] = 1;
                            if (one_shape.rank == 0) {
                                one_shape.rank = 1; one_shape.dims[0] = 1;
                            }
                            f32 v1 = 1.0f;
                            Term scalar = thvm_tensor(ctx, &v1, one_shape);
                            Term y_ones = (ys.rank == 0)
                                ? scalar : thvm_expand(ctx, scalar, ys);
                            u32 pp[MAX_DIM * 2];
                            for (u32 j = 0; j < nd; j++) {
                                pp[j*2]   = sf[j*2];
                                pp[j*2+1] = a_shape.dims[j] - sf[j*2+1];
                            }
                            whnf = thvm_pad(ctx, y_ones, pp, nd);
                            WNF_FIRED();
                            continue;
                        } else { // UOP_PAD
                            Shape y_shape = SHAPE(1);
                            const View *yv2 = st_get(wloc);
                            if (yv2) y_shape = yv2->shape;
                            if (y_shape.rank > 0) {
                                Shape one_shape = {.rank = y_shape.rank};
                                for (u32 i = 0; i < y_shape.rank; i++) one_shape.dims[i] = 1;
                                if (one_shape.rank == 0) {
                                    one_shape.rank = 1; one_shape.dims[0] = 1;
                                }
                                f32 v1 = 1.0f;
                                Term scalar = thvm_tensor(ctx, &v1, one_shape);
                                Term y_ones = thvm_expand(ctx, scalar, y_shape);
                                u32 sp[MAX_DIM * 2];
                                for (u32 j = 0; j < nd; j++) {
                                    sp[j*2]   = sf[j*2];
                                    sp[j*2+1] = sf[j*2] + a_shape.dims[j];
                                }
                                whnf = thvm_shrink(ctx, y_ones, sp, nd);
                                WNF_FIRED();
                                continue;
                            }
                        }
                    }
                    // Fallback: recurse on a with plain GRAD frame — the
                    // view drops in bwd (matches legacy pass-through).
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // MM: fwd is Leibniz (push both operands); rev is direct
                // VJP emit (ones(y.shape) @ bᵀ for a-match, aᵀ @ ones for
                // b-match; zeros otherwise).  No recursion in rev mode.
                if (wuop == UOP_MM) {
                    Term a = heap_read(ctx, wloc + 0);
                    Term b = heap_read(ctx, wloc + 1);
                    if (is_fwd) {
                        WnfFrame ph1 = {
                            .kind = WNF_F_GRAD_MM_FWD_PHASE1, .flags = 1,
                            .t0 = a, .t1 = b, .t2 = tgt, .t3 = 0
                        };
                        wnf_stack_push(ph1);
                        WnfFrame inner = {
                            .kind = WNF_F_GRAD, .flags = 1,
                            .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                        };
                        wnf_stack_push(inner);
                        next = a;
                        goto enter;
                    }
                    // Reverse mode: leaf-match direct emit via tensor id check.
                    u32 a_tid = (term_tag(a) == TAG_TEN) ? (u32)term_val(a) : ~0u;
                    u32 b_tid = (term_tag(b) == TAG_TEN) ? (u32)term_val(b) : ~0u;
                    u32 t_tid = (u32)term_val(tgt);
                    Shape ysh = SHAPE(1);
                    const View *yv = st_get(wloc); if (yv) ysh = yv->shape;
                    Shape tsh = (t_tid < ctx->tensor_count)
                        ? ctx->tensors[t_tid].view.shape : SHAPE(1);
                    if (ysh.rank != 2) {
                        // Non-2D MM — zeros.
                        Shape one_shape = {.rank = tsh.rank};
                        for (u32 i = 0; i < tsh.rank; i++) one_shape.dims[i] = 1;
                        if (one_shape.rank == 0) { one_shape.rank = 1; one_shape.dims[0] = 1; }
                        f32 z = 0.0f;
                        Term scalar = thvm_tensor(ctx, &z, one_shape);
                        int is_sc = (tsh.rank == 0) || (tsh.rank == 1 && tsh.dims[0] == 1);
                        whnf = is_sc ? scalar : thvm_expand(ctx, scalar, tsh);
                        WNF_FIRED();
                        continue;
                    }
                    Shape one_ysh = {.rank = ysh.rank};
                    for (u32 i = 0; i < ysh.rank; i++) one_ysh.dims[i] = 1;
                    f32 v1 = 1.0f;
                    Term ones_scalar = thvm_tensor(ctx, &v1, one_ysh);
                    Term gy = thvm_expand(ctx, ones_scalar, ysh);
                    u32 ax[2] = {1, 0};
                    if (t_tid == a_tid && a_tid != ~0u) {
                        Term bT = thvm_permute(ctx, b, ax, 2);
                        whnf = thvm_op(ctx, UOP_MM, gy, bT);
                        WNF_FIRED();
                        continue;
                    }
                    if (t_tid == b_tid && b_tid != ~0u) {
                        Term aT = thvm_permute(ctx, a, ax, 2);
                        whnf = thvm_op(ctx, UOP_MM, aT, gy);
                        WNF_FIRED();
                        continue;
                    }
                    // Neither leaf matches → zeros of target shape.
                    Shape one_shape = {.rank = tsh.rank};
                    for (u32 i = 0; i < tsh.rank; i++) one_shape.dims[i] = 1;
                    if (one_shape.rank == 0) { one_shape.rank = 1; one_shape.dims[0] = 1; }
                    f32 z = 0.0f;
                    Term scalar = thvm_tensor(ctx, &z, one_shape);
                    int is_sc = (tsh.rank == 0) || (tsh.rank == 1 && tsh.dims[0] == 1);
                    whnf = is_sc ? scalar : thvm_expand(ctx, scalar, tsh);
                    WNF_FIRED();
                    continue;
                }

                // MAX Leibniz: d(max(a,b))/dt = (a>=b)*da + (a<b)*db.
                if (wuop == UOP_MAX) {
                    Term a = heap_read(ctx, wloc + 0);
                    Term b = heap_read(ctx, wloc + 1);
                    WnfFrame ph1 = {
                        .kind = WNF_F_GRAD_MAX_PHASE1, .flags = (u8)is_fwd,
                        .t0 = a, .t1 = b, .t2 = tgt, .t3 = 0
                    };
                    wnf_stack_push(ph1);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // WHERE(cond, a, b): grad distributes linearly through a,b.
                // Fires as two-phase; cond is treated as constant w.r.t. target.
                if (wuop == UOP_WHERE) {
                    Term cond = heap_read(ctx, wloc + 0);
                    Term a = heap_read(ctx, wloc + 1);
                    Term b = heap_read(ctx, wloc + 2);
                    WnfFrame ph1 = {
                        .kind = WNF_F_GRAD_WHERE_PHASE1, .flags = (u8)is_fwd,
                        .t0 = cond, .t1 = a, .t2 = b, .t3 = tgt
                    };
                    wnf_stack_push(ph1);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // SUM: fwd → SUM(da, axes); rev → EXPAND(da, a_shape).
                if (wuop == UOP_SUM) {
                    Term a = heap_read(ctx, wloc + 0);
                    Term axes = heap_read(ctx, wloc + 1);
                    WnfFrame post = {
                        .kind = WNF_F_GRAD_SUM_POST, .flags = (u8)is_fwd,
                        .t0 = a, .t1 = axes, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(post);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a;
                    goto enter;
                }

                // RMAX: dA = da * mask, mask = (a >= expand(rmax(a))).
                // Build the mask eagerly at enter-time using 3 copies of a
                // (via two DUPs), then recurse on the first copy for da.
                if (wuop == UOP_RMAX) {
                    Term a = heap_read(ctx, wloc + 0);
                    Term axes = heap_read(ctx, wloc + 1);
                    Shape a_shape = SHAPE(1);
                    if (term_tag(a) == TAG_TEN) {
                        u32 aid = (u32)term_val(a);
                        if (aid < ctx->tensor_count)
                            a_shape = ctx->tensors[aid].view.shape;
                    } else if (term_tag(a) == TAG_TOP) {
                        const View *av = st_get(term_val(a));
                        if (av) a_shape = av->shape;
                    }
                    Term a0, a1, a1b, a2;
                    thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
                    thvm_dup(ctx, thvm_fresh_label(ctx), a1, &a1b, &a2);
                    Term rm = thvm_op_raw(ctx, UOP_RMAX, a1b, axes);
                    Term rm_bc = (a_shape.rank != 0)
                        ? thvm_expand(ctx, rm, a_shape) : rm;
                    Term gt = thvm_op_raw(ctx, UOP_CMP, rm_bc, a2);
                    Term one = term_num_f32(1.0f);
                    Term mask = thvm_op_raw(ctx, UOP_SUB, one, gt);
                    WnfFrame post = {
                        .kind = WNF_F_GRAD_RMAX_POST, .flags = (u8)is_fwd,
                        .t0 = mask, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(post);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = a0;
                    goto enter;
                }

                // ASSIGN: gradient flows through dst only.  Re-enter with
                // a fresh GRAD frame on dst.
                if (wuop == UOP_ASSIGN) {
                    Term dst = heap_read(ctx, wloc + 0);
                    WnfFrame inner = {
                        .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                        .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
                    };
                    wnf_stack_push(inner);
                    next = dst;
                    goto enter;
                }

                // Unhandled compute TOP: fall back (SUM, RMAX, EXPAND,
                // RESHAPE, PERMUTE, SHRINK, PAD, MM, WHERE).
                whnf = wnf_grad_rebuild(ctx, tgt, whnf, is_fwd);
                continue;
            }

            // Other whnf shapes (DP-stuck, VAR, etc.): fall back.
            whnf = wnf_grad_rebuild(ctx, tgt, whnf, is_fwd);
            continue;
        }

        case WNF_F_VJP: {
            // Reverse-mode (VJP) apply.  Cotangent g is carried in t1;
            // seed at outermost entry if t1 == 0.
            Term tgt = f.t0;
            Term g   = f.t1;
            if (g == 0) {
                // Outermost: seed g = ones(y.shape).
                Shape ys = wnf_shape_of(ctx, whnf);
                g = wnf_ones_of_shape(ctx, ys);
            }

            // Leaf rules.
            if (wtag == TAG_TEN) {
                if (term_tag(tgt) == TAG_TEN) {
                    u32 wtid = (u32)term_val(whnf);
                    u32 ttid = (u32)term_val(tgt);
                    WNF_FIRED();
                    whnf = (wtid == ttid) ? g : term_era();
                    continue;
                }
                // Target not yet resolved — rebuild stuck GRAD(whnf, tgt).
                whnf = wnf_grad_rebuild(ctx, tgt, whnf, 0);
                continue;
            }
            if (wtag == TAG_NUM || wtag == TAG_ERA) {
                whnf = term_era();
                WNF_FIRED();
                continue;
            }

            if (wtag != TAG_TOP) {
                // DP-stuck / VAR / other — rebuild as stuck.
                whnf = wnf_grad_rebuild(ctx, tgt, whnf, 0);
                continue;
            }

            u32 wuop = term_ext(whnf);
            u64 wloc = term_val(whnf);

            // CMP and DETACH are non-differentiable.
            if (wuop == UOP_CMP || wuop == UOP_DETACH) {
                whnf = term_era();
                WNF_FIRED();
                continue;
            }

            // Helper macro: push inner VJP frame on operand `op` with
            // cotangent `gv`, then enter it.
            #define VJP_RECURSE_INTO(op, gv) do { \
                WnfFrame _inner = { .kind = WNF_F_VJP, .flags = 0, \
                    .t0 = tgt, .t1 = (gv), .t2 = 0, .t3 = 0 }; \
                wnf_stack_push(_inner); \
                next = (op); \
                goto enter; \
            } while (0)

            // Helper: push accumulator for binary op.  Descend into a with
            // cotangent g_a; on return we'll descend into b with g_b and
            // sum.
            #define VJP_BINARY(av, bv, ga, gb) do { \
                WnfFrame _ph1 = { .kind = WNF_F_VJP_BIN1, .flags = 0, \
                    .t0 = tgt, .t1 = (bv), .t2 = (gb), .t3 = 0 }; \
                wnf_stack_push(_ph1); \
                WnfFrame _inner = { .kind = WNF_F_VJP, .flags = 0, \
                    .t0 = tgt, .t1 = (ga), .t2 = 0, .t3 = 0 }; \
                wnf_stack_push(_inner); \
                next = (av); \
                goto enter; \
            } while (0)

            // ADD: grad_a = g, grad_b = g (cotangent passes through).
            if (wuop == UOP_ADD) {
                Term a = heap_read(ctx, wloc + 0);
                Term b = heap_read(ctx, wloc + 1);
                Term g0, g1;
                thvm_dup(ctx, thvm_fresh_label(ctx), g, &g0, &g1);
                VJP_BINARY(a, b, g0, g1);
            }
            // SUB: grad_a = g, grad_b = -g.
            if (wuop == UOP_SUB) {
                Term a = heap_read(ctx, wloc + 0);
                Term b = heap_read(ctx, wloc + 1);
                Term g0, g1;
                thvm_dup(ctx, thvm_fresh_label(ctx), g, &g0, &g1);
                Term g1_neg = thvm_op_raw(ctx, UOP_NEG, g1, term_era());
                VJP_BINARY(a, b, g0, g1_neg);
            }
            // NEG: grad_a = -g.
            if (wuop == UOP_NEG) {
                Term a = heap_read(ctx, wloc + 0);
                Term ng = thvm_op_raw(ctx, UOP_NEG, g, term_era());
                VJP_RECURSE_INTO(a, ng);
            }
            // MUL: grad_a = g * b, grad_b = g * a.
            if (wuop == UOP_MUL) {
                Term a = heap_read(ctx, wloc + 0);
                Term b = heap_read(ctx, wloc + 1);
                Term g0, g1; thvm_dup(ctx, thvm_fresh_label(ctx), g, &g0, &g1);
                Term a0, a1; thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
                Term b0, b1; thvm_dup(ctx, thvm_fresh_label(ctx), b, &b0, &b1);
                Term g_a = thvm_op_raw(ctx, UOP_MUL, g0, b0);
                Term g_b = thvm_op_raw(ctx, UOP_MUL, g1, a1);
                VJP_BINARY(a0, b1, g_a, g_b);
                (void)a0; (void)b1; // silence unused if macros short-circuit
            }
            // DIV: grad_a = g / b, grad_b = -g * a / (b*b).
            if (wuop == UOP_DIV) {
                Term a = heap_read(ctx, wloc + 0);
                Term b = heap_read(ctx, wloc + 1);
                Term g0, g1; thvm_dup(ctx, thvm_fresh_label(ctx), g, &g0, &g1);
                Term a0, a1; thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
                Term b0, b1; thvm_dup(ctx, thvm_fresh_label(ctx), b, &b0, &b1);
                Term b1a, b1b; thvm_dup(ctx, thvm_fresh_label(ctx), b1, &b1a, &b1b);
                Term g_a = thvm_op_raw(ctx, UOP_DIV, g0, b0);
                // grad_b = -g * a / (b*b)
                Term bb = thvm_op_raw(ctx, UOP_MUL, b1a, b1b);
                Term ga_num = thvm_op_raw(ctx, UOP_MUL, g1, a1);
                Term ga_neg = thvm_op_raw(ctx, UOP_NEG, ga_num, term_era());
                Term g_b = thvm_op_raw(ctx, UOP_DIV, ga_neg, bb);
                VJP_BINARY(a0, b0, g_a, g_b);
            }
            // EXP: grad_a = g * y (where y = exp(a), = whnf).
            if (wuop == UOP_EXP) {
                Term a = heap_read(ctx, wloc + 0);
                Term y0, y1; thvm_dup(ctx, thvm_fresh_label(ctx), whnf, &y0, &y1);
                Term g_a = thvm_op_raw(ctx, UOP_MUL, g, y0);
                (void)y1; // y was the output; the other copy is dead but keep
                          // the DUP to match other nodes' patterns — TEN DUP
                          // just incref'd, no real cost.
                VJP_RECURSE_INTO(a, g_a);
            }
            // LOG: grad_a = g / a.
            if (wuop == UOP_LOG) {
                Term a = heap_read(ctx, wloc + 0);
                Term a0, a1; thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
                Term g_a = thvm_op_raw(ctx, UOP_DIV, g, a0);
                VJP_RECURSE_INTO(a1, g_a);
            }
            // SQRT: grad_a = g / (2 * y).
            if (wuop == UOP_SQRT) {
                Term a = heap_read(ctx, wloc + 0);
                Term y0 = whnf;
                f32 two_v = 2.0f;
                Shape ys = wnf_shape_of(ctx, whnf);
                Shape one_shape = {.rank = ys.rank};
                for (u32 i = 0; i < ys.rank; i++) one_shape.dims[i] = 1;
                if (one_shape.rank == 0) { one_shape.rank = 1; one_shape.dims[0] = 1; }
                Term two_t = thvm_tensor(ctx, &two_v, one_shape);
                int is_sc = (ys.rank == 0) || (ys.rank == 1 && ys.dims[0] == 1);
                Term two_b = is_sc ? two_t : thvm_expand(ctx, two_t, ys);
                Term den = thvm_op_raw(ctx, UOP_MUL, two_b, y0);
                Term g_a = thvm_op_raw(ctx, UOP_DIV, g, den);
                VJP_RECURSE_INTO(a, g_a);
            }
            // RELU: grad_a = g * (a > 0).  Represent (a > 0) as 1-CMP(0,a).
            // Simpler: use mask = CMP(0, a) returning 1 where a>0 else 0, then
            // g_a = g * mask.
            if (wuop == UOP_RELU) {
                Term a = heap_read(ctx, wloc + 0);
                Term a0, a1; thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
                // CMP returns 1 if x > y else 0: we want (a > 0).
                Shape ash = wnf_shape_of(ctx, a0);
                f32 zero_v = 0.0f;
                Shape one_shape = {.rank = ash.rank};
                for (u32 i = 0; i < ash.rank; i++) one_shape.dims[i] = 1;
                if (one_shape.rank == 0) { one_shape.rank = 1; one_shape.dims[0] = 1; }
                Term zero_t = thvm_tensor(ctx, &zero_v, one_shape);
                int is_sc = (ash.rank == 0) || (ash.rank == 1 && ash.dims[0] == 1);
                Term zero_b = is_sc ? zero_t : thvm_expand(ctx, zero_t, ash);
                Term mask = thvm_op_raw(ctx, UOP_CMP, a0, zero_b);
                Term g_a = thvm_op_raw(ctx, UOP_MUL, g, mask);
                VJP_RECURSE_INTO(a1, g_a);
            }
            // SUM: grad_a = expand(g, a.shape).
            if (wuop == UOP_SUM) {
                Term a = heap_read(ctx, wloc + 0);
                Shape ash = wnf_shape_of(ctx, a);
                Term g_a = (ash.rank != 0) ? thvm_expand(ctx, g, ash) : g;
                VJP_RECURSE_INTO(a, g_a);
            }
            // EXPAND: grad_a = sum_to_shape(g, a.shape).  Sum over broadcast
            // axes where output > input.
            if (wuop == UOP_EXPAND) {
                Term a = heap_read(ctx, wloc + 0);
                Shape ash = wnf_shape_of(ctx, a);
                Shape ysh = SHAPE(1);
                const View *yv = st_get(wloc); if (yv) ysh = yv->shape;
                Term g_a = (ysh.rank != 0 && ash.rank != 0)
                    ? sum_to_shape(ctx, g, ysh, ash) : g;
                VJP_RECURSE_INTO(a, g_a);
            }
            // RESHAPE: grad_a = reshape(g, a.shape).
            if (wuop == UOP_RESHAPE) {
                Term a = heap_read(ctx, wloc + 0);
                Shape ash = wnf_shape_of(ctx, a);
                Term g_a = (ash.rank != 0) ? thvm_reshape(ctx, g, ash) : g;
                VJP_RECURSE_INTO(a, g_a);
            }
            // PERMUTE: grad_a = permute(g, inv_perm).
            if (wuop == UOP_PERMUTE) {
                Term a = heap_read(ctx, wloc + 0);
                Term shape = heap_read(ctx, wloc + 1);
                Shape ash = wnf_shape_of(ctx, a);
                Term g_a = g;
                if (term_tag(shape) == TAG_TEN && ash.rank > 0) {
                    u32 pid = (u32)term_val(shape);
                    u32 nd = ash.rank;
                    u32 pf[MAX_DIM]; tensor_meta_read_u32(ctx, pid, pf, MAX_DIM);
                    u32 inv[MAX_DIM]; for (u32 j = 0; j < nd; j++) inv[pf[j]] = j;
                    g_a = thvm_permute(ctx, g, inv, nd);
                }
                VJP_RECURSE_INTO(a, g_a);
            }
            // SHRINK: grad_a = pad(g, derived_pairs).
            //   forward shrink(a, pairs) keeps a[pair[0]:pair[1]].
            //   Adjoint pads g with zeros: pad(g, [pair[0], a_dim - pair[1]]).
            if (wuop == UOP_SHRINK) {
                Term a = heap_read(ctx, wloc + 0);
                Term shape_t = heap_read(ctx, wloc + 1);
                Shape ash = wnf_shape_of(ctx, a);
                Term g_a = g;
                if (term_tag(shape_t) == TAG_TEN && ash.rank > 0) {
                    u32 sid = (u32)term_val(shape_t);
                    u32 nd = ash.rank;
                    u32 sf[MAX_DIM * 2]; tensor_meta_read_u32(ctx, sid, sf, MAX_DIM * 2);
                    u32 pp[MAX_DIM * 2];
                    for (u32 j = 0; j < nd; j++) {
                        pp[j*2]   = sf[j*2];
                        pp[j*2+1] = ash.dims[j] - sf[j*2+1];
                    }
                    g_a = thvm_pad(ctx, g, pp, nd);
                }
                VJP_RECURSE_INTO(a, g_a);
            }
            // PAD: grad_a = shrink(g, derived_pairs).
            //   forward pad(a, pairs) = wider output.
            //   Adjoint shrinks g back to a.shape: shrink(g, [pair[0], pair[0]+a_dim]).
            if (wuop == UOP_PAD) {
                Term a = heap_read(ctx, wloc + 0);
                Term shape_t = heap_read(ctx, wloc + 1);
                Shape ash = wnf_shape_of(ctx, a);
                Term g_a = g;
                if (term_tag(shape_t) == TAG_TEN && ash.rank > 0) {
                    u32 pid = (u32)term_val(shape_t);
                    u32 nd = ash.rank;
                    u32 pf[MAX_DIM * 2]; tensor_meta_read_u32(ctx, pid, pf, MAX_DIM * 2);
                    u32 sp[MAX_DIM * 2];
                    for (u32 j = 0; j < nd; j++) {
                        sp[j*2]   = pf[j*2];
                        sp[j*2+1] = pf[j*2] + ash.dims[j];
                    }
                    g_a = thvm_shrink(ctx, g, sp, nd);
                }
                VJP_RECURSE_INTO(a, g_a);
            }
            // RMAX: mask = (a == expand(rmax(a), a.shape)); grad_a = expand(g, a.shape) * mask.
            if (wuop == UOP_RMAX) {
                Term a = heap_read(ctx, wloc + 0);
                Term axes = heap_read(ctx, wloc + 1);
                Shape ash = wnf_shape_of(ctx, a);
                Term a0, a1; thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
                Term a1b, a2; thvm_dup(ctx, thvm_fresh_label(ctx), a1, &a1b, &a2);
                Term rm = thvm_op_raw(ctx, UOP_RMAX, a1b, axes);
                Term rm_bc = (ash.rank != 0) ? thvm_expand(ctx, rm, ash) : rm;
                // mask: 1 where a2 >= rm_bc else 0. CMP(x,y) = 1 if x>y else 0.
                // 1 - CMP(rm_bc, a2) == a2 >= rm_bc.
                Term gt = thvm_op_raw(ctx, UOP_CMP, rm_bc, a2);
                Term one = term_num_f32(1.0f);
                Term mask = thvm_op_raw(ctx, UOP_SUB, one, gt);
                Term g_bc = (ash.rank != 0) ? thvm_expand(ctx, g, ash) : g;
                Term g_a = thvm_op_raw(ctx, UOP_MUL, g_bc, mask);
                VJP_RECURSE_INTO(a0, g_a);
            }
            // MM: grad_a = g @ bᵀ, grad_b = aᵀ @ g.
            if (wuop == UOP_MM) {
                Term a = heap_read(ctx, wloc + 0);
                Term b = heap_read(ctx, wloc + 1);
                Term a0, a1; thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
                Term b0, b1; thvm_dup(ctx, thvm_fresh_label(ctx), b, &b0, &b1);
                Term g0, g1; thvm_dup(ctx, thvm_fresh_label(ctx), g, &g0, &g1);
                u32 ax[2] = {1, 0};
                Term bT = thvm_permute(ctx, b0, ax, 2);
                Term aT = thvm_permute(ctx, a0, ax, 2);
                Term g_a = thvm_op(ctx, UOP_MM, g0, bT);
                Term g_b = thvm_op(ctx, UOP_MM, aT, g1);
                VJP_BINARY(a1, b1, g_a, g_b);
            }
            // MAX: grad_a = where(a>=b, g, 0), grad_b = where(a<b, g, 0).
            if (wuop == UOP_MAX) {
                Term a = heap_read(ctx, wloc + 0);
                Term b = heap_read(ctx, wloc + 1);
                Term a0, a1; thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
                Term b0, b1; thvm_dup(ctx, thvm_fresh_label(ctx), b, &b0, &b1);
                Term g0, g1; thvm_dup(ctx, thvm_fresh_label(ctx), g, &g0, &g1);
                // mask_a = (a >= b) = 1 - CMP(b, a).  CMP(b,a)=1 if b>a.
                Term cmp_ba = thvm_op_raw(ctx, UOP_CMP, b0, a0);
                Term one = term_num_f32(1.0f);
                Term mask_a = thvm_op_raw(ctx, UOP_SUB, one, cmp_ba);
                // mask_b = (b > a) = CMP(b, a)  — but we consumed cmp_ba in mask_a.
                // Build a fresh CMP with DUPs already consumed; just use a different path.
                // Actually: mask_b = 1 - mask_a. Use DUP of mask_a if available, else re-cmp.
                Term cmp_ba2 = thvm_op_raw(ctx, UOP_CMP, b1, a1);  // re-build (a, b already dup'd)
                Term g_a = thvm_op_raw(ctx, UOP_MUL, g0, mask_a);
                Term g_b = thvm_op_raw(ctx, UOP_MUL, g1, cmp_ba2);
                // Use the original unduped a, b as recursion targets.  But we
                // already consumed them above; refetch by reading heap slots
                // again — the MUL node's slots are unchanged.
                Term a_for_rec = heap_read(ctx, wloc + 0);
                Term b_for_rec = heap_read(ctx, wloc + 1);
                // But those are the pre-DUP terms; they still work as recursion
                // starting points because DUP is non-destructive from term
                // viewpoint (the cell content is the original).
                VJP_BINARY(a_for_rec, b_for_rec, g_a, g_b);
            }
            // WHERE(c, a, b): grad_a = where(c, g, 0), grad_b = where(c, 0, g).
            // c is non-differentiable.  Resolve c and g to concrete TENs up
            // front (thvm_reduce flushes DP chains) so the two downstream
            // WHERE nodes don't both reference the same DUP cell — that
            // would dead-end one aux under IC linearity and the phase-2
            // reachability sweep can then collapse the result to ERA.
            if (wuop == UOP_WHERE) {
                Term c = heap_read(ctx, wloc + 0);
                Term a = heap_read(ctx, wloc + 1);
                Term b = heap_read(ctx, wloc + 2);
                Term c_r = thvm_reduce(ctx, c);
                Term g_r = thvm_reduce(ctx, g);
                Shape gsh = wnf_shape_of(ctx, g_r);
                Term z_a = wnf_zeros_of_shape(ctx, gsh);
                Term z_b = wnf_zeros_of_shape(ctx, gsh);
                Term g_a = thvm_where(ctx, c_r, g_r, z_a);
                Term g_b = thvm_where(ctx, c_r, z_b, g_r);
                VJP_BINARY(a, b, g_a, g_b);
            }
            // ASSIGN: gradient flows through dst (first arg).
            if (wuop == UOP_ASSIGN) {
                Term dst = heap_read(ctx, wloc + 0);
                VJP_RECURSE_INTO(dst, g);
            }

            #undef VJP_RECURSE_INTO
            #undef VJP_BINARY

            // Unhandled compute TOP: rebuild stuck GRAD.
            whnf = wnf_grad_rebuild(ctx, tgt, whnf, 0);
            continue;
        }

        case WNF_F_VJP_BIN1: {
            // whnf = grad_a (result of inner recursion on a).  Descend into b
            // with cotangent g_b; on return, BIN2 will sum grad_a + grad_b.
            Term grad_a = whnf;
            Term tgt = f.t0;
            Term b   = f.t1;
            Term g_b = f.t2;
            WnfFrame ph2 = {
                .kind = WNF_F_VJP_BIN2, .flags = 0,
                .t0 = grad_a, .t1 = 0, .t2 = 0, .t3 = 0
            };
            wnf_stack_push(ph2);
            WnfFrame inner = {
                .kind = WNF_F_VJP, .flags = 0,
                .t0 = tgt, .t1 = g_b, .t2 = 0, .t3 = 0
            };
            wnf_stack_push(inner);
            next = b;
            goto enter;
        }

        case WNF_F_VJP_BIN2: {
            // whnf = grad_b.  Return grad_a + grad_b (with ERA peepholes).
            Term grad_a = f.t0;
            Term grad_b = whnf;
            whnf = wnf_mk_add(ctx, grad_a, grad_b);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_AB_PHASE1: {
            // whnf = da.  Now compute db.
            Term da = whnf;
            Term b = f.t0, tgt = f.t1;
            u32 op = (u32)(u64)f.t2;
            int is_fwd = f.flags & 1;
            WnfFrame ph2 = {
                .kind = WNF_F_GRAD_AB_PHASE2, .flags = (u8)is_fwd,
                .t0 = da, .t1 = (Term)(u64)op, .t2 = 0, .t3 = 0
            };
            wnf_stack_push(ph2);
            WnfFrame inner = {
                .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
            };
            wnf_stack_push(inner);
            next = b;
            goto enter;
        }

        case WNF_F_GRAD_AB_PHASE2: {
            // whnf = db.  Assemble op(da, db).
            Term da = f.t0;
            u32 op = (u32)(u64)f.t1;
            whnf = (op == UOP_ADD) ? wnf_mk_add(ctx, da, whnf)
                                    : wnf_mk_sub(ctx, da, whnf);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_NEG: {
            // whnf = da.  Produce NEG(da).
            whnf = wnf_mk_neg(ctx, whnf);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_MUL_PHASE1: {
            // whnf = da.  Compute l = MUL(b, da), then descend into db.
            Term da = whnf;
            Term a = f.t0, b = f.t1, tgt = f.t2;
            int is_fwd = f.flags & 1;
            Term l = wnf_mk_mul(ctx, b, da);
            WnfFrame ph2 = {
                .kind = WNF_F_GRAD_MUL_PHASE2, .flags = (u8)is_fwd,
                .t0 = l, .t1 = a, .t2 = tgt, .t3 = 0
            };
            wnf_stack_push(ph2);
            WnfFrame inner = {
                .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
            };
            wnf_stack_push(inner);
            next = b;
            goto enter;
        }

        case WNF_F_GRAD_MUL_PHASE2: {
            // whnf = db.  Assemble ADD(l, MUL(a, db)).
            Term l = f.t0, a = f.t1;
            Term r = wnf_mk_mul(ctx, a, whnf);
            whnf = wnf_mk_add(ctx, l, r);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_DIV_PHASE1: {
            // whnf = da.  Compute l = MUL(da, b), then descend into GRAD(b, tgt).
            Term da = whnf;
            Term a = f.t0, b = f.t1, tgt = f.t2;
            int is_fwd = f.flags & 1;
            Term l = wnf_mk_mul(ctx, da, b);
            WnfFrame ph2 = {
                .kind = WNF_F_GRAD_DIV_PHASE2, .flags = (u8)is_fwd,
                .t0 = l, .t1 = a, .t2 = b, .t3 = 0
            };
            wnf_stack_push(ph2);
            WnfFrame inner = {
                .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
            };
            wnf_stack_push(inner);
            next = b;
            goto enter;
        }

        case WNF_F_GRAD_DIV_PHASE2: {
            // whnf = db.  Assemble (l - a*db) / b².
            Term l = f.t0, a = f.t1, b = f.t2;
            Term r = wnf_mk_mul(ctx, a, whnf);
            Term num = wnf_mk_sub(ctx, l, r);
            if (wnf_is_era(num)) { whnf = term_era(); WNF_FIRED(); continue; }
            Term den = thvm_op_raw(ctx, UOP_MUL, b, b);
            whnf = thvm_op_raw(ctx, UOP_DIV, num, den);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_EXP_POST: {
            // whnf = da.  Multiply by exp(a) (stored as y_whnf in f.t0).
            Term exp_a = f.t0;
            whnf = wnf_mk_mul(ctx, exp_a, whnf);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_LOG_POST: {
            // whnf = da.  Divide by a.
            Term a = f.t0;
            if (wnf_is_era(whnf)) { WNF_FIRED(); continue; }
            whnf = thvm_op_raw(ctx, UOP_DIV, whnf, a);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_SQRT_POST: {
            // whnf = da.  Divide by 2 * sqrt(a) (stored as y_whnf in f.t0).
            Term sq = f.t0;
            if (wnf_is_era(whnf)) { WNF_FIRED(); continue; }
            Term two = term_num_f32(2.0f);
            Term den = thvm_op_raw(ctx, UOP_MUL, two, sq);
            whnf = thvm_op_raw(ctx, UOP_DIV, whnf, den);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_RELU_POST: {
            // whnf = da.  Mask by (a > 0).
            Term a = f.t0;
            if (wnf_is_era(whnf)) { WNF_FIRED(); continue; }
            Term zero = term_num_f32(0.0f);
            Term mask = thvm_op_raw(ctx, UOP_CMP, a, zero);
            whnf = wnf_mk_mul(ctx, whnf, mask);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_VIEW_POST: {
            // whnf = da.  flags bit 0 = is_fwd, bits 1+ = uop (RESHAPE/PERMUTE/EXPAND).
            if (wnf_is_era(whnf)) { WNF_FIRED(); continue; }
            int is_fwd = f.flags & 1;
            u32 uop = f.flags >> 1;
            Term a = f.t0, shape = f.t1, y_tp = f.t2;
            Shape a_shape = SHAPE(1);
            if (term_tag(a) == TAG_TEN) {
                u32 id = (u32)term_val(a);
                if (id < ctx->tensor_count) a_shape = ctx->tensors[id].view.shape;
            } else if (term_tag(a) == TAG_TOP) {
                const View *v = st_get(term_val(a));
                if (v) a_shape = v->shape;
            }
            Shape y_shape = SHAPE(1);
            if (term_tag(y_tp) == TAG_TOP) {
                const View *v = st_get(term_val(y_tp));
                if (v) y_shape = v->shape;
            }
            Term out = whnf;
            if (uop == UOP_EXPAND && is_fwd && y_shape.rank > 0) {
                out = thvm_expand(ctx, whnf, y_shape);
            } else if (uop == UOP_RESHAPE) {
                Shape dst = is_fwd ? y_shape : a_shape;
                if (is_fwd) {
                    if (y_shape.rank > 0) out = thvm_reshape(ctx, whnf, y_shape);
                } else {
                    // Rev: reshape only when target numel matches a numel.
                    // Simplification: always reshape to a_shape when ranks match
                    // (caller ensures shape consistency elsewhere).
                    (void)dst;
                    if (a_shape.rank > 0) out = thvm_reshape(ctx, whnf, a_shape);
                }
            } else if (uop == UOP_PERMUTE &&
                       term_tag(shape) == TAG_TEN && a_shape.rank > 0) {
                u32 pid = (u32)term_val(shape);
                u32 nd = a_shape.rank;
                u32 pf[MAX_DIM]; tensor_meta_read_u32(ctx, pid, pf, MAX_DIM);
                if (is_fwd) {
                    out = thvm_permute(ctx, whnf, pf, nd);
                } else {
                    u32 inv[MAX_DIM]; for (u32 j = 0; j < nd; j++) inv[pf[j]] = j;
                    out = thvm_permute(ctx, whnf, inv, nd);
                }
            }
            whnf = out;
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_SHRINKPAD_FWD_POST: {
            // whnf = da.  Apply same shrink/pad.
            if (wnf_is_era(whnf)) { WNF_FIRED(); continue; }
            u32 uop = f.flags >> 1;
            Term shape = f.t0;
            whnf = thvm_op_raw(ctx, uop, whnf, shape);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_MM_FWD_PHASE1: {
            // whnf = da.  Compute l = MM(da, b), descend into GRAD(b, tgt).
            Term da = whnf;
            Term a = f.t0, b = f.t1, tgt = f.t2;
            Term l = wnf_is_era(da) ? term_era() : thvm_op(ctx, UOP_MM, da, b);
            WnfFrame ph2 = {
                .kind = WNF_F_GRAD_MM_FWD_PHASE2, .flags = 1,
                .t0 = l, .t1 = a, .t2 = 0, .t3 = 0
            };
            wnf_stack_push(ph2);
            WnfFrame inner = {
                .kind = WNF_F_GRAD, .flags = 1,
                .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
            };
            wnf_stack_push(inner);
            next = b;
            goto enter;
        }

        case WNF_F_GRAD_MM_FWD_PHASE2: {
            // whnf = db.  Assemble l + MM(a, db).
            Term l = f.t0, a = f.t1;
            Term db = whnf;
            Term r = wnf_is_era(db) ? term_era() : thvm_op(ctx, UOP_MM, a, db);
            whnf = wnf_mk_add(ctx, l, r);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_MAX_PHASE1: {
            // whnf = da.  Descend into b for db.
            Term a = f.t0, b = f.t1, tgt = f.t2;
            int is_fwd = f.flags & 1;
            WnfFrame ph2 = {
                .kind = WNF_F_GRAD_MAX_PHASE2, .flags = (u8)is_fwd,
                .t0 = a, .t1 = b, .t2 = whnf, .t3 = 0
            };
            wnf_stack_push(ph2);
            WnfFrame inner = {
                .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
            };
            wnf_stack_push(inner);
            next = b;
            goto enter;
        }

        case WNF_F_GRAD_MAX_PHASE2: {
            // whnf = db.  Assemble masked ADD.
            Term a = f.t0, b = f.t1, da = f.t2;
            Term db = whnf;
            if (wnf_is_era(da) && wnf_is_era(db)) { whnf = term_era(); WNF_FIRED(); continue; }
            Term mask_a = thvm_op_raw(ctx, UOP_CMP, a, b);
            Term one = term_num_f32(1.0f);
            Term mask_b = thvm_op_raw(ctx, UOP_SUB, one, mask_a);
            Term l = wnf_mk_mul(ctx, da, mask_a);
            Term r = wnf_mk_mul(ctx, db, mask_b);
            whnf = wnf_mk_add(ctx, l, r);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_WHERE_PHASE1: {
            // whnf = da.  Descend into b for db.
            Term cond = f.t0, b = f.t2, tgt = f.t3;
            int is_fwd = f.flags & 1;
            WnfFrame ph2 = {
                .kind = WNF_F_GRAD_WHERE_PHASE2, .flags = (u8)is_fwd,
                .t0 = cond, .t1 = b, .t2 = whnf, .t3 = 0
            };
            wnf_stack_push(ph2);
            WnfFrame inner = {
                .kind = WNF_F_GRAD, .flags = (u8)is_fwd,
                .t0 = tgt, .t1 = 0, .t2 = 0, .t3 = 0
            };
            wnf_stack_push(inner);
            next = b;
            goto enter;
        }

        case WNF_F_GRAD_WHERE_PHASE2: {
            // whnf = db.  out = WHERE(cond, da, db).
            Term cond = f.t0, da = f.t2;
            Term db = whnf;
            if (wnf_is_era(da) && wnf_is_era(db)) { whnf = term_era(); WNF_FIRED(); continue; }
            if (wnf_is_era(da)) da = thvm_expand(ctx, term_num_f32(0.0f), SHAPE(1));
            if (wnf_is_era(db)) db = thvm_expand(ctx, term_num_f32(0.0f), SHAPE(1));
            whnf = thvm_where(ctx, cond, da, db);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_RMAX_POST: {
            // whnf = da; frame t0 = pre-built mask.
            Term mask = f.t0;
            if (wnf_is_era(whnf)) { WNF_FIRED(); continue; }
            whnf = thvm_op_raw(ctx, UOP_MUL, whnf, mask);
            WNF_FIRED();
            continue;
        }

        case WNF_F_GRAD_SUM_POST: {
            // whnf = da.  Fwd: SUM(da, axes).  Rev: broadcast cotangent
            // back to operand shape — but under our propagate-up-from-leaf
            // semantics, da already carries target.shape (derived from the
            // leaf), so expanding to a_shape would over-shape it.  Only
            // expand when da is smaller than a (numel-wise).
            if (wnf_is_era(whnf)) { WNF_FIRED(); continue; }
            int is_fwd = f.flags & 1;
            Term a = f.t0, axes = f.t1;
            if (is_fwd) {
                whnf = thvm_op_raw(ctx, UOP_SUM, whnf, axes);
            } else {
                Shape a_shape = SHAPE(1);
                if (term_tag(a) == TAG_TEN) {
                    u32 id = (u32)term_val(a);
                    if (id < ctx->tensor_count) a_shape = ctx->tensors[id].view.shape;
                } else if (term_tag(a) == TAG_TOP) {
                    const View *v = st_get(term_val(a));
                    if (v) a_shape = v->shape;
                }
                // Compute da's numel from its view.  If already >= a_shape
                // numel, leave as-is (leaf seeded ones(target.shape) and
                // inner propagation already shaped it).
                u32 a_numel = 1;
                for (u32 i = 0; i < a_shape.rank; i++) a_numel *= a_shape.dims[i];
                const View *dv = NULL;
                u32 da_numel = 1;
                if (term_tag(whnf) == TAG_TEN) {
                    u32 id = (u32)term_val(whnf);
                    if (id < ctx->tensor_count) { dv = &ctx->tensors[id].view; }
                } else if (term_tag(whnf) == TAG_TOP) {
                    dv = st_get(term_val(whnf));
                }
                if (dv) da_numel = dv->numel;
                if (a_shape.rank != 0 && da_numel < a_numel) {
                    whnf = thvm_expand(ctx, whnf, a_shape);
                }
            }
            WNF_FIRED();
            continue;
        }

        case WNF_F_OP2_X: {
            // whnf = x (WHNF).
            Term op2_orig = f.t0;
            u64 op2_loc = (u64)f.t2;
            u32 opr = f.flags;
            Term y = heap_read(ctx, op2_loc + 1);
            // OP2-SUP (left): distribute.
            if (term_tag(whnf) == TAG_SUP) {
                u32 lab = term_ext(whnf);
                u64 sup_loc = term_val(whnf);
                Term x0 = heap_read(ctx, sup_loc + 0);
                Term x1 = heap_read(ctx, sup_loc + 1);
                u64 dup_loc = heap_alloc(ctx, 1);
                heap_set(ctx, dup_loc, y);
                Term y0 = term_new(TAG_DP0, lab, dup_loc);
                Term y1 = term_new(TAG_DP1, lab, dup_loc);
                WNF_FIRED();
                next = thvm_sup(ctx, lab,
                    thvm_op2(ctx, opr, x0, y0),
                    thvm_op2(ctx, opr, x1, y1));
                goto enter;
            }
            // x is WHNF (typically NUM); push phase 2 to reduce y.
            WnfFrame f2 = {
                .kind = WNF_F_OP2_Y, .flags = (u8)opr,
                .t0 = op2_orig, .t1 = whnf, .t2 = (Term)op2_loc, .t3 = 0
            };
            wnf_stack_push(f2);
            next = y;
            goto enter;
        }

        case WNF_F_OP2_Y: {
            // whnf = y (WHNF).  x is f.t1 (WHNF from phase 1).
            Term x = f.t1;
            u32 opr = f.flags;
            // OP2-SUP (right): distribute.
            if (term_tag(whnf) == TAG_SUP) {
                u32 lab = term_ext(whnf);
                u64 sup_loc = term_val(whnf);
                Term y0 = heap_read(ctx, sup_loc + 0);
                Term y1 = heap_read(ctx, sup_loc + 1);
                WNF_FIRED();
                next = thvm_sup(ctx, lab,
                    thvm_op2(ctx, opr, x, y0),
                    thvm_op2(ctx, opr, x, y1));
                goto enter;
            }
            // NUM op NUM → compute.
            if (term_tag(x) == TAG_NUM && term_tag(whnf) == TAG_NUM) {
                u32 xv = term_as_u32(x), yv = term_as_u32(whnf), r;
                switch (opr) {
                    case 0: r = xv + yv; break;
                    case 1: r = xv - yv; break;
                    case 2: r = xv * yv; break;
                    case 3: r = yv ? xv / yv : 0; break;
                    case 4: r = (xv == yv) ? 1 : 0; break;
                    case 5: r = yv ? xv % yv : 0; break;
                    default: r = 0;
                }
                WNF_FIRED();
                whnf = term_num_u32(r);
                continue;
            }
            // Stuck: rebuild OP2 with reduced x,y — return legacy OP2.
            Term op2_orig = f.t0;
            u64 op2_loc = (u64)f.t2;
            heap_set(ctx, op2_loc + 0, x);
            heap_set(ctx, op2_loc + 1, whnf);
            whnf = op2_orig;
            continue;
        }

        case WNF_F_DUP: {
            // frame: original DP term (t0); flags bit 0 = side (0=DP0, 1=DP1).
            // whnf is the body reduced to WHNF.
            Term dp_orig = f.t0;
            u32 side = f.flags & 1;
            u64 dup_loc = term_val(dp_orig);
            u32 dup_label = term_ext(dp_orig);

            // Transparent projection for pure compute TOPs: don't fire the
            // DUP.  Both auxes share the same TAG_TOP handle so the compute
            // materialises once; tensor refcounting on the shared TEN is
            // handled when the TOP later reduces.
            if (term_tag(whnf) == TAG_TOP) {
                u32 vuop = term_ext(whnf);
                if (vuop != UOP_DETACH && vuop != UOP_ASSIGN &&
                    vuop != UOP_KERNEL && vuop != UOP_EXEC &&
                    vuop != UOP_GRAD && vuop != UOP_GRAD_FWD) {
                    // Don't touch the DUP cell; return whnf unchanged.
                    continue;
                }
            }

            Term v0 = 0, v1 = 0;
            int fired = 0;
            u8 wtag = term_tag(whnf);

            // Atoms: DUP just copies.  TEN needs incref.
            if (wtag == TAG_TEN) {
                tensor_incref(ctx, (u32)term_val(whnf));
                v0 = whnf; v1 = whnf; fired = 1;
            } else if (wtag == TAG_ERA || wtag == TAG_NUM ||
                       wtag == TAG_ANY || wtag == TAG_CTR) {
                v0 = whnf; v1 = whnf; fired = 1;
            }
            // DUP ⊳ SUP: annihilate (same label) or commute (different).
            else if (wtag == TAG_SUP) {
                u32 sup_label = term_ext(whnf);
                u64 sup_loc = term_val(whnf);
                if (dup_label == sup_label) {
                    v0 = heap_read(ctx, sup_loc + 0);
                    v1 = heap_read(ctx, sup_loc + 1);
                } else {
                    Term b = heap_read(ctx, sup_loc + 1);
                    u64 du0 = sup_loc;
                    u64 du1 = heap_alloc(ctx, 1);
                    heap_set(ctx, du1, b);
                    u64 su0 = heap_alloc(ctx, 2);
                    u64 su1 = heap_alloc(ctx, 2);
                    heap_set(ctx, su0 + 0, term_new(TAG_DP0, dup_label, du0));
                    heap_set(ctx, su0 + 1, term_new(TAG_DP0, dup_label, du1));
                    heap_set(ctx, su1 + 0, term_new(TAG_DP1, dup_label, du0));
                    heap_set(ctx, su1 + 1, term_new(TAG_DP1, dup_label, du1));
                    v0 = term_new(TAG_SUP, sup_label, su0);
                    v1 = term_new(TAG_SUP, sup_label, su1);
                }
                fired = 1;
            }
            // DUP ⊳ USP: commute preserving unordered tag.
            else if (wtag == TAG_USP) {
                u32 usp_label = term_ext(whnf);
                u64 usp_loc = term_val(whnf);
                Term b = heap_read(ctx, usp_loc + 1);
                u64 du0 = usp_loc;
                u64 du1 = heap_alloc(ctx, 1);
                heap_set(ctx, du1, b);
                u64 su0 = heap_alloc(ctx, 2);
                u64 su1 = heap_alloc(ctx, 2);
                heap_set(ctx, su0 + 0, term_new(TAG_DP0, dup_label, du0));
                heap_set(ctx, su0 + 1, term_new(TAG_DP0, dup_label, du1));
                heap_set(ctx, su1 + 0, term_new(TAG_DP1, dup_label, du0));
                heap_set(ctx, su1 + 1, term_new(TAG_DP1, dup_label, du1));
                v0 = term_new(TAG_USP, usp_label, su0);
                v1 = term_new(TAG_USP, usp_label, su1);
                fired = 1;
            }
            // DUP ⊳ LAM: commutation — duplicate lambda.
            else if (wtag == TAG_LAM) {
                u64 lam_loc = term_val(whnf);
                Term body = heap_read(ctx, lam_loc + 1);
                u64 bdup = heap_alloc(ctx, 1);
                heap_set(ctx, bdup, body);
                Term var0, var1;
                v0 = thvm_lam(ctx, &var0, term_new(TAG_DP0, dup_label, bdup));
                v1 = thvm_lam(ctx, &var1, term_new(TAG_DP1, dup_label, bdup));
                u64 vsup = heap_alloc(ctx, 2);
                heap_set(ctx, vsup + 0, var0);
                heap_set(ctx, vsup + 1, var1);
                heap_set(ctx, lam_loc, term_new(TAG_SUP, dup_label, vsup));
                fired = 1;
            }
            // DUP ⊳ BRI: same as LAM but TAG_BRI.
            else if (wtag == TAG_BRI) {
                u64 bri_loc = term_val(whnf);
                Term body = heap_read(ctx, bri_loc + 1);
                u64 bdup = heap_alloc(ctx, 1);
                heap_set(ctx, bdup, body);
                Term var0, var1;
                v0 = thvm_bri(ctx, &var0, term_new(TAG_DP0, dup_label, bdup));
                v1 = thvm_bri(ctx, &var1, term_new(TAG_DP1, dup_label, bdup));
                u64 vsup = heap_alloc(ctx, 2);
                heap_set(ctx, vsup + 0, var0);
                heap_set(ctx, vsup + 1, var1);
                heap_set(ctx, bri_loc, term_new(TAG_SUP, dup_label, vsup));
                fired = 1;
            }
            // DUP ⊳ ANN: duplicate both term and type.
            else if (wtag == TAG_ANN) {
                u64 ann_loc = term_val(whnf);
                Term inner = heap_read(ctx, ann_loc);
                Term type  = heap_read(ctx, ann_loc + 1);
                u64 idup = heap_alloc(ctx, 1);
                heap_set(ctx, idup, inner);
                u64 tdup = heap_alloc(ctx, 1);
                heap_set(ctx, tdup, type);
                v0 = thvm_ann(ctx, term_new(TAG_DP0, dup_label, idup),
                                   term_new(TAG_DP0, dup_label, tdup));
                v1 = thvm_ann(ctx, term_new(TAG_DP1, dup_label, idup),
                                   term_new(TAG_DP1, dup_label, tdup));
                fired = 1;
            }
            // DUP ⊳ 2-slot compounds (OP2/APP/EQL/AND/OR/MAT/SEQ): DUP-NOD.
            else if (wtag == TAG_OP2 || wtag == TAG_APP ||
                     wtag == TAG_EQL || wtag == TAG_AND ||
                     wtag == TAG_OR  || wtag == TAG_MAT ||
                     wtag == TAG_SEQ) {
                u64 val_loc = term_val(whnf);
                u64 r0 = heap_alloc(ctx, 2);
                u64 r1 = heap_alloc(ctx, 2);
                for (u32 i = 0; i < 2; i++) {
                    Term child = heap_read(ctx, val_loc + i);
                    u64 cdup = heap_alloc(ctx, 1);
                    heap_set(ctx, cdup, child);
                    heap_set(ctx, r0 + i, term_new(TAG_DP0, dup_label, cdup));
                    heap_set(ctx, r1 + i, term_new(TAG_DP1, dup_label, cdup));
                }
                v0 = term_new(wtag, term_ext(whnf), r0);
                v1 = term_new(wtag, term_ext(whnf), r1);
                fired = 1;
            }
            // DUP ⊳ TOP (effectful uops only — pure compute took the
            // transparent projection above).  DETACH force, else DUP-NOD.
            else if (wtag == TAG_TOP) {
                u32 uop = term_ext(whnf);
                if (uop == UOP_DETACH) {
                    Term forced = thvm_eval(ctx, whnf);
                    u8 ft = term_tag(forced);
                    if (ft == TAG_TEN || ft == TAG_ERA || ft == TAG_NUM ||
                        ft == TAG_ANY || ft == TAG_CTR) {
                        if (ft == TAG_TEN) tensor_incref(ctx, (u32)term_val(forced));
                        v0 = forced; v1 = forced; fired = 1;
                    }
                }
                if (!fired) {
                    u64 val_loc = term_val(whnf);
                    u32 arity = thvm_uop_storage_arity(uop);
                    u64 r0 = heap_alloc(ctx, arity);
                    u64 r1 = heap_alloc(ctx, arity);
                    for (u32 i = 0; i < arity; i++) {
                        Term child = heap_read(ctx, val_loc + i);
                        u8 ct = term_tag(child);
                        if (ct == TAG_TEN || ct == TAG_NUM || ct == TAG_ERA ||
                            ct == TAG_ANY || ct == TAG_CTR || ct == TAG_ALO) {
                            heap_set(ctx, r0 + i, child);
                            heap_set(ctx, r1 + i, child);
                        } else {
                            u64 cdup = heap_alloc(ctx, 1);
                            heap_set(ctx, cdup, child);
                            heap_set(ctx, r0 + i, term_new(TAG_DP0, dup_label, cdup));
                            heap_set(ctx, r1 + i, term_new(TAG_DP1, dup_label, cdup));
                        }
                    }
                    v0 = term_new(TAG_TOP, uop, r0);
                    v1 = term_new(TAG_TOP, uop, r1);
                    const ShapeTracker *ast = st_get_tracker(val_loc);
                    if (ast) {
                        st_set_tracker(r0, ast);
                        st_set_tracker(r1, ast);
                    }
                    fired = 1;
                }
            }

            if (fired) {
                Term sibling = (side == 0) ? v1 : v0;
                Term mine    = (side == 0) ? v0 : v1;
                heap_set(ctx, dup_loc, term_set_sub(sibling));
                WNF_FIRED();
                whnf = mine;
                continue;
            }

            // Truly stuck (unknown tag): rebuild DP with body at cell.
            heap_set(ctx, dup_loc, whnf);
            whnf = dp_orig;
            continue;
        }

        case WNF_F_SEQ: {
            // frame: original SEQ term (t0), b (t1), SEQ loc (t2).
            // whnf is `a` in WHNF — dispatch on its tag.
            Term seq_orig = f.t0;
            Term b = f.t1;
            u64 seq_loc = (u64)f.t2;

            // SEQ ⊳ SUP: distribute through superposition.
            if (term_tag(whnf) == TAG_SUP) {
                u32 lab = term_ext(whnf);
                u64 sloc = term_val(whnf);
                Term a0 = heap_read(ctx, sloc + 0);
                Term a1 = heap_read(ctx, sloc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, b);
                u64 s0loc = heap_alloc(ctx, 2);
                heap_set(ctx, s0loc + 0, a0);
                heap_set(ctx, s0loc + 1, term_new(TAG_DP0, lab, dup));
                u64 s1loc = heap_alloc(ctx, 2);
                heap_set(ctx, s1loc + 0, a1);
                heap_set(ctx, s1loc + 1, term_new(TAG_DP1, lab, dup));
                WNF_FIRED();
                next = thvm_sup(ctx, lab,
                    term_new(TAG_SEQ, 0, s0loc),
                    term_new(TAG_SEQ, 0, s1loc));
                goto enter;
            }

            // SEQ ⊳ TAG_TOP: blocked — compute op not yet materialized.
            // Reinstall whnf in slot 0 and return the SEQ term as-is
            // (not further reducible in pure IC at this point).
            if (term_tag(whnf) == TAG_TOP) {
                heap_set(ctx, seq_loc + 0, whnf);
                whnf = seq_orig;
                continue;
            }

            // SEQ ⊳ {ERA, TEN, NUM, CTR, LAM, ...}: value — discard, return b.
            WNF_FIRED();
            next = b;
            goto enter;
        }

        case WNF_F_APP: {
            // frame: original APP term (t0), arg (t1), APP loc (t2).
            // whnf is fun in WHNF — dispatch.
            Term app_orig = f.t0;
            Term arg = f.t1;
            u64 app_loc = (u64)f.t2;

            // APP ⊳ LAM: beta.  Write arg into LAM's var slot, re-enter body.
            if (term_tag(whnf) == TAG_LAM) {
                u64 lam_loc = term_val(whnf);
                // DETACH handling: if arg is UOP_DETACH TOP and resolves to
                // a TEN, use the forced TEN so downstream var reads get a
                // materialized value.
                if (term_tag(arg) == TAG_TOP && term_ext(arg) == UOP_DETACH) {
                    Term forced = thvm_force_tensor_term(ctx, arg);
                    if (term_tag(forced) == TAG_TEN) arg = forced;
                }
                heap_set(ctx, lam_loc + 0, arg);
                heap_set(ctx, app_loc + 0, term_era());
                heap_set(ctx, app_loc + 1, term_era());
                WNF_FIRED();
                next = heap_read(ctx, lam_loc + 1);
                goto enter;
            }

            // APP ⊳ TEN: discard tensor, return arg (sequencing).
            if (term_tag(whnf) == TAG_TEN) {
                tensor_release(ctx, (u32)term_val(whnf));
                WNF_FIRED();
                next = arg;
                goto enter;
            }

            // APP ⊳ NUM: numeric sequencing terminal (grad bundles etc.).
            if (term_tag(whnf) == TAG_NUM) {
                WNF_FIRED();
                next = arg;
                goto enter;
            }

            // APP ⊳ BRI: (θx.body arg) → body[x ← arg].  Beta on bridge
            // binder; same shape as LAM but no APP-slot clearing (matches
            // legacy combinator behaviour).
            if (term_tag(whnf) == TAG_BRI) {
                u64 bri_loc = term_val(whnf);
                if (term_tag(arg) == TAG_TOP && term_ext(arg) == UOP_DETACH) {
                    Term forced = thvm_force_tensor_term(ctx, arg);
                    if (term_tag(forced) == TAG_TEN) arg = forced;
                }
                heap_set(ctx, bri_loc + 0, arg);
                WNF_FIRED();
                next = heap_read(ctx, bri_loc + 1);
                goto enter;
            }

            // APP ⊳ ERA: erasure propagates.  Discard arg via explicit ERA.
            if (term_tag(whnf) == TAG_ERA) {
                thvm_spawn_detached_era(ctx, arg);
                WNF_FIRED();
                whnf = term_era();
                continue;
            }

            // APP ⊳ SUP: distribute.
            //   (&L{f0,f1} arg) → !&L{a0,a1}=arg; &L{(f0 a0), (f1 a1)}
            if (term_tag(whnf) == TAG_SUP) {
                u32 lab = term_ext(whnf);
                u64 sup_loc = term_val(whnf);
                Term f0 = heap_read(ctx, sup_loc + 0);
                Term f1 = heap_read(ctx, sup_loc + 1);
                u64 dup_loc = heap_alloc(ctx, 1);
                heap_set(ctx, dup_loc, arg);
                Term arg0 = term_new(TAG_DP0, lab, dup_loc);
                Term arg1 = term_new(TAG_DP1, lab, dup_loc);
                WNF_FIRED();
                next = thvm_sup(ctx, lab,
                    thvm_app(ctx, f0, arg0),
                    thvm_app(ctx, f1, arg1));
                heap_set(ctx, app_loc + 0, term_era());
                heap_set(ctx, app_loc + 1, term_era());
                goto enter;
            }

            // APP ⊳ USP: same as SUP but preserves unordered tag.  UDP
            // consumers share the single dup slot.
            if (term_tag(whnf) == TAG_USP) {
                u32 lab = term_ext(whnf);
                u64 usp_loc = term_val(whnf);
                Term f0 = heap_read(ctx, usp_loc + 0);
                Term f1 = heap_read(ctx, usp_loc + 1);
                u64 dup_loc = heap_alloc(ctx, 1);
                heap_set(ctx, dup_loc, arg);
                WNF_FIRED();
                next = thvm_usp(ctx, lab,
                    thvm_app(ctx, f0, term_new(TAG_UDP, lab, dup_loc)),
                    thvm_app(ctx, f1, term_new(TAG_UDP, lab, dup_loc)));
                heap_set(ctx, app_loc + 0, term_era());
                heap_set(ctx, app_loc + 1, term_era());
                goto enter;
            }

            // APP ⊳ MAT: pattern-match combinator.  Need arg in WHNF
            // before we can dispatch on its shape (SUP/USP/CTR/NUM/ERA).
            // Push a second-phase frame and descend into arg.
            if (term_tag(whnf) == TAG_MAT) {
                WnfFrame mf = {
                    .kind = WNF_F_APP_MAT, .flags = 0,
                    .t0 = app_orig, .t1 = whnf, .t2 = (Term)app_loc, .t3 = 0
                };
                wnf_stack_push(mf);
                next = arg;
                goto enter;
            }

            // Unhandled whnf tag (REF, NUM, …): rebuild original APP with
            // updated fun and return as-is (stuck).
            heap_set(ctx, app_loc + 0, whnf);
            whnf = app_orig;
            continue;
        }

        case WNF_F_APP_MAT: {
            // frame: app_orig (t0), fun=MAT (t1), app_loc (t2).
            // whnf is arg reduced to WHNF — dispatch on its shape.
            Term app_orig = f.t0;
            Term fun      = f.t1;
            u64 app_loc   = (u64)f.t2;
            u64 mat_loc   = term_val(fun);
            u32 match_tag = term_ext(fun);
            u8  atag      = term_tag(whnf);

            // APP-MAT-SUP: distribute match through superposition.
            if (atag == TAG_SUP) {
                u32 lab = term_ext(whnf);
                u64 sloc = term_val(whnf);
                Term a = heap_read(ctx, sloc + 0);
                Term b = heap_read(ctx, sloc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, fun);
                WNF_FIRED();
                next = thvm_sup(ctx, lab,
                    thvm_app(ctx, term_new(TAG_DP0, lab, dup), a),
                    thvm_app(ctx, term_new(TAG_DP1, lab, dup), b));
                goto enter;
            }

            // APP-MAT-USP: same as APP-MAT-SUP for unordered SUP.
            if (atag == TAG_USP) {
                u32 lab = term_ext(whnf);
                u64 sloc = term_val(whnf);
                Term a = heap_read(ctx, sloc + 0);
                Term b = heap_read(ctx, sloc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, fun);
                WNF_FIRED();
                next = thvm_usp(ctx, lab,
                    thvm_app(ctx, term_new(TAG_UDP, lab, dup), a),
                    thvm_app(ctx, term_new(TAG_UDP, lab, dup), b));
                goto enter;
            }

            // APP-MAT-CTR: constructor match.  Bind handler or fallback
            // depending on whether tags match.
            if (atag == TAG_CTR) {
                u32 ctr_tag = term_ext(whnf);
                WNF_FIRED();
                if (match_tag == ctr_tag) {
                    u64 ctr_loc = term_val(whnf);
                    Term r = heap_read(ctx, mat_loc + 0);
                    for (u32 i = 0; i < ctr_tag; i++) {
                        r = thvm_app(ctx, r, heap_read(ctx, ctr_loc + i));
                    }
                    next = r;
                } else {
                    next = thvm_app(ctx, heap_read(ctx, mat_loc + 1), whnf);
                }
                goto enter;
            }

            // APP-MAT-NUM: numeric match.
            if (atag == TAG_NUM) {
                u32 num_val = term_as_u32(whnf);
                WNF_FIRED();
                if (match_tag == num_val) {
                    next = heap_read(ctx, mat_loc + 0);  // handler
                } else {
                    next = thvm_app(ctx, heap_read(ctx, mat_loc + 1), whnf);
                }
                goto enter;
            }

            // APP-MAT-ERA: erased arg → ERA.
            if (atag == TAG_ERA) {
                whnf = term_era();
                continue;
            }

            // Stuck: arg not in matchable form — rebuild APP(MAT, arg).
            heap_set(ctx, app_loc + 0, fun);
            heap_set(ctx, app_loc + 1, whnf);
            whnf = app_orig;
            continue;
        }

        case WNF_F_AND_OR: {
            // whnf = a reduced; frame t0=orig, t2=loc, flags bit 0 = is_or.
            Term orig = f.t0;
            u64 loc = (u64)f.t2;
            int is_or = f.flags & 1;
            u8 at = term_tag(whnf);

            // Distribute through SUP.
            if (at == TAG_SUP) {
                u32 lab = term_ext(whnf);
                u64 sloc = term_val(whnf);
                Term a0 = heap_read(ctx, sloc + 0);
                Term a1 = heap_read(ctx, sloc + 1);
                Term b = heap_read(ctx, loc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, b);
                WNF_FIRED();
                Term (*op)(TinyHVM *, Term, Term) = is_or ? thvm_or : thvm_and;
                next = thvm_sup(ctx, lab,
                    op(ctx, a0, term_new(TAG_DP0, lab, dup)),
                    op(ctx, a1, term_new(TAG_DP1, lab, dup)));
                goto enter;
            }
            if (at == TAG_USP) {
                u32 lab = term_ext(whnf);
                u64 sloc = term_val(whnf);
                Term a0 = heap_read(ctx, sloc + 0);
                Term a1 = heap_read(ctx, sloc + 1);
                Term b = heap_read(ctx, loc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, b);
                WNF_FIRED();
                Term (*op)(TinyHVM *, Term, Term) = is_or ? thvm_or : thvm_and;
                next = thvm_usp(ctx, lab,
                    op(ctx, a0, term_new(TAG_UDP, lab, dup)),
                    op(ctx, a1, term_new(TAG_UDP, lab, dup)));
                goto enter;
            }
            // ERA → ERA.
            if (at == TAG_ERA) { whnf = term_era(); WNF_FIRED(); continue; }
            // NUM: short-circuit.
            if (at == TAG_NUM) {
                u32 av = term_as_u32(whnf);
                WNF_FIRED();
                if (is_or) {
                    if (av == 0) { next = heap_read(ctx, loc + 1); goto enter; }
                    whnf = term_num_u32(1);
                    continue;
                } else {
                    if (av == 0) { whnf = term_num_u32(0); continue; }
                    next = heap_read(ctx, loc + 1);
                    goto enter;
                }
            }
            // Stuck: rebuild AND/OR with reduced left.
            heap_set(ctx, loc + 0, whnf);
            whnf = orig;
            continue;
        }

        case WNF_F_EQL_X: {
            // whnf = x reduced; frame t0=eql_orig, t2=eql_loc.
            Term eql_orig = f.t0;
            u64 loc = (u64)f.t2;
            u8 xt = term_tag(whnf);

            // EQL-SUP-L: distribute, cloning b.
            if (xt == TAG_SUP) {
                u32 lab = term_ext(whnf);
                u64 sloc = term_val(whnf);
                Term a0 = heap_read(ctx, sloc + 0);
                Term a1 = heap_read(ctx, sloc + 1);
                Term b = heap_read(ctx, loc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, b);
                WNF_FIRED();
                next = thvm_sup(ctx, lab,
                    thvm_eql(ctx, a0, term_new(TAG_DP0, lab, dup)),
                    thvm_eql(ctx, a1, term_new(TAG_DP1, lab, dup)));
                goto enter;
            }
            // EQL-USP-L.
            if (xt == TAG_USP) {
                u32 lab = term_ext(whnf);
                u64 sloc = term_val(whnf);
                Term a0 = heap_read(ctx, sloc + 0);
                Term a1 = heap_read(ctx, sloc + 1);
                Term b = heap_read(ctx, loc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, b);
                WNF_FIRED();
                next = thvm_usp(ctx, lab,
                    thvm_eql(ctx, a0, term_new(TAG_UDP, lab, dup)),
                    thvm_eql(ctx, a1, term_new(TAG_UDP, lab, dup)));
                goto enter;
            }
            // EQL-ERA-L.
            if (xt == TAG_ERA) { whnf = term_era(); WNF_FIRED(); continue; }
            // EQL-ANY-L.
            if (xt == TAG_ANY) { whnf = term_num_u32(1); WNF_FIRED(); continue; }
            // EQL-INC-L.
            if (xt == TAG_INC) {
                Term inner_a = heap_read(ctx, term_val(whnf));
                Term b = heap_read(ctx, loc + 1);
                WNF_FIRED();
                next = thvm_inc(ctx, thvm_eql(ctx, inner_a, b));
                goto enter;
            }

            // Otherwise: promote to phase 2.  Push Y frame with x=whnf,
            // descend into y.
            Term y = heap_read(ctx, loc + 1);
            WnfFrame yf = {
                .kind = WNF_F_EQL_Y, .flags = 0,
                .t0 = eql_orig, .t1 = whnf, .t2 = (Term)loc, .t3 = 0
            };
            wnf_stack_push(yf);
            next = y;
            goto enter;
        }

        case WNF_F_EQL_Y: {
            // whnf = y reduced; frame t0=eql_orig, t1=x atom, t2=eql_loc.
            Term eql_orig = f.t0;
            Term x        = f.t1;
            u64 loc       = (u64)f.t2;
            u8 yt = term_tag(whnf);

            // EQL-SUP-R: distribute — x is atom, no DUP needed.
            if (yt == TAG_SUP) {
                u32 lab = term_ext(whnf);
                u64 sloc = term_val(whnf);
                Term b0 = heap_read(ctx, sloc + 0);
                Term b1 = heap_read(ctx, sloc + 1);
                WNF_FIRED();
                next = thvm_sup(ctx, lab,
                    thvm_eql(ctx, x, b0),
                    thvm_eql(ctx, x, b1));
                goto enter;
            }
            // EQL-USP-R.
            if (yt == TAG_USP) {
                u32 lab = term_ext(whnf);
                u64 sloc = term_val(whnf);
                Term b0 = heap_read(ctx, sloc + 0);
                Term b1 = heap_read(ctx, sloc + 1);
                WNF_FIRED();
                next = thvm_usp(ctx, lab,
                    thvm_eql(ctx, x, b0),
                    thvm_eql(ctx, x, b1));
                goto enter;
            }
            if (yt == TAG_ERA) { whnf = term_era(); WNF_FIRED(); continue; }
            if (yt == TAG_ANY) { whnf = term_num_u32(1); WNF_FIRED(); continue; }
            // EQL-INC-R.
            if (yt == TAG_INC) {
                Term inner_b = heap_read(ctx, term_val(whnf));
                WNF_FIRED();
                next = thvm_inc(ctx, thvm_eql(ctx, x, inner_b));
                goto enter;
            }
            // EQL-NUM.
            if (term_tag(x) == TAG_NUM && yt == TAG_NUM) {
                WNF_FIRED();
                whnf = term_num_u32(term_as_u32(x) == term_as_u32(whnf) ? 1 : 0);
                continue;
            }
            // EQL-LAM: fresh-var compare bodies.
            if (term_tag(x) == TAG_LAM && yt == TAG_LAM) {
                u64 a_loc = term_val(x);
                u64 b_loc = term_val(whnf);
                Term a_body = heap_read(ctx, a_loc + 1);
                Term b_body = heap_read(ctx, b_loc + 1);
                u32 fresh = ctx->next_sup_label++;
                Term fresh_var = term_num_u32(fresh);
                heap_set(ctx, a_loc, fresh_var);
                heap_set(ctx, b_loc, fresh_var);
                WNF_FIRED();
                next = thvm_eql(ctx, a_body, b_body);
                goto enter;
            }
            // Different types (or stuck compound) — rebuild EQL and return.
            // Legacy rule emits 0 for genuinely different head tags; we
            // replicate that for all atomic / value-like forms.
            u8 xt = term_tag(x);
            int x_val = (xt == TAG_NUM || xt == TAG_TEN ||
                         xt == TAG_CTR || xt == TAG_LAM ||
                         xt == TAG_NUM);
            int y_val = (yt == TAG_NUM || yt == TAG_TEN ||
                         yt == TAG_CTR || yt == TAG_LAM);
            (void)eql_orig;
            if (x_val && y_val) {
                WNF_FIRED();
                whnf = term_num_u32(0);
                continue;
            }
            // Otherwise rebuild EQL with reduced operands.
            heap_set(ctx, loc + 0, x);
            heap_set(ctx, loc + 1, whnf);
            whnf = eql_orig;
            continue;
        }

        case WNF_F_DSU: {
            // frame: DSU orig (t0), DSU loc (t2); whnf = label in WHNF.
            Term dsu_orig = f.t0;
            u64 loc = (u64)f.t2;
            if (term_tag(whnf) != TAG_NUM) {
                // Label not ready — write back and return stuck.
                heap_set(ctx, loc + 0, whnf);
                whnf = dsu_orig;
                continue;
            }
            u32 label = term_as_u32(whnf);
            Term a = heap_read(ctx, loc + 1);
            Term b = heap_read(ctx, loc + 2);
            WNF_FIRED();
            next = thvm_sup(ctx, label, a, b);
            goto enter;
        }

        case WNF_F_DDU: {
            // frame: DDU orig (t0), DDU loc (t2); whnf = label in WHNF.
            Term ddu_orig = f.t0;
            u64 loc = (u64)f.t2;
            if (term_tag(whnf) != TAG_NUM) {
                heap_set(ctx, loc + 0, whnf);
                whnf = ddu_orig;
                continue;
            }
            u32 label = term_as_u32(whnf);
            Term val = heap_read(ctx, loc + 1);
            Term bod = heap_read(ctx, loc + 2);
            Term dp0, dp1;
            thvm_dup(ctx, label, val, &dp0, &dp1);
            WNF_FIRED();
            next = thvm_app(ctx, thvm_app(ctx, bod, dp0), dp1);
            goto enter;
        }

        case WNF_F_UDP: {
            // frame: UDP orig (t0), UDP loc (t2); whnf = body in WHNF.
            Term udp_orig = f.t0;
            u64 udp_loc = (u64)f.t2;
            u32 udp_label = term_ext(udp_orig);
            u8 vtag = term_tag(whnf);

            // UDP ⊳ USP (same label): consume one branch, keep producing from the
            // other; the UDP cell slides to the remainder.
            if (vtag == TAG_USP && term_ext(whnf) == udp_label) {
                u64 usp_loc = term_val(whnf);
                Term a = heap_read(ctx, usp_loc + 0);
                Term b = heap_read(ctx, usp_loc + 1);
                heap_set(ctx, udp_loc, b);
                WNF_FIRED();
                next = a;
                goto enter;
            }
            // UDP ⊳ USP (different label): commutation preserving USP.
            if (vtag == TAG_USP) {
                u32 usp_label = term_ext(whnf);
                u64 usp_loc = term_val(whnf);
                Term a = heap_read(ctx, usp_loc + 0);
                Term b = heap_read(ctx, usp_loc + 1);
                u64 du_a = heap_alloc(ctx, 1); heap_set(ctx, du_a, a);
                u64 du_b = heap_alloc(ctx, 1); heap_set(ctx, du_b, b);
                WNF_FIRED();
                next = thvm_usp(ctx, usp_label,
                    term_new(TAG_UDP, udp_label, du_a),
                    term_new(TAG_UDP, udp_label, du_b));
                goto enter;
            }
            // UDP ⊳ SUP: distribute UDP over ordered SUP branches.
            if (vtag == TAG_SUP) {
                u32 sup_label = term_ext(whnf);
                u64 sup_loc = term_val(whnf);
                Term a = heap_read(ctx, sup_loc + 0);
                Term b = heap_read(ctx, sup_loc + 1);
                u64 du_a = heap_alloc(ctx, 1); heap_set(ctx, du_a, a);
                u64 du_b = heap_alloc(ctx, 1); heap_set(ctx, du_b, b);
                WNF_FIRED();
                next = thvm_sup(ctx, sup_label,
                    term_new(TAG_UDP, udp_label, du_a),
                    term_new(TAG_UDP, udp_label, du_b));
                goto enter;
            }
            // UDP ⊳ atoms: UDP vanishes.
            if (vtag == TAG_NUM || vtag == TAG_ERA ||
                vtag == TAG_TEN || vtag == TAG_ANY) {
                // whnf is already the value; fall through.
                continue;
            }
            // UDP ⊳ LAM: commutation — wrap body in UDP, SUBST var with UDP of new var.
            if (vtag == TAG_LAM) {
                u64 lam_loc = term_val(whnf);
                Term body = heap_read(ctx, lam_loc + 1);
                u64 bdup = heap_alloc(ctx, 1);
                heap_set(ctx, bdup, body);
                Term var0;
                Term lam_new = thvm_lam(ctx, &var0, term_new(TAG_UDP, udp_label, bdup));
                u64 vdup = heap_alloc(ctx, 1);
                heap_set(ctx, vdup, var0);
                heap_set(ctx, lam_loc, term_new(TAG_UDP, udp_label, vdup));
                WNF_FIRED();
                whnf = lam_new;
                continue;
            }
            // Stuck: rebuild UDP with body at cell and return as-is.
            heap_set(ctx, udp_loc, whnf);
            whnf = udp_orig;
            continue;
        }

        default:
            // Unknown frame kind — panic-degrade.
            goto apply_done;
        }
    }
apply_done:
    // Budget-bound exit: if the budget fired before reduction hit WHNF,
    // drain remaining frames into a resumable Term.
    if (g_wnf_budget_hit && g_wnf_stack_pos > base) {
        whnf = wnf_drain_stack(ctx, whnf, base);
    }
    // Outermost boundary: if the GRAD reduced to ERA, synthesize a
    // zeros tensor of target.shape so callers get a materializable
    // value.  Inner ERAs during recursion participate in peepholes;
    // only the outer boundary materializes.
    if (outermost_is_grad && term_tag(whnf) == TAG_ERA) {
        Shape osh = outermost_tgt_shape;
        Shape one_shape = {.rank = osh.rank};
        for (u32 i = 0; i < osh.rank; i++) one_shape.dims[i] = 1;
        if (one_shape.rank == 0) { one_shape.rank = 1; one_shape.dims[0] = 1; }
        f32 z = 0.0f;
        Term scalar = thvm_tensor(ctx, &z, one_shape);
        int is_sc = (osh.rank == 0) ||
                    (osh.rank == 1 && osh.dims[0] == 1);
        whnf = is_sc ? scalar : thvm_expand(ctx, scalar, osh);
    }
    g_wnf_budget       = saved_budget;
    g_wnf_budget_fired = saved_budget_fired;
    g_wnf_budget_hit   = saved_budget_hit;
    return whnf;
}
}

static void wnf_stack_reset(void) {
    if (g_wnf_stack_buf) {
        free(g_wnf_stack_buf);
        g_wnf_stack_buf = NULL;
        g_wnf_stack_cap = 0;
        g_wnf_stack_pos = 0;
    }
    (void)wnf_stack_reset;
}
