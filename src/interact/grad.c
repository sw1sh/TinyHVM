// src/interact/grad.c - UOP_GRAD (VJP) / UOP_GRAD_FWD (JVP) interaction rule.
// Included inline from src/interact/_.c under case TAG_TOP.
//
// Reverse mode (UOP_GRAD):  GRAD(y, target) → tensor of target.shape
//     = ∂(Σ y)/∂target  (VJP with implicit ones cotangent on y).
// Forward mode (UOP_GRAD_FWD):  GRAD_FWD(y, target) → tensor of y.shape
//     = (∂y/∂target) · ones(target.shape)  (JVP with implicit ones seed).
//
// The two modes share all elementwise/diagonal-Jacobian rules (ADD, SUB, NEG,
// MUL, DIV, EXP, LOG, SQRT, RELU, MAX, CMP, WHERE).  They branch on polarity
// (`is_fwd`) for leaf shape and the non-diagonal ops: MM, SUM, EXPAND,
// RESHAPE, PERMUTE, SHRINK, PAD.

if (uop == UOP_GRAD || uop == UOP_GRAD_FWD) {
    const int is_fwd = (uop == UOP_GRAD_FWD);
    Term y   = heap_read(ctx, loc + 0);
    Term tgt = heap_read(ctx, loc + 1);
    if (term_is_sub(y))   y   = term_strip_sub(y);
    if (term_is_sub(tgt)) tgt = term_strip_sub(tgt);

    // In forward mode every recursive GRAD stays in forward mode too.
    #define GRAD_REC(_y, _t) (is_fwd ? thvm_grad_fwd(ctx, (_y), (_t)) \
                                      : thvm_grad(ctx, (_y), (_t)))

    // Helper: build a scalar-valued tensor of tsh shape via
    // rank-matched ones-shape + EXPAND.  Handles the recurring
    // pattern across all leaf/CMP emissions.
    #define GRAD_SCALAR_TEN(_val, _tsh) ({                 \
        Shape _one = {.rank = (_tsh).rank};                 \
        for (u32 _i = 0; _i < (_tsh).rank; _i++) _one.dims[_i] = 1; \
        if (_one.rank == 0) { _one.rank = 1; _one.dims[0] = 1; } \
        f32 _v = (_val);                                    \
        Term _t = thvm_tensor(ctx, &_v, _one);              \
        int _scalar = ((_tsh).rank == 0) ||                 \
                      ((_tsh).rank == 1 && (_tsh).dims[0] == 1); \
        _scalar ? _t : thvm_expand(ctx, _t, (_tsh));        \
    })
    // Emit ones tensor of the given shape (rank-matched ones-shape TEN
    // then EXPAND).  Used by view-op direct-emit bwd.
    // Read a term's shape (from TEN meta or TOP shape-table).
    // Stores into _out (Shape lvalue); leaves it as SHAPE(1) if none.
    #define GRAD_TERM_SHAPE(_term, _out) do {               \
        Term _tm = (_term);                                  \
        if (term_tag(_tm) == TAG_TEN) {                      \
            u32 _id = (u32)term_val(_tm);                    \
            if (_id < ctx->tensor_count)                     \
                (_out) = ctx->tensors[_id].view.shape;       \
        } else if (term_tag(_tm) == TAG_TOP) {               \
            const View *_av = st_get(term_val(_tm));         \
            if (_av) (_out) = _av->shape;                    \
        }                                                    \
    } while (0)
    #define GRAD_ONES_OF(_shape) ({                         \
        Shape _os = {.rank = (_shape).rank};                 \
        for (u32 _i = 0; _i < (_shape).rank; _i++) _os.dims[_i] = 1; \
        if (_os.rank == 0) { _os.rank = 1; _os.dims[0] = 1; } \
        f32 _v = 1.0f;                                       \
        Term _t = thvm_tensor(ctx, &_v, _os);                \
        ((_shape).rank == 0) ? _t : thvm_expand(ctx, _t, (_shape)); \
    })
    // Leaf output shape: VJP returns target.shape, JVP returns y.shape.
    // Value: 1 if y-leaf matches target (then y.shape == target.shape), else 0.
    // NUM constant: always 0 of output shape.
    if (term_tag(y) == TAG_NUM && term_tag(tgt) == TAG_TEN) {
        Shape osh = SHAPE(1);
        if (is_fwd) {
            // y is NUM scalar; result shape = [1].
            osh = SHAPE(1);
        } else {
            u32 ttid = (u32)term_val(tgt);
            if (ttid < ctx->tensor_count) osh = ctx->tensors[ttid].view.shape;
        }
        Term out = GRAD_SCALAR_TEN(0.0f, osh);
        ctx->itrs++; return out;
    }
    if (term_tag(y) == TAG_TEN && term_tag(tgt) == TAG_TEN) {
        u32 ytid = (u32)term_val(y);
        u32 ttid = (u32)term_val(tgt);
        Shape osh = SHAPE(1);
        if (is_fwd) {
            if (ytid < ctx->tensor_count) osh = ctx->tensors[ytid].view.shape;
        } else {
            if (ttid < ctx->tensor_count) osh = ctx->tensors[ttid].view.shape;
        }
        // Forward mode feeds this tangent directly into MM/etc., so emit a
        // dense materialized tensor instead of a stride-0 EXPAND view (which
        // can lose its broadcast semantics across the MM decomposition).
        if (is_fwd && osh.rank > 0) {
            u32 n = 1; for (u32 i = 0; i < osh.rank; i++) n *= osh.dims[i];
            f32 val = (ytid == ttid) ? 1.0f : 0.0f;
            f32 *buf = (f32 *)malloc(n * sizeof(f32));
            for (u32 i = 0; i < n; i++) buf[i] = val;
            Term out = thvm_tensor(ctx, buf, osh);
            free(buf);
            ctx->itrs++; return out;
        }
        Term out = GRAD_SCALAR_TEN(ytid == ttid ? 1.0f : 0.0f, osh);
        ctx->itrs++;
        return out;
    }
    // Reachability: does any leaf TEN descended from `y` have id == ttid?
    // Bounded DFS over TOP children (shape meta and reduction operands
    // are skipped — reachability checks compute operands only).  DP/VAR
    // chains are resolved one hop.  Used to prune dead branches in ADD,
    // SUB, MUL, DIV, EXP, LOG etc. — no reason to emit zero-seeds or
    // recurse into operands that don't contain the target.
    #define GRAD_REACHES_TARGET(_y, _ttid) ({ \
        u32 _tt = (_ttid); int _found = 0; \
        /* stack: pairs of (term, depth) — iterative DFS */ \
        Term _stk[64]; u32 _sp = 0; _stk[_sp++] = (_y); \
        while (_sp > 0 && !_found) { \
            Term _t = _stk[--_sp]; \
            /* Resolve DP/VAR one hop */ \
            for (int _it = 0; _it < 8; _it++) { \
                u8 _tg = term_tag(_t); \
                if (_tg == TAG_DP0 || _tg == TAG_DP1 || _tg == TAG_VAR) { \
                    u64 _l = term_val(_t); \
                    if (_l == 0 || _l >= ctx->heap_pos) break; \
                    Term _n = heap_read(ctx, _l); \
                    if (term_is_sub(_n)) _n = term_strip_sub(_n); \
                    if (_n == _t) break; _t = _n; continue; \
                } break; \
            } \
            u8 _tg = term_tag(_t); \
            if (_tg == TAG_TEN) { \
                if ((u32)term_val(_t) == _tt) { _found = 1; break; } \
                continue; \
            } \
            if (_tg == TAG_TOP) { \
                u32 _u = term_ext(_t); u64 _l = term_val(_t); \
                /* Compute-op arities: unary=1, binary=2, ternary=3. */ \
                u32 _ar = 2; \
                if (_u == UOP_WHERE || _u == UOP_IFZ) _ar = 3; \
                else if (_u == UOP_NEG || _u == UOP_EXP || _u == UOP_LOG || \
                         _u == UOP_RELU || _u == UOP_SQRT || _u == UOP_CAST || \
                         _u == UOP_DETACH) _ar = 1; \
                /* For view ops and SUM/RMAX, only first operand is data; \
                   second is shape/axes meta — skip it. */ \
                u32 _data_ar = _ar; \
                if (_u >= UOP_RESHAPE && _u <= UOP_PAD) _data_ar = 1; \
                else if (_u == UOP_SUM || _u == UOP_RMAX) _data_ar = 1; \
                for (u32 _i = 0; _i < _data_ar && _sp < 63; _i++) { \
                    if (_l + _i < ctx->heap_pos) _stk[_sp++] = heap_read(ctx, _l + _i); \
                } \
            } \
        } _found; \
    })

    // TOP chain-rule dispatch.  Recursively emits GRAD on sub-
    // operands; target is DUP'd to keep linearity.
    if (term_tag(y) == TAG_TOP && term_tag(tgt) == TAG_TEN) {
        u32 yuop = term_ext(y);
        u64 yloc = term_val(y);
        u32 ttid = (u32)term_val(tgt);
        Shape tsh_guard = SHAPE(1);
        if (ttid < ctx->tensor_count) tsh_guard = ctx->tensors[ttid].view.shape;
        // Pre-flight reachability on the whole y: if target doesn't appear
        // anywhere, the entire GRAD is zero (or identity for leaf-match
        // already handled above).  Saves every rule the work of recursing
        // into a dead subtree.
        if (!GRAD_REACHES_TARGET(y, ttid)) {
            ctx->itrs++;
            if (is_fwd) {
                Shape ysh_out = SHAPE(1);
                const View *yv = st_get(yloc); if (yv) ysh_out = yv->shape;
                return GRAD_SCALAR_TEN(0.0f, ysh_out);
            }
            return GRAD_SCALAR_TEN(0.0f, tsh_guard);
        }
        if (yuop == UOP_ADD || yuop == UOP_SUB) {
            Term a = heap_read(ctx, yloc + 0);
            Term b = heap_read(ctx, yloc + 1);
            int a_reaches = GRAD_REACHES_TARGET(a, ttid);
            int b_reaches = GRAD_REACHES_TARGET(b, ttid);
            Term da = a_reaches ? GRAD_REC(a, tgt) : (Term)0;
            Term db = b_reaches ? GRAD_REC(b, tgt) : (Term)0;
            ctx->itrs++;
            if (a_reaches && b_reaches) return thvm_op_raw(ctx, yuop, da, db);
            if (a_reaches) return yuop == UOP_SUB ? da : da;  // a + 0 = a; a - 0 = a
            if (b_reaches) return yuop == UOP_SUB
                ? thvm_op_raw(ctx, UOP_NEG, db, term_era())  // 0 - b = -b
                : db;                                         // 0 + b = b
            // Unreachable — pre-flight caught this.  Keep a safe fallback.
            return GRAD_SCALAR_TEN(0.0f, tsh_guard);
        }
        // NEG: d(-a)/dt = -(da).
        if (yuop == UOP_NEG) {
            Term a = heap_read(ctx, yloc + 0);
            Term da = GRAD_REC( a, tgt);
            Term out = thvm_op_raw(ctx, UOP_NEG, da, term_era());
            ctx->itrs++; return out;
        }
        // EXP: d(exp a)/dt = exp(a) * da.  a used twice.
        if (yuop == UOP_EXP) {
            Term a = heap_read(ctx, yloc + 0);
            Term a0, a1;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            Term da = GRAD_REC( a0, tgt);
            Term exp_a = thvm_op_raw(ctx, UOP_EXP, a1, term_era());
            Term out = thvm_op_raw(ctx, UOP_MUL, da, exp_a);
            ctx->itrs++; return out;
        }
        // LOG: d(log a)/dt = (1/a) * da.  a used twice.
        if (yuop == UOP_LOG) {
            Term a = heap_read(ctx, yloc + 0);
            Term a0, a1;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            Term da = GRAD_REC( a0, tgt);
            Term out = thvm_op_raw(ctx, UOP_DIV, da, a1);
            ctx->itrs++; return out;
        }
        // SQRT: d(sqrt a)/dt = da / (2*sqrt(a)).  a used twice.
        if (yuop == UOP_SQRT) {
            Term a = heap_read(ctx, yloc + 0);
            Term a0, a1;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            Term da = GRAD_REC( a0, tgt);
            Term sq = thvm_op_raw(ctx, UOP_SQRT, a1, term_era());
            Term two = term_num_f32(2.0f);
            Term den = thvm_op_raw(ctx, UOP_MUL, two, sq);
            Term out = thvm_op_raw(ctx, UOP_DIV, da, den);
            ctx->itrs++; return out;
        }
        // RELU: d(relu a)/dt = (a>0) * da.  a used twice.
        if (yuop == UOP_RELU) {
            Term a = heap_read(ctx, yloc + 0);
            Term a0, a1;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            Term da = GRAD_REC( a0, tgt);
            Term zero = term_num_f32(0.0f);
            Term mask = thvm_op_raw(ctx, UOP_CMP, a1, zero);
            Term out = thvm_op_raw(ctx, UOP_MUL, da, mask);
            ctx->itrs++; return out;
        }
        // DIV (quotient): d(a/b)/dt = (da*b - a*db) / b².
        // a used 2x, b used 4x (fwd, 2 cross-terms, denom b²),
        // target used 2x.
        if (yuop == UOP_DIV) {
            Term a = heap_read(ctx, yloc + 0);
            Term b = heap_read(ctx, yloc + 1);
            Term a0, a1, b0, b1, b2, b3, t0, t1;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            thvm_dup(ctx, thvm_fresh_label(ctx), b, &b0, &b1);
            Term b_lo;
            thvm_dup(ctx, thvm_fresh_label(ctx), b1, &b_lo, &b2);
            thvm_dup(ctx, thvm_fresh_label(ctx), b_lo, &b1, &b3);
            thvm_dup(ctx, thvm_fresh_label(ctx), tgt, &t0, &t1);
            Term da = GRAD_REC( a0, t0);
            Term db = GRAD_REC( b0, t1);
            Term l  = thvm_op_raw(ctx, UOP_MUL, da, b1);
            Term r  = thvm_op_raw(ctx, UOP_MUL, a1, db);
            Term num = thvm_op_raw(ctx, UOP_SUB, l, r);
            Term den = thvm_op_raw(ctx, UOP_MUL, b2, b3);
            Term out = thvm_op_raw(ctx, UOP_DIV, num, den);
            ctx->itrs++; return out;
        }
        // MAX: d(max(a,b))/dt = (a>=b)*da + (a<b)*db.
        if (yuop == UOP_MAX) {
            Term a = heap_read(ctx, yloc + 0);
            Term b = heap_read(ctx, yloc + 1);
            Term a0, a1, b0, b1, t0, t1;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            thvm_dup(ctx, thvm_fresh_label(ctx), b, &b0, &b1);
            thvm_dup(ctx, thvm_fresh_label(ctx), tgt, &t0, &t1);
            Term da = GRAD_REC( a0, t0);
            Term db = GRAD_REC( b0, t1);
            Term mask_a = thvm_op_raw(ctx, UOP_CMP, a1, b1);
            Term m0, m1;
            thvm_dup(ctx, thvm_fresh_label(ctx), mask_a, &m0, &m1);
            Term one = term_num_f32(1.0f);
            Term mask_b = thvm_op_raw(ctx, UOP_SUB, one, m1);
            Term l = thvm_op_raw(ctx, UOP_MUL, da, m0);
            Term r = thvm_op_raw(ctx, UOP_MUL, db, mask_b);
            Term out = thvm_op_raw(ctx, UOP_ADD, l, r);
            ctx->itrs++; return out;
        }
        // RESHAPE/PERMUTE: movement ops. dA = inverse_view(da).
        // Shapes/perm taken from the second operand TEN.
        // RESHAPE / PERMUTE are 1:1 data maps — y[*] corresponds bijectively
        // to a[*]. sum over y of dy/d_target = sum over a of da/d_target =
        // GRAD(a, target) unchanged.  Guard: only apply a wrap when the
        // grad's shape still matches a_shape (identity target case).
        if (yuop == UOP_RESHAPE || yuop == UOP_PERMUTE) {
            Term a     = heap_read(ctx, yloc + 0);
            Term shape = heap_read(ctx, yloc + 1);
            Shape a_shape = SHAPE(1);
            GRAD_TERM_SHAPE(a, a_shape);
            Term da = GRAD_REC( a, tgt);
            Term out;
            if (is_fwd) {
                // JVP: apply the forward movement op to the tangent.
                Shape y_shape = SHAPE(1);
                const View *yv = st_get(yloc); if (yv) y_shape = yv->shape;
                if (yuop == UOP_RESHAPE && y_shape.rank > 0) {
                    out = thvm_reshape(ctx, da, y_shape);
                } else if (yuop == UOP_PERMUTE &&
                           term_tag(shape) == TAG_TEN && a_shape.rank > 0) {
                    u32 pid = (u32)term_val(shape);
                    u32 nd = a_shape.rank;
                    u32 pf[MAX_DIM]; tensor_meta_read_u32(ctx, pid, pf, MAX_DIM);
                    out = thvm_permute(ctx, da, pf, nd);
                } else {
                    out = da;
                }
                ctx->itrs++; return out;
            }
            u32 ttid = (u32)term_val(tgt);
            Shape tsh = (ttid < ctx->tensor_count)
                ? ctx->tensors[ttid].view.shape : SHAPE(1);
            u32 a_numel = 1, t_numel = 1;
            for (u32 i = 0; i < a_shape.rank; i++) a_numel *= a_shape.dims[i];
            for (u32 i = 0; i < tsh.rank; i++) t_numel *= tsh.dims[i];
            if (a_numel == t_numel && yuop == UOP_RESHAPE && a_shape.rank != 0) {
                out = thvm_reshape(ctx, da, a_shape);
            } else if (a_numel == t_numel && yuop == UOP_PERMUTE &&
                       term_tag(shape) == TAG_TEN && a_shape.rank > 0) {
                u32 pid = (u32)term_val(shape);
                u32 nd = a_shape.rank;
                u32 pf[MAX_DIM]; tensor_meta_read_u32(ctx, pid, pf, MAX_DIM);
                u32 inv[MAX_DIM]; for (u32 j = 0; j < nd; j++) inv[pf[j]] = j;
                out = thvm_permute(ctx, da, inv, nd);
            } else {
                out = da;  // pass-through when target shape differs from a's
            }
            ctx->itrs++; return out;
        }
        // SUM.  VJP: expand da to a_shape (broadcast cotangent back).
        //         JVP: apply the same SUM axes to the tangent.
        if (yuop == UOP_SUM) {
            Term a    = heap_read(ctx, yloc + 0);
            Term axes = heap_read(ctx, yloc + 1);
            Shape a_shape = SHAPE(1);
            GRAD_TERM_SHAPE(a, a_shape);
            Term da = GRAD_REC( a, tgt);
            Term out;
            if (is_fwd) {
                out = thvm_op_raw(ctx, UOP_SUM, da, axes);
            } else {
                out = (a_shape.rank != 0) ? thvm_expand(ctx, da, a_shape) : da;
            }
            ctx->itrs++; return out;
        }
        // SHRINK/PAD: JVP applies the same op to the tangent.
        if ((yuop == UOP_SHRINK || yuop == UOP_PAD) && is_fwd) {
            Term a     = heap_read(ctx, yloc + 0);
            Term shape = heap_read(ctx, yloc + 1);
            Term da    = GRAD_REC(a, tgt);
            Term out   = thvm_op_raw(ctx, yuop, da, shape);
            ctx->itrs++; return out;
        }
        // SHRINK: dA = pad(da, complementary pairs of shrink pairs).
        if (yuop == UOP_SHRINK) {
            Term a     = heap_read(ctx, yloc + 0);
            Term shape = heap_read(ctx, yloc + 1);
            Shape a_shape = SHAPE(1);
            GRAD_TERM_SHAPE(a, a_shape);
            // SHRINK shape-preserving bwd.  For target == a (leaf
            // identity), emit PAD(ones(y_shape), pad_pairs) directly
            // so the output has target.shape with mask (=1 in kept
            // region, 0 in padded).  Recursion via thvm_grad(a,t)
            // returns target.shape (not y.shape), so we'd over-pad.
            Term out;
            if (term_tag(shape) == TAG_TEN && a_shape.rank > 0) {
                u32 sid = (u32)term_val(shape);
                u32 nd = a_shape.rank;
                u32 sf[MAX_DIM * 2]; tensor_meta_read_u32(ctx, sid, sf, MAX_DIM * 2);
                // y_shape = shrink output shape
                Shape ys = {.rank = nd};
                for (u32 j = 0; j < nd; j++) ys.dims[j] = sf[j*2+1] - sf[j*2];
                Term y_ones = GRAD_ONES_OF(ys);
                // Pad back to target shape using complementary pairs.
                u32 pp[MAX_DIM * 2];
                for (u32 j = 0; j < nd; j++) {
                    pp[j*2]   = sf[j*2];
                    pp[j*2+1] = a_shape.dims[j] - sf[j*2+1];
                }
                out = thvm_pad(ctx, y_ones, pp, nd);
            } else {
                out = GRAD_REC( a, tgt);
            }
            ctx->itrs++; return out;
        }
        // PAD: direct emit — dA = SHRINK(ones(y_shape), unpad_pairs).
        if (yuop == UOP_PAD) {
            Term a     = heap_read(ctx, yloc + 0);
            Term shape = heap_read(ctx, yloc + 1);
            Shape a_shape = SHAPE(1);
            GRAD_TERM_SHAPE(a, a_shape);
            Shape y_shape = SHAPE(1);
            const View *yv = st_get(yloc);
            if (yv) y_shape = yv->shape;
            Term out;
            if (term_tag(shape) == TAG_TEN && a_shape.rank > 0 && y_shape.rank > 0) {
                u32 pid = (u32)term_val(shape);
                u32 nd = a_shape.rank;
                u32 pf[MAX_DIM * 2]; tensor_meta_read_u32(ctx, pid, pf, MAX_DIM * 2);
                Term y_ones = GRAD_ONES_OF(y_shape);
                u32 sp[MAX_DIM * 2];
                for (u32 j = 0; j < nd; j++) {
                    sp[j*2]   = pf[j*2];
                    sp[j*2+1] = pf[j*2] + a_shape.dims[j];
                }
                out = thvm_shrink(ctx, y_ones, sp, nd);
            } else {
                out = GRAD_REC( a, tgt);
            }
            ctx->itrs++; return out;
        }
        // EXPAND: y = expand(a, y_shape). Under the ones-seed contract,
        //   d(sum y)/dt = sum_to_shape(ones(y_shape), tgt.shape)  when
        // target flows through the operand. If the operand is a leaf TEN
        // whose id doesn't match the target (projection-stripped), the
        // expand contributes nothing — emit zeros(tgt.shape).
        if (yuop == UOP_EXPAND) {
            Term a = heap_read(ctx, yloc + 0);
            Shape y_shape = SHAPE(1);
            const View *yv = st_get(yloc);
            if (yv) y_shape = yv->shape;
            if (is_fwd) {
                // JVP: expand the tangent forward to y.shape.
                Term da = GRAD_REC(a, tgt);
                Term out = (y_shape.rank != 0) ? thvm_expand(ctx, da, y_shape) : da;
                ctx->itrs++; return out;
            }
            u32 ttid = (u32)term_val(tgt);
            Shape tsh = (ttid < ctx->tensor_count)
                ? ctx->tensors[ttid].view.shape : SHAPE(1);
            // Leaf-mismatch short-circuit.
            if (term_tag(a) == TAG_TEN && term_tag(tgt) == TAG_TEN) {
                u32 a_tid = (u32)term_val(a);
                if (a_tid != ttid) {
                    ctx->itrs++; return GRAD_SCALAR_TEN(0.0f, tsh);
                }
            }
            Term y_ones = GRAD_ONES_OF(y_shape);
            Term out = (y_shape.rank != 0 && tsh.rank != 0)
                ? sum_to_shape(ctx, y_ones, y_shape, tsh) : y_ones;
            ctx->itrs++; return out;
        }
        // RMAX: dA = da * (a == expand(rmax(a))).  a used 3x.
        if (yuop == UOP_RMAX) {
            Term a    = heap_read(ctx, yloc + 0);
            Term axes = heap_read(ctx, yloc + 1);
            Shape a_shape = SHAPE(1);
            GRAD_TERM_SHAPE(a, a_shape);
            Term a0, a1, a2;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            Term a1b;
            thvm_dup(ctx, thvm_fresh_label(ctx), a1, &a1b, &a2);
            Term da = GRAD_REC( a0, tgt);
            Term rm = thvm_op_raw(ctx, UOP_RMAX, a1b, axes);
            Term rm_bc = (a_shape.rank != 0)
                ? thvm_expand(ctx, rm, a_shape) : rm;
            // mask = 1 - (rm_bc > a2)  == (a2 >= rm_bc).
            Term gt = thvm_op_raw(ctx, UOP_CMP, rm_bc, a2);
            Term one = term_num_f32(1.0f);
            Term mask = thvm_op_raw(ctx, UOP_SUB, one, gt);
            Term out = thvm_op_raw(ctx, UOP_MUL, da, mask);
            ctx->itrs++; return out;
        }
        // MM reverse-mode (scalar-loss, ones seed on y):
        //   y = a @ b   (y.shape = [M,N], a.shape = [M,K], b.shape = [K,N])
        //   gy = ones(y.shape)
        //   grad_a = gy @ b^T   shape [M,K] = a.shape
        //   grad_b = a^T @ gy   shape [K,N] = b.shape
        // GRAD contract requires result of target.shape. For a leaf target
        // matching operand a (TEN-id equal after projection strip), emit
        // grad_a; for leaf matching b, emit grad_b; else zeros(target).
        if (yuop == UOP_MM) {
            Term a = heap_read(ctx, yloc + 0);
            Term b = heap_read(ctx, yloc + 1);
            // JVP: Leibniz is exactly right — JVP(a@b) = JVP(a)@b + a@JVP(b).
            // Tangent of each operand has the operand's shape, so the MMs
            // compose shape-wise into y.shape.
            if (is_fwd) {
                // JVP(a@b) = JVP(a)@b + a@JVP(b).  Leibniz composes naturally.
                Term a0, a1, b0, b1, t0, t1;
                thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
                thvm_dup(ctx, thvm_fresh_label(ctx), b, &b0, &b1);
                thvm_dup(ctx, thvm_fresh_label(ctx), tgt, &t0, &t1);
                Term da = GRAD_REC(a0, t0);
                Term db = GRAD_REC(b0, t1);
                Term l  = thvm_op(ctx, UOP_MM, da, b1);
                Term r  = thvm_op(ctx, UOP_MM, a1, db);
                Term out = thvm_op_raw(ctx, UOP_ADD, l, r);
                ctx->itrs++; return out;
            }
            // Resolve TEN id through DP/VAR chain.
            #define MM_TID(_tm) ({ \
                Term _t = (_tm); u32 _id = ~0u; \
                for (int _i = 0; _i < 32; _i++) { \
                    u8 _tg = term_tag(_t); \
                    if (_tg == TAG_TEN) { _id = (u32)term_val(_t); break; } \
                    if (_tg == TAG_DP0 || _tg == TAG_DP1) { \
                        u64 _l = term_val(_t); \
                        if (_l == 0 || _l >= ctx->heap_pos) break; \
                        Term _n = heap_read(ctx, _l); \
                        if (term_is_sub(_n)) _n = term_strip_sub(_n); \
                        if (_n == _t) break; _t = _n; continue; \
                    } \
                    if (_tg == TAG_VAR) { \
                        u64 _l = term_val(_t); \
                        if (_l == 0 || _l >= ctx->heap_pos) break; \
                        Term _s = heap_read(ctx, _l); \
                        if (term_is_sub(_s)) break; \
                        if (_s == _t) break; _t = _s; continue; \
                    } \
                    break; \
                } _id; \
            })
            u32 a_tid = MM_TID(a), b_tid = MM_TID(b), t_tid = MM_TID(tgt);
            Shape ysh = SHAPE(1);
            const View *yv = st_get(yloc); if (yv) ysh = yv->shape;
            if (ysh.rank != 2) {
                Shape tsh = SHAPE(1); GRAD_TERM_SHAPE(tgt, tsh);
                ctx->itrs++; return GRAD_SCALAR_TEN(0.0f, tsh);
            }
            if (t_tid != ~0u && t_tid == a_tid) {
                Term gy = GRAD_ONES_OF(ysh);
                u32 ax[2] = {1,0};
                Term bT = thvm_permute(ctx, b, ax, 2);
                Term out = thvm_op(ctx, UOP_MM, gy, bT);
                ctx->itrs++; return out;
            }
            if (t_tid != ~0u && t_tid == b_tid) {
                Term gy = GRAD_ONES_OF(ysh);
                u32 ax[2] = {1,0};
                Term aT = thvm_permute(ctx, a, ax, 2);
                Term out = thvm_op(ctx, UOP_MM, aT, gy);
                ctx->itrs++; return out;
            }
            Shape tsh = SHAPE(1); GRAD_TERM_SHAPE(tgt, tsh);
            ctx->itrs++; return GRAD_SCALAR_TEN(0.0f, tsh);
            #undef MM_TID
        }
        // ASSIGN: forward-only. Gradient flows through dst only.
        if (yuop == UOP_ASSIGN) {
            Term dst = heap_read(ctx, yloc + 0);
            Term out = GRAD_REC( dst, tgt);
            ctx->itrs++; return out;
        }
        // WHERE(cond, a, b): dy/dt = where(cond, da, db).
        // cond is constant w.r.t. t; gradient distributes linearly.
        if (yuop == UOP_WHERE) {
            Term c = heap_read(ctx, yloc + 0);
            Term a = heap_read(ctx, yloc + 1);
            Term b = heap_read(ctx, yloc + 2);
            Term t0, t1;
            thvm_dup(ctx, thvm_fresh_label(ctx), tgt, &t0, &t1);
            Term da = GRAD_REC( a, t0);
            Term db = GRAD_REC( b, t1);
            Term out = thvm_where(ctx, c, da, db);
            ctx->itrs++; return out;
        }
        // CMP: non-differentiable. Zero contribution, shape-matched.
        if (yuop == UOP_CMP) {
            u32 ttid = (u32)term_val(tgt);
            Shape tsh = (ttid < ctx->tensor_count)
                ? ctx->tensors[ttid].view.shape : SHAPE(1);
            Term out = GRAD_SCALAR_TEN(0.0f, tsh);
            ctx->itrs++; return out;
        }
        // MUL (Leibniz): d(a*b)/dt = da*b + a*db.
        // a and b each appear twice (fwd + bwd cross-term),
        // target appears twice (one per recursive GRAD). DUPs.
        if (yuop == UOP_MUL) {
            Term a = heap_read(ctx, yloc + 0);
            Term b = heap_read(ctx, yloc + 1);
            Term a0, a1, b0, b1, t0, t1;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            thvm_dup(ctx, thvm_fresh_label(ctx), b, &b0, &b1);
            thvm_dup(ctx, thvm_fresh_label(ctx), tgt, &t0, &t1);
            Term da = GRAD_REC( a0, t0);
            Term db = GRAD_REC( b0, t1);
            Term l = thvm_op_raw(ctx, UOP_MUL, da, b1);
            Term r = thvm_op_raw(ctx, UOP_MUL, a1, db);
            Term out = thvm_op_raw(ctx, UOP_ADD, l, r);
            ctx->itrs++;
            return out;
        }
    }
    // y not yet WNF or pattern unhandled: leave alone; trampoline
    // retries when y reduces further.
}
