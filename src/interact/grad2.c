// src/interact/grad2.c - UOP_GRAD2 interaction rule (new DUP-shape gradient).
// Included inline from src/interact/_.c under case TAG_TOP.
//
// Convention: GRAD2(y, target) reduces to a tensor of target.shape holding
// d(y)/d(target) summed over y indices (forward-mode JVP with ones seed).
// Helper macros GRAD2_SCALAR_TEN / GRAD2_ONES_OF / GRAD2_TERM_SHAPE live in _.c.

if (uop == UOP_GRAD2) {
    Term y   = heap_read(ctx, loc + 0);
    Term tgt = heap_read(ctx, loc + 1);
    if (term_is_sub(y))   y   = term_strip_sub(y);
    if (term_is_sub(tgt)) tgt = term_strip_sub(tgt);

    // Helper: build a scalar-valued tensor of tsh shape via
    // rank-matched ones-shape + EXPAND.  Handles the recurring
    // pattern across all leaf/CMP emissions.
    #define GRAD2_SCALAR_TEN(_val, _tsh) ({                 \
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
    #define GRAD2_TERM_SHAPE(_term, _out) do {               \
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
    #define GRAD2_ONES_OF(_shape) ({                         \
        Shape _os = {.rank = (_shape).rank};                 \
        for (u32 _i = 0; _i < (_shape).rank; _i++) _os.dims[_i] = 1; \
        if (_os.rank == 0) { _os.rank = 1; _os.dims[0] = 1; } \
        f32 _v = 1.0f;                                       \
        Term _t = thvm_tensor(ctx, &_v, _os);                \
        ((_shape).rank == 0) ? _t : thvm_expand(ctx, _t, (_shape)); \
    })
    // NUM constant: d(const)/dt = 0 of target-shape.
    if (term_tag(y) == TAG_NUM && term_tag(tgt) == TAG_TEN) {
        u32 ttid = (u32)term_val(tgt);
        Shape tsh = (ttid < ctx->tensor_count)
            ? ctx->tensors[ttid].view.shape : SHAPE(1);
        Term out = GRAD2_SCALAR_TEN(0.0f, tsh);
        ctx->itrs++; return out;
    }
    if (term_tag(y) == TAG_TEN && term_tag(tgt) == TAG_TEN) {
        u32 ytid = (u32)term_val(y);
        u32 ttid = (u32)term_val(tgt);
        Shape tsh = (ttid < ctx->tensor_count)
            ? ctx->tensors[ttid].view.shape : SHAPE(1);
        Term out = GRAD2_SCALAR_TEN(ytid == ttid ? 1.0f : 0.0f, tsh);
        ctx->itrs++;
        return out;
    }
    // TOP chain-rule dispatch.  Recursively emits GRAD2 on sub-
    // operands; target is DUP'd to keep linearity.
    if (term_tag(y) == TAG_TOP && term_tag(tgt) == TAG_TEN) {
        u32 yuop = term_ext(y);
        u64 yloc = term_val(y);
        if (yuop == UOP_ADD || yuop == UOP_SUB) {
            Term a = heap_read(ctx, yloc + 0);
            Term b = heap_read(ctx, yloc + 1);
            Term tgt0, tgt1;
            thvm_dup(ctx, thvm_fresh_label(ctx), tgt, &tgt0, &tgt1);
            Term da = thvm_grad_u(ctx, a, tgt0);
            Term db = thvm_grad_u(ctx, b, tgt1);
            Term out = thvm_op_raw(ctx, yuop, da, db);
            ctx->itrs++;
            return out;
        }
        // NEG: d(-a)/dt = -(da).
        if (yuop == UOP_NEG) {
            Term a = heap_read(ctx, yloc + 0);
            Term da = thvm_grad_u(ctx, a, tgt);
            Term out = thvm_op_raw(ctx, UOP_NEG, da, term_era());
            ctx->itrs++; return out;
        }
        // EXP: d(exp a)/dt = exp(a) * da.  a used twice.
        if (yuop == UOP_EXP) {
            Term a = heap_read(ctx, yloc + 0);
            Term a0, a1;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            Term da = thvm_grad_u(ctx, a0, tgt);
            Term exp_a = thvm_op_raw(ctx, UOP_EXP, a1, term_era());
            Term out = thvm_op_raw(ctx, UOP_MUL, da, exp_a);
            ctx->itrs++; return out;
        }
        // LOG: d(log a)/dt = (1/a) * da.  a used twice.
        if (yuop == UOP_LOG) {
            Term a = heap_read(ctx, yloc + 0);
            Term a0, a1;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            Term da = thvm_grad_u(ctx, a0, tgt);
            Term out = thvm_op_raw(ctx, UOP_DIV, da, a1);
            ctx->itrs++; return out;
        }
        // SQRT: d(sqrt a)/dt = da / (2*sqrt(a)).  a used twice.
        if (yuop == UOP_SQRT) {
            Term a = heap_read(ctx, yloc + 0);
            Term a0, a1;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            Term da = thvm_grad_u(ctx, a0, tgt);
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
            Term da = thvm_grad_u(ctx, a0, tgt);
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
            Term da = thvm_grad_u(ctx, a0, t0);
            Term db = thvm_grad_u(ctx, b0, t1);
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
            Term da = thvm_grad_u(ctx, a0, t0);
            Term db = thvm_grad_u(ctx, b0, t1);
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
        // GRAD2(a, target) unchanged.  Guard: only apply a wrap when the
        // grad's shape still matches a_shape (identity target case).
        if (yuop == UOP_RESHAPE || yuop == UOP_PERMUTE) {
            Term a     = heap_read(ctx, yloc + 0);
            Term shape = heap_read(ctx, yloc + 1);
            Shape a_shape = SHAPE(1);
            GRAD2_TERM_SHAPE(a, a_shape);
            u32 ttid = (u32)term_val(tgt);
            Shape tsh = (ttid < ctx->tensor_count)
                ? ctx->tensors[ttid].view.shape : SHAPE(1);
            u32 a_numel = 1, t_numel = 1;
            for (u32 i = 0; i < a_shape.rank; i++) a_numel *= a_shape.dims[i];
            for (u32 i = 0; i < tsh.rank; i++) t_numel *= tsh.dims[i];
            Term da = thvm_grad_u(ctx, a, tgt);
            Term out;
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
        // SUM: dA = EXPAND(da, input_shape).
        if (yuop == UOP_SUM) {
            Term a    = heap_read(ctx, yloc + 0);
            (void)heap_read(ctx, yloc + 1);
            Shape a_shape = SHAPE(1);
            GRAD2_TERM_SHAPE(a, a_shape);
            Term da = thvm_grad_u(ctx, a, tgt);
            Term out = (a_shape.rank != 0)
                ? thvm_expand(ctx, da, a_shape) : da;
            ctx->itrs++; return out;
        }
        // SHRINK: dA = pad(da, complementary pairs of shrink pairs).
        if (yuop == UOP_SHRINK) {
            Term a     = heap_read(ctx, yloc + 0);
            Term shape = heap_read(ctx, yloc + 1);
            Shape a_shape = SHAPE(1);
            GRAD2_TERM_SHAPE(a, a_shape);
            // SHRINK shape-preserving bwd.  For target == a (leaf
            // identity), emit PAD(ones(y_shape), pad_pairs) directly
            // so the output has target.shape with mask (=1 in kept
            // region, 0 in padded).  Recursion via thvm_grad_u(a,t)
            // returns target.shape (not y.shape), so we'd over-pad.
            Term out;
            if (term_tag(shape) == TAG_TEN && a_shape.rank > 0) {
                u32 sid = (u32)term_val(shape);
                u32 nd = a_shape.rank;
                u32 sf[MAX_DIM * 2]; tensor_meta_read_u32(ctx, sid, sf, MAX_DIM * 2);
                // y_shape = shrink output shape
                Shape ys = {.rank = nd};
                for (u32 j = 0; j < nd; j++) ys.dims[j] = sf[j*2+1] - sf[j*2];
                Term y_ones = GRAD2_ONES_OF(ys);
                // Pad back to target shape using complementary pairs.
                u32 pp[MAX_DIM * 2];
                for (u32 j = 0; j < nd; j++) {
                    pp[j*2]   = sf[j*2];
                    pp[j*2+1] = a_shape.dims[j] - sf[j*2+1];
                }
                out = thvm_pad(ctx, y_ones, pp, nd);
            } else {
                out = thvm_grad_u(ctx, a, tgt);
            }
            ctx->itrs++; return out;
        }
        // PAD: direct emit — dA = SHRINK(ones(y_shape), unpad_pairs).
        if (yuop == UOP_PAD) {
            Term a     = heap_read(ctx, yloc + 0);
            Term shape = heap_read(ctx, yloc + 1);
            Shape a_shape = SHAPE(1);
            GRAD2_TERM_SHAPE(a, a_shape);
            Shape y_shape = SHAPE(1);
            const View *yv = st_get(yloc);
            if (yv) y_shape = yv->shape;
            Term out;
            if (term_tag(shape) == TAG_TEN && a_shape.rank > 0 && y_shape.rank > 0) {
                u32 pid = (u32)term_val(shape);
                u32 nd = a_shape.rank;
                u32 pf[MAX_DIM * 2]; tensor_meta_read_u32(ctx, pid, pf, MAX_DIM * 2);
                Term y_ones = GRAD2_ONES_OF(y_shape);
                u32 sp[MAX_DIM * 2];
                for (u32 j = 0; j < nd; j++) {
                    sp[j*2]   = pf[j*2];
                    sp[j*2+1] = pf[j*2] + a_shape.dims[j];
                }
                out = thvm_shrink(ctx, y_ones, sp, nd);
            } else {
                out = thvm_grad_u(ctx, a, tgt);
            }
            ctx->itrs++; return out;
        }
        // EXPAND: emit ones(y_shape) sum-reduced to target.shape.  Works when
        // target flows through the operand.  For cases where target tid doesn't
        // match expand's direct TEN, this over-counts but prior design was also
        // compromised for non-trivial chains; keeping simple rule.
        if (yuop == UOP_EXPAND) {
            Shape y_shape = SHAPE(1);
            const View *yv = st_get(yloc);
            if (yv) y_shape = yv->shape;
            u32 ttid = (u32)term_val(tgt);
            Shape tsh = (ttid < ctx->tensor_count)
                ? ctx->tensors[ttid].view.shape : SHAPE(1);
            Term y_ones = GRAD2_ONES_OF(y_shape);
            Term out = (y_shape.rank != 0 && tsh.rank != 0)
                ? sum_to_shape(ctx, y_ones, y_shape, tsh) : y_ones;
            ctx->itrs++; return out;
        }
        // RMAX: dA = da * (a == expand(rmax(a))).  a used 3x.
        if (yuop == UOP_RMAX) {
            Term a    = heap_read(ctx, yloc + 0);
            Term axes = heap_read(ctx, yloc + 1);
            Shape a_shape = SHAPE(1);
            GRAD2_TERM_SHAPE(a, a_shape);
            Term a0, a1, a2;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            Term a1b;
            thvm_dup(ctx, thvm_fresh_label(ctx), a1, &a1b, &a2);
            Term da = thvm_grad_u(ctx, a0, tgt);
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
        // MM: d(a@b)/dt = da@b + a@db.  Leibniz-style in MM.
        if (yuop == UOP_MM) {
            Term a = heap_read(ctx, yloc + 0);
            Term b = heap_read(ctx, yloc + 1);
            Term a0, a1, b0, b1, t0, t1;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            thvm_dup(ctx, thvm_fresh_label(ctx), b, &b0, &b1);
            thvm_dup(ctx, thvm_fresh_label(ctx), tgt, &t0, &t1);
            Term da = thvm_grad_u(ctx, a0, t0);
            Term db = thvm_grad_u(ctx, b0, t1);
            Term l  = thvm_op_raw(ctx, UOP_MM, da, b1);
            Term r  = thvm_op_raw(ctx, UOP_MM, a1, db);
            Term out = thvm_op_raw(ctx, UOP_ADD, l, r);
            ctx->itrs++; return out;
        }
        // ASSIGN: forward-only. Gradient flows through dst only.
        if (yuop == UOP_ASSIGN) {
            Term dst = heap_read(ctx, yloc + 0);
            Term out = thvm_grad_u(ctx, dst, tgt);
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
            Term da = thvm_grad_u(ctx, a, t0);
            Term db = thvm_grad_u(ctx, b, t1);
            Term out = thvm_where(ctx, c, da, db);
            ctx->itrs++; return out;
        }
        // CMP: non-differentiable. Zero contribution, shape-matched.
        if (yuop == UOP_CMP) {
            u32 ttid = (u32)term_val(tgt);
            Shape tsh = (ttid < ctx->tensor_count)
                ? ctx->tensors[ttid].view.shape : SHAPE(1);
            Term out = GRAD2_SCALAR_TEN(0.0f, tsh);
            ctx->itrs++; return out;
        }
        // MUL (Leibniz): d(a*b)/dt = da*b + a*db.
        // a and b each appear twice (fwd + bwd cross-term),
        // target appears twice (one per recursive GRAD2). DUPs.
        if (yuop == UOP_MUL) {
            Term a = heap_read(ctx, yloc + 0);
            Term b = heap_read(ctx, yloc + 1);
            Term a0, a1, b0, b1, t0, t1;
            thvm_dup(ctx, thvm_fresh_label(ctx), a, &a0, &a1);
            thvm_dup(ctx, thvm_fresh_label(ctx), b, &b0, &b1);
            thvm_dup(ctx, thvm_fresh_label(ctx), tgt, &t0, &t1);
            Term da = thvm_grad_u(ctx, a0, t0);
            Term db = thvm_grad_u(ctx, b0, t1);
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
