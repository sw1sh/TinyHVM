// nn/conv.c — Conv2d, MaxPool2d, Pool (tinygrad-style UOp composition)
static Term thvm_repeat(TinyHVM *ctx, Term x, const u32 *repeats, u32 ndim) {
    const View *vx = term_view(ctx, x);
    assert(vx && "thvm_repeat: input must have tracked shape");

    u32 unsq[MAX_DIM], exp[MAX_DIM], fin[MAX_DIM];
    for (u32 i = 0; i < ndim; i++) {
        unsq[i * 2]     = 1;
        unsq[i * 2 + 1] = vx->shape.dims[i];
        exp[i * 2]      = repeats[i];
        exp[i * 2 + 1]  = vx->shape.dims[i];
        fin[i]          = repeats[i] * vx->shape.dims[i];
    }
    Term t = thvm_reshape(ctx, x, shape_of(unsq, ndim * 2));
    t = thvm_expand(ctx, t, shape_of(exp, ndim * 2));
    t = thvm_reshape(ctx, t, shape_of(fin, ndim));
    return t;
}

Term thvm_pool(TinyHVM *ctx, Term x, const u32 *kernel, const u32 *stride_,
               u32 n_spatial) {
    const View *vx = term_view(ctx, x);
    assert(vx && "thvm_pool: input must have tracked shape");
    u32 ndim = vx->shape.rank;
    u32 bd = ndim - n_spatial;

    u32 i_[MAX_DIM], o_[MAX_DIM], s_[MAX_DIM], k_[MAX_DIM];
    for (u32 j = 0; j < n_spatial; j++) {
        i_[j] = vx->shape.dims[bd + j];
        s_[j] = stride_[j];
        k_[j] = kernel[j];
        o_[j] = (i_[j] - k_[j]) / s_[j] + 1;  // floor division for conv
    }

    // Check if we need the complex path (k > s for any spatial dim)
    int need_complex = 0;
    for (u32 j = 0; j < n_spatial; j++)
        if (k_[j] > s_[j]) need_complex = 1;

    Term t = x;

    if (need_complex) {
        // Tinygrad's _pool pattern: repeat → shrink → reshape → shrink → reshape → permute
        // All standard UOps — backward flows through standard rules, no special POOL_GATHER op.
        // See tinygrad tensor.py _pool() lines 2285-2296
        assert(n_spatial == 2);

        // f_ = scaling factors (always 1 for s=1, d=1)
        u32 f_[2];
        for (u32 j = 0; j < n_spatial; j++)
            f_[j] = 1 + (o_[j] * s_[j] > (i_[j] - (k_[j] - 1)) ? 1 : 0);

        // repeat: tile input so we can extract overlapping windows without padding
        u32 reps[MAX_DIM];
        for (u32 j = 0; j < bd; j++) reps[j] = 1;
        for (u32 j = 0; j < n_spatial; j++) {
            u32 ij = i_[j], kj = k_[j], fj = f_[j];
            reps[bd + j] = (kj * (ij * fj + 1) + ij - 1) / ij;  // ceildiv
        }
        t = thvm_repeat(ctx, t, reps, ndim);

        // shrink to [batch, k0*(i0*f0+1), k1*(i1*f1+1)]
        u32 sh1[MAX_DIM * 2];
        for (u32 j = 0; j < bd; j++) { sh1[j*2] = 0; sh1[j*2+1] = vx->shape.dims[j]; }
        for (u32 j = 0; j < n_spatial; j++) {
            sh1[(bd+j)*2] = 0;
            sh1[(bd+j)*2+1] = k_[j] * (i_[j] * f_[j] + 1);
        }
        t = thvm_shrink(ctx, t, sh1, ndim);

        // reshape to [batch, k0, i0*f0+1, k1, i1*f1+1]
        u32 rs1[MAX_DIM], rs1_rank = bd + n_spatial * 2;
        for (u32 j = 0; j < bd; j++) rs1[j] = vx->shape.dims[j];
        for (u32 j = 0; j < n_spatial; j++) {
            rs1[bd + j*2] = k_[j];
            rs1[bd + j*2 + 1] = i_[j] * f_[j] + 1;
        }
        t = thvm_reshape(ctx, t, shape_of(rs1, rs1_rank));

        // shrink to [batch, k0, o0*s0, k1, o1*s1]
        u32 sh2[MAX_DIM * 2];
        for (u32 j = 0; j < bd; j++) { sh2[j*2] = 0; sh2[j*2+1] = vx->shape.dims[j]; }
        for (u32 j = 0; j < n_spatial; j++) {
            sh2[(bd + j*2)*2] = 0;
            sh2[(bd + j*2)*2 + 1] = k_[j];
            sh2[(bd + j*2+1)*2] = 0;
            sh2[(bd + j*2+1)*2 + 1] = o_[j] * s_[j];
        }
        t = thvm_shrink(ctx, t, sh2, rs1_rank);

        // reshape to [batch, k0, o0, s0, k1, o1, s1]
        u32 rs2[MAX_DIM], rs2_rank = bd + n_spatial * 3;
        for (u32 j = 0; j < bd; j++) rs2[j] = vx->shape.dims[j];
        for (u32 j = 0; j < n_spatial; j++) {
            rs2[bd + j*3] = k_[j];
            rs2[bd + j*3 + 1] = o_[j];
            rs2[bd + j*3 + 2] = s_[j];
        }
        t = thvm_reshape(ctx, t, shape_of(rs2, rs2_rank));

        // shrink stride dim to 1: [batch, k0, o0, 1, k1, o1, 1]
        u32 sh3[MAX_DIM * 2];
        for (u32 j = 0; j < rs2_rank; j++) { sh3[j*2] = 0; sh3[j*2+1] = rs2[j]; }
        for (u32 j = 0; j < n_spatial; j++) {
            u32 si = bd + j*3 + 2;
            sh3[si*2+1] = 1;
        }
        t = thvm_shrink(ctx, t, sh3, rs2_rank);

        // reshape to [batch, k0, o0, k1, o1]
        u32 rs3[MAX_DIM], rs3_rank = bd + n_spatial * 2;
        for (u32 j = 0; j < bd; j++) rs3[j] = vx->shape.dims[j];
        for (u32 j = 0; j < n_spatial; j++) {
            rs3[bd + j*2] = k_[j];
            rs3[bd + j*2+1] = o_[j];
        }
        t = thvm_reshape(ctx, t, shape_of(rs3, rs3_rank));

        // permute to [batch, o0, o1, k0, k1]
        u32 perm[MAX_DIM], pi = 0;
        for (u32 j = 0; j < bd; j++) perm[pi++] = j;
        for (u32 j = 0; j < n_spatial; j++) perm[pi++] = bd + j*2 + 1;  // o dims
        for (u32 j = 0; j < n_spatial; j++) perm[pi++] = bd + j*2;      // k dims
        t = thvm_permute(ctx, t, perm, rs3_rank);

        return t;
    }

    // Simple path: k <= s (e.g., maxpool 2x2/2)
    // pad → shrink → reshape → shrink → permute

    // Step 1: pad to make divisible, then shrink to o*s
    u32 pad_pairs[MAX_DIM * 2];
    memset(pad_pairs, 0, sizeof(pad_pairs));
    int need_pad = 0;
    for (u32 j = 0; j < n_spatial; j++) {
        u32 pad_amt = (o_[j] * s_[j] > i_[j]) ? (o_[j] * s_[j] - i_[j]) : 0;
        pad_pairs[(bd + j) * 2 + 1] = pad_amt;
        if (pad_amt > 0) need_pad = 1;
    }
    if (need_pad) t = thvm_pad(ctx, t, pad_pairs, ndim);

    // Shrink to [batch..., o*s]
    u32 shrink_pairs[MAX_DIM * 2];
    for (u32 j = 0; j < ndim; j++) {
        shrink_pairs[j * 2] = 0;
        shrink_pairs[j * 2 + 1] = (j >= bd) ? o_[j - bd] * s_[j - bd] : vx->shape.dims[j];
    }
    t = thvm_shrink(ctx, t, shrink_pairs, ndim);

    // Step 2: reshape to [batch..., o0, s0, o1, s1, ...]
    u32 rs_dims[MAX_DIM], rs_rank = bd + n_spatial * 2;
    for (u32 j = 0; j < bd; j++) rs_dims[j] = vx->shape.dims[j];
    for (u32 j = 0; j < n_spatial; j++) {
        rs_dims[bd + j * 2]     = o_[j];
        rs_dims[bd + j * 2 + 1] = s_[j];
    }
    t = thvm_reshape(ctx, t, shape_of(rs_dims, rs_rank));

    // Step 3: shrink (o, k) from (o, s)
    u32 shrink2[MAX_DIM * 2];
    for (u32 j = 0; j < rs_rank; j++) {
        shrink2[j * 2] = 0;
        shrink2[j * 2 + 1] = rs_dims[j];
    }
    for (u32 j = 0; j < n_spatial; j++) {
        u32 dim_idx = bd + j * 2 + 1;
        shrink2[dim_idx * 2 + 1] = k_[j];
    }
    t = thvm_shrink(ctx, t, shrink2, rs_rank);

    // Step 4: permute to [batch..., o0, o1, ..., k0, k1, ...]
    u32 perm[MAX_DIM], pi = 0;
    for (u32 j = 0; j < bd; j++) perm[pi++] = j;
    for (u32 j = 0; j < n_spatial; j++) perm[pi++] = bd + j * 2;
    for (u32 j = 0; j < n_spatial; j++) perm[pi++] = bd + j * 2 + 1;
    t = thvm_permute(ctx, t, perm, rs_rank);

    return t;
}
// ============================================================
// Conv2d as UOp composition — matches tinygrad tensor.py:2476-2484
// ============================================================

Term thvm_conv2d(TinyHVM *ctx, Term x, Term w, Term bias,
                 u32 groups, const u32 *stride_, const u32 *padding_) {
    // x: [BS, Cin, H, W], w: [Cout, Cin/groups, KH, KW], bias: [Cout] or NULL
    const View *vx = term_view(ctx, x);
    const View *vw = term_view(ctx, w);
    assert(vx && vw && "thvm_conv2d: inputs must have tracked shapes");

    u32 bs = vx->shape.dims[0];
    u32 cin = vx->shape.dims[1];
    u32 cout = vw->shape.dims[0];
    u32 cin_g = vw->shape.dims[1];
    u32 KH = vw->shape.dims[2];
    u32 KW = vw->shape.dims[3];
    (void)cin_g;
    assert(groups * cin_g == cin);

    // Step 1: pad input
    u32 pad_pairs[MAX_DIM * 2] = {0};
    pad_pairs[2*2] = padding_[0]; pad_pairs[2*2+1] = padding_[1];  // H before/after
    pad_pairs[3*2] = padding_[2]; pad_pairs[3*2+1] = padding_[3];  // W before/after
    Term padded = x;
    if (padding_[0] || padding_[1] || padding_[2] || padding_[3]) {
        padded = thvm_pad(ctx, x, pad_pairs, 4);
    }

    // Step 2: _pool to create sliding windows
    u32 k[] = {KH, KW};
    u32 s[] = {stride_[0], stride_[1]};
    // Compute output spatial dims algebraically — no thvm_reduce needed.
    // input after padding: H + pad[0]+pad[1], W + pad[2]+pad[3]
    u32 IH = vx->shape.dims[2] + padding_[0] + padding_[1];
    u32 IW = vx->shape.dims[3] + padding_[2] + padding_[3];
    u32 oy = (IH - KH) / stride_[0] + 1;
    u32 ox = (IW - KW) / stride_[1] + 1;
    Term pooled = thvm_pool(ctx, padded, k, s, 2);
    // pooled: [BS, Cin, OY, OX, KH, KW]
    Term pr = pooled;  // lazy — shape is now known algebraically

    u32 rcout = cout / groups;

    // Step 3: reshape + expand + permute for broadcasting
    // pooled: [BS, groups, cin_g, 1, OY, OX, KH, KW]
    Term x_rs = thvm_reshape(ctx, pr,
        shape_of((u32[]){bs, groups, cin_g, 1, oy, ox, KH, KW}, 8));
    // expand to: [BS, groups, cin_g, rcout, OY, OX, KH, KW]
    Term x_exp = thvm_expand(ctx, x_rs,
        shape_of((u32[]){bs, groups, cin_g, rcout, oy, ox, KH, KW}, 8));
    // permute to: [BS, groups, rcout, OY, OX, cin_g, KH, KW]
    u32 conv_perm[] = {0, 1, 3, 4, 5, 2, 6, 7};
    Term x_perm = thvm_permute(ctx, x_exp, conv_perm, 8);

    // Step 4: reshape weight to [1, groups, rcout, 1, 1, cin_g, KH, KW]
    Term w_rs = thvm_reshape(ctx, w,
        shape_of((u32[]){1, groups, rcout, 1, 1, cin_g, KH, KW}, 8));

    // prod: [BS, groups, rcout, OY, OX, cin_g, KH, KW]
    Term prod = thvm_op(ctx, UOP_MUL, x_perm, w_rs);
    // Fused multi-axis SUM(MUL) — reduce axes 5,6,7 (cin_g, KH, KW)
    u32 reduce_axes[] = {5, 6, 7};
    Term summed = thvm_sum_axes(ctx, prod, reduce_axes, 3);
    // → [BS, groups, rcout, OY, OX, 1, 1, 1]
    Term out = thvm_reshape(ctx, summed, shape_of((u32[]){bs, cout, oy, ox}, 4));

    // Add bias
    if (term_tag(bias) != TAG_ERA) {
        Term b_rs = thvm_reshape(ctx, bias, shape_of((u32[]){1, cout, 1, 1}, 4));
        out = thvm_op(ctx, UOP_ADD, out, b_rs);
    }

    return out;
}

// ============================================================
// MaxPool2d as UOp composition — tinygrad tensor.py:2404-2405
// ============================================================

Term thvm_maxpool2d(TinyHVM *ctx, Term x, const u32 *kernel, const u32 *stride_) {
    u32 k[] = {kernel[0], kernel[1]};
    u32 s[] = {stride_[0], stride_[1]};

    const View *vx = term_view(ctx, x);
    assert(vx && "thvm_maxpool2d: input must have tracked shape");
    u32 bs = vx->shape.dims[0];
    u32 c  = vx->shape.dims[1];
    u32 IH = vx->shape.dims[2];
    u32 IW = vx->shape.dims[3];
    u32 kh = kernel[0], kw2 = kernel[1];
    u32 oy = (IH - kh) / stride_[0] + 1;
    u32 ox = (IW - kw2) / stride_[1] + 1;

    // Single pool call — pooled shape: [BS, C, OY, OX, KH, KW]
    Term pool_t = thvm_pool(ctx, x, k, s, 2);

    Term r1 = thvm_op(ctx, UOP_RMAX, pool_t, term_era());
    // → [BS, C, OY, OX, KH, 1], squeeze:
    r1 = thvm_reshape(ctx, r1, shape_of((u32[]){bs, c, oy, ox, kh}, 5));

    Term r2 = thvm_op(ctx, UOP_RMAX, r1, term_era());
    // → [BS, C, OY, OX, 1], squeeze:
    (void)kw2;
    return thvm_reshape(ctx, r2, shape_of((u32[]){bs, c, oy, ox}, 4));
}


