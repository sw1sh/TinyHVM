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

    // Direct stride computation for sliding window (no thvm_repeat needed).
    // pool[..., o0, o1, k0, k1] = input[..., o0*s0+k0, o1*s1+k1]
    // Strides: output dims use input_stride*s, kernel dims use input_stride*1.
    // This is a single view alias — no expand, no contiguify.
    if (n_spatial == 2) {
        int is_ten = (term_tag(x) == TAG_TEN);
        u32 src_id = is_ten ? (u32)term_val(x) : 0;
        TensorMeta *m = is_ten ? &ctx->tensors[src_id] : NULL;
        if (!is_ten || m->buf_id != 0 || m->creator_op) {
            // Build output shape: [batch..., o0, o1, k0, k1]
            u32 out_rank = bd + n_spatial * 2;
            Shape out_shape = {.rank = out_rank};
            for (u32 j = 0; j < bd; j++) out_shape.dims[j] = vx->shape.dims[j];
            for (u32 j = 0; j < n_spatial; j++) {
                out_shape.dims[bd + j] = o_[j];               // output spatial
                out_shape.dims[bd + n_spatial + j] = k_[j];   // kernel window
            }

            // Build strides from input view
            View ov = {0};
            ov.shape = out_shape;
            ov.offset = vx->offset;
            ov.numel = 1;
            for (u32 j = 0; j < out_rank; j++) ov.numel *= out_shape.dims[j];

            // Batch dims: same strides as input
            for (u32 j = 0; j < bd; j++)
                ov.strides[j] = vx->strides[j];

            // Output spatial dims: stride = input_spatial_stride * pool_stride
            for (u32 j = 0; j < n_spatial; j++)
                ov.strides[bd + j] = vx->strides[bd + j] * (i32)s_[j];

            // Kernel window dims: stride = input_spatial_stride * 1 (dilation=1)
            for (u32 j = 0; j < n_spatial; j++)
                ov.strides[bd + n_spatial + j] = vx->strides[bd + j];

            ov.contiguous = 0;
            // Mask propagation: the padded input's mask is per-dim on the
            // original spatial dims. For the pool view, the mask needs to
            // express which (oy,kh) combinations are valid — but this couples
            // two dims and can't be expressed as independent per-dim masks.
            // Solution: propagate the mask through the stride mapping.
            // For padded spatial dim with valid range [begin,end):
            //   output dim oy is valid for all oy (0..o-1) since oy*s+kh < padded_size
            //   kernel dim kh is valid for all kh (0..k-1)
            // The out-of-bounds access is handled by the original mask check:
            //   the index oy*s+kh maps to the padded spatial dim, and the
            //   mask on that dim checks [pad_before, pad_before+orig_size).
            // So we DON'T set has_mask on the pool output — the leaf's mask
            // handles it when the codegen generates index expressions.
            // EXCEPT: the leaf view is now the pool view, not the padded view.
            // The codegen sees the pool view's strides. The mask must be on
            // the padded spatial dims, which are the inner dims of the pool.
            //
            // Propagate pad mask: for each spatial dim, the validity condition
            // is pad_begin <= oy*s + kh*d < pad_end. We encode this as a mask
            // on the KERNEL dim with begin/end derived from the pad bounds.
            // The codegen checks mask_begin[d] <= c_d < mask_end[d] per dim.
            // For the output dim: always valid (full range).
            // For the kernel dim: conservatively valid, but we add the compound
            // check by setting mask on a "virtual" basis.
            //
            // Actually: the mask check needs to be on the COMPOSED coordinate.
            // The View mask system checks per-dim only. For compound checks,
            // we rely on the buffer offset going out of range — the PAD view
            // originally stored has_mask with begin/end on the spatial dim.
            // Since the pool view SHARES the padded buffer, and the strides
            // map (oy, kh) to the padded spatial index, the codegen will
            // compute index = oy*s + kh which may land in the padding zone.
            // The original PAD mask on the padded tensor checked this.
            //
            // But the pool view is a NEW tensor with its OWN view. The mask
            // must be on the pool view's dims. Set has_mask=1 and encode
            // the pad bounds on the kernel dims:
            //   kh valid when: pad_begin <= oy*stride_oy/stride_kh + kh < pad_end
            // This can't be per-dim. So encode as: the pool view itself is
            // unmasked, but the BUFFER has padding zeros outside [begin,end).
            //
            // If the original PAD used offset to shift the view start into
            // the valid region, the pool view inherits that offset. Elements
            // where oy*s+kh lands in the padding zone read from memory that
            // was initialized to 0 by the PAD (via mask→0.0f in codegen).
            //
            // The correct solution: set has_mask on the pool view with
            // the mask checking the compound condition. Extend View to
            // support compound masks, or just pass the pad bounds through.
            //
            // Pragmatic: the mask check IS the condition
            //   mask_begin_spatial <= oy*s + kh < mask_end_spatial
            // We can't express this per-dim. BUT: the offset-based access
            // naturally returns 0 if the PAD created a zero-filled buffer.
            // PAD in TinyHVM uses has_mask (NOT zero-fill). So we NEED the mask.
            //
            // Simplest correct fix: DON'T set has_mask on pool view.
            // Instead, check: does the PAD use offset to make the valid
            // region start at element 0? If so, pool strides work without mask.
            // Pad mask propagation: the padded input has mask_begin/mask_end on
            // spatial dims. For the pool view, validity is:
            //   mask_begin_H <= oy*s + kh < mask_end_H
            //   mask_begin_W <= ox*s + kw < mask_end_W
            // Express this as per-dim masks using the PAD bounds:
            //   For oy dim: begin = ceil((pad_begin_H) / s), but this loses the kh dependency
            // Instead: set has_mask=1 on pool view. Store pad bounds in mask_begin/mask_end
            // for the kernel dims. The codegen checks c_d >= begin && c_d < end per dim.
            // For the OUTPUT dims (oy, ox): full range [0, o).
            // For the KERNEL dims (kh, kw): the mask is a PROXY — the codegen
            // computes the composed index (c_oy*s + c_kh) and checks against pad bounds.
            //
            // Encode: mask_begin[kh_dim] = mask_begin_H of padded input
            //         mask_end[kh_dim] = mask_end_H of padded input
            // The codegen generates: c_kh >= begin && c_kh < end — WRONG (ignores oy*s).
            //
            // True fix: extend the codegen's mask check for pool views to use
            // compound coordinates. For now, the pool view inherits has_mask
            // and stores the original pad bounds. The codegen already generates
            // the index as offset + c_oy*stride_oy + c_kh*stride_kh. When this
            // index is negative or past buffer end, the mask must catch it.
            //
            // ACTUAL solution: use the PAD's mask bounds as-is. The pool view's
            // coordinates (oy, kh) map to the padded spatial coordinate via
            // c_spatial = oy*s + kh. The PAD mask checks c_spatial in [begin, end).
            // We replicate this by NOT decomposing the spatial dim — the mask stays
            // on the original spatial coordinate.
            //
            // But the pool view has SEPARATE oy and kh dims, not a single spatial dim.
            // The codegen needs to CHECK: is (c_oy * s + c_kh) within the valid range?
            //
            // Implementation: set has_mask=1 on pool view. For each spatial pair (j):
            //   mask_begin[oy_dim] = 0; mask_end[oy_dim] = o_j  (always valid per-dim)
            //   mask_begin[kh_dim] = 0; mask_end[kh_dim] = k_j  (always valid per-dim)
            // Then add compound mask info: pool_mask_spatial_begin/end per spatial dim.
            // The codegen generates the compound check.
            //
            // Simpler: just use the existing mask system but set it so that
            // the check on kh is pad_begin <= kh < pad_end. For kh=0, this means
            // pad_begin must be 0. For pad=1, mask_begin[kh]=1 → kh=0 is MASKED.
            // kh=1,2 are valid. oy=0: spatial = 0+kh. kh=1 → spatial=1 ✓ (pad_begin=1).
            //
            // This IS correct for s=1! When s=1, c_spatial = c_oy + c_kh.
            // mask_begin[kh] = pad_begin_H means kh >= pad_begin_H.
            // Combined: spatial = oy + kh >= oy + pad_begin_H >= pad_begin_H ✓
            // But also need: spatial < pad_end. oy + kh < pad_end → kh < pad_end - oy.
            // This depends on oy → per-dim mask on kh ALONE is too conservative.
            //
            // For s=1, d=1: set mask_begin[kh]=pad_begin, mask_end[kh]=pad_end.
            // Then per-dim check: pad_begin <= kh < pad_end.
            // Combined: spatial = oy + kh, and oy in [0, o).
            // When oy=0, kh must be in [pad_begin, pad_end) for validity.
            // When oy=1, kh must be in [pad_begin-1, pad_end-1) for spatial validity.
            // But per-dim check is [pad_begin, pad_end) regardless of oy → CONSERVATIVE.
            // Some valid (oy=1, kh=0) where spatial=1 >= pad_begin=1 ✓ but kh=0 < pad_begin=1 → masked → reads 0.
            // This is WRONG — valid data masked to 0. Loses information.
            //
            // So per-dim masks DON'T work. Must extend codegen.
            // For now: fallback for padded inputs.
            if (vx->has_mask) {
                // Compound mask: validity is begin <= c_oy*s + c_kh < end per spatial dim.
                ov.has_mask = 1;
                // Per-dim masks: full range (compound check handles validity)
                for (u32 d = 0; d < out_rank; d++) {
                    ov.mask_begin[d] = 0;
                    ov.mask_end[d] = out_shape.dims[d];
                }
                // Compound masks for each spatial dim
                ov.n_compound_masks = (u8)n_spatial;
                for (u32 j = 0; j < n_spatial && j < 2; j++) {
                    ov.compound_masks[j].dim_a = (u8)(bd + j);              // output dim (oy/ox)
                    ov.compound_masks[j].dim_b = (u8)(bd + n_spatial + j);  // kernel dim (kh/kw)
                    ov.compound_masks[j].stride_a = (i32)s_[j];            // stride multiplier
                    ov.compound_masks[j].begin = vx->mask_begin[bd + j];   // pad begin
                    ov.compound_masks[j].end = vx->mask_end[bd + j];       // pad end
                }
            }

            if (is_ten) {
                // TAG_TEN: create view alias with real buffer
                u32 vid = tensor_view_of(ctx, src_id, ov);
                ctx->tensors[vid].creator_op = UOP_RESHAPE;
                ctx->tensors[vid].src_ids[0] = src_id;
                if (m->requires_grad) ctx->tensors[vid].requires_grad = 1;
                if (n_spatial == 2) {
                    ctx->tensors[vid].pool_n_spatial = 2;
                    ctx->tensors[vid].pool_kernel[0] = (u8)k_[0];
                    ctx->tensors[vid].pool_kernel[1] = (u8)k_[1];
                    ctx->tensors[vid].pool_stride[0] = (u8)s_[0];
                    ctx->tensors[vid].pool_stride[1] = (u8)s_[1];
                }
                return term_ten(vid, m->dtype);
            } else {
                // TAG_TOP: pool view with custom strides in shape table.
                // Create a shape tensor so the interact handler can resolve it.
                f32 shape_f[MAX_DIM];
                for (u32 j = 0; j < out_rank; j++) shape_f[j] = (f32)out_shape.dims[j];
                Term shape_t = thvm_tensor(ctx, shape_f,
                    (Shape){.dims={out_rank}, .rank=1});
                Term pool_top = thvm_op(ctx, UOP_RESHAPE, x, shape_t);
                // Override the st_set from thvm_op with pool view (custom strides)
                st_set(term_val(pool_top), &ov);
                return pool_top;
            }
        }
    }

    // Fallback: original repeat-based path (also used for masked/padded inputs)
pool_fallback:;
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

    // Record conv info for specialized backward (avoids 8-dim contiguify chain).
    // The FUSING tensor inherits this when the rewrite rules fuse the chain.
    // Record conv info for specialized backward.
    // Record conv info for specialized backward (avoids 8-dim contiguify chain).
    // NOTE: currently disabled — out is TAG_TOP (lazy) so this never fires.
    // Enabling requires fixing thvm_pool's complex path interaction with no_fuse=1.
    if (term_tag(out) == TAG_TEN) {
        u32 out_id = (u32)term_val(out);
        TensorMeta *om = &ctx->tensors[out_id];
        om->conv_input_id = (term_tag(padded) == TAG_TEN) ? (u32)term_val(padded) : 0;
        om->conv_weight_id = (term_tag(w) == TAG_TEN) ? (u32)term_val(w) : 0;
        om->conv_groups = (u16)groups;
        om->conv_KH = (u8)KH; om->conv_KW = (u8)KW;
        om->conv_stride[0] = (u8)stride_[0]; om->conv_stride[1] = (u8)stride_[1];
        om->conv_padding[0] = (u8)padding_[0]; om->conv_padding[1] = (u8)padding_[1];
        om->conv_padding[2] = (u8)padding_[2]; om->conv_padding[3] = (u8)padding_[3];
    }

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
    const View *vx = term_view(ctx, x);
    assert(vx && "thvm_maxpool2d: input must have tracked shape");
    u32 bs = vx->shape.dims[0];
    u32 c  = vx->shape.dims[1];
    u32 IH = vx->shape.dims[2];
    u32 IW = vx->shape.dims[3];
    u32 kh = kernel[0], kw = kernel[1];
    u32 sh = stride_[0], sw = stride_[1];
    u32 oy = (IH - kh) / sh + 1;
    u32 ox = (IW - kw) / sw + 1;

    // Non-overlapping pool (stride == kernel): decompose into reshape + permute.
    // Forward: [BS,C,H,W] → reshape [BS,C,OY,KH,OX,KW] → permute [BS,C,OY,OX,KH,KW]
    // Backward correctly reverses: inv_permute → reshape → [BS,C,H,W]
    // (The old thvm_pool strided view had wrong backward: plain reshape scrambles elements)
    if (sh == kh && sw == kw) {
        Term r1 = thvm_reshape(ctx, x, shape_of((u32[]){bs, c, oy, kh, ox, kw}, 6));
        Term r2 = thvm_permute(ctx, r1, (u32[]){0,1,2,4,3,5}, 6);
        Term r = thvm_rmax_axes(ctx, r2, (u32[]){4, 5}, 2);
        return thvm_reshape(ctx, r, shape_of((u32[]){bs, c, oy, ox}, 4));
    }

    // Overlapping pool: use strided view via thvm_pool
    u32 k[] = {kernel[0], kernel[1]};
    u32 s[] = {stride_[0], stride_[1]};
    Term pool_t = thvm_pool(ctx, x, k, s, 2);
    Term r = thvm_rmax_axes(ctx, pool_t, (u32[]){4, 5}, 2);
    return thvm_reshape(ctx, r, shape_of((u32[]){bs, c, oy, ox}, 4));
}


