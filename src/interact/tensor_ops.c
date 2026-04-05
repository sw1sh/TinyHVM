            // === Phase 3 inet ops ===

            if (uop == UOP_ASSIGN) {
                // Args WNF from trampoline. When reached via inet_step
                // (combinator chain), args may be unreduced — return t
                // so the trampoline re-enters and reduces args.
                Term dst_r = heap_read(ctx, loc);
                Term src_t = heap_read(ctx, loc + 1);
                if (term_tag(dst_r) != TAG_TEN || term_tag(src_t) != TAG_TEN)
                    return t;  // trampoline will re-enter and reduce args
                {
                    u32 dst_id = (u32)term_val(dst_r);
                    u32 src_id = (u32)term_val(src_t);
                    if (ctx->defer_all && ctx->tensors[src_id].buf_id == 0) return t; // scheduler handles
                    if (dst_id != src_id) {
                        TensorMeta *md = &ctx->tensors[dst_id];
                        TensorMeta *ms = &ctx->tensors[src_id];
                        // ASSIGN elision: if src is deferred ew with same numel,
                        // tell materializer to write directly into dst's buffer.
                        if (ms->buf_id == 0 && ms->creator_op &&
                            is_elementwise(ms->creator_op) &&
                            ms->view.contiguous && ms->view.numel == md->view.numel &&
                            ms->defer_consumers == 0) {
                            ms->assign_target = md->buf_id;
                            ENSURE(ctx, src_id);
                            ms = &ctx->tensors[src_id];
                            ms->assign_target = 0;
                            // If materializer used the target, no blit needed
                            if (ms->buf_id == md->buf_id) goto assign_done;
                        } else {
                            ENSURE(ctx, src_id);
                            ms = &ctx->tensors[src_id]; // refresh after ENSURE
                            md = &ctx->tensors[dst_id];
                            if (md->backend->buf_copy && ms->view.contiguous &&
                                ms->view.numel == md->view.numel) {
                                md->backend->buf_copy(md->buf_id, ms->buf_id,
                                    (u64)md->view.numel * sizeof(f32));
                            } else if (md->backend->contiguify) {
                                md->backend->contiguify(md->buf_id, md->view.numel,
                                                         ms->buf_id, &ms->view);
                            } else {
                                u64 nbytes = (u64)md->view.numel * sizeof(f32);
                                f32 *src_host = (f32 *)thvm_to_host(ctx, src_t);
                                md->backend->buf_write(md->buf_id, src_host, nbytes);
                            }
                        }
                    assign_done:
                        if (md->host_ptr) { free(md->host_ptr); md->host_ptr = NULL; }
                        heap_set(ctx, loc + 1, dst_r);
                    }
                    ctx->itrs++;
                    RETURN_REDUCED(dst_r);
                }
            }

            if (uop == UOP_TODEVICE) {
                // UOP_TODEVICE(tensor, device_idx_scalar)
                // Read tensor from source backend, write to target backend
                Term src_t = heap_read(ctx, loc);     // WNF from trampoline
                Term dev_t = heap_read(ctx, loc + 1); // WNF from trampoline
                if (term_tag(src_t) == TAG_TEN && term_tag(dev_t) == TAG_TEN) {
                    u32 src_id = (u32)term_val(src_t);
                    ENSURE(ctx, src_id);
                    TensorMeta *ms = &ctx->tensors[src_id];
                    u32 dev_id = (u32)term_val(dev_t); ENSURE(ctx, dev_id);
                    f32 dev_f; ctx->tensors[dev_id].backend->buf_read(
                        ctx->tensors[dev_id].buf_id, &dev_f, sizeof(f32));
                    u32 dev_idx = (u32)dev_f;
                    Backend *dst_be = ctx->backends[dev_idx];
                    if (!dst_be || ms->backend == dst_be) {
                        ctx->itrs++;
                        RETURN_REDUCED(src_t); // no-op: same device or invalid
                    }
                    // Read to host, allocate on target device
                    f32 *host = (f32 *)thvm_to_host(ctx, src_t);
                    u32 new_id = tensor_create(ctx, ms->view.shape, ms->dtype);
                    TensorMeta *mn = &ctx->tensors[new_id];
                    mn->backend->buf_free(mn->buf_id);
                    mn->backend = dst_be;
                    mn->buf_id = dst_be->buf_alloc((u64)mn->view.numel * dtype_size(mn->dtype));
                    dst_be->buf_write(mn->buf_id, host, (u64)mn->view.numel * dtype_size(mn->dtype));
                    mn->requires_grad = ms->requires_grad;
                    ctx->itrs++;
                    RETURN_REDUCED(term_ten(new_id, mn->dtype));
                }
                RETURN_REDUCED(term_era());
            }

            if (uop == UOP_WHERE) {
                // UOP_WHERE(cond_ten, then_ten, else_ten) — elementwise ternary select
                // Per tinyspec §ElementwiseOps: result[i] = A[i] if P[i]!=0 else B[i]
                Term cond_t = heap_read(ctx, loc);     // WNF from trampoline
                Term then_t = heap_read(ctx, loc + 1); // WNF from trampoline
                Term else_t = heap_read(ctx, loc + 2); // WNF from trampoline (TAG_TOP2)
                if (term_tag(cond_t) != TAG_TEN || term_tag(then_t) != TAG_TEN ||
                    term_tag(else_t) != TAG_TEN) RETURN_REDUCED(term_era());
                u32 c_id = (u32)term_val(cond_t);
                u32 a_wid = (u32)term_val(then_t);
                u32 b_wid = (u32)term_val(else_t);
                TensorMeta *mc = &ctx->tensors[c_id];
                TensorMeta *ma_w = &ctx->tensors[a_wid];
                TensorMeta *mb_w = &ctx->tensors[b_wid];
                u32 n = ma_w->view.numel;
                assert(mc->view.numel == n && mb_w->view.numel == n);
                // Materialize all three
                f32 *cp = malloc(n * sizeof(f32)), *ap = malloc(n * sizeof(f32)),
                    *bp = malloc(n * sizeof(f32)), *rp = malloc(n * sizeof(f32));
                mc->backend->buf_read(mc->buf_id, cp, n*sizeof(f32));
                ma_w->backend->buf_read(ma_w->buf_id, ap, n*sizeof(f32));
                mb_w->backend->buf_read(mb_w->buf_id, bp, n*sizeof(f32));
                for (u32 i = 0; i < n; i++) rp[i] = (cp[i] != 0.0f) ? ap[i] : bp[i];
                free(cp); free(ap); free(bp);
                u32 dst_wid = tensor_create(ctx, ma_w->view.shape, ma_w->dtype);
                ctx->tensors[dst_wid].backend->buf_write(ctx->tensors[dst_wid].buf_id, rp, n*sizeof(f32));
                free(rp);
                ctx->itrs++;
                RETURN_REDUCED(term_ten(dst_wid, ma_w->dtype));
            }

            // UOP_IFZ(counter, zero_case, succ_lam)
            // counter==0 → zero_case; counter>0 → APP(succ_lam, TEN(counter-1))
            if (uop == UOP_IFZ) {
                Term ctr = heap_read(ctx, loc); // WNF from trampoline
                if (term_tag(ctr) != TAG_TEN) RETURN_REDUCED(term_era());
                f32 *val = thvm_to_host(ctx, ctr);
                ctx->itrs++;
                if (val[0] <= 0.0f) {
                    // Base case: return zero_case (slot 1)
                    return heap_read(ctx, loc + 1);
                } else {
                    // Recursive case: decrement, apply succ_lam
                    f32 nv = val[0] - 1.0f;
                    Term dec = thvm_tensor(ctx, &nv, SHAPE(1));
                    Term succ_lam = heap_read(ctx, loc + 2);
                    return thvm_app(ctx, succ_lam, dec);
                }
            }

            // UOP_LOG_PRINT(tensor) — print scalar value, return tensor unchanged
            if (uop == UOP_LOG_PRINT) {
                Term t = heap_read(ctx, loc); // WNF from trampoline
                ctx->itrs++;
                if (term_tag(t) == TAG_TEN) {
                    u32 tid = (u32)term_val(t);
                    f32 *val = thvm_to_host(ctx, t);
                    u32 n = ctx->tensors[tid].view.numel;
                    if (n == 1) printf("  loss: %.6f\n", val[0]);
                    else        printf("  tensor[%u]: %.6f ...\n", n, val[0]);
                }
                RETURN_REDUCED(t);
            }

            // === Fusion handled by rewrite rules (rewrite/_.c) ===
            // No FUSE handlers or SUM(MUL) pattern needed here.
            // The trampoline's rewrite_apply() handles fusion before
            // depth-first reduction. This interact handler only fires
            // for ops after their args are reduced to TAG_TEN.

            // (Elementwise chain FUSE, SUM(MUL) pattern — handled by rewrite rules)

            // Fusion check moved to thvm_reduce (reduce/_.c) — fires BEFORE
            // the trampoline reduces args, so lazy TAG_TOP chains are intact.

            // Movement ops: modify View, share buffer
            int is_movement = (uop >= UOP_RESHAPE && uop <= UOP_PAD);

            // Args resolved by thvm_reduce's enter/apply loop before firing.
            // Direct reads here — no thvm_reduce recursion.
            Term a = heap_read(ctx, loc);
            heap_set(ctx, loc, a);

            int is_binary = (uop >= UOP_ADD && uop <= UOP_SUB) || uop == UOP_MM;
            int is_reduce = (uop == UOP_SUM || uop == UOP_RMAX);
            Term b = term_era();
            if (is_binary || is_movement) {
                b = heap_read(ctx, loc + 1);
                heap_set(ctx, loc + 1, b);
            }

            // === TOP-SUP: distribute tensor op across superposition ===
            // TOP(op, &L{a0,a1}, b) → &L{TOP(op,a0,DP0_L(b)), TOP(op,a1,DP1_L(b))}
            if (term_tag(a) == TAG_SUP) {
                u32 lab = term_ext(a);
                u64 sup_loc = term_val(a);
                Term a0 = heap_read(ctx, sup_loc + 0);
                Term a1 = heap_read(ctx, sup_loc + 1);
                Term b0 = b, b1 = b;
                if (is_binary || is_movement) {
                    // DUP the other arg with the SUP's label
                    u64 dup_loc = heap_alloc(ctx, 1);
                    heap_set(ctx, dup_loc, b);
                    b0 = term_new(TAG_DP0, lab, dup_loc);
                    b1 = term_new(TAG_DP1, lab, dup_loc);
                }
                ctx->itrs++;
                RETURN_REDUCED(thvm_sup(ctx, lab,
                    thvm_op_raw(ctx, uop, a0, b0),
                    thvm_op_raw(ctx, uop, a1, b1)));
            }
            if (term_tag(b) == TAG_SUP && (is_binary || is_movement)) {
                u32 lab = term_ext(b);
                u64 sup_loc = term_val(b);
                Term b0 = heap_read(ctx, sup_loc + 0);
                Term b1 = heap_read(ctx, sup_loc + 1);
                // a is already reduced (TEN/NUM/ERA) — atoms copy freely
                ctx->itrs++;
                RETURN_REDUCED(thvm_sup(ctx, lab,
                    thvm_op_raw(ctx, uop, a, b0),
                    thvm_op_raw(ctx, uop, a, b1)));
            }

            // TAG_NUM fast path: inline scalar compute without tensors/buffers.
            // ~5ns per interaction vs ~30ns with tensor_create + buf ops.
            if (term_tag(a) == TAG_NUM && !is_movement && !is_reduce && uop != UOP_MM) {
                f32 va = term_as_f32(a);
                f32 vb = (is_binary && term_tag(b) == TAG_NUM) ? term_as_f32(b) : 0;
                f32 vr;
                if (is_binary && term_tag(b) != TAG_NUM) goto num_skip;
                switch (uop) {
                    case UOP_ADD: vr=va+vb; break; case UOP_SUB: vr=va-vb; break;
                    case UOP_MUL: vr=va*vb; break; case UOP_DIV: vr=vb!=0?va/vb:0; break;
                    case UOP_NEG: vr=-va; break;   case UOP_RELU: vr=va>0?va:0; break;
                    case UOP_EXP: vr=__builtin_expf(va); break;
                    case UOP_LOG: vr=__builtin_logf(va); break;
                    case UOP_SQRT: vr=__builtin_sqrtf(va); break;
                    case UOP_MAX: vr=va>vb?va:vb; break;
                    case UOP_CMP: vr=va>vb?1.f:0.f; break;
                    default: goto num_skip;
                }
                ctx->itrs++;
                RETURN_REDUCED(term_num_f32(vr));
            }
            num_skip:

            // ERA-as-identity for grad domain: ADD/SUB/MUL/DIV with ERA operands
            // ADD(ERA,x)→x, ADD(x,ERA)→x, ADD(ERA,ERA)→ERA, MUL(*,ERA)→ERA
            // tensor_release: ERA-discarded tensors are genuinely dead — no GRAD
            // handler will traverse them (gradient signal is dead).
            if (is_binary) {
                Term b_check = heap_read(ctx, loc + 1);
                u8 at = term_tag(a), bt2 = term_tag(b_check);
                int a_era = (at == TAG_ERA), b_era = (bt2 == TAG_ERA);
                int a_ten = (at == TAG_TEN), b_ten = (bt2 == TAG_TEN);
                if (a_era || b_era) {
                    // Both ERA → ERA
                    if (a_era && b_era) RETURN_REDUCED(term_era());
                    // ERA+TEN or TEN+ERA: identity returns the survivor
                    if (uop == UOP_ADD) RETURN_REDUCED(a_era ? b_check : a);
                    if (uop == UOP_SUB) {
                        if (a_era && b_ten) tensor_release(ctx, (u32)term_val(b_check));
                        RETURN_REDUCED(a_era ? term_era() : a);
                    }
                    if (uop == UOP_MUL || uop == UOP_DIV) {
                        if (!a_era && a_ten) tensor_release(ctx, (u32)term_val(a));
                        if (!b_era && b_ten) tensor_release(ctx, (u32)term_val(b_check));
                        RETURN_REDUCED(term_era());
                    }
                    (void)a_ten; (void)b_ten;
                }
            }
            // ERA-TOP (Phase 0): any non-binary op with ERA arg0 → ERA
            // Dead gradient branches self-eliminate through ERA propagation.
            if (!is_binary && term_tag(a) == TAG_ERA) RETURN_REDUCED(term_era());
            if (term_tag(a) != TAG_TEN) return t;
            if (is_binary && term_tag(b) != TAG_TEN) return t;

            // Compute ops: return t (stay TAG_TOP). Scheduler rewrites to UOP_FUSING.

            u32 a_id = (u32)term_val(a);
            TensorMeta *ma = &ctx->tensors[a_id];

            // (old defer_all path disabled)
            if (0) {
                u32 b_id = is_binary ? (u32)term_val(b) : 0;
                TensorMeta *mb = is_binary ? &ctx->tensors[b_id] : NULL;
                u32 out_shape[MAX_DIM]; u32 out_ndim;
                if (is_reduce) {
                    out_ndim = ma->view.shape.rank;
                    for (u32 i = 0; i < out_ndim; i++) out_shape[i] = ma->view.shape.dims[i];
                    Term sa = heap_read(ctx, loc + 1);
                    if (term_tag(sa) == TAG_TEN) {
                        u32 ax_id = (u32)term_val(sa);
                        TensorMeta *axt = &ctx->tensors[ax_id];
                        f32 af[MAX_DIM]; META_READ(axt->backend, axt->buf_id, af, axt->view.numel * 4);
                        for (u32 i = 0; i < axt->view.numel; i++) out_shape[(u32)af[i]] = 1;
                        if (!b_id) b_id = ax_id;
                    } else { for (int d=(int)out_ndim-1;d>=0;d--) if(out_shape[d]>1){out_shape[d]=1;break;} }
                } else if (is_binary) {
                    View av, bv; view_broadcast(&ma->view, &mb->view, &av, &bv, out_shape, &out_ndim);
                } else if (is_movement && term_tag(b)==TAG_TEN) {
                    u32 bv_id = (u32)term_val(b); b_id = bv_id;
                    TensorMeta *bm = &ctx->tensors[bv_id];
                    if (uop==UOP_PERMUTE) {
                        out_ndim = ma->view.shape.rank;
                        f32 pf[MAX_DIM]; META_READ(bm->backend, bm->buf_id, pf, out_ndim*4);
                        for (u32 i=0;i<out_ndim;i++) out_shape[i]=ma->view.shape.dims[(u32)pf[i]];
                    } else {
                        out_ndim = bm->view.numel;
                        f32 df[MAX_DIM]; META_READ(bm->backend, bm->buf_id, df, out_ndim*4);
                        for (u32 i=0;i<out_ndim;i++) out_shape[i]=(u32)df[i];
                    }
                } else {
                    out_ndim = ma->view.shape.rank;
                    for (u32 i=0;i<out_ndim;i++) out_shape[i]=ma->view.shape.dims[i];
                }
                u32 dst_id = ctx->tensor_count++;
                TensorMeta *md = &ctx->tensors[dst_id];
                memset(md, 0, sizeof(*md));
                md->dtype = ma->dtype; md->refcount = 1; md->backend = ma->backend;
                md->view = view_create(shape_of(out_shape, out_ndim));
                md->creator_op = uop; md->src_ids[0] = a_id; md->src_ids[1] = b_id;
                if (ma->requires_grad || (mb && mb->requires_grad)) md->requires_grad = 1;
                // Track consumers (needed by tensor_materialize for side outputs)
                if (a_id) ctx->tensors[a_id].defer_consumers++;
                if (b_id && is_binary) ctx->tensors[b_id].defer_consumers++;
                ctx->itrs++;
                RETURN_REDUCED(term_ten(dst_id, ma->dtype));
            }

            // Movement ops: create view, share buffer, no compute
            if (is_movement) {
                // b encodes the new shape/args as a tensor with shape data
                u32 b_id = 0;
                if (term_tag(b) == TAG_TEN) b_id = (u32)term_val(b);
                if (b_id) { ENSURE(ctx, b_id); }
                View new_view;
                int needs_materialize = 0;
                switch (uop) {
                    case UOP_RESHAPE: {
                        // b is a 1D tensor whose elements are the new dims
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        Shape ns = {.rank = mb->view.numel};
                        f32 *dims = malloc(ns.rank * sizeof(f32));
                        META_READ(mb->backend, mb->buf_id, dims, ns.rank * sizeof(f32));
                        for (u32 i = 0; i < ns.rank; i++) ns.dims[i] = (u32)dims[i];
                        free(dims);
                        // Pool stride view: numel mismatch (overlapping windows).
                        // Return self unchanged — fuser handles via st_get.
                        { u32 nn = 1; for (u32 i = 0; i < ns.rank; i++) nn *= ns.dims[i];
                          if (nn != ma->view.numel) return t; }

                        // Identity reshape: same shape → keep original view (no contiguify)
                        int same_shape = (ma->view.shape.rank == ns.rank);
                        if (same_shape)
                            for (u32 d = 0; d < ns.rank; d++)
                                if (ma->view.shape.dims[d] != ns.dims[d]) { same_shape = 0; break; }
                        if (same_shape) {
                            new_view = ma->view;
                        } else {
                            new_view = view_reshape(ma->view, ns);
                        }
                        // Materialize only for failed reshapes (invalid strides).
                        int needs_materialize = 0;
                        if (!same_shape && !new_view.contiguous) {
                            int looks_contiguous = 1;
                            i32 exp_st = 1;
                            for (int d2 = (int)ns.rank - 1; d2 >= 0; d2--) {
                                if (ns.dims[d2] > 1 && new_view.strides[d2] != exp_st) { looks_contiguous = 0; break; }
                                exp_st *= (i32)ns.dims[d2];
                            }
                            if (looks_contiguous) needs_materialize = 1;
                        }
                        if (needs_materialize) {
                            ENSURE(ctx, a_id); ma = &ctx->tensors[a_id];
                            u32 numel = ma->view.numel;
                            u32 orig_a_id = a_id;
                            u32 mat_id = tensor_create(ctx, ma->view.shape, ma->dtype);
                            if (ma->backend->contiguify) {
                                ma->backend->contiguify(ctx->tensors[mat_id].buf_id, numel,
                                                         ma->buf_id, &ma->view);
                            } else {
                                // CPU path: strided gather
                                f32 *src = malloc(numel * sizeof(f32));
                                u32 buf_bytes = (ma->view.offset > 0) ? (u32)ma->view.offset : 0;
                                for (u32 d = 0; d < ma->view.shape.rank; d++)
                                    if (ma->view.strides[d] > 0)
                                        buf_bytes += (ma->view.shape.dims[d]-1) * (u32)ma->view.strides[d];
                                buf_bytes += 1;
                                if (buf_bytes == 0) buf_bytes = 1;
                                f32 *raw = malloc(buf_bytes * sizeof(f32));
                                ma->backend->buf_read(ma->buf_id, raw, buf_bytes * sizeof(f32));
                                for (u32 flat = 0; flat < numel; flat++) {
                                    u32 rem = flat; i32 idx = ma->view.offset;
                                    int msk = 0;
                                    for (int d = (int)ma->view.shape.rank - 1; d >= 0; d--) {
                                        u32 coord = rem % ma->view.shape.dims[d];
                                        rem /= ma->view.shape.dims[d];
                                        if (ma->view.has_mask && (coord < ma->view.mask_begin[d] || coord >= ma->view.mask_end[d])) msk = 1;
                                        idx += (i32)coord * (ma->view.strides[d] > 0 ? ma->view.strides[d] : 0);
                                    }
                                    src[flat] = (msk || idx < 0 || (u32)idx >= buf_bytes) ? 0.f : raw[(u32)idx];
                                }
                                free(raw);
                                ctx->tensors[mat_id].backend->buf_write(ctx->tensors[mat_id].buf_id, src, numel * sizeof(f32));
                                free(src);
                            }

                            // Record provenance so GRAD can walk through materialized copies.
                            // Must be unconditional — gradient tensors don't have requires_grad
                            // but GRAD still needs to traverse their provenance chain.
                            {
                                ctx->tensors[mat_id].requires_grad = ctx->tensors[orig_a_id].requires_grad;
                                ctx->tensors[mat_id].creator_op = UOP_RESHAPE;
                                ctx->tensors[mat_id].src_ids[0] = orig_a_id;
                                ctx->tensors[mat_id].src_ids[1] = b_id;
                            }
                            a_id = mat_id;
                            ma = &ctx->tensors[a_id];
                            new_view = view_reshape(ma->view, ns);
                        }
                        break;
                    }
                    case UOP_EXPAND: {
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        Shape ns = {.rank = mb->view.numel};
                        f32 *dims = malloc(ns.rank * sizeof(f32));
                        META_READ(mb->backend, mb->buf_id, dims, ns.rank * sizeof(f32));
                        for (u32 i = 0; i < ns.rank; i++) ns.dims[i] = (u32)dims[i];
                        free(dims);
                        // Guard: GRAD backward may create rank-mismatched expands
                        if (ns.rank != ma->view.shape.rank) return t;
                        new_view = view_expand(ma->view, ns);
                        break;
                    }
                    case UOP_PERMUTE: {
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        u32 rank = mb->view.numel;
                        f32 *axes_f = malloc(rank * sizeof(f32));
                        META_READ(mb->backend, mb->buf_id, axes_f, rank * sizeof(f32));
                        u32 axes[MAX_DIM];
                        for (u32 i = 0; i < rank; i++) axes[i] = (u32)axes_f[i];
                        free(axes_f);
                        new_view = view_permute(ma->view, axes);
                        break;
                    }
                    case UOP_SHRINK: {
                        // b is a 1D tensor: [start0, end0, start1, end1, ...]
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        u32 n_pairs = mb->view.numel;
                        f32 *pairs;
                        if (mb->host_ptr && n_pairs <= MAX_DIM * 2) {
                            pairs = (f32*)mb->host_ptr; // CPU-cached, no GPU flush
                        } else {
                            pairs = malloc(n_pairs * sizeof(f32));
                            mb->backend->buf_read(mb->buf_id, pairs, n_pairs * sizeof(f32));
                        }
                        u32 starts[MAX_DIM], ends[MAX_DIM];
                        for (u32 i = 0; i < n_pairs / 2; i++) {
                            starts[i] = (u32)pairs[i * 2];
                            ends[i]   = (u32)pairs[i * 2 + 1];
                        }
                        if (pairs != (f32*)mb->host_ptr) free(pairs);
                        new_view = view_shrink(ma->view, starts, ends);
                        break;
                    }
                    case UOP_PAD: {
                        // b is a 1D tensor: [before0, after0, before1, after1, ...]
                        // Zero-copy: use view_pad to set mask on the view
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        u32 n_pairs = mb->view.numel;
                        f32 *pairs = malloc(n_pairs * sizeof(f32));
                        META_READ(mb->backend, mb->buf_id, pairs, n_pairs * sizeof(f32));
                        u32 pad_before[MAX_DIM], pad_after[MAX_DIM];
                        for (u32 i = 0; i < n_pairs / 2; i++) {
                            pad_before[i] = (u32)pairs[i * 2];
                            pad_after[i]  = (u32)pairs[i * 2 + 1];
                        }
                        free(pairs);
                        new_view = view_pad(ma->view, pad_before, pad_after);
                        break;
                    }
                    default:
                        assert(0 && "unknown movement op");
                        new_view = ma->view;
                }
                // Let buf_id=0 propagate from deferred bases through views.
                // tensor_materialize handles the full chain at real boundaries.
                u32 contiguify_buf = 0; // set if RESHAPE created a contiguify buffer
                if (uop == UOP_RESHAPE && needs_materialize && ma->buf_id)
                    contiguify_buf = ma->buf_id;
                u32 dst_id = tensor_view_of(ctx, a_id, new_view);
                // Release contiguify source's extra reference: view_of now holds
                // the sole needed reference. This enables mid-step buffer stealing.
                if (contiguify_buf && ma->backend->buf_decref)
                    ma->backend->buf_decref(contiguify_buf);
                // Always record provenance (needed for materialize + backward)
                {
                    TensorMeta *md = &ctx->tensors[dst_id];
                    md->creator_op = uop;
                    md->src_ids[0] = a_id;
                    md->src_ids[1] = b_id;
                    if (ma->requires_grad) md->requires_grad = 1;
                }
                ctx->itrs++;
                RETURN_REDUCED(term_ten(dst_id, ma->dtype));
            }

            if (!ctx_default_backend(ctx)) return t;

            // ERA propagation for compute ops (Phase 0, ic_native_backprop.md):
            // Dead GRAD branches produce ERA. These must be absorbed:
            //   ADD(ERA, x) → x,  ADD(x, ERA) → x     (identity)
            //   MUL(ERA, x) → ERA, SUB(ERA, x) → ERA   (annihilation)
            //   Unary(ERA) → ERA                        (propagation)
            if (term_tag(a) == TAG_ERA) {
                if (!is_binary) RETURN_REDUCED(term_era());
                if (uop == UOP_ADD) RETURN_REDUCED(b);
                RETURN_REDUCED(term_era());
            }
            if (is_binary && term_tag(b) == TAG_ERA) {
                if (uop == UOP_ADD || uop == UOP_SUB) RETURN_REDUCED(a);
                RETURN_REDUCED(term_era());
            }

            // ALL compute ops stay as TAG_TOP — scheduler rewrites them to UOP_FUSING.
            // Second reduce fires UOP_FUSING → dispatch → TAG_TEN.
            return t;

            // ---- DEAD CODE BELOW (kept for reference during migration) ----
            // For reduces, extract axes tensor ID to enable SUM deferral
            u32 b_id = 0;
            if (is_binary) {
                b_id = (u32)term_val(b);
            } else if (is_reduce) {
                Term sum_arg = heap_read(ctx, loc + 1); // WNF from trampoline
                if (term_tag(sum_arg) == TAG_TEN)
                    b_id = (u32)term_val(sum_arg);
            }
            TensorMeta *mb = is_binary ? &ctx->tensors[b_id] : NULL;


            // Determine output shape
            u32 out_shape[MAX_DIM];
            u32 out_ndim;
            View av_bc, bv_bc;  // broadcast views

            if (uop == UOP_MM) {
                // matmul: [M,K] x [K,N] → [M,N]
                assert(ma->view.shape.rank == 2 && mb->view.shape.rank == 2);
                assert(ma->view.shape.dims[1] == mb->view.shape.dims[0]);
                out_shape[0] = ma->view.shape.dims[0];
                out_shape[1] = mb->view.shape.dims[1];
                out_ndim = 2;
            } else if (is_reduce) {
                // Reduce: check if explicit axes are provided (from thvm_sum_axes)
                out_ndim = ma->view.shape.rank;
                for (u32 i = 0; i < out_ndim; i++) out_shape[i] = ma->view.shape.dims[i];

                Term sum_arg = heap_read(ctx, loc + 1); // WNF from trampoline
                if (term_tag(sum_arg) == TAG_TEN) {
                    // Explicit axes from thvm_sum_axes (already reduced)
                    Term axes_t = sum_arg;
                    {
                        u32 ax_id = (u32)term_val(axes_t);
                        TensorMeta *max_t = &ctx->tensors[ax_id];
                        u32 n_axes = max_t->view.numel;
                        f32 *axes_f = malloc(n_axes * sizeof(f32));
                        META_READ(max_t->backend, max_t->buf_id, axes_f, n_axes * sizeof(f32));
                        for (u32 i = 0; i < n_axes; i++) {
                            u32 ax = (u32)axes_f[i];
                            out_shape[ax] = 1;
                        }
                        free(axes_f);
                    }
                } else {
                    // Old behavior: last non-1 dim
                    int reduce_axis = -1;
                    for (int i = (int)out_ndim - 1; i >= 0; i--) {
                        if (out_shape[i] > 1) { reduce_axis = i; break; }
                    }
                    if (reduce_axis >= 0) {
                        out_shape[reduce_axis] = 1;
                    } else {
                        out_shape[out_ndim - 1] = 1;
                    }
                }
            } else if (is_binary) {
                // Cross-device mismatch check
                assert((!ma->backend || !mb->backend || ma->backend == mb->backend) &&
                       "cross-device op: use thvm_to_device first");
                // Binary: broadcast shapes
                int ok = view_broadcast(&ma->view, &mb->view, &av_bc, &bv_bc, out_shape, &out_ndim);
                if (!ok) {
                    printf("broadcast fail: uop=%u a=[", uop);
                    for (u32 d=0;d<ma->view.shape.rank;d++) printf("%u,",ma->view.shape.dims[d]);
                    printf("] b=[");
                    for (u32 d=0;d<mb->view.shape.rank;d++) printf("%u,",mb->view.shape.dims[d]);
                    printf("]\n");
                }
                assert(ok && "shape broadcast failed");
            } else {
                // Unary: output = input shape
                out_ndim = ma->view.shape.rank;
                for (u32 i = 0; i < out_ndim; i++) out_shape[i] = ma->view.shape.dims[i];
            }

            u32 dst_id;
            TensorMeta *md;

            // Elementwise ops: DEFER dispatch for GPU fusion.
            // All ew ops defer unconditionally — both forward and backward.
            // Backward reduces don't defer (line 1118 guard), so deferred backward
            // ew chains get materialized when the non-deferred reduce EENSUREs them.
            u32 _out_n = 1; for(u32 _d=0;_d<out_ndim;_d++) _out_n*=out_shape[_d];
            (void)_out_n;
            if (is_elementwise(uop) && ma->backend->dispatch_kernel_rs) {
                dst_id = ctx->tensor_count++;
                md = &ctx->tensors[dst_id];
                memset(md, 0, sizeof(*md));
                md->dtype = ma->dtype;
                md->refcount = 1;
                md->backend = ma->backend;
                md->view = view_create(shape_of(out_shape, out_ndim));
                md->creator_op = uop;
                md->src_ids[0] = a_id;
                md->src_ids[1] = b_id;
                md->creator_loc = loc;
                int needs = ma->requires_grad || (mb && mb->requires_grad);
                if (needs) md->requires_grad = 1;
                // Track: mark inputs as consumed by a deferred op
                if (a_id) ctx->tensors[a_id].defer_consumers++;
                if (b_id) ctx->tensors[b_id].defer_consumers++;
                ctx->itrs++;
                RETURN_REDUCED(term_ten(dst_id, ma->dtype));
            }

            // Defer reduce when input is deferred ew or view-of-deferred-ew.
            // During backward (no_grad_alloc=1): DON'T defer reduces. Deferred reduces
            // become TAG_TEN with buf_id=0, but the reducer treats TAG_TEN as WNF
            // and never materializes them. This causes zero gradients.
            if (!ctx->no_grad_alloc &&
                ((uop == UOP_SUM && b_id != 0) || uop == UOP_RMAX) &&
                ma->buf_id == 0 && ma->creator_op &&
                (is_elementwise(ma->creator_op) ||
                 (is_view_op(ma->creator_op) && ma->src_ids[0] &&
                  ctx->tensors[ma->src_ids[0]].buf_id == 0 &&
                  is_elementwise(ctx->tensors[ma->src_ids[0]].creator_op))) &&
                ma->defer_consumers == 0 &&
                ma->backend->dispatch_kernel_rs) {
                dst_id = ctx->tensor_count++;
                md = &ctx->tensors[dst_id];
                memset(md, 0, sizeof(*md));
                md->dtype = ma->dtype;
                md->refcount = 1;
                md->backend = ma->backend;
                md->view = view_create(shape_of(out_shape, out_ndim));
                md->creator_op = uop;
                md->src_ids[0] = a_id;
                md->src_ids[1] = b_id;
                md->creator_loc = loc;
                if (ma->requires_grad) md->requires_grad = 1;
                ma->defer_consumers++;
                ctx->itrs++;
                RETURN_REDUCED(term_ten(dst_id, ma->dtype));
            }

            dst_id = tensor_create(ctx, shape_of(out_shape, out_ndim), ma->dtype);
            md = &ctx->tensors[dst_id];

            // Record provenance for autograd
            {
                int needs = ma->requires_grad || (mb && mb->requires_grad);
                if (needs) {
                    md->requires_grad = 1;
                    md->creator_op = uop;
                    md->src_ids[0] = a_id;
                    md->src_ids[1] = b_id;
                    md->creator_loc = loc;
                }
            }

            // View-through fusion: SUM(PERMUTE(ew_chain), axes)
            // Walk through PERMUTE, permute leaf views to the PERMUTED index space,
            // then fuse reduce+ew in one kernel. Correct because PERMUTE only
            // rearranges strides — same data access, just different iteration order.
            if (is_reduce && ma->buf_id == 0 && ma->creator_op == UOP_PERMUTE &&
                ma->backend && ma->backend->dispatch_kernel_rs) {
                u32 perm_base_id = ma->src_ids[0];
                TensorMeta *perm_base = &ctx->tensors[perm_base_id];
                u32 perm_tid = ma->src_ids[1];
                if (perm_base->buf_id == 0 && perm_base->creator_op &&
                    is_elementwise(perm_base->creator_op) && perm_tid) {
                    ENSURE(ctx, perm_tid);
                    TensorMeta *pt = &ctx->tensors[perm_tid];
                    u32 rank = pt->view.numel;
                    f32 pf[MAX_DIM]; META_READ(pt->backend, pt->buf_id, pf, rank * 4);
                    u32 perm[MAX_DIM];
                    for (u32 i = 0; i < rank; i++) perm[i] = (u32)pf[i];

                    FusedOp fops[32]; u32 fn = 0, ftids[32];
                    u32 flids[16]; const View *flvs[16]; u32 fnl = 0;
                    int fr = materialize_walk(ctx, perm_base_id, fops, &fn, ftids, flids, flvs, &fnl);
                    if (fr >= 0 && fn > 0) {
                        // Permute all leaf views into the PERMUTED index space
                        View pviews[16]; const View *pvptrs[16];
                        for (u32 i = 0; i < fnl; i++) {
                            pviews[i] = view_permute(*flvs[i], perm);
                            pvptrs[i] = &pviews[i];
                        }
                        for (u32 i = 0; i < fn; i++) {
                            if (fops[i].arg_a >= 16) fops[i].arg_a = fnl + (fops[i].arg_a - 16);
                            if (fops[i].arg_b >= 16) fops[i].arg_b = fnl + (fops[i].arg_b - 16);
                        }
                        // ReduceSpec in PERMUTED space (no axis remapping)
                        ReduceSpec ers = {0}; ers.reduce_type = uop;
                        Term sum_arg_e = heap_read(ctx, loc + 1);
                        if (term_tag(sum_arg_e) == TAG_TEN || term_tag(sum_arg_e) == TAG_TOP) {
                            Term axes_e = sum_arg_e; // WNF from trampoline
                            if (term_tag(axes_e) == TAG_TEN) {
                                u32 axid = (u32)term_val(axes_e); ENSURE(ctx, axid);
                                TensorMeta *axt = &ctx->tensors[axid];
                                f32 af[MAX_DIM]; META_READ(axt->backend, axt->buf_id, af, axt->view.numel*4);
                                for (u32 i = 0; i < axt->view.numel; i++) {
                                    u32 ax = (u32)af[i]; if (ax < rank) ers.is_reduce[ax] = 1;
                                }
                            }
                        } else {
                            for (int d = (int)rank - 1; d >= 0; d--)
                                if (ma->view.shape.dims[d] > 1) { ers.is_reduce[d] = 1; break; }
                        }
                        u32 sbufs[8], sops[8]; u32 ns = 0;
                        for (u32 i = 0; i < fn && ns < 8; i++) {
                            TensorMeta *sm = &ctx->tensors[ftids[i]];
                            if (sm->buf_id != 0 && sm->defer_consumers > 0) {
                                sbufs[ns] = sm->buf_id; sops[ns] = fnl + i; ns++;
                            }
                        }
                        u32 fbufs[16];
                        for (u32 i = 0; i < fnl; i++) fbufs[i] = ctx->tensors[flids[i]].buf_id;
                        Shape pshape = ma->view.shape;
                        perm_base->backend->dispatch_kernel_rs(md->buf_id, fbufs, pvptrs, fnl,
                            fops, fn, &pshape, &ers, ns ? sbufs : NULL, ns ? sops : NULL, ns);
                        ctx->itrs++;
                        RETURN_REDUCED(term_ten(dst_id, ma->dtype));
                    }
                }
            }

            // View-through fusion: SUM(view*(ew_chain), axes)
            // Walk through chains of view ops to find a deferred ew base.
            if (is_reduce && ma->buf_id == 0 && is_view_op(ma->creator_op) &&
                ma->creator_op != UOP_PERMUTE && // permute handled above
                ma->backend && ma->backend->dispatch_kernel_rs) {
                // Walk through view ops to find ew base
                u32 view_base_id = ma->src_ids[0];
                for (u32 _vd = 0; _vd < 5; _vd++) {
                    TensorMeta *vt = &ctx->tensors[view_base_id];
                    if (vt->buf_id != 0 || !vt->creator_op) break;
                    if (is_elementwise(vt->creator_op)) break; // found ew
                    if (is_view_op(vt->creator_op) && vt->creator_op != UOP_PERMUTE)
                        view_base_id = vt->src_ids[0]; // follow view chain
                    else break;
                }
                TensorMeta *vb = &ctx->tensors[view_base_id];
                if (vb->buf_id == 0 && vb->creator_op &&
                    is_elementwise(vb->creator_op)) {
                    FusedOp fops[32]; u32 fn = 0, ftids[32];
                    u32 flids[16]; const View *flvs[16]; u32 fnl = 0;
                    int fr = materialize_walk(ctx, view_base_id, fops, &fn, ftids, flids, flvs, &fnl);
                    if (fr >= 0 && fn > 0) {
                        for (u32 i = 0; i < fn; i++) {
                            if (fops[i].arg_a >= 16) fops[i].arg_a = fnl + (fops[i].arg_a - 16);
                            if (fops[i].arg_b >= 16) fops[i].arg_b = fnl + (fops[i].arg_b - 16);
                        }
                        ReduceSpec ers = {0}; ers.reduce_type = uop;
                        Term sum_arg_e = heap_read(ctx, loc + 1);
                        if (term_tag(sum_arg_e) == TAG_TEN || term_tag(sum_arg_e) == TAG_TOP) {
                            Term axes_e = sum_arg_e; // WNF from trampoline
                            if (term_tag(axes_e) == TAG_TEN) {
                                u32 axid = (u32)term_val(axes_e); ENSURE(ctx, axid);
                                TensorMeta *axt = &ctx->tensors[axid];
                                f32 af[MAX_DIM]; META_READ(axt->backend, axt->buf_id, af, axt->view.numel*4);
                                for (u32 i = 0; i < axt->view.numel; i++) {
                                    u32 ax = (u32)af[i]; if (ax < ma->view.shape.rank) ers.is_reduce[ax] = 1;
                                }
                            }
                        } else {
                            for (int d = (int)ma->view.shape.rank - 1; d >= 0; d--)
                                if (ma->view.shape.dims[d] > 1) { ers.is_reduce[d] = 1; break; }
                        }
                        u32 sbufs[8], sops[8]; u32 ns = 0;
                        for (u32 i = 0; i < fn && ns < 8; i++) {
                            TensorMeta *sm = &ctx->tensors[ftids[i]];
                            if (sm->buf_id != 0 && sm->defer_consumers > 0) {
                                sbufs[ns] = sm->buf_id; sops[ns] = fnl + i; ns++;
                            }
                        }
                        u32 fbufs[16];
                        for (u32 i = 0; i < fnl; i++) fbufs[i] = ctx->tensors[flids[i]].buf_id;
                        // full_shape = reshaped shape (ma->view.shape), leaves in original shape
                        Shape rshape = ma->view.shape;
                        vb->backend->dispatch_kernel_rs(md->buf_id, fbufs, flvs, fnl,
                            fops, fn, &rshape, &ers, ns ? sbufs : NULL, ns ? sops : NULL, ns);
                        ctx->itrs++;
                        RETURN_REDUCED(term_ten(dst_id, ma->dtype));
                    }
                }
            }

            // Eager reduce fusion: if reduce input is deferred ew, fuse at dispatch time.
            if (is_reduce && ma->buf_id == 0 && ma->creator_op &&
                is_elementwise(ma->creator_op)) {
                // Build ReduceSpec
                ReduceSpec ers = {0}; ers.reduce_type = uop;
                Term sum_arg_e = heap_read(ctx, loc + 1);
                if (term_tag(sum_arg_e) == TAG_TEN || term_tag(sum_arg_e) == TAG_TOP) {
                    Term axes_e = sum_arg_e; // WNF from trampoline
                    if (term_tag(axes_e) == TAG_TEN) {
                        u32 axid = (u32)term_val(axes_e); ENSURE(ctx, axid);
                        TensorMeta *axt = &ctx->tensors[axid];
                        f32 af[MAX_DIM]; META_READ(axt->backend, axt->buf_id, af, axt->view.numel*4);
                        for (u32 i=0;i<axt->view.numel;i++) {
                            u32 ax=(u32)af[i]; if(ax<ma->view.shape.rank) ers.is_reduce[ax]=1;
                        }
                    }
                } else {
                    for (int d=(int)ma->view.shape.rank-1;d>=0;d--)
                        if(ma->view.shape.dims[d]>1){ers.is_reduce[d]=1;break;}
                }
                if (tensor_materialize_reduce(ctx, a_id, md->buf_id, &ers)) {
                    ctx->itrs++;
                    RETURN_REDUCED(term_ten(dst_id, ma->dtype));
                }
            }

            // Defer MM: create deferred tensor, materialize later.
            // This lets the backward graph build fully before dispatching.
            if (uop == UOP_MM) {
                md->buf_id = 0;
                md->creator_op = UOP_MM;
                md->src_ids[0] = a_id;
                md->src_ids[1] = b_id;
                if (a_id) ctx->tensors[a_id].defer_consumers++;
                if (b_id) ctx->tensors[b_id].defer_consumers++;
                ctx->itrs++;
                RETURN_REDUCED(term_ten(dst_id, ma->dtype));
            }

            // Materialize lazy inputs before dispatch
            {
                ENSURE(ctx, a_id); ma = &ctx->tensors[a_id];
                if (b_id) { ENSURE(ctx, b_id); if (is_binary) mb = &ctx->tensors[b_id]; }
            }
            // Dispatch
            if (is_reduce) {
                // Check if explicit axes are provided
                Term sum_arg2 = heap_read(ctx, loc + 1);
                int has_explicit_axes = (term_tag(sum_arg2) == TAG_TOP || term_tag(sum_arg2) == TAG_TEN);
                u32 n_explicit = 0;
                u32 explicit_axes[MAX_DIM];

                if (has_explicit_axes) {
                    Term axes_t2 = sum_arg2; // WNF from trampoline
                    if (term_tag(axes_t2) == TAG_TEN) {
                        u32 ax_id = (u32)term_val(axes_t2);
                        TensorMeta *axt = &ctx->tensors[ax_id];
                        n_explicit = axt->view.numel;
                        f32 *af = malloc(n_explicit * sizeof(f32));
                        META_READ(axt->backend, axt->buf_id, af, n_explicit * sizeof(f32));
                        for (u32 i = 0; i < n_explicit; i++) explicit_axes[i] = (u32)af[i];
                        free(af);
                    }
                }

                if (n_explicit > 0) {
                    // Multi-axis reduce via GPU: SUM(x, axes) = mul_reduce_sum(x, 1, axes)
                    u32 rank = ma->view.shape.rank;
                    u32 out_numel = md->view.numel;

                    u8 is_ra[MAX_DIM] = {0};
                    for (u32 i = 0; i < n_explicit; i++) is_ra[explicit_axes[i]] = 1;

                    // Build reduce dims and strides for mul_reduce_sum
                    u32 rdims[MAX_DIM], rstrides_a[MAX_DIM], rstrides_ones[MAX_DIM];
                    u32 rn = 0;
                    for (u32 d = 0; d < rank; d++) {
                        if (is_ra[d]) {
                            rdims[rn] = ma->view.shape.dims[d];
                            rstrides_a[rn] = (u32)(ma->view.strides[d] > 0 ? ma->view.strides[d] : 0);
                            rstrides_ones[rn] = 0;
                            rn++;
                        }
                    }

                    if (ma->backend->dispatch_kernel_rs) {
                        ReduceSpec rs2 = {0};
                        rs2.reduce_type = uop;
                        for (u32 d = 0; d < rank; d++)
                            if (is_ra[d]) rs2.is_reduce[d] = 1;
                        u32 leaf_bufs2[] = { ma->buf_id };
                        const View *leaf_views2[] = { &ma->view };
                        ma->backend->dispatch_kernel_rs(md->buf_id, leaf_bufs2, leaf_views2, 1,
                                                         NULL, 0, &ma->view.shape, &rs2, NULL, NULL, 0);
                    } else {
                        // CPU fallback: strided iteration
                        u32 in_numel = ma->view.numel;
                        u32 max_buf_idx = (u32)ma->view.offset;
                        for (u32 d = 0; d < rank; d++)
                            if (ma->view.strides[d] > 0)
                                max_buf_idx += (ma->view.shape.dims[d]-1) * (u32)ma->view.strides[d];
                        f32 *raw = malloc((max_buf_idx+1) * sizeof(f32));
                        if (ma->backend->end_batch) ma->backend->end_batch();
                        ma->backend->buf_read(ma->buf_id, raw, (u64)(max_buf_idx+1)*sizeof(f32));
                        if (ma->backend->begin_batch) ma->backend->begin_batch();
                        f32 *in_data = malloc(in_numel * sizeof(f32));
                        for (u32 flat = 0; flat < in_numel; flat++) {
                            u32 rem = flat, phys = (u32)ma->view.offset;
                            for (int d = (int)rank - 1; d >= 0; d--) {
                                u32 c = rem % ma->view.shape.dims[d]; rem /= ma->view.shape.dims[d];
                                phys += c * (u32)(ma->view.strides[d] > 0 ? ma->view.strides[d] : 0);
                            }
                            in_data[flat] = raw[phys];
                        }
                        free(raw);
                        f32 *out_data = malloc(out_numel * sizeof(f32));
                        for (u32 _oi = 0; _oi < out_numel; _oi++)
                            out_data[_oi] = (uop == UOP_RMAX) ? -1e30f : 0.0f;
                        u32 out_strides[MAX_DIM];
                        out_strides[rank-1] = 1;
                        for (int d = (int)rank - 2; d >= 0; d--)
                            out_strides[d] = out_strides[d+1] * out_shape[d+1];
                        for (u32 flat = 0; flat < in_numel; flat++) {
                            u32 coords[MAX_DIM], rem = flat;
                            for (int d = (int)rank - 1; d >= 0; d--) {
                                coords[d] = rem % ma->view.shape.dims[d]; rem /= ma->view.shape.dims[d];
                            }
                            u32 of = 0;
                            for (u32 d = 0; d < rank; d++) of += (is_ra[d] ? 0 : coords[d]) * out_strides[d];
                            if (uop == UOP_SUM) out_data[of] += in_data[flat];
                            else if (in_data[flat] > out_data[of]) out_data[of] = in_data[flat];
                        }
                        md->backend->buf_write(md->buf_id, out_data, out_numel * sizeof(f32));
                        free(in_data); free(out_data);
                    }
                } else {
                    // Single-axis reduce: last non-1 dim
                    // Single-axis reduce: find the last non-1 dimension
                    u32 reduce_dim = 1;
                    int ra = -1;
                    for (int i = (int)ma->view.shape.rank - 1; i >= 0; i--) {
                        if (ma->view.shape.dims[i] > 1 && ra < 0) ra = i;
                        if (ra >= 0) { reduce_dim *= ma->view.shape.dims[i]; break; }
                    }
                    if (ra < 0) ra = (int)ma->view.shape.rank - 1;

                    if (ma->backend->dispatch_kernel_rs) {
                        ReduceSpec rs = {0};
                        rs.reduce_type = uop;
                        rs.is_reduce[ra] = 1;
                        u32 leaf_bufs[] = { ma->buf_id };
                        const View *leaf_views[] = { &ma->view };
                        ma->backend->dispatch_kernel_rs(md->buf_id, leaf_bufs, leaf_views, 1,
                                                         NULL, 0, &ma->view.shape, &rs, NULL, NULL, 0);
                    } else {
                        // CPU: pre-compiled reduce kernel
                        u32 src_numel = ma->view.numel;
                        u32 use_buf = ma->buf_id;
                        if (!ma->view.contiguous) {
                            if (ma->backend->contiguify) {
                                use_buf = ma->backend->buf_alloc(src_numel * sizeof(f32));
                                ma->backend->contiguify(use_buf, src_numel, ma->buf_id, &ma->view);
                            } else {
                                u32 max_buf_idx = 0;
                                for (u32 d = 0; d < ma->view.shape.rank; d++)
                                    if (ma->view.strides[d] > 0)
                                        max_buf_idx += (ma->view.shape.dims[d]-1) * (u32)ma->view.strides[d];
                                max_buf_idx += (ma->view.offset > 0) ? (u32)ma->view.offset : 0;
                                f32 *raw = malloc((max_buf_idx+1) * sizeof(f32));
                                ma->backend->buf_read(ma->buf_id, raw, (u64)(max_buf_idx+1)*sizeof(f32));
                                f32 *mat_src = malloc(src_numel * sizeof(f32));
                                for (u32 flat = 0; flat < src_numel; flat++) {
                                    u32 rem = flat; i32 phys = ma->view.offset; int msk = 0;
                                    for (int d = (int)ma->view.shape.rank - 1; d >= 0; d--) {
                                        u32 c = rem % ma->view.shape.dims[d]; rem /= ma->view.shape.dims[d];
                                        if (ma->view.has_mask && (c < ma->view.mask_begin[d] || c >= ma->view.mask_end[d])) msk = 1;
                                        phys += (i32)c * (ma->view.strides[d] > 0 ? ma->view.strides[d] : 0);
                                    }
                                    mat_src[flat] = (msk || phys < 0 || (u32)phys > max_buf_idx) ? 0.f : raw[(u32)phys];
                                }
                                free(raw);
                                use_buf = ma->backend->buf_alloc(src_numel * sizeof(f32));
                                ma->backend->buf_write(use_buf, mat_src, src_numel * sizeof(f32));
                                free(mat_src);
                            }
                        }
                        md->backend->op_reduce(uop, md->buf_id, md->view.numel,
                                                use_buf, src_numel, reduce_dim);
                        if (use_buf != ma->buf_id) ma->backend->buf_free(use_buf);
                    }
                }
            } else if (is_binary) {
                md->backend->op_binary(uop, md->buf_id, &md->view,
                                    ma->buf_id, &av_bc,
                                    ctx->tensors[b_id].buf_id, &bv_bc);
            } else {
                md->backend->op_unary(uop, md->buf_id, &md->view,
                                   ma->buf_id, &ma->view);
            }

            ctx->itrs++;
            RETURN_REDUCED(term_ten(dst_id, ma->dtype));
