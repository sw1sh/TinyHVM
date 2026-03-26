// Read small metadata (axes, shapes, pad specs) without GPU flush.
// These are CPU-written and never GPU-modified.
#ifdef __APPLE__
#define META_READ(ctx, buf_id, out, bytes) \
    ((ctx)->backend == &metal_backend ? \
     metal_buf_read_nosync((buf_id), (out), (bytes)) : \
     (ctx)->backend->buf_read((buf_id), (out), (bytes)))
#else
#define META_READ(ctx, buf_id, out, bytes) \
    (ctx)->backend->buf_read((buf_id), (out), (bytes))
#endif

static Term thvm_interact(TinyHVM *ctx, Term t) {
    // If result is TAG_TOP, reduce it before returning (ensures the trampoline
    // drives lazy ops to completion before handing back to the caller).
    #define RETURN_REDUCED(result) do { \
        Term _r = (result); \
        if (term_tag(_r) == TAG_TOP) _r = thvm_reduce(ctx, _r); \
        return _r; \
    } while(0)
    // GRAD iterative step: instead of GRAD_STEP(GRAD3(...)) which recurses
    // O(chain_depth) via thvm_reduce, loop back to inet_step in the same frame.
    // Only works for pure GRAD3 results (not GRAD_ADD which needs reduction first).
    #define GRAD_STEP(result) do { t = (result); goto inet_step; } while(0)

    u32 tag;
inet_step:
    tag = term_tag(t);

    switch (tag) {
        case TAG_TOP: {
            u32 uop = term_ext(t);
            u64 loc = term_val(t);

            // === UOP_GRAD: DUP-op interaction (IC-native backward) ===
            // GRAD(y, grad_y, x) = "given ∂L/∂y = grad_y, compute ∂L/∂x"
            // No reduce — everything lazy. The chain rule produces new TOP nodes.
            if (uop == UOP_GRAD) {
                Term y  = heap_read(ctx, loc);
                Term gy = heap_read(ctx, loc + 1);
                Term x  = heap_read(ctx, loc + 2);

                // Reduce x if lazy (trampoline only auto-reduces slots 0,1; slot 2 stays raw)
                if (term_tag(x) != TAG_TEN && term_tag(x) != TAG_ERA) {
                    x = thvm_reduce(ctx, x);
                    heap_set(ctx, loc + 2, x);  // cache for re-entries
                }

                if (term_tag(y) == TAG_TEN) {
                    u32 y_id = (u32)term_val(y);
                    u32 x_id = (term_tag(x) == TAG_TEN) ? (u32)term_val(x) : ~0u;

                    // Base case: y == x → return grad_y
                    if (term_tag(x) == TAG_TEN && (u32)term_val(x) == y_id)
                        RETURN_REDUCED(thvm_reduce(ctx, gy));

                    TensorMeta *my = &ctx->tensors[y_id];

                    // Leaf (no provenance, not x) → ERA (no gradient path)
                    if (!my->creator_op) {
                        RETURN_REDUCED(term_era());
                    }

                    // DUP-op interaction via provenance
                    u32 cop = my->creator_op;
                    u32 aid = my->src_ids[0], bid = my->src_ids[1];
                    TensorMeta *ma = &ctx->tensors[aid];
                    Term at = term_ten(aid, ma->dtype);

                    int is_bin = (cop==UOP_ADD||cop==UOP_SUB||cop==UOP_MUL||
                                  cop==UOP_DIV||cop==UOP_MAX||cop==UOP_MM||cop==UOP_CMP);
                    TensorMeta *mb_p = is_bin ? &ctx->tensors[bid] : NULL;
                    Term bt = is_bin ? term_ten(bid, mb_p->dtype) : term_era();

                    // REACHES: can tensor `from` reach tensor `target` via src_ids?
                    // Returns 1 if from==target or any provenance ancestor reaches target.
                    // Used to prune dead GRAD branches in binary ops.
                    int a_reaches = 1, b_reaches = 1;
                    if (is_bin && term_tag(x) == TAG_TEN) {
                        // Quick DFS with cycle-safe depth limit
                        #define REACHES_MAX 64
                        u32 x_tgt = (u32)term_val(x);
                        // Check if aid can reach x_tgt
                        { u32 stk_r[REACHES_MAX]; int rp = 0;
                          stk_r[rp++] = aid;
                          a_reaches = 0;
                          while (rp > 0 && rp < REACHES_MAX) {
                              u32 cur = stk_r[--rp];
                              if (cur == x_tgt) { a_reaches = 1; break; }
                              TensorMeta *mc = &ctx->tensors[cur];
                              if (mc->creator_op && mc->src_ids[0] < ctx->tensor_count)
                                  stk_r[rp++] = mc->src_ids[0];
                              if (mc->creator_op && mc->src_ids[1] < ctx->tensor_count &&
                                  mc->src_ids[1] != mc->src_ids[0])
                                  stk_r[rp++] = mc->src_ids[1];
                          }
                        }
                        { u32 stk_r[REACHES_MAX]; int rp = 0;
                          stk_r[rp++] = bid;
                          b_reaches = 0;
                          while (rp > 0 && rp < REACHES_MAX) {
                              u32 cur = stk_r[--rp];
                              if (cur == x_tgt) { b_reaches = 1; break; }
                              TensorMeta *mc = &ctx->tensors[cur];
                              if (mc->creator_op && mc->src_ids[0] < ctx->tensor_count)
                                  stk_r[rp++] = mc->src_ids[0];
                              if (mc->creator_op && mc->src_ids[1] < ctx->tensor_count &&
                                  mc->src_ids[1] != mc->src_ids[0])
                                  stk_r[rp++] = mc->src_ids[1];
                          }
                        }
                        #undef REACHES_MAX
                    }

                    // Helper macro: create recursive GRAD term (3 heap slots)
                    #define GRAD3(y_,gy_,x_) ({ \
                        u64 _l = heap_alloc(ctx, 3); \
                        heap_set(ctx, _l, y_); heap_set(ctx, _l+1, gy_); heap_set(ctx, _l+2, x_); \
                        term_new(TAG_TOP, UOP_GRAD, _l); })
                    // Combine two gradient branches — skip ERA (no-path) operands.
                    // Single live branch → GRAD_STEP (iterative, no depth increase).
                    // Both live → RETURN_REDUCED (must reduce both, adds depth).
                    #define GRAD_COMBINE(ga, gb) do { \
                        Term _ga = (ga), _gb = (gb); \
                        if (term_tag(_ga)==TAG_ERA && term_tag(_gb)==TAG_ERA) \
                            RETURN_REDUCED(term_era()); \
                        if (term_tag(_gb)==TAG_ERA) GRAD_STEP(_ga); \
                        if (term_tag(_ga)==TAG_ERA) GRAD_STEP(_gb); \
                        RETURN_REDUCED(thvm_op(ctx, UOP_ADD, _ga, _gb)); \
                    } while(0)

                    switch (cop) {
                        case UOP_ADD: {
                            Term ga = term_era(), gb = term_era();
                            if (a_reaches) {
                                Term da = sum_to_shape(ctx, gy, my->view.shape, ma->view.shape);
                                ga = GRAD3(at,da,x);
                            }
                            if (b_reaches) {
                                Term db = sum_to_shape(ctx, gy, my->view.shape, mb_p->view.shape);
                                gb = GRAD3(bt,db,x);
                            }
                            GRAD_COMBINE(ga, gb);
                        }
                        case UOP_SUB: {
                            Term ga = term_era(), gb = term_era();
                            if (a_reaches) {
                                Term da = sum_to_shape(ctx, gy, my->view.shape, ma->view.shape);
                                ga = GRAD3(at,da,x);
                            }
                            if (b_reaches) {
                                Term neg = thvm_op(ctx, UOP_NEG,
                                    sum_to_shape(ctx, gy, my->view.shape, mb_p->view.shape), term_era());
                                gb = GRAD3(bt,neg,x);
                            }
                            GRAD_COMBINE(ga, gb);
                        }
                        case UOP_MUL: {
                            Term ga = term_era(), gb = term_era();
                            if (a_reaches) {
                                Term da = sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, bt),
                                                       my->view.shape, ma->view.shape);
                                ga = GRAD3(at,da,x);
                            }
                            if (b_reaches) {
                                Term db = sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, at),
                                                       my->view.shape, mb_p->view.shape);
                                gb = GRAD3(bt,db,x);
                            }
                            GRAD_COMBINE(ga, gb);
                        }
                        case UOP_MM: {
                            Term ga = term_era(), gb = term_era();
                            if (a_reaches) {
                                u32 bt_id = tensor_transpose_2d(ctx, bid);
                                Term da = thvm_op(ctx, UOP_MM, gy, term_ten(bt_id, mb_p->dtype));
                                ga = GRAD3(at,da,x);
                            }
                            if (b_reaches) {
                                u32 at_id = tensor_transpose_2d(ctx, aid);
                                Term db = thvm_op(ctx, UOP_MM, term_ten(at_id, ma->dtype), gy);
                                gb = GRAD3(bt,db,x);
                            }
                            GRAD_COMBINE(ga, gb);
                        }
                        case UOP_RELU: {
                            f32 z = 0.0f;
                            Term mask = thvm_op(ctx, UOP_CMP, at, thvm_tensor(ctx, &z, SHAPE(1)));
                            GRAD_STEP(GRAD3(at, thvm_op(ctx, UOP_MUL, gy, mask), x));
                        }
                        case UOP_NEG:
                            GRAD_STEP(GRAD3(at, thvm_op(ctx, UOP_NEG, gy, term_era()), x));
                        case UOP_EXP:
                            GRAD_STEP(GRAD3(at, thvm_op(ctx, UOP_MUL, gy, y), x));
                        case UOP_LOG:
                            GRAD_STEP(GRAD3(at, thvm_op(ctx, UOP_DIV, gy, at), x));
                        case UOP_SQRT: {
                            f32 two = 2.0f;
                            Term denom = thvm_op(ctx, UOP_MUL, thvm_tensor(ctx, &two, SHAPE(1)), y);
                            GRAD_STEP(GRAD3(at, thvm_op(ctx, UOP_DIV, gy, denom), x));
                        }
                        case UOP_DIV: {
                            Term da = thvm_op(ctx, UOP_DIV, gy, bt);
                            Term ng = thvm_op(ctx, UOP_NEG, gy, term_era());
                            Term db = thvm_op(ctx, UOP_DIV,
                                thvm_op(ctx, UOP_MUL, ng, at),
                                thvm_op(ctx, UOP_MUL, bt, bt));
                            { Term _a = GRAD3(at,da,x), _b = GRAD3(bt,db,x); GRAD_COMBINE(_a, _b); }
                        }
                        case UOP_MAX: {
                            // mask = (a > b): gradient flows to a where a was the max
                            Term mask = thvm_op(ctx, UOP_CMP, at, bt);
                            Term da = sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, mask),
                                                   my->view.shape, ma->view.shape);
                            f32 one = 1.0f;
                            Term inv = thvm_op(ctx, UOP_SUB, thvm_tensor(ctx, &one, SHAPE(1)), mask);
                            Term db = sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, inv),
                                                   my->view.shape, mb_p->view.shape);
                            { Term _a = GRAD3(at,da,x), _b = GRAD3(bt,db,x); GRAD_COMBINE(_a, _b); }
                        }
                        case UOP_FUSING: {
                            // General FUSING backward: re-reduce the original unfused subnet
                            // with provenance recording, then GRAD through the result.
                            u64 orig_loc = ctx->tensors[y_id].fusing_loc;
                            u32 orig_uop = ctx->tensors[y_id].fusing_uop;

                            u8 saved_nf = ctx->no_fuse;
                            ctx->no_fuse = 1; // prevent re-fusing the same pattern

                            Term orig = term_new(TAG_TOP, orig_uop, orig_loc);
                            Term unfused = thvm_reduce(ctx, orig);

                            ctx->no_fuse = saved_nf;

                            // GRAD through the unfused result
                            GRAD_STEP(GRAD3(unfused, gy, x));
                        }
                        case UOP_SUM: {
                            // SUM backward: expand gy to input shape.
                            // expand needs TAG_TEN (rank must match), so reduce gy first.
                            Term gy_r = (term_tag(gy) == TAG_TEN) ? gy : thvm_reduce(ctx, gy);
                            Term g = thvm_expand(ctx, gy_r, ma->view.shape);
                            GRAD_STEP(GRAD3(at, g, x));
                        }

                        case UOP_RMAX: {
                            Term max_bc = thvm_expand(ctx,
                                thvm_reshape(ctx, y, my->view.shape), ma->view.shape);
                            // mask = 1 where input >= max (CMP is strict >, so invert)
                            f32 one = 1.0f;
                            Term mask = thvm_op(ctx, UOP_SUB,
                                thvm_tensor(ctx, &one, SHAPE(1)),
                                thvm_op(ctx, UOP_CMP, max_bc, at));
                            Term gbc = thvm_expand(ctx, gy, ma->view.shape);
                            GRAD_STEP(GRAD3(at, thvm_op(ctx, UOP_MUL, gbc, mask), x));
                        }
                        case UOP_RESHAPE:
                            GRAD_STEP(GRAD3(at, thvm_reshape(ctx, gy, ma->view.shape), x));
                        case UOP_EXPAND:
                            GRAD_STEP(GRAD3(at,
                                sum_to_shape(ctx, gy, my->view.shape, ma->view.shape), x));
                        case UOP_PERMUTE: {
                            u32 rank = ctx->tensors[bid].view.numel;
                            f32 *af = malloc(rank * sizeof(f32));
                            META_READ(ctx, ctx->tensors[bid].buf_id, af, rank*sizeof(f32));
                            u32 inv[MAX_DIM];
                            for (u32 j = 0; j < rank; j++) inv[(u32)af[j]] = j;
                            free(af);
                            GRAD_STEP(GRAD3(at, thvm_permute(ctx, gy, inv, rank), x));
                        }
                        case UOP_PAD: {
                            TensorMeta *mp = &ctx->tensors[bid];
                            u32 nd = mp->view.numel / 2;
                            f32 *pf = malloc(mp->view.numel * sizeof(f32));
                            META_READ(ctx, mp->buf_id, pf, mp->view.numel*sizeof(f32));
                            u32 sp[MAX_DIM*2];
                            for (u32 j=0;j<nd;j++){sp[j*2]=(u32)pf[j*2];sp[j*2+1]=(u32)pf[j*2]+ma->view.shape.dims[j];}
                            free(pf);
                            GRAD_STEP(GRAD3(at, thvm_shrink(ctx, gy, sp, nd), x));
                        }
                        case UOP_SHRINK: {
                            TensorMeta *ms2 = &ctx->tensors[bid];
                            u32 nd = ms2->view.numel / 2;
                            f32 *sf = malloc(ms2->view.numel * sizeof(f32));
                            META_READ(ctx, ms2->buf_id, sf, ms2->view.numel*sizeof(f32));
                            u32 pp[MAX_DIM*2];
                            for (u32 j=0;j<nd;j++){pp[j*2]=(u32)sf[j*2];pp[j*2+1]=ma->view.shape.dims[j]-(u32)sf[j*2+1];}
                            free(sf);
                            GRAD_STEP(GRAD3(at, thvm_pad(ctx, gy, pp, nd), x));
                        }
                        default: {
                            // Unknown op → zero
                            u32 zid = tensor_fill(ctx, my->view.shape, 0.0f);
                            RETURN_REDUCED(term_ten(zid, my->dtype));
                        }
                    }
                    #undef GRAD3
                    #undef GRAD_COMBINE
                }
                // y is not TAG_TEN (still lazy or ERA) → reduce y first, then retry
                if (term_tag(y) == TAG_TOP) {
                    Term yr = thvm_reduce(ctx, y);
                    heap_set(ctx, loc, yr);
                    return thvm_reduce(ctx, t); // retry with reduced y
                }
                RETURN_REDUCED(term_era());
            }

            // === Phase 3 inet ops ===

            if (uop == UOP_ASSIGN) {
                // UOP_ASSIGN(dst, src) — reduce src, blit into dst's buffer in-place
                Term dst_t = heap_read(ctx, loc);
                Term src_raw = heap_read(ctx, loc + 1);
                Term src_t = thvm_reduce(ctx, src_raw);
                Term dst_r = thvm_reduce(ctx, dst_t);
                if (term_tag(dst_r) == TAG_TEN && term_tag(src_t) == TAG_TEN) {
                    u32 dst_id = (u32)term_val(dst_r);
                    u32 src_id = (u32)term_val(src_t);
                    if (dst_id != src_id) {
                        // Blit src into dst's buffer — GPU copy via contiguify
                        TensorMeta *md = &ctx->tensors[dst_id];
                        TensorMeta *ms = &ctx->tensors[src_id];
                        #ifdef __APPLE__
                        if (ctx->backend == &metal_backend) {
                            metal_contiguify(md->buf_id, md->view.numel,
                                             ms->buf_id, &ms->view);
                        } else
                        #endif
                        {
                            u64 nbytes = (u64)md->view.numel * sizeof(f32);
                            f32 *src_host = (f32 *)thvm_to_host(ctx, src_t);
                            ctx->backend->buf_write(md->buf_id, src_host, nbytes);
                        }
                        if (md->host_ptr) { free(md->host_ptr); md->host_ptr = NULL; }
                        heap_set(ctx, loc + 1, dst_r);
                    }
                    ctx->itrs++;
                    RETURN_REDUCED(dst_r);
                }
                RETURN_REDUCED(term_era());
            }

            if (uop == UOP_WHERE) {
                // UOP_WHERE(cond_ten, then_ten, else_ten) — elementwise ternary select
                // Per tinyspec §ElementwiseOps: result[i] = A[i] if P[i]!=0 else B[i]
                Term cond_t = thvm_reduce(ctx, heap_read(ctx, loc));
                Term then_t = thvm_reduce(ctx, heap_read(ctx, loc + 1));
                Term else_t = thvm_reduce(ctx, heap_read(ctx, loc + 2));
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
                ctx->backend->buf_read(mc->buf_id, cp, n*sizeof(f32));
                ctx->backend->buf_read(ma_w->buf_id, ap, n*sizeof(f32));
                ctx->backend->buf_read(mb_w->buf_id, bp, n*sizeof(f32));
                for (u32 i = 0; i < n; i++) rp[i] = (cp[i] != 0.0f) ? ap[i] : bp[i];
                free(cp); free(ap); free(bp);
                u32 dst_wid = tensor_create(ctx, ma_w->view.shape, ma_w->dtype);
                ctx->backend->buf_write(ctx->tensors[dst_wid].buf_id, rp, n*sizeof(f32));
                free(rp);
                ctx->itrs++;
                RETURN_REDUCED(term_ten(dst_wid, ma_w->dtype));
            }

            // UOP_IFZ(counter, zero_case, succ_lam)
            // counter==0 → zero_case; counter>0 → APP(succ_lam, TEN(counter-1))
            if (uop == UOP_IFZ) {
                Term ctr = thvm_reduce(ctx, heap_read(ctx, loc));
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
                Term t = thvm_reduce(ctx, heap_read(ctx, loc));
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

            // === FUSED MUL+SUM: pattern match SUM(MUL(a, b)) ===
            // Avoids materializing the huge MUL intermediate buffer.
            if (uop == UOP_SUM && !ctx->no_fuse) {
                Term child = heap_read(ctx, loc);
                if (term_tag(child) == TAG_TOP && term_ext(child) == UOP_MUL) {
                    u64 mul_loc = term_val(child);
                    Term ma_t = thvm_reduce(ctx, heap_read(ctx, mul_loc));
                    Term mb_t = thvm_reduce(ctx, heap_read(ctx, mul_loc + 1));
                    if (term_tag(ma_t) == TAG_TEN && term_tag(mb_t) == TAG_TEN) {
                        u32 ma_id = (u32)term_val(ma_t);
                        u32 mb_id = (u32)term_val(mb_t);
                        TensorMeta *ma = &ctx->tensors[ma_id];
                        TensorMeta *mb = &ctx->tensors[mb_id];
                        // Always fuse — GRAD will walk back through the original subnet.
                        // The fused output gets creator_op=UOP_FUSING and the SUM's
                        // heap loc in src_ids so the GRAD handler can re-read a and b.
                        View av_bc, bv_bc;
                        u32 out_shape[MAX_DIM];
                        u32 out_ndim;
                        int ok = view_broadcast(&ma->view, &mb->view, &av_bc, &bv_bc,
                                                out_shape, &out_ndim);
                        if (ok) {
                            // Determine reduce axes: explicit (from thvm_sum_axes) or last-non-1
                            u32 n_reduce = 0;
                            u32 reduce_axes[MAX_DIM];
                            Term sum_arg = heap_read(ctx, loc + 1);
                            if (term_tag(sum_arg) == TAG_TOP || term_tag(sum_arg) == TAG_TEN) {
                                // Explicit axes from thvm_sum_axes
                                Term axes_t = thvm_reduce(ctx, sum_arg);
                                if (term_tag(axes_t) == TAG_TEN) {
                                    u32 ax_id = (u32)term_val(axes_t);
                                    TensorMeta *max = &ctx->tensors[ax_id];
                                    n_reduce = max->view.numel;
                                    f32 *axes_f = malloc(n_reduce * sizeof(f32));
                                    META_READ(ctx, max->buf_id, axes_f, n_reduce * sizeof(f32));
                                    for (u32 i = 0; i < n_reduce; i++) reduce_axes[i] = (u32)axes_f[i];
                                    free(axes_f);
                                }
                            } else {
                                // Old behavior: last non-1 dim
                                for (int i = (int)out_ndim - 1; i >= 0; i--) {
                                    if (out_shape[i] > 1) { reduce_axes[0] = (u32)i; n_reduce = 1; break; }
                                }
                            }
                            if (n_reduce > 0) {
                                // Build reduce dim sizes and strides arrays
                                u32 reduce_dims[MAX_DIM];
                                u32 reduce_strides_a[MAX_DIM], reduce_strides_b[MAX_DIM];
                                u32 reduce_numel = 1;
                                for (u32 i = 0; i < n_reduce; i++) {
                                    u32 ax = reduce_axes[i];
                                    reduce_dims[i] = out_shape[ax];
                                    reduce_strides_a[i] = (u32)av_bc.strides[ax];
                                    reduce_strides_b[i] = (u32)bv_bc.strides[ax];
                                    reduce_numel *= out_shape[ax];
                                }

                                // Output shape: broadcast with reduce axes = 1
                                u32 dst_shape[MAX_DIM];
                                for (u32 i = 0; i < out_ndim; i++) dst_shape[i] = out_shape[i];
                                for (u32 i = 0; i < n_reduce; i++) dst_shape[reduce_axes[i]] = 1;
                                Shape dst_s = shape_of(dst_shape, out_ndim);
                                u32 dst_numel = 1;
                                for (u32 i = 0; i < out_ndim; i++) dst_numel *= dst_shape[i];

                                View ov = view_create(dst_s);

                                // Create output tensor
                                u32 dst_id = tensor_create(ctx, dst_s, ma->dtype);
                                TensorMeta *md = &ctx->tensors[dst_id];

                                // Record FUSING provenance: store the SUM heap loc so the
                                // GRAD handler can walk back through the unfused subnet.
                                // loc split across two u32 src_ids (heap fits in 38-bit VAL).
                                if (ma->requires_grad || mb->requires_grad) {
                                    md->requires_grad = 1;
                                    md->creator_op = UOP_FUSING;
                                    md->src_ids[0] = ma_id;  // MUL input a (for REACHES)
                                    md->src_ids[1] = mb_id;  // MUL input b (for REACHES)
                                    md->fusing_loc = loc;    // heap loc of SUM TAG_TOP
                                    md->fusing_uop = UOP_SUM; // outermost fused op
                                }

                                // Dispatch: Metal or CPU fallback
                                if (ctx->backend == &metal_backend) {
                                    metal_mul_reduce_sum(
                                        md->buf_id, dst_numel,
                                        ma->buf_id, &av_bc,
                                        mb->buf_id, &bv_bc,
                                        &ov, n_reduce,
                                        reduce_dims, reduce_strides_a, reduce_strides_b);
                                } else {
                                    // Compute actual buffer footprint (stride-0 broadcasts)
                                    u32 a_buf = ma->view.offset + 1, b_buf = mb->view.offset + 1;
                                    for (u32 d = 0; d < ma->view.shape.rank; d++)
                                        if (ma->view.strides[d] > 0)
                                            a_buf += (ma->view.shape.dims[d]-1) * (u32)ma->view.strides[d];
                                    for (u32 d = 0; d < mb->view.shape.rank; d++)
                                        if (mb->view.strides[d] > 0)
                                            b_buf += (mb->view.shape.dims[d]-1) * (u32)mb->view.strides[d];

                                    f32 *a_ptr = malloc(a_buf * sizeof(f32));
                                    f32 *b_ptr = malloc(b_buf * sizeof(f32));
                                    ctx->backend->buf_read(ma->buf_id, a_ptr, a_buf * sizeof(f32));
                                    ctx->backend->buf_read(mb->buf_id, b_ptr, b_buf * sizeof(f32));
                                    f32 *dst_ptr = calloc(dst_numel, sizeof(f32));
                                    for (u32 o = 0; o < dst_numel; o++) {
                                        u32 coords[MAX_DIM], rem = o;
                                        for (int d = (int)ov.shape.rank - 1; d >= 0; d--) {
                                            coords[d] = rem % ov.shape.dims[d];
                                            rem /= ov.shape.dims[d];
                                        }
                                        u32 ba = (u32)av_bc.offset, bb = (u32)bv_bc.offset;
                                        for (u32 d = 0; d < ov.shape.rank; d++) {
                                            ba += coords[d] * (u32)av_bc.strides[d];
                                            bb += coords[d] * (u32)bv_bc.strides[d];
                                        }
                                        f32 acc = 0.0f;
                                        for (u32 r = 0; r < reduce_numel; r++) {
                                            u32 off_a = 0, off_b = 0, rr = r;
                                            for (u32 ri = 0; ri < n_reduce; ri++) {
                                                u32 rc = rr % reduce_dims[ri];
                                                rr /= reduce_dims[ri];
                                                off_a += rc * reduce_strides_a[ri];
                                                off_b += rc * reduce_strides_b[ri];
                                            }
                                            acc += a_ptr[ba + off_a] * b_ptr[bb + off_b];
                                        }
                                        dst_ptr[o] = acc;
                                    }
                                    ctx->backend->buf_write(md->buf_id, dst_ptr, dst_numel * sizeof(f32));
                                    free(a_ptr); free(b_ptr); free(dst_ptr);
                                }
                                ctx->itrs++;
                                RETURN_REDUCED(term_ten(dst_id, ma->dtype));
                            }
                        } // if (ok)
                    }
                }
            }

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

            // ERA-as-identity for grad domain: ADD/SUB/MUL/DIV with ERA operands
            // ADD(ERA,x)→x, ADD(x,ERA)→x, ADD(ERA,ERA)→ERA, MUL(*,ERA)→ERA
            if (is_binary) {
                Term b_check = heap_read(ctx, loc + 1);
                u8 at = term_tag(a), bt2 = term_tag(b_check);
                int a_era = (at == TAG_ERA), b_era = (bt2 == TAG_ERA);
                int a_ten = (at == TAG_TEN), b_ten = (bt2 == TAG_TEN);
                if (a_era || b_era) {
                    // Both ERA → ERA
                    if (a_era && b_era) RETURN_REDUCED(term_era());
                    // ERA+TEN or TEN+ERA
                    if (uop == UOP_ADD) RETURN_REDUCED(a_era ? b_check : a);
                    if (uop == UOP_SUB) RETURN_REDUCED(a_era ? term_era() : a); // ERA-x=ERA, x-ERA=x
                    if (uop == UOP_MUL || uop == UOP_DIV) RETURN_REDUCED(term_era());
                    (void)a_ten; (void)b_ten;
                }
            }
            if (term_tag(a) != TAG_TEN) return t;
            if (is_binary && term_tag(b) != TAG_TEN) return t;

            u32 a_id = (u32)term_val(a);
            TensorMeta *ma = &ctx->tensors[a_id];

            // Movement ops: create view, share buffer, no compute
            if (is_movement) {
                // b encodes the new shape/args as a tensor with shape data
                u32 b_id = 0;
                if (term_tag(b) == TAG_TEN) b_id = (u32)term_val(b);
                View new_view;
                switch (uop) {
                    case UOP_RESHAPE: {
                        // b is a 1D tensor whose elements are the new dims
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        Shape ns = {.rank = mb->view.numel};
                        f32 *dims = malloc(ns.rank * sizeof(f32));
                        META_READ(ctx, mb->buf_id, dims, ns.rank * sizeof(f32));
                        for (u32 i = 0; i < ns.rank; i++) ns.dims[i] = (u32)dims[i];
                        free(dims);

                        // If input is non-contiguous (e.g. from expand with stride-0),
                        // we must materialize before reshape. Otherwise the new view's
                        // contiguous strides will index beyond the 1-element buffer.
                        new_view = view_reshape(ma->view, ns);
                        if (!new_view.contiguous) {
                            // Materialize non-contiguous view to contiguous buffer
                            u32 numel = ma->view.numel;
                            u32 orig_a_id = a_id;
                            u32 mat_id = tensor_create(ctx, ma->view.shape, ma->dtype);

                            #ifdef __APPLE__
                            if (ctx->backend == &metal_backend) {
                                // GPU path: strided copy via fused identity kernel
                                metal_contiguify(ctx->tensors[mat_id].buf_id, numel,
                                                 ma->buf_id, &ma->view);
                            } else
                            #endif
                            {
                                // CPU path: strided gather
                                f32 *src = malloc(numel * sizeof(f32));
                                u32 buf_bytes = (ma->view.offset > 0) ? (u32)ma->view.offset : 0;
                                for (u32 d = 0; d < ma->view.shape.rank; d++)
                                    if (ma->view.strides[d] > 0)
                                        buf_bytes += (ma->view.shape.dims[d]-1) * (u32)ma->view.strides[d];
                                buf_bytes += 1;
                                if (buf_bytes == 0) buf_bytes = 1;
                                f32 *raw = malloc(buf_bytes * sizeof(f32));
                                ctx->backend->buf_read(ma->buf_id, raw, buf_bytes * sizeof(f32));
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
                                ctx->backend->buf_write(ctx->tensors[mat_id].buf_id, src, numel * sizeof(f32));
                                free(src);
                            }

                            // Record provenance for backward chain
                            if (ctx->tensors[orig_a_id].requires_grad) {
                                ctx->tensors[mat_id].requires_grad = 1;
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
                        META_READ(ctx, mb->buf_id, dims, ns.rank * sizeof(f32));
                        for (u32 i = 0; i < ns.rank; i++) ns.dims[i] = (u32)dims[i];
                        free(dims);
                        new_view = view_expand(ma->view, ns);
                        break;
                    }
                    case UOP_PERMUTE: {
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        u32 rank = mb->view.numel;
                        f32 *axes_f = malloc(rank * sizeof(f32));
                        META_READ(ctx, mb->buf_id, axes_f, rank * sizeof(f32));
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
                        f32 *pairs = malloc(n_pairs * sizeof(f32));
                        META_READ(ctx, mb->buf_id, pairs, n_pairs * sizeof(f32));
                        u32 starts[MAX_DIM], ends[MAX_DIM];
                        for (u32 i = 0; i < n_pairs / 2; i++) {
                            starts[i] = (u32)pairs[i * 2];
                            ends[i]   = (u32)pairs[i * 2 + 1];
                        }
                        free(pairs);
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
                        META_READ(ctx, mb->buf_id, pairs, n_pairs * sizeof(f32));
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
                u32 dst_id = tensor_view_of(ctx, a_id, new_view);
                // Record provenance
                if (ma->requires_grad) {
                    TensorMeta *md = &ctx->tensors[dst_id];
                    md->requires_grad = 1;
                    md->creator_op = uop;
                    md->src_ids[0] = a_id;
                    md->src_ids[1] = b_id;
                }
                ctx->itrs++;
                RETURN_REDUCED(term_ten(dst_id, ma->dtype));
            }

            if (!ctx->backend) return t;

            u32 b_id = is_binary ? (u32)term_val(b) : 0;
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

                Term sum_arg = heap_read(ctx, loc + 1);
                if (term_tag(sum_arg) == TAG_TOP || term_tag(sum_arg) == TAG_TEN) {
                    // Explicit axes from thvm_sum_axes
                    Term axes_t = thvm_reduce(ctx, sum_arg);
                    if (term_tag(axes_t) == TAG_TEN) {
                        u32 ax_id = (u32)term_val(axes_t);
                        TensorMeta *max_t = &ctx->tensors[ax_id];
                        u32 n_axes = max_t->view.numel;
                        f32 *axes_f = malloc(n_axes * sizeof(f32));
                        META_READ(ctx, max_t->buf_id, axes_f, n_axes * sizeof(f32));
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

            u32 dst_id = tensor_create(ctx, shape_of(out_shape, out_ndim), ma->dtype);
            TensorMeta *md = &ctx->tensors[dst_id];

            // Record provenance for autograd
            {
                int needs = ma->requires_grad || (mb && mb->requires_grad);
                if (needs) {
                    md->requires_grad = 1;
                    md->creator_op = uop;
                    md->src_ids[0] = a_id;
                    md->src_ids[1] = b_id;
                }
            }

            // Dispatch
            if (uop == UOP_MM) {
                u32 M = ma->view.shape.dims[0], K = ma->view.shape.dims[1], N = mb->view.shape.dims[1];
                ctx->backend->op_mm(md->buf_id, ma->buf_id, &ma->view,
                                mb->buf_id, &mb->view, M, K, N);
            } else if (is_reduce) {
                // Check if explicit axes are provided
                Term sum_arg2 = heap_read(ctx, loc + 1);
                int has_explicit_axes = (term_tag(sum_arg2) == TAG_TOP || term_tag(sum_arg2) == TAG_TEN);
                u32 n_explicit = 0;
                u32 explicit_axes[MAX_DIM];

                if (has_explicit_axes) {
                    Term axes_t2 = thvm_reduce(ctx, sum_arg2);
                    if (term_tag(axes_t2) == TAG_TEN) {
                        u32 ax_id = (u32)term_val(axes_t2);
                        TensorMeta *axt = &ctx->tensors[ax_id];
                        n_explicit = axt->view.numel;
                        f32 *af = malloc(n_explicit * sizeof(f32));
                        META_READ(ctx, axt->buf_id, af, n_explicit * sizeof(f32));
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

                    #ifdef __APPLE__
                    if (uop == UOP_SUM && ctx->backend == &metal_backend) {
                        // GPU path: mul_reduce_sum(input, ones=1.0, reduce_axes)
                        // Allocate ones_buf each time (freed by pool_reset, reused from free_list)
                        f32 one_val = 1.0f;
                        u32 ones_buf = ctx->backend->buf_alloc(sizeof(f32));
                        ctx->backend->buf_write(ones_buf, &one_val, sizeof(f32));
                        // ones view: same rank as output, all strides=0 (broadcast scalar)
                        View ones_v = md->view;
                        for (u32 d = 0; d < ones_v.shape.rank; d++) ones_v.strides[d] = 0;
                        ones_v.offset = 0;
                        ones_v.contiguous = 0;
                        metal_mul_reduce_sum(
                            md->buf_id, out_numel,
                            ma->buf_id, &ma->view,
                            ones_buf, &ones_v,
                            &md->view, rn, rdims, rstrides_a, rstrides_ones);
                        // Don't free ones_buf here — GPU hasn't executed yet.
                        // It'll be reclaimed by pool_reset.
                    } else
                    #endif
                    {
                        // CPU fallback: strided iteration
                        u32 in_numel = ma->view.numel;
                        u32 max_buf_idx = (u32)ma->view.offset;
                        for (u32 d = 0; d < rank; d++)
                            if (ma->view.strides[d] > 0)
                                max_buf_idx += (ma->view.shape.dims[d]-1) * (u32)ma->view.strides[d];
                        f32 *raw = malloc((max_buf_idx+1) * sizeof(f32));
                        if (ctx->backend->end_batch) ctx->backend->end_batch();
                        ctx->backend->buf_read(ma->buf_id, raw, (u64)(max_buf_idx+1)*sizeof(f32));
                        if (ctx->backend->begin_batch) ctx->backend->begin_batch();
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
                        f32 *out_data = calloc(out_numel, sizeof(f32));
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
                        ctx->backend->buf_write(md->buf_id, out_data, out_numel * sizeof(f32));
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

                    #ifdef __APPLE__
                    if (uop == UOP_SUM && ctx->backend == &metal_backend && !ma->view.contiguous) {
                        // Use mul_reduce_sum(input, ones, reduce_axis) — handles non-contiguous
                        // via ViewParams. Avoids contiguify dispatch.
                        u32 rdims[] = {(u32)reduce_dim};
                        u32 rstrides_a[] = {(u32)(ma->view.strides[ra] > 0 ? ma->view.strides[ra] : 0)};
                        u32 rstrides_ones[] = {0};
                        f32 one_val = 1.0f;
                        u32 ones_buf = ctx->backend->buf_alloc(sizeof(f32));
                        ctx->backend->buf_write(ones_buf, &one_val, sizeof(f32));
                        View ones_v = md->view;
                        for (u32 d2 = 0; d2 < ones_v.shape.rank; d2++) ones_v.strides[d2] = 0;
                        ones_v.offset = 0; ones_v.contiguous = 0;
                        metal_mul_reduce_sum(
                            md->buf_id, md->view.numel,
                            ma->buf_id, &ma->view,
                            ones_buf, &ones_v,
                            &md->view, 1, rdims, rstrides_a, rstrides_ones);
                    } else
                    #endif
                    {
                        // Contiguify if needed, then use simple reduce kernel
                        u32 src_numel = ma->view.numel;
                        u32 use_buf = ma->buf_id;
                        if (!ma->view.contiguous) {
                            #ifdef __APPLE__
                            if (ctx->backend == &metal_backend) {
                                use_buf = ctx->backend->buf_alloc(src_numel * sizeof(f32));
                                metal_contiguify(use_buf, src_numel, ma->buf_id, &ma->view);
                            } else
                            #endif
                            {
                                u32 max_buf_idx = 0;
                                for (u32 d = 0; d < ma->view.shape.rank; d++)
                                    if (ma->view.strides[d] > 0)
                                        max_buf_idx += (ma->view.shape.dims[d]-1) * (u32)ma->view.strides[d];
                                max_buf_idx += (ma->view.offset > 0) ? (u32)ma->view.offset : 0;
                                f32 *raw = malloc((max_buf_idx+1) * sizeof(f32));
                                ctx->backend->buf_read(ma->buf_id, raw, (u64)(max_buf_idx+1)*sizeof(f32));
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
                                use_buf = ctx->backend->buf_alloc(src_numel * sizeof(f32));
                                ctx->backend->buf_write(use_buf, mat_src, src_numel * sizeof(f32));
                                free(mat_src);
                            }
                        }
                        ctx->backend->op_reduce(uop, md->buf_id, md->view.numel,
                                                use_buf, src_numel, reduce_dim);
                        if (use_buf != ma->buf_id) ctx->backend->buf_free(use_buf);
                    }
                }
            } else if (is_binary) {
                ctx->backend->op_binary(uop, md->buf_id, &md->view,
                                    ma->buf_id, &av_bc,
                                    ctx->tensors[b_id].buf_id, &bv_bc);
            } else {
                ctx->backend->op_unary(uop, md->buf_id, &md->view,
                                   ma->buf_id, &ma->view);
            }

            ctx->itrs++;
            RETURN_REDUCED(term_ten(dst_id, ma->dtype));
        }

        // TAG_APP: beta reduction — APP(LAM(x, body), arg) → body[x ← arg]
        // Rule fires as an active pair: APP + LAM → relink ports, continue loop.
        case TAG_APP: {
            u64 loc = term_val(t);
            Term fun = thvm_reduce(ctx, heap_read(ctx, loc));  // reduce fun (not tail)
            heap_set(ctx, loc, fun);
            if (term_tag(fun) == TAG_LAM) {
                u64 lam_loc = term_val(fun);
                Term arg = heap_read(ctx, loc + 1);
                heap_set(ctx, lam_loc, arg);      // link: write arg at var slot
                ctx->itrs++;
                t = heap_read(ctx, lam_loc + 1);  // body is the next term to reduce
                goto inet_step;                   // loop — no recursive call
            }
            // APP-TEN: APP(TEN, x) → x — tensor in function pos, discard and return arg
            // Enables sequencing: APP(ASSIGN(...), continuation) forces ASSIGN, then continues.
            if (term_tag(fun) == TAG_TEN) {
                tensor_decref(ctx, (u32)term_val(fun));
                ctx->itrs++;
                t = heap_read(ctx, loc + 1);
                goto inet_step;
            }
            if (term_tag(fun) == TAG_ERA) {
                ctx->itrs++;
                t = heap_read(ctx, loc + 1);
                goto inet_step;
            }
            return t;
        }

        // TAG_LAM: lambda abstraction — returned as-is until applied
        case TAG_LAM:
            return t;

        // TAG_REF: unfold definition — deep-clone body (ALO semantics).
        // Each unfold gets fresh heap nodes so recursive defs work correctly.
        case TAG_REF: {
            u32 name = (u32)term_ext(t);
            assert(name < ctx->def_count);
            ctx->itrs++;
            t = term_clone(ctx, ctx->defs[name]);
            goto inet_step;
        }

        // TAG_SUP: superposition — returned as-is until projected by DP0/DP1
        case TAG_SUP:
            return t;

        // TAG_DP0/DP1: superposition projection — active pair with SUP.
        // Fire the rule: link to the chosen branch, continue loop.
        case TAG_DP0: {
            u64 loc = term_val(t);
            Term sup = thvm_reduce(ctx, heap_read(ctx, loc));  // reduce the sup (not tail)
            if (term_tag(sup) == TAG_SUP) {
                ctx->itrs++;
                t = heap_read(ctx, term_val(sup));  // link to left branch
                goto inet_step;
            }
            heap_set(ctx, loc, sup);
            return t;
        }

        case TAG_DP1: {
            u64 loc = term_val(t);
            Term sup = thvm_reduce(ctx, heap_read(ctx, loc));
            if (term_tag(sup) == TAG_SUP) {
                ctx->itrs++;
                t = heap_read(ctx, term_val(sup) + 1);  // link to right branch
                goto inet_step;
            }
            heap_set(ctx, loc, sup);
            return t;
        }


        case TAG_OP2: {
            u64 loc = term_val(t);
            u32 opr = term_ext(t);
            Term x = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, x);
            Term y = thvm_reduce(ctx, heap_read(ctx, loc + 1));
            heap_set(ctx, loc + 1, y);
            if (term_tag(x) == TAG_NUM && term_tag(y) == TAG_NUM) {
                u32 xv = term_as_u32(x), yv = term_as_u32(y), r;
                switch (opr) {
                    case 0: r = xv + yv; break;
                    case 1: r = xv - yv; break;
                    case 2: r = xv * yv; break;
                    case 3: r = yv ? xv / yv : 0; break;
                    default: r = 0;
                }
                ctx->itrs++;
                return term_num_u32(r);
            }
            return t;
        }

        case TAG_VAR: {
            u64 loc = term_val(t);
            Term sub = heap_read(ctx, loc);
            if (term_is_sub(sub)) return t;
            return thvm_reduce(ctx, sub);
        }

        default:
            return t;
    }
}

// ============================================================
// thvm_reduce: enter/apply trampoline — HVM wnf/_.c style.
//
// Two phases, mirrored exactly from HVM4:
//
//   ENTER: walk the head, pushing eliminators as frames.
//     TAG_APP  → push frame, enter fun (slot 0)
//     TAG_DP0/DP1 → push frame, enter sup (slot 0)
//     TAG_REF  → unfold definition, go enter
//     TAG_TOP  → push frame, enter arg-slot 0 (strict left arg)
//     WNF (TEN/ERA/NUM/LAM/SUP) → jump to APPLY
//
//   APPLY: we have a WHNF in `whnf`. Pop frames and dispatch:
//     APP frame + LAM whnf  → beta-reduce, goto enter on body
//     APP frame + SUP whnf  → APP-SUP rule, continue apply
//     TAG_TOP frame + TEN whnf → check arg1: if TEN → fire rule
//                              → else push TOP1_TEN frame, enter arg1
//     TOP1_TEN frame + TEN whnf → arg1 ready, fire the rule
//     DP0/DP1 frame + SUP whnf → DUP-SUP annihilate, goto enter
//     DP0/DP1 frame + LAM whnf → DUP-LAM, goto enter
//     (stuck): rebuild term, continue
//
// For tensor ops with 2 required TAG_TEN args this is parallel to
// HVM's OP2 treatment: enter x, then enter y, then fire op(x,y).
// For unary ops (1 TAG_TEN arg), arg1 = term_era() which is WNF,
// so TOP1_TEN fires immediately after checking arg1.
//
// thvm_interact is unchanged: it fires a complete rule given a term
// whose args are ready (used from APPLY). The RETURN_REDUCED inside
// interact functions becomes the single result cache point.
// ============================================================

// Frame tags for staged tensor-op arg resolution (mimic HVM's F_OP2_NUM)
// TAG_TOP1: "arg0 is TEN (in heap[loc+0]), now reducing arg1"
// TAG_TOP2: reserved for 3-arg ops (GRAD etc.) — not yet used.
#define TAG_TOP1 0x7E  // sentinel for "arg0 done, entering arg1"
#define TAG_TOP2 0x7F  // sentinel for "arg1 done, entering arg2" (3-arg ops)

// Frame stack lives on the C stack (via recurse thvm_reduce calls are bounded
// by graph depth, not arity). For the outer-level call we use a heap buffer.
#define FRAME_CAP 65536

