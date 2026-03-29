// Read small metadata (axes, shapes, pad specs) without GPU flush.
// Uses buf_read_nosync if the backend supports it (avoids pipeline stall).
#define META_READ(be, buf_id, out, bytes) \
    ((be)->buf_read_nosync ? \
     (be)->buf_read_nosync((buf_id), (out), (bytes)) : \
     (be)->buf_read((buf_id), (out), (bytes)))

// Forward declarations (defined in rewrite/_.c, included after interact)
static int is_view_op(u32 uop);
static int is_elementwise(u32 uop);

static Term thvm_interact(TinyHVM *ctx, Term t) {
    // Return result to trampoline. Eagerly reduce TAG_TOP so the interaction
    // handler always returns WNF (TAG_TEN/ERA/NUM/etc.), not lazy terms.
    // This is required because GRAD_STEP (goto inet_step) expects resolved args.
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
                // Disable rewrite fusion during backward: let deferred chains grow
                // longer. Without this, rewrite_apply materializes each backward
                // formula immediately (1 dispatch per formula = 82 dispatches).
                // With this, deferred chains accumulate and tensor_materialize
                // fuses them into fewer, larger kernels at ENSURE boundaries.
                u8 _saved_nf = ctx->no_fuse;
                ctx->no_fuse = 1;
                #define GRAD_RETURN(r) do { \
                    Term _gr = (r); \
                    if (term_tag(_gr) == TAG_TOP) _gr = thvm_reduce(ctx, _gr); \
                    ctx->no_fuse = _saved_nf; \
                    return _gr; \
                } while(0)
                Term y  = heap_read(ctx, loc);
                Term gy = heap_read(ctx, loc + 1);
                Term x  = heap_read(ctx, loc + 2);

                // Reduce x if lazy (trampoline only auto-reduces slots 0,1; slot 2 stays raw)
                if (term_tag(x) != TAG_TEN && term_tag(x) != TAG_ERA && term_tag(x) != TAG_CTR) {
                    x = thvm_reduce(ctx, x);
                    heap_set(ctx, loc + 2, x);
                }

                if (term_tag(y) == TAG_TEN) {
                    u32 y_id = (u32)term_val(y);

                    // Base case: y == x → return grad_y
                    if (term_tag(x) == TAG_TEN && (u32)term_val(x) == y_id) {
                        GRAD_RETURN(thvm_reduce(ctx, gy));
                    }

                    // Multi-target: x = TAG_CTR encoding (param, slot) pairs.
                    // When y matches a param, deposit gy via ASSIGN into the slot.
                    if (term_tag(x) == TAG_CTR) {
                        u64 tgt_loc = term_val(x);
                        u32 n_tgt = term_as_u32(heap_read(ctx, tgt_loc));
                        for (u32 _gi = 0; _gi < n_tgt; _gi++) {
                            Term p = heap_read(ctx, tgt_loc + 1 + 2*_gi);
                            if (term_tag(p) == TAG_TEN && (u32)term_val(p) == y_id) {
                                Term slot = heap_read(ctx, tgt_loc + 1 + 2*_gi + 1);
                                Term accum = thvm_op(ctx, UOP_ADD, slot, gy);
                                GRAD_RETURN(thvm_app(ctx,
                                    thvm_assign(ctx, slot, accum), term_era()));
                            }
                        }
                    }

                    TensorMeta *my = &ctx->tensors[y_id];

                    // Leaf (no provenance, not target) → ERA
                    if (!my->creator_op) {
                            GRAD_RETURN(term_era());
                    }

                    // Shared tensor: multiple GRAD visits expected (diamond in DAG).
                    // Park gy, accumulate across visits, walk once with combined gy.
                    if (my->grad_refs > 1) {
                        my->grad_refs--;
                        if (my->grad_cache) {
                            my->grad_cache = (Term)(u64)thvm_op(ctx, UOP_ADD,
                                (Term)my->grad_cache, gy);
                        } else {
                            my->grad_cache = (Term)(u64)gy;
                        }
                        GRAD_RETURN(term_era());
                    }
                    // Last (or only) visit: combine with any parked gradient
                    if (my->grad_cache) {
                        gy = thvm_op(ctx, UOP_ADD, (Term)my->grad_cache, gy);
                        heap_set(ctx, loc + 1, gy);
                        my->grad_cache = 0;
                    }

                    // DUP-op interaction via provenance
                    u32 cop = my->creator_op;
                    u32 aid = my->src_ids[0], bid = my->src_ids[1];

                    // Lazy ENSURE: backward formulas create lazy ops wrapping at/bt.
                    // Only ENSURE where backward reads data: MM, RMAX, LOG, DIV, MAX.
                    TensorMeta *ma = &ctx->tensors[aid];
                    Term at = term_ten(aid, ma->dtype);

                    int is_bin = (cop==UOP_ADD||cop==UOP_SUB||cop==UOP_MUL||
                                  cop==UOP_DIV||cop==UOP_MAX||cop==UOP_MM||cop==UOP_CMP);
                    TensorMeta *mb_p = is_bin ? &ctx->tensors[bid] : NULL;
                    Term bt = is_bin ? term_ten(bid, mb_p->dtype) : term_era();

                    #define GRAD3(y_,gy_,x_) ({ \
                        u64 _l = heap_alloc(ctx, 3); \
                        heap_set(ctx, _l, y_); heap_set(ctx, _l+1, gy_); heap_set(ctx, _l+2, x_); \
                        term_new(TAG_TOP, UOP_GRAD, _l); })
                    // No GRAD3_FWD — gradient accumulation handled by additive
                    // ASSIGN at the base case. Each GRAD path walks independently.
                    #define BIN_GRAD(da_, db_) do { \
                        int _a_live = ma->requires_grad; \
                        int _b_live = mb_p && mb_p->requires_grad; \
                        if (_a_live && _b_live) { \
                            GRAD_RETURN(thvm_op(ctx, UOP_ADD, \
                                GRAD3(at, da_, x), \
                                GRAD3(bt, db_, x))); \
                        } else if (_a_live) { \
                            UN_GRAD(da_); \
                        } else if (_b_live) { \
                            Term _bg = GRAD3(bt, db_, x); \
                            if (term_tag(_bg) == TAG_TOP) GRAD_STEP(_bg); \
                            GRAD_RETURN(_bg); \
                        } else { \
                            GRAD_RETURN(term_era()); \
                        } \
                    } while(0)
                    #define UN_GRAD(da_) do { \
                        Term _ug = GRAD3(at, da_, x); \
                        if (term_tag(_ug) == TAG_TOP) GRAD_STEP(_ug); \
                        GRAD_RETURN(_ug); \
                    } while(0)

                    switch (cop) {
                        case UOP_ADD:
                            BIN_GRAD(
                                sum_to_shape(ctx, gy, my->view.shape, ma->view.shape),
                                sum_to_shape(ctx, gy, my->view.shape, mb_p->view.shape));
                        case UOP_SUB:
                            BIN_GRAD(
                                sum_to_shape(ctx, gy, my->view.shape, ma->view.shape),
                                thvm_op(ctx, UOP_NEG,
                                    sum_to_shape(ctx, gy, my->view.shape, mb_p->view.shape), term_era()));
                        case UOP_MUL:
                            BIN_GRAD(
                                sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, bt), my->view.shape, ma->view.shape),
                                sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, at), my->view.shape, mb_p->view.shape));
                        case UOP_MM: {
                            ENSURE(ctx, aid); ENSURE(ctx, bid);
                            ma = &ctx->tensors[aid]; mb_p = &ctx->tensors[bid];
                            u32 bt_id = tensor_transpose_2d(ctx, bid);
                            u32 at_id = tensor_transpose_2d(ctx, aid);
                            BIN_GRAD(
                                thvm_op(ctx, UOP_MM, gy, term_ten(bt_id, mb_p->dtype)),
                                thvm_op(ctx, UOP_MM, term_ten(at_id, ma->dtype), gy));
                        }
                        case UOP_RELU: {
                            // Use y (output) not at (input) for mask: y>0 iff at>0
                            // This allows RELU to appear in fused reduce chains
                            // (virtual intermediates don't have input data, but y is real)
                            f32 z = 0.0f;
                            Term mask = thvm_op(ctx, UOP_CMP, y, thvm_tensor(ctx, &z, SHAPE(1)));
                            UN_GRAD(thvm_op(ctx, UOP_MUL, gy, mask));
                        }
                        case UOP_NEG:   UN_GRAD(thvm_op(ctx, UOP_NEG, gy, term_era()));
                        case UOP_EXP:   UN_GRAD(thvm_op(ctx, UOP_MUL, gy, y));
                        case UOP_LOG: { ENSURE(ctx, aid); at = term_ten(aid, ctx->tensors[aid].dtype);
                            UN_GRAD(thvm_op(ctx, UOP_DIV, gy, at)); }
                        case UOP_SQRT: {
                            f32 two = 2.0f;
                            Term denom = thvm_op(ctx, UOP_MUL, thvm_tensor(ctx, &two, SHAPE(1)), y);
                            UN_GRAD(thvm_op(ctx, UOP_DIV, gy, denom));
                        }
                        case UOP_DIV: {
                            ENSURE(ctx, aid); ENSURE(ctx, bid);
                            at = term_ten(aid, ctx->tensors[aid].dtype);
                            bt = term_ten(bid, ctx->tensors[bid].dtype);
                            mb_p = &ctx->tensors[bid];
                            Term ng = thvm_op(ctx, UOP_NEG, gy, term_era());
                            BIN_GRAD(
                                thvm_op(ctx, UOP_DIV, gy, bt),
                                thvm_op(ctx, UOP_DIV, thvm_op(ctx, UOP_MUL, ng, at),
                                    thvm_op(ctx, UOP_MUL, bt, bt)));
                        }
                        case UOP_MAX: {
                            ENSURE(ctx, aid); ENSURE(ctx, bid);
                            at = term_ten(aid, ctx->tensors[aid].dtype);
                            bt = term_ten(bid, ctx->tensors[bid].dtype);
                            mb_p = &ctx->tensors[bid];
                            Term mask = thvm_op(ctx, UOP_CMP, at, bt);
                            f32 one = 1.0f;
                            Term inv = thvm_op(ctx, UOP_SUB, thvm_tensor(ctx, &one, SHAPE(1)), mask);
                            BIN_GRAD(
                                sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, mask), my->view.shape, ma->view.shape),
                                sum_to_shape(ctx, thvm_op(ctx, UOP_MUL, gy, inv), my->view.shape, mb_p->view.shape));
                        }
                        case UOP_FUSING: {
                            u64 orig_loc2 = ctx->tensors[y_id].fusing_loc;
                            u32 orig_uop2 = ctx->tensors[y_id].fusing_uop;
                            u8 saved_nf = ctx->no_fuse;
                            ctx->no_fuse = 1;
                            Term orig = term_new(TAG_TOP, orig_uop2, orig_loc2);
                            Term unfused = thvm_reduce(ctx, orig);
                            ctx->no_fuse = saved_nf;
                            GRAD_RETURN(GRAD3(unfused, gy, x));
                        }
                        case UOP_SUM: {
                            Term gy_r = (term_tag(gy) == TAG_TEN) ? gy : thvm_reduce(ctx, gy);
                            gy_r = thvm_reshape(ctx, gy_r, my->view.shape);
                            UN_GRAD(thvm_expand(ctx, gy_r, ma->view.shape));
                        }
                        case UOP_RMAX: {
                            ENSURE(ctx, aid); ma = &ctx->tensors[aid];
                            at = term_ten(aid, ma->dtype);
                            Term max_bc = thvm_expand(ctx,
                                thvm_reshape(ctx, y, my->view.shape), ma->view.shape);
                            f32 one = 1.0f;
                            Term mask = thvm_op(ctx, UOP_SUB,
                                thvm_tensor(ctx, &one, SHAPE(1)),
                                thvm_op(ctx, UOP_CMP, max_bc, at));
                            UN_GRAD(thvm_op(ctx, UOP_MUL, thvm_expand(ctx, gy, ma->view.shape), mask));
                        }
                        case UOP_RESHAPE: {
                            // Force lazy reshape in backward: keeps the chain composable.
                            Shape rs = ma->view.shape;
                            f32 dims[MAX_DIM];
                            for (u32 j = 0; j < rs.rank; j++) dims[j] = (f32)rs.dims[j];
                            Term shape_t = thvm_tensor(ctx, dims, SHAPE(rs.rank));
                            UN_GRAD(thvm_op(ctx, UOP_RESHAPE, gy, shape_t));
                        }
                        case UOP_EXPAND:
                            UN_GRAD(sum_to_shape(ctx, gy, my->view.shape, ma->view.shape));
                        case UOP_PERMUTE: {
                            u32 rank = ctx->tensors[bid].view.numel;
                            f32 *af = malloc(rank * sizeof(f32));
                            META_READ(ctx->tensors[bid].backend, ctx->tensors[bid].buf_id, af, rank*sizeof(f32));
                            u32 inv[MAX_DIM];
                            for (u32 j = 0; j < rank; j++) inv[(u32)af[j]] = j;
                            free(af);
                            // Force lazy permute: keeps the chain composable by the fuser.
                            // Eager permute creates non-contiguous TAG_TEN that forces
                            // downstream RESHAPEs to materialize (contiguify dispatch).
                            // Lazy permute stays TAG_TOP → fuser composes it into the
                            // consumer's kernel → zero extra dispatches.
                            f32 inv_f[MAX_DIM];
                            for (u32 j = 0; j < rank; j++) inv_f[j] = (f32)inv[j];
                            Term inv_t = thvm_tensor(ctx, inv_f, SHAPE(rank));
                            UN_GRAD(thvm_op(ctx, UOP_PERMUTE, gy, inv_t));
                        }
                        case UOP_PAD: {
                            TensorMeta *mp = &ctx->tensors[bid];
                            u32 nd = mp->view.numel / 2;
                            f32 *pf = malloc(mp->view.numel * sizeof(f32));
                            META_READ(mp->backend, mp->buf_id, pf, mp->view.numel*sizeof(f32));
                            u32 sp[MAX_DIM*2];
                            for (u32 j=0;j<nd;j++){sp[j*2]=(u32)pf[j*2];sp[j*2+1]=(u32)pf[j*2]+ma->view.shape.dims[j];}
                            free(pf);
                            UN_GRAD(thvm_shrink(ctx, gy, sp, nd));
                        }
                        case UOP_SHRINK: {
                            TensorMeta *ms2 = &ctx->tensors[bid];
                            u32 nd = ms2->view.numel / 2;
                            f32 *sf = malloc(ms2->view.numel * sizeof(f32));
                            META_READ(ms2->backend, ms2->buf_id, sf, ms2->view.numel*sizeof(f32));
                            u32 pp[MAX_DIM*2];
                            for (u32 j=0;j<nd;j++){pp[j*2]=(u32)sf[j*2];pp[j*2+1]=ma->view.shape.dims[j]-(u32)sf[j*2+1];}
                            free(sf);
                            UN_GRAD(thvm_pad(ctx, gy, pp, nd));
                        }
                        default: {
                            u32 zid = tensor_fill(ctx, my->view.shape, 0.0f);
                            GRAD_RETURN(term_ten(zid, my->dtype));
                        }
                    }
                    #undef GRAD3
                    #undef BIN_GRAD
                    #undef UN_GRAD
                }
                // y is not TAG_TEN (still lazy or ERA) → reduce y first, then retry
                if (term_tag(y) == TAG_TOP) {
                    Term yr = thvm_reduce(ctx, y);
                    heap_set(ctx, loc, yr);
                    return thvm_reduce(ctx, t); // retry with reduced y
                }
                GRAD_RETURN(term_era());
                #undef GRAD_RETURN
            }

            // === Phase 3 inet ops ===

            if (uop == UOP_ASSIGN) {
                    Term _dt=heap_read(ctx,loc); Term _st=heap_read(ctx,loc+1);
                // UOP_ASSIGN(dst, src) — reduce src, blit into dst's buffer in-place
                Term dst_t = heap_read(ctx, loc);
                Term src_raw = heap_read(ctx, loc + 1);
                Term src_t = thvm_reduce(ctx, src_raw);
                Term dst_r = thvm_reduce(ctx, dst_t);
                if (term_tag(dst_r) == TAG_TEN && term_tag(src_t) == TAG_TEN) {
                    u32 dst_id = (u32)term_val(dst_r);
                    u32 src_id = (u32)term_val(src_t);
                    if (dst_id != src_id) {
                        ENSURE(ctx, src_id);
                        TensorMeta *md = &ctx->tensors[dst_id];
                        TensorMeta *ms = &ctx->tensors[src_id];
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
                        if (md->host_ptr) { free(md->host_ptr); md->host_ptr = NULL; }
                        heap_set(ctx, loc + 1, dst_r);
                    }
                    ctx->itrs++;
                    RETURN_REDUCED(dst_r);
                }
                RETURN_REDUCED(term_era());
            }

            if (uop == UOP_TODEVICE) {
                // UOP_TODEVICE(tensor, device_idx_scalar)
                // Read tensor from source backend, write to target backend
                Term src_t = thvm_reduce(ctx, heap_read(ctx, loc));
                Term dev_t = thvm_reduce(ctx, heap_read(ctx, loc + 1));
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
            // ERA-TOP (Phase 0): any non-binary op with ERA arg0 → ERA
            // Dead gradient branches self-eliminate through ERA propagation.
            if (!is_binary && term_tag(a) == TAG_ERA) RETURN_REDUCED(term_era());
            if (term_tag(a) != TAG_TEN) return t;
            if (is_binary && term_tag(b) != TAG_TEN) return t;

            u32 a_id = (u32)term_val(a);
            TensorMeta *ma = &ctx->tensors[a_id];

            // Movement ops: create view, share buffer, no compute
            if (is_movement) {
                // b encodes the new shape/args as a tensor with shape data
                u32 b_id = 0;
                if (term_tag(b) == TAG_TEN) b_id = (u32)term_val(b);
                if (b_id) { ENSURE(ctx, b_id); }
                View new_view;
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

                        // If input is non-contiguous (e.g. from expand with stride-0),
                        // we must materialize before reshape. Otherwise the new view's
                        // contiguous strides will index beyond the 1-element buffer.
                        new_view = view_reshape(ma->view, ns);
                        // Materialize only for failed reshapes (invalid strides).
                        // Masked views with valid strides (from PAD on permuted
                        // gradients) are handled by the codegen inline — the fuser
                        // propagates masks through view composition.
                        int needs_materialize = 0;
                        if (!new_view.contiguous) {
                            // Check if view_reshape produced fallback (contiguous-pattern) strides
                            // This indicates a failed merge-split — strides don't match data layout
                            int looks_contiguous = 1;
                            i32 exp_st = 1;
                            for (int d2 = (int)ns.rank - 1; d2 >= 0; d2--) {
                                if (ns.dims[d2] > 1 && new_view.strides[d2] != exp_st) { looks_contiguous = 0; break; }
                                exp_st *= (i32)ns.dims[d2];
                            }
                            if (looks_contiguous) needs_materialize = 1; // failed reshape fallback
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
                        // Use full buf_read (not META_READ) — pairs may be GPU-computed
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        u32 n_pairs = mb->view.numel;
                        f32 *pairs = malloc(n_pairs * sizeof(f32));
                        mb->backend->buf_read(mb->buf_id, pairs, n_pairs * sizeof(f32));
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
                u32 dst_id = tensor_view_of(ctx, a_id, new_view);
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

            // For reduces, extract axes tensor ID to enable SUM deferral
            u32 b_id = 0;
            if (is_binary) {
                b_id = (u32)term_val(b);
            } else if (is_reduce) {
                Term sum_arg = heap_read(ctx, loc + 1);
                if (term_tag(sum_arg) == TAG_TEN)
                    b_id = (u32)term_val(sum_arg);
                else if (term_tag(sum_arg) == TAG_TOP) {
                    sum_arg = thvm_reduce(ctx, sum_arg);
                    if (term_tag(sum_arg) == TAG_TEN) b_id = (u32)term_val(sum_arg);
                }
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

                Term sum_arg = heap_read(ctx, loc + 1);
                if (term_tag(sum_arg) == TAG_TOP || term_tag(sum_arg) == TAG_TEN) {
                    // Explicit axes from thvm_sum_axes
                    Term axes_t = thvm_reduce(ctx, sum_arg);
                    if (term_tag(axes_t) == TAG_TEN) {
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

            // Elementwise ops: DEFER dispatch when BOTH inputs already have data.
            // Creates tensor with buf_id=0 — materialized later by ENSURE at
            // fusion boundaries. This fuses chains of backward ops into one kernel.
            // Defer elementwise: create tensor with buf_id=0, let tensor_materialize
            // fuse chains at boundaries. Shared intermediates get multi-output kernels.
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

            // Defer reduce when input is deferred, elementwise, unshared.
            // Only when backend has codegen (dispatch_kernel_rs) for fused reduce+ew.
            if (((uop == UOP_SUM && b_id != 0 && ma->view.shape.rank >= 4) || uop == UOP_RMAX) &&
                ma->buf_id == 0 && ma->creator_op &&
                is_elementwise(ma->creator_op) && ma->defer_consumers == 0 &&
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
                            Term axes_e = thvm_reduce(ctx, sum_arg_e);
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

            // Eager reduce fusion: if reduce input is deferred ew, fuse at dispatch time.
            if (is_reduce && ma->buf_id == 0 && ma->creator_op &&
                is_elementwise(ma->creator_op)) {
                // Build ReduceSpec
                ReduceSpec ers = {0}; ers.reduce_type = uop;
                Term sum_arg_e = heap_read(ctx, loc + 1);
                if (term_tag(sum_arg_e) == TAG_TEN || term_tag(sum_arg_e) == TAG_TOP) {
                    Term axes_e = thvm_reduce(ctx, sum_arg_e);
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

            // Materialize lazy inputs before dispatch
            {
                ENSURE(ctx, a_id); ma = &ctx->tensors[a_id];
                if (b_id) { ENSURE(ctx, b_id); if (is_binary) mb = &ctx->tensors[b_id]; }
            }
            // Dispatch
            if (uop == UOP_MM) {
                u32 M = ma->view.shape.dims[0], K = ma->view.shape.dims[1], N = mb->view.shape.dims[1];
                md->backend->op_mm(md->buf_id, ma->buf_id, &ma->view,
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

        // DUP interaction: DP0/DP1 are auxiliary port labels of a single DUP node.
        // The interaction fires between the DUP's principal port and the value.
        // ONE rule, branching on dp_index for which projection to return.
        case TAG_DP0:
        case TAG_DP1: {
            u32 dp_index = (tag == TAG_DP1) ? 1 : 0;
            u64 loc = term_val(t);
            Term val = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, val);  // cache reduced value for other projection
            // DUP ⊳ SUP: annihilation (same label) or commutation (diff label)
            if (term_tag(val) == TAG_SUP) {
                ctx->itrs++;
                t = heap_read(ctx, term_val(val) + dp_index);
                goto inet_step;
            }
            // DUP ⊳ TEN: copy atom — both projections get same TEN (shared buffer)
            if (term_tag(val) == TAG_TEN) return val;
            // DUP ⊳ ERA: both projections get ERA
            if (term_tag(val) == TAG_ERA) return val;
            // DUP ⊳ NUM: copy atom
            if (term_tag(val) == TAG_NUM) return val;
            // Not yet reducible — return DUP as-is
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

