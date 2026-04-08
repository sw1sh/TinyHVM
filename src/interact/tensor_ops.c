            // === Phase 3 inet ops ===

            if (uop == UOP_ASSIGN) {
                // Args WNF from trampoline. When reached via inet_step
                // (combinator chain), args may be unreduced — return t
                // so the trampoline re-enters and reduces args.
                Term dst_r = heap_read(ctx, loc);
                Term src_t = heap_read(ctx, loc + 1);
                if (getenv("THVM_SCHED_DIAG"))
                    fprintf(stderr, "ASSIGN: dst_tag=%u src_tag=%u src_ext=%u\n",
                        term_tag(dst_r), term_tag(src_t),
                        term_tag(src_t)==TAG_TOP?term_ext(src_t):0);
                // Resolve WNF view chain on src: dispatch inner FUSINGs,
                // then create TAG_TEN view aliases for the view ops.
                if (term_tag(src_t) == TAG_TOP && term_ext(src_t) != UOP_ASSIGN) {
                    u32 _su = term_ext(src_t);
                    int _is_view = (_su >= UOP_RESHAPE && _su <= UOP_PAD);
                    if (_is_view) {
                        // Walk through view chain to find the innermost non-view term
                        Term _inner = src_t;
                        u64 _locs[16]; u32 _terms_n = 0;
                        while (term_tag(_inner) == TAG_TOP && _terms_n < 16) {
                            u32 _iu = term_ext(_inner);
                            if (_iu < UOP_RESHAPE || _iu > UOP_PAD) break;
                            _locs[_terms_n++] = term_val(_inner);
                            Term _next = heap_read(ctx, term_val(_inner));
                            if (term_tag(_next) == TAG_DP0 || term_tag(_next) == TAG_DP1)
                                _next = heap_read(ctx, term_val(_next));
                            _inner = _next;
                        }
                        // Dispatch the inner term (FUSING or compute op)
                        if (getenv("THVM_SCHED_DIAG"))
                            fprintf(stderr, "ASSIGN_WALK: depth=%u inner_tag=%u inner_ext=%u\n",
                                _terms_n, term_tag(_inner), term_tag(_inner)==TAG_TOP?term_ext(_inner):0);
                        if (term_tag(_inner) == TAG_TOP) {
                            Term _r = thvm_reduce(ctx, _inner);
                            if (term_tag(_r) == TAG_TEN) {
                                // Now resolve view chain bottom-up using st_get
                                u32 _tid = (u32)term_val(_r);
                                for (int _vi = (int)_terms_n - 1; _vi >= 0; _vi--) {
                                    const View *_sv = st_get(_locs[_vi]);
                                    if (_sv) {
                                        u32 _nvid = tensor_view_of(ctx, _tid, *_sv);
                                        _tid = _nvid;
                                    }
                                }
                                heap_set(ctx, loc + 1, term_ten(_tid, DTYPE_F32));
                                src_t = heap_read(ctx, loc + 1);
                            }
                        }
                    }
                }
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
                            if (ms->buf_id == 0) { fprintf(stderr, "ASSIGN_NULL_BUF: dst=%u src=%u\n", dst_id, src_id); goto assign_done; }
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

            // UOP_FUSING: scheduled kernel — dispatch from KernelEntry.
            // Fired by second thvm_reduce. Reads kernel spec from global table.
            if (uop == UOP_FUSING) {
                extern KernelEntry sched_kernels[];
                extern Term kid_results[];
                extern u32 sched_kernel_count;
                Term kid_term = heap_read(ctx, loc + 1); // arg1 = kernel index
                u32 kid = (u32)term_val(kid_term);
                // Dedup: same kid fired only once (multi-consumer propagation).
                if (kid < sched_kernel_count && term_tag(kid_results[kid]) != TAG_ERA)
                    RETURN_REDUCED(kid_results[kid]);
                KernelEntry *ke = &sched_kernels[kid];

                // Dead kernel: redirect to absorber
                if (ke->n_ops == 0 && ke->n_leaves == 0) {
                    extern u32 sched_absorber[];
                    u32 abs_kid = sched_absorber[kid];
                    if (abs_kid < sched_kernel_count &&
                        term_tag(kid_results[abs_kid]) != TAG_ERA) {
                        kid_results[kid] = kid_results[abs_kid];
                        RETURN_REDUCED(kid_results[abs_kid]);
                    }
                    RETURN_REDUCED(term_era());
                }

                // Allocate output buffer
                u32 dst_id = tensor_create(ctx, ke->out_shape, DTYPE_F32);
                TensorMeta *md = &ctx->tensors[dst_id];

                // Collect leaf buffers — resolve placeholder/lazy IDs from UOP_FUSING deps
                u32 bufs[FUSE_MAX_LEAVES];
                const View *views[FUSE_MAX_LEAVES];
                for (u32 i = 0; i < ke->n_leaves; i++) {
                    u32 lid = ke->leaf_ids[i];
                    if (lid == 0 || (lid & 0x80000000u)) {
                        // Placeholder (0) or lazy leaf (bit 31 set).
                        // Try reducing the leaf term — FUSING leaves fire and return TAG_TEN.
                        Term lt = ke->leaf_terms[i];
                        Term lr = thvm_reduce(ctx, lt);
                        if (term_tag(lr) == TAG_TEN) { lid = (u32)term_val(lr); }
                        else {
                            // WNF compute op: the scheduler replaced the TAG_TOP with FUSING
                            // on the heap but leaf_terms holds the stale TAG_TOP.
                            // Scan heap for the FUSING that replaced this TAG_TOP.
                            for (u64 _sh = 1; _sh < ctx->heap_pos; _sh++) {
                                Term _ht = ctx->heap[_sh];
                                if (term_tag(_ht) == TAG_TOP && term_ext(_ht) == UOP_FUSING) {
                                    extern KernelEntry sched_kernels[];
                                    Term _kid_t = heap_read(ctx, term_val(_ht) + 1);
                                    u32 _kid = (u32)term_val(_kid_t);
                                    if (_kid < sched_kernel_count && sched_kernels[_kid].original_term == lt) {
                                        lr = thvm_reduce(ctx, _ht);
                                        if (term_tag(lr) == TAG_TEN) { lid = (u32)term_val(lr); break; }
                                    }
                                }
                            }
                            // View chain: leaf is RESHAPE/EXPAND/etc wrapping a compute op.
                            // Walk through view ops, dispatch inner FUSING, create view aliases.
                            if (!(lid && lid != (lid | 0x80000000u)) &&
                                term_tag(lr) == TAG_TOP && is_view_op(term_ext(lr))) {
                                Term _inner = lr;
                                u64 _vlocs[16]; u32 _vn = 0;
                                while (term_tag(_inner) == TAG_TOP && _vn < 16) {
                                    u32 _vu = term_ext(_inner);
                                    if (!is_view_op(_vu)) break;
                                    _vlocs[_vn++] = term_val(_inner);
                                    Term _next = heap_read(ctx, term_val(_inner));
                                    if (term_tag(_next) == TAG_DP0 || term_tag(_next) == TAG_DP1)
                                        _next = heap_read(ctx, term_val(_next));
                                    _inner = _next;
                                }
                                // Try to dispatch the inner term
                                Term _ir = thvm_reduce(ctx, _inner);
                                if (term_tag(_ir) != TAG_TEN) {
                                    // Scan heap for FUSING that replaced inner
                                    for (u64 _sh2 = 1; _sh2 < ctx->heap_pos; _sh2++) {
                                        Term _ht2 = ctx->heap[_sh2];
                                        if (term_tag(_ht2) == TAG_TOP && term_ext(_ht2) == UOP_FUSING) {
                                            Term _kid_t2 = heap_read(ctx, term_val(_ht2) + 1);
                                            u32 _kid2 = (u32)term_val(_kid_t2);
                                            if (_kid2 < sched_kernel_count && sched_kernels[_kid2].original_term == _inner) {
                                                _ir = thvm_reduce(ctx, _ht2);
                                                if (term_tag(_ir) == TAG_TEN) break;
                                            }
                                        }
                                    }
                                }
                                if (term_tag(_ir) == TAG_TEN) {
                                    // Resolve view chain bottom-up
                                    u32 _tid = (u32)term_val(_ir);
                                    for (int _vi = (int)_vn - 1; _vi >= 0; _vi--) {
                                        const View *_sv = st_get(_vlocs[_vi]);
                                        if (_sv) {
                                            u32 _nvid = tensor_view_of(ctx, _tid, *_sv);
                                            _tid = _nvid;
                                        }
                                    }
                                    lid = _tid;
                                }
                            }
                            if (!(lid && lid != (lid | 0x80000000u))) {
                                fprintf(stderr, "FUSING: leaf %u unresolved (tag=%u ext=%u lt_tag=%u lt_ext=%u)\n",
                                    i, term_tag(lr),
                                    term_tag(lr)==TAG_TOP ? term_ext(lr) : 0,
                                    term_tag(lt),
                                    term_tag(lt)==TAG_TOP ? term_ext(lt) : 0);
                                RETURN_REDUCED(term_era());
                            }
                        }
                    }
                    ENSURE(ctx, lid);
                    bufs[i] = ctx->tensors[lid].buf_id;
                    views[i] = &ke->leaf_views[i];
                }

                // Build ST pointer array for multi-view leaves
                const ShapeTracker *st_ptrs[FUSE_MAX_LEAVES];
                int has_multiview = 0;
                for (u32 i = 0; i < ke->n_leaves; i++) {
                    st_ptrs[i] = &ke->leaf_sts[i];
                    if (ke->leaf_sts[i].n_views >= 2) has_multiview = 1;
                }
                md->backend->dispatch_kernel_rs(
                    md->buf_id, bufs, views,
                    has_multiview ? st_ptrs : NULL, ke->n_leaves,
                    ke->ops, ke->n_ops, &ke->full_shape,
                    ke->has_reduce ? &ke->reduce : NULL,
                    NULL, NULL, 0);


                // Post-reduce reshape (if SUM was wrapped by RESHAPE)
                if (ke->has_reduce && term_tag(ke->reshape_term) != TAG_ERA) {
                    u64 rs_loc = term_val(ke->reshape_term);
                    Term shape_t = heap_read(ctx, rs_loc + 1);
                    if (term_tag(shape_t) == TAG_DP0 || term_tag(shape_t) == TAG_DP1)
                        shape_t = heap_read(ctx, term_val(shape_t));
                    if (term_tag(shape_t) == TAG_TEN) {
                        TensorMeta *ms = &ctx->tensors[(u32)term_val(shape_t)];
                        u32 rank = ms->view.numel;
                        f32 dims_f[MAX_DIM];
                        META_READ(ms->backend, ms->buf_id, dims_f, rank * sizeof(f32));
                        Shape ns = {.rank = rank};
                        for (u32 i = 0; i < rank; i++) ns.dims[i] = (u32)dims_f[i];
                        u32 rs_id = tensor_view_of(ctx, dst_id, view_create(ns));
                        ctx->tensors[rs_id].creator_op = UOP_RESHAPE;
                        ctx->tensors[rs_id].src_ids[0] = dst_id;
                        ctx->itrs++;
                        Term result = term_ten(rs_id, DTYPE_F32);
                        kid_results[kid] = result;
                        RETURN_REDUCED(result);
                    }
                }
                ctx->itrs++;
                { Term result = term_ten(dst_id, DTYPE_F32);
                  kid_results[kid] = result;
                  RETURN_REDUCED(result); }
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
            // Movement ops fire immediately when args are TAG_TEN.

            u32 a_id = (u32)term_val(a);
            TensorMeta *ma = &ctx->tensors[a_id];

            if (!is_movement) goto skip_movement;
            {
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
                        // view_reshape returns numel=0 when merge fails → materialize
                        int needs_materialize = (new_view.numel == 0);
                        if (!needs_materialize && !same_shape && !new_view.contiguous) {
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
            skip_movement:;

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

            // Compute ops stay as TAG_TOP. Scheduler rewrites to UOP_FUSING.
            return t;
