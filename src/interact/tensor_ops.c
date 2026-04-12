            // === Phase 3 inet ops ===

            if (uop == UOP_ASSIGN) {
                // ASSIGN: pure interaction. Both args must be TAG_TEN.
                // No gate, no nested reduce. Fires when args are ready.
                Term dst_r = heap_read(ctx, loc);
                Term src_t = heap_read(ctx, loc + 1);
                if (getenv("THVM_SCHED_DIAG"))
                    fprintf(stderr, "ASSIGN: dst_tag=%u src_tag=%u src_ext=%u\n",
                        term_tag(dst_r), term_tag(src_t),
                        term_tag(src_t)==TAG_TOP?term_ext(src_t):0);
                if (term_tag(dst_r) != TAG_TEN || term_tag(src_t) != TAG_TEN)
                    return t;  // not ready — reducer will reduce args first
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
                                ms->view.numel == md->view.numel &&
                                ms->dtype == md->dtype) {
                                md->backend->buf_copy(md->buf_id, ms->buf_id,
                                    (u64)md->view.numel * dtype_size(md->dtype));
                            } else if (md->backend->contiguify && ms->dtype == md->dtype) {
                                md->backend->contiguify(md->buf_id, md->view.numel,
                                                         ms->buf_id, &ms->view);
                            } else {
                                u32 src_dtype = ms->dtype;
                                void *src_host = thvm_to_host_raw(ctx, src_t, &src_dtype, NULL);
                                u64 n = md->view.numel;
                                u64 nbytes = n * dtype_size(md->dtype);
                                if (!src_host) goto assign_done;
                                if (src_dtype == md->dtype) {
                                    md->backend->buf_write(md->buf_id, src_host, nbytes);
                                } else {
                                    void *dst_host = malloc((size_t)nbytes);
                                    for (u32 i = 0; i < n; i++)
                                        dtype_store_from_f32(dst_host, md->dtype, i,
                                                             dtype_load_as_f32(src_host, src_dtype, i));
                                    md->backend->buf_write(md->buf_id, dst_host, nbytes);
                                    free(dst_host);
                                }
                            }
                        }
                    assign_done:
                        if (md->host_ptr) { free(md->host_ptr); md->host_ptr = NULL; }
                        heap_set(ctx, loc + 1, dst_r);
                        // Bump buffer epoch — invalidates cached kernel results
                        // that read from this buffer.
                        { extern u32 buf_epoch[];
                          if (md->buf_id < MAX_BUF_EPOCHS)
                              buf_epoch[md->buf_id]++; }
                    }
                    ctx->itrs++;
                    RETURN_REDUCED(dst_r);
                }
            }

            // UOP_KERNEL: scheduled kernel — dispatch from KernelEntry.
            // Fired by second thvm_reduce. Reads kernel spec from global table.
            // UOP_KERNEL: scheduled kernel — dispatch immediately.
            // Created by local UOP_FUSE. SEQ chains enforce ordering.
            if (uop == UOP_KERNEL) {
                extern Term kid_results[];
                extern u32 sched_kernel_count;
                extern u32 buf_epoch[];
                extern u32 kid_input_bufs[][KERNEL_MAX_INPUTS];
                extern u32 kid_input_epochs[][KERNEL_MAX_INPUTS];
                extern u32 kid_n_inputs[];
                Term kid_term = heap_read(ctx, loc + 1); // arg1 = kernel index
                u32 kid = (u32)term_val(kid_term);
                if (getenv("THVM_SCHED_DIAG"))
                    fprintf(stderr, "KERNEL kid=%u cached=%d n_inputs=%u epoch_check=",
                            kid, term_tag(kid_results[kid]) != TAG_ERA, kid_n_inputs[kid]);
                if (getenv("THVM_SCHED_DIAG") && term_tag(kid_results[kid]) != TAG_ERA) {
                    for (u32 _di = 0; _di < kid_n_inputs[kid]; _di++)
                        fprintf(stderr, "b%u:%u/%u ", kid_input_bufs[kid][_di],
                                kid_input_epochs[kid][_di],
                                kid_input_bufs[kid][_di] < MAX_BUF_EPOCHS ? buf_epoch[kid_input_bufs[kid][_di]] : 0);
                }
                if (getenv("THVM_SCHED_DIAG")) fprintf(stderr, "\n");
                // Cache check: valid only if all input buffer epochs match
                if (kid < sched_kernel_count && term_tag(kid_results[kid]) != TAG_ERA) {
                    int cache_valid = 1;
                    for (u32 i = 0; i < kid_n_inputs[kid]; i++) {
                        u32 bid = kid_input_bufs[kid][i];
                        if (bid < MAX_BUF_EPOCHS && buf_epoch[bid] != kid_input_epochs[kid][i]) {
                            cache_valid = 0;
                            break;
                        }
                    }
                    if (cache_valid)
                        RETURN_REDUCED(kid_results[kid]);
                    // Stale — invalidate and re-dispatch
                    kid_results[kid] = term_era();
                }
                {
                    Term result = thvm_sched_dispatch_kernel(ctx, kid);
                    if (kid < sched_kernel_count) {
                        kid_results[kid] = result;
                        // Record input buffer epochs for future cache checks
                        extern KernelEntry sched_kernels[];
                        KernelEntry *ke = &sched_kernels[kid];
                        u32 ni = 0;
                        for (u32 i = 0; i < ke->n_leaves && ni < KERNEL_MAX_INPUTS; i++) {
                            u32 tid = ke->leaf_ids[i];
                            if (tid && tid < ctx->tensor_count) {
                                u32 bid = ctx->tensors[tid].buf_id;
                                if (bid) {
                                    kid_input_bufs[kid][ni] = bid;
                                    kid_input_epochs[kid][ni] = (bid < MAX_BUF_EPOCHS) ? buf_epoch[bid] : 0;
                                    ni++;
                                }
                            }
                        }
                        kid_n_inputs[kid] = ni;
                    }
                    RETURN_REDUCED(result);
                }
            }

            // UOP_FUSE: propagating IC agent — like GRAD but for fusion.
            // Propagates through compute ops, wrapping children in FUSE.
            // When all children are TAG_TEN, creates UOP_KERNEL and dispatches.
            // Distributes through SEQ/CTR/ASSIGN transparently.
            if (uop == UOP_FUSE) {
                Term payload = heap_read(ctx, loc);
                u8 ptag = term_tag(payload);

                // TAG_TEN/ERA/NUM: leaf — pass through (nothing to fuse)
                if (ptag == TAG_TEN || ptag == TAG_ERA || ptag == TAG_NUM)
                    RETURN_REDUCED(payload);

                // TAG_TOP: dispatch by UOP
                if (ptag == TAG_TOP) {
                    u32 puop = term_ext(payload);

                    // Already fused or meta — pass through
                    if (puop == UOP_KERNEL || puop == UOP_FUSE || puop == UOP_SCHED)
                        RETURN_REDUCED(payload);

                    // ASSIGN: wrap src in FUSE, pass through
                    if (puop == UOP_ASSIGN) {
                        u64 aloc = term_val(payload);
                        Term src = heap_read(ctx, aloc + 1);
                        if (term_tag(src) != TAG_TEN && term_tag(src) != TAG_ERA &&
                            !(term_tag(src) == TAG_TOP && term_ext(src) == UOP_KERNEL)) {
                            u64 fl = heap_alloc(ctx, 1);
                            heap_set(ctx, fl, src);
                            heap_set(ctx, aloc + 1, term_new(TAG_TOP, UOP_FUSE, fl));
                        }
                        RETURN_REDUCED(payload);
                    }

                    // Compute op (MUL, ADD, SUM, etc.): wrap children in FUSE
                    // and convert this op into a single-op KERNEL.
                    // Multi-op fusion is a future optimization.
                    if (is_elementwise(puop) || puop == UOP_SUM || puop == UOP_RMAX) {
                        u64 ploc = term_val(payload);
                        u32 arity = is_binary(puop) ? 2 : 1;

                        // Wrap each child in FUSE
                        for (u32 i = 0; i < arity; i++) {
                            Term child = heap_read(ctx, ploc + i);
                            if (term_tag(child) != TAG_TEN && term_tag(child) != TAG_NUM &&
                                term_tag(child) != TAG_ERA &&
                                !(term_tag(child) == TAG_TOP && term_ext(child) == UOP_KERNEL)) {
                                u64 fl = heap_alloc(ctx, 1);
                                heap_set(ctx, fl, child);
                                heap_set(ctx, ploc + i, term_new(TAG_TOP, UOP_FUSE, fl));
                            }
                        }
                        // Return the compute op — its children are now FUSE-wrapped.
                        // The reducer will reduce the FUSE children to TAG_TEN,
                        // then the compute op will have TAG_TEN inputs and can dispatch.
                        RETURN_REDUCED(payload);
                    }

                    // Other TAG_TOP (LOG_PRINT, IFZ, etc.): pass through
                    RETURN_REDUCED(payload);
                }

                // TAG_SEQ: distribute → SEQ(FUSE(a), FUSE(b))
                if (ptag == TAG_SEQ) {
                    u64 sloc = term_val(payload);
                    for (u32 i = 0; i < 2; i++) {
                        Term c = heap_read(ctx, sloc + i);
                        if (term_tag(c) != TAG_TEN && term_tag(c) != TAG_ERA) {
                            u64 fl = heap_alloc(ctx, 1);
                            heap_set(ctx, fl, c);
                            heap_set(ctx, sloc + i, term_new(TAG_TOP, UOP_FUSE, fl));
                        }
                    }
                    RETURN_REDUCED(payload);
                }

                // TAG_CTR: distribute → CTR(FUSE(c0), FUSE(c1), ...)
                if (ptag == TAG_CTR) {
                    u64 cloc = term_val(payload);
                    u32 arity = term_ext(payload);
                    for (u32 i = 0; i < arity; i++) {
                        Term c = heap_read(ctx, cloc + i);
                        if (term_tag(c) != TAG_TEN && term_tag(c) != TAG_ERA) {
                            u64 fl = heap_alloc(ctx, 1);
                            heap_set(ctx, fl, c);
                            heap_set(ctx, cloc + i, term_new(TAG_TOP, UOP_FUSE, fl));
                        }
                    }
                    RETURN_REDUCED(payload);
                }

                // DP0/DP1: propagate through DUP projections
                if (ptag == TAG_DP0 || ptag == TAG_DP1) {
                    // Let the reducer resolve the DP first, then re-fuse
                    RETURN_REDUCED(payload);
                }

                // Anything else: pass through
                RETURN_REDUCED(payload);
            }

            // UOP_SCHED: pass-through. With local FUSE + clean ASSIGN + SEQ,
            // scheduling signals are unnecessary — KERNEL fires immediately,
            // ASSIGN fires when args are TAG_TEN, SEQ handles ordering.
            if (uop == UOP_SCHED) {
                RETURN_REDUCED(heap_read(ctx, loc));
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
                    u32 dev_dtype = DTYPE_U32;
                    void *dev_raw = thvm_to_host_raw(ctx, dev_t, &dev_dtype, NULL);
                    u32 dev_idx = dev_raw ? dtype_load_as_u32(dev_raw, dev_dtype, 0) : 0;
                    Backend *dst_be = ctx->backends[dev_idx];
                    if (!dst_be || ms->backend == dst_be) {
                        ctx->itrs++;
                        RETURN_REDUCED(src_t); // no-op: same device or invalid
                    }
                    // Read to host, allocate on target device
                    void *host = thvm_to_host_raw(ctx, src_t, NULL, NULL);
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

            if (uop == UOP_DETACH) {
                Term memo = heap_read(ctx, loc + 1);
                if (term_tag(memo) == TAG_TEN) {
                    RETURN_REDUCED(memo);
                }
                Term src = heap_read(ctx, loc); // WNF from trampoline when possible
                Term src_expr = src;
                if (getenv("THVM_LOOP_DIAG")) {
                    fprintf(stderr, "DETACH_ENTER loc=%llu src_tag=%u src_ext=%u src_val=%llu\n",
                            (unsigned long long)loc,
                            (u32)term_tag(src),
                            (u32)term_ext(src),
                            (unsigned long long)term_val(src));
                    if (term_tag(src) == TAG_TOP) {
                        u64 src_loc = term_val(src);
                        Term a0 = heap_read(ctx, src_loc + 0);
                        Term a1 = heap_read(ctx, src_loc + 1);
                        fprintf(stderr,
                                "DETACH_SRC0  loc=%llu a0_tag=%u a0_ext=%u a0_val=%llu a1_tag=%u a1_ext=%u a1_val=%llu\n",
                                (unsigned long long)loc,
                                (u32)term_tag(a0), (u32)term_ext(a0), (unsigned long long)term_val(a0),
                                (u32)term_tag(a1), (u32)term_ext(a1), (unsigned long long)term_val(a1));
                        if (term_tag(a1) == TAG_TOP) {
                            u64 a1_loc = term_val(a1);
                            Term b0 = heap_read(ctx, a1_loc + 0);
                            Term b1 = heap_read(ctx, a1_loc + 1);
                            fprintf(stderr,
                                    "DETACH_SRC1  loc=%llu b0_tag=%u b0_ext=%u b0_val=%llu b1_tag=%u b1_ext=%u b1_val=%llu\n",
                                    (unsigned long long)loc,
                                    (u32)term_tag(b0), (u32)term_ext(b0), (unsigned long long)term_val(b0),
                                    (u32)term_tag(b1), (u32)term_ext(b1), (unsigned long long)term_val(b1));
                            if (term_tag(b0) == TAG_VAR || term_tag(b0) == TAG_DP0 || term_tag(b0) == TAG_DP1) {
                                Term subb0 = heap_read(ctx, term_val(b0));
                                fprintf(stderr,
                                        "DETACH_SRC1A loc=%llu subb0_tag=%u subb0_ext=%u subb0_val=%llu\n",
                                        (unsigned long long)loc,
                                        (u32)term_tag(subb0), (u32)term_ext(subb0), (unsigned long long)term_val(subb0));
                            }
                            if (term_tag(b1) == TAG_VAR || term_tag(b1) == TAG_DP0 || term_tag(b1) == TAG_DP1) {
                                Term subb1 = heap_read(ctx, term_val(b1));
                                fprintf(stderr,
                                        "DETACH_SRC1B loc=%llu subb1_tag=%u subb1_ext=%u subb1_val=%llu\n",
                                        (unsigned long long)loc,
                                        (u32)term_tag(subb1), (u32)term_ext(subb1), (unsigned long long)term_val(subb1));
                                if (term_tag(subb1) == TAG_TOP) {
                                    u64 subb1_loc = term_val(subb1);
                                    Term c0 = heap_read(ctx, subb1_loc + 0);
                                    Term c1 = heap_read(ctx, subb1_loc + 1);
                                    fprintf(stderr,
                                            "DETACH_SRC1C loc=%llu c0_tag=%u c0_ext=%u c0_val=%llu c1_tag=%u c1_ext=%u c1_val=%llu\n",
                                            (unsigned long long)loc,
                                            (u32)term_tag(c0), (u32)term_ext(c0), (unsigned long long)term_val(c0),
                                            (u32)term_tag(c1), (u32)term_ext(c1), (unsigned long long)term_val(c1));
                                }
                            }
                        }
                        if (term_tag(a0) == TAG_VAR || term_tag(a0) == TAG_DP0 || term_tag(a0) == TAG_DP1) {
                            Term sub0 = heap_read(ctx, term_val(a0));
                            fprintf(stderr,
                                    "DETACH_SRC0A loc=%llu sub0_tag=%u sub0_ext=%u sub0_val=%llu\n",
                                    (unsigned long long)loc,
                                    (u32)term_tag(sub0), (u32)term_ext(sub0), (unsigned long long)term_val(sub0));
                        }
                        if (term_tag(a1) == TAG_VAR || term_tag(a1) == TAG_DP0 || term_tag(a1) == TAG_DP1) {
                            Term sub1 = heap_read(ctx, term_val(a1));
                            fprintf(stderr,
                                    "DETACH_SRC1A loc=%llu sub1_tag=%u sub1_ext=%u sub1_val=%llu\n",
                                    (unsigned long long)loc,
                                    (u32)term_tag(sub1), (u32)term_ext(sub1), (unsigned long long)term_val(sub1));
                        }
                    }
                }
                src = thvm_eval(ctx, src);
                if (getenv("THVM_LOOP_DIAG")) {
                    fprintf(stderr, "DETACH_EVAL  loc=%llu src_tag=%u src_ext=%u src_val=%llu\n",
                            (unsigned long long)loc,
                            (u32)term_tag(src),
                            (u32)term_ext(src),
                            (unsigned long long)term_val(src));
                    if (term_tag(src) == TAG_TOP) {
                        u64 src_loc = term_val(src);
                        Term a0 = heap_read(ctx, src_loc + 0);
                        Term a1 = heap_read(ctx, src_loc + 1);
                        fprintf(stderr,
                                "DETACH_EVAL0 loc=%llu a0_tag=%u a0_ext=%u a0_val=%llu a1_tag=%u a1_ext=%u a1_val=%llu\n",
                                (unsigned long long)loc,
                                (u32)term_tag(a0), (u32)term_ext(a0), (unsigned long long)term_val(a0),
                                (u32)term_tag(a1), (u32)term_ext(a1), (unsigned long long)term_val(a1));
                        if (term_tag(a1) == TAG_TOP) {
                            u64 a1_loc = term_val(a1);
                            Term b0 = heap_read(ctx, a1_loc + 0);
                            Term b1 = heap_read(ctx, a1_loc + 1);
                            fprintf(stderr,
                                    "DETACH_EVAL1 loc=%llu b0_tag=%u b0_ext=%u b0_val=%llu b1_tag=%u b1_ext=%u b1_val=%llu\n",
                                    (unsigned long long)loc,
                                    (u32)term_tag(b0), (u32)term_ext(b0), (unsigned long long)term_val(b0),
                                    (u32)term_tag(b1), (u32)term_ext(b1), (unsigned long long)term_val(b1));
                            if (term_tag(b0) == TAG_VAR || term_tag(b0) == TAG_DP0 || term_tag(b0) == TAG_DP1) {
                                Term subb0 = heap_read(ctx, term_val(b0));
                                fprintf(stderr,
                                        "DETACH_EVAL1A loc=%llu subb0_tag=%u subb0_ext=%u subb0_val=%llu\n",
                                        (unsigned long long)loc,
                                        (u32)term_tag(subb0), (u32)term_ext(subb0), (unsigned long long)term_val(subb0));
                            }
                            if (term_tag(b1) == TAG_VAR || term_tag(b1) == TAG_DP0 || term_tag(b1) == TAG_DP1) {
                                Term subb1 = heap_read(ctx, term_val(b1));
                                fprintf(stderr,
                                        "DETACH_EVAL1B loc=%llu subb1_tag=%u subb1_ext=%u subb1_val=%llu\n",
                                        (unsigned long long)loc,
                                        (u32)term_tag(subb1), (u32)term_ext(subb1), (unsigned long long)term_val(subb1));
                                if (term_tag(subb1) == TAG_TOP) {
                                    u64 subb1_loc = term_val(subb1);
                                    Term c0 = heap_read(ctx, subb1_loc + 0);
                                    Term c1 = heap_read(ctx, subb1_loc + 1);
                                    fprintf(stderr,
                                            "DETACH_EVAL1C loc=%llu c0_tag=%u c0_ext=%u c0_val=%llu c1_tag=%u c1_ext=%u c1_val=%llu\n",
                                            (unsigned long long)loc,
                                            (u32)term_tag(c0), (u32)term_ext(c0), (unsigned long long)term_val(c0),
                                            (u32)term_tag(c1), (u32)term_ext(c1), (unsigned long long)term_val(c1));
                                }
                            }
                        }
                        if (term_tag(a0) == TAG_VAR || term_tag(a0) == TAG_DP0 || term_tag(a0) == TAG_DP1) {
                            Term sub0 = heap_read(ctx, term_val(a0));
                            fprintf(stderr,
                                    "DETACH_EVALA loc=%llu sub0_tag=%u sub0_ext=%u sub0_val=%llu\n",
                                    (unsigned long long)loc,
                                    (u32)term_tag(sub0), (u32)term_ext(sub0), (unsigned long long)term_val(sub0));
                        }
                        if (term_tag(a1) == TAG_VAR || term_tag(a1) == TAG_DP0 || term_tag(a1) == TAG_DP1) {
                            Term sub1 = heap_read(ctx, term_val(a1));
                            fprintf(stderr,
                                    "DETACH_EVALB loc=%llu sub1_tag=%u sub1_ext=%u sub1_val=%llu\n",
                                    (unsigned long long)loc,
                                    (u32)term_tag(sub1), (u32)term_ext(sub1), (unsigned long long)term_val(sub1));
                        }
                    }
                }
                if (term_tag(src) == TAG_ERA) {
                    ctx->itrs++;
                    RETURN_REDUCED(term_era());
                }
                u32 src_dtype = DTYPE_F32;
                Shape src_shape = SHAPE(1);
                void *host = thvm_to_host_raw(ctx, src, &src_dtype, &src_shape);
                if (!host) return t;

                Term src_real = thvm_force_tensor_term(ctx, src);
                TensorMeta *ms = (term_tag(src_real) == TAG_TEN)
                               ? &ctx->tensors[(u32)term_val(src_real)]
                               : NULL;
                u32 new_id = tensor_create(ctx, src_shape, src_dtype);
                TensorMeta *md = &ctx->tensors[new_id];
                u64 nbytes = (u64)md->view.numel * dtype_size(md->dtype);
                md->backend->buf_write(md->buf_id, host, nbytes);
                md->requires_grad = ms ? ms->requires_grad : 0;
                md->creator_op = 0;
                md->creator_loc = 0;
                md->src_ids[0] = 0;
                md->src_ids[1] = 0;
                md->defer_consumers = 0;
                heap_set(ctx, loc + 1, term_ten(new_id, md->dtype));

                thvm_spawn_detached_era(ctx, src_expr);
                ctx->itrs++;
                RETURN_REDUCED(term_ten(new_id, md->dtype));
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
                u32 cond_dtype = mc->dtype, then_dtype = ma_w->dtype, else_dtype = mb_w->dtype;
                void *cp = thvm_to_host_raw(ctx, cond_t, &cond_dtype, NULL);
                void *ap = thvm_to_host_raw(ctx, then_t, &then_dtype, NULL);
                void *bp = thvm_to_host_raw(ctx, else_t, &else_dtype, NULL);
                if (!cp || !ap || !bp) RETURN_REDUCED(term_era());
                u32 dst_wid = tensor_create(ctx, ma_w->view.shape, ma_w->dtype);
                u64 nbytes = (u64)n * dtype_size(ma_w->dtype);
                void *rp = malloc((size_t)nbytes);
                for (u32 i = 0; i < n; i++) {
                    f32 pick = dtype_load_as_f32(cp, cond_dtype, i) != 0.0f
                             ? dtype_load_as_f32(ap, then_dtype, i)
                             : dtype_load_as_f32(bp, else_dtype, i);
                    dtype_store_from_f32(rp, ma_w->dtype, i, pick);
                }
                ctx->tensors[dst_wid].backend->buf_write(ctx->tensors[dst_wid].buf_id, rp, nbytes);
                free(rp);
                ctx->itrs++;
                RETURN_REDUCED(term_ten(dst_wid, ma_w->dtype));
            }

            // UOP_IFZ(counter, zero_case, succ_lam)
            // counter==0 → zero_case; counter>0 → APP(succ_lam, TEN(counter-1))
            if (uop == UOP_IFZ) {
                Term ctr = heap_read(ctx, loc); // WNF from trampoline
                if (term_tag(ctr) != TAG_TEN) RETURN_REDUCED(term_era());
                u32 ctr_dtype = ctx->tensors[(u32)term_val(ctr)].dtype;
                void *val = thvm_to_host_raw(ctx, ctr, &ctr_dtype, NULL);
                f32 ctr_val = val ? dtype_load_as_f32(val, ctr_dtype, 0) : 0.0f;
                Term zero_case = heap_read(ctx, loc + 1);
                Term succ_lam = heap_read(ctx, loc + 2);
                if (getenv("THVM_LOOP_DIAG")) {
                    fprintf(stderr,
                            "IFZ loc=%llu ctr=%.3f zero_tag=%u succ_tag=%u succ_val=%llu\n",
                            (unsigned long long)loc,
                            ctr_val,
                            (u32)term_tag(zero_case),
                            (u32)term_tag(succ_lam),
                            (unsigned long long)term_val(succ_lam));
                }
                ctx->itrs++;
                thvm_spawn_detached_era(ctx, ctr);
                if (ctr_val <= 0.0f) {
                    // Base case: return zero_case (slot 1)
                    thvm_spawn_detached_era(ctx, succ_lam);
                    return zero_case;
                } else {
                    // Recursive case: decrement, apply succ_lam
                    f32 nv = ctr_val - 1.0f;
                    Term dec = thvm_scalar_typed(ctx, nv, ctr_dtype);
                    thvm_spawn_detached_era(ctx, zero_case);
                    return thvm_app(ctx, succ_lam, dec);
                }
            }

            // UOP_LOG_PRINT(tensor) — print and consume the branch
            if (uop == UOP_LOG_PRINT) {
                Term t = heap_read(ctx, loc); // WNF from trampoline
                ctx->itrs++;
                if (term_tag(t) == TAG_TEN) {
                    u32 tid = (u32)term_val(t);
                    u32 dtype = ctx->tensors[tid].dtype;
                    void *val = thvm_to_host_raw(ctx, t, &dtype, NULL);
                    u32 n = ctx->tensors[tid].view.numel;
                    if (n == 1) {
                        if (dtype == DTYPE_I32)      printf("  loss: %d\n", ((i32 *)val)[0]);
                        else if (dtype == DTYPE_U32) printf("  loss: %u\n", ((u32 *)val)[0]);
                        else                         printf("  loss: %.6f\n", dtype_load_as_f32(val, dtype, 0));
                    } else {
                        if (dtype == DTYPE_I32)      printf("  tensor[%u]: %d ...\n", n, ((i32 *)val)[0]);
                        else if (dtype == DTYPE_U32) printf("  tensor[%u]: %u ...\n", n, ((u32 *)val)[0]);
                        else                         printf("  tensor[%u]: %.6f ...\n", n, dtype_load_as_f32(val, dtype, 0));
                    }
                    tensor_release(ctx, tid);
                } else if (term_tag(t) == TAG_NUM) {
                    printf("  loss: %.6f\n", term_as_f32(t));
                }
                RETURN_REDUCED(term_era());
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
            int is_cast = (uop == UOP_CAST);

            // Args resolved by thvm_reduce's enter/apply loop before firing.
            // Direct reads here — no thvm_reduce recursion.
            Term a = heap_read(ctx, loc);
            heap_set(ctx, loc, a);

            int is_binary = (uop >= UOP_ADD && uop <= UOP_SUB) || uop == UOP_MM;
            int is_reduce = (uop == UOP_SUM || uop == UOP_RMAX);
            int has_aux = is_binary || is_movement || is_reduce || is_cast;
            Term b = term_era();
            if (has_aux) {
                b = heap_read(ctx, loc + 1);
                heap_set(ctx, loc + 1, b);
            }
            Term b_arg = has_aux ? heap_read(ctx, loc + 1) : term_era();

            // Strict ERA commutation for lazy UOP nodes:
            // if any aux port is ERA, the UOP disappears and ERA propagates to
            // all remaining aux payloads as explicit future interactions.
            if (term_tag(a) == TAG_ERA || (has_aux && term_tag(b_arg) == TAG_ERA)) {
                int a_has_era = term_tag(a) == TAG_ERA;
                int b_has_era = has_aux && term_tag(b_arg) == TAG_ERA;
                int a_is_active_era = a_has_era && term_val(a) != 0;
                int b_is_active_era = b_has_era && term_val(b_arg) != 0;
                Term era_term = a_has_era ? a : b_arg;
                Term other = a_has_era ? b_arg : a;
                // ADD treats an erased branch as additive zero: drop the ADD,
                // preserve the other branch, and let the ERA continue erasing
                // its own payload as a detached interaction.
                if (uop == UOP_ADD) {
                    heap_set(ctx, loc + 0, term_era());
                    heap_set(ctx, loc + 1, term_era());
                    if (a_has_era && b_has_era) {
                        if (a_is_active_era && b_is_active_era)
                            thvm_spawn_detached_era(ctx, b_arg);
                        ctx->itrs++;
                        if (a_is_active_era) RETURN_REDUCED(a);
                        if (b_is_active_era) RETURN_REDUCED(b_arg);
                        RETURN_REDUCED(term_era());
                    }
                    if (a_is_active_era || b_is_active_era)
                        thvm_spawn_detached_era(ctx, era_term);
                    ctx->itrs++;
                    RETURN_REDUCED(other);
                }
                if (has_aux && !(a_has_era && b_has_era))
                    thvm_spawn_detached_era(ctx, other);
                heap_set(ctx, loc + 0, term_era());
                if (has_aux)
                    heap_set(ctx, loc + 1, term_era());
                ctx->itrs++;
                if (a_has_era && b_has_era) {
                    if (a_is_active_era && b_is_active_era)
                        thvm_spawn_detached_era(ctx, b_arg);
                    if (a_is_active_era) RETURN_REDUCED(a);
                    if (b_is_active_era) RETURN_REDUCED(b_arg);
                    RETURN_REDUCED(term_era());
                }
                RETURN_REDUCED(era_term);
            }

            // ADD with a literal zero branch is algebraically transparent.
            // Fire this eagerly in phase 1 so zero gradients don't leave
            // dead lazy ADD nodes in the step graph.
            if (uop == UOP_ADD) {
                int a_zero = term_tag(a) == TAG_NUM && term_as_f32(a) == 0.0f;
                int b_zero = term_tag(b) == TAG_NUM && term_as_f32(b) == 0.0f;
                if (a_zero || b_zero) {
                    heap_set(ctx, loc + 0, term_era());
                    heap_set(ctx, loc + 1, term_era());
                    ctx->itrs++;
                    if (a_zero && b_zero) RETURN_REDUCED(a);
                    RETURN_REDUCED(a_zero ? b : a);
                }
            }

            // === TOP-SUP: distribute tensor op across superposition ===
            // TOP(op, &L{a0,a1}, b) → &L{TOP(op,a0,DP0_L(b)), TOP(op,a1,DP1_L(b))}
            if (term_tag(a) == TAG_SUP) {
                u32 lab = term_ext(a);
                u64 sup_loc = term_val(a);
                Term a0 = heap_read(ctx, sup_loc + 0);
                Term a1 = heap_read(ctx, sup_loc + 1);
                Term b0 = b, b1 = b;
                if (has_aux) {
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
            if (term_tag(b) == TAG_SUP && has_aux) {
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

            if (term_tag(a) != TAG_TEN) return t;
            if (is_binary && term_tag(b) != TAG_TEN) return t;
            if (uop == UOP_CAST && term_tag(b) != TAG_TEN) return t;

            // Compute ops: return t (stay TAG_TOP). Scheduler rewrites to UOP_KERNEL.
            // Movement ops fire immediately when args are TAG_TEN.

            u32 a_id = (u32)term_val(a);
            TensorMeta *ma = &ctx->tensors[a_id];

            if (uop == UOP_CAST) {
                if (!ctx->dispatch_mode) return t;
                u32 arg_dtype = DTYPE_U32;
                void *dtype_raw = thvm_to_host_raw(ctx, b, &arg_dtype, NULL);
                if (!dtype_raw) RETURN_REDUCED(term_era());
                u32 dst_dtype = dtype_load_as_u32(dtype_raw, arg_dtype, 0);
                if (dst_dtype >= DTYPE_COUNT) RETURN_REDUCED(term_era());
                if (dst_dtype == ma->dtype) {
                    ctx->itrs++;
                    RETURN_REDUCED(a);
                }
                void *src_raw = thvm_to_host_raw(ctx, a, NULL, NULL);
                if (!src_raw) RETURN_REDUCED(term_era());
                u32 n = ma->view.numel;
                u32 dst_id = tensor_create(ctx, ma->view.shape, dst_dtype);
                TensorMeta *md = &ctx->tensors[dst_id];
                size_t bytes = (size_t)n * dtype_size(dst_dtype);
                void *dst_raw = malloc(bytes);
                for (u32 i = 0; i < n; i++)
                    dtype_store_from_f32(dst_raw, dst_dtype, i, dtype_load_as_f32(src_raw, ma->dtype, i));
                if (md->backend) md->backend->buf_write(md->buf_id, dst_raw, bytes);
                if (n <= MAX_DIM * 2) {
                    md->host_ptr = malloc(bytes);
                    memcpy(md->host_ptr, dst_raw, bytes);
                }
                free(dst_raw);
                md->requires_grad = ma->requires_grad;
                md->creator_op = UOP_CAST;
                md->src_ids[0] = a_id;
                md->src_ids[1] = (u32)term_val(b);
                ctx->itrs++;
                RETURN_REDUCED(term_ten(dst_id, dst_dtype));
            }

            if (!is_movement) goto skip_movement;
            {
                u32 b_id = is_binary ? (u32)term_val(b) : 0;
                TensorMeta *mb = is_binary ? &ctx->tensors[b_id] : NULL;
                u32 out_shape[MAX_DIM]; u32 out_ndim;
                const View *ov = st_get(loc);
                if (is_reduce) {
                    out_ndim = ma->view.shape.rank;
                    for (u32 i = 0; i < out_ndim; i++) out_shape[i] = ma->view.shape.dims[i];
                    Term sa = heap_read(ctx, loc + 1);
                    if (term_tag(sa) == TAG_TEN) {
                        u32 ax_id = (u32)term_val(sa);
                        TensorMeta *axt = &ctx->tensors[ax_id];
                        u32 af[MAX_DIM];
                        u32 an = tensor_meta_read_u32(ctx, ax_id, af, MAX_DIM);
                        for (u32 i = 0; i < an; i++) out_shape[af[i]] = 1;
                        if (!b_id) b_id = ax_id;
                    } else { for (int d=(int)out_ndim-1;d>=0;d--) if(out_shape[d]>1){out_shape[d]=1;break;} }
                } else if (is_binary) {
                    if (ov) {
                        out_ndim = ov->shape.rank;
                        for (u32 i = 0; i < out_ndim; i++) out_shape[i] = ov->shape.dims[i];
                    } else {
                        out_ndim = ma->view.shape.rank;
                        for (u32 i = 0; i < out_ndim; i++) out_shape[i] = ma->view.shape.dims[i];
                    }
                } else if (is_movement && term_tag(b)==TAG_TEN) {
                    u32 bv_id = (u32)term_val(b); b_id = bv_id;
                    TensorMeta *bm = &ctx->tensors[bv_id];
                    if (uop==UOP_PERMUTE) {
                        out_ndim = ma->view.shape.rank;
                        u32 pf[MAX_DIM];
                        tensor_meta_read_u32(ctx, bv_id, pf, MAX_DIM);
                        for (u32 i=0;i<out_ndim;i++) out_shape[i]=ma->view.shape.dims[pf[i]];
                    } else {
                        out_ndim = bm->view.numel;
                        u32 df[MAX_DIM];
                        tensor_meta_read_u32(ctx, bv_id, df, MAX_DIM);
                        for (u32 i=0;i<out_ndim;i++) out_shape[i]=df[i];
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
                        u32 dims[MAX_DIM];
                        tensor_meta_read_u32(ctx, b_id, dims, MAX_DIM);
                        for (u32 i = 0; i < ns.rank; i++) ns.dims[i] = dims[i];
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
                                u32 elem_bytes = dtype_size(ma->dtype);
                                u8 *src = malloc((size_t)numel * elem_bytes);
                                u32 buf_bytes = (ma->view.offset > 0) ? (u32)ma->view.offset : 0;
                                for (u32 d = 0; d < ma->view.shape.rank; d++)
                                    if (ma->view.strides[d] > 0)
                                        buf_bytes += (ma->view.shape.dims[d]-1) * (u32)ma->view.strides[d];
                                buf_bytes += 1;
                                if (buf_bytes == 0) buf_bytes = 1;
                                u8 *raw = malloc((size_t)buf_bytes * elem_bytes);
                                ma->backend->buf_read(ma->buf_id, raw, (u64)buf_bytes * elem_bytes);
                                for (u32 flat = 0; flat < numel; flat++) {
                                    u32 rem = flat; i32 idx = ma->view.offset;
                                    int msk = 0;
                                    for (int d = (int)ma->view.shape.rank - 1; d >= 0; d--) {
                                        u32 coord = rem % ma->view.shape.dims[d];
                                        rem /= ma->view.shape.dims[d];
                                        if (ma->view.has_mask && (coord < ma->view.mask_begin[d] || coord >= ma->view.mask_end[d])) msk = 1;
                                        idx += (i32)coord * (ma->view.strides[d] > 0 ? ma->view.strides[d] : 0);
                                    }
                                    u8 *dstp = src + (size_t)flat * elem_bytes;
                                    if (msk || idx < 0 || (u32)idx >= buf_bytes) {
                                        memset(dstp, 0, elem_bytes);
                                    } else {
                                        memcpy(dstp, raw + (size_t)(u32)idx * elem_bytes, elem_bytes);
                                    }
                                }
                                free(raw);
                                ctx->tensors[mat_id].backend->buf_write(ctx->tensors[mat_id].buf_id, src,
                                                                        (u64)numel * elem_bytes);
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
                        u32 dims[MAX_DIM];
                        tensor_meta_read_u32(ctx, b_id, dims, MAX_DIM);
                        for (u32 i = 0; i < ns.rank; i++) ns.dims[i] = dims[i];
                        // Guard: GRAD backward may create rank-mismatched expands
                        if (ns.rank != ma->view.shape.rank) return t;
                        new_view = view_expand(ma->view, ns);
                        break;
                    }
                    case UOP_PERMUTE: {
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        u32 rank = mb->view.numel;
                        u32 axes[MAX_DIM];
                        tensor_meta_read_u32(ctx, b_id, axes, MAX_DIM);
                        new_view = view_permute(ma->view, axes);
                        break;
                    }
                    case UOP_SHRINK: {
                        // b is a 1D tensor: [start0, end0, start1, end1, ...]
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        u32 n_pairs = mb->view.numel;
                        u32 pairs[MAX_DIM * 2];
                        tensor_meta_read_u32(ctx, b_id, pairs, MAX_DIM * 2);
                        u32 starts[MAX_DIM], ends[MAX_DIM];
                        for (u32 i = 0; i < n_pairs / 2; i++) {
                            starts[i] = pairs[i * 2];
                            ends[i]   = pairs[i * 2 + 1];
                        }
                        new_view = view_shrink(ma->view, starts, ends);
                        break;
                    }
                    case UOP_PAD: {
                        // b is a 1D tensor: [before0, after0, before1, after1, ...]
                        // Zero-copy: use view_pad to set mask on the view
                        assert(b_id);
                        TensorMeta *mb = &ctx->tensors[b_id];
                        u32 n_pairs = mb->view.numel;
                        u32 pairs[MAX_DIM * 2];
                        tensor_meta_read_u32(ctx, b_id, pairs, MAX_DIM * 2);
                        u32 pad_before[MAX_DIM], pad_after[MAX_DIM];
                        for (u32 i = 0; i < n_pairs / 2; i++) {
                            pad_before[i] = pairs[i * 2];
                            pad_after[i]  = pairs[i * 2 + 1];
                        }
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

            // Compute ops: dispatch directly when args are TAG_TEN.
            // Single-op dispatch via fuse_build_kernel + backend.
            // FUSE propagation ensures inputs are TAG_TEN before we get here.
            {
                extern KernelEntry sched_kernels[];
                extern u32 sched_kernel_count;
                extern Term kid_results[];
                extern u32 kid_n_inputs[];

                KernelEntry ke;
                memset(&ke, 0, sizeof(ke));
                fuse_set_schedule_boundaries(NULL, NULL, NULL, 0, term_val(t));
                if (!fuse_build_kernel(ctx, t, &ke)) {
                    fuse_clear_schedule_boundaries();
                    return t; // can't fuse — stay WNF
                }
                fuse_clear_schedule_boundaries();

                u32 kid = sched_kernel_count++;
                ke.original_term = t;
                sched_kernels[kid] = ke;
                kid_results[kid] = term_era();
                kid_n_inputs[kid] = 0;

                u32 out_tid = tensor_create_unbacked(ctx, ke.out_shape, ma->dtype);
                ctx->tensors[out_tid].creator_op = UOP_KERNEL;
                ke.output_tid = out_tid;
                ke.raw_output_tid = out_tid;
                sched_kernels[kid] = ke;

                Term result = thvm_sched_dispatch_kernel(ctx, kid);
                kid_results[kid] = result;
                ctx->itrs++;
                RETURN_REDUCED(result);
            }
