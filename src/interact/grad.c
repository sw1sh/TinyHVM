            if (uop == UOP_GRAD) {
                // Disable rewrite fusion during backward: let deferred chains grow
                // longer. Without this, rewrite_apply materializes each backward
                // formula immediately (1 dispatch per formula = 82 dispatches).
                // With this, deferred chains accumulate and tensor_materialize
                // fuses them into fewer, larger kernels at ENSURE boundaries.
                u8 _saved_nf = ctx->no_fuse;
                u8 _saved_nga = ctx->no_grad_alloc;
                ctx->no_fuse = 0;
                ctx->no_grad_alloc = 1;
                #define GRAD_RETURN(r) do { \
                    Term _gr = (r); \
                    if (term_tag(_gr) == TAG_TOP) _gr = thvm_reduce(ctx, _gr); \
                    ctx->no_fuse = _saved_nf; \
                    ctx->no_grad_alloc = _saved_nga; \
                    return _gr; \
                } while(0)
                Term y  = heap_read(ctx, loc);
                Term gy = heap_read(ctx, loc + 1);
                Term x  = heap_read(ctx, loc + 2);
                // x is guaranteed WNF by the trampoline's TAG_TOP2 phase.

                // Deferred grad_prescan: run ONCE when forward provenance is ready.
                // Only the top-level GRAD needs prescan; recursive GRAD3s reuse the counts.
                if (term_tag(x) == TAG_CTR && term_tag(y) == TAG_TEN &&
                    !ctx->prescan_done) {
                    grad_prescan(ctx, y);
                    ctx->prescan_done = 1;
                }

                // GRAD-SUP: gradient through superposition
                // GRAD(&L{y0,y1}, gy, x) → &L{GRAD(y0, DP0_L(gy), DP0_L(x)), ...}
                if (term_tag(y) == TAG_SUP) {
                    u32 lab = term_ext(y);
                    u64 sup_loc = term_val(y);
                    Term y0 = heap_read(ctx, sup_loc + 0);
                    Term y1 = heap_read(ctx, sup_loc + 1);
                    u64 gy_dup = heap_alloc(ctx, 1);
                    heap_set(ctx, gy_dup, gy);
                    u64 x_dup = heap_alloc(ctx, 1);
                    heap_set(ctx, x_dup, x);
                    // Inline GRAD3 (macro not yet in scope)
                    u64 _l0 = heap_alloc(ctx, 3);
                    heap_set(ctx, _l0, y0);
                    heap_set(ctx, _l0+1, term_new(TAG_DP0, lab, gy_dup));
                    heap_set(ctx, _l0+2, term_new(TAG_DP0, lab, x_dup));
                    u64 _l1 = heap_alloc(ctx, 3);
                    heap_set(ctx, _l1, y1);
                    heap_set(ctx, _l1+1, term_new(TAG_DP1, lab, gy_dup));
                    heap_set(ctx, _l1+2, term_new(TAG_DP1, lab, x_dup));
                    ctx->itrs++;
                    GRAD_RETURN(thvm_sup(ctx, lab,
                        term_new(TAG_TOP, UOP_GRAD, _l0),
                        term_new(TAG_TOP, UOP_GRAD, _l1)));
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
                        /* Shape track: GRAD output shape = gy shape */ \
                        { const View *_gv = NULL; \
                          Term _gt = (gy_); \
                          if (term_tag(_gt) == TAG_TEN) { \
                              u32 _gid = (u32)term_val(_gt); \
                              if (_gid < ctx->tensor_count) _gv = &ctx->tensors[_gid].view; \
                          } else if (term_tag(_gt) == TAG_TOP) { \
                              _gv = st_get(term_val(_gt)); \
                          } \
                          if (_gv) st_set(_l, _gv); } \
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
                            // Must reduce with no_fuse=1 to prevent re-fusion cycle
                            u8 saved_nf2 = ctx->no_fuse;
                            ctx->no_fuse = 1;
                            Term orig = term_new(TAG_TOP, orig_uop2, orig_loc2);
                            Term unfused = thvm_reduce(ctx, orig);
                            ctx->no_fuse = saved_nf2;
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
                            // Auto-detect conv pattern: RESHAPE([BS,cout,OY,OX]) ← SUM(MUL(8D))
                            // TODO: enable matmul-based conv backward once dW verified correct
                            if (0 && !my->conv_input_id && my->view.shape.rank == 4 &&
                                ma && ma->creator_op == UOP_SUM && ma->src_ids[0] &&
                                ma->backend && ma->backend->op_mm) {
                                u32 mul_id = ma->src_ids[0];
                                TensorMeta *mmul = &ctx->tensors[mul_id];
                                if (mmul->creator_op == UOP_MUL && mmul->view.shape.rank == 8 &&
                                    mmul->src_ids[0] && mmul->src_ids[1]) {
                                    TensorMeta *mw_src = &ctx->tensors[mmul->src_ids[1]];
                                    // Weight shape: [1,groups,rcout,1,1,cin_g,KH,KW]
                                    if (mw_src->view.shape.dims[0] == 1 &&
                                        mw_src->view.shape.dims[3] == 1 && mw_src->view.shape.dims[4] == 1) {
                                        // Extract conv params from shapes
                                        u32 _groups = mw_src->view.shape.dims[1];
                                        u32 _cin_g = mw_src->view.shape.dims[5];
                                        u32 _KH = mw_src->view.shape.dims[6], _KW = mw_src->view.shape.dims[7];
                                        // Find original weight (walk through reshapes)
                                        u32 _w_id = mmul->src_ids[1];
                                        while (_w_id && ctx->tensors[_w_id].creator_op == UOP_RESHAPE)
                                            _w_id = ctx->tensors[_w_id].src_ids[0];
                                        // Find original input: walk src[0] chain looking for tensor
                                        // matching expected conv input shape [BS,cin,OY+KH-1,OX+KW-1]
                                        u32 _BS = my->view.shape.dims[0];
                                        u32 _OY = my->view.shape.dims[2], _OX = my->view.shape.dims[3];
                                        u32 _exp_H = _OY + _KH - 1, _exp_W = _OX + _KW - 1;
                                        u32 _exp_cin = _groups * _cin_g;
                                        u32 _inp_id = mmul->src_ids[0];
                                        for (u32 _d = 0; _d < 40 && _inp_id; _d++) {
                                            TensorMeta *_m = &ctx->tensors[_inp_id];
                                            if (_m->view.shape.rank == 4 &&
                                                _m->view.shape.dims[0] == _BS &&
                                                _m->view.shape.dims[1] == _exp_cin &&
                                                _m->view.shape.dims[2] == _exp_H &&
                                                _m->view.shape.dims[3] == _exp_W) break;
                                            if (_m->src_ids[0]) _inp_id = _m->src_ids[0];
                                            else break;
                                        }
                                        if (_w_id && _inp_id && _groups == 1 &&
                                            ctx->tensors[_inp_id].view.shape.rank == 4) {
                                            TensorMeta *_mi = &ctx->tensors[_inp_id];
                                            // Verify shapes match before activating
                                            u32 _cin = _mi->view.shape.dims[1];
                                            u32 _OY = my->view.shape.dims[2];
                                            u32 _OX = my->view.shape.dims[3];
                                            u32 _exp_IH = _OY + _KH - 1; // stride=1
                                            u32 _exp_IW = _OX + _KW - 1;
                                            if (_mi->view.shape.dims[2] == _exp_IH &&
                                                _mi->view.shape.dims[3] == _exp_IW &&
                                                _cin == _cin_g * _groups) {
                                                my->conv_input_id = _inp_id;
                                                my->conv_weight_id = _w_id;
                                                my->conv_groups = 1;
                                                my->conv_KH = (u8)_KH; my->conv_KW = (u8)_KW;
                                                my->conv_stride[0] = 1; my->conv_stride[1] = 1;
                                            }
                                        }
                                    }
                                }
                            }
                            // Specialized conv backward: use MPS matmul for dW and dX
                            if (my->conv_input_id && my->conv_weight_id &&
                                my->conv_groups == 1 && ma->backend->op_mm) {
                                u32 inp_id = my->conv_input_id;
                                u32 w_id = my->conv_weight_id;
                                TensorMeta *mi = &ctx->tensors[inp_id];
                                TensorMeta *mw = &ctx->tensors[w_id];
                                ENSURE(ctx, inp_id); ENSURE(ctx, w_id);
                                mi = &ctx->tensors[inp_id]; mw = &ctx->tensors[w_id];
                                u32 KH = my->conv_KH, KW = my->conv_KW;
                                u32 sH = my->conv_stride[0], sW = my->conv_stride[1];
                                // Build im2col view via pool
                                u32 BS = my->view.shape.dims[0];
                                u32 cout = my->view.shape.dims[1];
                                u32 OY = my->view.shape.dims[2];
                                u32 OX = my->view.shape.dims[3];
                                u32 cin = mi->view.shape.dims[1];
                                // im2col: thvm_pool creates [BS,cin,OY,OX,KH,KW]
                                Term inp_t = term_ten(inp_id, mi->dtype);
                                Term w_t = term_ten(w_id, mw->dtype);
                                Term im2col = thvm_pool(ctx, inp_t, (u32[]){KH,KW}, (u32[]){sH,sW}, 2);
                                // Permute to [BS,OY,OX,cin,KH,KW] then reshape to [M,K]
                                u32 M = BS*OY*OX, K = cin*KH*KW, N = cout;
                                Term im_perm = thvm_permute(ctx, im2col, (u32[]){0,2,3,1,4,5}, 6);
                                Term im_flat = thvm_reshape(ctx, im_perm, SHAPE(M, K));
                                // gy: [BS,cout,OY,OX] → [BS*OY*OX, cout] = [M, N]
                                Term gy_r = (term_tag(gy) == TAG_TEN) ? gy : thvm_reduce(ctx, gy);
                                // Permute gy to [BS,OY,OX,cout] then reshape to [M,N]
                                Term gy_perm = thvm_permute(ctx, gy_r, (u32[]){0,2,3,1}, 4);
                                Term gy_flat = thvm_reshape(ctx, gy_perm, SHAPE(M, N));
                                // dW = im2col^T @ gy = [K,M] @ [M,N] = [K,N]
                                Term im_T = thvm_permute(ctx, im_flat, (u32[]){1,0}, 2);
                                Term dW_flat = thvm_op(ctx, UOP_MM, im_T, gy_flat);
                                // Reshape to [cin,KH,KW,cout] → permute to [cout,cin,KH,KW]
                                Term dW_rs = thvm_reshape(ctx, dW_flat, shape_of((u32[]){cin,KH,KW,cout}, 4));
                                Term dW = thvm_permute(ctx, dW_rs, (u32[]){3,0,1,2}, 4);
                                // dX via matmul + col2im (both verified numerically correct)
                                Term w_flat = thvm_reshape(ctx, w_t, SHAPE(N, K));
                                Term dX_flat = thvm_op(ctx, UOP_MM, gy_flat, w_flat);
                                Term dX_6d = thvm_reshape(ctx, dX_flat,
                                    shape_of((u32[]){BS,OY,OX,cin,KH,KW}, 6));
                                Term dX_perm = thvm_permute(ctx, dX_6d, (u32[]){0,3,1,2,4,5}, 6);
                                u32 IH = mi->view.shape.dims[2], IW = mi->view.shape.dims[3];
                                Term dX_acc = term_era();
                                for (u32 _kh=0;_kh<KH;_kh++) for (u32 _kw=0;_kw<KW;_kw++) {
                                    u32 sh6[12];
                                    for (u32 _j=0;_j<6;_j++) { sh6[_j*2]=0; sh6[_j*2+1]=(u32[]){BS,cin,OY,OX,KH,KW}[_j]; }
                                    sh6[4*2]=_kh; sh6[4*2+1]=_kh+1;
                                    sh6[5*2]=_kw; sh6[5*2+1]=_kw+1;
                                    Term sl = thvm_shrink(ctx, dX_perm, sh6, 6);
                                    sl = thvm_reshape(ctx, sl, SHAPE(BS, cin, OY, OX));
                                    u32 pp[8]; memset(pp,0,sizeof(pp));
                                    pp[2*2]=_kh*sH; pp[2*2+1]=IH-OY*sH-_kh*sH;
                                    pp[3*2]=_kw*sW; pp[3*2+1]=IW-OX*sW-_kw*sW;
                                    sl = thvm_pad(ctx, sl, pp, 4);
                                    if (term_tag(dX_acc)==TAG_ERA) dX_acc=sl;
                                    else dX_acc=thvm_op(ctx,UOP_ADD,dX_acc,sl);
                                }
                                // Route: dW to weight via GRAD3, dX to input via GRAD3
                                // Both bypass the generic SUM→MUL backward entirely.
                                int _a_live = mi->requires_grad;
                                int _b_live = mw->requires_grad;
                                if (_a_live && _b_live) {
                                    // Decrement weight's grad_refs since the generic MUL
                                    // backward (which prescan counted) will never visit it.
                                    // Without this, the weight thinks it has an unvisited ref.
                                    // (prescan counted 1 ref from MUL→weight, but we bypass MUL)
                                    // Actually: prescan might count refs through view ops too.
                                    // Just trust that GRAD3 handles it.
                                    GRAD_RETURN(thvm_op(ctx, UOP_ADD,
                                        GRAD3(w_t, dW, x),
                                        GRAD3(inp_t, dX_acc, x)));
                                } else if (_b_live) {
                                    Term _bg = GRAD3(w_t, dW, x);
                                    if (term_tag(_bg) == TAG_TOP) GRAD_STEP(_bg);
                                    GRAD_RETURN(_bg);
                                } else if (_a_live) {
                                    UN_GRAD(dX_acc);
                                } else {
                                    goto generic_reshape_backward;
                                }
                            }
                            generic_reshape_backward:;
                            // Pool view backward: gy [..,OY,OX,KH,KW] → input [..,H,W]
                            if (my->pool_n_spatial == 2) {
                                u32 pkh = my->pool_kernel[0], pkw = my->pool_kernel[1];
                                u32 psh = my->pool_stride[0], psw = my->pool_stride[1];
                                u32 _r = my->view.shape.rank;
                                u32 bd2 = _r - 4; // batch dims (before OY,OX,KH,KW)
                                u32 ih = ma->view.shape.dims[bd2], iw = ma->view.shape.dims[bd2+1];
                                if (psh == pkh && psw == pkw) {
                                    // Non-overlapping: permute [..,OY,OX,KH,KW]→[..,OY,KH,OX,KW], reshape
                                    u32 pp[MAX_DIM], _pi=0;
                                    for (u32 _j=0;_j<bd2;_j++) pp[_pi++]=_j;
                                    pp[_pi++]=bd2; pp[_pi++]=bd2+2; pp[_pi++]=bd2+1; pp[_pi++]=bd2+3;
                                    Term gp = thvm_permute(ctx, gy, pp, _r);
                                    u32 tgt[MAX_DIM];
                                    for (u32 _j=0;_j<bd2;_j++) tgt[_j]=ma->view.shape.dims[_j];
                                    tgt[bd2]=ih; tgt[bd2+1]=iw;
                                    f32 tgtf[MAX_DIM];
                                    for (u32 _j=0;_j<bd2+2;_j++) tgtf[_j]=(f32)tgt[_j];
                                    UN_GRAD(thvm_op(ctx, UOP_RESHAPE, gp,
                                        thvm_tensor(ctx, tgtf, SHAPE(bd2+2))));
                                }
                                // Overlapping (k > s): col2im via pad+sum over kernel positions
                                // grad_input[..,h,w] = sum_{kh,kw} gy[..,h-kh,w-kw,kh,kw]
                                // For each (kh,kw): extract slice, pad to [..,H,W], accumulate
                                u32 _oy = my->view.shape.dims[bd2], _ox = my->view.shape.dims[bd2+1];
                                Term acc = term_era();
                                for (u32 _kh=0;_kh<pkh;_kh++) for (u32 _kw=0;_kw<pkw;_kw++) {
                                    // Shrink to extract slice gy[..,_kh,_kw] at dims bd2+2, bd2+3
                                    u32 sh[MAX_DIM*2];
                                    for (u32 _j=0;_j<_r;_j++) {
                                        sh[_j*2]=0; sh[_j*2+1]=my->view.shape.dims[_j];
                                    }
                                    sh[(bd2+2)*2]=_kh; sh[(bd2+2)*2+1]=_kh+1;
                                    sh[(bd2+3)*2]=_kw; sh[(bd2+3)*2+1]=_kw+1;
                                    Term sl = thvm_shrink(ctx, gy, sh, _r);
                                    // Reshape to remove kernel dims: [..,OY,OX,1,1] → [..,OY,OX]
                                    u32 slr[MAX_DIM]; u32 slrk=0;
                                    for (u32 _j=0;_j<bd2;_j++) slr[slrk++]=my->view.shape.dims[_j];
                                    slr[slrk++]=_oy; slr[slrk++]=_ox;
                                    sl = thvm_reshape(ctx, sl, shape_of(slr, slrk));
                                    // Pad to [..,H,W]: pad_top=_kh*psh, pad_bottom=ih-_oy*psh-_kh*psh
                                    // For stride=1: pad_top=_kh, pad_bottom=pkh-1-_kh
                                    u32 pp2[MAX_DIM*2];
                                    memset(pp2,0,sizeof(pp2));
                                    pp2[(bd2)*2]  =_kh*psh;
                                    pp2[(bd2)*2+1]=ih - _oy*psh - _kh*psh;
                                    pp2[(bd2+1)*2]  =_kw*psw;
                                    pp2[(bd2+1)*2+1]=iw - _ox*psw - _kw*psw;
                                    sl = thvm_pad(ctx, sl, pp2, slrk);
                                    if (term_tag(acc) == TAG_ERA) acc = sl;
                                    else acc = thvm_op(ctx, UOP_ADD, acc, sl);
                                }
                                UN_GRAD(acc);
                            }
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
                // y is guaranteed WNF by the trampoline (Phase 1: TAG_TOP2 reduces arg0)
                // Dead path: TAG_TOP y can't happen anymore.
                GRAD_RETURN(term_era());
                #undef GRAD_RETURN
            }
