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
    // Return result directly — the trampoline handles TAG_TOP results
    // via `next = r; goto enter;` (no need to force-reduce here).
    #define RETURN_REDUCED(result) do { return (result); } while(0)
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
                u8 _saved_nga = ctx->no_grad_alloc;
                ctx->no_fuse = 0;
                ctx->no_grad_alloc = 1;
                // Return from GRAD handler — restore flags, let trampoline reduce
                #define GRAD_RETURN(r) do { \
                    ctx->no_fuse = _saved_nf; \
                    ctx->no_grad_alloc = _saved_nga; \
                    return (r); \
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

            u32 a_id = (u32)term_val(a);
            TensorMeta *ma = &ctx->tensors[a_id];

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
                            Term axes_e = thvm_reduce(ctx, sum_arg_e);
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
        }

        // TAG_APP: beta reduction + APP-SUP distribution
        case TAG_APP: {
            u64 loc = term_val(t);
            Term fun = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, fun);

            // APP-SUP (fun position): (&L{f0,f1} arg) → !&L{a0,a1}=arg; &L{(f0 a0),(f1 a1)}
            if (term_tag(fun) == TAG_SUP) {
                u32 lab = term_ext(fun);
                u64 sup_loc = term_val(fun);
                Term f0 = heap_read(ctx, sup_loc + 0);
                Term f1 = heap_read(ctx, sup_loc + 1);
                Term arg = heap_read(ctx, loc + 1);
                // Clone arg with the SUP's label
                u64 dup_loc = heap_alloc(ctx, 1);
                heap_set(ctx, dup_loc, arg);
                Term arg0 = term_new(TAG_DP0, lab, dup_loc);
                Term arg1 = term_new(TAG_DP1, lab, dup_loc);
                ctx->itrs++;
                t = thvm_sup(ctx, lab,
                    thvm_app(ctx, f0, arg0),
                    thvm_app(ctx, f1, arg1));
                goto inet_step;
            }

            // APP-BRI: (θx.body arg) → body[x ← arg]  (same beta rule as LAM)
            if (term_tag(fun) == TAG_BRI) {
                u64 bri_loc = term_val(fun);
                Term arg = heap_read(ctx, loc + 1);
                heap_set(ctx, bri_loc, arg);
                ctx->itrs++;
                t = heap_read(ctx, bri_loc + 1);
                goto inet_step;
            }

            // APP-LAM: beta reduction — (λx.body arg) → body[x ← arg]
            if (term_tag(fun) == TAG_LAM) {
                u64 lam_loc = term_val(fun);
                Term arg = heap_read(ctx, loc + 1);
                heap_set(ctx, lam_loc, arg);      // link: write arg at var slot
                ctx->itrs++;
                t = heap_read(ctx, lam_loc + 1);  // body
                goto inet_step;
            }
            // APP-TEN: discard tensor, return arg (sequencing)
            if (term_tag(fun) == TAG_TEN) {
                tensor_decref(ctx, (u32)term_val(fun));
                ctx->itrs++;
                t = heap_read(ctx, loc + 1);
                goto inet_step;
            }
            // APP-ERA: erasure propagation
            if (term_tag(fun) == TAG_ERA) {
                ctx->itrs++;
                t = heap_read(ctx, loc + 1);
                goto inet_step;
            }

            // APP-MAT: pattern match fired by application
            if (term_tag(fun) == TAG_MAT) {
                Term arg = thvm_reduce(ctx, heap_read(ctx, loc + 1));
                heap_set(ctx, loc + 1, arg);

                // APP-MAT-SUP: (MAT &L{a,b}) → clone MAT, &L{(MAT₀ a), (MAT₁ b)}
                if (term_tag(arg) == TAG_SUP) {
                    u32 lab = term_ext(arg);
                    u64 sloc = term_val(arg);
                    Term a = heap_read(ctx, sloc + 0);
                    Term b = heap_read(ctx, sloc + 1);
                    u64 dup = heap_alloc(ctx, 1);
                    heap_set(ctx, dup, fun);
                    ctx->itrs++;
                    t = thvm_sup(ctx, lab,
                        thvm_app(ctx, term_new(TAG_DP0, lab, dup), a),
                        thvm_app(ctx, term_new(TAG_DP1, lab, dup), b));
                    goto inet_step;
                }
                // APP-MAT-USP: same for unordered SUP
                if (term_tag(arg) == TAG_USP) {
                    u32 lab = term_ext(arg);
                    u64 sloc = term_val(arg);
                    Term a = heap_read(ctx, sloc + 0);
                    Term b = heap_read(ctx, sloc + 1);
                    u64 dup = heap_alloc(ctx, 1);
                    heap_set(ctx, dup, fun);
                    ctx->itrs++;
                    t = thvm_usp(ctx, lab,
                        thvm_app(ctx, term_new(TAG_UDP, lab, dup), a),
                        thvm_app(ctx, term_new(TAG_UDP, lab, dup), b));
                    goto inet_step;
                }

                // APP-MAT-NUM: match on numeric tag
                if (term_tag(arg) == TAG_NUM) {
                    u64 mat_loc = term_val(fun);
                    u32 match_tag = term_ext(fun);
                    u32 num_val = term_as_u32(arg);
                    ctx->itrs++;
                    if (match_tag == num_val) {
                        t = heap_read(ctx, mat_loc + 0);  // handler
                    } else {
                        t = thvm_app(ctx, heap_read(ctx, mat_loc + 1), arg);  // fallback(value)
                    }
                    goto inet_step;
                }

                // APP-MAT-ERA: erased argument → ERA
                if (term_tag(arg) == TAG_ERA) return term_era();

                return t;  // stuck (arg not yet reduced to matchable form)
            }

            // APP-USP (fun position): same as APP-SUP but preserves unordered
            if (term_tag(fun) == TAG_USP) {
                u32 lab = term_ext(fun);
                u64 usp_loc = term_val(fun);
                Term f0 = heap_read(ctx, usp_loc + 0);
                Term f1 = heap_read(ctx, usp_loc + 1);
                Term arg = heap_read(ctx, loc + 1);
                u64 dup_loc = heap_alloc(ctx, 1);
                heap_set(ctx, dup_loc, arg);
                ctx->itrs++;
                t = thvm_usp(ctx, lab,
                    thvm_app(ctx, f0, term_new(TAG_UDP, lab, dup_loc)),
                    thvm_app(ctx, f1, term_new(TAG_UDP, lab, dup_loc)));
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
        // ONE rule, branching on dp_index for which projection to return.
        // Label in EXT field: same label SUP → annihilate, different → commute.
        case TAG_DP0:
        case TAG_DP1: {
            u32 dp_index = (tag == TAG_DP1) ? 1 : 0;
            u32 dup_label = term_ext(t);
            u64 dup_loc = term_val(t);
            Term val = thvm_reduce(ctx, heap_read(ctx, dup_loc));
            heap_set(ctx, dup_loc, val);

            // DUP ⊳ SUP
            if (term_tag(val) == TAG_SUP) {
                u32 sup_label = term_ext(val);
                u64 sup_loc = term_val(val);
                ctx->itrs++;

                if (dup_label == sup_label) {
                    // ANNIHILATION: same label — project directly
                    Term tm0 = heap_read(ctx, sup_loc + 0);
                    Term tm1 = heap_read(ctx, sup_loc + 1);
                    if (dp_index == 0) {
                        heap_set(ctx, dup_loc, tm1); // other projection gets tm1
                        t = tm0;
                    } else {
                        heap_set(ctx, dup_loc, tm0); // other projection gets tm0
                        t = tm1;
                    }
                    goto inet_step;
                } else {
                    // COMMUTATION: different label — pass through, preserve both labels
                    Term b = heap_read(ctx, sup_loc + 1);
                    // Reuse sup_loc for du0 (a is already at sup_loc+0), alloc new for du1
                    u64 du0 = sup_loc;
                    u64 du1 = heap_alloc(ctx, 1);
                    heap_set(ctx, du1, b);
                    // Create 2 new SUPs with the ORIGINAL sup_label
                    u64 su0 = heap_alloc(ctx, 2);
                    u64 su1 = heap_alloc(ctx, 2);
                    heap_set(ctx, su0 + 0, term_new(TAG_DP0, dup_label, du0));
                    heap_set(ctx, su0 + 1, term_new(TAG_DP0, dup_label, du1));
                    heap_set(ctx, su1 + 0, term_new(TAG_DP1, dup_label, du0));
                    heap_set(ctx, su1 + 1, term_new(TAG_DP1, dup_label, du1));
                    Term new_sup0 = term_new(TAG_SUP, sup_label, su0);
                    Term new_sup1 = term_new(TAG_SUP, sup_label, su1);
                    if (dp_index == 0) {
                        heap_set(ctx, dup_loc, new_sup1);
                        t = new_sup0;
                    } else {
                        heap_set(ctx, dup_loc, new_sup0);
                        t = new_sup1;
                    }
                    goto inet_step;
                }
            }

            // DUP ⊳ LAM: commutation — duplicate lambda
            if (term_tag(val) == TAG_LAM) {
                u64 lam_loc = term_val(val);
                Term body = heap_read(ctx, lam_loc + 1);
                // Clone body with DUP label (single shared slot)
                u64 bdup = heap_alloc(ctx, 1);
                heap_set(ctx, bdup, body);
                // Create two fresh lambdas
                Term var0, var1;
                Term lam0 = thvm_lam(ctx, &var0, term_new(TAG_DP0, dup_label, bdup));
                Term lam1 = thvm_lam(ctx, &var1, term_new(TAG_DP1, dup_label, bdup));
                // Original var gets SUP of new vars (with dup_label)
                u64 vsup = heap_alloc(ctx, 2);
                heap_set(ctx, vsup + 0, var0);
                heap_set(ctx, vsup + 1, var1);
                heap_set(ctx, lam_loc, term_new(TAG_SUP, dup_label, vsup));
                ctx->itrs++;
                if (dp_index == 0) {
                    heap_set(ctx, dup_loc, lam1);
                    t = lam0;
                } else {
                    heap_set(ctx, dup_loc, lam0);
                    t = lam1;
                }
                goto inet_step;
            }

            // DUP ⊳ BRI: commutation — duplicate bridge (same as DUP-LAM but TAG_BRI)
            if (term_tag(val) == TAG_BRI) {
                u64 bri_loc = term_val(val);
                Term body = heap_read(ctx, bri_loc + 1);
                u64 bdup = heap_alloc(ctx, 1);
                heap_set(ctx, bdup, body);
                Term var0, var1;
                Term bri0 = thvm_bri(ctx, &var0, term_new(TAG_DP0, dup_label, bdup));
                Term bri1 = thvm_bri(ctx, &var1, term_new(TAG_DP1, dup_label, bdup));
                u64 vsup = heap_alloc(ctx, 2);
                heap_set(ctx, vsup + 0, var0);
                heap_set(ctx, vsup + 1, var1);
                heap_set(ctx, bri_loc, term_new(TAG_SUP, dup_label, vsup));
                ctx->itrs++;
                if (dp_index == 0) {
                    heap_set(ctx, dup_loc, bri1);
                    t = bri0;
                } else {
                    heap_set(ctx, dup_loc, bri0);
                    t = bri1;
                }
                goto inet_step;
            }

            // DUP ⊳ ANN: dup through annotation — DUP both term and type
            if (term_tag(val) == TAG_ANN) {
                u64 ann_loc = term_val(val);
                Term inner = heap_read(ctx, ann_loc);
                Term type  = heap_read(ctx, ann_loc + 1);
                u64 idup = heap_alloc(ctx, 1);
                heap_set(ctx, idup, inner);
                u64 tdup = heap_alloc(ctx, 1);
                heap_set(ctx, tdup, type);
                Term a0 = thvm_ann(ctx, term_new(TAG_DP0, dup_label, idup),
                                        term_new(TAG_DP0, dup_label, tdup));
                Term a1 = thvm_ann(ctx, term_new(TAG_DP1, dup_label, idup),
                                        term_new(TAG_DP1, dup_label, tdup));
                ctx->itrs++;
                if (dp_index == 0) {
                    heap_set(ctx, dup_loc, a1);
                    t = a0;
                } else {
                    heap_set(ctx, dup_loc, a0);
                    t = a1;
                }
                goto inet_step;
            }

            // DUP ⊳ atoms: copy (both projections get same value)
            if (term_tag(val) == TAG_TEN) return val;
            if (term_tag(val) == TAG_ERA) return val;
            if (term_tag(val) == TAG_NUM) return val;
            if (term_tag(val) == TAG_ANY) return val;

            // DUP ⊳ USP: commutation (same as DUP-SUP but preserves TAG_USP)
            if (term_tag(val) == TAG_USP) {
                u32 usp_label = term_ext(val);
                u64 usp_loc = term_val(val);
                Term b = heap_read(ctx, usp_loc + 1);
                u64 du0 = usp_loc;  // reuse (a already at usp_loc+0)
                u64 du1 = heap_alloc(ctx, 1);
                heap_set(ctx, du1, b);
                u64 su0 = heap_alloc(ctx, 2);
                u64 su1 = heap_alloc(ctx, 2);
                heap_set(ctx, su0 + 0, term_new(TAG_DP0, dup_label, du0));
                heap_set(ctx, su0 + 1, term_new(TAG_DP0, dup_label, du1));
                heap_set(ctx, su1 + 0, term_new(TAG_DP1, dup_label, du0));
                heap_set(ctx, su1 + 1, term_new(TAG_DP1, dup_label, du1));
                Term new_usp0 = term_new(TAG_USP, usp_label, su0);
                Term new_usp1 = term_new(TAG_USP, usp_label, su1);
                ctx->itrs++;
                if (dp_index == 0) {
                    heap_set(ctx, dup_loc, new_usp1);
                    t = new_usp0;
                } else {
                    heap_set(ctx, dup_loc, new_usp0);
                    t = new_usp1;
                }
                goto inet_step;
            }

            // DUP ⊳ EQL/AND/OR/MAT: generic 2-slot compound duplication (same as DUP-OP2)
            // DUP ⊳ OP2/APP: generic compound node duplication (DUP-NOD)
            if (term_tag(val) == TAG_OP2 || term_tag(val) == TAG_APP ||
                term_tag(val) == TAG_EQL || term_tag(val) == TAG_AND ||
                term_tag(val) == TAG_OR  || term_tag(val) == TAG_MAT) {
                u64 val_loc = term_val(val);
                u64 r0 = heap_alloc(ctx, 2);
                u64 r1 = heap_alloc(ctx, 2);
                for (u32 i = 0; i < 2; i++) {
                    Term child = heap_read(ctx, val_loc + i);
                    u64 cdup = heap_alloc(ctx, 1);
                    heap_set(ctx, cdup, child);
                    heap_set(ctx, r0 + i, term_new(TAG_DP0, dup_label, cdup));
                    heap_set(ctx, r1 + i, term_new(TAG_DP1, dup_label, cdup));
                }
                Term n0 = term_new(term_tag(val), term_ext(val), r0);
                Term n1 = term_new(term_tag(val), term_ext(val), r1);
                ctx->itrs++;
                if (dp_index == 0) {
                    heap_set(ctx, dup_loc, n1);
                    t = n0;
                } else {
                    heap_set(ctx, dup_loc, n0);
                    t = n1;
                }
                goto inet_step;
            }

            // Not yet reducible — return DUP as-is
            return t;
        }


        case TAG_OP2: {
            u64 loc = term_val(t);
            u32 opr = term_ext(t);
            Term x = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, x);

            // OP2-SUP (left): OP2(opr, &L{x0,x1}, y) → &L{OP2(opr,x0,y0), OP2(opr,x1,y1)}
            if (term_tag(x) == TAG_SUP) {
                u32 lab = term_ext(x);
                u64 sup_loc = term_val(x);
                Term x0 = heap_read(ctx, sup_loc + 0);
                Term x1 = heap_read(ctx, sup_loc + 1);
                Term y = heap_read(ctx, loc + 1);
                // Clone y with SUP's label
                u64 dup_loc = heap_alloc(ctx, 1);
                heap_set(ctx, dup_loc, y);
                Term y0 = term_new(TAG_DP0, lab, dup_loc);
                Term y1 = term_new(TAG_DP1, lab, dup_loc);
                ctx->itrs++;
                t = thvm_sup(ctx, lab,
                    thvm_op2(ctx, opr, x0, y0),
                    thvm_op2(ctx, opr, x1, y1));
                goto inet_step;
            }

            Term y = thvm_reduce(ctx, heap_read(ctx, loc + 1));
            heap_set(ctx, loc + 1, y);

            // OP2-SUP (right): x is already NUM (atom), just reuse — no DUP needed
            if (term_tag(y) == TAG_SUP) {
                u32 lab = term_ext(y);
                u64 sup_loc = term_val(y);
                Term y0 = heap_read(ctx, sup_loc + 0);
                Term y1 = heap_read(ctx, sup_loc + 1);
                ctx->itrs++;
                t = thvm_sup(ctx, lab,
                    thvm_op2(ctx, opr, x, y0),
                    thvm_op2(ctx, opr, x, y1));
                goto inet_step;
            }

            if (term_tag(x) == TAG_NUM && term_tag(y) == TAG_NUM) {
                u32 xv = term_as_u32(x), yv = term_as_u32(y), r;
                switch (opr) {
                    case 0: r = xv + yv; break;
                    case 1: r = xv - yv; break;
                    case 2: r = xv * yv; break;
                    case 3: r = yv ? xv / yv : 0; break;
                    case 4: r = (xv == yv) ? 1 : 0; break;  // EQ
                    case 5: r = yv ? xv % yv : 0; break;     // MOD
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

        // TAG_BRI: bridge — returned as-is until applied (WNF, like LAM)
        case TAG_BRI:
            return t;

        // TAG_ANN: annotation — transparent, strip and return inner term
        case TAG_ANN: {
            u64 loc = term_val(t);
            ctx->itrs++;
            t = heap_read(ctx, loc);  // the term (slot 0)
            goto inet_step;
        }

        // TAG_DSU: dynamic SUP — reduce label_expr to NUM, then create normal SUP
        case TAG_DSU: {
            u64 loc = term_val(t);
            Term label_expr = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, label_expr);
            if (term_tag(label_expr) != TAG_NUM) return t;  // not ready
            u32 label = term_as_u32(label_expr);
            Term a = heap_read(ctx, loc + 1);
            Term b = heap_read(ctx, loc + 2);
            ctx->itrs++;
            t = thvm_sup(ctx, label, a, b);
            goto inet_step;
        }

        // TAG_DDU: dynamic DUP — reduce label_expr to NUM, then dup val, apply bod
        case TAG_DDU: {
            u64 loc = term_val(t);
            Term label_expr = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, label_expr);
            if (term_tag(label_expr) != TAG_NUM) return t;  // not ready
            u32 label = term_as_u32(label_expr);
            Term val = heap_read(ctx, loc + 1);
            Term bod = heap_read(ctx, loc + 2);
            // DUP val with dynamic label
            Term dp0, dp1;
            thvm_dup(ctx, label, val, &dp0, &dp1);
            ctx->itrs++;
            // Apply bod to both copies: APP(APP(bod, dp0), dp1)
            t = thvm_app(ctx, thvm_app(ctx, bod, dp0), dp1);
            goto inet_step;
        }

        // TAG_INC: priority wrapper — transparent during normal reduction
        case TAG_INC: {
            u64 loc = term_val(t);
            ctx->itrs++;
            t = heap_read(ctx, loc);
            goto inet_step;
        }

        // ── Type enumeration machinery ───────────────────────────────────

        // TAG_EQL: structural equality — reduce both sides, compare
        // Pattern: same as TAG_OP2 (reduce left, check SUP, reduce right, check SUP, compare)
        case TAG_EQL: {
            u64 loc = term_val(t);
            // Reduce left operand
            Term a = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, a);

            // EQL-SUP-L: (&L{a0,a1} === b) → clone b, &L{(a0===B₀), (a1===B₁)}
            if (term_tag(a) == TAG_SUP) {
                u32 lab = term_ext(a);
                u64 sloc = term_val(a);
                Term a0 = heap_read(ctx, sloc + 0);
                Term a1 = heap_read(ctx, sloc + 1);
                Term b = heap_read(ctx, loc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, b);
                ctx->itrs++;
                t = thvm_sup(ctx, lab,
                    thvm_eql(ctx, a0, term_new(TAG_DP0, lab, dup)),
                    thvm_eql(ctx, a1, term_new(TAG_DP1, lab, dup)));
                goto inet_step;
            }
            // EQL-USP-L: same for unordered SUP
            if (term_tag(a) == TAG_USP) {
                u32 lab = term_ext(a);
                u64 sloc = term_val(a);
                Term a0 = heap_read(ctx, sloc + 0);
                Term a1 = heap_read(ctx, sloc + 1);
                Term b = heap_read(ctx, loc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, b);
                ctx->itrs++;
                t = thvm_usp(ctx, lab,
                    thvm_eql(ctx, a0, term_new(TAG_UDP, lab, dup)),
                    thvm_eql(ctx, a1, term_new(TAG_UDP, lab, dup)));
                goto inet_step;
            }
            // EQL-ERA-L: (ERA === b) → ERA
            if (term_tag(a) == TAG_ERA) return term_era();
            // EQL-ANY-L: (ANY === b) → 1
            if (term_tag(a) == TAG_ANY) { ctx->itrs++; return term_num_u32(1); }
            // EQL-INC-L: (INC(a) === b) → INC(a === b)
            if (term_tag(a) == TAG_INC) {
                Term inner_a = heap_read(ctx, term_val(a));
                Term b = heap_read(ctx, loc + 1);
                ctx->itrs++;
                t = thvm_inc(ctx, thvm_eql(ctx, inner_a, b));
                goto inet_step;
            }

            // Reduce right operand
            Term b = thvm_reduce(ctx, heap_read(ctx, loc + 1));
            heap_set(ctx, loc + 1, b);

            // EQL-SUP-R: (a === &L{b0,b1}) → clone a, &L{(A₀===b0), (A₁===b1)}
            if (term_tag(b) == TAG_SUP) {
                u32 lab = term_ext(b);
                u64 sloc = term_val(b);
                Term b0 = heap_read(ctx, sloc + 0);
                Term b1 = heap_read(ctx, sloc + 1);
                // a is already NUM or other atom (no SUP), no DUP needed — reuse directly
                ctx->itrs++;
                t = thvm_sup(ctx, lab,
                    thvm_eql(ctx, a, b0),
                    thvm_eql(ctx, a, b1));
                goto inet_step;
            }
            // EQL-USP-R
            if (term_tag(b) == TAG_USP) {
                u32 lab = term_ext(b);
                u64 sloc = term_val(b);
                Term b0 = heap_read(ctx, sloc + 0);
                Term b1 = heap_read(ctx, sloc + 1);
                ctx->itrs++;
                t = thvm_usp(ctx, lab,
                    thvm_eql(ctx, a, b0),
                    thvm_eql(ctx, a, b1));
                goto inet_step;
            }
            // EQL-ERA-R: (a === ERA) → ERA
            if (term_tag(b) == TAG_ERA) return term_era();
            // EQL-ANY-R: (a === ANY) → 1
            if (term_tag(b) == TAG_ANY) { ctx->itrs++; return term_num_u32(1); }
            // EQL-INC-R: (a === INC(b)) → INC(a === b)
            if (term_tag(b) == TAG_INC) {
                Term inner_b = heap_read(ctx, term_val(b));
                ctx->itrs++;
                t = thvm_inc(ctx, thvm_eql(ctx, a, inner_b));
                goto inet_step;
            }

            // EQL-NUM: (#a === #b) → NUM(a == b ? 1 : 0)
            if (term_tag(a) == TAG_NUM && term_tag(b) == TAG_NUM) {
                ctx->itrs++;
                return term_num_u32(term_as_u32(a) == term_as_u32(b) ? 1 : 0);
            }
            // EQL-LAM: (λx.af === λy.bf) → compare bodies with fresh shared var
            if (term_tag(a) == TAG_LAM && term_tag(b) == TAG_LAM) {
                u64 a_loc = term_val(a);
                u64 b_loc = term_val(b);
                Term a_body = heap_read(ctx, a_loc + 1);
                Term b_body = heap_read(ctx, b_loc + 1);
                // Substitute both binders with a fresh NUM (unique ID)
                u32 fresh = ctx->next_sup_label++;
                Term fresh_var = term_num_u32(fresh);
                heap_set(ctx, a_loc, fresh_var);  // substitute a's VAR
                heap_set(ctx, b_loc, fresh_var);  // substitute b's VAR
                ctx->itrs++;
                t = thvm_eql(ctx, a_body, b_body);
                goto inet_step;
            }
            // Different types → not equal
            ctx->itrs++;
            return term_num_u32(0);
        }

        // TAG_AND: short-circuit boolean AND — reduce left, short-circuit on 0
        case TAG_AND: {
            u64 loc = term_val(t);
            Term a = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, a);

            // AND-SUP: distribute
            if (term_tag(a) == TAG_SUP) {
                u32 lab = term_ext(a);
                u64 sloc = term_val(a);
                Term a0 = heap_read(ctx, sloc + 0);
                Term a1 = heap_read(ctx, sloc + 1);
                Term b = heap_read(ctx, loc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, b);
                ctx->itrs++;
                t = thvm_sup(ctx, lab,
                    thvm_and(ctx, a0, term_new(TAG_DP0, lab, dup)),
                    thvm_and(ctx, a1, term_new(TAG_DP1, lab, dup)));
                goto inet_step;
            }
            if (term_tag(a) == TAG_USP) {
                u32 lab = term_ext(a);
                u64 sloc = term_val(a);
                Term a0 = heap_read(ctx, sloc + 0);
                Term a1 = heap_read(ctx, sloc + 1);
                Term b = heap_read(ctx, loc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, b);
                ctx->itrs++;
                t = thvm_usp(ctx, lab,
                    thvm_and(ctx, a0, term_new(TAG_UDP, lab, dup)),
                    thvm_and(ctx, a1, term_new(TAG_UDP, lab, dup)));
                goto inet_step;
            }
            // AND-ZER: short-circuit false
            if (term_tag(a) == TAG_NUM && term_as_u32(a) == 0) {
                ctx->itrs++;
                return term_num_u32(0);
            }
            // AND-ONE: nonzero → pass through to b
            if (term_tag(a) == TAG_NUM) {
                ctx->itrs++;
                t = heap_read(ctx, loc + 1);
                goto inet_step;
            }
            // AND-ERA: erased → ERA
            if (term_tag(a) == TAG_ERA) return term_era();
            return t;
        }

        // TAG_OR: short-circuit boolean OR — reduce left, short-circuit on nonzero
        case TAG_OR: {
            u64 loc = term_val(t);
            Term a = thvm_reduce(ctx, heap_read(ctx, loc));
            heap_set(ctx, loc, a);

            // OR-SUP: distribute
            if (term_tag(a) == TAG_SUP) {
                u32 lab = term_ext(a);
                u64 sloc = term_val(a);
                Term a0 = heap_read(ctx, sloc + 0);
                Term a1 = heap_read(ctx, sloc + 1);
                Term b = heap_read(ctx, loc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, b);
                ctx->itrs++;
                t = thvm_sup(ctx, lab,
                    thvm_or(ctx, a0, term_new(TAG_DP0, lab, dup)),
                    thvm_or(ctx, a1, term_new(TAG_DP1, lab, dup)));
                goto inet_step;
            }
            if (term_tag(a) == TAG_USP) {
                u32 lab = term_ext(a);
                u64 sloc = term_val(a);
                Term a0 = heap_read(ctx, sloc + 0);
                Term a1 = heap_read(ctx, sloc + 1);
                Term b = heap_read(ctx, loc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, b);
                ctx->itrs++;
                t = thvm_usp(ctx, lab,
                    thvm_or(ctx, a0, term_new(TAG_UDP, lab, dup)),
                    thvm_or(ctx, a1, term_new(TAG_UDP, lab, dup)));
                goto inet_step;
            }
            // OR-ZER: zero → pass through to b
            if (term_tag(a) == TAG_NUM && term_as_u32(a) == 0) {
                ctx->itrs++;
                t = heap_read(ctx, loc + 1);
                goto inet_step;
            }
            // OR-ONE: nonzero → short-circuit true
            if (term_tag(a) == TAG_NUM) {
                ctx->itrs++;
                return term_num_u32(1);
            }
            // OR-ERA: erased → ERA
            if (term_tag(a) == TAG_ERA) return term_era();
            return t;
        }

        // ── Unordered DUP ────────────────────────────────────────────────

        // TAG_UDP: unordered DUP — single output port, infinite producer
        case TAG_UDP: {
            u32 udp_label = term_ext(t);
            u64 udp_loc = term_val(t);
            Term val = thvm_reduce(ctx, heap_read(ctx, udp_loc));
            heap_set(ctx, udp_loc, val);

            // UDP ⊳ USP (same label): consume ONE branch, keep producing from other
            if (term_tag(val) == TAG_USP && term_ext(val) == udp_label) {
                u64 usp_loc = term_val(val);
                Term a = heap_read(ctx, usp_loc + 0);
                Term b = heap_read(ctx, usp_loc + 1);
                // UDP keeps pointing to b (unconsumed remainder)
                heap_set(ctx, udp_loc, b);
                ctx->itrs++;
                t = a;
                goto inet_step;
            }

            // UDP ⊳ USP (different label): commutation
            if (term_tag(val) == TAG_USP) {
                u32 usp_label = term_ext(val);
                u64 usp_loc = term_val(val);
                Term a = heap_read(ctx, usp_loc + 0);
                Term b = heap_read(ctx, usp_loc + 1);
                u64 du_a = heap_alloc(ctx, 1); heap_set(ctx, du_a, a);
                u64 du_b = heap_alloc(ctx, 1); heap_set(ctx, du_b, b);
                ctx->itrs++;
                t = thvm_usp(ctx, usp_label,
                    term_new(TAG_UDP, udp_label, du_a),
                    term_new(TAG_UDP, udp_label, du_b));
                goto inet_step;
            }

            // UDP ⊳ ordered SUP: commutation (distribute UDP over SUP branches)
            if (term_tag(val) == TAG_SUP) {
                u32 sup_label = term_ext(val);
                u64 sup_loc = term_val(val);
                Term a = heap_read(ctx, sup_loc + 0);
                Term b = heap_read(ctx, sup_loc + 1);
                u64 du_a = heap_alloc(ctx, 1); heap_set(ctx, du_a, a);
                u64 du_b = heap_alloc(ctx, 1); heap_set(ctx, du_b, b);
                ctx->itrs++;
                t = thvm_sup(ctx, sup_label,
                    term_new(TAG_UDP, udp_label, du_a),
                    term_new(TAG_UDP, udp_label, du_b));
                goto inet_step;
            }

            // UDP ⊳ atoms: return value directly (UDP vanishes)
            if (term_tag(val) == TAG_NUM || term_tag(val) == TAG_ERA ||
                term_tag(val) == TAG_TEN || term_tag(val) == TAG_ANY) {
                return val;
            }

            // UDP ⊳ LAM: commutation — wrap body in UDP
            if (term_tag(val) == TAG_LAM) {
                u64 lam_loc = term_val(val);
                Term body = heap_read(ctx, lam_loc + 1);
                u64 bdup = heap_alloc(ctx, 1);
                heap_set(ctx, bdup, body);
                Term var0;
                Term lam_new = thvm_lam(ctx, &var0, term_new(TAG_UDP, udp_label, bdup));
                // Substitute original var with UDP of new var
                u64 vdup = heap_alloc(ctx, 1);
                heap_set(ctx, vdup, var0);
                heap_set(ctx, lam_loc, term_new(TAG_UDP, udp_label, vdup));
                ctx->itrs++;
                t = lam_new;
                goto inet_step;
            }

            return t;  // stuck
        }

        // TAG_USP: unordered superposition — WNF (like TAG_SUP)
        case TAG_USP:
            return t;

        // TAG_MAT: pattern match — WNF atom (fired by APP)
        case TAG_MAT:
            return t;

        // TAG_ANY: wildcard — WNF atom
        case TAG_ANY:
            return t;

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

