            if (uop == UOP_GRAD) {
                // (linear_use validates first_loc before patching)
                // Disable rewrite fusion during backward: let deferred chains grow
                // longer. Without this, rewrite_apply materializes each backward
                // formula immediately (1 dispatch per formula = 82 dispatches).
                // With this, deferred chains accumulate and tensor_materialize
                // fuses them into fewer, larger kernels at ENSURE boundaries.
                u8 _saved_nf = ctx->no_fuse;
                u8 _saved_nga = ctx->no_grad_alloc;
                ctx->no_fuse = 0;
                ctx->no_grad_alloc = 1;
                term_use_clear();

	                Term y_src  = heap_read(ctx, loc);
	                Term gy_src = heap_read(ctx, loc + 1);
	                Term y  = y_src;
	                Term gy = gy_src;
	                // Restore flags, return directly — trampoline reduces TAG_TOP results
	                #define GRAD_RETURN(r) do { \
	                    heap_set(ctx, loc, term_era()); \
	                    heap_set(ctx, loc + 1, term_era()); \
	                    ctx->no_fuse = _saved_nf; \
	                    ctx->no_grad_alloc = _saved_nga; \
	                    /* don't set dispatch_mode — let lazy trampoline handle gradient TAG_TOPs */ \
	                    return (r); \
	                } while(0)
	                #define GRAD_ERASE(term_to_erase) ({ \
	                    Term _ge = (term_to_erase); \
	                    thvm_make_active_era(ctx, _ge); \
	                })
	                #define GRAD_ERASE2(term_a, term_b) ({ \
	                    Term _qa = (term_a); \
	                    Term _qb = (term_b); \
	                    thvm_spawn_detached_era(ctx, _qb); \
	                    thvm_make_active_era(ctx, _qa); \
	                })
	                #define GRAD_SPAWN_ERA(term_to_erase) do { \
	                    Term _gs = (term_to_erase); \
	                    thvm_spawn_detached_era(ctx, _gs); \
	                } while (0)
	                #define GRAD_ZERO(term_to_drop) do { \
	                    Term _gz = (term_to_drop); \
	                    GRAD_SPAWN_ERA(_gz); \
	                    GRAD_RETURN(term_num_f32(0.0f)); \
	                } while (0)
	                Term x  = thvm_grad_target_get(ctx, loc);
                if (getenv("THVM_SCHED_DIAG")) {
                    static int _re=0; _re++; if (_re<=20)
                    fprintf(stderr, "  GRAD_ENTRY[%d]: y_tag=%u y_val=%llu gy_tag=%u x_tag=%u\n",
                        _re, term_tag(y), term_val(y), term_tag(gy), term_tag(x));
                }
                // Resolve DP0/DP1 on y (from BG DUP)
                for (int _dp=0; _dp<20 && (term_tag(y)==TAG_DP0||term_tag(y)==TAG_DP1); _dp++)
                    y = heap_read(ctx, term_val(y));
                if (getenv("THVM_SCHED_DIAG")) {
                    static int _yc=0; _yc++; if (_yc<=20)
                    fprintf(stderr, "  GRAD_RESOLVE[%d]: y_tag=%u y_ext=%u y_val=%llu gy_tag=%u\n", _yc, term_tag(y), term_ext(y), term_val(y), term_tag(gy));
                }

                // GRAD-SUP: gradient through superposition
                // GRAD(&L{y0,y1}, gy) → &L{GRAD(y0, DP0_L(gy)), ...}
                if (term_tag(y) == TAG_SUP) {
                    u32 lab = term_ext(y);
                    u64 sup_loc = term_val(y);
                    Term y0 = heap_read(ctx, sup_loc + 0);
                    Term y1 = heap_read(ctx, sup_loc + 1);
                    u64 gy_dup = heap_alloc(ctx, 1);
                    heap_set(ctx, gy_dup, gy);
                    // Inline GRAD2 (macro not yet in scope)
                    u64 _l0 = heap_alloc(ctx, 2);
                    heap_set(ctx, _l0, y0);
                    heap_set(ctx, _l0+1, term_new(TAG_DP0, lab, gy_dup));
                    thvm_grad_target_set(ctx, _l0, x);
                    u64 _l1 = heap_alloc(ctx, 2);
                    heap_set(ctx, _l1, y1);
                    heap_set(ctx, _l1+1, term_new(TAG_DP1, lab, gy_dup));
                    thvm_grad_target_set(ctx, _l1, x);
                    ctx->itrs++;
                    GRAD_RETURN(thvm_sup(ctx, lab,
                        term_new(TAG_TOP, UOP_GRAD, _l0),
                        term_new(TAG_TOP, UOP_GRAD, _l1)));
                }

                // ── GRAD on TAG_TOP: pure graph rewrite ──
                // y is a lazy compute op. Read UOP + args from heap.
                if (term_tag(y) == TAG_TOP && term_ext(y) != UOP_GRAD) {
                    static u32 _gc = 0; _gc++; if (_gc <= 100 && getenv("THVM_SCHED_DIAG")) fprintf(stderr, "GRAD_TOP[%u]: uop=%s\n", _gc, term_ext(y)<UOP_COUNT?uop_names[term_ext(y)]:"?");
	                    u32 cop = term_ext(y);
	                    u64 y_loc = term_val(y);
	                    Term at = heap_read(ctx, y_loc);
	                    Term bt = heap_read(ctx, y_loc + 1);
	                    const View *yv = st_get(y_loc);
                    Shape y_shape = yv ? yv->shape : SHAPE(1);
                    Shape a_shape = SHAPE(1), b_shape = SHAPE(1);
                    { Term _at = at;
                      if (term_tag(_at)==TAG_DP0||term_tag(_at)==TAG_DP1) _at=heap_read(ctx,term_val(_at));
                      if (term_tag(_at)==TAG_TEN) a_shape=ctx->tensors[(u32)term_val(_at)].view.shape;
                      else if (term_tag(_at)==TAG_TOP) { const View *av=st_get(term_val(_at)); if(av) a_shape=av->shape; } }
	                    { Term _bt = bt;
	                      if (term_tag(_bt)==TAG_DP0||term_tag(_bt)==TAG_DP1) _bt=heap_read(ctx,term_val(_bt));
	                      if (term_tag(_bt)==TAG_TEN) b_shape=ctx->tensors[(u32)term_val(_bt)].view.shape;
	                      else if (term_tag(_bt)==TAG_TOP) { const View *bv=st_get(term_val(_bt)); if(bv) b_shape=bv->shape; } }
	                    // Split `at` only when a rule needs both consumed and
	                    // continuation views.
	                    Term at_cons = at;
	                    u8 _at_split = 0;
	                    #define GRAD_SPLIT_AT() do { \
	                        if (!_at_split) { \
	                            u64 _ad = heap_alloc(ctx, 1); \
	                            heap_set(ctx, _ad, at); \
	                            at_cons = term_new(TAG_DP0, 0, _ad); \
	                            at      = term_new(TAG_DP1, 0, _ad); \
	                            _at_split = 1; \
	                        } \
	                    } while (0)

	                    #define GRAD2_H(y_,gy_,x_) ({ \
	                        u64 _l = heap_alloc(ctx, 2); \
	                        heap_set(ctx, _l, y_); heap_set(ctx, _l+1, gy_); thvm_grad_target_set(ctx, _l, x_); \
	                        { const View *_gv = NULL; Term _gt = (gy_); \
	                          if (term_tag(_gt)==TAG_TEN && (u32)term_val(_gt)<ctx->tensor_count) _gv=&ctx->tensors[(u32)term_val(_gt)].view; \
	                          else if (term_tag(_gt)==TAG_TOP) _gv=st_get(term_val(_gt)); \
	                          if (_gv) st_set(_l, _gv); } \
	                        term_new(TAG_TOP, UOP_GRAD, _l); })
	                    // Binary rule where both branches only need duplicated gy.
	                    #define BG_GY(da_of_gy, db_of_gy) do { \
	                        u64 _gyd = heap_alloc(ctx, 1); \
	                        heap_set(ctx, _gyd, gy); \
	                        Term gy0 = term_new(TAG_DP0, 0, _gyd); \
	                        Term gy1 = term_new(TAG_DP1, 0, _gyd); \
	                        Term _da; { Term gy = gy0; (void)gy; _da = (da_of_gy); } \
	                        Term _db; { Term gy = gy1; (void)gy; _db = (db_of_gy); } \
	                        Term _ga = GRAD2_H(at, _da, x); \
	                        Term _gb = GRAD2_H(bt, _db, x); \
	                        GRAD_RETURN(thvm_op_raw(ctx, UOP_ADD, _ga, _gb)); \
	                    } while (0)
		                    // BG: DUP bt/gy and keep two GRAD branches connected by ADD.
		                    #define BG(da_of_gy_bt, db_of_gy_at) do { \
		                        GRAD_SPLIT_AT(); \
		                        u64 _gyd = heap_alloc(ctx, 1); \
		                        heap_set(ctx, _gyd, gy); \
		                        Term gy0 = term_new(TAG_DP0, 0, _gyd); \
		                        Term gy1 = term_new(TAG_DP1, 0, _gyd); \
		                        u64 _btd = heap_alloc(ctx, 1); \
		                        heap_set(ctx, _btd, bt); \
		                        Term bt0 = term_new(TAG_DP0, 0, _btd); \
		                        Term bt1 = term_new(TAG_DP1, 0, _btd); \
		                        Term _da; { Term gy=gy0,bt=bt0; (void)gy;(void)bt; _da=(da_of_gy_bt); } \
		                        Term _db; { Term gy=gy1; Term _at=at; Term at=_at; (void)gy;(void)at; _db=(db_of_gy_at); } \
		                        /* table cleared at GRAD entry */ \
		                        Term _ga=GRAD2_H(at_cons,_da,x), _gb=GRAD2_H(bt1,_db,x); \
		                        GRAD_RETURN(thvm_op_raw(ctx, UOP_ADD, _ga, _gb)); \
		                    } while(0)
	                    // UG: keep sub-GRAD on the returned net.
	                    #define UG(da_) do { \
	                        Term _da_ug = (da_); \
	                        Term _u=GRAD2_H(at,_da_ug,x); \
	                        GRAD_RETURN(_u); \
	                    } while(0)
	                    #define UG_DROP(drop_, da_) do { \
	                        GRAD_SPAWN_ERA(drop_); \
	                        Term _da_ug = (da_); \
	                        Term _u=GRAD2_H(at,_da_ug,x); \
	                        GRAD_RETURN(_u); \
	                    } while(0)

	                    switch (cop) {
	                        case UOP_ADD: BG_GY(sum_to_shape(ctx,gy,y_shape,a_shape), sum_to_shape(ctx,gy,y_shape,b_shape));
	                        case UOP_SUB: BG_GY(sum_to_shape(ctx,gy,y_shape,a_shape), thvm_op_raw(ctx,UOP_NEG,sum_to_shape(ctx,gy,y_shape,b_shape),term_era()));
	                        case UOP_MUL: BG(sum_to_shape(ctx,thvm_op_raw(ctx,UOP_MUL,gy,bt),y_shape,a_shape), sum_to_shape(ctx,thvm_op_raw(ctx,UOP_MUL,gy,at),y_shape,b_shape));
                        case UOP_NEG: UG(thvm_op_raw(ctx,UOP_NEG,gy,term_era()));
	                        case UOP_RELU: { \
	                            f32 z = 0; \
	                            GRAD_SPLIT_AT(); \
	                            Term relu_mask = thvm_op_raw(ctx, UOP_CMP, at_cons, thvm_tensor(ctx, &z, SHAPE(1))); \
	                            UG(thvm_op_raw(ctx, UOP_MUL, gy, relu_mask)); \
	                        }
                        case UOP_EXP: UG(thvm_op_raw(ctx,UOP_MUL,gy,y));
                        case UOP_LOG: UG(thvm_op_raw(ctx,UOP_DIV,gy,at));
                        case UOP_SQRT: { f32 two=2; UG(thvm_op_raw(ctx,UOP_DIV,gy,thvm_op_raw(ctx,UOP_MUL,thvm_tensor(ctx,&two,SHAPE(1)),y))); }
                        case UOP_DIV: { Term ng=thvm_op_raw(ctx,UOP_NEG,gy,term_era()); BG(thvm_op_raw(ctx,UOP_DIV,gy,bt), thvm_op_raw(ctx,UOP_DIV,thvm_op_raw(ctx,UOP_MUL,ng,at),thvm_op_raw(ctx,UOP_MUL,bt,bt))); }
                        case UOP_MAX: { Term mask=thvm_op_raw(ctx,UOP_CMP,at,bt); f32 one=1; Term inv=thvm_op_raw(ctx,UOP_SUB,thvm_tensor(ctx,&one,SHAPE(1)),mask);
                            BG(sum_to_shape(ctx,thvm_op_raw(ctx,UOP_MUL,gy,mask),y_shape,a_shape), sum_to_shape(ctx,thvm_op_raw(ctx,UOP_MUL,gy,inv),y_shape,b_shape)); }
	                        case UOP_CMP: GRAD_RETURN(GRAD_ERASE(gy));
                        case UOP_SUM: { Term gy_rs=thvm_reshape(ctx,gy,y_shape); UG_DROP(bt, thvm_expand(ctx,gy_rs,a_shape)); }
                        case UOP_RMAX: { Term max_bc=thvm_expand(ctx,thvm_reshape(ctx,y,y_shape),a_shape); f32 one=1;
                            Term mask=thvm_op_raw(ctx,UOP_SUB,thvm_tensor(ctx,&one,SHAPE(1)),thvm_op_raw(ctx,UOP_CMP,max_bc,at));
                            UG_DROP(bt, thvm_op_raw(ctx,UOP_MUL,thvm_expand(ctx,gy,a_shape),mask)); }
                        case UOP_RESHAPE: UG_DROP(bt, thvm_reshape(ctx,gy,a_shape));
                        case UOP_PERMUTE: {
                            if (term_tag(bt)==TAG_TEN) { u32 pid=(u32)term_val(bt); TensorMeta *mp=&ctx->tensors[pid];
                                u32 nd=mp->view.numel; f32 pf[MAX_DIM]; META_READ(mp->backend,mp->buf_id,pf,nd*4);
                                u32 inv[MAX_DIM]; for(u32 j=0;j<nd;j++) inv[(u32)pf[j]]=j;
                                UG_DROP(bt, thvm_permute(ctx,gy,inv,nd));
	                            } else GRAD_RETURN(GRAD_ERASE(gy)); }
                        case UOP_EXPAND: UG_DROP(bt, sum_to_shape(ctx,gy,y_shape,a_shape));
                        case UOP_SHRINK: {
                            if (term_tag(bt)==TAG_TEN) { u32 sid=(u32)term_val(bt); TensorMeta *ms=&ctx->tensors[sid];
                                u32 nd=ms->view.numel/2; f32 sf[MAX_DIM*2]; META_READ(ms->backend,ms->buf_id,sf,nd*2*4);
                                u32 pp[MAX_DIM*2]; for(u32 j=0;j<nd;j++){pp[j*2]=(u32)sf[j*2];pp[j*2+1]=a_shape.dims[j]-(u32)sf[j*2+1];}
                                UG_DROP(bt, thvm_pad(ctx,gy,pp,nd));
	                            } else GRAD_RETURN(GRAD_ERASE(gy)); }
                        case UOP_PAD: {
                            if (term_tag(bt)==TAG_TEN) { u32 pid2=(u32)term_val(bt); TensorMeta *mp2=&ctx->tensors[pid2];
                                u32 nd=mp2->view.numel/2; f32 pf2[MAX_DIM*2]; META_READ(mp2->backend,mp2->buf_id,pf2,nd*2*4);
                                u32 sp[MAX_DIM*2]; for(u32 j=0;j<nd;j++){sp[j*2]=(u32)pf2[j*2];sp[j*2+1]=(u32)pf2[j*2]+a_shape.dims[j];}
                                UG_DROP(bt, thvm_shrink(ctx,gy,sp,nd));
	                            } else GRAD_RETURN(GRAD_ERASE(gy)); }
	                        default: GRAD_RETURN(GRAD_ERASE(gy));
	                    }
		                    #undef GRAD2_H
		                    #undef BG_GY
		                    #undef GRAD_SPLIT_AT
		                    #undef BG
		                    #undef UG
		                    #undef UG_DROP
                }

                if (term_tag(y) == TAG_TEN) {
                    u32 y_id = (u32)term_val(y);
                    TensorMeta *my = &ctx->tensors[y_id];
                    if (getenv("THVM_SCHED_DIAG")) { static int _gc=0; _gc++; if (_gc<=10) fprintf(stderr,"  GRAD_TEN[%d]: y_id=%u tag_x=%u\n",_gc,y_id,term_tag(x)); }
                    // Non-grad tensors are constants: produce explicit zero,
                    // and erase the consumed incoming gy branch separately.
                    if (!my->requires_grad) {
                        GRAD_ZERO(gy);
                    }
                    // Base case: y == x → return grad_y (trampoline reduces)
                    if (term_tag(x) == TAG_TEN && (u32)term_val(x) == y_id) {
                        GRAD_RETURN(gy);
                    }

                    // Multi-target: x = ANY pattern. Param->slot mapping lives in
                    // the internal GRAD target table and is resolved at TAG_TEN leaves.
                    // x may be DP0/DP1 reference (from BG DUP) — resolve through.
                    { Term _xr = x;
                      while (term_tag(_xr) == TAG_DP0 || term_tag(_xr) == TAG_DP1)
                          _xr = heap_read(ctx, term_val(_xr));
                      x = _xr; }
                    if (getenv("THVM_SCHED_DIAG") && y_id < 10)
                        fprintf(stderr, "  GRAD_TEN_X: y_id=%u x_tag=%u x_ext=%u\n", y_id, term_tag(x), term_ext(x));
                    if (term_tag(x) == TAG_ANY) {
                        Term slot = term_era();
	                        if (thvm_grad_targets_find_slot(ctx, y_id, &slot)) {
	                            // Leaf target hit: materialize explicit gradient update.
	                            // The GRAD branch itself disappears; gy now belongs to
	                            // the detached ASSIGN side effect and must not be erased.
	                            u64 _sd = heap_alloc(ctx, 1);
	                            heap_set(ctx, _sd, slot);
	                            Term slot0 = term_new(TAG_DP0, 0, _sd);
	                            Term slot1 = term_new(TAG_DP1, 0, _sd);
	                            Term accum = thvm_op_raw(ctx, UOP_ADD, slot0, gy);
	                            u64 _ah = heap_alloc(ctx, 1);
	                            heap_set(ctx, _ah, thvm_assign(ctx, slot1, accum));
	                            if (getenv("THVM_SCHED_DIAG")) fprintf(stderr, "  ASSIGN_CREATE: target_tid=%u\n", y_id);
	                            GRAD_RETURN(term_era());
	                        }
	                    }

                    // "all params" pattern: only requires_grad leaves survive.
                    // If this leaf is not in the explicit multi-target table,
                    // gy itself is the leaf gradient value.
	                    if (term_tag(x) == TAG_ANY) {
	                        if (my->requires_grad) GRAD_RETURN(gy);
	                        GRAD_ZERO(gy);
	                    }

                    // Leaf (no provenance, not target) is constant wrt the
                    // current target, so its local derivative is zero.
	                    if (!my->creator_op) {
	                            GRAD_ZERO(gy);
	                    }

                    // Movement ops fire eagerly → TAG_TEN with creator_op.
                    // Walk backward through provenance for movement ops only.
                    {
                        u32 cop = my->creator_op;
                        u32 aid = my->src_ids[0], bid = my->src_ids[1];
                        // (term_use_table cleared at GRAD entry)
                        // Cycle detection: provenance must go to earlier tensors
                        if (getenv("THVM_SCHED_DIAG")) fprintf(stderr, "  PROV: y_id=%u cop=%u aid=%u bid=%u tc=%u\n", y_id, cop, aid, bid, ctx->tensor_count);
	                        if (aid >= y_id || aid >= ctx->tensor_count) GRAD_RETURN(GRAD_ERASE(gy));
                        Term at = term_ten(aid, ctx->tensors[aid].dtype);

                        // Rebuild GRAD(input, backward_gy) and recurse
                        #define GRAD2_TEN(y_,gy_,x_) ({ \
                            u64 _l = heap_alloc(ctx, 2); \
                            heap_set(ctx, _l, y_); heap_set(ctx, _l+1, gy_); thvm_grad_target_set(ctx, _l, x_); \
                            term_new(TAG_TOP, UOP_GRAD, _l); })
	                        // WALK: place sub-GRAD on heap — one interaction per step
	                        #define WALK(da_) do { \
	                            Term _da_val = (da_); \
	                            /* table cleared at GRAD entry */ \
	                            Term _w = GRAD2_TEN(at, _da_val, x); \
	                            if (getenv("THVM_SCHED_DIAG")) { \
	                                u64 _gl = term_val(_w); \
	                                fprintf(stderr, "  WALK: grad_loc=%llu y_tag=%u y_val=%llu\n", \
	                                    _gl, term_tag(ctx->heap[_gl]), term_val(ctx->heap[_gl])); \
	                            } \
	                            GRAD_RETURN(_w); \
	                        } while(0)

                        switch (cop) {
                            case UOP_RESHAPE: WALK(thvm_reshape(ctx, gy, ctx->tensors[aid].view.shape));
                            case UOP_PERMUTE: {
                                if (bid) {
                                    TensorMeta *mp = &ctx->tensors[bid];
                                    u32 nd = mp->view.numel;
                                    f32 pf[MAX_DIM]; META_READ(mp->backend, mp->buf_id, pf, nd*4);
                                    u32 inv[MAX_DIM]; for(u32 j=0;j<nd;j++) inv[(u32)pf[j]]=j;
                                    WALK(thvm_permute(ctx, gy, inv, nd));
                                }
	                                GRAD_RETURN(GRAD_ERASE(gy));
	                            }
                            case UOP_EXPAND: WALK(sum_to_shape(ctx, gy, my->view.shape, ctx->tensors[aid].view.shape));
                            case UOP_SHRINK: {
                                if (bid) {
                                    TensorMeta *ms = &ctx->tensors[bid];
                                    u32 nd = ms->view.numel/2;
                                    f32 sf[MAX_DIM*2]; META_READ(ms->backend, ms->buf_id, sf, nd*2*4);
                                    u32 pp[MAX_DIM*2];
                                    for(u32 j=0;j<nd;j++){pp[j*2]=(u32)sf[j*2];pp[j*2+1]=ctx->tensors[aid].view.shape.dims[j]-(u32)sf[j*2+1];}
                                    WALK(thvm_pad(ctx, gy, pp, nd));
                                }
	                                GRAD_RETURN(GRAD_ERASE(gy));
	                            }
                            case UOP_PAD: {
                                if (bid) {
                                    TensorMeta *mp = &ctx->tensors[bid];
                                    u32 nd = mp->view.numel/2;
                                    f32 pf[MAX_DIM*2]; META_READ(mp->backend, mp->buf_id, pf, nd*2*4);
                                    u32 sp[MAX_DIM*2];
                                    for(u32 j=0;j<nd;j++){sp[j*2]=(u32)pf[j*2];sp[j*2+1]=(u32)pf[j*2]+ctx->tensors[aid].view.shape.dims[j];}
                                    WALK(thvm_shrink(ctx, gy, sp, nd));
                                }
	                                GRAD_RETURN(GRAD_ERASE(gy));
	                            }
	                            default: GRAD_RETURN(GRAD_ERASE(gy));
	                        }
                        #undef GRAD2_TEN
                        #undef WALK
                    }
                }
	                // y is TAG_TOP but not handled by GRAD_TOP above → ERA
	                if (getenv("THVM_SCHED_DIAG")) fprintf(stderr, "  GRAD_FALLTHROUGH: y_tag=%u y_ext=%u y_val=%llu\n", term_tag(y), term_ext(y), term_val(y));
	                GRAD_RETURN(GRAD_ERASE(gy));
	                #undef GRAD_SPAWN_ERA
	                #undef GRAD_ZERO
	                #undef GRAD_ERASE
	                #undef GRAD_ERASE2
	                #undef GRAD_RETURN
	            }
