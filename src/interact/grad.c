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

                // Restore flags, return directly — trampoline reduces TAG_TOP results
                #define GRAD_RETURN(r) do { \
                    ctx->no_fuse = _saved_nf; \
                    ctx->no_grad_alloc = _saved_nga; \
                    /* don't set dispatch_mode — let lazy trampoline handle gradient TAG_TOPs */ \
                    return (r); \
                } while(0)
                Term y  = heap_read(ctx, loc);
                Term gy = heap_read(ctx, loc + 1);
                Term x  = heap_read(ctx, loc + 2);
                // GRAD's child slots (y, gy, x) left intact for ERA propagation.
                // The worklist replaces heap[h] with ERA. A separate pass
                // fires ERA⊳TEN/TOP to clean consumed subtrees.
                // Resolve DP0/DP1 on y and x (from BG DUP)
                for (int _dp=0; _dp<20 && (term_tag(y)==TAG_DP0||term_tag(y)==TAG_DP1); _dp++)
                    y = heap_read(ctx, term_val(y));
                for (int _dp=0; _dp<20 && (term_tag(x)==TAG_DP0||term_tag(x)==TAG_DP1); _dp++)
                    x = heap_read(ctx, term_val(x));

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

                // ── GRAD on TAG_TOP: pure graph rewrite ──
                // y is a lazy compute op. Read UOP + args from heap.
                if (term_tag(y) == TAG_TOP && term_ext(y) != UOP_GRAD) {
                    static u32 _gc = 0; _gc++; if (_gc <= 100 && getenv("THVM_SCHED_DIAG")) fprintf(stderr, "GRAD_TOP[%u]: uop=%s\n", _gc, term_ext(y)<UOP_COUNT?uop_names[term_ext(y)]:"?");
                    u32 cop = term_ext(y);
                    u64 y_loc = term_val(y);
                    Term at = heap_read(ctx, y_loc);
                    Term bt = heap_read(ctx, y_loc + 1);
                    // y-op's children left in place — consumed but visible in graph
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

                    #define GRAD3_H(y_,gy_,x_) ({ \
                        u64 _l = heap_alloc(ctx, 3); \
                        heap_set(ctx, _l, y_); heap_set(ctx, _l+1, gy_); heap_set(ctx, _l+2, x_); \
                        { const View *_gv = NULL; Term _gt = (gy_); \
                          if (term_tag(_gt)==TAG_TEN && (u32)term_val(_gt)<ctx->tensor_count) _gv=&ctx->tensors[(u32)term_val(_gt)].view; \
                          else if (term_tag(_gt)==TAG_TOP) _gv=st_get(term_val(_gt)); \
                          if (_gv) st_set(_l, _gv); } \
                        term_new(TAG_TOP, UOP_GRAD, _l); })
                    // Chain gradients: both branches must fire (side-effect ASSIGNs).
                    // DUP all shared terms (at, bt, gy, x) so each branch gets its own port.
                    // _gb continues inline, _ga is placed on heap for Phase 1 scan.
                    // Binary gradient: both branches share at, bt, gy, x.
                    // Only DUP gy (used in both da_expr and db_expr).
                    // at/bt/x are NOT DUP'd: at goes to branch A (GRAD target),
                    // bt goes to branch B (GRAD target), x goes to branch B.
                    // Branch A's _ga is placed on heap; Branch B's _gb continues inline.
                    // Iterative chain rule: place ALL sub-GRADs on heap.
                    // Phase 1 worklist processes them without recursive goto.
                    #define BG(da_of_gy_bt, db_of_gy_at) do { \
                        u64 _ds = heap_alloc(ctx, 6); u32 _di = 0; \
                        Term gy0,gy1,at0,at1,bt0,bt1,x0,x1; \
                        if(term_tag(gy)==TAG_TEN||term_tag(gy)==TAG_ERA||term_tag(gy)==TAG_NUM||term_tag(gy)==TAG_CTR){gy0=gy;gy1=gy;} \
                        else{heap_set(ctx,_ds+_di,gy);gy0=term_new(TAG_DP0,0,_ds+_di);gy1=term_new(TAG_DP1,0,_ds+_di);_di++;} \
                        if(term_tag(at)==TAG_TEN||term_tag(at)==TAG_ERA||term_tag(at)==TAG_NUM||term_tag(at)==TAG_CTR){at0=at;at1=at;} \
                        else{heap_set(ctx,_ds+_di,at);at0=term_new(TAG_DP0,0,_ds+_di);at1=term_new(TAG_DP1,0,_ds+_di);_di++;} \
                        if(term_tag(bt)==TAG_TEN||term_tag(bt)==TAG_ERA||term_tag(bt)==TAG_NUM||term_tag(bt)==TAG_CTR){bt0=bt;bt1=bt;} \
                        else{heap_set(ctx,_ds+_di,bt);bt0=term_new(TAG_DP0,0,_ds+_di);bt1=term_new(TAG_DP1,0,_ds+_di);_di++;} \
                        if(term_tag(x)==TAG_TEN||term_tag(x)==TAG_ERA||term_tag(x)==TAG_NUM||term_tag(x)==TAG_CTR){x0=x;x1=x;} \
                        else{heap_set(ctx,_ds+_di,x);x0=term_new(TAG_DP0,0,_ds+_di);x1=term_new(TAG_DP1,0,_ds+_di);_di++;} \
                        Term _da; { Term gy=gy0,bt=bt0; (void)gy;(void)bt; _da=(da_of_gy_bt); } \
                        Term _db; { Term gy=gy1,at=at1; (void)gy;(void)at; _db=(db_of_gy_at); } \
                        Term _ga=GRAD3_H(at0,_da,x0), _gb=GRAD3_H(bt1,_db,x1); \
                        heap_set(ctx, _ds+4, _ga); \
                        heap_set(ctx, _ds+5, _gb); \
                        GRAD_RETURN(term_era()); \
                    } while(0)
                    // UG: place sub-GRAD on heap — one interaction per step
                    #define UG(da_) do { \
                        Term _u=GRAD3_H(at,da_,x); \
                        if(term_tag(_u)==TAG_TOP) { \
                            u64 _slot = heap_alloc(ctx, 1); \
                            heap_set(ctx, _slot, _u); \
                        } \
                        GRAD_RETURN(term_era()); \
                    } while(0)

                    switch (cop) {
                        case UOP_ADD: BG(sum_to_shape(ctx,gy,y_shape,a_shape), sum_to_shape(ctx,gy,y_shape,b_shape));
                        case UOP_SUB: BG(sum_to_shape(ctx,gy,y_shape,a_shape), thvm_op_raw(ctx,UOP_NEG,sum_to_shape(ctx,gy,y_shape,b_shape),term_era()));
                        case UOP_MUL: BG(sum_to_shape(ctx,thvm_op_raw(ctx,UOP_MUL,gy,bt),y_shape,a_shape), sum_to_shape(ctx,thvm_op_raw(ctx,UOP_MUL,gy,at),y_shape,b_shape));
                        case UOP_NEG: UG(thvm_op_raw(ctx,UOP_NEG,gy,term_era()));
                        case UOP_RELU: { f32 z=0; UG(thvm_op_raw(ctx,UOP_MUL,gy,thvm_op_raw(ctx,UOP_CMP,y,thvm_tensor(ctx,&z,SHAPE(1))))); }
                        case UOP_EXP: UG(thvm_op_raw(ctx,UOP_MUL,gy,y));
                        case UOP_LOG: UG(thvm_op_raw(ctx,UOP_DIV,gy,at));
                        case UOP_SQRT: { f32 two=2; UG(thvm_op_raw(ctx,UOP_DIV,gy,thvm_op_raw(ctx,UOP_MUL,thvm_tensor(ctx,&two,SHAPE(1)),y))); }
                        case UOP_DIV: { Term ng=thvm_op_raw(ctx,UOP_NEG,gy,term_era()); BG(thvm_op_raw(ctx,UOP_DIV,gy,bt), thvm_op_raw(ctx,UOP_DIV,thvm_op_raw(ctx,UOP_MUL,ng,at),thvm_op_raw(ctx,UOP_MUL,bt,bt))); }
                        case UOP_MAX: { Term mask=thvm_op_raw(ctx,UOP_CMP,at,bt); f32 one=1; Term inv=thvm_op_raw(ctx,UOP_SUB,thvm_tensor(ctx,&one,SHAPE(1)),mask);
                            BG(sum_to_shape(ctx,thvm_op_raw(ctx,UOP_MUL,gy,mask),y_shape,a_shape), sum_to_shape(ctx,thvm_op_raw(ctx,UOP_MUL,gy,inv),y_shape,b_shape)); }
                        case UOP_CMP: GRAD_RETURN(term_era());
                        case UOP_SUM: { Term gy_rs=thvm_reshape(ctx,gy,y_shape); UG(thvm_expand(ctx,gy_rs,a_shape)); }
                        case UOP_RMAX: { Term max_bc=thvm_expand(ctx,thvm_reshape(ctx,y,y_shape),a_shape); f32 one=1;
                            Term mask=thvm_op_raw(ctx,UOP_SUB,thvm_tensor(ctx,&one,SHAPE(1)),thvm_op_raw(ctx,UOP_CMP,max_bc,at));
                            UG(thvm_op_raw(ctx,UOP_MUL,thvm_expand(ctx,gy,a_shape),mask)); }
                        case UOP_RESHAPE: UG(thvm_reshape(ctx,gy,a_shape));
                        case UOP_PERMUTE: {
                            if (term_tag(bt)==TAG_TEN) { u32 pid=(u32)term_val(bt); TensorMeta *mp=&ctx->tensors[pid];
                                u32 nd=mp->view.numel; f32 pf[MAX_DIM]; META_READ(mp->backend,mp->buf_id,pf,nd*4);
                                u32 inv[MAX_DIM]; for(u32 j=0;j<nd;j++) inv[(u32)pf[j]]=j;
                                UG(thvm_permute(ctx,gy,inv,nd));
                            } else GRAD_RETURN(term_era()); }
                        case UOP_EXPAND: UG(sum_to_shape(ctx,gy,y_shape,a_shape));
                        case UOP_SHRINK: {
                            if (term_tag(bt)==TAG_TEN) { u32 sid=(u32)term_val(bt); TensorMeta *ms=&ctx->tensors[sid];
                                u32 nd=ms->view.numel/2; f32 sf[MAX_DIM*2]; META_READ(ms->backend,ms->buf_id,sf,nd*2*4);
                                u32 pp[MAX_DIM*2]; for(u32 j=0;j<nd;j++){pp[j*2]=(u32)sf[j*2];pp[j*2+1]=a_shape.dims[j]-(u32)sf[j*2+1];}
                                UG(thvm_pad(ctx,gy,pp,nd));
                            } else GRAD_RETURN(term_era()); }
                        case UOP_PAD: {
                            if (term_tag(bt)==TAG_TEN) { u32 pid2=(u32)term_val(bt); TensorMeta *mp2=&ctx->tensors[pid2];
                                u32 nd=mp2->view.numel/2; f32 pf2[MAX_DIM*2]; META_READ(mp2->backend,mp2->buf_id,pf2,nd*2*4);
                                u32 sp[MAX_DIM*2]; for(u32 j=0;j<nd;j++){sp[j*2]=(u32)pf2[j*2];sp[j*2+1]=(u32)pf2[j*2]+a_shape.dims[j];}
                                UG(thvm_shrink(ctx,gy,sp,nd));
                            } else GRAD_RETURN(term_era()); }
                        default: GRAD_RETURN(term_era());
                    }
                    #undef GRAD3_H
                    #undef BG
                    #undef UG
                }

                if (term_tag(y) == TAG_TEN) {
                    u32 y_id = (u32)term_val(y);
                    if (getenv("THVM_SCHED_DIAG")) { static int _gc=0; _gc++; if (_gc<=10) fprintf(stderr,"  GRAD_TEN[%d]: y_id=%u tag_x=%u\n",_gc,y_id,term_tag(x)); }
                    // Base case: y == x → return grad_y (trampoline reduces)
                    if (term_tag(x) == TAG_TEN && (u32)term_val(x) == y_id) {
                        GRAD_RETURN(gy);
                    }

                    // Multi-target: x = TAG_CTR encoding (param, slot) pairs.
                    // When y matches a param, deposit gy via ASSIGN into the slot.
                    // x may be DP0/DP1 reference (from BG DUP) — resolve through.
                    { Term _xr = x;
                      while (term_tag(_xr) == TAG_DP0 || term_tag(_xr) == TAG_DP1)
                          _xr = heap_read(ctx, term_val(_xr));
                      x = _xr; }
                    if (getenv("THVM_SCHED_DIAG") && y_id < 10)
                        fprintf(stderr, "  GRAD_TEN_X: y_id=%u x_tag=%u x_ext=%u\n", y_id, term_tag(x), term_ext(x));
                    if (term_tag(x) == TAG_CTR) {
                        u64 tgt_loc = term_val(x);
                        u32 n_tgt = term_as_u32(heap_read(ctx, tgt_loc));
                        for (u32 _gi = 0; _gi < n_tgt; _gi++) {
                            Term p = heap_read(ctx, tgt_loc + 1 + 2*_gi);
                            if (term_tag(p) == TAG_TEN && (u32)term_val(p) == y_id) {
                                Term slot = heap_read(ctx, tgt_loc + 1 + 2*_gi + 1);
                                // DUP slot for ADD (reads old) + ASSIGN (writes new)
                                u64 _sd = heap_alloc(ctx, 2); // 1 for slot DUP + 1 for pending
                                heap_set(ctx, _sd, slot);
                                Term slot0 = term_new(TAG_DP0, 0, _sd);
                                Term slot1 = term_new(TAG_DP1, 0, _sd);
                                Term accum = thvm_op_raw(ctx, UOP_ADD, slot0, gy);
                                u64 _ah = _sd + 1;
                                heap_set(ctx, _ah, thvm_assign(ctx, slot1, accum));
                                if (getenv("THVM_SCHED_DIAG")) fprintf(stderr, "  ASSIGN_CREATE: target=%u slot=%u\n", y_id, _gi);
                                GRAD_RETURN(term_era());
                            }
                        }
                    }

                    TensorMeta *my = &ctx->tensors[y_id];

                    // Leaf (no provenance, not target) → ERA
                    if (!my->creator_op) {
                            GRAD_RETURN(term_era());
                    }

                    // Movement ops fire eagerly → TAG_TEN with creator_op.
                    // Walk backward through provenance for movement ops only.
                    {
                        u32 cop = my->creator_op;
                        u32 aid = my->src_ids[0], bid = my->src_ids[1];
                        // Cycle detection: provenance must go to earlier tensors
                        if (aid >= y_id || aid >= ctx->tensor_count) GRAD_RETURN(term_era());
                        Term at = term_ten(aid, ctx->tensors[aid].dtype);

                        // Rebuild GRAD(input, backward_gy, x) and recurse
                        #define GRAD3_TEN(y_,gy_,x_) ({ \
                            u64 _l = heap_alloc(ctx, 3); \
                            heap_set(ctx, _l, y_); heap_set(ctx, _l+1, gy_); heap_set(ctx, _l+2, x_); \
                            term_new(TAG_TOP, UOP_GRAD, _l); })
                        // WALK: place sub-GRAD on heap — one interaction per step
                        #define WALK(da_) do { \
                            Term _w = GRAD3_TEN(at, da_, x); \
                            if (term_tag(_w) == TAG_TOP) { \
                                u64 _slot = heap_alloc(ctx, 1); \
                                heap_set(ctx, _slot, _w); \
                            } \
                            GRAD_RETURN(term_era()); \
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
                                GRAD_RETURN(term_era());
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
                                GRAD_RETURN(term_era());
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
                                GRAD_RETURN(term_era());
                            }
                            default: GRAD_RETURN(term_era());
                        }
                        #undef GRAD3_TEN
                        #undef WALK
                    }
                }
                // y is TAG_TOP but not handled by GRAD_TOP above → ERA
                GRAD_RETURN(term_era());
                #undef GRAD_RETURN
            }
