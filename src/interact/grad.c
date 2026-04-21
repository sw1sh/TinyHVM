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
                        Term _gr = (r); \
                        if (thvm_grad_mode_get(ctx, loc) == GRAD_MODE_KEEP && \
                            term_tag(_gr) == TAG_ERA && term_val(_gr) == 0) { \
                            _gr = term_num_f32(0.0f); \
                        } \
	                    heap_set(ctx, loc, term_era()); \
	                    heap_set(ctx, loc + 1, term_era()); \
	                    ctx->no_fuse = _saved_nf; \
	                    ctx->no_grad_alloc = _saved_nga; \
	                    /* don't set dispatch_mode — let lazy trampoline handle gradient TAG_TOPs */ \
	                    return _gr; \
	                } while(0)
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
	                #define GRAD_DROP(term_to_drop) do { \
	                    Term _gd = (term_to_drop); \
	                    GRAD_SPAWN_ERA(_gd); \
	                    GRAD_RETURN(term_era()); \
	                } while (0)
                    #define GRAD_RESOLVE_ALIAS(_tt) do { \
                        for (int _i = 0; _i < 32; _i++) { \
                            u8 _tag = term_tag(_tt); \
                            if (_tag == TAG_DP0 || _tag == TAG_DP1) { \
                                (_tt) = heap_read(ctx, term_val(_tt)); \
                                continue; \
                            } \
                            if (_tag == TAG_VAR) { \
                                Term _sub = heap_read(ctx, term_val(_tt)); \
                                if (term_is_sub(_sub)) break; \
                                (_tt) = _sub; \
                                continue; \
                            } \
                            break; \
                        } \
	                } while (0)
	                Term x  = thvm_grad_target_get(ctx, loc);
                    u32 grad_mode = thvm_grad_mode_get(ctx, loc);
                    Term keep_bundle = (grad_mode == GRAD_MODE_KEEP) ? thvm_grad_keep_bundle_get(ctx, loc) : term_era();
                    int has_keep_bundle = thvm_grad_keep_app_loc_get(ctx, loc) != 0 ||
                                          !(term_tag(keep_bundle) == TAG_ERA && term_val(keep_bundle) == 0);
                    #define GRAD_SLOT_SEQ2(a_, b_) ({ \
                        Term _ga_seq = (a_); \
                        Term _gb_seq = (b_); \
                        Term _gr_seq; \
                        if (term_tag(_ga_seq) == TAG_ERA && term_val(_ga_seq) == 0) { \
                            _gr_seq = _gb_seq; \
                        } else if (term_tag(_gb_seq) == TAG_ERA && term_val(_gb_seq) == 0) { \
                            _gr_seq = _ga_seq; \
                        } else { \
                            /* Slot-backed gradients are phase-3 effects. */ \
                            /* Sequence them with APP so the whole grad term */ \
                            /* collapses to ERA after all deposits fire. */ \
                            _gr_seq = thvm_app(ctx, _ga_seq, _gb_seq); \
                        } \
                        _gr_seq; \
                    })
                    #define GRAD_KEEP_SEQ2(a_, b_) ({ \
                        Term _ga_keep = (a_); \
                        Term _gb_keep = (b_); \
                        Term _gr_keep; \
                        int _ga_zero = (term_tag(_ga_keep) == TAG_ERA && term_val(_ga_keep) == 0) || \
                                       (term_tag(_ga_keep) == TAG_NUM && term_val(_ga_keep) == 0); \
                        int _gb_zero = (term_tag(_gb_keep) == TAG_ERA && term_val(_gb_keep) == 0) || \
                                       (term_tag(_gb_keep) == TAG_NUM && term_val(_gb_keep) == 0); \
                        if (_ga_zero) { \
                            _gr_keep = _gb_keep; \
                        } else if (_gb_zero) { \
                            _gr_keep = _ga_keep; \
                        } else { \
                            _gr_keep = thvm_app(ctx, _ga_keep, _gb_keep); \
                        } \
                        _gr_keep; \
                    })
                    // GRAD_COMBINE: combine two branch gradients.
                    // Always zero-aware: if one branch is NUM(0) / ERA, the
                    // other branch passes through unchanged. Without this, a
                    // binary rule where only one operand is a target (or
                    // neither is a target) produces an ADD(gy, 0) TOP node
                    // that clutters the graph with a spurious ADD.
                    #define GRAD_COMBINE(a_, b_) ({ \
                        Term _ga_comb = (a_); \
                        Term _gb_comb = (b_); \
                        Term _gr_comb; \
                        int _ga_z = (term_tag(_ga_comb) == TAG_ERA && term_val(_ga_comb) == 0) || \
                                    (term_tag(_ga_comb) == TAG_NUM && term_val(_ga_comb) == 0); \
                        int _gb_z = (term_tag(_gb_comb) == TAG_ERA && term_val(_gb_comb) == 0) || \
                                    (term_tag(_gb_comb) == TAG_NUM && term_val(_gb_comb) == 0); \
                        if (_ga_z) _gr_comb = _gb_comb; \
                        else if (_gb_z) _gr_comb = _ga_comb; \
                        else if (grad_mode == GRAD_MODE_SLOT) _gr_comb = GRAD_SLOT_SEQ2(_ga_comb, _gb_comb); \
                        else if (grad_mode == GRAD_MODE_KEEP) _gr_comb = GRAD_KEEP_SEQ2(_ga_comb, _gb_comb); \
                        else _gr_comb = thvm_op_raw(ctx, UOP_ADD, _ga_comb, _gb_comb); \
                        _gr_comb; \
                    })
                    #define GRAD_SLOT_ACCUM(slot_term_, gy_term_) ({ \
                        Term _slot_src = (slot_term_); \
                        Term _slot_gy  = (gy_term_); \
                        /* Slot-backed gradients are phase-3 effects, but the */ \
                        /* target must still be a real tensor, not DP1(slot). */ \
                        /* Tensors are atomic/copyable, so reuse the slot term */ \
                        /* directly as both the assign target and ADD lhs. */ \
                        Term _accum = thvm_op_raw(ctx, UOP_ADD, _slot_src, _slot_gy); \
                        thvm_app(ctx, thvm_assign(ctx, _slot_src, _accum), term_era()); \
                    })
                if (getenv("THVM_SCHED_DIAG")) {
                    static int _re=0; _re++; if (_re<=20)
                    fprintf(stderr, "  GRAD_ENTRY[%d]: y_tag=%u y_val=%llu gy_tag=%u x_tag=%u\n",
                        _re, term_tag(y), term_val(y), term_tag(gy), term_tag(x));
                }
                if (getenv("THVM_LOOP_DIAG")) {
                    fprintf(stderr,
                            "GRAD_ENTRY loc=%llu y_tag=%u y_val=%llu gy_tag=%u gy_val=%llu x_tag=%u x_ext=%u x_val=%llu mode=%u has_keep=%d targets=%u\n",
                            (unsigned long long)loc,
                            (u32)term_tag(y),
                            (unsigned long long)term_val(y),
                            (u32)term_tag(gy),
                            (unsigned long long)term_val(gy),
                            (u32)term_tag(x),
                            (u32)term_ext(x),
                            (unsigned long long)term_val(x),
                            grad_mode,
                            has_keep_bundle,
                            thvm_grad_targets_count_at(ctx, loc));
                }
                // Free-port gy (TAG_ERA, val=0) = Jacobian-mode seed "I am
                // unit gradient of y's shape". Lift locally to a scalar TEN(1.0)
                // so the per-rule formulas (sum_to_shape, thvm_expand) have a
                // concrete shape-aware operand to work with. The heap cell stays
                // a free port so step graphs still render the gy port as an
                // outward arrow to a small circle.
                if (term_tag(gy) == TAG_ERA && term_val(gy) == 0) {
                    gy = thvm_scalar_typed(ctx, 1.0f, DTYPE_F32);
                }
                // DROP mode used to short-circuit here with ERA2(y, gy) —
                // but that hid the chain rule, which is the main thing the
                // Phase-1 step graphs are supposed to show. Fall through so
                // the same rule firings happen as in KEEP/SLOT modes; drop
                // mode just doesn't deposit into a bundle.
                GRAD_RESOLVE_ALIAS(y);
                GRAD_RESOLVE_ALIAS(x);
                if (getenv("THVM_LOOP_DIAG")) {
                    fprintf(stderr,
                            "GRAD_RESOLVED loc=%llu y_tag=%u y_ext=%u y_val=%llu x_tag=%u x_ext=%u x_val=%llu gy_tag=%u gy_ext=%u gy_val=%llu\n",
                            (unsigned long long)loc,
                            (u32)term_tag(y),
                            (u32)term_ext(y),
                            (unsigned long long)term_val(y),
                            (u32)term_tag(x),
                            (u32)term_ext(x),
                            (unsigned long long)term_val(x),
                            (u32)term_tag(gy),
                            (u32)term_ext(gy),
                            (unsigned long long)term_val(gy));
                }
                if (getenv("THVM_SCHED_DIAG")) {
                    static int _yc=0; _yc++; if (_yc<=20)
                    fprintf(stderr, "  GRAD_RESOLVE[%d]: y_tag=%u y_ext=%u y_val=%llu gy_tag=%u\n", _yc, term_tag(y), term_ext(y), term_val(y), term_tag(gy));
                }

                // GRAD⊳TEN target match: y is the target leaf, so dy/dx = 1.
                // Return gy (the accumulated gradient at this path); any
                // previous short-circuit that erased gy here was hiding the
                // chain rule's result at the root.
                if (term_tag(x) != TAG_ANY && x == y) {
                    GRAD_RETURN(gy);
                }

                // Constants / erased branches contribute zero local derivative.
                // Do not erase gy here: in KEEP mode, erasing can collapse
                // sequencing APP chains and drop the gradient bundle itself.
                if (term_tag(y) == TAG_ERA || term_tag(y) == TAG_NUM) {
                    if (grad_mode == GRAD_MODE_SLOT) GRAD_DROP(gy);
                    GRAD_ZERO(gy);
                }

                if (term_tag(x) == TAG_ANY) {
                    Term slot = term_era();
                    u32 target_index = 0;
                    u64 keep_app_loc = thvm_grad_keep_app_loc_get(ctx, loc);
                    int matched_target = 0;
                    if (term_tag(y) == TAG_TEN) {
                        matched_target =
                            thvm_grad_targets_find_index_at(ctx, loc, (u32)term_val(y), &target_index, &slot) ||
                            thvm_grad_targets_find_term_at(ctx, loc, y, &target_index, &slot);
                    }
                    if (matched_target) {
                        if (getenv("THVM_LOOP_DIAG")) {
                            fprintf(stderr,
                                    "GRAD_TARGET_MULTI_HIT loc=%llu y_id=%u idx=%u mode=%u has_keep=%d keep_tag=%u keep_ext=%u keep_val=%llu keep_app=%llu slot_tag=%u slot_ext=%u slot_val=%llu gy_tag=%u gy_ext=%u gy_val=%llu\n",
                                    (unsigned long long)loc,
                                    (u32)term_val(y),
                                    target_index,
                                    grad_mode,
                                    has_keep_bundle,
                                    (u32)term_tag(keep_bundle),
                                    (u32)term_ext(keep_bundle),
                                    (unsigned long long)term_val(keep_bundle),
                                    (unsigned long long)keep_app_loc,
                                    (u32)term_tag(slot),
                                    (u32)term_ext(slot),
                                    (unsigned long long)term_val(slot),
                                    (u32)term_tag(gy),
                                    (u32)term_ext(gy),
                                    (unsigned long long)term_val(gy));
                        }
                        if (grad_mode == GRAD_MODE_SLOT &&
                            !(term_tag(slot) == TAG_ERA && term_val(slot) == 0)) {
                            GRAD_RETURN(GRAD_SLOT_ACCUM(slot, gy));
                        }
                        if (grad_mode == GRAD_MODE_KEEP && has_keep_bundle) {
                            thvm_grad_bundle_accum(ctx, loc, target_index, gy);
                            GRAD_RETURN(term_num_f32(0.0f));
                        }
                        if (grad_mode == GRAD_MODE_DROP) GRAD_DROP(gy);
                        if (grad_mode == GRAD_MODE_SLOT) GRAD_DROP(gy);
                        GRAD_ZERO(gy);
                    }
                    if (term_tag(y) == TAG_TEN && getenv("THVM_LOOP_DIAG")) {
                        u32 nt = thvm_grad_targets_count_at(ctx, loc);
                        fprintf(stderr, "GRAD_TARGET_MISS loc=%llu y_id=%u nt=%u\n",
                                (unsigned long long)loc,
                                (u32)term_val(y),
                                nt);
                        for (u32 i = 0; i < nt; i++) {
                            Term pt = thvm_grad_targets_get_term_at(ctx, loc, i);
                            Term ps = thvm_grad_targets_get_slot_at(ctx, loc, i);
                            fprintf(stderr,
                                    "  target[%u]=tag%u ext%u val%llu tid=%u slot=(tag%u ext%u val%llu)\n",
                                    i,
                                    (u32)term_tag(pt),
                                    (u32)term_ext(pt),
                                    (unsigned long long)term_val(pt),
                                    thvm_grad_targets_get_tid_at(ctx, loc, i),
                                    (u32)term_tag(ps),
                                    (u32)term_ext(ps),
                                    (unsigned long long)term_val(ps));
                        }
                    }
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
                    thvm_grad_targets_share(ctx, _l0, loc);
                    thvm_grad_target_set(ctx, _l0, x);
                    u64 _l1 = heap_alloc(ctx, 2);
                    heap_set(ctx, _l1, y1);
                    heap_set(ctx, _l1+1, term_new(TAG_DP1, lab, gy_dup));
                    thvm_grad_targets_share(ctx, _l1, loc);
                    thvm_grad_target_set(ctx, _l1, x);
                    ctx->itrs++;
                    if (grad_mode == GRAD_MODE_SLOT) {
                        Term _g0 = term_new(TAG_TOP, UOP_GRAD, _l0);
                        Term _g1 = term_new(TAG_TOP, UOP_GRAD, _l1);
                        GRAD_RETURN(GRAD_SLOT_SEQ2(_g0, _g1));
                    }
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
                    // Default arg shapes to y_shape: elementwise ops preserve
                    // shape, so when the arg is a DP/VAR/ALO (not resolvable
                    // yet to TEN or TOP), y_shape is the correct assumption.
                    // Reshape/expand/permute overrides this via the TEN/TOP
                    // lookup below.
                    Shape a_shape = y_shape, b_shape = y_shape;
                    { Term _at = at;
                      GRAD_RESOLVE_ALIAS(_at);
                      if (term_tag(_at)==TAG_TEN) a_shape=ctx->tensors[(u32)term_val(_at)].view.shape;
                      else if (term_tag(_at)==TAG_TOP) { const View *av=st_get(term_val(_at)); if(av) a_shape=av->shape; } }
	                    { Term _bt = bt;
	                      GRAD_RESOLVE_ALIAS(_bt);
	                      if (term_tag(_bt)==TAG_TEN) b_shape=ctx->tensors[(u32)term_val(_bt)].view.shape;
	                      else if (term_tag(_bt)==TAG_TOP) { const View *bv=st_get(term_val(_bt)); if(bv) b_shape=bv->shape; } }
	                    /* LINEAR IR: consume y_loc's arg cells now. The grad
	                     * rule is the sole consumer of y's args. Leaving DP
	                     * refs in y_loc+0/+1 creates non-linear placements
	                     * when subsequent rule construction (GRAD2_H,
	                     * GRAD_SPLIT_AT, thvm_op_raw) stores at/bt in new
	                     * heap cells. Clearing here removes y_loc's
	                     * tracking slot for any DP, so downstream placements
	                     * are the sole holders. Values stay in local
	                     * variables at/bt so grad formulas can still use
	                     * them. Gated on THVM_GRAD_LINEAR_IR. */
	                    if (getenv("THVM_GRAD_LINEAR_IR")) {
	                        u8 _lia = term_tag(at);
	                        u8 _lib = term_tag(bt);
	                        u8 _ligy = term_tag(gy);
	                        if (getenv("LINEAR_IR_TRACE"))
	                            fprintf(stderr, "LIN_IR y=%u/%u@%llu at_tag=%u bt_tag=%u gy_tag=%u\n",
	                                (u32)term_tag(y), (u32)term_ext(y), (unsigned long long)y_loc,
	                                _lia, _lib, _ligy);
	                        if (_lia == TAG_DP0 || _lia == TAG_DP1)
	                            heap_set(ctx, y_loc, term_era());
	                        if (_lib == TAG_DP0 || _lib == TAG_DP1)
	                            heap_set(ctx, y_loc + 1, term_era());
	                        if (_ligy == TAG_DP0 || _ligy == TAG_DP1)
	                            heap_set(ctx, loc + 1, term_era());
	                    }
                    // Recover y_shape from operands when the TOP node lost its
                    // View tracker (e.g., ALO realize copied LAM body without
                    // tracker). For elementwise ops the output shape is the
                    // broadcast of operand shapes; without this, y_shape stays
                    // SHAPE(1) and sum_to_shape squashes grads to one element
                    // and rebroadcasts uniformly.
                    if (!yv) {
                        u32 ar = a_shape.rank, br = b_shape.rank;
                        u32 r = ar > br ? ar : br;
                        if (r > 0 && r <= MAX_DIM) {
                            Shape s = {.rank = r};
                            int ok = 1;
                            for (u32 i = 0; i < r && ok; i++) {
                                int ia = (int)ar - 1 - (int)i;
                                int ib = (int)br - 1 - (int)i;
                                u32 da = ia >= 0 ? a_shape.dims[ia] : 1;
                                u32 db = ib >= 0 ? b_shape.dims[ib] : 1;
                                u32 d;
                                if (da == db) d = da;
                                else if (da == 1) d = db;
                                else if (db == 1) d = da;
                                else { ok = 0; d = 0; }
                                s.dims[r - 1 - i] = d;
                            }
                            if (ok) y_shape = s;
                        }
                    }
                    if (getenv("THVM_SCHED_DIAG")) {
                        static int _gs = 0;
                        if (++_gs <= 100) {
                            fprintf(stderr,
                                    "  GRAD_SHAPES[%d]: y=[",
                                    _gs);
                            for (u32 _d = 0; _d < y_shape.rank; _d++)
                                fprintf(stderr, "%u%s", y_shape.dims[_d], _d + 1 < y_shape.rank ? "," : "");
                            fprintf(stderr, "] a=[");
                            for (u32 _d = 0; _d < a_shape.rank; _d++)
                                fprintf(stderr, "%u%s", a_shape.dims[_d], _d + 1 < a_shape.rank ? "," : "");
                            fprintf(stderr, "] b=[");
                            for (u32 _d = 0; _d < b_shape.rank; _d++)
                                fprintf(stderr, "%u%s", b_shape.dims[_d], _d + 1 < b_shape.rank ? "," : "");
                            fprintf(stderr, "] y_loc=%llu\n", (unsigned long long)y_loc);
                        }
                    }
	                    // Split `at` only when a rule needs both consumed and
	                    // continuation views.
	                    Term at_cons = at;
	                    u8 _at_split = 0;
	                    /* Split at into (at_cons, at):
	                       - at_cons goes into recursive GRAD walk (provenance)
	                       - at is used as VALUE by the BG arm's MUL(gy, at)
	                       For ALO/TOP at, force-to-TEN for VALUE and keep
	                       original for GRAD walk. */
	                    /* Peek ALO's inner book term; force only for pure
	                       view/reduce compute ops that don't dispatch to
	                       backend kernels or side-effect. Exclude MM (BLAS),
	                       ASSIGN, KERNEL, IFZ, DETACH, LOG_PRINT, TODEVICE,
	                       GRAD (would recurse). */
	                    #define GRAD_ALO_SAFE_TO_FORCE(_t) ({ \
	                        int _ok = 0; \
	                        if (term_tag(_t) == TAG_ALO) { \
	                            u64 _al = term_val(_t); \
	                            if (_al + 1 < ctx->heap_pos) { \
	                                Term _inner = heap_read(ctx, _al + 0); \
	                                u8 _it = term_tag(_inner); \
	                                u32 _iu = term_ext(_inner); \
	                                if (_it == TAG_TOP) { \
	                                    if (_iu != UOP_GRAD && _iu != UOP_MM && \
	                                        _iu != UOP_ASSIGN && _iu != UOP_KERNEL && \
	                                        _iu != UOP_DETACH && _iu != UOP_IFZ && \
	                                        _iu != UOP_TODEVICE && _iu != UOP_LOG_PRINT && \
	                                        _iu != UOP_RESHAPE) _ok = 1; \
	                                } \
	                                else if (_it == TAG_DP0 || _it == TAG_DP1) _ok = 1; \
	                            } \
	                        } \
	                        _ok; })
	                    #define GRAD_SPLIT_AT() do { \
	                        if (!_at_split) { \
	                            u64 _ad = heap_alloc(ctx, 1); \
	                            heap_set(ctx, _ad, at); \
	                            u32 _alab = thvm_fresh_label(ctx); \
	                            at_cons = term_new(TAG_DP0, _alab, _ad); \
	                            at      = term_new(TAG_DP1, _alab, _ad); \
	                            _at_split = 1; \
	                        } \
	                    } while (0)

	                    #define GRAD2_H(y_,gy_,x_) ({ \
	                        u64 _l = heap_alloc(ctx, 2); \
	                        heap_set(ctx, _l, y_); heap_set(ctx, _l+1, gy_); thvm_grad_targets_share(ctx, _l, loc); thvm_grad_target_set(ctx, _l, x_); \
	                        { const View *_gv = NULL; Term _gt = (gy_); \
	                          GRAD_RESOLVE_ALIAS(_gt); \
	                          if (term_tag(_gt)==TAG_TEN && (u32)term_val(_gt)<ctx->tensor_count) _gv=&ctx->tensors[(u32)term_val(_gt)].view; \
	                          else if (term_tag(_gt)==TAG_TOP) _gv=st_get(term_val(_gt)); \
	                          if (_gv) st_set(_l, _gv); } \
	                        term_new(TAG_TOP, UOP_GRAD, _l); })
	                    // Binary rule where both branches only need duplicated gy.
	                    #define BG_GY(da_of_gy, db_of_gy) do { \
	                        u64 _gyd = heap_alloc(ctx, 1); \
	                        heap_set(ctx, _gyd, gy); \
	                        u32 _gylab = thvm_fresh_label(ctx); \
	                        Term gy0 = term_new(TAG_DP0, _gylab, _gyd); \
	                        Term gy1 = term_new(TAG_DP1, _gylab, _gyd); \
	                        Term _da; { Term gy = gy0; (void)gy; _da = (da_of_gy); } \
	                        Term _db; { Term gy = gy1; (void)gy; _db = (db_of_gy); } \
	                        Term _ga = GRAD2_H(at, _da, x); \
	                        Term _gb = GRAD2_H(bt, _db, x); \
	                        GRAD_RETURN(GRAD_COMBINE(_ga, _gb)); \
	                    } while (0)
			                    /* BG: DUP bt/gy and keep two GRAD branches
			                       connected by ADD. For ALO-wrapped TOP bt,
			                       force to TEN so the a-arm's MUL(gy, bt0)
			                       reads bt per-element; keep original bt as
			                       the b-arm's GRAD walk target (provenance). */
			                    /* Short-circuit: if bt resolves through ALO/RESHAPE */ \
			                    /* wrappers to a non-req-grad TEN (a constant), skip */ \
			                    /* DUP(gy) and the b-arm entirely. The bt-DUP over a */ \
			                    /* const-ALO(RESHAPE(TEN)) halves gy via IC orphans */ \
			                    /* (see test_sum_sq_ref_scale). */ \
			                    #define BG_BT_CONST_PEEK(_bt) ({ \
			                        u32 _tid = UINT32_MAX; \
			                        Term _c = (_bt); \
			                        for (int _d = 0; _d < 8; _d++) { \
			                            if (term_tag(_c) == TAG_ALO) { \
			                                u64 _aloc = term_val(_c); \
			                                if (_aloc + 1 >= ctx->heap_pos) break; \
			                                _c = heap_read(ctx, _aloc + 0); continue; \
			                            } \
			                            if (term_tag(_c) == TAG_TOP && term_ext(_c) == UOP_RESHAPE) { \
			                                u64 _tloc = term_val(_c); \
			                                if (_tloc + 1 >= ctx->heap_pos) break; \
			                                _c = heap_read(ctx, _tloc + 0); continue; \
			                            } \
			                            if (term_tag(_c) == TAG_TEN) { \
			                                u32 _tt = (u32)term_val(_c); \
			                                if (_tt < ctx->tensor_count && !ctx->tensors[_tt].requires_grad) _tid = _tt; \
			                            } \
			                            break; \
			                        } \
			                        _tid; })
			                    #define BG(da_of_gy_bt, db_of_gy_at) do { \
			                        u32 _bt_const = BG_BT_CONST_PEEK(bt); \
			                        if (_bt_const != UINT32_MAX) { \
			                            Term bt_val = term_new(TAG_TEN, 0, _bt_const); \
			                            Term _da; { Term bt=bt_val; (void)bt; _da=(da_of_gy_bt); } \
			                            GRAD_SPAWN_ERA(bt); \
			                            Term _ga = GRAD2_H(at, _da, x); \
			                            GRAD_RETURN(_ga); \
			                        } \
			                        GRAD_SPLIT_AT(); \
			                        u64 _gyd = heap_alloc(ctx, 1); \
			                        heap_set(ctx, _gyd, gy); \
			                        u32 _gylab = thvm_fresh_label(ctx); \
			                        Term gy0 = term_new(TAG_DP0, _gylab, _gyd); \
			                        Term gy1 = term_new(TAG_DP1, _gylab, _gyd); \
			                        Term bt0 = term_era(), bt1 = term_era(); \
			                        { \
			                            u64 _btd = heap_alloc(ctx, 1); \
			                            heap_set(ctx, _btd, bt); \
			                            u32 _btlab = thvm_fresh_label(ctx); \
			                            bt0 = term_new(TAG_DP0, _btlab, _btd); \
			                            bt1 = term_new(TAG_DP1, _btlab, _btd); \
			                        } \
			                        Term _da; { Term gy=gy0,bt=bt0; (void)gy;(void)bt; _da=(da_of_gy_bt); } \
			                        Term _db; { Term gy=gy1; Term _at=at; Term at=_at; (void)gy;(void)at; _db=(db_of_gy_at); } \
			                        Term _ga=GRAD2_H(at_cons,_da,x), _gb=GRAD2_H(bt1,_db,x); \
			                        GRAD_RETURN(GRAD_COMBINE(_ga, _gb)); \
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

	                    if (getenv("THVM_SCHED_DIAG")) {
	                        static int _mt=0; _mt++;
	                        if (_mt<=10) fprintf(stderr, "  GRAD_MUL_TAGS[%d]: at.tag=%u at.ext=%u at.val=%llu bt.tag=%u bt.ext=%u bt.val=%llu (cop=%u)\n",
	                            _mt, (u32)term_tag(at), (u32)term_ext(at), (unsigned long long)term_val(at),
	                                 (u32)term_tag(bt), (u32)term_ext(bt), (unsigned long long)term_val(bt), cop);
	                    }
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
                        /* d(a/b)/da = 1/b = gy/bt
                           d(a/b)/db = -a/b^2 = -(a/b)/b = -y/bt
                           Using `y` in the b-arm avoids MUL(bt, bt) which
                           reused bt without DUP (linearity violation when
                           bt is a compute TOP like e_sum_bc in softmax). */
                        case UOP_DIV: { BG(thvm_op_raw(ctx,UOP_DIV,gy,bt), thvm_op_raw(ctx,UOP_NEG,thvm_op_raw(ctx,UOP_DIV,thvm_op_raw(ctx,UOP_MUL,gy,y),bt),term_era())); }
                        case UOP_MAX: { Term mask=thvm_op_raw(ctx,UOP_CMP,at,bt); f32 one=1; Term inv=thvm_op_raw(ctx,UOP_SUB,thvm_tensor(ctx,&one,SHAPE(1)),mask);
                            BG(sum_to_shape(ctx,thvm_op_raw(ctx,UOP_MUL,gy,mask),y_shape,a_shape), sum_to_shape(ctx,thvm_op_raw(ctx,UOP_MUL,gy,inv),y_shape,b_shape)); }
	                        case UOP_CMP: GRAD_ZERO(gy);
                        case UOP_SUM: { UG_DROP(bt, thvm_expand(ctx, gy, a_shape)); }
                        case UOP_RMAX: { Term max_bc=thvm_expand(ctx,thvm_reshape(ctx,y,y_shape),a_shape); f32 one=1;
                            Term mask=thvm_op_raw(ctx,UOP_SUB,thvm_tensor(ctx,&one,SHAPE(1)),thvm_op_raw(ctx,UOP_CMP,max_bc,at));
                            UG_DROP(bt, thvm_op_raw(ctx,UOP_MUL,thvm_expand(ctx,gy,a_shape),mask)); }
                        case UOP_RESHAPE: UG_DROP(bt, thvm_reshape(ctx,gy,a_shape));
                        case UOP_PERMUTE: {
                            if (term_tag(bt)==TAG_TEN) { u32 pid=(u32)term_val(bt); TensorMeta *mp=&ctx->tensors[pid];
                                u32 nd=mp->view.numel; u32 pf[MAX_DIM]; tensor_meta_read_u32(ctx, pid, pf, MAX_DIM);
                                u32 inv[MAX_DIM]; for(u32 j=0;j<nd;j++) inv[pf[j]]=j;
                                UG_DROP(bt, thvm_permute(ctx,gy,inv,nd));
	                            } else GRAD_ZERO(gy); }
                        case UOP_EXPAND: UG_DROP(bt, sum_to_shape(ctx,gy,y_shape,a_shape));
                        case UOP_SHRINK: {
                            if (term_tag(bt)==TAG_TEN) { u32 sid=(u32)term_val(bt); TensorMeta *ms=&ctx->tensors[sid];
                                u32 nd=ms->view.numel/2; u32 sf[MAX_DIM*2]; tensor_meta_read_u32(ctx, sid, sf, MAX_DIM*2);
                                u32 pp[MAX_DIM*2]; for(u32 j=0;j<nd;j++){pp[j*2]=sf[j*2];pp[j*2+1]=a_shape.dims[j]-sf[j*2+1];}
                                UG_DROP(bt, thvm_pad(ctx,gy,pp,nd));
	                            } else GRAD_ZERO(gy); }
                        case UOP_PAD: {
                            if (term_tag(bt)==TAG_TEN) { u32 pid2=(u32)term_val(bt); TensorMeta *mp2=&ctx->tensors[pid2];
                                u32 nd=mp2->view.numel/2; u32 pf2[MAX_DIM*2]; tensor_meta_read_u32(ctx, pid2, pf2, MAX_DIM*2);
                                u32 sp[MAX_DIM*2]; for(u32 j=0;j<nd;j++){sp[j*2]=pf2[j*2];sp[j*2+1]=pf2[j*2]+a_shape.dims[j];}
                                UG_DROP(bt, thvm_shrink(ctx,gy,sp,nd));
	                            } else GRAD_ZERO(gy); }
                        // ASSIGN(dst, src) returns dst; gradient flows through dst
                        // and drops the src branch (src only feeds the side effect).
                        case UOP_ASSIGN: UG_DROP(bt, gy);
	                        default: GRAD_ZERO(gy);
	                    }
		                    #undef GRAD2_H
	                    #undef BG_GY
	                    #undef GRAD_SPLIT_AT
	                    #undef BG
	                    #undef UG
		                    #undef UG_DROP
                }

                // GRAD⊳TEN: pure matcher.
                //   y == target   → NUM(1)·gy (accumulate gy into bundle/slot, or return gy)
                //   y ∉ targets   → NUM(0)      (and ERA the incoming gy branch)
                // No requires_grad / creator_op / src_ids inspection. No
                // provenance walk. Movement ops that materialize eagerly into
                // TEN are handled by keeping them as TOPs (scheduler's job).
                if (term_tag(y) == TAG_TEN) {
                    u32 y_id = (u32)term_val(y);

                    // Single-target base case: y == x.
                    if (term_tag(x) == TAG_TEN && (u32)term_val(x) == y_id) {
                        if (grad_mode == GRAD_MODE_DROP) GRAD_DROP(gy);
                        GRAD_RETURN(gy);
                    }

                    // Multi-target matcher (x = ANY). Check target table.
                    if (term_tag(x) == TAG_ANY) {
                        Term slot = term_era();
                        u32 target_index = 0;
                        if (thvm_grad_targets_find_term_at(ctx, loc, y, &target_index, &slot) ||
                            thvm_grad_targets_find_index_at(ctx, loc, y_id, &target_index, &slot)) {
                            if (grad_mode == GRAD_MODE_SLOT &&
                                !(term_tag(slot) == TAG_ERA && term_val(slot) == 0)) {
                                GRAD_RETURN(GRAD_SLOT_ACCUM(slot, gy));
                            }
                            if (grad_mode == GRAD_MODE_KEEP && has_keep_bundle) {
                                thvm_grad_bundle_accum(ctx, loc, target_index, gy);
                                GRAD_RETURN(term_num_f32(0.0f));
                            }
                            // DROP mode or unbacked target: zero.
                            GRAD_ZERO(gy);
                        }
                    }

                    // Non-matching TEN leaf: zero local derivative.
                    GRAD_ZERO(gy);
                }
	                // Unknown y shape → no local derivative contribution → NUM(0).
	                if (getenv("THVM_SCHED_DIAG")) fprintf(stderr, "  GRAD_FALLTHROUGH: y_tag=%u y_ext=%u y_val=%llu\n", term_tag(y), term_ext(y), term_val(y));
	                GRAD_ZERO(gy);
                    #undef GRAD_SLOT_ACCUM
                    #undef GRAD_COMBINE
                    #undef GRAD_KEEP_SEQ2
                    #undef GRAD_SLOT_SEQ2
	                #undef GRAD_SPAWN_ERA
	                #undef GRAD_ZERO
                    #undef GRAD_DROP
	                #undef GRAD_ERASE2
	                #undef GRAD_RETURN
	            }
