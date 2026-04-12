        // TAG_APP: beta reduction + APP-SUP distribution
        case TAG_APP: {
            u64 loc = term_val(t);
            Term fun = heap_read(ctx, loc); // WNF from trampoline
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
                INET_RECURSE();
            }

            // APP-BRI: (θx.body arg) → body[x ← arg]  (same beta rule as LAM)
            if (term_tag(fun) == TAG_BRI) {
                u64 bri_loc = term_val(fun);
                Term arg = heap_read(ctx, loc + 1);
                if (term_tag(arg) == TAG_TOP && term_ext(arg) == UOP_DETACH) {
                    Term src0 = heap_read(ctx, term_val(arg));
                    Term forced = thvm_force_tensor_term(ctx, arg);
                    if (getenv("THVM_LOOP_DIAG")) {
                        fprintf(stderr,
                                "APP_BRI_DETACH loc=%llu arg_loc=%llu src_tag=%u src_ext=%u src_val=%llu forced_tag=%u forced_ext=%u forced_val=%llu\n",
                                (unsigned long long)bri_loc,
                                (unsigned long long)term_val(arg),
                                (u32)term_tag(src0),
                                (u32)term_ext(src0),
                                (unsigned long long)term_val(src0),
                                (u32)term_tag(forced),
                                (u32)term_ext(forced),
                                (unsigned long long)term_val(forced));
                    }
                    if (term_tag(forced) == TAG_TEN) arg = forced;
                }
                heap_set(ctx, bri_loc, arg);
                ctx->itrs++;
                t = heap_read(ctx, bri_loc + 1);
                INET_RECURSE();
            }

            // APP-LAM: beta reduction — (λx.body arg) → body[x ← arg]
            if (term_tag(fun) == TAG_LAM) {
                u64 lam_loc = term_val(fun);
                Term arg = heap_read(ctx, loc + 1);
                if (term_tag(arg) == TAG_TOP && term_ext(arg) == UOP_DETACH) {
                    Term src0 = heap_read(ctx, term_val(arg));
                    Term forced = thvm_force_tensor_term(ctx, arg);
                    if (getenv("THVM_LOOP_DIAG")) {
                        fprintf(stderr,
                                "APP_LAM_DETACH loc=%llu arg_loc=%llu src_tag=%u src_ext=%u src_val=%llu forced_tag=%u forced_ext=%u forced_val=%llu\n",
                                (unsigned long long)lam_loc,
                                (unsigned long long)term_val(arg),
                                (u32)term_tag(src0),
                                (u32)term_ext(src0),
                                (unsigned long long)term_val(src0),
                                (u32)term_tag(forced),
                                (u32)term_ext(forced),
                                (unsigned long long)term_val(forced));
                    }
                    if (term_tag(forced) == TAG_TEN) arg = forced;
                }
                heap_set(ctx, lam_loc, arg);      // link: write arg at var slot
                ctx->itrs++;
                t = heap_read(ctx, lam_loc + 1);  // body
                INET_RECURSE();
            }
            // APP-TEN: discard tensor, return arg (sequencing)
            if (term_tag(fun) == TAG_TEN) {
                tensor_release(ctx, (u32)term_val(fun));
                ctx->itrs++;
                t = heap_read(ctx, loc + 1);
                INET_RECURSE();
            }
            // APP-NUM: numeric value in function position is a sequencing
            // terminal for graph/debug carriers such as grad bundles.
            if (term_tag(fun) == TAG_NUM) {
                ctx->itrs++;
                t = heap_read(ctx, loc + 1);
                INET_RECURSE();
            }
            // APP-ERA: HVM4 semantics — erase the whole application
            if (term_tag(fun) == TAG_ERA) {
                ctx->itrs++;
                t = term_era();
                INET_RECURSE();
            }

            // APP-MAT: pattern match — arg WNF from TAG_MAT_1 frame
            if (term_tag(fun) == TAG_MAT) {
                Term arg = heap_read(ctx, loc + 1); // WNF from trampoline

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
                    INET_RECURSE();
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
                    INET_RECURSE();
                }

                // APP-MAT-CTR: constructor match on TAG_CTR ext. If the
                // constructor code matches, apply the handler to each child.
                if (term_tag(arg) == TAG_CTR) {
                    u64 mat_loc = term_val(fun);
                    u32 match_tag = term_ext(fun);
                    u32 ctr_tag = term_ext(arg);
                    ctx->itrs++;
                    if (match_tag == ctr_tag) {
                        Term r = heap_read(ctx, mat_loc + 0);
                        u64 ctr_loc = term_val(arg);
                        for (u32 i = 0; i < ctr_tag; i++)
                            r = thvm_app(ctx, r, heap_read(ctx, ctr_loc + i));
                        t = r;
                    } else {
                        t = thvm_app(ctx, heap_read(ctx, mat_loc + 1), arg);
                    }
                    INET_RECURSE();
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
                    INET_RECURSE();
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
                INET_RECURSE();
            }

            return t;
        }

        // TAG_LAM: lambda abstraction — returned as-is until applied
        case TAG_LAM:
            return t;

        // TAG_ERA: active eraser (val = slot containing the erased target)
        // ERA(0) is inert.
        case TAG_ERA: {
            u64 era_loc = term_val(t);
            if (era_loc == 0 || era_loc >= ctx->heap_pos) return t;

            Term cur = thvm_era_payload(ctx, heap_read(ctx, era_loc));
            #define ERA_TERM_ARITY(_tg, _ext) ({ \
                u32 _ar = 0; \
                switch ((_tg)) { \
                    case TAG_TOP: \
                        if ((_ext) == UOP_KERNEL) _ar = 0; \
                        else if ((_ext) == UOP_WHERE || (_ext) == UOP_IFZ) _ar = 3; \
                        else if ((_ext) == UOP_GRAD) _ar = 2; \
                        else if ((_ext) == UOP_LOG_PRINT || (_ext) == UOP_DETACH) _ar = 1; \
                        else if (!is_binary((_ext)) && is_elementwise((_ext))) _ar = 1; \
                        else _ar = 2; \
                        break; \
                    case TAG_APP: \
                    case TAG_LAM: \
                    case TAG_BRI: \
                    case TAG_SUP: \
                    case TAG_USP: \
                    case TAG_OP2: \
                    case TAG_EQL: \
                    case TAG_AND: \
                    case TAG_OR: \
                    case TAG_MAT: \
                    case TAG_ANN: \
                        _ar = 2; \
                        break; \
                    case TAG_DSU: \
                    case TAG_DDU: \
                        _ar = 3; \
                        break; \
                    case TAG_DP0: \
                    case TAG_DP1: \
                    case TAG_UDP: \
                    case TAG_ERA: \
                    case TAG_VAR: \
                    case TAG_INC: \
                        _ar = 1; \
                        break; \
                    default: \
                        _ar = 0; \
                        break; \
                } \
                _ar; \
            })

            u8 vtag = term_tag(cur);
            u32 vext = term_ext(cur);
            u64 vval = term_val(cur);
            Term next = term_era();
            int have_next = 0;

            // DUP collapse: if ERA consumes one DP port, project shared value
            // to the opposite port and clear the shared slot.
            if (vtag == TAG_DP0 || vtag == TAG_DP1) {
                u64 dl = vval;
                if (dl < ctx->heap_pos) {
                    u32 other_pi = (vtag == TAG_DP0) ? 1 : 0;
                    u64 other_slot = thvm_dup_port_slot(ctx, dl, other_pi);
                    if (other_slot > 0 && other_slot < ctx->heap_pos)
                        heap_set(ctx, other_slot, heap_read(ctx, dl));
                    thvm_dup_ports_clear_entry(ctx, dl);
                    heap_set(ctx, dl, term_era());
                }
                goto era_done;
            } else if (vtag == TAG_TEN) {
                tensor_release(ctx, (u32)vval);
                goto era_done;
            } else if (vtag == TAG_CTR) {
                u32 ar = vext;
                if (vval < ctx->heap_pos) {
                    for (u32 i = 0; i < ar; i++) {
                        Term child = thvm_era_payload(ctx, heap_read(ctx, vval + i));
                        if (term_tag(child) == TAG_ERA && term_val(child) == 0) continue;
                        if (!have_next) {
                            next = child;
                            have_next = 1;
                        } else {
                            thvm_spawn_detached_era(ctx, child);
                        }
                    }
                    if (have_next) goto era_continue;
                }
            } else {
                u32 ar = ERA_TERM_ARITY(vtag, vext);
                if (ar > 0 && vval < ctx->heap_pos) {
                    for (u32 i = 0; i < ar; i++) {
                        Term child = thvm_era_payload(ctx, heap_read(ctx, vval + i));
                        if (term_tag(child) == TAG_ERA && term_val(child) == 0) continue;
                        if (!have_next) {
                            next = child;
                            have_next = 1;
                        } else {
                                thvm_spawn_detached_era(ctx, child);
                        }
                    }
                    if (have_next) goto era_continue;
                }
            }
era_done:
            heap_set(ctx, era_loc, term_era());
            ctx->itrs++;
            #undef ERA_TERM_ARITY
            return term_era();
era_continue:
            heap_set(ctx, era_loc, next);
            ctx->itrs++;
            #undef ERA_TERM_ARITY
            return term_new(TAG_ERA, term_ext(t) ^ 1u, era_loc);
        }

        // TAG_REF: unfold definition — clone body for fresh variable bindings.
        // TODO: optimize for loops — clone only lambda scaffolding, share compute.
        case TAG_REF: {
            u32 name = (u32)term_ext(t);
            assert(name < ctx->def_count);
            ctx->itrs++;
            t = term_clone(ctx, ctx->defs[name]);
            INET_RECURSE();
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
            Term val = heap_read(ctx, dup_loc); // WNF from trampoline
            heap_set(ctx, dup_loc, val);
            #define DUP_STATE_RETURN(_src, _dp0v, _dp1v) do { \
                Term _v0 = (_dp0v); \
                Term _v1 = (_dp1v); \
                u64 _s0 = thvm_dup_port_slot(ctx, dup_loc, 0); \
                u64 _s1 = thvm_dup_port_slot(ctx, dup_loc, 1); \
                if (_s0 > 0 && _s0 < ctx->heap_pos) heap_set(ctx, _s0, _v0); \
                if (_s1 > 0 && _s1 < ctx->heap_pos) heap_set(ctx, _s1, _v1); \
                thvm_dup_ports_clear_entry(ctx, dup_loc); \
                heap_set(ctx, dup_loc, term_era()); \
                ctx->itrs++; \
                RETURN_REDUCED(dp_index == 0 ? _v0 : _v1); \
            } while (0)

            // DUP ⊳ SUP
            if (term_tag(val) == TAG_SUP) {
                u32 sup_label = term_ext(val);
                u64 sup_loc = term_val(val);

                if (dup_label == sup_label) {
                    // ANNIHILATION: same label — project directly
                    Term tm0 = heap_read(ctx, sup_loc + 0);
                    Term tm1 = heap_read(ctx, sup_loc + 1);
                    DUP_STATE_RETURN(val, tm0, tm1);
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
                    DUP_STATE_RETURN(val, new_sup0, new_sup1);
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
                DUP_STATE_RETURN(val, lam0, lam1);
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
                DUP_STATE_RETURN(val, bri0, bri1);
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
                DUP_STATE_RETURN(val, a0, a1);
            }

            // DUP ⊳ atoms: copy (both projections get same value)
            if (term_tag(val) == TAG_TEN) DUP_STATE_RETURN(val, val, val);
            if (term_tag(val) == TAG_ERA) DUP_STATE_RETURN(val, val, val);
            if (term_tag(val) == TAG_NUM) DUP_STATE_RETURN(val, val, val);
            if (term_tag(val) == TAG_ANY) DUP_STATE_RETURN(val, val, val);
            if (term_tag(val) == TAG_CTR) DUP_STATE_RETURN(val, val, val);
	            // DUP ⊳ TOP: commute by duplicating the node and splitting children.
	            if (term_tag(val) == TAG_TOP) {
	                u32 uop = term_ext(val);
	                if (uop == UOP_KERNEL) {
	                    // Create independent dispatch instance: same kid, fresh heap slot.
	                    // One copy consumed this iteration, the other survives for next.
	                    u64 old_loc = term_val(val);
	                    u64 new_loc = heap_alloc(ctx, 1);
	                    heap_set(ctx, new_loc, heap_read(ctx, old_loc)); // copy kid
	                    Term copy = term_new(TAG_TOP, UOP_KERNEL, new_loc);
	                    DUP_STATE_RETURN(val, val, copy);
	                }
                    if (uop == UOP_DETACH) {
                        Term forced = thvm_eval(ctx, val);
                        if (getenv("THVM_LOOP_DIAG")) {
                            fprintf(stderr,
                                    "DUP_DETACH val_loc=%llu forced_tag=%u forced_ext=%u forced_val=%llu\n",
                                    (unsigned long long)term_val(val),
                                    (u32)term_tag(forced),
                                    (u32)term_ext(forced),
                                    (unsigned long long)term_val(forced));
                        }
                        if (term_tag(forced) == TAG_TEN ||
                            term_tag(forced) == TAG_ERA ||
                            term_tag(forced) == TAG_NUM ||
                            term_tag(forced) == TAG_ANY ||
                            term_tag(forced) == TAG_CTR) {
                            DUP_STATE_RETURN(forced, forced, forced);
                        }
                    }
	                u64 val_loc = term_val(val);
	                u32 arity = (uop == UOP_WHERE || uop == UOP_IFZ) ? 3 : 2;
	                if (!is_binary(uop) && is_elementwise(uop)) arity = 1;

	                u64 r0 = heap_alloc(ctx, arity);
	                u64 r1 = heap_alloc(ctx, arity);
	                for (u32 i = 0; i < arity; i++) {
	                    Term child = heap_read(ctx, val_loc + i);
	                    u8 ct = term_tag(child);
	                    // Copyable atoms commute by shared reference.
	                    if (ct == TAG_TEN || ct == TAG_NUM || ct == TAG_ERA ||
	                        ct == TAG_ANY || ct == TAG_CTR) {
	                        heap_set(ctx, r0 + i, child);
	                        heap_set(ctx, r1 + i, child);
	                    } else {
	                        u64 cdup = heap_alloc(ctx, 1);
	                        heap_set(ctx, cdup, child);
	                        heap_set(ctx, r0 + i, term_new(TAG_DP0, dup_label, cdup));
	                        heap_set(ctx, r1 + i, term_new(TAG_DP1, dup_label, cdup));
	                    }
	                }

	                Term n0 = term_new(TAG_TOP, uop, r0);
	                Term n1 = term_new(TAG_TOP, uop, r1);
	                const View *vv = st_get(val_loc);
	                if (vv) {
	                    st_set(r0, vv);
	                    st_set(r1, vv);
	                }
	                if (uop == UOP_GRAD) {
	                    thvm_grad_targets_share(ctx, r0, val_loc);
	                    thvm_grad_targets_share(ctx, r1, val_loc);
	                }

	                DUP_STATE_RETURN(val, n0, n1);
	            }

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
                DUP_STATE_RETURN(val, new_usp0, new_usp1);
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
                DUP_STATE_RETURN(val, n0, n1);
            }

            // Not yet reducible — return DUP as-is
            #undef DUP_STATE_RETURN
            return t;
        }


        case TAG_OP2: {
            u64 loc = term_val(t);
            u32 opr = term_ext(t);
            Term x = heap_read(ctx, loc); // WNF from trampoline
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
                INET_RECURSE();
            }

            Term y = heap_read(ctx, loc + 1); // WNF from TAG_OP2_1 frame

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
                INET_RECURSE();
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
            if (term_tag(sub) == TAG_TOP && term_ext(sub) == UOP_DETACH) {
                Term forced = thvm_force_tensor_term(ctx, sub);
                if (term_tag(forced) == TAG_TEN) {
                    heap_set(ctx, loc, forced);
                    ctx->itrs++;
                    return forced;
                }
                return forced;
            }
            return sub; // trampoline re-enters
        }

        // TAG_BRI: bridge — returned as-is until applied (WNF, like LAM)
        case TAG_BRI:
            return t;

        // TAG_ANN: annotation — transparent, strip and return inner term
        case TAG_ANN: {
            u64 loc = term_val(t);
            ctx->itrs++;
            t = heap_read(ctx, loc);  // the term (slot 0)
            INET_RECURSE();
        }

        // TAG_DSU: dynamic SUP — reduce label_expr to NUM, then create normal SUP
        case TAG_DSU: {
            u64 loc = term_val(t);
            Term label_expr = heap_read(ctx, loc); // WNF from trampoline
            heap_set(ctx, loc, label_expr);
            if (term_tag(label_expr) != TAG_NUM) return t;  // not ready
            u32 label = term_as_u32(label_expr);
            Term a = heap_read(ctx, loc + 1);
            Term b = heap_read(ctx, loc + 2);
            ctx->itrs++;
            t = thvm_sup(ctx, label, a, b);
            INET_RECURSE();
        }

        // TAG_DDU: dynamic DUP — reduce label_expr to NUM, then dup val, apply bod
        case TAG_DDU: {
            u64 loc = term_val(t);
            Term label_expr = heap_read(ctx, loc); // WNF from trampoline
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
            INET_RECURSE();
        }

        // TAG_INC: priority wrapper — transparent during normal reduction
        case TAG_INC: {
            u64 loc = term_val(t);
            ctx->itrs++;
            t = heap_read(ctx, loc);
            INET_RECURSE();
        }

        // ── Type enumeration machinery ───────────────────────────────────

        // TAG_EQL: structural equality — reduce both sides, compare
        // Pattern: same as TAG_OP2 (reduce left, check SUP, reduce right, check SUP, compare)
        case TAG_EQL: {
            u64 loc = term_val(t);
            // Reduce left operand
            Term a = heap_read(ctx, loc); // WNF from trampoline
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
                INET_RECURSE();
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
                INET_RECURSE();
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
                INET_RECURSE();
            }

            Term b = heap_read(ctx, loc + 1); // WNF from TAG_EQL_1 frame

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
                INET_RECURSE();
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
                INET_RECURSE();
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
                INET_RECURSE();
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
                INET_RECURSE();
            }
            // Different types → not equal
            ctx->itrs++;
            return term_num_u32(0);
        }

        // TAG_AND: short-circuit boolean AND — reduce left, short-circuit on 0
        case TAG_AND: {
            u64 loc = term_val(t);
            Term a = heap_read(ctx, loc); // WNF from trampoline
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
                INET_RECURSE();
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
                INET_RECURSE();
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
                INET_RECURSE();
            }
            // AND-ERA: erased → ERA
            if (term_tag(a) == TAG_ERA) return term_era();
            return t;
        }

        // TAG_OR: short-circuit boolean OR — reduce left, short-circuit on nonzero
        case TAG_OR: {
            u64 loc = term_val(t);
            Term a = heap_read(ctx, loc); // WNF from trampoline
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
                INET_RECURSE();
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
                INET_RECURSE();
            }
            // OR-ZER: zero → pass through to b
            if (term_tag(a) == TAG_NUM && term_as_u32(a) == 0) {
                ctx->itrs++;
                t = heap_read(ctx, loc + 1);
                INET_RECURSE();
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
            Term val = heap_read(ctx, udp_loc); // WNF from trampoline

            // UDP ⊳ USP (same label): consume ONE branch, keep producing from other
            if (term_tag(val) == TAG_USP && term_ext(val) == udp_label) {
                u64 usp_loc = term_val(val);
                Term a = heap_read(ctx, usp_loc + 0);
                Term b = heap_read(ctx, usp_loc + 1);
                // UDP keeps pointing to b (unconsumed remainder)
                heap_set(ctx, udp_loc, b);
                ctx->itrs++;
                t = a;
                INET_RECURSE();
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
                INET_RECURSE();
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
                INET_RECURSE();
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
                INET_RECURSE();
            }

            return t;  // stuck
        }

        // TAG_USP: unordered superposition — WNF (like TAG_SUP)
        case TAG_USP:
            return t;

        // TAG_CTR: output bundle carrier / sequencing tuple.
        // - CTR(N=1){x} -> x
        // - CTR(N>1){ERA, xs...} -> CTR(N-1){xs...}
        case TAG_CTR: {
            u32 ar = term_ext(t);
            u64 loc = term_val(t);
            if (ar == 0 || loc == 0 || loc >= ctx->heap_pos) return t;
            Term head = heap_read(ctx, loc);
            if (ar == 1) {
                ctx->itrs++;
                return head;
            }
            if (term_tag(head) == TAG_ERA) {
                ctx->itrs++;
                return term_new(TAG_CTR, (u8)(ar - 1), loc + 1);
            }
            return t;
        }

        // TAG_MAT: pattern match — WNF atom (fired by APP)
        case TAG_MAT:
            return t;

        // TAG_ANY: wildcard — WNF atom
        case TAG_ANY:
            return t;
