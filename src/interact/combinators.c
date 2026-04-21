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
                // Clear the APP's arg slot so ALO/DP tokens move exclusively to
                // the lambda binder. Without this, a linear ALO stays aliased
                // from both heap[loc+1] and heap[lam_loc], and the second
                // reader walks into a stale memoised projection.
                heap_set(ctx, loc + 0, term_era());
                heap_set(ctx, loc + 1, term_era());
                ctx->itrs++;
                t = heap_read(ctx, lam_loc + 1);  // body (may be ALO; let it interact explicitly next)
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
                    /* Retry-from-fixed-point for multi-target GRAD.
                     *
                     * When a CTR child is an unreduced compute TOP
                     * (ADD/MUL/.../KERNEL/EXEC), defer this rule. The
                     * scheduler's global pass then materialises each
                     * child to TEN before MAT-CTR re-fires and binds.
                     *
                     * Why: in a training loop the body of MAT is a
                     * SEQ chain of ASSIGNs that blit-mutate forward
                     * tensors in place. If the n'th ASSIGN's src is
                     * still an unreduced backward TOP at binding
                     * time, reducing it later (from inside the SEQ)
                     * fires after the (n-1)'th ASSIGN has already
                     * blitted — so the n'th backward reads stale
                     * forward values.
                     *
                     * Pure view ops (RESHAPE/PERMUTE/EXPAND/SHRINK/
                     * PAD) are exempt: they re-view a single source
                     * and don't pull values from multiple forward
                     * tensors, so no stale-read risk. Also required
                     * for correctness: backward chains that are
                     * trivial EXPAND(scalar) (e.g. sum(w+b) gradient)
                     * never get picked up by the scheduler's global
                     * pass, so deferring on them would livelock. */
                    /* Previously this deferred when CTR children were
                     * unreduced compute TOPs (ADD/MUL/etc.), waiting
                     * for the scheduler to materialize them to TENs
                     * first. Under the transparent-DP-projection rule
                     * for atom-shared DUP⊳TOP, scheduler can't build
                     * kernels for backward chains that contain VARs
                     * bound by this very MAT-CTR — creating a
                     * deadlock. Bind eagerly: the LAM body can use
                     * compute TOP children just fine; they'll reduce
                     * when consumed. */
                    (void)match_tag; (void)ctr_tag;
                    ctx->itrs++;
                    if (thvm_loop_diag_enabled()) {
                        u64 ctr_loc_dbg = term_val(arg);
                        fprintf(stderr,
                                "MAT_CTR match_tag=%u ctr_tag=%u ctr_loc=%llu\n",
                                match_tag, ctr_tag,
                                (unsigned long long)ctr_loc_dbg);
                        for (u32 i = 0; i < ctr_tag && i < 8; i++) {
                            Term child = heap_read(ctx, ctr_loc_dbg + i);
                            fprintf(stderr, "  [%u]=%u/%u@%llu", i,
                                    (u32)term_tag(child), (u32)term_ext(child),
                                    (unsigned long long)term_val(child));
                            thvm_probe_print_ten(ctx, child, "v");
                            /* If child is TAG_TOP with 2 args (like ADD),
                               recurse one level to show args, and for each
                               arg that's a DP, show what's at its DUP cell. */
                            if (term_tag(child) == TAG_TOP) {
                                u64 cloc = term_val(child);
                                if (cloc > 0 && cloc + 1 < ctx->heap_pos) {
                                    for (u32 ai = 0; ai < 2; ai++) {
                                        Term a = heap_read(ctx, cloc + ai);
                                        fprintf(stderr, " arg%u=%u/%u@%llu",
                                                ai, (u32)term_tag(a),
                                                (u32)term_ext(a),
                                                (unsigned long long)term_val(a));
                                        thvm_probe_print_ten(ctx, a, "v");
                                        /* If arg is DP, show heap[dup_loc] */
                                        if (term_tag(a) == TAG_DP0 ||
                                            term_tag(a) == TAG_DP1) {
                                            u64 dl = term_val(a);
                                            if (dl > 0 && dl < ctx->heap_pos) {
                                                Term dv = heap_read(ctx, dl);
                                                fprintf(stderr,
                                                        " dup[%llu]=%u/%u@%llu",
                                                        (unsigned long long)dl,
                                                        (u32)term_tag(dv),
                                                        (u32)term_ext(dv),
                                                        (unsigned long long)term_val(dv));
                                            }
                                        }
                                    }
                                }
                            }
                            fputc('\n', stderr);
                        }
                    }
                    if (match_tag == ctr_tag) {
                        u64 ctr_loc = term_val(arg);
                        /* Fix for multi-grad-in-loop stale-read bug.
                         * THVM_MAT_FORCE_WNF=1: eagerly thvm_eval each
                         * TAG_TOP CTR child to materialise as TEN before
                         * binding. Works for the target bug but:
                         *   - Violates "no eager eval in interaction
                         *     rules" policy.
                         *   - Aborts (SIGTRAP) on UOP_EXPAND children
                         *     in twoparam_sum (scheduler assertion).
                         * Three lazy alternatives (FUSE-wrap children,
                         * SEQ-chain FUSE wrappers, FUSE-wrap the CTR
                         * arg to trigger FUSE⊳CTR distribution) all
                         * tried and all fail — the FUSE-based paths
                         * disrupt the APP-MAT-CTR binding flow such
                         * that ASSIGNs never fire. See round 12-13
                         * in resources/plans/recursive_grad_loop_fix.md.
                         */
                        int do_eager = getenv("THVM_MAT_FORCE_WNF") &&
                                       getenv("THVM_MAT_FORCE_WNF")[0] != '0';
                        if (do_eager) {
                            for (u32 i = 0; i < ctr_tag; i++) {
                                Term child = heap_read(ctx, ctr_loc + i);
                                if (term_tag(child) == TAG_TOP) {
                                    Term reduced = thvm_eval(ctx, child);
                                    if (term_tag(reduced) == TAG_TEN ||
                                        term_tag(reduced) == TAG_NUM ||
                                        term_tag(reduced) == TAG_ERA) {
                                        heap_set(ctx, ctr_loc + i, reduced);
                                    }
                                }
                            }
                        }
                        Term r = heap_read(ctx, mat_loc + 0);
                        for (u32 i = 0; i < ctr_tag; i++) {
                            r = thvm_app(ctx, r, heap_read(ctx, ctr_loc + i));
                        }
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
                        _ar = thvm_uop_storage_arity((_ext)); \
                        break; \
                    case TAG_APP: \
                    case TAG_LAM: \
                    case TAG_BRI: \
                    case TAG_SEQ: \
                    case TAG_SUP: \
                    case TAG_USP: \
                    case TAG_OP2: \
                    case TAG_EQL: \
                    case TAG_AND: \
                    case TAG_OR: \
                    case TAG_MAT: \
                    case TAG_ANN: \
                    case TAG_ALO: \
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
            if (vtag == TAG_DP0 || vtag == TAG_DP1 ||
                vtag == TAG_GF  || vtag == TAG_GB) {
                /* HVM4-style: ERA on an aux DP port is handled via the
                 * DUP cell itself, not via a side channel into the
                 * sibling's consumer slot. Two cases:
                 *  a) cell has SUB bit — the sibling already fired and
                 *     stored its clone here for us. We got erased before
                 *     consuming it; emit an explicit detached ERA on the
                 *     orphan clone so the step graph shows the sweep.
                 *  b) cell has no SUB bit — we are the first aux to
                 *     drop. Body survives so the sibling can take it
                 *     as identity. Mark the cell SUB-of-body. */
                u64 dl = vval;
                if (dl < ctx->heap_pos) {
                    Term cell = heap_read(ctx, dl);
                    if (term_is_sub(cell)) {
                        Term orphan = term_strip_sub(cell);
                        if (term_tag(orphan) != TAG_ERA)
                            thvm_spawn_detached_era(ctx, orphan);
                        heap_set(ctx, dl, term_era());
                    } else {
                        heap_set(ctx, dl, term_set_sub(cell));
                    }
                }
                goto era_done;
            } else if (vtag == TAG_VAR) {
                // ERA ⊳ VAR erases the binder-input cell (VAR slot), not an
                // arbitrary consumer edge. If VAR is already substituted, keep
                // erasing through the resolved value.
                if (vval >= ctx->heap_pos) goto era_done;
                Term sub = heap_read(ctx, vval);
                if (term_is_sub(sub)) {
                    // Binder still unsubstituted: materialize active ERA only in the
                    // binder cell (local rule). Consumer VAR@loc sites stay as VAR
                    // until they reduce and read the slot (see TAG_VAR path).
                    u64 el2 = heap_alloc(ctx, 1);
                    heap_set(ctx, el2, term_era());
                    heap_set(ctx, vval, term_era_at(el2));
                    goto era_done;
                }
                next = thvm_era_payload(ctx, sub);
                if (!(term_tag(next) == TAG_ERA && term_val(next) == 0))
                    goto era_continue;
                goto era_done;
            } else if (vtag == TAG_TEN) {
                tensor_release(ctx, (u32)vval);
                goto era_done;
            } else if (vtag == TAG_CTR) {
                u32 ar = vext;
                if (vval < ctx->heap_pos) {
                    for (u32 i = 0; i < ar; i++) {
                        Term child = thvm_era_payload(ctx, heap_read(ctx, vval + i));
                        heap_set(ctx, vval + i, term_era());
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
                        heap_set(ctx, vval + i, term_era());
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

        // TAG_REF: convert named definition reference into lazy allocation frontier.
        case TAG_REF: {
            u32 name = (u32)term_ext(t);
            assert(name < ctx->def_count);
            if (ctx->def_books[name] == 0)
                ctx->def_books[name] = thvm_book_from_dynamic(ctx, ctx->defs[name]);
            ctx->itrs++;
            t = thvm_alo_realize(ctx, ctx->def_books[name], 0);
            INET_RECURSE();
        }

        // TAG_ALO: force exactly one static/book layer into dynamic net.
        case TAG_ALO: {
            ctx->itrs++;
            t = thvm_alo_force(ctx, t);
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
            Term cell = heap_read(ctx, dup_loc);
            /* HVM4-style consumer-driven DUP:
             *  - cell has SUB bit → sibling fired and stored our value
             *    here, or our sibling was ERA'd and the DUP collapsed
             *    to identity-of-body. Either way, take the stored
             *    value and we're done. */
            if (term_is_sub(cell)) {
                Term v = term_strip_sub(cell);
                /* TEN refcount: the SUB cell counted once for the
                 * sibling's stored ref. Reading it into this aux's
                 * output creates another live holder. */
                if (term_tag(v) == TAG_TEN)
                    tensor_incref(ctx, (u32)term_val(v));
                ctx->itrs++;
                RETURN_REDUCED(v);
            }
            Term val = cell; // body, WNF from trampoline
            /* Transparent projection for pure compute TOPs:
             * return body directly without firing DUP so both auxes
             * share the same TAG_TOP (materialize once, share tensor).
             * Only non-effectful compute uops qualify. */
            if (term_tag(val) == TAG_TOP) {
                u32 _vuop = term_ext(val);
                if (_vuop != UOP_DETACH && _vuop != UOP_ASSIGN &&
                    _vuop != UOP_KERNEL && _vuop != UOP_EXEC &&
                    _vuop != UOP_GRAD) {
                    RETURN_REDUCED(val);
                }
            }
            /* HVM4 heap_subst_cop: write the SIBLING's clone into this
             * DUP cell with SUB bit set, return the FIRING aux's clone.
             * The sibling, when its consumer later pulls it, will read
             * the SUB cell above and take its value directly — no side
             * channel, no sibling-slot lookup. */
            #define DUP_STATE_RETURN(_src, _dp0v, _dp1v) do { \
                Term _v0 = (_dp0v); \
                Term _v1 = (_dp1v); \
                Term _sibling = dp_index == 0 ? _v1 : _v0; \
                Term _mine    = dp_index == 0 ? _v0 : _v1; \
                heap_set(ctx, dup_loc, term_set_sub(_sibling)); \
                ctx->itrs++; \
                RETURN_REDUCED(_mine); \
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

            // DUP ⊳ atoms: copy (both projections get same value).
            // For TEN: duplicate tensor reference — both DP projections alias
            // the same tensor_id. Bump tensor refcount so that an ERA on one
            // branch doesn't tensor_release the still-live other branch.
            if (term_tag(val) == TAG_TEN) {
                tensor_incref(ctx, (u32)term_val(val));
                DUP_STATE_RETURN(val, val, val);
            }
            if (term_tag(val) == TAG_ERA) {
                if (thvm_dup_diag_enabled()) {
                    fprintf(stderr,
                            "DUP_ON_ERA dup=%llu lab=%u fire=DP%u val_era=%llu [cell was erased by prior fire]\n",
                            (unsigned long long)dup_loc, dup_label, dp_index,
                            (unsigned long long)term_val(val));
                }
                DUP_STATE_RETURN(val, val, val);
            }
            if (term_tag(val) == TAG_NUM) DUP_STATE_RETURN(val, val, val);
            if (term_tag(val) == TAG_ANY) DUP_STATE_RETURN(val, val, val);
            if (term_tag(val) == TAG_CTR) DUP_STATE_RETURN(val, val, val);
	            // DUP ⊳ TOP: commute by duplicating the node and splitting children.
	            // Note: pure compute TOPs are handled earlier as transparent DP
	            // projection (body returned without firing DUP). This branch
	            // only runs for effectful TOPs (ASSIGN/KERNEL/EXEC/DETACH/GRAD).
	            if (term_tag(val) == TAG_TOP) {
	                u32 uop = term_ext(val);
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
	                u32 arity = thvm_uop_storage_arity(uop);

	                u64 r0 = heap_alloc(ctx, arity);
	                u64 r1 = heap_alloc(ctx, arity);
	                for (u32 i = 0; i < arity; i++) {
	                    Term child = heap_read(ctx, val_loc + i);
	                    u8 ct = term_tag(child);
	                    // Copyable atoms commute by shared reference.
	                    // TAG_ALO is memoized — sharing by reference is safe
	                    // because force returns the same realized value for
	                    // every consumer (or fresh-DUP-wraps for DP results).
	                    if (ct == TAG_TEN || ct == TAG_NUM || ct == TAG_ERA ||
	                        ct == TAG_ANY || ct == TAG_CTR || ct == TAG_ALO) {
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
	                /* Copy the full multi-view ShapeTracker, not just the
	                 * last view. For composed view chains (reshape+expand
	                 * that can't merge into a single View), flattening
	                 * here loses the stride-0 broadcast info and makes
	                 * downstream kernels index the buffer as if it were
	                 * dense contiguous — reading the same element for
	                 * every broadcast position. Hits the matmul-in-loop
	                 * backward where both BG branches DUP-commute through
	                 * the EXPAND wrappers. */
	                const ShapeTracker *ast = st_get_tracker(val_loc);
	                if (ast) {
	                    st_set_tracker(r0, ast);
	                    st_set_tracker(r1, ast);
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
                term_tag(val) == TAG_OR  || term_tag(val) == TAG_MAT ||
                term_tag(val) == TAG_SEQ) {
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

        // ─── GRAD pair: (fwd, bwd) = GRAD(y) ───────────────────────────
        // TAG_GF and TAG_GB share one rule with branching on `is_bwd`,
        // mirroring the DP0/DP1 pattern. The cell at term_val(t) holds
        // y; the ext field carries the target tensor id.
        //
        // When the first aux fires: read body y (WNF from trampoline),
        // compute both forward and backward outputs via the chain rule
        // for y's shape, write the sibling's clone into the cell with
        // the SUB bit, return the firing aux's clone.
        //
        // When the second aux fires later: cell has SUB bit set →
        // strip it and return directly (HVM4 heap_subst_cop pattern).
        case TAG_GF:
        case TAG_GB: {
            u32 is_bwd = (tag == TAG_GB) ? 1 : 0;
            u32 target_tid = term_ext(t);
            u64 cell_loc = term_val(t);
            Term cell = heap_read(ctx, cell_loc);

            if (term_is_sub(cell)) {
                Term v = term_strip_sub(cell);
                if (term_tag(v) == TAG_TEN)
                    tensor_incref(ctx, (u32)term_val(v));
                ctx->itrs++;
                RETURN_REDUCED(v);
            }
            /* Resolve y through DP / VAR so the rule's tag dispatch sees
             * the underlying TEN or compute TOP, not the projection. */
            Term y = cell;
            for (int _i = 0; _i < 32; _i++) {
                u8 yt = term_tag(y);
                if (yt == TAG_DP0 || yt == TAG_DP1) {
                    u64 l = term_val(y);
                    if (l == 0 || l >= ctx->heap_pos) break;
                    Term nxt = heap_read(ctx, l);
                    if (term_is_sub(nxt)) nxt = term_strip_sub(nxt);
                    if (nxt == y) break;
                    y = nxt;
                    continue;
                }
                if (yt == TAG_VAR) {
                    u64 l = term_val(y);
                    if (l == 0 || l >= ctx->heap_pos) break;
                    Term sub = heap_read(ctx, l);
                    if (term_is_sub(sub)) break;
                    if (sub == y) break;
                    y = sub;
                    continue;
                }
                break;
            }

            #define GRAD_STATE_RETURN(_fwd, _bwd) do { \
                Term _sibling = is_bwd ? (_fwd) : (_bwd); \
                Term _mine    = is_bwd ? (_bwd) : (_fwd); \
                heap_set(ctx, cell_loc, term_set_sub(_sibling)); \
                ctx->itrs++; \
                RETURN_REDUCED(_mine); \
            } while (0)

            // GRAD ⊳ TEN(t):  fwd = t, bwd = 1 if t == target else 0
            if (term_tag(y) == TAG_TEN) {
                u32 tid = (u32)term_val(y);
                Term fwd = y;
                Term bwd = (tid == target_tid) ? term_num_f32(1.0f)
                                               : term_num_f32(0.0f);
                GRAD_STATE_RETURN(fwd, bwd);
            }

            // GRAD ⊳ TOP(uop, args): chain rule per uop.
            if (term_tag(y) == TAG_TOP) {
                u32 uop = term_ext(y);
                u64 yloc = term_val(y);

                // Binary elementwise rules (both operands live at yloc+0/1).
                if (uop == UOP_ADD || uop == UOP_SUB || uop == UOP_MUL) {
                    Term a = heap_read(ctx, yloc + 0);
                    Term b = heap_read(ctx, yloc + 1);
                    Term a_fwd, a_bwd, b_fwd, b_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    thvm_grad_pair(ctx, target_tid, b, &b_fwd, &b_bwd);
                    Term fwd, bwd;
                    if (uop == UOP_ADD) {
                        // d(a+b) = da + db. No operand sharing.
                        fwd = thvm_op_raw(ctx, UOP_ADD, a_fwd, b_fwd);
                        bwd = thvm_op_raw(ctx, UOP_ADD, a_bwd, b_bwd);
                    } else if (uop == UOP_SUB) {
                        // d(a-b) = da - db. No operand sharing.
                        fwd = thvm_op_raw(ctx, UOP_SUB, a_fwd, b_fwd);
                        bwd = thvm_op_raw(ctx, UOP_SUB, a_bwd, b_bwd);
                    } else {
                        // Leibniz: d(a*b) = da*b + a*db. The forward ports
                        // a_fwd/b_fwd are each needed twice (forward MUL and
                        // backward cross-term), so DUP them before use —
                        // otherwise the second consumer reads a SUB-cell
                        // written by the first and gets the sibling value.
                        Term a_fwd_0, a_fwd_1, b_fwd_0, b_fwd_1;
                        thvm_dup(ctx, thvm_fresh_label(ctx), a_fwd, &a_fwd_0, &a_fwd_1);
                        thvm_dup(ctx, thvm_fresh_label(ctx), b_fwd, &b_fwd_0, &b_fwd_1);
                        fwd = thvm_op_raw(ctx, UOP_MUL, a_fwd_0, b_fwd_0);
                        Term l = thvm_op_raw(ctx, UOP_MUL, a_bwd, b_fwd_1);
                        Term r = thvm_op_raw(ctx, UOP_MUL, a_fwd_1, b_bwd);
                        bwd = thvm_op_raw(ctx, UOP_ADD, l, r);
                    }
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // MM: Leibniz same shape as MUL — emit
                // fwd = MM(fa, fb),  bwd = MM(da, fb') + MM(fa', db).
                if (uop == UOP_MM) {
                    Term a = heap_read(ctx, yloc + 0);
                    Term b = heap_read(ctx, yloc + 1);
                    Term a_fwd, a_bwd, b_fwd, b_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    thvm_grad_pair(ctx, target_tid, b, &b_fwd, &b_bwd);
                    Term af0, af1, bf0, bf1;
                    thvm_dup(ctx, thvm_fresh_label(ctx), a_fwd, &af0, &af1);
                    thvm_dup(ctx, thvm_fresh_label(ctx), b_fwd, &bf0, &bf1);
                    Term fwd = thvm_op_raw(ctx, UOP_MM, af0, bf0);
                    Term l = thvm_op_raw(ctx, UOP_MM, a_bwd, bf1);
                    Term r = thvm_op_raw(ctx, UOP_MM, af1, b_bwd);
                    Term bwd = thvm_op_raw(ctx, UOP_ADD, l, r);
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // Unary NEG: d(-a) = -da. No operand reuse.
                if (uop == UOP_NEG) {
                    Term a = heap_read(ctx, yloc + 0);
                    Term a_fwd, a_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    Term fwd = thvm_op_raw(ctx, UOP_NEG, a_fwd, term_era());
                    Term bwd = thvm_op_raw(ctx, UOP_NEG, a_bwd, term_era());
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // DIV: d(a/b) = (da·b - a·db) / b²
                // fwd = DIV(fa, fb), bwd = DIV(SUB(MUL(da, fb'), MUL(fa', db)),
                //                             MUL(fb'', fb'''))
                // fa used twice (fwd, bwd cross), fb used FOUR times (fwd,
                // bwd.l, bwd.r numerator, bwd denom twice).
                if (uop == UOP_DIV) {
                    Term a = heap_read(ctx, yloc + 0);
                    Term b = heap_read(ctx, yloc + 1);
                    Term a_fwd, a_bwd, b_fwd, b_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    thvm_grad_pair(ctx, target_tid, b, &b_fwd, &b_bwd);
                    Term af0, af1, bf0, bf1, bf2, bf3, bftmp;
                    thvm_dup(ctx, thvm_fresh_label(ctx), a_fwd, &af0, &af1);
                    thvm_dup(ctx, thvm_fresh_label(ctx), b_fwd, &bf0, &bftmp);
                    thvm_dup(ctx, thvm_fresh_label(ctx), bftmp, &bf1, &bf2);
                    /* bf2 needs one more DUP for denom (b²) */
                    (void)bf3; /* denom gets bf2 itself, then bf1 for numerator b */
                    Term fwd = thvm_op_raw(ctx, UOP_DIV, af0, bf0);
                    Term num_l = thvm_op_raw(ctx, UOP_MUL, a_bwd, bf1);
                    Term num_r = thvm_op_raw(ctx, UOP_MUL, af1, b_bwd);
                    Term num   = thvm_op_raw(ctx, UOP_SUB, num_l, num_r);
                    /* denom = b² — need two more copies of b_fwd */
                    Term bf_d0, bf_d1;
                    thvm_dup(ctx, thvm_fresh_label(ctx), bf2, &bf_d0, &bf_d1);
                    Term denom = thvm_op_raw(ctx, UOP_MUL, bf_d0, bf_d1);
                    Term bwd = thvm_op_raw(ctx, UOP_DIV, num, denom);
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // Unary EXP: d(exp a) = exp(a) · da.
                // Reuses fa (once in forward, once in derivative MUL).
                if (uop == UOP_EXP) {
                    Term a = heap_read(ctx, yloc + 0);
                    Term a_fwd, a_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    Term af0, af1;
                    thvm_dup(ctx, thvm_fresh_label(ctx), a_fwd, &af0, &af1);
                    Term fwd = thvm_op_raw(ctx, UOP_EXP, af0, term_era());
                    Term expa = thvm_op_raw(ctx, UOP_EXP, af1, term_era());
                    Term bwd = thvm_op_raw(ctx, UOP_MUL, a_bwd, expa);
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // Unary LOG: d(log a) = da / a.  a used twice.
                if (uop == UOP_LOG) {
                    Term a = heap_read(ctx, yloc + 0);
                    Term a_fwd, a_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    Term af0, af1;
                    thvm_dup(ctx, thvm_fresh_label(ctx), a_fwd, &af0, &af1);
                    Term fwd = thvm_op_raw(ctx, UOP_LOG, af0, term_era());
                    Term bwd = thvm_op_raw(ctx, UOP_DIV, a_bwd, af1);
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // Unary SQRT: d(√a) = da / (2·√a).  y = √a, so reuse fwd.
                if (uop == UOP_SQRT) {
                    Term a = heap_read(ctx, yloc + 0);
                    Term a_fwd, a_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    Term af0, af1;
                    thvm_dup(ctx, thvm_fresh_label(ctx), a_fwd, &af0, &af1);
                    Term fwd = thvm_op_raw(ctx, UOP_SQRT, af0, term_era());
                    Term sqrt_a = thvm_op_raw(ctx, UOP_SQRT, af1, term_era());
                    Term two = term_num_f32(2.0f);
                    Term denom = thvm_op_raw(ctx, UOP_MUL, two, sqrt_a);
                    Term bwd = thvm_op_raw(ctx, UOP_DIV, a_bwd, denom);
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // Unary RELU: d(relu a) = (a > 0) · da.  a used twice.
                if (uop == UOP_RELU) {
                    Term a = heap_read(ctx, yloc + 0);
                    Term a_fwd, a_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    Term af0, af1;
                    thvm_dup(ctx, thvm_fresh_label(ctx), a_fwd, &af0, &af1);
                    Term fwd = thvm_op_raw(ctx, UOP_RELU, af0, term_era());
                    Term zero = term_num_f32(0.0f);
                    Term mask = thvm_op_raw(ctx, UOP_CMP, af1, zero);
                    Term bwd = thvm_op_raw(ctx, UOP_MUL, a_bwd, mask);
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // CMP: non-differentiable → bwd = NUM(0). fa/fb erased.
                if (uop == UOP_CMP) {
                    Term a = heap_read(ctx, yloc + 0);
                    Term b = heap_read(ctx, yloc + 1);
                    Term a_fwd, a_bwd, b_fwd, b_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    thvm_grad_pair(ctx, target_tid, b, &b_fwd, &b_bwd);
                    Term fwd = thvm_op_raw(ctx, UOP_CMP, a_fwd, b_fwd);
                    thvm_spawn_detached_era(ctx, a_bwd);
                    thvm_spawn_detached_era(ctx, b_bwd);
                    Term bwd = term_num_f32(0.0f);
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // MAX: d(max(a,b)) = (a >= b)·da + (a < b)·db. Needs both
                // operand values to build the mask. a/b used twice each.
                if (uop == UOP_MAX) {
                    Term a = heap_read(ctx, yloc + 0);
                    Term b = heap_read(ctx, yloc + 1);
                    Term a_fwd, a_bwd, b_fwd, b_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    thvm_grad_pair(ctx, target_tid, b, &b_fwd, &b_bwd);
                    Term af0, af1, bf0, bf1;
                    thvm_dup(ctx, thvm_fresh_label(ctx), a_fwd, &af0, &af1);
                    thvm_dup(ctx, thvm_fresh_label(ctx), b_fwd, &bf0, &bf1);
                    Term fwd = thvm_op_raw(ctx, UOP_MAX, af0, bf0);
                    /* mask = (a >= b) represented via CMP(a, b)  */
                    Term mask_a = thvm_op_raw(ctx, UOP_CMP, af1, bf1);
                    Term one = term_num_f32(1.0f);
                    Term mask_b = thvm_op_raw(ctx, UOP_SUB, one, mask_a);
                    Term l = thvm_op_raw(ctx, UOP_MUL, a_bwd, mask_a);
                    Term r = thvm_op_raw(ctx, UOP_MUL, b_bwd, mask_b);
                    Term bwd = thvm_op_raw(ctx, UOP_ADD, l, r);
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // View ops: forward applies the view to fa, backward sends
                // da through the inverse view. For RESHAPE we always have
                // the operand's shape available; for EXPAND we sum over the
                // broadcast axes. SHRINK / PAD / PERMUTE topology-only for
                // now (need more metadata than the shape-tensor operand
                // reveals, but the inverse op exists — future pass).
                if (uop == UOP_RESHAPE || uop == UOP_EXPAND ||
                    uop == UOP_SHRINK  || uop == UOP_PAD    ||
                    uop == UOP_PERMUTE) {
                    Term a     = heap_read(ctx, yloc + 0);
                    Term shape = heap_read(ctx, yloc + 1);
                    Term a_fwd, a_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    Term fwd = thvm_op_raw(ctx, uop, a_fwd, shape);

                    // Resolve y's shape (output) and a's shape (input).
                    Shape a_shape = SHAPE(1);
                    Shape y_shape = SHAPE(1);
                    if (term_tag(a) == TAG_TEN) {
                        u32 aid = (u32)term_val(a);
                        if (aid < ctx->tensor_count)
                            a_shape = ctx->tensors[aid].view.shape;
                    } else if (term_tag(a) == TAG_TOP) {
                        const View *av = st_get(term_val(a));
                        if (av) a_shape = av->shape;
                    }
                    const View *yv = st_get(yloc);
                    if (yv) y_shape = yv->shape;

                    Term bwd;
                    if (uop == UOP_EXPAND) {
                        // dA = sum_to_shape(da, y_shape, a_shape): reduces
                        // the broadcasted axes. Needs both shapes; skip the
                        // reduction if shapes are unknown/equal.
                        if (y_shape.rank != 0 && a_shape.rank != 0)
                            bwd = sum_to_shape(ctx, a_bwd, y_shape, a_shape);
                        else
                            bwd = a_bwd;
                    } else if (uop == UOP_RESHAPE) {
                        // dA = reshape(da, a_shape).
                        if (a_shape.rank != 0)
                            bwd = thvm_reshape(ctx, a_bwd, a_shape);
                        else
                            bwd = a_bwd;
                    } else if (uop == UOP_PERMUTE) {
                        // dA = permute(da, inverse_perm).
                        // Second operand `shape` is a TEN holding the
                        // permutation. Read its data, invert.
                        if (term_tag(shape) == TAG_TEN && a_shape.rank > 0) {
                            u32 pid = (u32)term_val(shape);
                            u32 nd = a_shape.rank;
                            u32 pf[MAX_DIM];
                            tensor_meta_read_u32(ctx, pid, pf, MAX_DIM);
                            u32 inv[MAX_DIM];
                            for (u32 j = 0; j < nd; j++) inv[pf[j]] = j;
                            bwd = thvm_permute(ctx, a_bwd, inv, nd);
                        } else {
                            bwd = a_bwd;
                        }
                    } else if (uop == UOP_SHRINK) {
                        // dA = pad(da, complementary_pairs).
                        // SHRINK's `shape` TEN holds [start_0, end_0,
                        // start_1, end_1, ...]. Pad pairs are
                        // (start_i, a_shape.dims[i] - end_i).
                        if (term_tag(shape) == TAG_TEN && a_shape.rank > 0) {
                            u32 sid = (u32)term_val(shape);
                            u32 nd = a_shape.rank;
                            u32 sf[MAX_DIM * 2];
                            tensor_meta_read_u32(ctx, sid, sf, MAX_DIM * 2);
                            u32 pp[MAX_DIM * 2];
                            for (u32 j = 0; j < nd; j++) {
                                pp[j*2]   = sf[j*2];
                                pp[j*2+1] = a_shape.dims[j] - sf[j*2+1];
                            }
                            bwd = thvm_pad(ctx, a_bwd, pp, nd);
                        } else {
                            bwd = a_bwd;
                        }
                    } else if (uop == UOP_PAD) {
                        // dA = shrink(da, complementary_pairs).
                        // PAD's `shape` TEN holds pad pairs; shrink start
                        // is pad_start, shrink end is pad_start + a_size.
                        if (term_tag(shape) == TAG_TEN && a_shape.rank > 0) {
                            u32 pid = (u32)term_val(shape);
                            u32 nd = a_shape.rank;
                            u32 pf[MAX_DIM * 2];
                            tensor_meta_read_u32(ctx, pid, pf, MAX_DIM * 2);
                            u32 sp[MAX_DIM * 2];
                            for (u32 j = 0; j < nd; j++) {
                                sp[j*2]   = pf[j*2];
                                sp[j*2+1] = pf[j*2] + a_shape.dims[j];
                            }
                            bwd = thvm_shrink(ctx, a_bwd, sp, nd);
                        } else {
                            bwd = a_bwd;
                        }
                    } else {
                        bwd = a_bwd;
                    }
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // RMAX (reduce-max): fwd = rmax(a_fwd, axes).
                // Backward emits gy * (a == expanded(y)) — only the argmax
                // positions get the gradient.
                if (uop == UOP_RMAX) {
                    Term a    = heap_read(ctx, yloc + 0);
                    Term axes = heap_read(ctx, yloc + 1);
                    Term a_fwd, a_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    Term af0, af1;
                    thvm_dup(ctx, thvm_fresh_label(ctx), a_fwd, &af0, &af1);
                    Term fwd = thvm_op_raw(ctx, UOP_RMAX, af0, axes);

                    Shape a_shape = SHAPE(1), y_shape = SHAPE(1);
                    if (term_tag(a) == TAG_TEN) {
                        u32 aid = (u32)term_val(a);
                        if (aid < ctx->tensor_count)
                            a_shape = ctx->tensors[aid].view.shape;
                    } else if (term_tag(a) == TAG_TOP) {
                        const View *av = st_get(term_val(a));
                        if (av) a_shape = av->shape;
                    }
                    const View *yv = st_get(yloc);
                    if (yv) y_shape = yv->shape;

                    Term bwd;
                    if (a_shape.rank != 0 && y_shape.rank != 0) {
                        /* Re-run rmax on the second forward copy for mask. */
                        Term y_local = thvm_op_raw(ctx, UOP_RMAX, af1, axes);
                        Term y_bc = thvm_expand(ctx, thvm_reshape(ctx, y_local, y_shape), a_shape);
                        /* Mask: a == expanded(y). Using CMP as placeholder
                         * for equality (CMP is <, but legacy uses this
                         * pattern; proper EQ would be ideal). */
                        Term a_for_mask;
                        {
                            Term af1a, af1b;
                            thvm_dup(ctx, thvm_fresh_label(ctx), af1, &af1a, &af1b);
                            (void)af1b;
                            a_for_mask = af1a;
                        }
                        Term mask = thvm_op_raw(ctx, UOP_CMP, y_bc, a_for_mask);
                        Term one = term_num_f32(1.0f);
                        Term mask1 = thvm_op_raw(ctx, UOP_SUB, one, mask);
                        Term gy_bc = thvm_expand(ctx, a_bwd, a_shape);
                        bwd = thvm_op_raw(ctx, UOP_MUL, gy_bc, mask1);
                    } else {
                        bwd = a_bwd;
                    }
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // ASSIGN(dst, src): y == dst (side effect updates dst).
                // The pair just threads the dst pair through; src branch
                // receives a NUM(0) backward (non-contributing to grad w.r.t.
                // dst) and src_fwd is eager-evaluated as part of the effect.
                if (uop == UOP_ASSIGN) {
                    Term dst = heap_read(ctx, yloc + 0);
                    Term src = heap_read(ctx, yloc + 1);
                    Term d_fwd, d_bwd, s_fwd, s_bwd;
                    thvm_grad_pair(ctx, target_tid, dst, &d_fwd, &d_bwd);
                    thvm_grad_pair(ctx, target_tid, src, &s_fwd, &s_bwd);
                    Term fwd = thvm_op_raw(ctx, UOP_ASSIGN, d_fwd, s_fwd);
                    thvm_spawn_detached_era(ctx, s_bwd);
                    Term bwd = d_bwd;
                    GRAD_STATE_RETURN(fwd, bwd);
                }

                // Reduction SUM: forward keeps axes; backward expands da
                // back to a's shape.
                //   fwd = SUM(a_fwd, axes)
                //   bwd = EXPAND(a_bwd, a_shape)      (after RESHAPE to
                //         match a's rank if axes collapsed dims)
                if (uop == UOP_SUM) {
                    Term a    = heap_read(ctx, yloc + 0);
                    Term axes = heap_read(ctx, yloc + 1);
                    Term a_fwd, a_bwd;
                    thvm_grad_pair(ctx, target_tid, a, &a_fwd, &a_bwd);
                    Term fwd = thvm_op_raw(ctx, UOP_SUM, a_fwd, axes);

                    Shape a_shape = SHAPE(1);
                    if (term_tag(a) == TAG_TEN) {
                        u32 aid = (u32)term_val(a);
                        if (aid < ctx->tensor_count)
                            a_shape = ctx->tensors[aid].view.shape;
                    } else if (term_tag(a) == TAG_TOP) {
                        const View *av = st_get(term_val(a));
                        if (av) a_shape = av->shape;
                    }
                    Term bwd;
                    if (a_shape.rank != 0)
                        bwd = thvm_expand(ctx, a_bwd, a_shape);
                    else
                        bwd = a_bwd;
                    GRAD_STATE_RETURN(fwd, bwd);
                }
            }

            // Not yet reducible — leave the aux as-is; trampoline will
            // retry once the body (y) reduces further.
            #undef GRAD_STATE_RETURN
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
            if (term_tag(sub) == TAG_ERA && term_val(sub) != 0) {
                // Each consumer gets its own local active ERA, but collapse
                // through any ERA indirection first so VAR substitution does
                // not manufacture ERA->ERA shells.
                ctx->itrs++;
                return thvm_make_active_era(ctx, sub);
            }
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

        // ── Sequencing ──────────────────────────────────────────────────

        // TAG_SEQ: strict sequencing — reduce left, discard, return right.
        // SEQ(a, b): forces a to WNF, then returns b. The value of a is discarded.
        // Used for dependency ordering: SEQ(ASSIGN(w,...), kernel_reading_w).
        case TAG_SEQ: {
            u64 loc = term_val(t);
            Term a = heap_read(ctx, loc); // WNF from trampoline
            if (getenv("THVM_SCHED_DIAG"))
                fprintf(stderr, "SEQ_INTERACT: a_tag=%u a_ext=%u loc=%llu\n",
                        term_tag(a), term_ext(a), (unsigned long long)loc);

            // SEQ-SUP: distribute through superposition
            if (term_tag(a) == TAG_SUP) {
                u32 lab = term_ext(a);
                u64 sloc = term_val(a);
                Term a0 = heap_read(ctx, sloc + 0);
                Term a1 = heap_read(ctx, sloc + 1);
                Term b = heap_read(ctx, loc + 1);
                u64 dup = heap_alloc(ctx, 1);
                heap_set(ctx, dup, b);
                ctx->itrs++;
                u64 s0loc = heap_alloc(ctx, 2);
                heap_set(ctx, s0loc, a0);
                heap_set(ctx, s0loc + 1, term_new(TAG_DP0, lab, dup));
                u64 s1loc = heap_alloc(ctx, 2);
                heap_set(ctx, s1loc, a1);
                heap_set(ctx, s1loc + 1, term_new(TAG_DP1, lab, dup));
                t = thvm_sup(ctx, lab,
                    term_new(TAG_SEQ, 0, s0loc),
                    term_new(TAG_SEQ, 0, s1loc));
                INET_RECURSE();
            }
            // SEQ-ERA: erased left → return right (effect was erased)
            if (term_tag(a) == TAG_ERA) {
                ctx->itrs++;
                t = heap_read(ctx, loc + 1);
                INET_RECURSE();
            }
            // SEQ-val: left reached WNF (TEN, NUM, CTR, etc.) → discard, return right
            if (term_tag(a) != TAG_TOP) {
                ctx->itrs++;
                t = heap_read(ctx, loc + 1);
                INET_RECURSE();
            }
            // SEQ-TOP: left is still TAG_TOP (WNF compute op) → blocked, return as-is
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
