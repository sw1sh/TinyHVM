// interact/_.c — interaction handler dispatch
// Split into sub-files for readability:
//   grad.c        — UOP_GRAD (backward chain rule)
//   tensor_ops.c  — ASSIGN, TODEVICE, WHERE, IFZ, LOG_PRINT, ew/reduce/view ops
//   combinators.c — APP, LAM, REF, SUP, DP0/DP1, OP2, VAR, etc.

// Read small metadata (axes, shapes, pad specs) without GPU flush.
#define META_READ(be, buf_id, out, bytes) \
    ((be)->buf_read_nosync ? \
     (be)->buf_read_nosync((buf_id), (out), (bytes)) : \
     (be)->buf_read((buf_id), (out), (bytes)))

// Forward declarations (defined in rewrite/_.c, included after interact)
static int is_view_op(u32 uop);
static int is_elementwise(u32 uop);
static void thvm_grad_slot_accum(TinyHVM *ctx, Term slot, Term grad);
static Term thvm_force_tensor_term(TinyHVM *ctx, Term t);
static Term thvm_eval_exec_fixed_point(TinyHVM *ctx, Term t);
static Term thvm_force_dispatch_kid(TinyHVM *ctx, u32 kid, u32 depth);
static int thvm_kernel_register(TinyHVM *ctx, Term kernel, u32 *out_kid);
static Term thvm_alo_force(TinyHVM *ctx, Term alo);
// Defined in grad/_.c (included after interact). Used by diag sites here.
static u32 thvm_probe_ten_tid(TinyHVM *ctx, Term t);
static void thvm_probe_print_ten(TinyHVM *ctx, Term t, const char *label);

static Term thvm_era_payload(TinyHVM *ctx, Term item) {
    while (term_tag(item) == TAG_ERA) {
        u64 el = term_val(item);
        if (el == 0 || el >= ctx->heap_pos) return term_era();
        item = heap_read(ctx, el);
    }
    return item;
}

static Term thvm_make_active_era(TinyHVM *ctx, Term item) {
    item = thvm_era_payload(ctx, item);
    if (term_tag(item) == TAG_ERA && term_val(item) == 0) return term_era();
    u64 el = heap_alloc(ctx, 1);
    heap_set(ctx, el, item);
    return term_era_at(el);
}

static int thvm_term_is_active_era_like(TinyHVM *ctx, Term item, Term *era_out) {
    if (term_tag(item) == TAG_ERA && term_val(item) != 0) {
        if (era_out) *era_out = item;
        return 1;
    }
    if (term_tag(item) == TAG_VAR) {
        u64 loc = term_val(item);
        if (loc < ctx->heap_pos) {
            Term sub = heap_read(ctx, loc);
            if (term_tag(sub) == TAG_ERA && term_val(sub) != 0) {
                if (era_out) *era_out = sub;
                return 1;
            }
        }
    }
    return 0;
}

static void thvm_spawn_detached_era(TinyHVM *ctx, Term item) {
    Term era = thvm_make_active_era(ctx, item);
    if (term_tag(era) == TAG_ERA && term_val(era) != 0) {
        u64 slot = heap_alloc(ctx, 1);
        heap_set(ctx, slot, era);
    }
}

static u32 thvm_uop_storage_arity(u32 ext) {
    if (ext == UOP_KERNEL) return 3;
    if (ext == UOP_EXEC) return 3;   // [NUM(kid), deps, NUM(flags)]
    if (ext == UOP_FUSE) return 1;
    if (ext == UOP_WHERE || ext == UOP_IFZ) return 3;
    if (ext == UOP_GRAD) return 2;
    if (ext == UOP_LOG_PRINT || ext == UOP_DETACH) return 1;
    if (!is_binary(ext) && is_elementwise(ext)) return 1;
    return 2;
}

static u32 thvm_uop_visible_arity(u32 ext) {
    if (ext == UOP_KERNEL) return 2;
    return thvm_uop_storage_arity(ext);
}

static void thvm_copy_shape_state_loc(TinyHVM *ctx, u64 src_loc, u64 dst_loc) {
    if (!ctx || src_loc == 0 || dst_loc == 0) return;
    const ShapeTracker *tracker = st_get_tracker(src_loc);
    if (tracker) {
        st_set_tracker(dst_loc, tracker);
        return;
    }
    const View *view = st_get(src_loc);
    if (view) st_set(dst_loc, view);
}

static void thvm_fuse_copy_public_shape(TinyHVM *ctx, Term src, u64 dst_loc) {
    if (!ctx || term_tag(src) != TAG_TOP || dst_loc == 0) return;
    thvm_copy_shape_state_loc(ctx, term_val(src), dst_loc);
}

static u32 thvm_kernel_root_uop(TinyHVM *ctx, Term kernel);

static int thvm_kernel_is_monolithic(TinyHVM *ctx, Term kernel) {
    if (!ctx || term_tag(kernel) != TAG_TOP || term_ext(kernel) != UOP_KERNEL) return 0;
    u64 loc = term_val(kernel);
    if (loc == 0 || loc + 2 >= ctx->heap_pos) return 0;
    return term_tag(heap_read(ctx, loc + 1)) == TAG_ANY;
}

static Term thvm_kernel_monolithic_payload(TinyHVM *ctx, Term kernel) {
    if (!thvm_kernel_is_monolithic(ctx, kernel)) return term_era();
    u64 loc = term_val(kernel);
    return heap_read(ctx, loc + 0);
}

static Term thvm_make_public_kernel(TinyHVM *ctx, Term payload) {
    if (!ctx || term_tag(payload) != TAG_TOP) return payload;
    u32 uop = term_ext(payload);
    u64 kloc = heap_alloc(ctx, 3);
    heap_set(ctx, kloc + 0, payload);
    heap_set(ctx, kloc + 1, thvm_any());  // monolithic public region marker
    heap_set(ctx, kloc + 2, term_num_u32(uop));
    thvm_fuse_copy_public_shape(ctx, payload, kloc);
    return term_new(TAG_TOP, UOP_KERNEL, kloc);
}

static Term thvm_make_growing_kernel_from_uop(TinyHVM *ctx, u32 uop, Term left, Term right, Term shape_src) {
    if (!ctx || uop >= UOP_COUNT) return 0;
    u64 kloc = heap_alloc(ctx, 3);
    heap_set(ctx, kloc + 0, left);
    heap_set(ctx, kloc + 1, right);
    heap_set(ctx, kloc + 2, term_num_u32(uop));
    if (term_tag(shape_src) == TAG_TOP)
        thvm_fuse_copy_public_shape(ctx, shape_src, kloc);
    return term_new(TAG_TOP, UOP_KERNEL, kloc);
}

static int thvm_kernel_compute_uop(u32 uop) {
    return is_binary(uop) || is_elementwise(uop) ||
           uop == UOP_SUM || uop == UOP_RMAX || is_view_op(uop);
}

static int thvm_fuse_child_is_compute_like(Term child) {
    if (term_tag(child) != TAG_TOP) return 0;
    return thvm_kernel_compute_uop(term_ext(child));
}

static void fuse_wrap_memo_reset(void) { (void)0; }

static Term thvm_fuse_wrap_child(TinyHVM *ctx, Term child) {
    if (!ctx || !thvm_fuse_child_is_compute_like(child)) return child;
    u64 floc = heap_alloc(ctx, 1);
    heap_set(ctx, floc, child);
    return term_new(TAG_TOP, UOP_FUSE, floc);
}

static int thvm_kernel_local_child_ready(TinyHVM *ctx, Term child) {
    u8 tag = term_tag(child);
    if (tag == TAG_DP0 || tag == TAG_DP1) return 0;
    if (tag == TAG_TOP) {
        u32 ext = term_ext(child);
        if (ext == UOP_FUSE) return 0;
        if (ext == UOP_KERNEL) return thvm_kernel_is_monolithic(ctx, child);
        // Raw TOP children are not locally ready. They still need phase-1
        // reduction/fusion before a parent public KERNEL can treat them as a
        // stable leaf. Otherwise unresolved view/admin structure can leak into
        // the coarse graph and fail later kernel lowering.
        return 0;
    }
    return tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM ||
           tag == TAG_SEQ || tag == TAG_CTR || tag == TAG_LAM ||
           tag == TAG_SUP || tag == TAG_ANY;
}

static int thvm_public_kernel_absorb_child(TinyHVM *ctx, Term child, Term *out, u32 depth) {
    if (!out || depth > 64) return 0;
    if (term_tag(child) == TAG_TOP && term_ext(child) == UOP_KERNEL) {
        if (thvm_kernel_is_monolithic(ctx, child)) {
            *out = thvm_kernel_monolithic_payload(ctx, child);
            return 1;
        }
        return 0;
    }
    *out = child;
    return 1;
}

static Term thvm_make_public_kernel_from_uop(TinyHVM *ctx, u32 uop, Term left, Term right, Term shape_src) {
    if (!ctx || uop >= UOP_COUNT) return 0;
    Term raw_left = left;
    Term raw_right = right;
    if (!thvm_public_kernel_absorb_child(ctx, left, &raw_left, 0)) return 0;
    if (!thvm_public_kernel_absorb_child(ctx, right, &raw_right, 0)) return 0;
    int unary = (!is_binary(uop) && is_elementwise(uop) && term_tag(right) == TAG_ERA);
    u64 ploc = heap_alloc(ctx, unary ? 1 : 2);
    heap_set(ctx, ploc + 0, raw_left);
    if (!unary) heap_set(ctx, ploc + 1, raw_right);
    if (term_tag(shape_src) == TAG_TOP)
        thvm_fuse_copy_public_shape(ctx, shape_src, ploc);
    return thvm_make_public_kernel(ctx, term_new(TAG_TOP, uop, ploc));
}

static Term thvm_make_visible_kernel_from_uop(TinyHVM *ctx, u32 uop, Term left, Term right, Term shape_src) {
    // Emit a monolithic kernel only when both immediate children are already
    // locally ready; otherwise keep the visible growing shell so inner FUSE
    // work can still show up as its own step.
    if (thvm_kernel_local_child_ready(ctx, left) &&
        thvm_kernel_local_child_ready(ctx, right)) {
        Term public_term = thvm_make_public_kernel_from_uop(ctx, uop, left, right, shape_src);
        if (public_term) return public_term;
    }
    return thvm_make_growing_kernel_from_uop(ctx, uop, left, right, shape_src);
}

static Term thvm_fuse_public_term(TinyHVM *ctx, Term t, u32 depth) {
    if (!ctx || depth > 64) return t;
    if (term_tag(t) != TAG_TOP) return t;

    u32 uop = term_ext(t);
    if (uop == UOP_KERNEL || uop == UOP_FUSE) return t;

    u64 loc = term_val(t);
    if (loc == 0 || loc >= ctx->heap_pos) return t;

    if (is_binary(uop)) {
        if (loc + 1 >= ctx->heap_pos) return t;
        Term left = thvm_fuse_wrap_child(ctx, heap_read(ctx, loc + 0));
        Term right = thvm_fuse_wrap_child(ctx, heap_read(ctx, loc + 1));
        return thvm_make_visible_kernel_from_uop(ctx, uop, left, right, t);
    }

    if (is_elementwise(uop)) {
        Term left = thvm_fuse_wrap_child(ctx, heap_read(ctx, loc + 0));
        return thvm_make_visible_kernel_from_uop(ctx, uop, left, term_era(), t);
    }

    if (uop == UOP_SUM || uop == UOP_RMAX || is_view_op(uop)) {
        if (loc + 1 >= ctx->heap_pos) return t;
        Term left = thvm_fuse_wrap_child(ctx, heap_read(ctx, loc + 0));
        Term right = heap_read(ctx, loc + 1);
        return thvm_make_visible_kernel_from_uop(ctx, uop, left, right, t);
    }

    return t;
}

static u32 thvm_term_dtype_hint(TinyHVM *ctx, Term t) {
    switch (term_tag(t)) {
        case TAG_TEN:
            return ctx->tensors[(u32)term_val(t)].dtype;
        case TAG_NUM:
            return term_ext(t) == NUM_U32 ? DTYPE_U32 : DTYPE_F32;
        case TAG_TOP: {
            u32 uop = term_ext(t);
            u64 loc = term_val(t);
            if (uop == UOP_KERNEL) {
                if (thvm_kernel_is_monolithic(ctx, t))
                    return thvm_term_dtype_hint(ctx, thvm_kernel_monolithic_payload(ctx, t));
                if (loc > 0 && loc + 2 < ctx->heap_pos)
                    return thvm_term_dtype_hint(ctx, heap_read(ctx, loc + 0));
                return DTYPE_F32;
            }
            if (uop == UOP_CAST && loc > 0 && loc + 1 < ctx->heap_pos) {
                Term meta = heap_read(ctx, loc + 1);
                if (term_tag(meta) == TAG_TEN) {
                    u32 tid = (u32)term_val(meta);
                    if (tid < ctx->tensor_count) {
                        u32 raw[MAX_DIM];
                        if (tensor_meta_read_u32(ctx, tid, raw, MAX_DIM) == 1 && raw[0] < DTYPE_COUNT)
                            return raw[0];
                    }
                }
            }
            if ((is_view_op(uop) || is_elementwise(uop) || uop == UOP_CAST ||
                 uop == UOP_SUM || uop == UOP_RMAX) &&
                loc > 0 && loc < ctx->heap_pos)
                return thvm_term_dtype_hint(ctx, heap_read(ctx, loc + 0));
            return DTYPE_F32;
        }
        default:
            return DTYPE_F32;
    }
}

static u32 thvm_kernel_root_uop(TinyHVM *ctx, Term kernel) {
    if (term_tag(kernel) != TAG_TOP || term_ext(kernel) != UOP_KERNEL) return UOP_COUNT;
    u64 loc = term_val(kernel);
    if (loc == 0 || loc + 2 >= ctx->heap_pos) return UOP_COUNT;
    Term op = heap_read(ctx, loc + 2);
    return term_tag(op) == TAG_NUM ? term_as_u32(op) : UOP_COUNT;
}

static int thvm_kernel_child_ready(Term t) {
    u8 tag = term_tag(t);
    if (tag == TAG_DP0 || tag == TAG_DP1) return 0;
    if (tag == TAG_TOP) {
        u32 ext = term_ext(t);
        if (ext == UOP_FUSE) return 0;
        // Child UOP_KERNEL must be dispatched (have a cached result) before
        // the parent can proceed — no recursive flattening.
        if (ext == UOP_KERNEL) return 0;
        return thvm_kernel_compute_uop(ext);
    }
    return tag == TAG_TEN || tag == TAG_ERA || tag == TAG_NUM ||
           tag == TAG_SEQ || tag == TAG_CTR || tag == TAG_LAM ||
           tag == TAG_SUP || tag == TAG_ANY;
}

// Resolve a child UOP_KERNEL to its cached result (TAG_TEN).
// Returns 0 if the child has not yet been dispatched (not ready).
static int thvm_kernel_child_resolve(TinyHVM *ctx, Term child, Term *out) {
    extern Term kid_results[];
    extern u32 sched_kernel_count;
    extern u64 sched_kernel_locs[];
    if (term_tag(child) == TAG_TOP && term_ext(child) == UOP_KERNEL) {
        u64 cloc = term_val(child);
        for (u32 kid = 0; kid < sched_kernel_count; kid++) {
            if (sched_kernel_locs[kid] == cloc && term_tag(kid_results[kid]) != TAG_ERA) {
                *out = kid_results[kid];
                return 1;
            }
        }
        if (ctx && ctx->dispatch_enabled) {
            u32 kid = 0;
            if (thvm_kernel_register(ctx, child, &kid)) {
                Term forced = thvm_force_dispatch_kid(ctx, kid, 0);
                if (term_tag(forced) != TAG_ERA || term_val(forced) != 0) {
                    *out = forced;
                    return 1;
                }
            }
        }
        return 0;  // child kernel not yet dispatched
    }
    *out = child;
    return 1;
}

static int thvm_exec_dep_force(TinyHVM *ctx, Term *dep_io, u32 depth) {
    if (!dep_io || depth > 64) return 0;
    Term dep = *dep_io;
    if (term_tag(dep) == TAG_CTR) {
        u64 loc = term_val(dep);
        if (loc == 0 || loc + 1 >= ctx->heap_pos) return 0;
        Term d0 = heap_read(ctx, loc + 0);
        Term d1 = heap_read(ctx, loc + 1);
        int r0 = thvm_exec_dep_force(ctx, &d0, depth + 1);
        int r1 = thvm_exec_dep_force(ctx, &d1, depth + 1);
        if (d0 != heap_read(ctx, loc + 0)) heap_set(ctx, loc + 0, d0);
        if (d1 != heap_read(ctx, loc + 1)) heap_set(ctx, loc + 1, d1);
        return r0 && r1;
    }
    if (term_tag(dep) == TAG_TOP) {
        if (term_ext(dep) == UOP_EXEC) {
            Term next = thvm_eval_exec_fixed_point(ctx, dep);
            if (next != dep) *dep_io = next;
            dep = *dep_io;
            return !(term_tag(dep) == TAG_TOP && term_ext(dep) == UOP_EXEC);
        }
        if (term_ext(dep) == UOP_KERNEL) {
            Term next = dep;
            if (thvm_kernel_child_resolve(ctx, dep, &next)) {
                *dep_io = next;
                dep = next;
            }
            return !(term_tag(dep) == TAG_TOP && term_ext(dep) == UOP_KERNEL);
        }
    }
    return thvm_kernel_child_ready(dep);
}

static int thvm_kernel_normalize_compute(TinyHVM *ctx, Term t, Term *out, u32 depth) {
    if (!ctx || !out || depth > 64) return 0;
    for (u32 iter = 0; iter < 32; iter++) {
        u8 tag = term_tag(t);
        if (tag == TAG_TEN || tag == TAG_NUM || tag == TAG_ERA || tag == TAG_ANY) {
            *out = t;
            return 1;
        }

        if (tag == TAG_TOP && term_ext(t) == UOP_KERNEL) {
            Term resolved = term_era();
            if (!thvm_kernel_child_resolve(ctx, t, &resolved)) return 0;
            t = resolved;
            continue;
        }

        if (tag == TAG_TOP && is_view_op(term_ext(t))) {
            u32 uop = term_ext(t);
            u64 loc = term_val(t);
            if (loc == 0 || loc + 1 >= ctx->heap_pos) return 0;

            Term a = term_era();
            if (!thvm_kernel_normalize_compute(ctx, heap_read(ctx, loc + 0), &a, depth + 1))
                return 0;

            Term b = heap_read(ctx, loc + 1);
            if (term_tag(b) == TAG_TOP || term_tag(b) == TAG_SEQ) {
                b = thvm_force_tensor_term(ctx, b);
            } else {
                b = thvm_reduce(ctx, b);
            }
            if (term_tag(b) != TAG_TEN) return 0;

            u64 nloc = heap_alloc(ctx, 2);
            heap_set(ctx, nloc + 0, a);
            heap_set(ctx, nloc + 1, b);
            *out = term_new(TAG_TOP, uop, nloc);
            {
                const ShapeTracker *ast = st_get_tracker(loc);
                if (ast) st_set_tracker(nloc, ast);
            }
            return 1;
        }

        if (tag == TAG_TOP &&
            (is_binary(term_ext(t)) || is_elementwise(term_ext(t)) ||
             term_ext(t) == UOP_SUM || term_ext(t) == UOP_RMAX)) {
            u32 uop = term_ext(t);
            u64 loc = term_val(t);
            if (loc == 0 || loc >= ctx->heap_pos) return 0;

            Term a = term_era();
            Term b = term_era();
            if (!thvm_kernel_normalize_compute(ctx, heap_read(ctx, loc + 0), &a, depth + 1))
                return 0;

            int unary = (!is_binary(uop) && is_elementwise(uop));
            if (!unary) {
                if (loc + 1 >= ctx->heap_pos) return 0;
                if (!thvm_kernel_normalize_compute(ctx, heap_read(ctx, loc + 1), &b, depth + 1))
                    return 0;
            }

            u64 nloc = heap_alloc(ctx, unary ? 1 : 2);
            heap_set(ctx, nloc + 0, a);
            if (!unary) heap_set(ctx, nloc + 1, b);
            *out = term_new(TAG_TOP, uop, nloc);
            {
                const ShapeTracker *ast = st_get_tracker(loc);
                if (ast) st_set_tracker(nloc, ast);
            }
            return 1;
        }

        Term next = thvm_reduce(ctx, t);
        if (next == t) {
            if (tag == TAG_TOP) {
                u32 uop = term_ext(t);
                int maybe_force = (uop == UOP_KERNEL || uop == UOP_EXEC ||
                                   uop == UOP_ASSIGN || uop == UOP_DETACH ||
                                   uop == UOP_FUSE ||
                                   uop == UOP_IFZ || uop == UOP_WHERE ||
                                   uop == UOP_GRAD || uop == UOP_GRAD2 ||
                                   uop == UOP_LOG_PRINT ||
                                   uop == UOP_TODEVICE);
                if (maybe_force) {
                    next = (uop == UOP_EXEC) ? thvm_eval_exec_fixed_point(ctx, t)
                                             : thvm_eval(ctx, t);
                    if (next != t) {
                        t = next;
                        continue;
                    }
                }
                if (uop != UOP_FUSE && uop != UOP_GRAD && uop != UOP_GRAD2 &&
                    uop != UOP_EXEC && uop != UOP_KERNEL &&
                    uop != UOP_ASSIGN && uop != UOP_DETACH &&
                    uop != UOP_IFZ && uop != UOP_WHERE &&
                    uop != UOP_LOG_PRINT && uop != UOP_TODEVICE &&
                    st_get(term_val(t)) != NULL) {
                    *out = t;
                    return 1;
                }
            }
            return 0;
        }
        t = next;
    }
    return 0;
}

static int thvm_kernel_to_compute(TinyHVM *ctx, Term t, Term *out_compute, u32 depth) {
    if (!out_compute || depth > 64) return 0;
    if (term_tag(t) == TAG_TOP && term_ext(t) == UOP_KERNEL) {
        if (thvm_kernel_is_monolithic(ctx, t)) {
            Term payload = thvm_kernel_monolithic_payload(ctx, t);
            return thvm_kernel_normalize_compute(ctx, payload, out_compute, depth + 1);
        }
        u64 loc = term_val(t);
        if (loc == 0 || loc + 2 >= ctx->heap_pos) return 0;
        Term left = heap_read(ctx, loc + 0);
        Term right = heap_read(ctx, loc + 1);
        u32 kop = thvm_kernel_root_uop(ctx, t);
        if (kop >= UOP_COUNT || kop == UOP_COUNT) return 0;
        // Child UOP_KERNELs are opaque boundaries — use their cached
        // result (TAG_TEN) instead of recursively flattening.
        Term raw_left = left;
        Term raw_right = right;
        if (term_tag(left) == TAG_TOP && term_ext(left) == UOP_KERNEL) {
            if (!thvm_kernel_child_resolve(ctx, left, &raw_left)) return 0;
        } else if (!thvm_kernel_child_ready(left)) {
            return 0;
        }
        if (term_tag(right) == TAG_TOP && term_ext(right) == UOP_KERNEL) {
            if (!thvm_kernel_child_resolve(ctx, right, &raw_right)) return 0;
        } else if (!thvm_kernel_child_ready(right)) {
            return 0;
        }
        int unary = (!is_binary(kop) && is_elementwise(kop) && term_tag(right) == TAG_ERA);
        u64 oploc = heap_alloc(ctx, unary ? 1 : 2);
        heap_set(ctx, oploc + 0, raw_left);
        if (!unary) heap_set(ctx, oploc + 1, raw_right);
        *out_compute = term_new(TAG_TOP, kop, oploc);
        {
            const View *sv = st_get(loc);
            if (sv) st_set(oploc, sv);
        }
        return 1;
    }
    if (!thvm_kernel_child_ready(t)) return 0;
    *out_compute = t;
    return 1;
}

static int thvm_kernel_lookup_kid(u64 loc, u32 *out_kid) {
    extern u32 sched_kernel_count;
    extern u64 sched_kernel_locs[];
    extern KernelEntry sched_kernels[];
    for (u32 kid = 0; kid < sched_kernel_count; kid++) {
        Term original = sched_kernels[kid].original_term;
        if (sched_kernel_locs[kid] == loc ||
            (term_tag(original) == TAG_TOP && term_val(original) == loc)) {
            if (out_kid) *out_kid = kid;
            return 1;
        }
    }
    return 0;
}

static int thvm_kernel_register(TinyHVM *ctx, Term kernel, u32 *out_kid) {
    extern KernelEntry sched_kernels[];
    extern u32 sched_kernel_count;
    extern Term kid_results[];
    extern u32 kid_n_inputs[];
    extern u64 sched_kernel_locs[];
    u64 loc = term_val(kernel);
    u32 existing = 0;
    if (thvm_kernel_lookup_kid(loc, &existing)) {
        if (out_kid) *out_kid = existing;
        return 1;
    }
    Term compute = 0;
    if (!thvm_kernel_to_compute(ctx, kernel, &compute, 0)) {
        if (getenv("THVM_LOOP_DIAG")) {
            fprintf(stderr,
                    "KERNEL_REGISTER normalize_fail kernel=(tag=%u ext=%u val=%llu)\n",
                    (u32)term_tag(kernel), (u32)term_ext(kernel),
                    (unsigned long long)term_val(kernel));
        }
        return 0;
    }
    if (sched_kernel_count >= SCHED_MAX_KERNELS) return 0;

    KernelEntry ke;
    memset(&ke, 0, sizeof(ke));
    fuse_set_schedule_boundaries(NULL, NULL, NULL, 0, term_val(compute));
    if (!fuse_build_kernel(ctx, compute, &ke)) {
        if (getenv("THVM_LOOP_DIAG")) {
            fprintf(stderr,
                    "KERNEL_REGISTER build_fail kernel=(tag=%u ext=%u val=%llu) compute=(tag=%u ext=%u val=%llu) fail_code=%u\n",
                    (u32)term_tag(kernel), (u32)term_ext(kernel),
                    (unsigned long long)term_val(kernel),
                    (u32)term_tag(compute), (u32)term_ext(compute),
                    (unsigned long long)term_val(compute),
                    ke.fail_code);
            if (term_tag(compute) == TAG_TOP) {
                u64 cloc = term_val(compute);
                if (cloc != 0 && cloc + 1 < ctx->heap_pos) {
                    Term a = heap_read(ctx, cloc + 0);
                    Term b = heap_read(ctx, cloc + 1);
                    fprintf(stderr,
                            "KERNEL_REGISTER build_fail_slots a=(tag=%u ext=%u val=%llu) b=(tag=%u ext=%u val=%llu)\n",
                            (u32)term_tag(a), (u32)term_ext(a), (unsigned long long)term_val(a),
                            (u32)term_tag(b), (u32)term_ext(b), (unsigned long long)term_val(b));
                }
            }
        }
        fuse_clear_schedule_boundaries();
        return 0;
    }
    fuse_clear_schedule_boundaries();

    u32 kid = sched_kernel_count++;
    ke.original_term = compute;
    {
        u32 out_tid = tensor_create_unbacked(ctx, ke.out_shape, thvm_term_dtype_hint(ctx, kernel));
        ctx->tensors[out_tid].creator_op = UOP_KERNEL;
        ctx->tensors[out_tid].fusing_loc = loc;
        ctx->tensors[out_tid].fusing_uop = thvm_kernel_root_uop(ctx, kernel);
        ke.raw_output_tid = out_tid;
        ke.output_tid = out_tid;
    }
    sched_kernels[kid] = ke;
    kid_results[kid] = term_era();
    kid_n_inputs[kid] = 0;
    sched_kernel_locs[kid] = loc;
    {
        View fv = view_create(ke.out_shape);
        st_set(loc, &fv);
    }
    if (out_kid) *out_kid = kid;
    return 1;
}

static u32 thvm_alo_state_push(TinyHVM *ctx, u32 parent, u8 kind, u8 bind_tag, u64 bind_book, u64 bind_dyn, u32 label_old, u32 label_new) {
    if (!ctx->alo_states) return 0;
    if (ctx->alo_state_count >= ctx->alo_state_cap) {
        u32 new_cap = ctx->alo_state_cap ? (ctx->alo_state_cap << 1) : (1u << 16);
        ctx->alo_states = (AloState *)realloc(ctx->alo_states, (size_t)new_cap * sizeof(AloState));
        memset(ctx->alo_states + ctx->alo_state_cap, 0, (size_t)(new_cap - ctx->alo_state_cap) * sizeof(AloState));
        ctx->alo_state_cap = new_cap;
    }
    u32 id = ctx->alo_state_count++;
    ctx->alo_states[id] = (AloState){
        .parent = parent,
        .bind_book = bind_book,
        .bind_dyn = bind_dyn,
        .label_old = label_old,
        .label_new = label_new,
        .kind = kind,
        .bind_tag = bind_tag
    };
    return id;
}

static int thvm_alo_lookup_bind(TinyHVM *ctx, u32 state_id, u64 bind_book, u64 *out_dyn) {
    for (u32 sid = state_id; sid != 0; sid = ctx->alo_states[sid].parent) {
        AloState *s = &ctx->alo_states[sid];
        if (s->kind == 1 && s->bind_book == bind_book) {
            if (out_dyn) *out_dyn = s->bind_dyn;
            return 1;
        }
    }
    return 0;
}

static int thvm_alo_lookup_alias(TinyHVM *ctx, u32 state_id, u64 book_loc, u64 *out_alo_loc) {
    for (u32 sid = state_id; sid != 0; sid = ctx->alo_states[sid].parent) {
        AloState *s = &ctx->alo_states[sid];
        if (s->kind == 3 && s->bind_book == book_loc) {
            if (out_alo_loc) *out_alo_loc = s->bind_dyn;
            return 1;
        }
    }
    return 0;
}

static int thvm_alo_lookup_node(TinyHVM *ctx, u32 state_id, u64 book_loc, u64 *out_dyn_loc) {
    for (u32 sid = state_id; sid != 0; sid = ctx->alo_states[sid].parent) {
        AloState *s = &ctx->alo_states[sid];
        if (s->kind == 4 && s->bind_book == book_loc) {
            if (out_dyn_loc) *out_dyn_loc = s->bind_dyn;
            return 1;
        }
    }
    return 0;
}

static u32 thvm_alo_get_or_add_label(TinyHVM *ctx, u32 state_id, u32 old_label, u32 *io_state_id) {
    for (u32 sid = state_id; sid != 0; sid = ctx->alo_states[sid].parent) {
        AloState *s = &ctx->alo_states[sid];
        if (s->kind == 2 && s->label_old == old_label) return s->label_new;
    }
    u32 fresh = thvm_fresh_label(ctx);
    u32 next_state = thvm_alo_state_push(ctx, state_id, 2, 0, 0, 0, old_label, fresh);
    if (io_state_id) *io_state_id = next_state;
    return fresh;
}

static Term thvm_alo_make(TinyHVM *ctx, Term book_term, u32 state_id) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc + 0, book_term);
    heap_set(ctx, loc + 1, term_num_u32(state_id));
    return term_new(TAG_ALO, 0, loc);
}

static Term thvm_alo_suspend_child(TinyHVM *ctx, Term child, u32 state_id) {
    u8 tag = term_tag(child);
    if (tag == TAG_NUM || tag == TAG_TEN || tag == TAG_ERA || tag == TAG_ANY || tag == TAG_REF)
        return child;
    if (tag == TAG_VAR) {
        u64 dyn_loc = 0;
        if (thvm_alo_lookup_bind(ctx, state_id, term_val(child), &dyn_loc))
            return term_new(TAG_VAR, term_ext(child), dyn_loc);
        return child;
    }
    // DP0/DP1 book tokens for the same DUP location L are DIFFERENT linear
    // consumer ports — they must NOT share an ALO cell. Each gets its own
    // ALO storing its specific tag; memoization of the shared dynamic DUP
    // cell happens inside alo_realize via the kind=1 bind lookup on L.
    u64 old_loc = term_val(child);
    if (old_loc != 0 && tag != TAG_DP0 && tag != TAG_DP1) {
        u64 alo_loc = 0;
        if (thvm_alo_lookup_alias(ctx, state_id, old_loc, &alo_loc))
            return term_new(TAG_ALO, 0, alo_loc);
    }
    Term alo = thvm_alo_make(ctx, child, state_id);
    if (old_loc != 0 && tag != TAG_DP0 && tag != TAG_DP1)
        (void)thvm_alo_state_push(ctx, state_id, 3, 0, old_loc, term_val(alo), 0, 0);
    return alo;
}

static int thvm_alo_term_is_shared_alias(Term t) {
    return term_tag(t) == TAG_ALO;
}

static int thvm_alo_try_share_child(TinyHVM *ctx, Term child, Term *out) {
    Term cur = child;
    for (u32 depth = 0; depth < 8; depth++) {
        if (term_tag(cur) == TAG_VAR) {
            u64 loc = term_val(cur);
            if (loc < ctx->heap_pos) {
                Term sub = heap_read(ctx, loc);
                if (!term_is_sub(sub)) {
                    cur = sub;
                    continue;
                }
            }
        }
        break;
    }
    if (!thvm_alo_term_is_shared_alias(cur)) return 0;
    if (out) *out = cur;
    return 1;
}

static u32 thvm_alo_top_arity(u32 ext) {
    return thvm_uop_storage_arity(ext);
}

static u32 thvm_alo_book_arity(Term t) {
    switch (term_tag(t)) {
        case TAG_APP:
        case TAG_LAM:
        case TAG_BRI:
        case TAG_SEQ:
        case TAG_SUP:
        case TAG_USP:
        case TAG_OP2:
        case TAG_EQL:
        case TAG_AND:
        case TAG_OR:
        case TAG_MAT:
        case TAG_ANN:
            return 2;
        case TAG_DSU:
        case TAG_DDU:
            return 3;
        case TAG_INC:
        case TAG_UDP:
            return 1;
        case TAG_CTR:
            return term_ext(t);
        case TAG_TOP:
            return thvm_alo_top_arity(term_ext(t));
        default:
            return 0;
    }
}

static Term thvm_alo_realize(TinyHVM *ctx, Term book_term, u32 state_id) {
    u8 tag = term_tag(book_term);
    if (tag == TAG_REF || tag == TAG_NUM || tag == TAG_TEN || tag == TAG_ERA || tag == TAG_ANY)
        return book_term;

    if (tag == TAG_VAR) {
        u64 old_loc = term_val(book_term);
        u64 dyn_loc = 0;
        if (thvm_alo_lookup_bind(ctx, state_id, old_loc, &dyn_loc))
            return term_new(TAG_VAR, term_ext(book_term), dyn_loc);
        return book_term;
    }

    if (tag == TAG_DP0 || tag == TAG_DP1) {
        u32 walk_state = state_id;
        u32 new_lab = thvm_alo_get_or_add_label(ctx, walk_state, term_ext(book_term), &walk_state);
        u64 old_loc = term_val(book_term);
        u64 dyn_loc = 0;
        if (!thvm_alo_lookup_bind(ctx, walk_state, old_loc, &dyn_loc)) {
            dyn_loc = heap_alloc(ctx, 1);
            Term child = (old_loc > 0 && old_loc < ctx->book_heap_pos) ? ctx->book_heap[old_loc] : term_era();
            Term suspended = thvm_alo_suspend_child(ctx, child, walk_state);
            Term shared = term_era();
            if (thvm_alo_try_share_child(ctx, suspended, &shared)) {
                Term forced = thvm_alo_force(ctx, shared);
                if (term_tag(forced) == TAG_TEN ||
                    term_tag(forced) == TAG_NUM ||
                    (term_tag(forced) == TAG_ERA && term_val(forced) == 0))
                    return forced;
            }
            heap_set(ctx, dyn_loc, suspended);
            walk_state = thvm_alo_state_push(ctx, walk_state, 1, term_tag(book_term), old_loc, dyn_loc, 0, 0);
        }
        return term_new(tag, new_lab, dyn_loc);
    }

    if (tag == TAG_SUP || tag == TAG_USP || tag == TAG_UDP) {
        u32 walk_state = state_id;
        u32 new_lab = thvm_alo_get_or_add_label(ctx, walk_state, term_ext(book_term), &walk_state);
        u32 ar = thvm_alo_book_arity(book_term);
        if (ar == 0) return term_new(tag, new_lab, term_val(book_term));
        u64 old_loc = term_val(book_term);
        u64 new_loc = heap_alloc(ctx, ar);
        for (u32 i = 0; i < ar; i++) {
            Term child = (old_loc > 0 && old_loc + i < ctx->book_heap_pos) ? ctx->book_heap[old_loc + i] : term_era();
            heap_set(ctx, new_loc + i, thvm_alo_suspend_child(ctx, child, walk_state));
        }
        return term_new(tag, new_lab, new_loc);
    }

    if (tag == TAG_LAM || tag == TAG_BRI) {
        u64 old_loc = term_val(book_term);
        u64 new_loc = heap_alloc(ctx, 2);
        Term var = term_new(TAG_VAR, 0, new_loc);
        heap_set(ctx, new_loc + 0, term_set_sub(var));
        thvm_copy_shape_state_loc(ctx, thvm_st_book_loc_key(old_loc), new_loc);
        u32 body_state = thvm_alo_state_push(ctx, state_id, 1, tag, old_loc, new_loc, 0, 0);
        Term body = (old_loc > 0 && old_loc + 1 < ctx->book_heap_pos) ? ctx->book_heap[old_loc + 1] : term_era();
        heap_set(ctx, new_loc + 1, thvm_alo_suspend_child(ctx, body, body_state));
        return term_new(tag, term_ext(book_term), new_loc);
    }

    u32 ar = thvm_alo_book_arity(book_term);
    if (ar == 0) return book_term;
    u64 old_loc = term_val(book_term);
    u64 new_loc = heap_alloc(ctx, ar);
    thvm_copy_shape_state_loc(ctx, thvm_st_book_loc_key(old_loc), new_loc);
    u32 node_state = thvm_alo_state_push(ctx, state_id, 4, 0, old_loc, new_loc, 0, 0);
    for (u32 i = 0; i < ar; i++) {
        Term child = (old_loc > 0 && old_loc + i < ctx->book_heap_pos) ? ctx->book_heap[old_loc + i] : term_era();
        heap_set(ctx, new_loc + i, thvm_alo_suspend_child(ctx, child, node_state));
    }
    if (tag == TAG_APP && ar == 2) {
        Term fun_book = (old_loc > 0 && old_loc < ctx->book_heap_pos) ? ctx->book_heap[old_loc + 0] : term_era();
        Term fun_dyn  = heap_read(ctx, new_loc + 0);
        if (term_tag(fun_book) == TAG_TOP && term_ext(fun_book) == UOP_GRAD &&
            term_tag(fun_dyn)  == TAG_TOP && term_ext(fun_dyn)  == UOP_GRAD) {
            u64 grad_loc = term_val(fun_dyn);
            thvm_grad_keep_app_loc_set(ctx, grad_loc, new_loc);
            thvm_grad_keep_bundle_set(ctx, grad_loc, heap_read(ctx, new_loc + 1));
        }
    }
    if (tag == TAG_TOP && term_ext(book_term) == UOP_GRAD) {
        u64 book_grad_loc = thvm_grad_book_loc_key(old_loc);
        Term dyn_target = thvm_alo_suspend_child(ctx, thvm_grad_target_get(ctx, book_grad_loc), state_id);
        u32 dyn_mode = thvm_grad_mode_get(ctx, book_grad_loc);
        thvm_grad_target_set(ctx, new_loc, dyn_target);
        thvm_grad_mode_set(ctx, new_loc, dyn_mode);
        u32 nt = thvm_grad_targets_count_at(ctx, book_grad_loc);
        if (nt > 0) {
            Term params[THVM_GRAD_TARGETS_MAX];
            Term slots[THVM_GRAD_TARGETS_MAX];
            assert(nt <= THVM_GRAD_TARGETS_MAX);
            for (u32 i = 0; i < nt; i++) {
                params[i] = thvm_alo_suspend_child(ctx,
                                                   thvm_grad_targets_get_term_at(ctx, book_grad_loc, i),
                                                   state_id);
                slots[i] = thvm_alo_suspend_child(ctx,
                                                  thvm_grad_targets_get_slot_at(ctx, book_grad_loc, i),
                                                  state_id);
            }
            thvm_grad_targets_set_for_loc(ctx, new_loc, params, slots, nt);
        }
        Term bundle = thvm_grad_keep_bundle_get(ctx, book_grad_loc);
        if (!(term_tag(bundle) == TAG_ERA && term_val(bundle) == 0)) {
            thvm_grad_keep_bundle_set(ctx, new_loc, thvm_alo_suspend_child(ctx, bundle, state_id));
        }
        u64 book_app_loc = thvm_grad_keep_app_loc_get(ctx, book_grad_loc);
        if (book_app_loc != 0) {
            u64 dyn_or_book_app_loc = book_app_loc;
            if (thvm_grad_is_book_loc(book_app_loc)) {
                u64 dyn_app_loc = 0;
                if (thvm_alo_lookup_node(ctx, state_id, thvm_grad_unkey_book_loc(book_app_loc), &dyn_app_loc))
                    dyn_or_book_app_loc = dyn_app_loc;
            }
            thvm_grad_keep_app_loc_set(ctx, new_loc, dyn_or_book_app_loc);
        }
        if (getenv("THVM_LOOP_DIAG")) {
            fprintf(stderr,
                    "ALO_GRAD book_loc=%llu dyn_loc=%llu mode=%u targets=%u target_tag=%u target_ext=%u target_val=%llu bundle_tag=%u bundle_ext=%u bundle_val=%llu\n",
                    (unsigned long long)old_loc,
                    (unsigned long long)new_loc,
                    dyn_mode,
                    nt,
                    (u32)term_tag(dyn_target),
                    (u32)term_ext(dyn_target),
                    (unsigned long long)term_val(dyn_target),
                    (u32)term_tag(bundle),
                    (u32)term_ext(bundle),
                    (unsigned long long)term_val(bundle));
            for (u32 i = 0; i < nt; i++) {
                Term pt = thvm_grad_targets_get_term_at(ctx, new_loc, i);
                Term ps = thvm_grad_targets_get_slot_at(ctx, new_loc, i);
                fprintf(stderr,
                        "  ALO_GRAD_TARGET[%u]=term(tag=%u ext=%u val=%llu) slot(tag=%u ext=%u val=%llu)\n",
                        i,
                        (u32)term_tag(pt), (u32)term_ext(pt), (unsigned long long)term_val(pt),
                        (u32)term_tag(ps), (u32)term_ext(ps), (unsigned long long)term_val(ps));
            }
        }
    }
    return term_new(tag, term_ext(book_term), new_loc);
}

static Term thvm_alo_force(TinyHVM *ctx, Term alo) {
    u64 alo_loc = term_val(alo);
    if (alo_loc == 0 || alo_loc + 1 >= ctx->heap_pos) return alo;
    Term book_term = heap_read(ctx, alo_loc + 0);
    Term sid_term = heap_read(ctx, alo_loc + 1);
    if (term_tag(sid_term) != TAG_NUM) {
        // ALO already realized (memoised). If the memoised value is a
        // linear DP0/DP1 port, a second consumer cannot reuse it — a
        // shared DP token would violate IC linearity (port_slot only
        // tracks one consumer per DUP port, so stale references cascade
        // ERA). Fresh-DUP-wrap: split the memoised DP into two new
        // linear ports, hand out one, push the other back as the new
        // memo so a third force can split again.
        u8 bt = term_tag(book_term);
        if (bt == TAG_DP0 || bt == TAG_DP1) {
            u64 new_dup = heap_alloc(ctx, 1);
            heap_set(ctx, new_dup, book_term);
            u32 new_lab = thvm_fresh_label(ctx);
            Term dp0 = term_new(TAG_DP0, new_lab, new_dup);
            Term dp1 = term_new(TAG_DP1, new_lab, new_dup);
            heap_set(ctx, alo_loc + 0, dp1);
            return dp0;
        }
        return book_term;
    }
    u32 state_id = term_as_u32(sid_term);
    Term out = thvm_alo_realize(ctx, book_term, state_id);
    if (getenv("THVM_LOOP_DIAG")) {
        fprintf(stderr,
                "ALO_FORCE loc=%llu state=%u book=%u/%u@%llu out=%u/%u@%llu",
                (unsigned long long)alo_loc, state_id,
                (u32)term_tag(book_term), (u32)term_ext(book_term),
                (unsigned long long)term_val(book_term),
                (u32)term_tag(out), (u32)term_ext(out),
                (unsigned long long)term_val(out));
        for (u32 sid = state_id, depth = 0; sid != 0 && depth < 8; sid = ctx->alo_states[sid].parent, depth++) {
            AloState *s = &ctx->alo_states[sid];
            fprintf(stderr,
                    " | S%u kind=%u tag=%u book=%llu dyn=%llu lab=%u->%u",
                    sid, (u32)s->kind, (u32)s->bind_tag,
                    (unsigned long long)s->bind_book,
                    (unsigned long long)s->bind_dyn,
                    (u32)s->label_old, (u32)s->label_new);
        }
        fputc('\n', stderr);
    }
    // #region agent log
    do {
        static u32 alo_force_dbg_count = 0;
        if (alo_force_dbg_count >= 12) break;
        alo_force_dbg_count++;
        u32 out_ar = thvm_alo_book_arity(out);
        u32 child0_tag = 255;
        u32 child1_tag = 255;
        u64 child0_val = 0;
        u64 child1_val = 0;
        u64 out_loc = term_val(out);
        if (out_ar > 0 && out_loc > 0 && out_loc < ctx->heap_pos) {
            Term child0 = heap_read(ctx, out_loc + 0);
            child0_tag = term_tag(child0);
            child0_val = term_val(child0);
            if (out_ar > 1 && out_loc + 1 < ctx->heap_pos) {
                Term child1 = heap_read(ctx, out_loc + 1);
                child1_tag = term_tag(child1);
                child1_val = term_val(child1);
            }
        }
        char _dbg[384];
        snprintf(_dbg, sizeof(_dbg),
                 "{\"alo_tag\":%u,\"alo_loc\":%llu,\"book_tag\":%u,\"book_ext\":%u,"
                 "\"book_val\":%llu,\"state_id\":%u,\"out_tag\":%u,\"out_ext\":%u,"
                 "\"out_val\":%llu,\"out_arity\":%u,\"out_child0_tag\":%u,"
                 "\"out_child0_val\":%llu,\"out_child1_tag\":%u,\"out_child1_val\":%llu}",
                 (u32)term_tag(alo), (unsigned long long)alo_loc,
                 (u32)term_tag(book_term), (u32)term_ext(book_term),
                 (unsigned long long)term_val(book_term), state_id,
                 (u32)term_tag(out), (u32)term_ext(out),
                 (unsigned long long)term_val(out), out_ar,
                 child0_tag, (unsigned long long)child0_val,
                 child1_tag, (unsigned long long)child1_val);
        thvm_agent_debug_log("pre-fix", "H4", "src/interact/_.c:220",
                             "alo_force_result", _dbg);
    } while (0);
    // #endregion
    thvm_step_alo_subst_record(ctx, alo_loc, out, book_term);
    heap_set(ctx, alo_loc + 0, out);
    heap_set(ctx, alo_loc + 1, term_era());
    return out;
}

static Term thvm_interact(TinyHVM *ctx, Term t) {
    // Pre-interaction step-graph hook: capture BEFORE metadata from live
    // heap while we still can (the interaction will mutate it). No-op
    // unless THVM_STEP_GRAPH is set.
    thvm_step_graph_on_pre_interaction(ctx, t);

    // Return result directly — the trampoline handles TAG_TOP results
    // via `next = r; goto enter;` (no need to force-reduce here).
    // Return directly — trampoline handles TAG_TOP via goto enter
    #define RETURN_REDUCED(result) do { return (result); } while(0)
    // One interact() call performs exactly one local rewrite.
    // Continued reduction is the reducer trampoline's job.
    #define INET_RECURSE() do { return t; } while (0)
    // No GRAD_STEP — all GRAD sub-terms are placed on heap iteratively

    u32 tag;
inet_step:
    tag = term_tag(t);

    switch (tag) {
        case TAG_TOP: {
            u32 uop = term_ext(t);
            u64 loc = term_val(t);

            // === UOP_GRAD2 (new UOP-shape gradient; TEN-leaf only for now)
            if (uop == UOP_GRAD2) {
                Term y   = heap_read(ctx, loc + 0);
                Term tgt = heap_read(ctx, loc + 1);
                if (term_is_sub(y))   y   = term_strip_sub(y);
                if (term_is_sub(tgt)) tgt = term_strip_sub(tgt);
                if (term_tag(y) == TAG_TEN && term_tag(tgt) == TAG_TEN) {
                    u32 ytid = (u32)term_val(y);
                    u32 ttid = (u32)term_val(tgt);
                    Shape tsh = (ttid < ctx->tensor_count)
                        ? ctx->tensors[ttid].view.shape : SHAPE(1);
                    Term scalar = (ytid == ttid) ? term_num_f32(1.0f)
                                                 : term_num_f32(0.0f);
                    Term out = (tsh.rank > 0 && !(tsh.rank == 1 && tsh.dims[0] == 1))
                        ? thvm_expand(ctx, scalar, tsh) : scalar;
                    ctx->itrs++;
                    RETURN_REDUCED(out);
                }
                // y not yet WNF or not TEN-leaf: leave alone; trampoline will
                // retry once y reduces further.  TOP dispatch for UOP_GRAD2
                // lands in follow-up ticks.
            }

            // === UOP_GRAD ===
#include "grad.c"

            // === Tensor ops (ASSIGN, ew, reduce, view, etc.) ===
#include "tensor_ops.c"
        } // end case TAG_TOP

        // === Combinators (APP, LAM, SUP, DP0/DP1, etc.) ===
#include "combinators.c"

        default:
            #undef INET_RECURSE
            return t;
    }
    #undef INET_RECURSE
    return t;
}

// ============================================================
// Trampoline frame tags (used by reduce/_.c)
// ============================================================
#define TAG_TOP1  0x7E  // TAG_TOP arg0 done, entering arg1
#define TAG_TOP2  0x7F  // TAG_TOP arg1 done, entering arg2 (GRAD/WHERE/IFZ)
#define TAG_OP2_1 0x7D  // OP2 arg0 done (non-SUP), entering arg1. EXT=opr, VAL=loc
#define TAG_EQL_1 0x7C  // EQL arg0 done (non-SUP), entering arg1. VAL=loc
#define TAG_MAT_1 0x7B  // APP fun=MAT, entering arg. VAL=app_loc

#define FRAME_CAP 65536
