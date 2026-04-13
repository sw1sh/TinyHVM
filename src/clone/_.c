// clone/_.c — Deep-clone a term tree (ALO: allocate fresh copies)
//
// Used by TAG_REF to produce a fresh copy of a definition body on each
// unfold.  Interaction nets are linear: each heap cell is consumed once,
// so recursive definitions need a fresh copy per invocation.
//
// Approach: eager recursive walk.  Each compound node (APP, LAM, SUP,
// TOP, DP0/DP1) gets a freshly-allocated heap copy with children
// cloned recursively.  Terminals (NUM, TEN, ERA, REF) are returned
// as-is.  LAM/VAR pairs are rebound via a small relocation table.

#define CLONE_MAX_RELOC 512
#define CLONE_MAX_LABELS 64
#define CLONE_MAX_TERMMAP 512

typedef struct { u64 old_loc; u64 new_loc; } Reloc;
typedef struct { u32 old_label; u32 new_label; } LabelMap;
typedef struct { Term old_term; Term new_term; } TermMap;

static u32 clone_fresh_label(TinyHVM *ctx, u32 old_label, LabelMap *lmap, u32 *n_labels) {
    for (u32 i = 0; i < *n_labels; i++)
        if (lmap[i].old_label == old_label)
            return lmap[i].new_label;
    assert(*n_labels < CLONE_MAX_LABELS);
    u32 fresh = thvm_fresh_label(ctx);
    lmap[*n_labels] = (LabelMap){ old_label, fresh };
    (*n_labels)++;
    return fresh;
}

static void clone_term_map_add(TermMap *tmap, u32 *n_tmap, Term old_term, Term new_term) {
    assert(*n_tmap < CLONE_MAX_TERMMAP);
    tmap[*n_tmap] = (TermMap){ old_term, new_term };
    (*n_tmap)++;
}

static Term clone_term_map_get(TermMap *tmap, u32 n_tmap, Term old_term) {
    for (u32 i = 0; i < n_tmap; i++) {
        if (tmap[i].old_term == old_term) return tmap[i].new_term;
    }
    return old_term;
}

static Term term_clone_r(TinyHVM *ctx, Term t, Reloc *relocs, u32 *n_relocs,
                          LabelMap *lmap, u32 *n_labels,
                          TermMap *tmap, u32 *n_tmap) {
    u32 tag = term_tag(t);
    Term mapped = clone_term_map_get(tmap, *n_tmap, t);
    if (mapped != t) return mapped;

    switch (tag) {
        // ── Terminals: no heap state ──────────────────────────────
        case TAG_NUM:
        case TAG_ERA:
        case TAG_REF:
            return t;
        case TAG_TEN:
            tensor_incref(ctx, (u32)term_val(t));
            return t;

        // ── VAR: look up relocation table for rebound location ───
        case TAG_VAR: {
            u64 old_loc = term_val(t);
            for (u32 i = 0; i < *n_relocs; i++) {
                if (relocs[i].old_loc == old_loc)
                    return term_new(TAG_VAR, term_ext(t), relocs[i].new_loc);
            }
            if (old_loc < ctx->heap_pos) {
                Term sub = heap_read(ctx, old_loc);
                // If this variable is already substituted, clone the resolved
                // value instead of retaining a live pointer into mutable net state.
                if (!term_is_sub(sub)) {
                    return term_clone_r(ctx, sub, relocs, n_relocs, lmap, n_labels, tmap, n_tmap);
                }
            }
            return t;  // external variable (not from a cloned LAM)
        }

        // ── LAM: allocate fresh var slot + clone body ────────────
        case TAG_LAM: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            // Register relocation: old var slot → new var slot
            assert(*n_relocs < CLONE_MAX_RELOC);
            relocs[*n_relocs] = (Reloc){ old_loc, new_loc };
            (*n_relocs)++;
            // Fresh var slot (unbound / substitution marker)
            Term var = term_new(TAG_VAR, 0, new_loc);
            heap_set(ctx, new_loc, term_set_sub(var));
            const ShapeTracker *vst = st_get_tracker(old_loc);
            if (vst) st_set_tracker(new_loc, vst);
            // Clone body
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            Term out = term_new(TAG_LAM, term_ext(t), new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── BRI: bridge (same layout as LAM — 2 slots with var reloc)
        case TAG_BRI: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            assert(*n_relocs < CLONE_MAX_RELOC);
            relocs[*n_relocs] = (Reloc){ old_loc, new_loc };
            (*n_relocs)++;
            Term var = term_new(TAG_VAR, 0, new_loc);
            heap_set(ctx, new_loc, term_set_sub(var));
            const ShapeTracker *vst = st_get_tracker(old_loc);
            if (vst) st_set_tracker(new_loc, vst);
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            Term out = term_new(TAG_BRI, term_ext(t), new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── APP / SEQ: 2 slots ──────────────────────────────────
        case TAG_SEQ:
        case TAG_APP: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            Term out = term_new(tag, term_ext(t), new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── SUP: 2 slots (freshen label) ────────────────────────
        case TAG_SUP: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            Term out = term_new(TAG_SUP, clone_fresh_label(ctx, term_ext(t), lmap, n_labels), new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── DP0/DP1: 1 slot (freshen label) ─────────────────────
        // If backing points to a lambda's var slot (in relocation table),
        // share the relocated slot directly so DUP reads the lambda's
        // substitution without VAR indirection.
        case TAG_DP0:
        case TAG_DP1: {
            u64 old_loc = term_val(t);
            u64 use_loc = 0;
            int mapped = 0;
            for (u32 i = 0; i < *n_relocs; i++) {
                if (relocs[i].old_loc == old_loc) {
                    use_loc = relocs[i].new_loc;
                    mapped = 1;
                    break;
                }
            }
            if (!mapped) {
                use_loc = heap_alloc(ctx, 1);
                assert(*n_relocs < CLONE_MAX_RELOC);
                relocs[*n_relocs] = (Reloc){ old_loc, use_loc };
                (*n_relocs)++;
                heap_set(ctx, use_loc,
                         term_clone_r(ctx, heap_read(ctx, old_loc),
                                      relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            }
            Term out = term_new(tag, clone_fresh_label(ctx, term_ext(t), lmap, n_labels), use_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── ANN: 2 slots (term, type) ─────────────────────────────
        case TAG_ANN: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            Term out = term_new(TAG_ANN, term_ext(t), new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── DSU/DDU: 3 slots ─────────────────────────────────────
        case TAG_DSU:
        case TAG_DDU: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 3);
            for (u32 i = 0; i < 3; i++)
                heap_set(ctx, new_loc + i,
                         term_clone_r(ctx, heap_read(ctx, old_loc + i),
                                      relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            Term out = term_new(tag, term_ext(t), new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── INC: 1 slot ──────────────────────────────────────────
        case TAG_INC: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 1);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            Term out = term_new(TAG_INC, term_ext(t), new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── OP2: 2 slots (x, y) ─────────────────────────────────
        case TAG_OP2: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            Term out = term_new(TAG_OP2, term_ext(t), new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── TOP: tensor op, 2 or 3 slots ────────────────────────
        case TAG_TOP: {
            u32 uop = term_ext(t);
            u64 old_loc = term_val(t);
            u32 arity = (uop == UOP_WHERE || uop == UOP_IFZ) ? 3 : 2;
            u64 new_loc = heap_alloc(ctx, arity);
            for (u32 i = 0; i < arity; i++)
                heap_set(ctx, new_loc + i,
                         term_clone_r(ctx, heap_read(ctx, old_loc + i),
                                      relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            const View *vv = st_get(old_loc);
            if (vv) st_set(new_loc, vv);
            if (uop == UOP_GRAD) {
                Term gx = term_clone_r(ctx, thvm_grad_target_get(ctx, old_loc),
                                       relocs, n_relocs, lmap, n_labels, tmap, n_tmap);
                thvm_grad_target_set(ctx, new_loc, gx);
                thvm_grad_mode_set(ctx, new_loc, thvm_grad_mode_get(ctx, old_loc));
                u32 nt = thvm_grad_targets_count_at(ctx, old_loc);
                if (nt > 0) {
                    Term params[THVM_GRAD_TARGETS_MAX];
                    Term slots[THVM_GRAD_TARGETS_MAX];
                    assert(nt <= THVM_GRAD_TARGETS_MAX);
                    for (u32 i = 0; i < nt; i++) {
                        params[i] = term_clone_r(ctx, thvm_grad_targets_get_term_at(ctx, old_loc, i),
                                                 relocs, n_relocs, lmap, n_labels, tmap, n_tmap);
                        slots[i] = term_clone_r(ctx, thvm_grad_targets_get_slot_at(ctx, old_loc, i),
                                                relocs, n_relocs, lmap, n_labels, tmap, n_tmap);
                    }
                    thvm_grad_targets_set_for_loc(ctx, new_loc, params, slots, nt);
                }
                Term gb = thvm_grad_keep_bundle_get(ctx, old_loc);
                Term gb_old = gb;
                gb = term_clone_r(ctx, gb, relocs, n_relocs, lmap, n_labels, tmap, n_tmap);
                if (!(term_tag(gb) == TAG_ERA && term_val(gb) == 0))
                    thvm_grad_keep_bundle_set(ctx, new_loc, gb);
                if (getenv("THVM_LOOP_DIAG")) {
                    fprintf(stderr,
                            "CLONE_GRAD old_loc=%llu new_loc=%llu old_bundle_tag=%u old_bundle_val=%llu new_bundle_tag=%u new_bundle_val=%llu\n",
                            (unsigned long long)old_loc,
                            (unsigned long long)new_loc,
                            (u32)term_tag(gb_old),
                            (unsigned long long)term_val(gb_old),
                            (u32)term_tag(gb),
                            (unsigned long long)term_val(gb));
                }
            }
            Term out = term_new(TAG_TOP, uop, new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── MAT: 2 slots (ok, fallback) ─────────────────────────
        case TAG_MAT: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            Term out = term_new(TAG_MAT, term_ext(t), new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── CTR: ext children ───────────────────────────────────
        case TAG_CTR: {
            u32 arity = term_ext(t);
            u64 old_loc = term_val(t);
            if (arity == 0 || old_loc == 0) return t;
            u64 new_loc = heap_alloc(ctx, arity);
            for (u32 i = 0; i < arity; i++) {
                heap_set(ctx, new_loc + i,
                         term_clone_r(ctx, heap_read(ctx, old_loc + i),
                                      relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            }
            Term out = term_new(TAG_CTR, term_ext(t), new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── USP: 2 slots (freshen label) ────────────────────────
        case TAG_USP: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            Term out = term_new(TAG_USP, clone_fresh_label(ctx, term_ext(t), lmap, n_labels), new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── UDP: 1 slot (freshen label) ─────────────────────────
        case TAG_UDP: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 1);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels, tmap, n_tmap));
            Term out = term_new(TAG_UDP, clone_fresh_label(ctx, term_ext(t), lmap, n_labels), new_loc);
            clone_term_map_add(tmap, n_tmap, t, out);
            return out;
        }

        // ── Unknown: return as-is ────────────────────────────────
        default:
            return t;
    }
}

// Public API: deep-clone a term tree (with fresh labels for DUP/SUP)
static Term term_clone(TinyHVM *ctx, Term t) {
    Reloc relocs[CLONE_MAX_RELOC];
    u32 n_relocs = 0;
    LabelMap lmap[CLONE_MAX_LABELS];
    u32 n_labels = 0;
    TermMap tmap[CLONE_MAX_TERMMAP];
    u32 n_tmap = 0;
    return term_clone_r(ctx, t, relocs, &n_relocs, lmap, &n_labels, tmap, &n_tmap);
}
