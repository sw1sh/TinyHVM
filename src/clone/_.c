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

#define CLONE_MAX_RELOC 128
#define CLONE_MAX_LABELS 64

typedef struct { u64 old_loc; u64 new_loc; } Reloc;
typedef struct { u32 old_label; u32 new_label; } LabelMap;

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

static Term term_clone_r(TinyHVM *ctx, Term t, Reloc *relocs, u32 *n_relocs,
                          LabelMap *lmap, u32 *n_labels) {
    u32 tag = term_tag(t);

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
            // Clone body
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels));
            return term_new(TAG_LAM, term_ext(t), new_loc);
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
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels));
            return term_new(TAG_BRI, term_ext(t), new_loc);
        }

        // ── APP: 2 slots (fun, arg) ──────────────────────────────
        case TAG_APP: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels));
            return term_new(TAG_APP, term_ext(t), new_loc);
        }

        // ── SUP: 2 slots (freshen label) ────────────────────────
        case TAG_SUP: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels));
            return term_new(TAG_SUP, clone_fresh_label(ctx, term_ext(t), lmap, n_labels), new_loc);
        }

        // ── DP0/DP1: 1 slot (freshen label) ─────────────────────
        // If backing points to a lambda's var slot (in relocation table),
        // share the relocated slot directly so DUP reads the lambda's
        // substitution without VAR indirection.
        case TAG_DP0:
        case TAG_DP1: {
            u64 old_loc = term_val(t);
            u64 use_loc = 0;
            int shared = 0;
            for (u32 i = 0; i < *n_relocs; i++) {
                if (relocs[i].old_loc == old_loc) {
                    use_loc = relocs[i].new_loc;
                    shared = 1;
                    break;
                }
            }
            if (!shared) {
                use_loc = heap_alloc(ctx, 1);
                heap_set(ctx, use_loc,
                         term_clone_r(ctx, heap_read(ctx, old_loc),
                                      relocs, n_relocs, lmap, n_labels));
            }
            return term_new(tag, clone_fresh_label(ctx, term_ext(t), lmap, n_labels), use_loc);
        }

        // ── ANN: 2 slots (term, type) ─────────────────────────────
        case TAG_ANN: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels));
            return term_new(TAG_ANN, term_ext(t), new_loc);
        }

        // ── DSU/DDU: 3 slots ─────────────────────────────────────
        case TAG_DSU:
        case TAG_DDU: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 3);
            for (u32 i = 0; i < 3; i++)
                heap_set(ctx, new_loc + i,
                         term_clone_r(ctx, heap_read(ctx, old_loc + i),
                                      relocs, n_relocs, lmap, n_labels));
            return term_new(tag, term_ext(t), new_loc);
        }

        // ── INC: 1 slot ──────────────────────────────────────────
        case TAG_INC: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 1);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels));
            return term_new(TAG_INC, term_ext(t), new_loc);
        }

        // ── OP2: 2 slots (x, y) ─────────────────────────────────
        case TAG_OP2: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs, lmap, n_labels));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs, lmap, n_labels));
            return term_new(TAG_OP2, term_ext(t), new_loc);
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
                                      relocs, n_relocs, lmap, n_labels));
            return term_new(TAG_TOP, uop, new_loc);
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
    return term_clone_r(ctx, t, relocs, &n_relocs, lmap, &n_labels);
}
