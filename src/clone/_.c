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

typedef struct { u64 old_loc; u64 new_loc; } Reloc;

static Term term_clone_r(TinyHVM *ctx, Term t, Reloc *relocs, u32 *n_relocs) {
    u32 tag = term_tag(t);

    switch (tag) {
        // ── Terminals: no heap state ──────────────────────────────
        case TAG_NUM:
        case TAG_TEN:
        case TAG_ERA:
        case TAG_REF:
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
                                  relocs, n_relocs));
            return term_new(TAG_LAM, term_ext(t), new_loc);
        }

        // ── APP: 2 slots (fun, arg) ──────────────────────────────
        case TAG_APP: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs));
            return term_new(TAG_APP, term_ext(t), new_loc);
        }

        // ── SUP: 2 slots ────────────────────────────────────────
        case TAG_SUP: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs));
            return term_new(TAG_SUP, term_ext(t), new_loc);
        }

        // ── DP0/DP1: 1 slot (the shared expr) ───────────────────
        case TAG_DP0:
        case TAG_DP1: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 1);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs));
            return term_new(tag, term_ext(t), new_loc);
        }

        // ── OP2: 2 slots (x, y) ─────────────────────────────────
        case TAG_OP2: {
            u64 old_loc = term_val(t);
            u64 new_loc = heap_alloc(ctx, 2);
            heap_set(ctx, new_loc,
                     term_clone_r(ctx, heap_read(ctx, old_loc),
                                  relocs, n_relocs));
            heap_set(ctx, new_loc + 1,
                     term_clone_r(ctx, heap_read(ctx, old_loc + 1),
                                  relocs, n_relocs));
            return term_new(TAG_OP2, term_ext(t), new_loc);
        }

        // ── TOP: tensor op, 2 or 3 slots ────────────────────────
        case TAG_TOP: {
            u32 uop = term_ext(t);
            u64 old_loc = term_val(t);
            u32 arity = (uop == UOP_GRAD || uop == UOP_WHERE || uop == UOP_IFZ) ? 3 : 2;
            u64 new_loc = heap_alloc(ctx, arity);
            for (u32 i = 0; i < arity; i++)
                heap_set(ctx, new_loc + i,
                         term_clone_r(ctx, heap_read(ctx, old_loc + i),
                                      relocs, n_relocs));
            return term_new(TAG_TOP, uop, new_loc);
        }

        // ── Unknown: return as-is ────────────────────────────────
        default:
            return t;
    }
}

// Public API: deep-clone a term tree
static Term term_clone(TinyHVM *ctx, Term t) {
    Reloc relocs[CLONE_MAX_RELOC];
    u32 n_relocs = 0;
    return term_clone_r(ctx, t, relocs, &n_relocs);
}
