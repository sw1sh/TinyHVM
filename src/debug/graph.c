// graph.c — BFS heap walker for inet graph visualization
// Returns node/edge arrays suitable for WL Graph construction.
static int heap_dot_root_only = 0; // skip global ERA/ASSIGN seeding in step graphs
void thvm_heap_dot_set_root_only(int v) { heap_dot_root_only = v ? 1 : 0; }
int thvm_heap_dot_get_root_only(void) { return heap_dot_root_only; }
static u64 heap_dot_node_hl = 0;  // node highlight (red border when edge hl fails)
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root);
static void thvm_heap_dot_set_highlight(u64 slot, Term term);
static void thvm_heap_dot_set_step_meta(const char *prev_name, const char *next_name);
static const char *thvm_step_tag_name_short(u8 tag);
static const char *thvm_step_graph_interaction_name_current_at(TinyHVM *ctx, u64 source_slot,
                                                               Term before, char *buf, size_t nbuf);
static void thvm_step_graph_append_focus_suffix(TinyHVM *ctx, u64 source_slot, Term before,
                                                u64 hl_slot, Term hl_term,
                                                char *buf, size_t nbuf);

// Simple open-addressing hash set for visited Term values
typedef struct {
    Term *keys;
    u32   cap;
    u32   count;
} TermSet;

static TermSet termset_create(u32 cap) {
    // Round up to power of 2
    u32 n = 1; while (n < cap) n <<= 1;
    Term *keys = (Term *)calloc(n, sizeof(Term));
    return (TermSet){keys, n, 0};
}

static void termset_free(TermSet *s) { free(s->keys); }

static int termset_insert(TermSet *s, Term t) {
    if (t == 0) return 0; // 0 = empty sentinel
    u32 mask = s->cap - 1;
    u32 h = (u32)(t * 0x9E3779B97F4A7C15ULL >> 32) & mask;
    for (;;) {
        if (s->keys[h] == 0) { s->keys[h] = t; s->count++; return 1; }
        if (s->keys[h] == t) return 0; // already present
        h = (h + 1) & mask;
    }
}

// Number of children for a given tag
static u32 tag_arity(u32 tag, u32 ext) {
    switch (tag) {
        case TAG_APP: return 2;
        case TAG_LAM: return 2; // var + body
        case TAG_SEQ: return 2;
        case TAG_ALO: return 0;
        case TAG_SUP: return 2;
        case TAG_DP0: case TAG_DP1: return 1; // 1-slot DUP: only heap[val]
        case TAG_OP2: return 2;
        case TAG_TOP:
            return thvm_uop_visible_arity(ext);
        case TAG_USP: return 2;
        case TAG_UDP: return 1;
        case TAG_EQL: return 2;
        case TAG_AND: return 2;
        case TAG_OR:  return 2;
        case TAG_MAT: return 2;
        case TAG_CTR: return ext; // arity in ext field
        // Leaves:
        case TAG_TEN: case TAG_ERA: case TAG_NUM:
        case TAG_VAR: case TAG_REF: case TAG_ANY:
            return 0;
        default: return 0;
    }
}

// Heap graph output (caller frees node_data and edge_data)
// node_data: [tag, ext, val_lo, heap_loc_lo] per node (4 ints per node)
// edge_data: [from_idx, to_idx] per edge (2 ints per edge)
void thvm_heap_graph(TinyHVM *ctx, Term root,
                     i32 **out_nodes, u32 *out_n_nodes,
                     i32 **out_edges, u32 *out_n_edges) {
    u32 max_n = 4096, max_e = 8192;
    i32 *nodes = (i32 *)malloc(max_n * 4 * sizeof(i32));
    i32 *edges = (i32 *)malloc(max_e * 2 * sizeof(i32));
    u32 nn = 0, ne = 0;

    // BFS queue
    u32 q_cap = 4096;
    Term *queue = (Term *)malloc(q_cap * sizeof(Term));
    u32 *parent_idx = (u32 *)malloc(q_cap * sizeof(u32)); // parent node index (or ~0)
    u32 qh = 0, qt = 0;

    TermSet visited = termset_create(8192);

    // Enqueue root
    queue[qt] = root; parent_idx[qt] = ~0u; qt++;
    termset_insert(&visited, root);

    while (qh < qt) {
        Term t = queue[qh];
        u32 pidx = parent_idx[qh];
        qh++;

        u32 tag = term_tag(t);
        u32 ext = term_ext(t);
        u64 val = term_val(t);

        // Add node
        u32 my_idx = nn;
        if (nn >= max_n) { max_n *= 2; nodes = (i32 *)realloc(nodes, max_n * 4 * sizeof(i32)); }
        nodes[nn * 4 + 0] = (i32)tag;
        nodes[nn * 4 + 1] = (i32)ext;
        nodes[nn * 4 + 2] = (i32)(val & 0xFFFFFFFF);
        // heap_loc: for nodes with children, val IS the heap location;
        // for leaves (TEN, NUM, ERA, VAR, REF), use -1 (no meaningful heap loc)
        u32 arity = tag_arity(tag, ext);
        nodes[nn * 4 + 3] = (arity > 0) ? (i32)(val & 0xFFFFFFFF) : -1;
        nn++;

        // Add edge: data flows child -> parent
        if (pidx != ~0u) {
            if (ne >= max_e) { max_e *= 2; edges = (i32 *)realloc(edges, max_e * 2 * sizeof(i32)); }
            edges[ne * 2 + 0] = (i32)my_idx;
            edges[ne * 2 + 1] = (i32)pidx;
            ne++;
        }

        // Enqueue children
        for (u32 i = 0; i < arity; i++) {
            Term child = heap_read(ctx, val + i);
            if (child == 0) continue; // uninitialized

            if (termset_insert(&visited, child)) {
                // Grow queue if needed
                if (qt >= q_cap) {
                    q_cap *= 2;
                    queue = (Term *)realloc(queue, q_cap * sizeof(Term));
                    parent_idx = (u32 *)realloc(parent_idx, q_cap * sizeof(u32));
                }
                queue[qt] = child;
                parent_idx[qt] = my_idx;
                qt++;
            } else {
                // Already visited — still add edge to existing node
                // Find existing node index
                for (u32 j = 0; j < nn; j++) {
                    u32 jtag = (u32)nodes[j * 4 + 0];
                    u32 jval = (u32)nodes[j * 4 + 2];
                    if (jtag == term_tag(child) && jval == (u32)(term_val(child) & 0xFFFFFFFF)) {
                        if (ne >= max_e) { max_e *= 2; edges = (i32 *)realloc(edges, max_e * 2 * sizeof(i32)); }
                        edges[ne * 2 + 0] = (i32)j;
                        edges[ne * 2 + 1] = (i32)my_idx;
                        ne++;
                        break;
                    }
                }
            }
        }
    }

    free(queue);
    free(parent_idx);
    termset_free(&visited);

    *out_nodes = nodes;
    *out_n_nodes = nn;
    *out_edges = edges;
    *out_n_edges = ne;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step-graph hooks fired from thvm_reduce interaction path
// ─────────────────────────────────────────────────────────────────────────────

static int step_graph_active = 0;
static u32 step_graph_n = 0;
static u64 step_graph_last_sig = 0;
static u64 step_graph_last_dot_sig = 0; // signature of unhighlighted structural DOT
static u64 step_graph_last_render_sig = 0; // signature of last written visible DOT before meta rewrite
static Term step_graph_root_term = 0;
static Term step_graph_before_grad_y = 0;
static Term step_graph_before_era_payload = 0;
static int step_graph_before_top_had_era = 0;
static int step_graph_before_top_had_add_zero = 0;
static Term step_graph_before_top_partner = 0;
static u64  step_graph_before_top_partner_slot = 0;
static char step_graph_last_name[160] = {0};
static char step_graph_last_prev_name[160] = {0};
static char step_graph_last_file[384] = {0};
static char step_graph_lower_anchor_name[160] = {0};
static u32 step_graph_lower_anchor_index = 0;

static const char *thvm_step_graph_dir(void) {
    const char *dir = getenv("THVM_STEP_GRAPH_DIR");
    return (dir && dir[0]) ? dir : "graphs";
}

static int thvm_step_graph_is_active(void) {
    return step_graph_active ? 1 : 0;
}

static u32 thvm_step_graph_current_index(void) {
    return step_graph_n;
}

static const char *thvm_step_graph_last_name(void) {
    return step_graph_last_name[0] ? step_graph_last_name : step_graph_last_prev_name;
}

static const char *thvm_step_graph_lower_anchor_name(void) {
    return step_graph_lower_anchor_name;
}

static u32 thvm_step_graph_lower_anchor_index(void) {
    return step_graph_lower_anchor_index;
}

static u32 thvm_step_graph_max_steps(void) {
    const char *env = getenv("THVM_STEP_GRAPH_MAX");
    if (env && env[0]) {
        unsigned long n = strtoul(env, NULL, 10);
        if (n > 0) return (u32)n;
    }
    return 512;
}



static void thvm_step_graph_set_before_grad_y(Term y) {
    step_graph_before_grad_y = y;
}

static void thvm_step_graph_set_before_era_payload(Term payload) {
    step_graph_before_era_payload = payload;
}

static void thvm_step_graph_set_before_top_era(int had_era) {
    step_graph_before_top_had_era = had_era;
}

static void thvm_step_graph_set_before_top_add_zero(int had_add_zero) {
    step_graph_before_top_had_add_zero = had_add_zero;
}

static void thvm_step_graph_set_before_top_partner(Term partner, u64 slot) {
    step_graph_before_top_partner = partner;
    step_graph_before_top_partner_slot = slot;
}

static u64 thvm_file_sig(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    u64 h = 1469598103934665603ULL;
    int c;
    while ((c = fgetc(f)) != EOF) {
        h ^= (u8)c;
        h *= 1099511628211ULL;
    }
    fclose(f);
    return h;
}


static u32 thvm_step_top_arity(u32 ext) {
    return thvm_uop_visible_arity(ext);
}

static u32 thvm_step_term_arity(Term t);

static int thvm_step_term_is_active_era_like(TinyHVM *ctx, Term t, Term *era_out) {
    if (term_tag(t) == TAG_ERA && term_val(t) != 0) {
        if (era_out) *era_out = t;
        return 1;
    }
    if (term_tag(t) == TAG_VAR) {
        u64 loc = term_val(t);
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

static int thvm_step_top_has_era_arg(TinyHVM *ctx, Term t, u64 *slot_out, Term *term_out) {
    if (term_tag(t) != TAG_TOP) return 0;
    u64 loc = term_val(t);
    u32 arity = thvm_step_top_arity(term_ext(t));
    for (u32 i = 0; i < arity; i++) {
        Term child = heap_read(ctx, loc + i);
        Term era_child = 0;
        if (thvm_step_term_is_active_era_like(ctx, child, &era_child)) {
            if (slot_out) *slot_out = loc + i;
            if (term_out) *term_out = era_child;
            return 1;
        }
    }
    return 0;
}

static int thvm_step_top_has_add_zero_arg(TinyHVM *ctx, Term t, u64 *slot_out, Term *term_out) {
    if (term_tag(t) != TAG_TOP || term_ext(t) != UOP_ADD) return 0;
    u64 loc = term_val(t);
    for (u32 i = 0; i < 2; i++) {
        Term child = heap_read(ctx, loc + i);
        if (term_tag(child) == TAG_NUM && term_as_f32(child) == 0.0f) {
            if (slot_out) *slot_out = loc + i;
            if (term_out) *term_out = child;
            return 1;
        }
    }
    return 0;
}

static int thvm_step_slot_is_ifz_cond(TinyHVM *ctx, u64 slot) {
    if (slot == 0 || slot >= ctx->heap_pos) return 0;
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term p = ctx->heap[h];
        if (term_tag(p) != TAG_TOP || term_ext(p) != UOP_IFZ) continue;
        u64 ploc = term_val(p);
        if (ploc == 0 || ploc + 2 >= ctx->heap_pos) continue;
        if (ploc == slot) return 1;
    }
    return 0;
}

static u32 thvm_step_term_arity(Term t) {
    u8 tag = term_tag(t);
    u32 ext = term_ext(t);
    switch (tag) {
        case TAG_TOP:
            return thvm_uop_visible_arity(ext);
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
        case TAG_ALO:
            return 0;
        case TAG_DSU:
        case TAG_DDU:
            return 3;
        case TAG_CTR:
            return ext;
        case TAG_DP0:
        case TAG_DP1:
        case TAG_UDP:
        case TAG_ERA:
        case TAG_VAR:
        case TAG_INC:
            return 1;
        default:
            return 0;
    }
}

static int thvm_step_slot_is_rendered_parent_arg(TinyHVM *ctx, u64 slot) {
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term p = ctx->heap[h];
        u8 tag = term_tag(p);
        if (tag == TAG_TOP) {
            u64 loc = term_val(p);
            u32 arity = thvm_uop_visible_arity(term_ext(p));
            for (u32 i = 0; i < arity; i++) if (loc + i == slot) return 1;
            continue;
        }
        if (tag == TAG_CTR) {
            u64 cl = term_val(p);
            u32 arity = term_ext(p);
            if (cl == 0 || cl + arity > ctx->heap_pos) continue;
            for (u32 i = 0; i < arity; i++) if (cl + i == slot) return 1;
            continue;
        }
        if (tag == TAG_ERA && term_val(p) == slot) return 1;
    }
    return 0;
}

static int thvm_step_find_active_era_for_payload_slot(TinyHVM *ctx, u64 payload_slot,
                                                      u64 *out_era_slot, Term *out_era_term) {
    if (!ctx || payload_slot == 0 || payload_slot >= ctx->heap_pos) return 0;
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term p = ctx->heap[h];
        if (term_tag(p) != TAG_ERA) continue;
        if (term_val(p) != payload_slot) continue;
        if (out_era_slot) *out_era_slot = h;
        if (out_era_term) *out_era_term = p;
        return 1;
    }
    return 0;
}

static Term thvm_step_parent_term_for_slot(TinyHVM *ctx, u64 slot) {
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term p = ctx->heap[h];
        u32 arity = thvm_step_term_arity(p);
        u64 loc = term_val(p);
        if (term_tag(p) == TAG_ERA) continue;
        if (arity == 0 || loc == 0 || loc + arity > ctx->heap_pos) continue;
        for (u32 i = 0; i < arity; i++) {
            if (loc + i == slot) return p;
        }
    }
    return 0;
}






// Second heap index involved in the redex (for filenames). Returns 0 if unknown.
static u64 thvm_step_graph_term_display_loc(TinyHVM *ctx, u64 slot_hint, Term t) {
    switch (term_tag(t)) {
        case TAG_REF:
            return slot_hint;
        case TAG_TEN:
            if (ctx && slot_hint > 0 && slot_hint < ctx->heap_pos &&
                ctx->heap[slot_hint] == t && thvm_step_slot_is_rendered_parent_arg(ctx, slot_hint))
                return slot_hint;
            for (u64 h = 1; ctx && h < ctx->heap_pos; h++) {
                if (ctx->heap[h] == t && thvm_step_slot_is_rendered_parent_arg(ctx, h)) return h;
            }
            for (u64 h = 1; ctx && h < ctx->heap_pos; h++) {
                if (ctx->heap[h] == t) return h;
            }
            return slot_hint;
        case TAG_NUM:
            return slot_hint;
        default:
            return term_val(t);
    }
}

static u64 thvm_step_graph_partner_loc(TinyHVM *ctx, u64 source_slot, Term before,
                                       u64 hl_slot, Term hl_term, u64 self_loc) {
    u8 tag = term_tag(before);
    if (tag == TAG_ALO) {
        u64 era_slot = 0;
        Term era_term = 0;
        if (hl_slot != 0 &&
            thvm_step_find_active_era_for_payload_slot(ctx, hl_slot, &era_slot, &era_term) &&
            era_slot != self_loc)
            return era_slot;
        if (source_slot != 0 &&
            thvm_step_find_active_era_for_payload_slot(ctx, source_slot, &era_slot, &era_term) &&
            era_slot != self_loc)
            return era_slot;
    }
    if (tag == TAG_ERA) {
        Term payload = step_graph_before_era_payload;
        if (payload == 0 || (term_tag(payload) == TAG_ERA && term_val(payload) == 0)) {
            u64 el = term_val(before);
            payload = (el > 0 && el < ctx->heap_pos) ? thvm_era_payload(ctx, heap_read(ctx, el))
                                                     : term_era();
        }
        u64 el = term_val(before);
        u64 ploc = thvm_step_graph_term_display_loc(ctx, el ? el : (hl_slot ? hl_slot : source_slot), payload);
        if (ploc) return ploc;
    }
    if (tag == TAG_TOP && step_graph_before_top_partner != 0 &&
        term_tag(step_graph_before_top_partner) != TAG_ERA) {
        u64 ploc = thvm_step_graph_term_display_loc(
            ctx,
            step_graph_before_top_partner_slot ? step_graph_before_top_partner_slot : source_slot,
            step_graph_before_top_partner
        );
        if (ploc && ploc != self_loc) return ploc;
    }

    if (tag == TAG_TOP && hl_term != 0 && term_tag(hl_term) == TAG_ERA) {
        u64 el = term_val(hl_term);
        if (el > 0 && el < ctx->heap_pos) {
            Term payload = thvm_era_payload(ctx, heap_read(ctx, el));
            u64 ploc = thvm_step_graph_term_display_loc(ctx, el, payload);
            if (ploc && ploc != self_loc) return ploc;
        }
    }

    if (hl_term != 0) {
        u64 hloc = thvm_step_graph_term_display_loc(ctx, hl_slot ? hl_slot : source_slot, hl_term);
        if (hloc && hloc != self_loc) return hloc;
    }

    if (hl_slot != 0 && hl_slot < ctx->heap_pos) {
        Term parent = thvm_step_parent_term_for_slot(ctx, hl_slot);
        u64 ploc = thvm_step_graph_term_display_loc(ctx, hl_slot, parent);
        if (ploc && ploc != self_loc) return ploc;
    }

    if (tag == TAG_APP || tag == TAG_SEQ) {
        u64 loc = term_val(before);
        if (loc > 0 && loc + 1 < ctx->heap_pos) {
            u64 other_slot = 0;
            if (hl_slot == loc) other_slot = loc + 1;
            else if (hl_slot == loc + 1) other_slot = loc;
            else if (source_slot == loc) other_slot = loc + 1;
            else if (source_slot == loc + 1) other_slot = loc;
            if (other_slot) {
                u64 other_loc = thvm_step_graph_term_display_loc(ctx, other_slot, heap_read(ctx, other_slot));
                if (other_loc) return other_loc;
            }
        }
    }

    return 0;
}

static const char *thvm_step_graph_focus_term_name(Term t, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return "";
    buf[0] = '\0';
    if (term_tag(t) == TAG_TOP && term_ext(t) < UOP_COUNT) {
        snprintf(buf, nbuf, "%s", uop_names[term_ext(t)]);
        return buf;
    }
    snprintf(buf, nbuf, "%s", thvm_step_tag_name_short(term_tag(t)));
    return buf;
}

static Term thvm_step_graph_partner_term(TinyHVM *ctx, u64 source_slot, Term before,
                                         u64 hl_slot, Term hl_term) {
    u8 tag = term_tag(before);
    if (tag == TAG_ALO) {
        u64 era_slot = 0;
        Term era_term = 0;
        if (hl_slot != 0 &&
            thvm_step_find_active_era_for_payload_slot(ctx, hl_slot, &era_slot, &era_term))
            return era_term;
        if (source_slot != 0 &&
            thvm_step_find_active_era_for_payload_slot(ctx, source_slot, &era_slot, &era_term))
            return era_term;
    }
    if (tag == TAG_ERA) {
        Term payload = step_graph_before_era_payload;
        if (payload == 0 || (term_tag(payload) == TAG_ERA && term_val(payload) == 0)) {
            u64 el = term_val(before);
            payload = (el > 0 && el < ctx->heap_pos) ? thvm_era_payload(ctx, heap_read(ctx, el))
                                                     : term_era();
        }
        return payload;
    }
    if (tag == TAG_TOP && step_graph_before_top_partner != 0 &&
        term_tag(step_graph_before_top_partner) != TAG_ERA)
        return step_graph_before_top_partner;
    if (tag == TAG_TOP && hl_term != 0 && term_tag(hl_term) == TAG_ERA) {
        u64 el = term_val(hl_term);
        if (el > 0 && el < ctx->heap_pos)
            return thvm_era_payload(ctx, heap_read(ctx, el));
        return term_era();
    }
    if (hl_term != 0 && hl_slot != 0 && term_tag(hl_term) != TAG_ERA)
        return hl_term;
    if (tag == TAG_APP || tag == TAG_SEQ) {
        u64 loc = term_val(before);
        if (loc > 0 && loc + 1 < ctx->heap_pos) {
            u64 other_slot = 0;
            if (hl_slot == loc) other_slot = loc + 1;
            else if (hl_slot == loc + 1) other_slot = loc;
            else if (source_slot == loc) other_slot = loc + 1;
            else if (source_slot == loc + 1) other_slot = loc;
            if (other_slot) return heap_read(ctx, other_slot);
        }
    }
    if (hl_slot != 0 && hl_slot < ctx->heap_pos) return hl_term;
    if (source_slot != 0 && source_slot < ctx->heap_pos) return heap_read(ctx, source_slot);
    return before;
}




static void thvm_step_graph_append_focus_suffix(TinyHVM *ctx, u64 source_slot, Term before,
                                                u64 hl_slot, Term hl_term,
                                                char *buf, size_t nbuf) {
    if (!buf || !buf[0] || nbuf == 0) return;
    int era_like = 0;
    if (term_tag(before) == TAG_ERA) era_like = 1;
    if (term_tag(before) == TAG_TOP && hl_term != 0 && term_tag(hl_term) == TAG_ERA) era_like = 1;

    u64 self_loc = era_like
        ? (hl_slot ? hl_slot : source_slot)
        : thvm_step_graph_term_display_loc(ctx, source_slot ? source_slot : hl_slot, before);
    if (self_loc == 0) self_loc = hl_slot ? hl_slot : source_slot;

    Term other_term = thvm_step_graph_partner_term(ctx, source_slot, before, hl_slot, hl_term);
    u64 era_other_hint = hl_slot ? hl_slot : source_slot;
    if (term_tag(before) == TAG_ERA && term_val(before) != 0) {
        era_other_hint = term_val(before);
    } else if (term_tag(before) == TAG_TOP && hl_term != 0 &&
               term_tag(hl_term) == TAG_ERA && term_val(hl_term) != 0) {
        era_other_hint = term_val(hl_term);
    }
    u64 other_loc = era_like
        ? thvm_step_graph_term_display_loc(ctx, era_other_hint, other_term)
        : thvm_step_graph_partner_loc(ctx, source_slot, before, hl_slot, hl_term, self_loc);
    if (other_loc == 0) {
        if (term_tag(before) == TAG_VAR && self_loc != 0) other_loc = self_loc;
        if (hl_slot != 0 && hl_slot != self_loc) other_loc = hl_slot;
        else if (source_slot != 0 && source_slot != self_loc) other_loc = source_slot;
    }

    if (self_loc == 0) self_loc = other_loc;
    if (other_loc == 0) other_loc = self_loc;

    if (self_loc != 0) {
        char other_name[32];
        thvm_step_graph_focus_term_name(other_term, other_name, sizeof(other_name));
        snprintf(buf + strlen(buf), nbuf - strlen(buf), "_h%llu_%s_h%llu",
                 (unsigned long long)self_loc,
                 other_name[0] ? other_name : "UNK",
                 (unsigned long long)other_loc);
    }
}



static const char *thvm_step_tag_name_short(u8 tag) {
    switch (tag) {
        case TAG_ERA: return "ERA";
        case TAG_TOP: return "TOP";
        case TAG_TEN: return "TEN";
        case TAG_DP0: return "DP0";
        case TAG_DP1: return "DP1";
        case TAG_SUP: return "SUP";
        case TAG_APP: return "APP";
        case TAG_CTR: return "CTR";
        case TAG_NUM: return "NUM";
        case TAG_REF: return "REF";
        case TAG_VAR: return "VAR";
        case TAG_LAM: return "LAM";
        case TAG_BRI: return "BRI";
        case TAG_SEQ: return "SEQ";
        case TAG_OP2: return "OP2";
        case TAG_USP: return "USP";
        case TAG_UDP: return "UDP";
        case TAG_EQL: return "EQL";
        case TAG_AND: return "AND";
        case TAG_OR:  return "OR";
        case TAG_MAT: return "MAT";
        case TAG_ANY: return "ANY";
        case TAG_ALO: return "ALO";
        default: return "UNK";
    }
}



static const char *thvm_step_graph_interaction_name_current_at(TinyHVM *ctx, u64 source_slot,
                                                               Term before, char *buf, size_t nbuf) {
    (void)source_slot;
    u8 tag = term_tag(before);
    if (tag == TAG_TOP) {
        u32 ext = term_ext(before);
        if (ext == UOP_GRAD) {
            snprintf(buf, nbuf, "GRAD");
            return buf;
        }
        if (ext < UOP_COUNT) {
            if (thvm_step_top_has_era_arg(ctx, before, NULL, NULL))
                snprintf(buf, nbuf, "ERA");
            else
                snprintf(buf, nbuf, "%s", uop_names[ext]);
            return buf;
        }
    }
    if (tag == TAG_ERA) {
        snprintf(buf, nbuf, "ERA");
        return buf;
    }
    snprintf(buf, nbuf, "%s", thvm_step_tag_name_short(tag));
    return buf;
}

// Pre-interaction hook — called from thvm_interact entry. Reads BEFORE
// metadata from live heap (GRAD y, ERA payload, TOP era/add-zero args)
// while the interaction hasn't fired yet.
void thvm_step_graph_on_pre_interaction(TinyHVM *ctx, Term before) {
    if (!getenv("THVM_STEP_GRAPH") || !step_graph_active) return;
    thvm_step_capture_step_before_meta(ctx, before);
}

// Post-interaction hook — called from TRACE_STEP in thvm_reduce_steps after
// each fired interaction. Lazy-initializes the dump session on first call
// when env is set. Updates step_root_slot's heap mirror so the dumper
// sees the current root, and computes the correct source_slot via the
// same helper structural_nf used.
void thvm_step_graph_on_post_interaction(TinyHVM *ctx, Term before, Term root) {
    // Disabled: the in-reducer hook path can't maintain the root mirror
    // reliably from deep reducer frames. The step-graph session now does
    // its own budget-1 tracing loop in thvm_trace_step_graph_session.
    (void)ctx; (void)before; (void)root;
}
