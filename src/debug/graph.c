// graph.c — BFS heap walker for inet graph visualization
// Returns node/edge arrays suitable for WL Graph construction.
static void thvm_heap_dot(TinyHVM *ctx, const char *path);
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root);
static void thvm_heap_dot_set_highlight(u64 slot, Term term);
static int thvm_heap_dot_highlight_was_drawn(void);
static void thvm_heap_dot_set_step_meta(const char *prev_name, const char *next_name);
static int thvm_phase1_find_next_actual(TinyHVM *ctx, Term root, u64 *out_source_slot, Term *out_before);
static const char *thvm_step_graph_interaction_name(TinyHVM *ctx, Term before,
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
        case TAG_SUP: return 2;
        case TAG_DP0: case TAG_DP1: return 1; // 1-slot DUP: only heap[val]
        case TAG_OP2: return 2;
        case TAG_TOP:
            if (ext == UOP_KERNEL) return 0;
            if (ext == UOP_WHERE) return 3;
            if (ext == UOP_GRAD) return 2;
            if (ext == UOP_LOG_PRINT) return 1;
            if (ext == UOP_DETACH) return 1;
            if (!is_binary(ext) && is_elementwise(ext)) return 1;
            return 2;
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
// node_data: [tag, ext, val_lo] per node (3 ints per node)
// edge_data: [from_idx, to_idx] per edge (2 ints per edge)
void thvm_heap_graph(TinyHVM *ctx, Term root,
                     i32 **out_nodes, u32 *out_n_nodes,
                     i32 **out_edges, u32 *out_n_edges) {
    u32 max_n = 4096, max_e = 8192;
    i32 *nodes = (i32 *)malloc(max_n * 3 * sizeof(i32));
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
        if (nn >= max_n) { max_n *= 2; nodes = (i32 *)realloc(nodes, max_n * 3 * sizeof(i32)); }
        nodes[nn * 3 + 0] = (i32)tag;
        nodes[nn * 3 + 1] = (i32)ext;
        nodes[nn * 3 + 2] = (i32)(val & 0xFFFFFFFF);
        nn++;

        // Add edge: data flows child -> parent
        if (pidx != ~0u) {
            if (ne >= max_e) { max_e *= 2; edges = (i32 *)realloc(edges, max_e * 2 * sizeof(i32)); }
            edges[ne * 2 + 0] = (i32)my_idx;
            edges[ne * 2 + 1] = (i32)pidx;
            ne++;
        }

        // Enqueue children
        u32 arity = tag_arity(tag, ext);
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
                    u32 jtag = (u32)nodes[j * 3 + 0];
                    u32 jval = (u32)nodes[j * 3 + 2];
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
static Term step_graph_root_term = 0;
static Term step_graph_before_grad_y = 0;
static Term step_graph_before_era_payload = 0;
static int step_graph_before_top_had_era = 0;
static int step_graph_before_top_had_add_zero = 0;
static char step_graph_last_prev_name[96] = {0};

static const char *thvm_step_graph_dir(void) {
    const char *dir = getenv("THVM_STEP_GRAPH_DIR");
    return (dir && dir[0]) ? dir : "thvm_steps";
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

static u64 thvm_step_graph_sig(TinyHVM *ctx) {
    u64 h = 1469598103934665603ULL; // FNV-1a 64-bit offset basis
    #define STEP_MIX(_x) do { h ^= (u64)(_x); h *= 1099511628211ULL; } while (0)
    STEP_MIX(ctx->heap_pos);
    STEP_MIX(ctx->tensor_count);
    for (u64 i = 1; i < ctx->heap_pos; i++) STEP_MIX(ctx->heap[i]);
    for (u32 i = 0; i < ctx->tensor_count; i++) {
        TensorMeta *m = &ctx->tensors[i];
        STEP_MIX(m->view.numel);
        STEP_MIX(m->view.shape.rank);
        for (u32 d = 0; d < m->view.shape.rank && d < MAX_DIM; d++)
            STEP_MIX(m->view.shape.dims[d]);
        STEP_MIX(m->requires_grad);
        STEP_MIX(m->creator_op);
        STEP_MIX(m->src_ids[0]);
        STEP_MIX(m->src_ids[1]);
    }
    #undef STEP_MIX
    return h;
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

static int thvm_file_has_substr(const char *path, const char *needle) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    size_t nlen = strlen(needle);
    if (nlen == 0) { fclose(f); return 1; }
    char *buf = (char *)malloc(nlen);
    if (!buf) { fclose(f); return 0; }
    size_t fill = 0;
    int ch;
    int found = 0;
    while ((ch = fgetc(f)) != EOF) {
        if (fill < nlen) {
            buf[fill++] = (char)ch;
            if (fill < nlen) continue;
        } else {
            memmove(buf, buf + 1, nlen - 1);
            buf[nlen - 1] = (char)ch;
        }
        if (memcmp(buf, needle, nlen) == 0) { found = 1; break; }
    }
    free(buf);
    fclose(f);
    return found;
}

static u32 thvm_step_top_arity(u32 ext) {
    if (ext == UOP_KERNEL) return 0;
    if (ext == UOP_WHERE || ext == UOP_IFZ) return 3;
    if (ext == UOP_GRAD) return 2;
    if (ext == UOP_DETACH) return 1;
    if (!is_binary(ext) && is_elementwise(ext)) return 1;
    return 2;
}

static u32 thvm_step_term_arity(Term t);
static int thvm_step_has_parent_ref(TinyHVM *ctx, Term target);

static int thvm_step_top_has_era_arg(TinyHVM *ctx, Term t, u64 *slot_out, Term *term_out) {
    if (term_tag(t) != TAG_TOP) return 0;
    u64 loc = term_val(t);
    u32 arity = thvm_step_top_arity(term_ext(t));
    for (u32 i = 0; i < arity; i++) {
        Term child = heap_read(ctx, loc + i);
        if (term_tag(child) == TAG_ERA) {
            if (slot_out) *slot_out = loc + i;
            if (term_out) *term_out = child;
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

static int thvm_step_term_maybe_active(TinyHVM *ctx, Term t) {
    u8 tag = term_tag(t);
    if (tag == TAG_TOP) {
        return term_ext(t) == UOP_GRAD ||
               thvm_step_top_has_era_arg(ctx, t, NULL, NULL) ||
               thvm_step_top_has_add_zero_arg(ctx, t, NULL, NULL);
    }
    if (tag == TAG_ERA) return term_val(t) != 0 && !thvm_step_has_parent_ref(ctx, t);
    if (tag == TAG_TEN || tag == TAG_NUM || tag == TAG_LAM || tag == TAG_SUP ||
        tag == TAG_BRI || tag == TAG_MAT || tag == TAG_ANY || tag == TAG_USP)
        return 0;
    return 1;
}

static int thvm_step_term_first_reachable_occurrence(TinyHVM *ctx, u64 h, Term t, const u8 *reach) {
    for (u64 i = 1; i < h; i++) {
        if (reach && !reach[i]) continue;
        if (ctx->heap[i] == t) return 0;
    }
    return 1;
}

static u32 thvm_step_term_arity(Term t) {
    u8 tag = term_tag(t);
    u32 ext = term_ext(t);
    switch (tag) {
        case TAG_TOP:
            if (ext == UOP_KERNEL) return 0;
            if (ext == UOP_WHERE || ext == UOP_IFZ) return 3;
            if (ext == UOP_GRAD) return 2;
            if (ext == UOP_DETACH) return 1;
            if (!is_binary(ext) && is_elementwise(ext)) return 1;
            return 2;
        case TAG_APP:
        case TAG_LAM:
        case TAG_BRI:
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

static void thvm_step_mark_reachable_slots(TinyHVM *ctx, Term root, u8 *reach) {
    if (!reach || ctx->heap_pos == 0) return;
    memset(reach, 0, (size_t)ctx->heap_pos);
    u8 *seen_slot = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    u8 *seen_dup  = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    u64 work_cap = ctx->heap_pos ? (ctx->heap_pos * 8) : 0;
    Term *work = work_cap ? (Term *)malloc(sizeof(Term) * (size_t)work_cap) : NULL;
    u64 wp = 0;
    #define STEP_PUSH(_tt) do { \
        if (work && wp < work_cap) work[wp++] = (_tt); \
    } while (0)

    if (!(term_tag(root) == TAG_ERA && term_val(root) == 0)) STEP_PUSH(root);
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) == TAG_ERA && term_val(ht) != 0) {
            reach[h] = 1;
            STEP_PUSH(ht);
        }
    }

    while (work && wp > 0) {
        Term tt = work[--wp];
        u8 tg = term_tag(tt);
        u64 tv = term_val(tt);

        if (tg == TAG_DP0 || tg == TAG_DP1) {
            u64 dl = tv;
            if (dl == 0 || dl >= ctx->heap_pos || (seen_dup && seen_dup[dl])) continue;
            if (seen_dup) seen_dup[dl] = 1;
            reach[dl] = 1;
            STEP_PUSH(heap_read(ctx, dl));
            continue;
        }

        if (tg == TAG_ERA) {
            if (tv == 0 || tv >= ctx->heap_pos) continue;
            if (!seen_slot || !seen_slot[tv]) {
                if (seen_slot) seen_slot[tv] = 1;
                reach[tv] = 1;
                STEP_PUSH(heap_read(ctx, tv));
            }
            continue;
        }

        u32 ar = thvm_step_term_arity(tt);
        for (u32 i = 0; i < ar; i++) {
            u64 p = tv + i;
            if (p < ctx->heap_pos && (!seen_slot || !seen_slot[p])) {
                if (seen_slot) seen_slot[p] = 1;
                reach[p] = 1;
                STEP_PUSH(heap_read(ctx, p));
            }
        }
    }

    free(work);
    free(seen_dup);
    free(seen_slot);
    #undef STEP_PUSH
}

static int thvm_step_slot_is_rendered_parent_arg(TinyHVM *ctx, u64 slot) {
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term p = ctx->heap[h];
        u8 tag = term_tag(p);
        if (tag == TAG_TOP) {
            u64 loc = term_val(p);
            u32 arity = 2;
            if (term_ext(p) == UOP_WHERE || term_ext(p) == UOP_IFZ) arity = 3;
            if (term_ext(p) == UOP_KERNEL) arity = 0;
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

static int thvm_step_has_parent_ref(TinyHVM *ctx, Term target) {
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term p = ctx->heap[h];
        u32 arity = thvm_step_term_arity(p);
        u64 loc = term_val(p);
        if (arity == 0 || loc == 0 || loc + arity > ctx->heap_pos) continue;
        for (u32 i = 0; i < arity; i++) {
            if (heap_read(ctx, loc + i) == target) return 1;
        }
    }
    return 0;
}

static int thvm_step_candidate_visible(TinyHVM *ctx, u64 slot, Term t) {
    u8 tag = term_tag(t);
    if (tag == TAG_ERA) {
        if (term_val(t) == 0 || term_val(t) >= ctx->heap_pos) return 0;
        return 1;
    }
    if (tag == TAG_DP0 || tag == TAG_DP1) {
        if (slot == term_val(t)) return 1;
        return thvm_step_slot_is_rendered_parent_arg(ctx, slot);
    }
    if (tag == TAG_TOP) {
        if (!(t == step_graph_root_term || thvm_step_has_parent_ref(ctx, t)))
            return 0;
        if (thvm_step_top_has_era_arg(ctx, t, NULL, NULL))
            return 1;
        u64 loc = term_val(t);
        u32 arity = thvm_step_top_arity(term_ext(t));
        for (u32 i = 0; i < arity; i++) {
            Term c = heap_read(ctx, loc + i);
            if (term_tag(c) == TAG_ERA && term_val(c) == 0) continue;
            if (term_ext(t) == UOP_GRAD && term_tag(c) == TAG_ANY) continue;
            return 1;
        }
        return 0;
    }
    return 1;
}

static int thvm_step_graph_highlight_from_before(TinyHVM *ctx, u64 source_slot, Term before,
                                                 u64 *out_slot, Term *out_term) {
    u8 tag = term_tag(before);
    if (tag == TAG_CTR) {
        u64 loc = term_val(before);
        if (loc == 0 || loc >= ctx->heap_pos) return 0;
        *out_slot = loc + 0;
        *out_term = heap_read(ctx, loc + 0);
        return 1;
    }
    if (tag == TAG_APP) {
        u64 loc = term_val(before);
        if (loc == 0 || loc >= ctx->heap_pos) return 0;
        *out_slot = loc + 0;
        *out_term = heap_read(ctx, loc + 0);
        return (*out_slot != 0 && *out_slot < ctx->heap_pos);
    }
    if (tag == TAG_DP0 || tag == TAG_DP1) {
        *out_slot = term_val(before);
        *out_term = before;
        return (*out_slot != 0 && *out_slot < ctx->heap_pos);
    }
    if (tag == TAG_ERA) {
        u64 el = term_val(before);
        if (el == 0 || el >= ctx->heap_pos) return 0;
        *out_slot = el;
        *out_term = before;
        return 1;
    }
    if (tag == TAG_TOP) {
        u64 loc = term_val(before);
        if (loc >= ctx->heap_pos) return 0;
        if (term_ext(before) == UOP_GRAD) {
            Term y = heap_read(ctx, loc + 0);
            if (term_tag(y) == TAG_DP0 || term_tag(y) == TAG_DP1) {
                *out_slot = term_val(y);
                *out_term = y;
                return (*out_slot != 0 && *out_slot < ctx->heap_pos);
            }
            *out_slot = loc + 0;
            *out_term = y;
            return 1;
        }
        if (thvm_step_top_has_era_arg(ctx, before, out_slot, out_term))
            return (*out_slot != 0 && *out_slot < ctx->heap_pos);
        if (thvm_step_top_has_add_zero_arg(ctx, before, out_slot, out_term))
            return (*out_slot != 0 && *out_slot < ctx->heap_pos);
        *out_slot = loc + 0;
        *out_term = heap_read(ctx, loc + 0);
        return (*out_slot != 0 && *out_slot < ctx->heap_pos);
    }
    if (source_slot != 0 && source_slot < ctx->heap_pos) {
        *out_slot = source_slot;
        *out_term = before;
        return 1;
    }
    return 0;
}

static int thvm_step_graph_find_next_interaction(TinyHVM *ctx, u64 *out_slot, Term *out_term,
                                                 Term *out_before) {
    u64 source_slot = 0;
    Term before = 0;
    if (!thvm_phase1_find_next_actual(ctx, step_graph_root_term, &source_slot, &before))
        return 0;
    if (out_before) *out_before = before;
    return thvm_step_graph_highlight_from_before(ctx, source_slot, before, out_slot, out_term);
}

static void thvm_step_graph_eval_begin(TinyHVM *ctx, Term root) {
    if (!getenv("THVM_STEP_GRAPH") || step_graph_active) return;
    const char *step_graph_dir = thvm_step_graph_dir();
    step_graph_root_term = root;
    step_graph_before_grad_y = 0;
    step_graph_before_era_payload = 0;
    step_graph_before_top_had_era = 0;
    step_graph_before_top_had_add_zero = 0;
    snprintf(step_graph_last_prev_name, sizeof(step_graph_last_prev_name), "%s", "state_init");
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s && rm -f %s/*.dot %s/*.png",
             step_graph_dir, step_graph_dir, step_graph_dir);
    system(cmd);
    char p0[256];
    snprintf(p0, sizeof(p0), "%s/step_%03u_state_init.dot", step_graph_dir, 0u);
    u64 hs = 0; Term ht = 0; Term next_before = 0;
    char next_name[96] = {0};
    if (thvm_step_graph_find_next_interaction(ctx, &hs, &ht, &next_before)) {
        thvm_step_graph_interaction_name(ctx, next_before, next_name, sizeof(next_name));
        thvm_heap_dot_set_highlight(hs, ht);
    } else {
        thvm_heap_dot_set_highlight(0, 0);
    }
    thvm_heap_dot_set_step_meta("state_init", next_name[0] ? next_name : "");
    thvm_heap_dot_root(ctx, p0, root);
    step_graph_last_sig = thvm_step_graph_sig(ctx);
    {
        char p0s[256];
        snprintf(p0s, sizeof(p0s), "%s/.tmp_step_struct.dot", step_graph_dir);
        thvm_heap_dot_set_highlight(0, 0);
        thvm_heap_dot_set_step_meta("", "");
        thvm_heap_dot_root(ctx, p0s, root);
        step_graph_last_dot_sig = thvm_file_sig(p0s);
        remove(p0s);
        if (thvm_step_graph_find_next_interaction(ctx, &hs, &ht, &next_before)) {
            thvm_step_graph_interaction_name(ctx, next_before, next_name, sizeof(next_name));
            thvm_heap_dot_set_highlight(hs, ht);
        } else {
            thvm_heap_dot_set_highlight(0, 0);
            next_name[0] = '\0';
        }
        thvm_heap_dot_set_step_meta("state_init", next_name);
    }
    step_graph_n = 1;
    step_graph_active = 1;
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
        case TAG_OP2: return "OP2";
        case TAG_USP: return "USP";
        case TAG_UDP: return "UDP";
        case TAG_EQL: return "EQL";
        case TAG_AND: return "AND";
        case TAG_OR:  return "OR";
        case TAG_MAT: return "MAT";
        case TAG_ANY: return "ANY";
        default: return "UNK";
    }
}

static const char *thvm_step_graph_interaction_name(TinyHVM *ctx, Term before,
                                                     char *buf, size_t nbuf) {
    u8 tag = term_tag(before);
    if (tag == TAG_TOP) {
        u32 ext = term_ext(before);
        if (ext == UOP_GRAD) {
            Term y = step_graph_before_grad_y;
            if (y == 0) {
                u64 gl = term_val(before);
                y = heap_read(ctx, gl);
            }
            if (term_tag(y) == TAG_DP0 || term_tag(y) == TAG_DP1)
                snprintf(buf, nbuf, "interact_grad_on_DUP");
            else if (term_tag(y) == TAG_TOP && term_ext(y) < UOP_COUNT)
                snprintf(buf, nbuf, "interact_grad_on_%s", uop_names[term_ext(y)]);
            else if (term_tag(y) == TAG_TEN)
                snprintf(buf, nbuf, "interact_grad_on_TEN");
            else
                snprintf(buf, nbuf, "interact_grad");
            return buf;
        }
        if (ext < UOP_COUNT) {
            if (step_graph_before_top_had_add_zero ||
                thvm_step_top_has_add_zero_arg(ctx, before, NULL, NULL))
                snprintf(buf, nbuf, "interact_%s", uop_names[ext]);
            else if (thvm_step_top_has_era_arg(ctx, before, NULL, NULL))
                snprintf(buf, nbuf, "interact_era_on_%s", uop_names[ext]);
            else if (step_graph_before_top_had_era)
                snprintf(buf, nbuf, "interact_era_on_%s", uop_names[ext]);
            else
                snprintf(buf, nbuf, "interact_%s", uop_names[ext]);
            return buf;
        }
    }
    if (tag == TAG_ERA) {
        Term payload = step_graph_before_era_payload;
        if (payload == 0) {
            u64 el = term_val(before);
            payload = (el > 0 && el < ctx->heap_pos) ? heap_read(ctx, el) : term_era();
        }
        if (term_tag(payload) == TAG_TOP && term_ext(payload) < UOP_COUNT) {
            snprintf(buf, nbuf, "interact_era_on_%s", uop_names[term_ext(payload)]);
        } else {
            snprintf(buf, nbuf, "interact_era_on_%s", thvm_step_tag_name_short(term_tag(payload)));
        }
        return buf;
    }
    snprintf(buf, nbuf, "interact_%s", thvm_step_tag_name_short(tag));
    return buf;
}

static void thvm_step_graph_after_interaction(TinyHVM *ctx, Term before, Term root) {
    if (!getenv("THVM_STEP_GRAPH") || !step_graph_active) return;
    if (step_graph_n >= 1000) return;
    const char *step_graph_dir = thvm_step_graph_dir();
    step_graph_root_term = root;
    u64 sig = thvm_step_graph_sig(ctx);
    if (sig == step_graph_last_sig) return;
    char prev_name[96];
    thvm_step_graph_interaction_name(ctx, before, prev_name, sizeof(prev_name));
    step_graph_before_grad_y = 0;
    step_graph_before_era_payload = 0;
    step_graph_before_top_had_era = 0;
    step_graph_before_top_had_add_zero = 0;
    char tmp_struct[256];
    snprintf(tmp_struct, sizeof(tmp_struct), "%s/.tmp_step_struct.dot", step_graph_dir);
    thvm_heap_dot_set_highlight(0, 0);
    thvm_heap_dot_set_step_meta("", "");
    thvm_heap_dot_root(ctx, tmp_struct, root);
    u64 dot_sig = thvm_file_sig(tmp_struct);
    if (dot_sig == step_graph_last_dot_sig) {
        step_graph_last_sig = sig;
        remove(tmp_struct);
        step_graph_before_grad_y = 0;
        step_graph_before_era_payload = 0;
        step_graph_before_top_had_era = 0;
        return;
    }
    remove(tmp_struct);
    snprintf(step_graph_last_prev_name, sizeof(step_graph_last_prev_name), "%s", prev_name);

    char tmp[256];
    snprintf(tmp, sizeof(tmp), "%s/.tmp_step.dot", step_graph_dir);
    u64 hs = 0; Term ht = 0; Term next_before = 0;
    char next_name[96] = {0};
    if (!thvm_step_graph_find_next_interaction(ctx, &hs, &ht, &next_before)) {
        // No further visible interaction from this state: emit explicit final state.
        thvm_heap_dot_set_highlight(0, 0);
        thvm_heap_dot_set_step_meta(prev_name, "");
        thvm_heap_dot_root(ctx, tmp, root);
        char p_final[256];
        snprintf(p_final, sizeof(p_final), "%s/step_%03u_state_final.dot", step_graph_dir, step_graph_n);
        rename(tmp, p_final);
        step_graph_n++;
        step_graph_last_sig = sig;
        step_graph_last_dot_sig = dot_sig;
        step_graph_before_grad_y = 0;
        step_graph_before_era_payload = 0;
        step_graph_before_top_had_era = 0;
        step_graph_before_top_had_add_zero = 0;
        return;
    }
    thvm_step_graph_interaction_name(ctx, next_before, next_name, sizeof(next_name));
    thvm_heap_dot_set_highlight(hs, ht);
    thvm_heap_dot_set_step_meta(prev_name, next_name);
    thvm_heap_dot_root(ctx, tmp, root);
    if (!thvm_heap_dot_highlight_was_drawn() || !thvm_file_has_substr(tmp, "#cc0000")) {
        // If candidate selection found a detached/non-rendered interaction,
        // keep this as a plain state transition instead of dropping the step.
        char p_state[256];
        snprintf(p_state, sizeof(p_state), "%s/step_%03u_state_no_highlight.dot",
                 step_graph_dir, step_graph_n);
        rename(tmp, p_state);
        step_graph_n++;
        step_graph_last_sig = sig;
        step_graph_last_dot_sig = dot_sig;
        step_graph_before_grad_y = 0;
        step_graph_before_era_payload = 0;
        step_graph_before_top_had_add_zero = 0;
        return;
    }
    char p[256];
    snprintf(p, sizeof(p), "%s/step_%03u_%s.dot", step_graph_dir, step_graph_n, prev_name);
    rename(tmp, p);
    step_graph_n++;
    step_graph_last_sig = sig;
    step_graph_last_dot_sig = dot_sig;
    step_graph_before_grad_y = 0;
    step_graph_before_era_payload = 0;
    step_graph_before_top_had_era = 0;
    step_graph_before_top_had_add_zero = 0;
}

static void thvm_step_graph_finalize(TinyHVM *ctx) {
    if (!step_graph_active || !getenv("THVM_STEP_GRAPH")) return;
    const char *step_graph_dir = thvm_step_graph_dir();
    u64 sig = thvm_step_graph_sig(ctx);
    char tmp_struct[256];
    snprintf(tmp_struct, sizeof(tmp_struct), "%s/.tmp_step_struct.dot", step_graph_dir);
    thvm_heap_dot_set_highlight(0, 0);
    thvm_heap_dot_set_step_meta("", "");
    thvm_heap_dot_root(ctx, tmp_struct, step_graph_root_term);
    u64 dot_sig = thvm_file_sig(tmp_struct);
    remove(tmp_struct);
    if (dot_sig != step_graph_last_dot_sig) {
        char tmp[256];
        snprintf(tmp, sizeof(tmp), "%s/.tmp_step.dot", step_graph_dir);
        u64 hs = 0; Term ht = 0; Term next_before = 0;
        if (thvm_step_graph_find_next_interaction(ctx, &hs, &ht, &next_before))
            thvm_heap_dot_set_highlight(hs, ht);
        else
            thvm_heap_dot_set_highlight(0, 0);
        thvm_heap_dot_set_step_meta(step_graph_last_prev_name, "");
        thvm_heap_dot_root(ctx, tmp, step_graph_root_term);
        char p[256];
        snprintf(p, sizeof(p), "%s/step_%03u_state_final.dot", step_graph_dir, step_graph_n);
        rename(tmp, p);
        step_graph_n++;
        step_graph_last_dot_sig = dot_sig;
    }
    step_graph_last_sig = sig;
    if (!getenv("THVM_STEP_GRAPH_NO_PNG")) {
        char cmd[256];
        snprintf(cmd, sizeof(cmd),
                 "for f in %s/*.dot; do dot -Tpng -Gdpi=150 \"$f\" -o \"${f%%.dot}.png\" 2>/dev/null; done",
                 step_graph_dir);
        system(cmd);
    }
    fprintf(stderr, "Step graphs (%u steps) → %s/\n", step_graph_n, step_graph_dir);
    thvm_heap_dot_set_highlight(0, 0);
    thvm_heap_dot_set_step_meta("", "");
    step_graph_before_grad_y = 0;
    step_graph_before_era_payload = 0;
    step_graph_active = 0;
    step_graph_last_prev_name[0] = '\0';
    step_graph_n = 0;
}
