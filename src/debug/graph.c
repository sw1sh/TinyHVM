// graph.c — BFS heap walker for inet graph visualization
// Returns node/edge arrays suitable for WL Graph construction.
static int heap_dot_root_only = 0; // skip global ERA/ASSIGN seeding in step graphs
static u64 heap_dot_node_hl = 0;  // node highlight (red border when edge hl fails)
static void thvm_heap_dot(TinyHVM *ctx, const char *path);
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root);
static void thvm_heap_dot_set_highlight(u64 slot, Term term);
static int thvm_heap_dot_highlight_was_drawn(void);
static void thvm_heap_dot_set_step_meta(const char *prev_name, const char *next_name);
static int thvm_step_find_next_actual(TinyHVM *ctx, Term root, u64 *out_source_slot, Term *out_before);
static u64 thvm_step_redex_source_slot(TinyHVM *ctx, u64 container_slot, Term container, Term before);
static const char *thvm_step_tag_name_short(u8 tag);
static const char *thvm_step_graph_interaction_name(TinyHVM *ctx, Term before,
                                                    char *buf, size_t nbuf);
static const char *thvm_step_graph_interaction_name_current(TinyHVM *ctx, Term before,
                                                            char *buf, size_t nbuf);
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
static Term step_graph_root_term = 0;
static Term step_graph_before_grad_y = 0;
static Term step_graph_before_era_payload = 0;
static int step_graph_before_top_had_era = 0;
static int step_graph_before_top_had_add_zero = 0;
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
    return step_graph_last_prev_name;
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

static void thvm_step_graph_set_root(Term root) {
    step_graph_root_term = root;
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

static void thvm_step_graph_rewrite_meta(const char *path, const char *prev_name, const char *next_name) {
    if (!path || !path[0]) return;
    FILE *in = fopen(path, "rb");
    if (!in) return;
    char tmp[320];
    snprintf(tmp, sizeof(tmp), "%s.meta.tmp", path);
    FILE *out = fopen(tmp, "wb");
    if (!out) { fclose(in); return; }
    char line[2048];
    while (fgets(line, sizeof(line), in)) {
        if (strncmp(line, "  // PREV_INTERACTION:", 22) == 0 ||
            strncmp(line, "// PREV_INTERACTION:", 21) == 0) {
            fprintf(out, "  // PREV_INTERACTION: %s\n", prev_name ? prev_name : "");
        } else if (strncmp(line, "  // NEXT_INTERACTION:", 22) == 0 ||
                   strncmp(line, "// NEXT_INTERACTION:", 21) == 0) {
            fprintf(out, "  // NEXT_INTERACTION: %s\n", next_name ? next_name : "");
        } else {
            fputs(line, out);
        }
    }
    fclose(in);
    fclose(out);
    rename(tmp, path);
}

static void thvm_step_graph_prune_isolated_tensor_nodes(const char *path) {
    if (!path || !path[0]) return;
    FILE *in = fopen(path, "rb");
    if (!in) return;
    fseek(in, 0, SEEK_END);
    long n = ftell(in);
    if (n <= 0) { fclose(in); return; }
    rewind(in);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(in); return; }
    size_t nr = fread(buf, 1, (size_t)n, in);
    fclose(in);
    buf[nr] = '\0';
    char tmp[320];
    snprintf(tmp, sizeof(tmp), "%s.tensor.tmp", path);
    FILE *out = fopen(tmp, "wb");
    if (!out) { free(buf); return; }
    char *cursor = buf;
    while (*cursor) {
        char *eol = strchr(cursor, '\n');
        size_t len = eol ? (size_t)(eol - cursor) : strlen(cursor);
        char line[2048];
        size_t copy_len = len < sizeof(line) - 1 ? len : sizeof(line) - 1;
        memcpy(line, cursor, copy_len);
        line[copy_len] = '\0';
        u32 tid = 0;
        int keep = 1;
        if (sscanf(line, "  t%u [", &tid) == 1) {
            char out_pat[32], in_pat[32];
            snprintf(out_pat, sizeof(out_pat), "t%u ->", tid);
            snprintf(in_pat, sizeof(in_pat), "-> t%u", tid);
            if (!strstr(buf, out_pat) && !strstr(buf, in_pat)) keep = 0;
        }
        if (keep) {
            fputs(line, out);
            fputc('\n', out);
        }
        if (!eol) break;
        cursor = eol + 1;
    }
    fclose(out);
    rename(tmp, path);
    free(buf);
}

static void thvm_step_graph_keep_only_root_tensor_result(const char *path, u32 root_tid) {
    if (!path || !path[0]) return;
    FILE *in = fopen(path, "rb");
    if (!in) return;
    fseek(in, 0, SEEK_END);
    long n = ftell(in);
    if (n <= 0) { fclose(in); return; }
    rewind(in);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(in); return; }
    size_t nr = fread(buf, 1, (size_t)n, in);
    fclose(in);
    buf[nr] = '\0';

    char tmp[320];
    snprintf(tmp, sizeof(tmp), "%s.final.tmp", path);
    FILE *out = fopen(tmp, "wb");
    if (!out) { free(buf); return; }

    char *cursor = buf;
    while (*cursor) {
        char *eol = strchr(cursor, '\n');
        size_t len = eol ? (size_t)(eol - cursor) : strlen(cursor);
        char line[2048];
        size_t copy_len = len < sizeof(line) - 1 ? len : sizeof(line) - 1;
        memcpy(line, cursor, copy_len);
        line[copy_len] = '\0';

        int keep = 1;
        u32 tid = 0, src_tid = 0, dst_tid = 0;
        if (sscanf(line, "  t%u -> rootout_t%u ", &src_tid, &dst_tid) == 2) {
            keep = (src_tid == root_tid && dst_tid == root_tid);
        } else if (strncmp(line, "  node [", 8) == 0 ||
                   strncmp(line, "  edge [", 8) == 0 ||
                   strncmp(line, "  graph [", 9) == 0) {
            keep = 1;
        } else if (strstr(line, "->")) {
            keep = 0;
        } else if (sscanf(line, "  t%u [", &tid) == 1) {
            keep = (tid == root_tid);
        } else if (sscanf(line, "  rootout_t%u [", &tid) == 1) {
            keep = (tid == root_tid);
        } else if (strchr(line, '[') && strncmp(line, "digraph ", 8) != 0) {
            keep = 0;
        }

        if (keep) {
            fputs(line, out);
            fputc('\n', out);
        }
        if (!eol) break;
        cursor = eol + 1;
    }
    fclose(out);
    rename(tmp, path);
    free(buf);
}

static u32 thvm_step_top_arity(u32 ext) {
    return thvm_uop_visible_arity(ext);
}

static u32 thvm_step_term_arity(Term t);
static int thvm_step_has_parent_ref(TinyHVM *ctx, Term target);

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
    if (tag == TAG_ERA) {
        Term payload = step_graph_before_era_payload;
        if (payload == 0 || (term_tag(payload) == TAG_ERA && term_val(payload) == 0)) {
            u64 el = term_val(before);
            payload = (el > 0 && el < ctx->heap_pos) ? thvm_era_payload(ctx, heap_read(ctx, el))
                                                     : term_era();
        }
        return payload;
    }
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

static const char *thvm_step_graph_display_name(TinyHVM *ctx, u64 source_slot, Term before,
                                                u64 hl_slot, Term hl_term,
                                                char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return "";
    buf[0] = '\0';
    Term focus = before;
    u64 focus_slot = source_slot;
    Term visible_peer = hl_term;
    if (hl_slot != 0 && hl_slot < ctx->heap_pos)
        visible_peer = heap_read(ctx, hl_slot);
    if (visible_peer != 0 && term_tag(visible_peer) == TAG_ERA) {
        focus = visible_peer;
        focus_slot = hl_slot ? hl_slot : source_slot;
        snprintf(buf, nbuf, "ERA");
    } else {
        if (hl_slot != 0 && hl_slot < ctx->heap_pos) {
            Term parent = thvm_step_parent_term_for_slot(ctx, hl_slot);
            if (parent != 0) {
                focus = parent;
                focus_slot = hl_slot;
            }
        }
        thvm_step_graph_interaction_name_current_at(ctx, focus_slot, focus, buf, nbuf);
    }
    thvm_step_graph_append_focus_suffix(ctx, focus_slot, focus, hl_slot, visible_peer, buf, nbuf);
    return buf;
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
        if (loc == 0 || loc + 1 >= ctx->heap_pos) return 0;
        if (source_slot >= loc && source_slot <= loc + 1) {
            *out_slot = source_slot;
            *out_term = heap_read(ctx, source_slot);
            return 1;
        }
        *out_slot = loc + 0;
        *out_term = heap_read(ctx, loc + 0);
        return 1;
    }
    if (tag == TAG_SEQ) {
        u64 loc = term_val(before);
        if (loc == 0 || loc + 1 >= ctx->heap_pos) return 0;
        if (source_slot >= loc && source_slot <= loc + 1) {
            *out_slot = source_slot;
            *out_term = heap_read(ctx, source_slot);
            return 1;
        }
        *out_slot = loc + 0;
        *out_term = heap_read(ctx, loc + 0);
        return 1;
    }
    if (tag == TAG_DP0 || tag == TAG_DP1) {
        *out_slot = term_val(before);
        *out_term = before;
        return (*out_slot != 0 && *out_slot < ctx->heap_pos);
    }
    if (tag == TAG_ERA) {
        Term want = step_graph_before_era_payload;
        if (term_tag(want) == TAG_ERA && term_val(want) == 0) want = 0;
        if (term_tag(want) == TAG_VAR) {
            u64 vloc = term_val(want);
            if (vloc > 0 && vloc < ctx->heap_pos) {
                Term et = heap_read(ctx, vloc);
                if (term_tag(et) == TAG_ERA && term_val(et) != 0) {
                    *out_slot = vloc;
                    *out_term = et;
                    return 1;
                }
            }
        }
        if (source_slot != 0 && source_slot < ctx->heap_pos) {
            Term et = heap_read(ctx, source_slot);
            if (term_tag(et) == TAG_ERA && term_val(et) != 0) {
                *out_slot = source_slot;
                *out_term = et;
                return 1;
            }
        }
        // Prefer matching by payload in post-state when the ERA stays in place
        // but its outgoing edge changed.
        if (want != 0) {
            for (u64 h = 1; h < ctx->heap_pos; h++) {
                Term et = ctx->heap[h];
                if (term_tag(et) != TAG_ERA) continue;
                u64 ev = term_val(et);
                if (ev == 0 || ev >= ctx->heap_pos) continue;
                Term payload = thvm_era_payload(ctx, heap_read(ctx, ev));
                if (payload == want) {
                    *out_slot = h;
                    *out_term = et;
                    return 1;
                }
            }
        }
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            if (ctx->heap[h] == before && term_tag(ctx->heap[h]) == TAG_ERA) {
                *out_slot = h;
                *out_term = ctx->heap[h];
                return 1;
            }
        }
        return 0;
    }
    if (tag == TAG_VAR) {
        // Prefer the binder/input edge feeding this VAR node. This is the
        // actual input being substituted/erased; consumer edges are fallback.
        u64 var_loc = term_val(before);
        if (source_slot != 0 && source_slot < ctx->heap_pos) {
            Term src = heap_read(ctx, source_slot);
            if (term_tag(src) == TAG_VAR && term_val(src) == var_loc) {
                for (u64 h = 1; h < ctx->heap_pos; h++) {
                    Term agent = ctx->heap[h];
                    if (term_tag(agent) == TAG_ERA && term_val(agent) == source_slot) {
                        *out_slot = h;
                        *out_term = agent;
                        return 1;
                    }
                }
            }
            if (term_tag(src) == TAG_ERA && term_val(src) != 0) {
                for (u64 h = 1; h < ctx->heap_pos; h++) {
                    if (h == source_slot) continue;
                    if (ctx->heap[h] == src) {
                        *out_slot = h;
                        *out_term = src;
                        return 1;
                    }
                }
                *out_slot = source_slot;
                *out_term = src;
                return 1;
            }
        }
        if (var_loc > 0 && var_loc < ctx->heap_pos) {
            Term sub = heap_read(ctx, var_loc);
            if (term_tag(sub) == TAG_ERA && term_val(sub) != 0) {
                for (u64 h = 1; h < ctx->heap_pos; h++) {
                    if (h == var_loc) continue;
                    if (ctx->heap[h] == sub) {
                        *out_slot = h;
                        *out_term = sub;
                        return 1;
                    }
                }
            }
            if (!term_is_sub(sub) || (term_tag(sub) == TAG_ERA && term_val(sub) != 0)) {
                *out_slot = var_loc;
                *out_term = before;
                return 1;
            }
        }
        // Otherwise highlight the actual parent-child edge where this VAR is consumed.
        u64 deferred_ifz_slot = 0;
        if (source_slot != 0 && source_slot < ctx->heap_pos) {
            if (heap_read(ctx, source_slot) == before) {
                if (!thvm_step_slot_is_ifz_cond(ctx, source_slot)) {
                    *out_slot = source_slot;
                    *out_term = before;
                    return 1;
                } else if (deferred_ifz_slot == 0) {
                    deferred_ifz_slot = source_slot;
                }
            }
            Term src = heap_read(ctx, source_slot);
            u64 ploc = term_val(src);
            u32 ar = thvm_step_term_arity(src);
            if (ploc > 0 && ploc + ar <= ctx->heap_pos) {
                for (u32 i = 0; i < ar; i++) {
                    if (heap_read(ctx, ploc + i) == before) {
                        u64 cand = ploc + i;
                        if (!thvm_step_slot_is_ifz_cond(ctx, cand)) {
                            *out_slot = cand;
                            *out_term = before;
                            return 1;
                        } else if (deferred_ifz_slot == 0) {
                            deferred_ifz_slot = cand;
                        }
                    }
                }
            }
        }
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            if (heap_read(ctx, h) == before) {
                if (!thvm_step_slot_is_ifz_cond(ctx, h)) {
                    if (!thvm_step_slot_is_rendered_parent_arg(ctx, h)) continue;
                    *out_slot = h;
                    *out_term = before;
                    return 1;
                } else if (deferred_ifz_slot == 0) {
                    deferred_ifz_slot = h;
                }
            }
        }
        if (deferred_ifz_slot != 0 && deferred_ifz_slot < ctx->heap_pos) {
            *out_slot = deferred_ifz_slot;
            *out_term = before;
            return 1;
        }
        if (var_loc > 0 && var_loc < ctx->heap_pos) {
            *out_slot = var_loc;
            *out_term = before;
            return 1;
        }
        if (out_slot) *out_slot = 0;
        return 0;
    }
    if (tag == TAG_REF) {
        // REF unfolds when entered by APP — find the slot.
        {
            if (source_slot != 0 && source_slot < ctx->heap_pos &&
                heap_read(ctx, source_slot) == before) {
                *out_slot = source_slot;
                *out_term = before;
                return 1;
            }
            // Walk APP chain from root to find the slot
            Term _walk = step_graph_root_term;
            for (int _depth = 0; _depth < 32; _depth++) {
                if (term_tag(_walk) != TAG_APP) break;
                u64 _aloc = term_val(_walk);
                if (_aloc == 0 || _aloc + 1 >= ctx->heap_pos) break;
                Term _fun = heap_read(ctx, _aloc);
                if (_fun == before) {
                    *out_slot = _aloc;
                    *out_term = before;
                    return 1;
                }
                _walk = _fun;
            }
        }
        // Once the REF is consumed, the slot still identifies the rewritten
        // APP fun position even though the current term there is no longer REF.
        if (source_slot != 0 && source_slot < ctx->heap_pos) {
            *out_slot = source_slot;
            *out_term = heap_read(ctx, source_slot);
            return 1;
        }
        // Fallback: find an exact heap slot containing this REF.
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            if (heap_read(ctx, h) == before) {
                *out_slot = h;
                *out_term = before;
                return 1;
            }
        }
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
        *out_slot = loc;
        *out_term = heap_read(ctx, *out_slot);
        return (*out_slot != 0 && *out_slot < ctx->heap_pos);
    }
    if (source_slot != 0 && source_slot < ctx->heap_pos) {
        *out_slot = source_slot;
        *out_term = before;
        return 1;
    }
    return 0;
}

static int thvm_step_graph_highlight_from_current_before(TinyHVM *ctx, u64 source_slot, Term before,
                                                         u64 *out_slot, Term *out_term) {
    if (term_tag(before) == TAG_ERA) {
        if (source_slot != 0 && source_slot < ctx->heap_pos) {
            Term cur = heap_read(ctx, source_slot);
            if (term_tag(cur) == TAG_ERA && cur == before) {
                *out_slot = source_slot;
                *out_term = before;
                return 1;
            }
        }
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            if (ctx->heap[h] == before && term_tag(ctx->heap[h]) == TAG_ERA) {
                *out_slot = h;
                *out_term = before;
                return 1;
            }
        }
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term cur = ctx->heap[h];
            if (term_tag(cur) == TAG_ERA && term_val(cur) == term_val(before) && term_val(cur) != 0) {
                *out_slot = h;
                *out_term = cur;
                return 1;
            }
        }
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term agent = ctx->heap[h];
            if (term_tag(agent) != TAG_ERA || term_val(agent) == 0 || term_val(agent) >= ctx->heap_pos) continue;
            if (heap_read(ctx, term_val(agent)) == before) {
                *out_slot = h;
                *out_term = agent;
                return 1;
            }
        }
        return 0;
    }
    return thvm_step_graph_highlight_from_before(ctx, source_slot, before, out_slot, out_term);
}

static int thvm_step_graph_find_next_interaction(TinyHVM *ctx, u64 *out_slot, Term *out_term,
                                                 u64 *out_source_slot, Term *out_before) {
    u64 source_slot = 0;
    Term before = 0;
    if (!thvm_step_find_next_actual(ctx, step_graph_root_term, &source_slot, &before))
        return 0;
    Term container = step_graph_root_term;
    if (source_slot != step_root_slot && source_slot < ctx->heap_pos)
        container = heap_read(ctx, source_slot);
    u64 graph_source_slot = thvm_step_redex_source_slot(ctx, source_slot, container, before);
    if (out_source_slot) *out_source_slot = graph_source_slot;
    if (out_before) *out_before = before;
    return thvm_step_graph_highlight_from_current_before(ctx, graph_source_slot, before, out_slot, out_term);
}

static int thvm_step_graph_prefer_kernel_child_fuse(TinyHVM *ctx, u64 source_slot, Term before,
                                                    u64 *out_hs, Term *out_ht,
                                                    u64 *out_source_slot, Term *out_before) {
    if (term_tag(before) != TAG_TOP) return 0;
    u32 uop = term_ext(before);
    if (uop != UOP_FUSE) return 0;
    if (source_slot == 0 || source_slot >= ctx->heap_pos) return 0;
    Term after = heap_read(ctx, source_slot);
    if (term_tag(after) != TAG_TOP || term_ext(after) != UOP_KERNEL) return 0;
    u64 kloc = term_val(after);
    if (kloc == 0 || kloc + 1 >= ctx->heap_pos) return 0;
    for (u32 ai = 0; ai < 2; ai++) {
        u64 child_slot = kloc + ai;
        Term child = heap_read(ctx, child_slot);
        if (term_tag(child) != TAG_TOP) continue;
        u32 cuop = term_ext(child);
        if (cuop != UOP_FUSE) continue;
        u64 hs = 0;
        Term ht = 0;
        if (!thvm_step_graph_highlight_from_before(ctx, child_slot, child, &hs, &ht))
            continue;
        if (out_hs) *out_hs = hs;
        if (out_ht) *out_ht = ht;
        if (out_source_slot) *out_source_slot = child_slot;
        if (out_before) *out_before = child;
        return 1;
    }
    return 0;
}

static void thvm_step_graph_eval_begin(TinyHVM *ctx, Term root) {
    if (!getenv("THVM_STEP_GRAPH") || step_graph_active) return;
    const char *step_graph_dir = thvm_step_graph_dir();
    step_graph_root_term = root;
    step_graph_lower_anchor_name[0] = '\0';
    step_graph_lower_anchor_index = 0;
    thvm_step_alo_substs_clear(ctx);
    step_graph_before_grad_y = 0;
    step_graph_before_era_payload = 0;
    step_graph_before_top_had_era = 0;
    step_graph_before_top_had_add_zero = 0;
    step_graph_last_prev_name[0] = '\0';
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s && rm -f %s/step_*.dot %s/step_*.png %s/.tmp_step*.dot",
             step_graph_dir, step_graph_dir, step_graph_dir, step_graph_dir);
    system(cmd);
    char tmp0[256];
    snprintf(tmp0, sizeof(tmp0), "%s/.tmp_step_init.dot", step_graph_dir);
    u64 hs = 0; Term ht = 0; u64 next_source_slot = 0; Term next_before = 0;
    char shown_name[160] = {0};
    if (thvm_step_graph_find_next_interaction(ctx, &hs, &ht, &next_source_slot, &next_before)) {
        thvm_step_graph_display_name(ctx, next_source_slot, next_before, hs, ht,
                                     shown_name, sizeof(shown_name));
        thvm_heap_dot_set_highlight(hs, ht);
    } else {
        thvm_heap_dot_set_highlight(0, 0);
    }
    char p0[384];
    thvm_heap_dot_set_step_meta(shown_name[0] ? shown_name : "state_init", "pending");
    heap_dot_root_only = 1;
    thvm_heap_dot_root(ctx, tmp0, root);
    heap_dot_root_only = 0;
    step_graph_last_sig = thvm_step_graph_sig(ctx);
    {
        char p0s[256];
        snprintf(p0s, sizeof(p0s), "%s/.tmp_step_struct.dot", step_graph_dir);
        thvm_heap_dot_set_highlight(0, 0);
        thvm_heap_dot_set_step_meta("", "");
        heap_dot_root_only = 1;
        thvm_heap_dot_root(ctx, p0s, root);
        heap_dot_root_only = 0;
        step_graph_last_dot_sig = thvm_file_sig(p0s);
        remove(p0s);
        if (thvm_step_graph_find_next_interaction(ctx, &hs, &ht, &next_source_slot, &next_before)) {
            thvm_step_graph_display_name(ctx, next_source_slot, next_before, hs, ht,
                                         shown_name, sizeof(shown_name));
            thvm_heap_dot_set_highlight(hs, ht);
        } else {
            thvm_heap_dot_set_highlight(0, 0);
            shown_name[0] = '\0';
        }
        thvm_heap_dot_set_step_meta(shown_name[0] ? shown_name : "state_init", "pending");
    }
    snprintf(p0, sizeof(p0), "%s/step_%03u_%s.dot", step_graph_dir, 0u,
             shown_name[0] ? shown_name : "state_init");
    rename(tmp0, p0);
    thvm_step_graph_prune_isolated_tensor_nodes(p0);
    thvm_step_graph_rewrite_meta(p0, shown_name[0] ? shown_name : "state_init", "");
    snprintf(step_graph_last_file, sizeof(step_graph_last_file), "%s", p0);
    snprintf(step_graph_last_prev_name, sizeof(step_graph_last_prev_name), "%s",
             shown_name[0] ? shown_name : "state_init");
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

static const char *thvm_step_graph_interaction_name(TinyHVM *ctx, Term before,
                                                     char *buf, size_t nbuf) {
    u8 tag = term_tag(before);
    if (tag == TAG_TOP) {
        u32 ext = term_ext(before);
        if (ext == UOP_GRAD) {
            snprintf(buf, nbuf, "GRAD");
            return buf;
        }
        if (ext < UOP_COUNT) {
            if (thvm_step_top_has_era_arg(ctx, before, NULL, NULL) || step_graph_before_top_had_era)
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

static const char *thvm_step_graph_interaction_name_current(TinyHVM *ctx, Term before,
                                                            char *buf, size_t nbuf) {
    return thvm_step_graph_interaction_name_current_at(ctx, 0, before, buf, nbuf);
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

static void thvm_step_graph_after_interaction(TinyHVM *ctx, u64 source_slot, Term before, Term root) {
    if (!getenv("THVM_STEP_GRAPH") || !step_graph_active) return;
    if (step_graph_n >= thvm_step_graph_max_steps()) {
        thvm_step_alo_substs_clear(ctx);
        return;
    }
    const char *step_graph_dir = thvm_step_graph_dir();
    step_graph_root_term = root;
    u64 sig = thvm_step_graph_sig(ctx);
    if (sig == step_graph_last_sig) {
        thvm_step_alo_substs_clear(ctx);
        return;
    }
    char tmp_struct[256];
    snprintf(tmp_struct, sizeof(tmp_struct), "%s/.tmp_step_struct.dot", step_graph_dir);
    thvm_heap_dot_set_highlight(0, 0);
    thvm_heap_dot_set_step_meta("", "");
    heap_dot_root_only = 1;
    thvm_heap_dot_root(ctx, tmp_struct, root);
    heap_dot_root_only = 0;
    u64 dot_sig = thvm_file_sig(tmp_struct);
    if (dot_sig == step_graph_last_dot_sig) {
        step_graph_last_sig = sig;
        remove(tmp_struct);
        step_graph_before_grad_y = 0;
        step_graph_before_era_payload = 0;
        step_graph_before_top_had_era = 0;
        thvm_step_alo_substs_clear(ctx);
        return;
    }
    remove(tmp_struct);
    char tmp[384];
    snprintf(tmp, sizeof(tmp), "%s/.tmp_step.dot", step_graph_dir);
    u64 hs = 0; Term ht = 0; u64 next_source_slot = 0; Term next_before = 0;
    u64 next_hs = 0; Term next_ht = 0;
    char shown_name[160] = {0};
    u64 shown_source_slot = source_slot;
    Term shown_before = before;
    // Highlight the NEXT interaction (forward-looking): the snapshot is
    // the post-fire state, so the edge of the redex about to fire still
    // exists. Prev (what just fired) is by definition consumed and often
    // can't be drawn — use it only as a fallback for terminal snapshots.
    int have_next = thvm_step_graph_find_next_interaction(ctx, &next_hs, &next_ht, &next_source_slot, &next_before);
    int have_prev_hl = 0;
    if (have_next) {
        hs = next_hs;
        ht = next_ht;
        shown_source_slot = next_source_slot;
        shown_before = next_before;
        have_prev_hl = 1;
    } else {
        have_prev_hl = thvm_step_graph_highlight_from_before(ctx, source_slot, before, &hs, &ht);
        if (!have_prev_hl &&
            thvm_step_graph_prefer_kernel_child_fuse(ctx, source_slot, before,
                                                     &hs, &ht, &shown_source_slot, &shown_before)) {
            have_prev_hl = 1;
        }
    }
    thvm_heap_dot_set_highlight(have_prev_hl ? hs : 0, have_prev_hl ? ht : 0);
    thvm_heap_dot_set_step_meta("pending", "pending");
    heap_dot_root_only = 1;
    thvm_heap_dot_root(ctx, tmp, root);
    heap_dot_root_only = 0;
    // Edge highlight is the only acceptable highlight for step graphs.
    // If it couldn't be drawn, leave the snapshot un-highlighted rather
    // than falling back to a node-highlight (which looks like a different
    // kind of event and confuses readers).
    int rendered_highlight = thvm_heap_dot_highlight_was_drawn() && thvm_file_has_substr(tmp, "#cc0000");
    if (!rendered_highlight) {
        // Highlight couldn't be drawn (pre-fire slot already mutated). Keep
        // the snapshot anyway with no highlight so the step isn't dropped —
        // the shown_name still captures what just fired.
    }
    thvm_step_graph_display_name(ctx, shown_source_slot, shown_before, hs, ht,
                                 shown_name, sizeof(shown_name));
    if (!shown_name[0]) snprintf(shown_name, sizeof(shown_name), "state_no_highlight");
    // If the interaction produced a tensor result, this is the terminal
    // state — name the file accordingly so finalize() can skip writing
    // a redundant state_final next to it.
    if (term_tag(root) == TAG_TEN)
        snprintf(shown_name, sizeof(shown_name), "state_final");
    char p[384];
    snprintf(p, sizeof(p), "%s/step_%03u_%s.dot", step_graph_dir, step_graph_n, shown_name);
    rename(tmp, p);
    if (term_tag(root) != TAG_TEN)
        thvm_step_graph_prune_isolated_tensor_nodes(p);
    thvm_step_graph_rewrite_meta(p, shown_name, "");
    if (step_graph_last_file[0])
        thvm_step_graph_rewrite_meta(step_graph_last_file, step_graph_last_prev_name, shown_name);
    snprintf(step_graph_last_file, sizeof(step_graph_last_file), "%s", p);
    snprintf(step_graph_last_prev_name, sizeof(step_graph_last_prev_name), "%s", shown_name);
    step_graph_n++;
    step_graph_last_sig = sig;
    step_graph_last_dot_sig = dot_sig;
    step_graph_before_grad_y = 0;
    step_graph_before_era_payload = 0;
    step_graph_before_top_had_era = 0;
    step_graph_before_top_had_add_zero = 0;
    thvm_step_alo_substs_clear(ctx);
}

static void thvm_step_graph_finalize(TinyHVM *ctx) {
    if (!step_graph_active || !getenv("THVM_STEP_GRAPH")) return;
    const char *step_graph_dir = thvm_step_graph_dir();
    u64 sig = thvm_step_graph_sig(ctx);
    if (term_tag(step_graph_root_term) == TAG_TOP && term_ext(step_graph_root_term) == UOP_KERNEL) {
        u64 loc = term_val(step_graph_root_term);
        u64 hl_slot = (loc > 0 && loc < ctx->heap_pos) ? loc : 0;
        Term hl_term = hl_slot ? heap_read(ctx, hl_slot) : 0;
        thvm_step_graph_display_name(ctx, step_root_slot, step_graph_root_term,
                                     hl_slot, hl_term,
                                     step_graph_lower_anchor_name, sizeof(step_graph_lower_anchor_name));
        step_graph_lower_anchor_index = step_graph_n;
    } else if (step_graph_last_prev_name[0]) {
        snprintf(step_graph_lower_anchor_name, sizeof(step_graph_lower_anchor_name),
                 "%s", step_graph_last_prev_name);
        step_graph_lower_anchor_index = step_graph_n > 0 ? step_graph_n - 1 : 0;
    }
    char tmp_struct[256];
    snprintf(tmp_struct, sizeof(tmp_struct), "%s/.tmp_step_struct.dot", step_graph_dir);
    thvm_heap_dot_set_highlight(0, 0);
    thvm_heap_dot_set_step_meta("", "");
    heap_dot_root_only = 1;
    thvm_heap_dot_root(ctx, tmp_struct, step_graph_root_term);
    heap_dot_root_only = 0;
    u64 dot_sig = thvm_file_sig(tmp_struct);
    remove(tmp_struct);
    // If the last per-interaction snapshot already captured this state,
    // don't write a redundant state_final duplicate.
    if (dot_sig == step_graph_last_dot_sig) {
        (void)sig;
        goto finalize_rewrite_last;
    }
    {
        char tmp[384];
        snprintf(tmp, sizeof(tmp), "%s/.tmp_step.dot", step_graph_dir);
        thvm_heap_dot_set_highlight(0, 0);
        thvm_heap_dot_set_step_meta(step_graph_last_prev_name, "pending");
        heap_dot_root_only = 1;
        thvm_heap_dot_root(ctx, tmp, step_graph_root_term);
        heap_dot_root_only = 0;
        char p[384];
        snprintf(p, sizeof(p), "%s/step_%03u_state_final.dot", step_graph_dir, step_graph_n);
        rename(tmp, p);
        if (term_tag(step_graph_root_term) == TAG_TEN)
            thvm_step_graph_keep_only_root_tensor_result(p, (u32)term_val(step_graph_root_term));
        else
            thvm_step_graph_prune_isolated_tensor_nodes(p);
        thvm_step_graph_rewrite_meta(step_graph_last_file, step_graph_last_prev_name, "");
        snprintf(step_graph_last_file, sizeof(step_graph_last_file), "%s", p);
        step_graph_n++;
        step_graph_last_dot_sig = dot_sig;
    }
finalize_rewrite_last:
    step_graph_last_sig = sig;
    if (!getenv("THVM_STEP_GRAPH_NO_PNG")) {
        char cmd[256];
        snprintf(cmd, sizeof(cmd),
                 "for f in %s/step_*.dot; do dot -Tpng -Gdpi=150 \"$f\" -o \"${f%%.dot}.png\" 2>/dev/null; done",
                 step_graph_dir);
        system(cmd);
    }
    fprintf(stderr, "Step graphs (%u steps) → %s/\n", step_graph_n, step_graph_dir);
    thvm_heap_dot_set_highlight(0, 0);
    thvm_heap_dot_set_step_meta("", "");
    thvm_step_alo_substs_clear(ctx);
    step_graph_before_grad_y = 0;
    step_graph_before_era_payload = 0;
    step_graph_before_top_had_era = 0;
    step_graph_before_top_had_add_zero = 0;
    step_graph_active = 0;
    step_graph_last_prev_name[0] = '\0';
    step_graph_last_file[0] = '\0';
    step_graph_n = 0;
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
