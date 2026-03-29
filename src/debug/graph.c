// graph.c — BFS heap walker for inet graph visualization
// Returns node/edge arrays suitable for WL Graph construction.

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
            if (ext == UOP_WHERE) return 3;
            return 2;
        case TAG_CTR: return ext; // arity in ext field
        // Leaves:
        case TAG_TEN: case TAG_ERA: case TAG_NUM:
        case TAG_VAR: case TAG_REF:
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
