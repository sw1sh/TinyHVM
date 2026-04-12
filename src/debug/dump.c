// debug/dump.c — Dump tensor provenance graph as DOT (graphviz) or JSON
// Usage: thvm_dump_dot(ctx, "graph.dot") or thvm_dump_json(ctx, "graph.json")
// Visualize: dot -Tpng graph.dot -o graph.png

#include <stdio.h>

// Step-graph overlay: highlight one predicted next interaction edge.
static int heap_dot_hl_on = 0;
static u64 heap_dot_hl_slot = 0;
static Term heap_dot_hl_term = 0;
static int heap_dot_hl_hit = 0;
static char heap_dot_prev_name[96] = {0};
static char heap_dot_next_name[96] = {0};
static int heap_dot_include_sched_kernels = 0;
static void thvm_heap_dot_set_highlight(u64 slot, Term term) {
    heap_dot_hl_on = (slot != 0);
    heap_dot_hl_slot = slot;
    heap_dot_hl_term = term;
    heap_dot_hl_hit = 0;
}
static int thvm_heap_dot_highlight_was_drawn(void) { return heap_dot_hl_hit; }
static void thvm_heap_dot_set_step_meta(const char *prev_name, const char *next_name) {
    snprintf(heap_dot_prev_name, sizeof(heap_dot_prev_name), "%s", prev_name ? prev_name : "");
    snprintf(heap_dot_next_name, sizeof(heap_dot_next_name), "%s", next_name ? next_name : "");
}
static void thvm_heap_dot_set_sched_kernels(int enabled) {
    heap_dot_include_sched_kernels = enabled ? 1 : 0;
}

static int dot_shape_is_valid(Shape s) {
    if (s.rank > MAX_DIM) return 0;
    for (u32 i = 0; i < s.rank; i++) {
        if (s.dims[i] == 0) return 0;
    }
    return 1;
}

static u32 dot_tensor_planned_slot(TinyHVM *ctx, u32 tid) {
    if (!ctx || tid >= ctx->tensor_count) return 0;
    TensorMeta *m = &ctx->tensors[tid];
    if (m->planned_slot) return m->planned_slot;
    extern KernelEntry sched_kernels[];
    extern u32 sched_kernel_count;
    for (u32 kid = 0; kid < sched_kernel_count; kid++) {
        KernelEntry *ke = &sched_kernels[kid];
        if (ke->output_tid == tid || ke->raw_output_tid == tid)
            return ke->output_slot;
    }
    return 0;
}

static const char *dot_backend_short(TinyHVM *ctx, Backend *backend) {
    if (!backend && ctx) backend = ctx_default_backend(ctx);
    if (!backend) return "?";
    if (ctx && ctx->backends[THVM_DEV_CPU] == backend) return "cpu";
    if (ctx && ctx->backends[THVM_DEV_METAL] == backend) return "mtl";
    return "dev";
}

static const char *dot_kernel_backend(TinyHVM *ctx, const KernelEntry *ke) {
    if (!ke) return "?";
    if (ke->output_tid && ke->output_tid < ctx->tensor_count) {
        Backend *bk = ctx->tensors[ke->output_tid].backend;
        if (bk) return dot_backend_short(ctx, bk);
    }
    if (ke->raw_output_tid && ke->raw_output_tid < ctx->tensor_count) {
        Backend *bk = ctx->tensors[ke->raw_output_tid].backend;
        if (bk) return dot_backend_short(ctx, bk);
    }
    return dot_backend_short(ctx, ctx_default_backend(ctx));
}

static int dot_shape_broadcast(Shape a, Shape b, Shape *out) {
    u32 r = a.rank > b.rank ? a.rank : b.rank;
    if (r == 0 || r > MAX_DIM) return 0;
    Shape s = {.rank = r};
    for (u32 i = 0; i < r; i++) {
        int ia = (int)a.rank - 1 - (int)i;
        int ib = (int)b.rank - 1 - (int)i;
        u32 da = ia >= 0 ? a.dims[ia] : 1;
        u32 db = ib >= 0 ? b.dims[ib] : 1;
        u32 d = 0;
        if (da == db) d = da;
        else if (da == 1) d = db;
        else if (db == 1) d = da;
        else return 0;
        s.dims[r - 1 - i] = d;
    }
    *out = s;
    return 1;
}

static const char *dot_uop_port_name(u32 uop, u32 argi) {
    if (uop == UOP_SUM || uop == UOP_RMAX) return argi == 0 ? "in" : "axes";
    if (uop == UOP_GRAD) return argi == 0 ? "y" : "gy";
    if (is_view_op(uop)) return argi == 0 ? "in" : "shape";
    if (is_binary(uop) || uop == UOP_CMP || uop == UOP_MM) return argi == 0 ? "a" : "b";
    if (uop == UOP_LOG_PRINT) return "in";
    if (uop == UOP_DETACH) return "in";
    return "in";
}

static int dot_visible_heap_loc_tag(u8 tag) {
    switch (tag) {
        case TAG_TOP:
        case TAG_APP:
        case TAG_LAM:
        case TAG_SUP:
        case TAG_BRI:
        case TAG_OP2:
        case TAG_USP:
        case TAG_EQL:
        case TAG_AND:
        case TAG_OR:
        case TAG_MAT:
        case TAG_ANN:
        case TAG_DSU:
        case TAG_DDU:
        case TAG_UDP:
        case TAG_CTR:
        case TAG_INC:
            return 1;
        default:
            return 0;
    }
}

static const char *dot_heap_tag_name(u8 tag) {
    switch (tag) {
        case TAG_APP: return "APP";
        case TAG_LAM: return "LAM";
        case TAG_SUP: return "SUP";
        case TAG_BRI: return "BRI";
        case TAG_OP2: return "OP2";
        case TAG_USP: return "USP";
        case TAG_EQL: return "EQL";
        case TAG_AND: return "AND";
        case TAG_OR:  return "OR";
        case TAG_MAT: return "MAT";
        case TAG_ANN: return "ANN";
        case TAG_DSU: return "DSU";
        case TAG_DDU: return "DDU";
        case TAG_UDP: return "UDP";
        case TAG_CTR: return "CTR";
        case TAG_REF: return "REF";
        case TAG_VAR: return "VAR";
        case TAG_INC: return "INC";
        default: return "?";
    }
}

static const char *dot_heap_node_shape(u8 tag) {
    switch (tag) {
        case TAG_APP: return "invtriangle";
        case TAG_SUP:
        case TAG_USP:
        case TAG_CTR: return "hexagon";
        case TAG_REF:
        case TAG_VAR: return "oval";
        default: return "box";
    }
}

static const char *dot_heap_node_color(u8 tag) {
    switch (tag) {
        case TAG_APP: return "#f3f3f3";
        case TAG_SUP:
        case TAG_USP: return "#e4d6fc";
        case TAG_MAT: return "#eaf2ff";
        case TAG_LAM:
        case TAG_BRI: return "#f2e8ff";
        case TAG_REF:
        case TAG_VAR: return "#eeeeee";
        default: return "#f0f0f0";
    }
}

static const char *dot_heap_port_name(u8 tag, u32 idx) {
    switch (tag) {
        case TAG_LAM:
        case TAG_BRI: return idx == 0 ? "var" : "body";
        case TAG_SUP:
        case TAG_USP: return idx == 0 ? "a" : "b";
        case TAG_MAT: return idx == 0 ? "ok" : "fb";
        case TAG_ANN: return idx == 0 ? "term" : "type";
        case TAG_DSU:
        case TAG_DDU: return idx == 0 ? "label" : (idx == 1 ? "a" : "b");
        case TAG_UDP:
        case TAG_INC: return "in";
        default: return idx == 0 ? "a" : "b";
    }
}

static void dot_kernel_leaf_label(TinyHVM *ctx, const KernelEntry *ke, u32 leaf_idx,
                                  char *buf, size_t bufsz) {
    if (bufsz == 0) return;
    buf[0] = '\0';
    for (u32 oi = 0; oi < ke->n_ops; oi++) {
        if (ke->ops[oi].arg_a == leaf_idx) {
            u32 u = ke->ops[oi].uop;
            snprintf(buf, bufsz, "%s:%s",
                     (u < UOP_COUNT) ? uop_names[u] : "?",
                     dot_uop_port_name(u, 0));
            return;
        }
        if (ke->ops[oi].arg_b == leaf_idx) {
            u32 u = ke->ops[oi].uop;
            snprintf(buf, bufsz, "%s:%s",
                     (u < UOP_COUNT) ? uop_names[u] : "?",
                     dot_uop_port_name(u, 1));
            return;
        }
    }
    if (ke->has_reduce && term_tag(ke->sum_term) == TAG_TOP) {
        Term axes = heap_read(ctx, term_val(ke->sum_term) + 1);
        if (term_tag(axes) == TAG_DP0 || term_tag(axes) == TAG_DP1)
            axes = heap_read(ctx, term_val(axes));
        if (ke->leaf_kinds[leaf_idx] == KERNEL_LEAF_TENSOR &&
            term_tag(axes) == TAG_TEN &&
            ke->leaf_ids[leaf_idx] == (u32)term_val(axes)) {
            snprintf(buf, bufsz, "%s:axes",
                     ke->has_reduce == UOP_SUM ? "SUM" : "RMAX");
            return;
        }
    }
    if (ke->leaf_kinds[leaf_idx] == KERNEL_LEAF_NUM) snprintf(buf, bufsz, "const");
    else snprintf(buf, bufsz, "leaf%u", leaf_idx);
}

static void dot_kernel_output_label(const KernelEntry *ke, char *buf, size_t bufsz) {
    if (bufsz == 0) return;
    if (term_tag(ke->original_term) == TAG_TOP) {
        u32 u = term_ext(ke->original_term);
        snprintf(buf, bufsz, "%s:out",
                 (u < UOP_COUNT) ? uop_names[u] : "out");
        return;
    }
    snprintf(buf, bufsz, "out");
}

static int dot_infer_top_shape(TinyHVM *ctx, u32 uop, u64 loc, Shape *out);

static int dot_term_shape(TinyHVM *ctx, Term t, Shape *out) {
    for (int d = 0; d < 16; d++) {
        u8 tag = term_tag(t);
        if (tag == TAG_DP0 || tag == TAG_DP1) {
            u64 dl = term_val(t);
            if (dl == 0 || dl >= ctx->heap_pos) return 0;
            t = heap_read(ctx, dl);
            continue;
        }
        if (tag == TAG_TEN) {
            u32 tid = (u32)term_val(t);
            if (tid < ctx->tensor_count) { *out = tensor_view_get(&ctx->tensors[tid])->shape; return 1; }
            return 0;
        }
        if (tag == TAG_TOP) {
            const View *v = st_get(term_val(t));
            if (v && dot_shape_is_valid(v->shape)) { *out = v->shape; return 1; }
            if (dot_infer_top_shape(ctx, term_ext(t), term_val(t), out) &&
                dot_shape_is_valid(*out)) return 1;
            return 0;
        }
        if (tag == TAG_NUM) { *out = SHAPE(1); return 1; }
        return 0;
    }
    return 0;
}

static int dot_meta_shape_from_tensor(TinyHVM *ctx, Term tmeta, Shape *out) {
    if (term_tag(tmeta) != TAG_TEN) return 0;
    u32 tid = (u32)term_val(tmeta);
    if (tid >= ctx->tensor_count) return 0;
    TensorMeta *m = &ctx->tensors[tid];
    u32 n = m->view.numel;
    if (n == 0 || n > MAX_DIM) return 0;
    u32 dims[MAX_DIM];
    if (!tensor_meta_read_u32(ctx, tid, dims, MAX_DIM)) return 0;
    Shape s = {.rank = n};
    for (u32 i = 0; i < n; i++) {
        s.dims[i] = dims[i];
        if (s.dims[i] == 0) return 0;
    }
    *out = s;
    return 1;
}

static int dot_infer_top_shape(TinyHVM *ctx, u32 uop, u64 loc, Shape *out) {
    Term a = heap_read(ctx, loc + 0);
    Term b = heap_read(ctx, loc + 1);
    Shape sa = SHAPE(1), sb = SHAPE(1);
    int has_a = dot_term_shape(ctx, a, &sa);
    int has_b = dot_term_shape(ctx, b, &sb);

    if (uop == UOP_RESHAPE || uop == UOP_EXPAND) {
        if (dot_meta_shape_from_tensor(ctx, b, out)) return 1;
        if (has_a) { *out = sa; return 1; }
        return 0;
    }
    if (uop == UOP_SUM || uop == UOP_RMAX) {
        if (!has_a) return 0;
        *out = sa;
        if (term_tag(b) == TAG_TEN) {
            u32 tid = (u32)term_val(b);
            if (tid < ctx->tensor_count) {
                TensorMeta *m = &ctx->tensors[tid];
                u32 n = m->view.numel;
                if (n <= MAX_DIM) {
                    u32 axes[MAX_DIM];
                    if (tensor_meta_read_u32(ctx, tid, axes, MAX_DIM)) {
                        for (u32 i = 0; i < n; i++) {
                            int ax = (int)axes[i];
                            if (ax >= 0 && ax < (int)out->rank) out->dims[ax] = 1;
                        }
                    }
                }
            }
        }
        return 1;
    }
    if (is_binary(uop) || uop == UOP_CMP) {
        if (has_a && has_b && dot_shape_broadcast(sa, sb, out)) return 1;
        if (has_a) { *out = sa; return 1; }
        if (has_b) { *out = sb; return 1; }
        return 0;
    }
    if (has_a) { *out = sa; return 1; }
    if (has_b) { *out = sb; return 1; }
    return 0;
}

static u64 dot_dup_canon_loc(TinyHVM *ctx, u64 dloc) {
    for (int depth = 0; depth < 64; depth++) {
        if (dloc == 0 || dloc >= ctx->heap_pos) break;
        Term dv = heap_read(ctx, dloc);
        u8 dt = term_tag(dv);
        if (dt != TAG_DP0 && dt != TAG_DP1) break;
        u64 up = term_val(dv);
        if (up == dloc || up == 0 || up >= ctx->heap_pos) break;
        dloc = up;
    }
    return dloc;
}

static u32 dot_term_arity(Term t) {
    u8 tag = term_tag(t);
    u32 ext = term_ext(t);
    switch (tag) {
        case TAG_TOP:
            if (ext == UOP_KERNEL) return 0;
            if (ext == UOP_WHERE || ext == UOP_IFZ) return 3;
            if (ext == UOP_GRAD) return 2;
            if (ext == UOP_LOG_PRINT) return 1;
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

static int dot_term_maybe_active(Term t) {
    u8 tag = term_tag(t);
    if (tag == TAG_TOP) return term_ext(t) == UOP_GRAD;
    if (tag == TAG_ERA) return term_val(t) != 0;
    return 0;
}

static void thvm_dump_dot(TinyHVM *ctx, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "dump_dot: can't open %s\n", path); return; }

    fprintf(f, "digraph G {\n");
    fprintf(f, "  graph [ordering=\"out\", nodesep=0.35, ranksep=0.55];\n");
    fprintf(f, "  rankdir=BT;\n");
    fprintf(f, "  node [shape=record, fontname=\"Courier\", fontsize=10];\n");
    fprintf(f, "  edge [fontsize=8];\n\n");

    for (u32 i = 0; i < ctx->tensor_count; i++) {
        TensorMeta *m = &ctx->tensors[i];
        if (m->view.numel == 0) continue;

        // Shape string
        char shape_str[128] = {0};
        int pos = 0;
        for (u32 d = 0; d < m->view.shape.rank && d < MAX_DIM; d++)
            pos += snprintf(shape_str + pos, sizeof(shape_str) - pos, "%s%u",
                           d > 0 ? "×" : "", m->view.shape.dims[d]);

        // Strides string
        char stride_str[128] = {0};
        pos = 0;
        for (u32 d = 0; d < m->view.shape.rank && d < MAX_DIM; d++)
            pos += snprintf(stride_str + pos, sizeof(stride_str) - pos, "%s%d",
                           d > 0 ? "," : "", m->view.strides[d]);

        // Color
        const char *color = "white";
        u32 planned_slot = dot_tensor_planned_slot(ctx, i);
        int planned = (m->creator_op == UOP_KERNEL && m->buf_id == 0 && planned_slot != 0);
        if (m->requires_grad) color = "#e8f4e8";
        if (m->view.has_mask) color = "#fff0e0";
        if (!m->creator_op && m->requires_grad) color = "#e0e8ff"; // param
        if (planned) color = "#d9f2e6";

        const char *op = (m->creator_op < UOP_COUNT) ? uop_names[m->creator_op] : "?";

        char slot_str[32] = {0};
        if (planned) snprintf(slot_str, sizeof(slot_str), "|slot=%u planned", planned_slot);
        else if (planned_slot) snprintf(slot_str, sizeof(slot_str), "|slot=%u", planned_slot);

        fprintf(f, "  t%u [label=\"{t%u|%s %s|buf=%u off=%d%s|strides=[%s]%s}\", "
                   "style=filled, fillcolor=\"%s\"];\n",
                i, i, m->creator_op ? op : "LEAF", shape_str,
                m->buf_id, m->view.offset,
                slot_str,
                stride_str,
                m->view.has_mask ? " MASKED" : "",
                color);

        // Edges from sources
        if (m->creator_op) {
            fprintf(f, "  t%u -> t%u [label=\"a\"];\n", m->src_ids[0], i);
            if (m->src_ids[1] && m->src_ids[1] < ctx->tensor_count) {
                TensorMeta *mb = &ctx->tensors[m->src_ids[1]];
                // Skip small metadata tensors (shapes, axes, pairs)
                if (mb->view.numel > MAX_DIM)
                    fprintf(f, "  t%u -> t%u [label=\"b\"];\n", m->src_ids[1], i);
            }
        }
    }

    fprintf(f, "}\n");
    fclose(f);
    fprintf(stderr, "dump_dot: wrote %u tensors to %s\n", ctx->tensor_count, path);
}

// Dump heap as DOT — flat walk, no BFS. Every combinator shown faithfully.
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root);
static void thvm_heap_dot(TinyHVM *ctx, const char *path) {
    thvm_heap_dot_root(ctx, path, term_era());
}
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "heap_dot: can't open %s\n", path); return; }
    fprintf(f, "digraph G {\n");
    if (heap_dot_prev_name[0]) fprintf(f, "  // PREV_INTERACTION: %s\n", heap_dot_prev_name);
    if (heap_dot_next_name[0]) fprintf(f, "  // NEXT_INTERACTION: %s\n", heap_dot_next_name);
    fprintf(f, "  rankdir=BT;\n");
    fprintf(f, "  node [fontname=\"Helvetica\", fontsize=10, style=filled, shape=box, margin=\"0.1,0.05\"];\n");
    fprintf(f, "  edge [fontsize=8, fontname=\"Helvetica\"];\n\n");

    // Track emitted tensor nodes (avoid duplicates)
    u8 ten_emitted[256]; memset(ten_emitted, 0, sizeof(ten_emitted));
    // Helper: emit tensor node if not yet emitted
    #define EMIT_TEN(tid) do { \
        if ((tid)<256 && !ten_emitted[tid] && (tid)<ctx->tensor_count) { \
            ten_emitted[tid]=1; TensorMeta *_m=&ctx->tensors[tid]; \
            char _sh[64]=""; int _p=0; \
            for (u32 _d=0;_d<_m->view.shape.rank;_d++) \
                _p+=snprintf(_sh+_p,sizeof(_sh)-_p,"%s%u",_d?",":"",_m->view.shape.dims[_d]); \
            const char *_dt=dtype_name(_m->dtype); \
            const char *_bk=_m->backend?((_m->backend==ctx->backends[0])?"cpu":"mtl"):"?"; \
            u32 _pslot = dot_tensor_planned_slot(ctx, (tid)); \
            int _planned = (_m->creator_op == UOP_KERNEL && _m->buf_id == 0 && _pslot != 0); \
            const char *_fc=_planned ? "#d9f2e6" : (_m->requires_grad?"#ffe0e0":"#e0e0e0"); \
            char _slot[32]=""; \
            if (_pslot) snprintf(_slot, sizeof(_slot), "\\nslot%u%s", _pslot, _planned ? " planned" : ""); \
            fprintf(f,"  t%u [label=\"t%u\\n[%s]\\n%s %s%s%s\",shape=box,fillcolor=\"%s\"];\n", \
                    (tid),(tid),_sh,_dt,_bk,_m->requires_grad?" grad":"",_slot,_fc); \
        } } while(0)

    u32 nn = 0; // node count for stats
    // Track emitted node locations (dedup TAG_TOP by val)
    #define NODE_DEDUP_MAX 1024
    u64 node_emitted[NODE_DEDUP_MAX]; u32 n_node_emitted = 0;
    #define NODE_SEEN(v) ({ int _s=0; for(u32 _i=0;_i<n_node_emitted;_i++) \
        if(node_emitted[_i]==(v)){_s=1;break;} _s; })
    #define NODE_MARK(v) do { if(n_node_emitted<NODE_DEDUP_MAX) node_emitted[n_node_emitted++]=(v); } while(0)
    u8 *dp_slot_emitted = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    u8 *top_live = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    u8 *slot_live = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    u8 *loc_live = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    u8 *ctr_children_emitted = (u8 *)calloc((size_t)ctx->heap_pos, 1);
    #define DP_SLOT_MARK(pos) do { if ((pos) < ctx->heap_pos) dp_slot_emitted[(pos)] = 1; } while(0)
    #define DP_SLOT_SEEN(pos) (((pos) < ctx->heap_pos) ? dp_slot_emitted[(pos)] : 0)
    #define RAW_NODE_KEY(v) ((v) + 0x200000)
    #define CTR_NODE_KEY(v) ((v) + 0x300000)
    #define ANY_NODE_KEY(v) ((v) + 0x400000)
    #define REF_NODE_KEY(v) ((v) + 0x500000)
    #define VAR_NODE_KEY(v) ((v) + 0x600000)
    #define EMIT_RAW_NODE(hid, tterm) do { \
        if (!NODE_SEEN(RAW_NODE_KEY(hid))) { \
            NODE_MARK(RAW_NODE_KEY(hid)); \
            Term _rt = (tterm); \
            fprintf(f, "  h%llu [label=\"h%llu\\ntag=%u\", shape=box, fillcolor=\"#dddddd\", fontsize=8];\n", \
                    (unsigned long long)(hid), (unsigned long long)(hid), (u32)term_tag(_rt)); \
        } \
    } while(0)
    #define EMIT_CTR_NODE(cloc) do { \
        if (!NODE_SEEN(CTR_NODE_KEY(cloc))) { \
            NODE_MARK(CTR_NODE_KEY(cloc)); \
            fprintf(f, "  ctr%llu [label=\"CTR\", shape=hexagon, fillcolor=\"#f3f3f3\", fontsize=9];\n", \
                    (unsigned long long)(cloc)); \
        } \
    } while(0)
    #define EMIT_ANY_NODE(apos) do { \
        if (!NODE_SEEN(ANY_NODE_KEY(apos))) { \
            NODE_MARK(ANY_NODE_KEY(apos)); \
            fprintf(f, "  any%llu [label=\"ANY\", shape=oval, fillcolor=\"#eeeeee\", fontsize=8];\n", (unsigned long long)(apos)); \
        } \
    } while(0)
    #define EMIT_REF_NODE(slot, name) do { \
        if (!NODE_SEEN(REF_NODE_KEY(slot))) { \
            NODE_MARK(REF_NODE_KEY(slot)); \
            fprintf(f, "  ref%llu [label=\"REF\\n#%u\", shape=oval, fillcolor=\"#eeeeee\", fontsize=8];\n", \
                    (unsigned long long)(slot), (u32)(name)); \
        } \
    } while(0)
    #define EMIT_VAR_NODE(slot, loc, is_sub) do { \
        if (!NODE_SEEN(VAR_NODE_KEY(slot))) { \
            NODE_MARK(VAR_NODE_KEY(slot)); \
            fprintf(f, "  var%llu [label=\"VAR\\n@%llu%s\", shape=oval, fillcolor=\"#eeeeee\", fontsize=8];\n", \
                    (unsigned long long)(slot), (unsigned long long)(loc), (is_sub) ? " sub" : ""); \
        } \
    } while(0)
    #define SLOT_LIVE(pos) (((pos) < ctx->heap_pos) ? slot_live[(pos)] : 0)
    #define LOC_LIVE(pos)  (((pos) < ctx->heap_pos) ? loc_live[(pos)] : 0)

    // Precompute live TOP locations by traversing from active agents/root only.
    // This hides stale detached TOP cells left in dead heap regions.
    if (top_live) {
        u8 *seen_slot = (u8 *)calloc((size_t)ctx->heap_pos, 1);
        u8 *seen_dup  = (u8 *)calloc((size_t)ctx->heap_pos, 1);
        u64 work_cap = ctx->heap_pos ? (ctx->heap_pos * 8) : 0;
        Term *work = work_cap ? (Term *)malloc(sizeof(Term) * (size_t)work_cap) : NULL;
        u64 wp = 0;
        #define PUSH_TERM(_tt) do { \
            if (work && wp < work_cap) work[wp++] = (_tt); \
        } while (0)

        if (!(term_tag(root) == TAG_ERA && term_val(root) == 0))
            PUSH_TERM(root);
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            if (ctx->heap[h] == root) slot_live[h] = 1;
        }
        // Also seed explicit active ERA agents. GRAD interactions can drop
        // discarded metadata onto detached ERA components that still belong to
        // the literal heap state and must be visible/firable step-by-step.
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term ht = ctx->heap[h];
            if (term_tag(ht) == TAG_ERA && term_val(ht) != 0) {
                slot_live[h] = 1;
                PUSH_TERM(ht);
            }
            if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_ASSIGN) {
                slot_live[h] = 1;
                PUSH_TERM(ht);
            }
            if (heap_dot_include_sched_kernels &&
                term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_KERNEL) {
                slot_live[h] = 1;
                PUSH_TERM(ht);
            }
        }

        while (work && wp > 0) {
            Term tt = work[--wp];
            u8 tg = term_tag(tt);
            u64 tv = term_val(tt);
            if (dot_visible_heap_loc_tag(tg) && tv > 0 && tv < ctx->heap_pos)
                loc_live[tv] = 1;
            if (tg == TAG_TOP) {
                if (tv == 0 || tv >= ctx->heap_pos || top_live[tv]) continue;
                top_live[tv] = 1;
                u32 ar = dot_term_arity(tt);
                for (u32 i = 0; i < ar; i++) {
                    u64 p = tv + i;
                    if (p < ctx->heap_pos && (!seen_slot || !seen_slot[p])) {
                        if (seen_slot) seen_slot[p] = 1;
                        if (slot_live) slot_live[p] = 1;
                        PUSH_TERM(heap_read(ctx, p));
                    }
                }
                continue;
            }
            if (tg == TAG_DP0 || tg == TAG_DP1) {
                u64 dl = tv;
                if (dl == 0 || dl >= ctx->heap_pos || (seen_dup && seen_dup[dl])) continue;
                if (seen_dup) seen_dup[dl] = 1;
                PUSH_TERM(heap_read(ctx, dl));
                continue;
            }
            if (tg == TAG_ERA) {
                if (tv == 0 || tv >= ctx->heap_pos) continue;
                if (!seen_slot || !seen_slot[tv]) {
                    if (seen_slot) seen_slot[tv] = 1;
                    if (slot_live) slot_live[tv] = 1;
                    PUSH_TERM(heap_read(ctx, tv));
                }
                continue;
            }
            u32 ar = dot_term_arity(tt);
            for (u32 i = 0; i < ar; i++) {
                u64 p = tv + i;
                if (p < ctx->heap_pos && (!seen_slot || !seen_slot[p])) {
                    if (seen_slot) seen_slot[p] = 1;
                    if (slot_live) slot_live[p] = 1;
                    PUSH_TERM(heap_read(ctx, p));
                }
            }
        }

        free(work);
        free(seen_dup);
        free(seen_slot);
        #undef PUSH_TERM
    }
    #define DOT_HL_MATCH(pos) ({ \
        int _m = 0; \
        if (heap_dot_hl_on) { \
            u64 _p = (u64)(pos); \
            if (_p == heap_dot_hl_slot) { \
                _m = 1; \
            } else if (heap_dot_hl_slot < ctx->heap_pos) { \
                /* Back-compat for DP highlights: if slot points at a DP child \
                   entry, highlight the DUP principal edge (the DP's target loc). */ \
                Term _ht = heap_read(ctx, heap_dot_hl_slot); \
                if ((term_tag(_ht) == TAG_DP0 || term_tag(_ht) == TAG_DP1) && \
                    term_val(_ht) == _p) { \
                    _m = 1; \
                } \
            } \
        } \
        if (_m) heap_dot_hl_hit = 1; \
        _m; \
    })
    #define EDGE_HL_ONLY(pos) (DOT_HL_MATCH(pos) ? " [color=\"#cc0000\",penwidth=2.0]" : "")
    #define EDGE_HL_LABEL(pos) (DOT_HL_MATCH(pos) ? ",color=\"#cc0000\",penwidth=2.0" : "")
    #define HL_SLOT_TERM() ((heap_dot_hl_on && heap_dot_hl_slot < ctx->heap_pos) ? heap_read(ctx, heap_dot_hl_slot) : 0)
    #define HL_SLOT_IS_DP() ({ Term _ht = HL_SLOT_TERM(); term_tag(_ht) == TAG_DP0 || term_tag(_ht) == TAG_DP1; })
    #define DUP_PORT_HAS_VISIBLE_CONSUMER(_dloc, _ptag) ({ \
        int _has = 0; \
        for (u64 _hh = 1; _hh < ctx->heap_pos && !_has; _hh++) { \
            Term _pp = ctx->heap[_hh]; \
            u8 _tg = term_tag(_pp); \
            if (_tg == TAG_TOP) { \
                u64 _pl = term_val(_pp); \
                if (top_live && (_pl >= ctx->heap_pos || !top_live[_pl])) continue; \
                u32 _pa = dot_term_arity(_pp); \
                for (u32 _pi = 0; _pi < _pa; _pi++) { \
                    Term _ch = heap_read(ctx, _pl + _pi); \
                    if (term_tag(_ch) != (_ptag)) continue; \
                    u64 _dl = term_val(_ch); \
                    if (_dl == 0 || _dl >= ctx->heap_pos) continue; \
                    if (dot_dup_canon_loc(ctx, _dl) == (_dloc)) { _has = 1; break; } \
                } \
                continue; \
            } \
            if (_tg == TAG_ERA) { \
                continue; \
            } \
        } \
        _has; \
    })
    #define EMIT_DUP_CHAIN(dloc) do { \
        u64 _cur = (dloc); \
        for (int _depth = 0; _depth < 32; _depth++) { \
            if (_cur == 0 || _cur >= ctx->heap_pos) break; \
            int _hl_principal = DOT_HL_MATCH(_cur); \
            int _cur_new = !NODE_SEEN(_cur + 0x100000); \
            if (_cur_new) { \
                NODE_MARK(_cur + 0x100000); \
                fprintf(f, "  dup%llu [label=\"DUP\", shape=invtriangle, fillcolor=\"#d4b8e8\", fontsize=9, width=0.7, height=0.5];\n", (unsigned long long)_cur); \
            } \
            Term _shared = heap_read(ctx, _cur); \
            u8 _stag = term_tag(_shared); \
            if (_stag == TAG_DP0 || _stag == TAG_DP1) { \
                u64 _up = term_val(_shared); \
                if (_up == _cur) break; \
                if (_cur_new) \
                    fprintf(f, "  dup%llu -> dup%llu [label=\"%s\"];\n", \
                        (unsigned long long)_up, (unsigned long long)_cur, _stag == TAG_DP1 ? "dp1" : "dp0"); \
                _cur = _up; \
                continue; \
            } \
            if (_cur_new) { \
                if (_stag == TAG_TOP) fprintf(f, "  n%llu -> dup%llu%s;\n", (unsigned long long)term_val(_shared), (unsigned long long)_cur, _hl_principal ? " [color=\"#cc0000\",penwidth=2.0]" : ""); \
                else if (_stag == TAG_TEN) { EMIT_TEN((u32)term_val(_shared)); fprintf(f, "  t%u -> dup%llu%s;\n", (u32)term_val(_shared), (unsigned long long)_cur, _hl_principal ? " [color=\"#cc0000\",penwidth=2.0]" : ""); } \
                else if (_stag == TAG_CTR) { EMIT_CTR_NODE(term_val(_shared)); fprintf(f, "  ctr%llu -> dup%llu%s;\n", (unsigned long long)term_val(_shared), (unsigned long long)_cur, _hl_principal ? " [color=\"#cc0000\",penwidth=2.0]" : ""); } \
                else if (_stag == TAG_REF) { EMIT_REF_NODE(_cur, term_ext(_shared)); fprintf(f, "  ref%llu -> dup%llu%s;\n", (unsigned long long)_cur, (unsigned long long)_cur, _hl_principal ? " [color=\"#cc0000\",penwidth=2.0]" : ""); } \
                else if (_stag == TAG_VAR) { EMIT_VAR_NODE(_cur, term_val(_shared), term_is_sub(_shared)); fprintf(f, "  var%llu -> dup%llu%s;\n", (unsigned long long)_cur, (unsigned long long)_cur, _hl_principal ? " [color=\"#cc0000\",penwidth=2.0]" : ""); } \
                else if (dot_visible_heap_loc_tag(_stag)) { fprintf(f, "  n%llu -> dup%llu%s;\n", (unsigned long long)term_val(_shared), (unsigned long long)_cur, _hl_principal ? " [color=\"#cc0000\",penwidth=2.0]" : ""); } \
                else if (_stag == TAG_NUM) { f32 _fv; u32 _bv=(u32)term_val(_shared); memcpy(&_fv,&_bv,4); \
                    fprintf(f, "  num_d%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", (unsigned long long)_cur, (double)_fv); \
                    fprintf(f, "  num_d%llu -> dup%llu%s;\n", (unsigned long long)_cur, (unsigned long long)_cur, _hl_principal ? " [color=\"#cc0000\",penwidth=2.0]" : ""); } \
                else if (_stag == TAG_ANY) { EMIT_ANY_NODE(_cur); fprintf(f, "  any%llu -> dup%llu%s;\n", (unsigned long long)_cur, (unsigned long long)_cur, _hl_principal ? " [color=\"#cc0000\",penwidth=2.0]" : ""); } \
                else { EMIT_RAW_NODE(_cur, _shared); fprintf(f, "  h%llu -> dup%llu%s;\n", (unsigned long long)_cur, (unsigned long long)_cur, _hl_principal ? " [color=\"#cc0000\",penwidth=2.0]" : ""); } \
            } \
            break; \
        } \
    } while(0)
    #define EMIT_DUP_STUB(dloc) EMIT_DUP_CHAIN(dloc)
    #define ERA_DEDUP_MAX 1024
    u64 era_emitted[ERA_DEDUP_MAX]; u32 n_era_emitted = 0;
    #define ERA_SEEN(v) ({ int _s=0; for(u32 _i=0;_i<n_era_emitted;_i++) \
        if(era_emitted[_i]==(v)){_s=1;break;} _s; })
    #define ERA_MARK(v) do { if(n_era_emitted<ERA_DEDUP_MAX) era_emitted[n_era_emitted++]=(v); } while(0)
    #define EMIT_ERA_NODE(epos, eterm) do { \
        if (!ERA_SEEN(epos)) { \
            ERA_MARK(epos); \
            fprintf(f, "  era%llu [label=\"ERA\",shape=circle,width=0.38,height=0.38,fixedsize=true,fillcolor=\"#ffb8b8\",fontsize=7];\n", (unsigned long long)(epos)); \
            Term _et = (eterm); \
            u64 _ev = term_val(_et); \
            if (_ev == 0) break; \
            if (_ev < ctx->heap_pos) { \
                Term _src = heap_read(ctx, _ev); \
                u8 _st = term_tag(_src); \
                if (_st == TAG_DP0 || _st == TAG_DP1) { \
                    u64 _dl = term_val(_src); \
                    DP_SLOT_MARK(_ev); \
                    EMIT_DUP_CHAIN(_dl); \
                    fprintf(f, "  dup%llu -> era%llu [label=\"%s\"%s];\n", _dl, (unsigned long long)(epos), _st == TAG_DP1 ? "dp1" : "dp0", EDGE_HL_LABEL(_ev)); \
                } else if (_st == TAG_TOP) { \
                    fprintf(f, "  n%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)term_val(_src), (unsigned long long)(epos), EDGE_HL_LABEL(_ev)); \
                } else if (_st == TAG_TEN) { \
                    EMIT_TEN((u32)term_val(_src)); \
                    fprintf(f, "  t%u -> era%llu [label=\"p\"%s];\n", (u32)term_val(_src), (unsigned long long)(epos), EDGE_HL_LABEL(_ev)); \
                } else if (_st == TAG_CTR) { \
                    EMIT_CTR_NODE(term_val(_src)); \
                    fprintf(f, "  ctr%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)term_val(_src), (unsigned long long)(epos), EDGE_HL_LABEL(_ev)); \
                } else if (_st == TAG_REF) { \
                    EMIT_REF_NODE(_ev, term_ext(_src)); \
                    fprintf(f, "  ref%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)_ev, (unsigned long long)(epos), EDGE_HL_LABEL(_ev)); \
                } else if (_st == TAG_VAR) { \
                    EMIT_VAR_NODE(_ev, term_val(_src), term_is_sub(_src)); \
                    fprintf(f, "  var%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)_ev, (unsigned long long)(epos), EDGE_HL_LABEL(_ev)); \
                } else if (dot_visible_heap_loc_tag(_st)) { \
                    fprintf(f, "  n%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)term_val(_src), (unsigned long long)(epos), EDGE_HL_LABEL(_ev)); \
                } else if (_st == TAG_NUM) { \
                    f32 _fv; u32 _bv=(u32)term_val(_src); memcpy(&_fv,&_bv,4); \
                    fprintf(f, "  num_era%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", (unsigned long long)(epos), (double)_fv); \
                    fprintf(f, "  num_era%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)(epos), (unsigned long long)(epos), EDGE_HL_LABEL(_ev)); \
                } else if (_st == TAG_ANY) { \
                    EMIT_ANY_NODE(_ev); \
                    fprintf(f, "  any%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)_ev, (unsigned long long)(epos), EDGE_HL_LABEL(_ev)); \
                } else if (_st == TAG_ERA) { \
                    if (!ERA_SEEN(_ev)) { \
                        ERA_MARK(_ev); \
                        fprintf(f, "  era%llu [label=\"ERA\",shape=circle,width=0.38,height=0.38,fixedsize=true,fillcolor=\"#ffb8b8\",fontsize=7];\n", (unsigned long long)_ev); \
                    } \
                    fprintf(f, "  era%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)_ev, (unsigned long long)(epos), EDGE_HL_LABEL(_ev)); \
                } else { \
                    EMIT_RAW_NODE(_ev, _src); \
                    fprintf(f, "  h%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)_ev, (unsigned long long)(epos), EDGE_HL_LABEL(_ev)); \
                } \
            } \
        } \
    } while(0)

    if (term_tag(root) == TAG_TEN) {
        EMIT_TEN((u32)term_val(root));
    } else if (term_tag(root) == TAG_NUM) {
        f32 _fv; u32 _bv = (u32)term_val(root); memcpy(&_fv, &_bv, 4);
        fprintf(f, "  num_root [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n",
                (double)_fv);
    }

    // Flat walk: every heap position
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term t = ctx->heap[h];
        u8 tag = term_tag(t); u32 ext = term_ext(t); u64 val = term_val(t);

        // --- ERA ---
        if (tag == TAG_ERA) {
            if (val != 0 && val < ctx->heap_pos) {
                EMIT_ERA_NODE(h, t);
                nn++;
            }
            continue;
        }

        // --- TAG_TEN: only shown when referenced as child of a node ---
        if (tag == TAG_TEN) {
            // Do not emit standalone tensor heap cells; show tensors only where
            // they participate in explicit net edges.
            continue;
        }

        // --- DUP (DP0/DP1 references) ---
        if (tag == TAG_DP0 || tag == TAG_DP1) {
            DP_SLOT_MARK(h);
            nn++;
            continue;
        }

        // --- TAG_TOP: op nodes (dedup by val) ---
        if (tag == TAG_TOP) {
            if (top_live && (val >= ctx->heap_pos || !top_live[val])) continue;
            if (NODE_SEEN(val)) continue;
            NODE_MARK(val);
            char label[128]; const char *color = "#f0f0f0"; const char *nshape = "box";
            const char *opn = (ext < UOP_COUNT) ? uop_names[ext] : "?";
            const View *v = st_get(val);
            Shape shp = SHAPE(1);
            int has_shape = 0;
            if (v && dot_shape_is_valid(v->shape)) {
                shp = v->shape;
                has_shape = 1;
            } else if (dot_infer_top_shape(ctx, ext, val, &shp) &&
                       dot_shape_is_valid(shp)) {
                has_shape = 1;
            }
            char sh[64] = "?";
            if (has_shape) {
                int p = 0;
                if (shp.rank == 0) {
                    snprintf(sh, sizeof(sh), "1");
                } else {
                    sh[0] = '\0';
                    for (u32 d = 0; d < shp.rank && p < 50; d++)
                        p += snprintf(sh + p, sizeof(sh) - p, "%s%u", d ? "," : "", shp.dims[d]);
                }
            }
            if (ext == UOP_KERNEL) {
                Term kid_t = heap_read(ctx, val + 1);
                u32 kid = (term_tag(kid_t) == TAG_NUM) ? (u32)term_val(kid_t) : 0;
                extern KernelEntry sched_kernels[];
                KernelEntry *ke = &sched_kernels[kid];
                char ops_s[64] = ""; int p = 0;
                if (ke->has_reduce) { p += snprintf(ops_s+p,sizeof(ops_s)-p,"%s",ke->has_reduce==UOP_SUM?"SUM":"RMAX");
                    if (ke->n_ops) p += snprintf(ops_s+p,sizeof(ops_s)-p,"+"); }
                u8 seen_op[UOP_COUNT]={0};
                for (u32 oi=0; oi<ke->n_ops && p<50; oi++) { u32 u=ke->ops[oi].uop;
                    if (u<UOP_COUNT && !seen_op[u]) { seen_op[u]=1;
                        if (p>0 && ops_s[p-1]!='+') p+=snprintf(ops_s+p,sizeof(ops_s)-p,"+");
                        p+=snprintf(ops_s+p,sizeof(ops_s)-p,"%s",uop_names[u]); }}
                if (ops_s[0] == '\0' && term_tag(ke->original_term) == TAG_TOP) {
                    u32 ou = term_ext(ke->original_term);
                    snprintf(ops_s, sizeof(ops_s), "%s",
                             (ou < UOP_COUNT) ? uop_names[ou] : "FUSING");
                }
                snprintf(label,sizeof(label),"K%u: %s\\n[%s]\\n%s",
                         kid, ops_s, sh, dot_kernel_backend(ctx, ke));
                color = "#ccffcc";
            } else if (ext == UOP_GRAD) {
                // GRAD bead: y (input below), gy (output above).
                Term gx = thvm_grad_target_get(ctx, val);
                char tgt[32] = "?";
                if (term_tag(gx) == TAG_ANY) {
                    int p = 0;
                    u32 nt = thvm_grad_targets_count_at(ctx, val);
                    if (nt == 0) {
                        snprintf(tgt, sizeof(tgt), "all");
                    } else {
                        for (u32 gi = 0; gi < nt && p < 24; gi++) {
                            u32 tid = thvm_grad_targets_get_tid_at(ctx, val, gi);
                            if (tid != ~0u)
                                p += snprintf(tgt + p, sizeof(tgt) - p, "%st%u", gi ? "," : "", tid);
                        }
                    }
                } else if (term_tag(gx) == TAG_TEN) {
                    snprintf(tgt, sizeof(tgt), "t%u", (u32)term_val(gx));
                }
                snprintf(label, sizeof(label), "GRAD\\nd/d(%s)", tgt);
                color = "#e8d0ff"; nshape = "box";
            } else {
                snprintf(label, sizeof(label), "%s\\n[%s]", opn, sh);
                if (ext == UOP_ASSIGN) color = "#ffd700";
                else if (is_elementwise(ext)) color = "#cce5ff";
                else if (ext == UOP_SUM || ext == UOP_RMAX) color = "#ffcccc";
                else if (is_view_op(ext)) color = "#fff3cd";
                else if (ext == UOP_MM) color = "#ffccff";
            }
            fprintf(f, "  n%llu [label=\"%s\", shape=%s, fillcolor=\"%s\"];\n", val, label, nshape, color);

            // Child edges: GRAD has 2 heap ports (y + gy). target pattern is metadata.
            u32 arity = 2;
            if (ext == UOP_GRAD) arity = 2;
            else if (ext == UOP_WHERE || ext == UOP_IFZ) arity = 3;
            else if (ext == UOP_KERNEL) arity = 0;
            else if (ext == UOP_LOG_PRINT) arity = 1;
            else if (ext == UOP_DETACH) arity = 1;
            else if (!is_binary(ext) && is_elementwise(ext)) arity = 1;
            for (u32 ai = 0; ai < arity; ai++) {
                Term child = heap_read(ctx, val + ai);
                u8 ctag = term_tag(child); u64 cval = term_val(child);
                u64 cpos = val + ai;
                int edge_hl = heap_dot_hl_on && cpos == heap_dot_hl_slot;
                if (edge_hl) heap_dot_hl_hit = 1;
                const char *elbl = "";
                if (ext == UOP_ASSIGN) elbl = ai==0 ? "tgt" : "src";
                else if (ext >= UOP_RESHAPE && ext <= UOP_PAD) elbl = ai==0 ? "in" : "shape";
                else if (ext == UOP_SUM || ext == UOP_RMAX) elbl = ai==0 ? "in" : "axes";
                else if (ext == UOP_GRAD) elbl = ai==0 ? "y" : "gy";
                else if (ext == UOP_DETACH) elbl = "in";
                else if (is_binary(ext)) elbl = ai==0 ? "a" : "b";
                int rev = 0;

                if (ctag == TAG_DP0 || ctag == TAG_DP1) {
                    // Always show full DUP triangle with both ports
                    u64 dl = cval;
                    DP_SLOT_MARK(cpos);
                    EMIT_DUP_CHAIN(dl);
                    // For DUP redexes, highlight only the DUP principal edge,
                    // never the auxiliary consumer edge.
                    fprintf(f, "  dup%llu -> n%llu [label=\"%s%s\"];\n", dl, val, elbl, ctag==TAG_DP1?" (dp1)":" (dp0)");
                } else if (ctag == TAG_TEN) {
                    EMIT_TEN((u32)cval);
                    if (rev) {
                        if (edge_hl) fprintf(f, "  n%llu -> t%u [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", val, (u32)cval, elbl);
                        else         fprintf(f, "  n%llu -> t%u [label=\"%s\"];\n", val, (u32)cval, elbl);
                    } else {
                        if (edge_hl) fprintf(f, "  t%u -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", (u32)cval, val, elbl);
                        else         fprintf(f, "  t%u -> n%llu [label=\"%s\"];\n", (u32)cval, val, elbl);
                    }
                } else if (ctag == TAG_CTR) {
                    EMIT_CTR_NODE(cval);
                    if (rev) {
                        if (edge_hl) fprintf(f, "  n%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", val, cval, elbl);
                        else         fprintf(f, "  n%llu -> ctr%llu [label=\"%s\"];\n", val, cval, elbl);
                    } else {
                        if (edge_hl) fprintf(f, "  ctr%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", cval, val, elbl);
                        else         fprintf(f, "  ctr%llu -> n%llu [label=\"%s\"];\n", cval, val, elbl);
                    }
                } else if (ctag == TAG_ANY) {
                    if (ext == UOP_GRAD) continue;
                    EMIT_ANY_NODE(cval);
                    if (rev) {
                        if (edge_hl) fprintf(f, "  n%llu -> any%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", val, cval, elbl);
                        else         fprintf(f, "  n%llu -> any%llu [label=\"%s\"];\n", val, cval, elbl);
                    } else {
                        if (edge_hl) fprintf(f, "  any%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", cval, val, elbl);
                        else         fprintf(f, "  any%llu -> n%llu [label=\"%s\"];\n", cval, val, elbl);
                    }
                } else if (ctag == TAG_ERA) {
                    u64 epos = cpos;
                    if (cval != 0) EMIT_ERA_NODE(epos, child);
                    else if (!ERA_SEEN(epos)) {
                        ERA_MARK(epos);
                        fprintf(f, "  era%llu [label=\"ERA\",shape=circle,width=0.38,height=0.38,fixedsize=true,fillcolor=\"#ffb8b8\",fontsize=7];\n", (unsigned long long)epos);
                    }
                    if (rev) {
                        if (edge_hl) fprintf(f, "  n%llu -> era%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", val, (unsigned long long)epos, elbl);
                        else         fprintf(f, "  n%llu -> era%llu [label=\"%s\"];\n", val, (unsigned long long)epos, elbl);
                    } else {
                        if (edge_hl) fprintf(f, "  era%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", (unsigned long long)epos, val, elbl);
                        else         fprintf(f, "  era%llu -> n%llu [label=\"%s\"];\n", (unsigned long long)epos, val, elbl);
                    }
                } else if (ctag == TAG_NUM) {
                    f32 fv; u32 bv=(u32)cval; memcpy(&fv,&bv,4);
                    fprintf(f, "  num%llu_%u [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", (unsigned long long)val, ai, (double)fv);
                    if (edge_hl) fprintf(f, "  num%llu_%u -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", val, ai, val, elbl);
                    else         fprintf(f, "  num%llu_%u -> n%llu [label=\"%s\"];\n", val, ai, val, elbl);
                } else if (ctag == TAG_REF) {
                    EMIT_REF_NODE(cpos, term_ext(child));
                    if (edge_hl) fprintf(f, "  ref%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                    else         fprintf(f, "  ref%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                } else if (ctag == TAG_VAR) {
                    EMIT_VAR_NODE(cpos, cval, term_is_sub(child));
                    if (edge_hl) fprintf(f, "  var%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                    else         fprintf(f, "  var%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                } else if (ctag == TAG_TOP) {
                    if (rev) {
                        if (edge_hl) fprintf(f, "  n%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", val, cval, elbl);
                        else         fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n", val, cval, elbl);
                    } else {
                        if (edge_hl) fprintf(f, "  n%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", cval, val, elbl);
                        else         fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n", cval, val, elbl);
                    }
                } else if (dot_visible_heap_loc_tag(ctag)) {
                    if (edge_hl) fprintf(f, "  n%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                    else         fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                } else {
                    EMIT_RAW_NODE(cpos, child);
                    if (edge_hl) fprintf(f, "  h%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                    else         fprintf(f, "  h%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                }
            }
            nn++;
            continue;
        }

        // --- TAG_APP: visible sequencing/control node ---
        if (tag == TAG_APP) {
            if (!LOC_LIVE(val)) continue;
            if (NODE_SEEN(val)) continue;
            NODE_MARK(val);
            fprintf(f, "  n%llu [label=\"APP\", shape=invtriangle, fillcolor=\"#f3f3f3\"];\n",
                    (unsigned long long)val);
            for (u32 ai = 0; ai < 2; ai++) {
                Term child = heap_read(ctx, val + ai);
                u8 ctag = term_tag(child);
                u64 cval = term_val(child);
                u64 cpos = val + ai;
                int edge_hl = heap_dot_hl_on && cpos == heap_dot_hl_slot;
                if (edge_hl) heap_dot_hl_hit = 1;
                const char *elbl = ai == 0 ? "fun" : "arg";

                if (ctag == TAG_DP0 || ctag == TAG_DP1) {
                    u64 dl = cval;
                    DP_SLOT_MARK(cpos);
                    EMIT_DUP_CHAIN(dl);
                    fprintf(f, "  dup%llu -> n%llu [label=\"%s%s\"%s];\n",
                            (unsigned long long)dl, (unsigned long long)val, elbl,
                            ctag == TAG_DP1 ? " (dp1)" : " (dp0)",
                            edge_hl ? " [color=\"#cc0000\",penwidth=2.0]" : "");
                } else if (ctag == TAG_TEN) {
                    EMIT_TEN((u32)cval);
                    if (edge_hl) fprintf(f, "  t%u -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (u32)cval, (unsigned long long)val, elbl);
                    else         fprintf(f, "  t%u -> n%llu [label=\"%s\"];\n",
                                         (u32)cval, (unsigned long long)val, elbl);
                } else if (ctag == TAG_CTR) {
                    EMIT_CTR_NODE(cval);
                    if (edge_hl) fprintf(f, "  ctr%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                    else         fprintf(f, "  ctr%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                } else if (ctag == TAG_NUM) {
                    f32 fv; u32 bv = (u32)cval; memcpy(&fv, &bv, 4);
                    fprintf(f, "  num_app%llu_%u [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n",
                            (unsigned long long)val, ai, (double)fv);
                    if (edge_hl) fprintf(f, "  num_app%llu_%u -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)val, ai, (unsigned long long)val, elbl);
                    else         fprintf(f, "  num_app%llu_%u -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)val, ai, (unsigned long long)val, elbl);
                } else if (ctag == TAG_ERA) {
                    u64 epos = cpos;
                    if (cval != 0) EMIT_ERA_NODE(epos, child);
                    else if (!ERA_SEEN(epos)) {
                        ERA_MARK(epos);
                        fprintf(f, "  era%llu [label=\"ERA\",shape=circle,width=0.38,height=0.38,fixedsize=true,fillcolor=\"#ffb8b8\",fontsize=7];\n",
                                (unsigned long long)epos);
                    }
                    if (edge_hl) fprintf(f, "  era%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)epos, (unsigned long long)val, elbl);
                    else         fprintf(f, "  era%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)epos, (unsigned long long)val, elbl);
                } else if (ctag == TAG_ANY) {
                    EMIT_ANY_NODE(cval);
                    if (edge_hl) fprintf(f, "  any%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                    else         fprintf(f, "  any%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                } else if (ctag == TAG_REF) {
                    EMIT_REF_NODE(cpos, term_ext(child));
                    if (edge_hl) fprintf(f, "  ref%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                    else         fprintf(f, "  ref%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                } else if (ctag == TAG_VAR) {
                    EMIT_VAR_NODE(cpos, cval, term_is_sub(child));
                    if (edge_hl) fprintf(f, "  var%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                    else         fprintf(f, "  var%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                } else if (dot_visible_heap_loc_tag(ctag)) {
                    if (edge_hl) fprintf(f, "  n%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                    else         fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                } else {
                    EMIT_RAW_NODE(cpos, child);
                    if (edge_hl) fprintf(f, "  h%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                    else         fprintf(f, "  h%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                }
            }
            nn++;
            continue;
        }

        // --- CTR ---
        if (tag == TAG_CTR) {
            if (!LOC_LIVE(val)) continue;
            if (!NODE_SEEN(CTR_NODE_KEY(val))) {
                NODE_MARK(CTR_NODE_KEY(val));
                fprintf(f, "  ctr%llu [label=\"CTR\\nN=%u\", shape=hexagon, fillcolor=\"#f3f3f3\", fontsize=9];\n",
                        (unsigned long long)val, ext);
            }
            if (val < ctx->heap_pos && (!ctr_children_emitted || !ctr_children_emitted[val])) {
                if (ctr_children_emitted) ctr_children_emitted[val] = 1;
                for (u32 gi = 0; gi < ext; gi++) {
                    Term cterm = heap_read(ctx, val + gi);
                    char clbl[16];
                    snprintf(clbl, sizeof(clbl), "c%u", gi);
                    u8 ctag = term_tag(cterm);
                    u64 cval = term_val(cterm);
                    u64 cpos = val + gi;
                    int edge_hl = heap_dot_hl_on && cpos == heap_dot_hl_slot;
                    if (edge_hl) heap_dot_hl_hit = 1;
                    if (ctag == TAG_TEN) {
                        EMIT_TEN((u32)cval);
                        if (edge_hl) fprintf(f, "  t%u -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (u32)cval, (unsigned long long)val, clbl);
                        else         fprintf(f, "  t%u -> ctr%llu [label=\"%s\"];\n",
                                             (u32)cval, (unsigned long long)val, clbl);
                    } else if (ctag == TAG_TOP) {
                        if (!NODE_SEEN(cval)) {
                            NODE_MARK(cval);
                            const char *opn = term_ext(cterm) < UOP_COUNT ? uop_names[term_ext(cterm)] : "?";
                            fprintf(f, "  n%llu [label=\"%s\", shape=box, fillcolor=\"#f0f0f0\"];\n",
                                    (unsigned long long)cval, opn);
                        }
                        if (edge_hl) fprintf(f, "  n%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)cval, (unsigned long long)val, clbl);
                        else         fprintf(f, "  n%llu -> ctr%llu [label=\"%s\"];\n",
                                             (unsigned long long)cval, (unsigned long long)val, clbl);
                    } else if (ctag == TAG_DP0 || ctag == TAG_DP1) {
                        u64 dl = cval;
                        DP_SLOT_MARK(cpos);
                        EMIT_DUP_CHAIN(dl);
                        if (edge_hl) fprintf(f, "  dup%llu -> ctr%llu [label=\"%s%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)dl, (unsigned long long)val, clbl,
                                             ctag == TAG_DP1 ? " (dp1)" : " (dp0)");
                        else         fprintf(f, "  dup%llu -> ctr%llu [label=\"%s%s\"];\n",
                                             (unsigned long long)dl, (unsigned long long)val, clbl,
                                             ctag == TAG_DP1 ? " (dp1)" : " (dp0)");
                    } else if (ctag == TAG_NUM) {
                        f32 fv; u32 bv = (u32)cval; memcpy(&fv, &bv, 4);
                        fprintf(f, "  num_ctr%llu_%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n",
                                (unsigned long long)val, (unsigned long long)cpos, (double)fv);
                        if (edge_hl) fprintf(f, "  num_ctr%llu_%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)val, (unsigned long long)cpos, (unsigned long long)val, clbl);
                        else         fprintf(f, "  num_ctr%llu_%llu -> ctr%llu [label=\"%s\"];\n",
                                             (unsigned long long)val, (unsigned long long)cpos, (unsigned long long)val, clbl);
                    } else if (ctag == TAG_ERA) {
                        if (cval != 0) EMIT_ERA_NODE(cpos, cterm);
                        else if (!ERA_SEEN(cpos)) {
                            ERA_MARK(cpos);
                            fprintf(f, "  era%llu [label=\"ERA\",shape=circle,width=0.38,height=0.38,fixedsize=true,fillcolor=\"#ffb8b8\",fontsize=7];\n",
                                    (unsigned long long)cpos);
                        }
                        if (edge_hl) fprintf(f, "  era%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)cpos, (unsigned long long)val, clbl);
                        else         fprintf(f, "  era%llu -> ctr%llu [label=\"%s\"];\n",
                                             (unsigned long long)cpos, (unsigned long long)val, clbl);
                    } else if (ctag == TAG_ANY) {
                        EMIT_ANY_NODE(cval);
                        if (edge_hl) fprintf(f, "  any%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)cval, (unsigned long long)val, clbl);
                        else         fprintf(f, "  any%llu -> ctr%llu [label=\"%s\"];\n",
                                             (unsigned long long)cval, (unsigned long long)val, clbl);
                    } else if (ctag == TAG_CTR) {
                        EMIT_CTR_NODE(cval);
                        if (edge_hl) fprintf(f, "  ctr%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)cval, (unsigned long long)val, clbl);
                        else         fprintf(f, "  ctr%llu -> ctr%llu [label=\"%s\"];\n",
                                             (unsigned long long)cval, (unsigned long long)val, clbl);
                    } else if (ctag == TAG_REF) {
                        EMIT_REF_NODE(cpos, term_ext(cterm));
                        if (edge_hl) fprintf(f, "  ref%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)cpos, (unsigned long long)val, clbl);
                        else         fprintf(f, "  ref%llu -> ctr%llu [label=\"%s\"];\n",
                                             (unsigned long long)cpos, (unsigned long long)val, clbl);
                    } else if (ctag == TAG_VAR) {
                        EMIT_VAR_NODE(cpos, cval, term_is_sub(cterm));
                        if (edge_hl) fprintf(f, "  var%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)cpos, (unsigned long long)val, clbl);
                        else         fprintf(f, "  var%llu -> ctr%llu [label=\"%s\"];\n",
                                             (unsigned long long)cpos, (unsigned long long)val, clbl);
                    } else if (dot_visible_heap_loc_tag(ctag)) {
                        if (edge_hl) fprintf(f, "  n%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)cval, (unsigned long long)val, clbl);
                        else         fprintf(f, "  n%llu -> ctr%llu [label=\"%s\"];\n",
                                             (unsigned long long)cval, (unsigned long long)val, clbl);
                    } else {
                        EMIT_RAW_NODE(cpos, cterm);
                        if (edge_hl) fprintf(f, "  h%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)cpos, (unsigned long long)val, clbl);
                        else         fprintf(f, "  h%llu -> ctr%llu [label=\"%s\"];\n",
                                             (unsigned long long)cpos, (unsigned long long)val, clbl);
                    }
                }
            }
            nn++;
            continue;
        }

        // --- Visible non-TOP combinators ---
        if (tag == TAG_REF) {
            if (!SLOT_LIVE(h)) continue;
            EMIT_REF_NODE(h, ext);
            nn++;
            continue;
        }

        if (tag == TAG_VAR) {
            if (!SLOT_LIVE(h)) continue;
            EMIT_VAR_NODE(h, val, term_is_sub(t));
            nn++;
            continue;
        }

        if (dot_visible_heap_loc_tag(tag) && tag != TAG_TOP && tag != TAG_APP && tag != TAG_CTR) {
            if (!LOC_LIVE(val)) continue;
            if (NODE_SEEN(val)) continue;
            NODE_MARK(val);
            char label[96];
            if (tag == TAG_SUP || tag == TAG_USP) {
                snprintf(label, sizeof(label), "%s #%u", dot_heap_tag_name(tag), ext);
            } else if (tag == TAG_MAT || tag == TAG_REF) {
                snprintf(label, sizeof(label), "%s\\n#%u", dot_heap_tag_name(tag), ext);
            } else {
                snprintf(label, sizeof(label), "%s", dot_heap_tag_name(tag));
            }
            fprintf(f, "  n%llu [label=\"%s\", shape=%s, fillcolor=\"%s\"];\n",
                    (unsigned long long)val, label, dot_heap_node_shape(tag), dot_heap_node_color(tag));
            u32 ar = dot_term_arity(t);
            for (u32 ai = 0; ai < ar; ai++) {
                Term child = heap_read(ctx, val + ai);
                u8 ctag = term_tag(child);
                u64 cval = term_val(child);
                u64 cpos = val + ai;
                int edge_hl = heap_dot_hl_on && cpos == heap_dot_hl_slot;
                if (edge_hl) heap_dot_hl_hit = 1;
                const char *elbl = dot_heap_port_name(tag, ai);
                if (ctag == TAG_DP0 || ctag == TAG_DP1) {
                    u64 dl = cval;
                    DP_SLOT_MARK(cpos);
                    EMIT_DUP_CHAIN(dl);
                    fprintf(f, "  dup%llu -> n%llu [label=\"%s%s\"%s];\n",
                            (unsigned long long)dl, (unsigned long long)val, elbl,
                            ctag == TAG_DP1 ? " (dp1)" : " (dp0)",
                            edge_hl ? " [color=\"#cc0000\",penwidth=2.0]" : "");
                } else if (ctag == TAG_TEN) {
                    EMIT_TEN((u32)cval);
                    if (edge_hl) fprintf(f, "  t%u -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (u32)cval, (unsigned long long)val, elbl);
                    else         fprintf(f, "  t%u -> n%llu [label=\"%s\"];\n",
                                         (u32)cval, (unsigned long long)val, elbl);
                } else if (ctag == TAG_NUM) {
                    f32 fv; u32 bv = (u32)cval; memcpy(&fv, &bv, 4);
                    fprintf(f, "  num_heap%llu_%u [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n",
                            (unsigned long long)val, ai, (double)fv);
                    if (edge_hl) fprintf(f, "  num_heap%llu_%u -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)val, ai, (unsigned long long)val, elbl);
                    else         fprintf(f, "  num_heap%llu_%u -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)val, ai, (unsigned long long)val, elbl);
                } else if (ctag == TAG_ERA) {
                    u64 epos = cpos;
                    if (cval != 0) EMIT_ERA_NODE(epos, child);
                    else if (!ERA_SEEN(epos)) {
                        ERA_MARK(epos);
                        fprintf(f, "  era%llu [label=\"ERA\",shape=circle,width=0.38,height=0.38,fixedsize=true,fillcolor=\"#ffb8b8\",fontsize=7];\n",
                                (unsigned long long)epos);
                    }
                    if (edge_hl) fprintf(f, "  era%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)epos, (unsigned long long)val, elbl);
                    else         fprintf(f, "  era%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)epos, (unsigned long long)val, elbl);
                } else if (ctag == TAG_ANY) {
                    EMIT_ANY_NODE(cval);
                    if (edge_hl) fprintf(f, "  any%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                    else         fprintf(f, "  any%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                } else if (ctag == TAG_CTR) {
                    EMIT_CTR_NODE(cval);
                    if (edge_hl) fprintf(f, "  ctr%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                    else         fprintf(f, "  ctr%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                } else if (ctag == TAG_REF) {
                    EMIT_REF_NODE(cpos, term_ext(child));
                    if (edge_hl) fprintf(f, "  ref%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                    else         fprintf(f, "  ref%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                } else if (ctag == TAG_VAR) {
                    EMIT_VAR_NODE(cpos, cval, term_is_sub(child));
                    if (edge_hl) fprintf(f, "  var%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                    else         fprintf(f, "  var%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                } else if (ctag == TAG_TOP || dot_visible_heap_loc_tag(ctag)) {
                    if (edge_hl) fprintf(f, "  n%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                    else         fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cval, (unsigned long long)val, elbl);
                } else {
                    EMIT_RAW_NODE(cpos, child);
                    if (edge_hl) fprintf(f, "  h%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                    else         fprintf(f, "  h%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                }
            }
            nn++;
            continue;
        }

        // Skip TAG_TEN, TAG_NUM at heap positions (shown when referenced as children)
    }

    u8 *fusing_leaf_done = ctx->heap_pos ? (u8 *)calloc((size_t)ctx->heap_pos, 1) : NULL;

    #define EMIT_TOP_NODE_LABEL(_loc, _ext) do { \
        if (!NODE_SEEN((_loc))) { \
            NODE_MARK((_loc)); \
            char label[128]; \
            const char *color = "#f0f0f0"; \
            const char *nshape = "box"; \
            const char *opn = ((_ext) < UOP_COUNT) ? uop_names[(_ext)] : "?"; \
            const View *v = st_get((_loc)); \
            Shape shp = SHAPE(1); \
            int has_shape = 0; \
            if (v && dot_shape_is_valid(v->shape)) { \
                shp = v->shape; \
                has_shape = 1; \
            } else if (dot_infer_top_shape(ctx, (_ext), (_loc), &shp) && \
                       dot_shape_is_valid(shp)) { \
                has_shape = 1; \
            } \
            char sh[64] = "?"; \
            if (has_shape) { \
                int p = 0; \
                if (shp.rank == 0) { \
                    snprintf(sh, sizeof(sh), "1"); \
                } else { \
                    sh[0] = '\0'; \
                    for (u32 d = 0; d < shp.rank && p < 50; d++) \
                        p += snprintf(sh + p, sizeof(sh) - p, "%s%u", d ? "," : "", shp.dims[d]); \
                } \
            } \
            if ((_ext) == UOP_KERNEL) { \
                Term kid_t2 = heap_read(ctx, (_loc) + 1); \
                u32 kid2 = (term_tag(kid_t2) == TAG_NUM) ? (u32)term_val(kid_t2) : 0; \
                KernelEntry *ke2 = &sched_kernels[kid2]; \
                char ops_s[64] = ""; int p = 0; \
                if (ke2->has_reduce) { \
                    p += snprintf(ops_s + p, sizeof(ops_s) - p, "%s", \
                                  ke2->has_reduce == UOP_SUM ? "SUM" : "RMAX"); \
                    if (ke2->n_ops) p += snprintf(ops_s + p, sizeof(ops_s) - p, "+"); \
                } \
                u8 seen_op[UOP_COUNT] = {0}; \
                for (u32 oi = 0; oi < ke2->n_ops && p < 50; oi++) { \
                    u32 u = ke2->ops[oi].uop; \
                    if (u < UOP_COUNT && !seen_op[u]) { \
                        seen_op[u] = 1; \
                        if (p > 0 && ops_s[p - 1] != '+') \
                            p += snprintf(ops_s + p, sizeof(ops_s) - p, "+"); \
                        p += snprintf(ops_s + p, sizeof(ops_s) - p, "%s", uop_names[u]); \
                    } \
                } \
                if (ops_s[0] == '\0' && term_tag(ke2->original_term) == TAG_TOP) { \
                    u32 ou2 = term_ext(ke2->original_term); \
                    snprintf(ops_s, sizeof(ops_s), "%s", \
                             (ou2 < UOP_COUNT) ? uop_names[ou2] : "FUSING"); \
                } \
                snprintf(label, sizeof(label), "K%u: %s\\n[%s]\\n%s", \
                         kid2, ops_s, sh, dot_kernel_backend(ctx, ke2)); \
                color = "#ccffcc"; \
            } else if ((_ext) == UOP_GRAD) { \
                Term gx = thvm_grad_target_get(ctx, (_loc)); \
                char tgt[32] = "?"; \
                if (term_tag(gx) == TAG_ANY) { \
                    int p = 0; \
                    u32 nt = thvm_grad_targets_count_at(ctx, (_loc)); \
                    if (nt == 0) { \
                        snprintf(tgt, sizeof(tgt), "all"); \
                    } else { \
                        for (u32 gi = 0; gi < nt && p < 24; gi++) { \
                            u32 tid = thvm_grad_targets_get_tid_at(ctx, (_loc), gi); \
                            if (tid != ~0u) \
                                p += snprintf(tgt + p, sizeof(tgt) - p, "%st%u", gi ? "," : "", tid); \
                        } \
                    } \
                } else if (term_tag(gx) == TAG_TEN) { \
                    snprintf(tgt, sizeof(tgt), "t%u", (u32)term_val(gx)); \
                } \
                snprintf(label, sizeof(label), "GRAD\\nd/d(%s)", tgt); \
                color = "#e8d0ff"; \
            } else { \
                snprintf(label, sizeof(label), "%s\\n[%s]", opn, sh); \
                if ((_ext) == UOP_ASSIGN) color = "#ffd700"; \
                else if (is_elementwise((_ext))) color = "#cce5ff"; \
                else if ((_ext) == UOP_SUM || (_ext) == UOP_RMAX) color = "#ffcccc"; \
                else if (is_view_op((_ext))) color = "#fff3cd"; \
                else if ((_ext) == UOP_MM) color = "#ffccff"; \
            } \
            fprintf(f, "  n%llu [label=\"%s\", shape=%s, fillcolor=\"%s\"];\n", \
                    (unsigned long long)(_loc), label, nshape, color); \
        } \
    } while(0)

    #define EMIT_FUSING_LEAF_CHILD(_child, _parent_loc, _edge_id, _label) do { \
        Term _cterm = (_child); \
        for (int _r = 0; _r < 10; _r++) { \
            if (term_tag(_cterm) == TAG_DP0 || term_tag(_cterm) == TAG_DP1) { \
                _cterm = heap_read(ctx, term_val(_cterm)); \
                continue; \
            } \
            break; \
        } \
        u8 _ctag = term_tag(_cterm); \
        u64 _cval = term_val(_cterm); \
        if (_ctag == TAG_TEN) { \
            EMIT_TEN((u32)_cval); \
            fprintf(f, "  t%u -> n%llu [label=\"%s\"];\n", (u32)_cval, (unsigned long long)(_parent_loc), (_label)); \
        } else if (_ctag == TAG_NUM) { \
            f32 _fv; u32 _bv = (u32)_cval; memcpy(&_fv, &_bv, 4); \
            fprintf(f, "  numleaf%llu_%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", \
                    (unsigned long long)(_parent_loc), (unsigned long long)(_edge_id), (double)_fv); \
            fprintf(f, "  numleaf%llu_%llu -> n%llu [label=\"%s\"];\n", \
                    (unsigned long long)(_parent_loc), (unsigned long long)(_edge_id), \
                    (unsigned long long)(_parent_loc), (_label)); \
        } else if (_ctag == TAG_ERA) { \
            u64 _epos = _cval ? _cval : ((u64)(_parent_loc) << 8) ^ (_edge_id); \
            if (_cval != 0) EMIT_ERA_NODE(_epos, _cterm); \
            else if (!ERA_SEEN(_epos)) { \
                ERA_MARK(_epos); \
                fprintf(f, "  era%llu [label=\"ERA\",shape=circle,width=0.38,height=0.38,fixedsize=true,fillcolor=\"#ffb8b8\",fontsize=7];\n", \
                        (unsigned long long)_epos); \
            } \
            fprintf(f, "  era%llu -> n%llu [label=\"%s\"];\n", \
                    (unsigned long long)_epos, (unsigned long long)(_parent_loc), (_label)); \
        } else if (_ctag == TAG_TOP) { \
            EMIT_TOP_NODE_LABEL(_cval, term_ext(_cterm)); \
            fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n", \
                    (unsigned long long)_cval, (unsigned long long)(_parent_loc), (_label)); \
        } else if (_ctag == TAG_CTR) { \
            EMIT_CTR_NODE(_cval); \
            fprintf(f, "  ctr%llu -> n%llu [label=\"%s\"];\n", \
                    (unsigned long long)_cval, (unsigned long long)(_parent_loc), (_label)); \
        } else if (_ctag == TAG_ANY) { \
            EMIT_ANY_NODE(_cval); \
            fprintf(f, "  any%llu -> n%llu [label=\"%s\"];\n", \
                    (unsigned long long)_cval, (unsigned long long)(_parent_loc), (_label)); \
        } else { \
            EMIT_RAW_NODE(_cval, _cterm); \
            fprintf(f, "  h%llu -> n%llu [label=\"%s\"];\n", \
                    (unsigned long long)_cval, (unsigned long long)(_parent_loc), (_label)); \
        } \
    } while(0)

    // FUSING leaf edges (for post-schedule graphs)
    if (heap_dot_include_sched_kernels) {
        extern KernelEntry sched_kernels[];
        extern u32 sched_kernel_count;
        u8 output_used[SCHED_MAX_KERNELS];
        memset(output_used, 0, sizeof(output_used));
        for (u32 kid = 0; kid < sched_kernel_count; kid++) {
            KernelEntry *consumer = &sched_kernels[kid];
            for (u32 li = 0; li < consumer->n_leaves; li++) {
                if (consumer->leaf_kinds[li] != KERNEL_LEAF_TENSOR) continue;
                u32 lid = consumer->leaf_ids[li];
                for (u32 pk = 0; pk < sched_kernel_count; pk++) {
                    if (sched_kernels[pk].output_tid == lid) {
                        output_used[pk] = 1;
                        break;
                    }
                }
            }
        }
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term t = ctx->heap[h];
            if (term_tag(t) != TAG_TOP || term_ext(t) != UOP_KERNEL) continue;
            u64 loc = term_val(t);
            if (top_live && (loc >= ctx->heap_pos || !top_live[loc])) continue;
            if (fusing_leaf_done && loc < ctx->heap_pos) {
                if (fusing_leaf_done[loc]) continue;
                fusing_leaf_done[loc] = 1;
            }
            Term kid_t = heap_read(ctx, loc + 1);
            if (term_tag(kid_t) != TAG_NUM) continue;
            u32 kid = (u32)term_val(kid_t);
            if (kid >= sched_kernel_count) continue;
            KernelEntry *ke = &sched_kernels[kid];
            if (output_used[kid] && ke->output_tid) {
                char out_lbl[48];
                dot_kernel_output_label(ke, out_lbl, sizeof(out_lbl));
                EMIT_TEN(ke->output_tid);
                fprintf(f, "  n%llu -> t%u [style=dashed,color=\"#009900\",fontcolor=\"#006600\",label=\"%s\"];\n",
                        (unsigned long long)loc, ke->output_tid, out_lbl);
            }
            for (u32 li = 0; li < ke->n_leaves; li++) {
                char leaf_lbl[48];
                dot_kernel_leaf_label(ctx, ke, li, leaf_lbl, sizeof(leaf_lbl));
                if (ke->leaf_kinds[li] == KERNEL_LEAF_TENSOR) {
                    EMIT_TEN(ke->leaf_ids[li]);
                    fprintf(f, "  t%u -> n%llu [style=dashed,color=\"#009900\",fontcolor=\"#006600\",label=\"%s\"];\n",
                            ke->leaf_ids[li], (unsigned long long)loc, leaf_lbl);
                } else if (ke->leaf_kinds[li] == KERNEL_LEAF_NUM) {
                    f32 fv = ke->leaf_nums[li];
                    fprintf(f, "  numleaf%llu_%u [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n",
                            (unsigned long long)loc, li, (double)fv);
                    fprintf(f, "  numleaf%llu_%u -> n%llu [style=dashed,color=\"#009900\",fontcolor=\"#006600\",label=\"%s\"];\n",
                            (unsigned long long)loc, li, (unsigned long long)loc, leaf_lbl);
                }
            }
        }
    }

    #undef EMIT_FUSING_LEAF_CHILD
    #undef EMIT_TOP_NODE_LABEL

    fprintf(f, "}\n");
    fclose(f);
    free(dp_slot_emitted);
    free(ctr_children_emitted);
    free(fusing_leaf_done);
    free(top_live);
    free(slot_live);
    free(loc_live);
    fprintf(stderr, "heap_dot: %u nodes → %s\n", nn, path);
    #undef EMIT_TEN
    #undef EMIT_RAW_NODE
    #undef RAW_NODE_KEY
    #undef EMIT_CTR_NODE
    #undef CTR_NODE_KEY
    #undef EMIT_ANY_NODE
    #undef ANY_NODE_KEY
    #undef EMIT_REF_NODE
    #undef REF_NODE_KEY
    #undef EMIT_VAR_NODE
    #undef VAR_NODE_KEY
    #undef SLOT_LIVE
    #undef LOC_LIVE
    #undef EMIT_DUP_STUB
    #undef EMIT_DUP_CHAIN
    #undef DUP_PORT_HAS_VISIBLE_CONSUMER
    #undef EDGE_HL_ONLY
    #undef EDGE_HL_LABEL
    #undef EMIT_ERA_NODE
    #undef ERA_DEDUP_MAX
    #undef ERA_SEEN
    #undef ERA_MARK
    #undef NODE_DEDUP_MAX
    #undef NODE_SEEN
    #undef NODE_MARK
    #undef DP_SLOT_MARK
    #undef DP_SLOT_SEEN
}

static void thvm_dump_json(TinyHVM *ctx, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "dump_json: can't open %s\n", path); return; }

    fprintf(f, "{\n  \"tensors\": [\n");
    for (u32 i = 0; i < ctx->tensor_count; i++) {
        TensorMeta *m = &ctx->tensors[i];
        if (m->view.numel == 0) { fprintf(f, "    null%s\n", i+1<ctx->tensor_count?",":""); continue; }

        fprintf(f, "    {\"id\":%u, \"op\":\"%s\", \"src\":[%u,%u], ",
                i, (m->creator_op < UOP_COUNT) ? uop_names[m->creator_op] : "LEAF",
                m->src_ids[0], m->src_ids[1]);

        fprintf(f, "\"shape\":[");
        for (u32 d = 0; d < m->view.shape.rank; d++)
            fprintf(f, "%s%u", d > 0 ? "," : "", m->view.shape.dims[d]);

        fprintf(f, "], \"strides\":[");
        for (u32 d = 0; d < m->view.shape.rank; d++)
            fprintf(f, "%s%d", d > 0 ? "," : "", m->view.strides[d]);

        fprintf(f, "], \"numel\":%u, \"buf\":%u, \"offset\":%d, \"grad\":%d, \"mask\":%d",
                m->view.numel, m->buf_id, m->view.offset, m->requires_grad, m->view.has_mask);

        if (m->view.has_mask) {
            fprintf(f, ", \"mask_begin\":[");
            for (u32 d = 0; d < m->view.shape.rank; d++)
                fprintf(f, "%s%u", d > 0 ? "," : "", m->view.mask_begin[d]);
            fprintf(f, "], \"mask_end\":[");
            for (u32 d = 0; d < m->view.shape.rank; d++)
                fprintf(f, "%s%u", d > 0 ? "," : "", m->view.mask_end[d]);
            fprintf(f, "]");
        }

        // First few values (for debugging)
        if (m->buf_id && m->view.numel <= 16 && m->view.contiguous) {
            f32 vals[16];
            m->backend->buf_read(m->buf_id, vals, m->view.numel * sizeof(f32));
            fprintf(f, ", \"vals\":[");
            for (u32 j = 0; j < m->view.numel; j++)
                fprintf(f, "%s%.6g", j > 0 ? "," : "", vals[j]);
            fprintf(f, "]");
        }

        fprintf(f, "}%s\n", i+1<ctx->tensor_count?",":"");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
    fprintf(stderr, "dump_json: wrote %u tensors to %s\n", ctx->tensor_count, path);
}
