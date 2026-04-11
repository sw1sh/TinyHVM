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

static int dot_shape_is_valid(Shape s) {
    if (s.rank > MAX_DIM) return 0;
    for (u32 i = 0; i < s.rank; i++) {
        if (s.dims[i] == 0) return 0;
    }
    return 1;
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
            if (tid < ctx->tensor_count) { *out = ctx->tensors[tid].view.shape; return 1; }
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
    f32 tmp[MAX_DIM];
    const f32 *pf = (const f32 *)m->host_ptr;
    if (!pf && m->backend) {
        META_READ(m->backend, m->buf_id, tmp, n * sizeof(f32));
        pf = tmp;
    }
    if (!pf) return 0;
    Shape s = {.rank = n};
    for (u32 i = 0; i < n; i++) {
        s.dims[i] = (u32)pf[i];
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
                    f32 tmp[MAX_DIM];
                    const f32 *pf = (const f32 *)m->host_ptr;
                    if (!pf && m->backend) {
                        META_READ(m->backend, m->buf_id, tmp, n * sizeof(f32));
                        pf = tmp;
                    }
                    if (pf) {
                        for (u32 i = 0; i < n; i++) {
                            int ax = (int)pf[i];
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
            if (ext == UOP_FUSING) return 0;
            if (ext == UOP_WHERE || ext == UOP_IFZ) return 3;
            if (ext == UOP_GRAD) return 2;
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
        if (m->requires_grad) color = "#e8f4e8";
        if (m->view.has_mask) color = "#fff0e0";
        if (!m->creator_op && m->requires_grad) color = "#e0e8ff"; // param

        const char *op = (m->creator_op < UOP_COUNT) ? uop_names[m->creator_op] : "?";

        fprintf(f, "  t%u [label=\"{t%u|%s %s|buf=%u off=%d|strides=[%s]%s}\", "
                   "style=filled, fillcolor=\"%s\"];\n",
                i, i, m->creator_op ? op : "LEAF", shape_str,
                m->buf_id, m->view.offset, stride_str,
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
            const char *_dt=_m->dtype==0?"f32":_m->dtype==1?"i32":"?"; \
            const char *_bk=_m->backend?((_m->backend==ctx->backends[0])?"cpu":"mtl"):"?"; \
            const char *_fc=_m->requires_grad?"#ffe0e0":"#e0e0e0"; \
            fprintf(f,"  t%u [label=\"t%u\\n[%s]\\n%s %s%s\",shape=box,fillcolor=\"%s\"];\n", \
                    (tid),(tid),_sh,_dt,_bk,_m->requires_grad?" grad":"",_fc); \
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
    #define DP_SLOT_MARK(pos) do { if ((pos) < ctx->heap_pos) dp_slot_emitted[(pos)] = 1; } while(0)
    #define DP_SLOT_SEEN(pos) (((pos) < ctx->heap_pos) ? dp_slot_emitted[(pos)] : 0)
    #define RAW_NODE_KEY(v) ((v) + 0x200000)
    #define CTR_NODE_KEY(v) ((v) + 0x300000)
    #define ANY_NODE_KEY(v) ((v) + 0x400000)
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
            u32 _n = 0; \
            if ((cloc) < ctx->heap_pos) { \
                Term _nt = heap_read(ctx, (cloc)); \
                if (term_tag(_nt) == TAG_NUM) _n = (u32)term_val(_nt); \
            } \
            fprintf(f, "  ctr%llu [label=\"CTR\\nN=%u\", shape=hexagon, fillcolor=\"#f3f3f3\", fontsize=9];\n", \
                    (unsigned long long)(cloc), _n); \
        } \
    } while(0)
    #define EMIT_ANY_NODE(apos) do { \
        if (!NODE_SEEN(ANY_NODE_KEY(apos))) { \
            NODE_MARK(ANY_NODE_KEY(apos)); \
            fprintf(f, "  any%llu [label=\"ANY\", shape=oval, fillcolor=\"#eeeeee\", fontsize=8];\n", (unsigned long long)(apos)); \
        } \
    } while(0)

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
        for (u32 gi = 0, gn = thvm_grad_detached_root_count(ctx); gi < gn; gi++) {
            Term groot = thvm_grad_detached_root_get(ctx, gi);
            if (!(term_tag(groot) == TAG_ERA && term_val(groot) == 0))
                PUSH_TERM(groot);
        }
        for (u32 gi = 0, gn = thvm_grad_output_count(ctx); gi < gn; gi++) {
            Term gout = thvm_grad_output_get(ctx, gi);
            if (!(term_tag(gout) == TAG_ERA && term_val(gout) == 0))
                PUSH_TERM(gout);
        }
        // Also seed explicit active ERA agents. GRAD interactions can drop
        // discarded metadata onto detached ERA components that still belong to
        // the literal heap state and must be visible/firable step-by-step.
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term ht = ctx->heap[h];
            if (term_tag(ht) == TAG_ERA && term_val(ht) != 0)
                PUSH_TERM(ht);
            if (term_tag(ht) == TAG_TOP && term_ext(ht) == UOP_ASSIGN)
                PUSH_TERM(ht);
        }

        while (work && wp > 0) {
            Term tt = work[--wp];
            u8 tg = term_tag(tt);
            u64 tv = term_val(tt);
            if (tg == TAG_TOP) {
                if (tv == 0 || tv >= ctx->heap_pos || top_live[tv]) continue;
                top_live[tv] = 1;
                u32 ar = dot_term_arity(tt);
                for (u32 i = 0; i < ar; i++) {
                    u64 p = tv + i;
                    if (p < ctx->heap_pos && (!seen_slot || !seen_slot[p])) {
                        if (seen_slot) seen_slot[p] = 1;
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
                    PUSH_TERM(heap_read(ctx, tv));
                }
                continue;
            }
            if (tg == TAG_CTR) {
                if (tv == 0 || tv >= ctx->heap_pos) continue;
                if (!seen_slot || !seen_slot[tv]) {
                    if (seen_slot) seen_slot[tv] = 1;
                    PUSH_TERM(heap_read(ctx, tv));
                }
                Term nt = heap_read(ctx, tv);
                if (term_tag(nt) == TAG_NUM) {
                    u32 n = (u32)term_val(nt);
                    if (n > 64) n = 64;
                    for (u32 i = 0; i < 2 * n; i++) {
                        u64 p = tv + 1 + i;
                        if (p < ctx->heap_pos && (!seen_slot || !seen_slot[p])) {
                            if (seen_slot) seen_slot[p] = 1;
                            PUSH_TERM(heap_read(ctx, p));
                        }
                    }
                }
                continue;
            }
            u32 ar = dot_term_arity(tt);
            for (u32 i = 0; i < ar; i++) {
                u64 p = tv + i;
                if (p < ctx->heap_pos && (!seen_slot || !seen_slot[p])) {
                    if (seen_slot) seen_slot[p] = 1;
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
            if (ext == UOP_FUSING) {
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
                snprintf(label,sizeof(label),"K%u: %s\\n[%s]",kid,ops_s,sh);
                color = "#ccffcc";
            } else if (ext == UOP_GRAD) {
                // GRAD bead: y (input below), gy (output above).
                Term gx = thvm_grad_target_get(ctx, val);
                char tgt[32] = "?";
                if (term_tag(gx) == TAG_ANY) {
                    int p = 0;
                    u32 nt = thvm_grad_targets_count(ctx);
                    if (nt == 0) {
                        snprintf(tgt, sizeof(tgt), "all");
                    } else {
                        for (u32 gi = 0; gi < nt && p < 24; gi++) {
                            u32 tid = thvm_grad_targets_get_tid(ctx, gi);
                            if (tid != ~0u) p += snprintf(tgt+p, sizeof(tgt)-p, "%st%u", gi?",":"", tid);
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
            else if (ext == UOP_WHERE) arity = 3;
            else if (ext == UOP_FUSING) arity = 0;
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

                    if (cval < ctx->heap_pos) {
                        Term nt = heap_read(ctx, cval);
                        u32 n_tgt = (term_tag(nt) == TAG_NUM) ? (u32)term_val(nt) : 0;
                        if (n_tgt > 64) n_tgt = 64;
                        for (u32 gi = 0; gi < n_tgt; gi++) {
                            Term pterm = heap_read(ctx, cval + 1 + 2*gi);
                            Term sterm = heap_read(ctx, cval + 1 + 2*gi + 1);
                            char plbl[16], slbl[16];
                            snprintf(plbl, sizeof(plbl), "p%u", gi);
                            snprintf(slbl, sizeof(slbl), "slot%u", gi);
                            #define EMIT_CTR_CHILD(cterm, clbl, cpos) do { \
                                u8 _ct = term_tag((cterm)); \
                                u64 _cv = term_val((cterm)); \
                                if (_ct == TAG_TEN) { \
                                    EMIT_TEN((u32)_cv); \
                                    fprintf(f, "  t%u -> ctr%llu [label=\"%s\"];\n", (u32)_cv, cval, (clbl)); \
                                } else if (_ct == TAG_TOP) { \
                                    fprintf(f, "  n%llu -> ctr%llu [label=\"%s\"];\n", _cv, cval, (clbl)); \
                                } else if (_ct == TAG_DP0 || _ct == TAG_DP1) { \
                                    u64 _dl = _cv; \
                                    DP_SLOT_MARK(cpos); \
                                    EMIT_DUP_CHAIN(_dl); \
                                    if (heap_dot_hl_on && (u64)(cpos) == heap_dot_hl_slot) \
                                        fprintf(f, "  dup%llu -> ctr%llu [label=\"%s%s\",color=\"#cc0000\",penwidth=2.0];\n", _dl, cval, (clbl), _ct==TAG_DP1 ? " (dp1)" : " (dp0)"); \
                                    else \
                                        fprintf(f, "  dup%llu -> ctr%llu [label=\"%s%s\"];\n", _dl, cval, (clbl), _ct==TAG_DP1 ? " (dp1)" : " (dp0)"); \
                                } else if (_ct == TAG_NUM) { \
                                    f32 _fv; u32 _bv=(u32)_cv; memcpy(&_fv,&_bv,4); \
                                    fprintf(f, "  num_ctr%llu_%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", (unsigned long long)cval, (unsigned long long)(cpos), (double)_fv); \
                                    fprintf(f, "  num_ctr%llu_%llu -> ctr%llu [label=\"%s\"];\n", cval, (unsigned long long)(cpos), cval, (clbl)); \
                                } else if (_ct == TAG_ERA) { \
                                    if (_cv != 0) { \
                                        EMIT_ERA_NODE((cpos), (cterm)); \
                                        fprintf(f, "  era%llu -> ctr%llu [label=\"%s\"];\n", (unsigned long long)(cpos), cval, (clbl)); \
                                    } \
                                } else { \
                                    EMIT_RAW_NODE(_cv, (cterm)); \
                                    fprintf(f, "  h%llu -> ctr%llu [label=\"%s\"];\n", (unsigned long long)_cv, cval, (clbl)); \
                                } \
                            } while(0)
                            EMIT_CTR_CHILD(pterm, plbl, cval + 1 + 2*gi);
                            EMIT_CTR_CHILD(sterm, slbl, cval + 1 + 2*gi + 1);
                            #undef EMIT_CTR_CHILD
                        }
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
                } else if (ctag == TAG_TOP) {
                    if (rev) {
                        if (edge_hl) fprintf(f, "  n%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", val, cval, elbl);
                        else         fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n", val, cval, elbl);
                    } else {
                        if (edge_hl) fprintf(f, "  n%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", cval, val, elbl);
                        else         fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n", cval, val, elbl);
                    }
                }
            }
            nn++;
            continue;
        }

        // --- CTR: skip (GRAD parameter, not standalone node) ---
        if (tag == TAG_CTR) continue;

        // --- SUP ---
        if (tag == TAG_SUP) {
            if (NODE_SEEN(val)) continue; NODE_MARK(val);
            fprintf(f, "  n%llu [label=\"SUP #%u\", shape=hexagon, fillcolor=\"#e4d6fc\"];\n", val, ext);
            nn++;
            continue;
        }

        // Skip TAG_TEN, TAG_NUM at heap positions (shown when referenced as children)
    }

    // FUSING leaf edges (for post-schedule graphs)
    {
        extern KernelEntry sched_kernels[];
        extern u32 sched_kernel_count;
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term t = ctx->heap[h];
            if (term_tag(t) != TAG_TOP || term_ext(t) != UOP_FUSING) continue;
            u64 loc = term_val(t);
            Term kid_t = heap_read(ctx, loc + 1);
            if (term_tag(kid_t) != TAG_NUM) continue;
            u32 kid = (u32)term_val(kid_t);
            if (kid >= sched_kernel_count) continue;
            KernelEntry *ke = &sched_kernels[kid];
            for (u32 li = 0; li < ke->n_leaves; li++) {
                Term lt = ke->leaf_terms[li];
                for (int _r = 0; _r < 10; _r++) {
                    if (term_tag(lt)==TAG_TOP && is_view_op(term_ext(lt))) { lt = heap_read(ctx, term_val(lt)); continue; }
                    if (term_tag(lt)==TAG_DP0||term_tag(lt)==TAG_DP1) { lt = heap_read(ctx, term_val(lt)); continue; }
                    break;
                }
                if (term_tag(lt) == TAG_TEN) {
                    EMIT_TEN((u32)term_val(lt));
                    fprintf(f, "  t%u -> n%llu [style=dashed,color=\"#009900\"];\n", (u32)term_val(lt), loc);
                } else if (term_tag(lt) == TAG_TOP)
                    fprintf(f, "  n%llu -> n%llu [style=dashed,color=\"#009900\"];\n", term_val(lt), loc);
            }
        }
    }

    fprintf(f, "}\n");
    fclose(f);
    free(dp_slot_emitted);
    free(top_live);
    fprintf(stderr, "heap_dot: %u nodes → %s\n", nn, path);
    #undef EMIT_TEN
    #undef EMIT_RAW_NODE
    #undef RAW_NODE_KEY
    #undef EMIT_CTR_NODE
    #undef CTR_NODE_KEY
    #undef EMIT_ANY_NODE
    #undef ANY_NODE_KEY
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
