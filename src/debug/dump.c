// debug/dump.c — Dump tensor provenance graph as DOT (graphviz) or JSON
// Usage: thvm_dump_dot(ctx, "graph.dot") or thvm_dump_json(ctx, "graph.json")
// Visualize: dot -Tpng graph.dot -o graph.png

#include <stdio.h>

static void thvm_dump_dot(TinyHVM *ctx, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "dump_dot: can't open %s\n", path); return; }

    fprintf(f, "digraph G {\n");
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

// Dump live heap compute graph as DOT (tinygrad-style: clean, compact, colored by op type).
// Walks TAG_TOP nodes on heap. Skips view ops (shows as edge labels).
// Colors: blue=ew, red=reduce, green=FUSING, yellow=view, gray=tensor, purple=DUP.
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root);
static void thvm_heap_dot(TinyHVM *ctx, const char *path) {
    thvm_heap_dot_root(ctx, path, term_era());
}
// Faithful heap graph: shows every reachable term exactly as stored.
// No filtering, no skipping. Disconnected nodes = real structural problem.
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "heap_dot: can't open %s\n", path); return; }
    fprintf(f, "digraph G {\n");
    fprintf(f, "  rankdir=BT;\n");
    fprintf(f, "  node [fontname=\"Helvetica\", fontsize=10, style=filled, shape=box, margin=\"0.1,0.05\"];\n");
    fprintf(f, "  edge [fontsize=8, fontname=\"Helvetica\"];\n\n");

    // BFS from roots — collect ALL reachable terms
    #define HDOT_MAX 4096
    typedef struct { Term term; } HNode;
    HNode nodes[HDOT_MAX]; u32 nn = 0;
    Term bfs[HDOT_MAX]; u32 qh = 0, qt = 0;

    #define HDOT_ENQ(t) do { \
        Term _t = (t); \
        int _dup = 0; \
        for (u32 _i = 0; _i < nn; _i++) if (nodes[_i].term == _t) { _dup = 1; break; } \
        if (!_dup && nn < HDOT_MAX) { nodes[nn++].term = _t; bfs[qt++] = _t; } \
    } while(0)

    // Seed from roots: explicit root term + ASSIGN/FUSING on heap
    if (term_tag(root) != TAG_ERA) HDOT_ENQ(root);
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term ht = ctx->heap[h];
        if (term_tag(ht) == TAG_TOP) {
            u32 ext = term_ext(ht);
            if (ext == UOP_ASSIGN || ext == UOP_FUSING)
                HDOT_ENQ(ht);
        }
    }
    // If no roots found (all reduced), seed ALL non-leaf heap terms
    if (nn == 0) {
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term ht = ctx->heap[h];
            u8 tag = term_tag(ht);
            if (tag == TAG_ERA || tag == TAG_NUM || tag == TAG_TEN ||
                tag == TAG_VAR || tag == TAG_ANY || tag == TAG_DP0 ||
                tag == TAG_DP1 || tag >= TAG_COUNT) continue;
            if (tag == TAG_TOP && term_ext(ht) == UOP_GRAD) continue;
            HDOT_ENQ(ht);
        }
    }
    // BFS: walk children (skip FUSING metadata)
    while (qh < qt) {
        Term t = bfs[qh++];
        u8 tag = term_tag(t); u32 ext = term_ext(t);
        if (tag == TAG_TOP && ext == UOP_FUSING) continue; // FUSING heap children are metadata
        u32 arity = tag_arity(tag, ext);
        u64 loc = term_val(t);
        for (u32 ai = 0; ai < arity; ai++) {
            Term child = heap_read(ctx, loc + ai);
            u8 ctag = term_tag(child);
            if (ctag == TAG_DP0 || ctag == TAG_DP1) {
                // Enqueue the shared value behind the DUP
                Term shared = heap_read(ctx, term_val(child));
                if (term_tag(shared) != TAG_ERA && term_tag(shared) != TAG_NUM &&
                    term_tag(shared) != TAG_TEN && term_tag(shared) != TAG_VAR)
                    HDOT_ENQ(shared);
            } else if (ctag != TAG_ERA && ctag != TAG_NUM && ctag != TAG_TEN &&
                       ctag != TAG_VAR && ctag != TAG_ANY) {
                HDOT_ENQ(child);
            }
        }
    }
    #undef HDOT_ENQ

    // --- Emit nodes and edges ---

    // Collect DUP info — each DP reference becomes an output edge
    #define HDOT_DUP_MAX 512
    #define HDOT_DUP_EDGE_MAX 1024
    typedef struct { u64 dp_loc; u64 consumer_loc; int is_dp1; } DupEdge;
    DupEdge dup_edges[HDOT_DUP_EDGE_MAX]; u32 n_dup_edges = 0;
    // Unique DUP locations
    u64 dup_locs[HDOT_DUP_MAX]; u32 n_dup_locs = 0;

    for (u32 ni = 0; ni < nn; ni++) {
        Term t = nodes[ni].term;
        u32 arity = tag_arity(term_tag(t), term_ext(t));
        u64 loc = term_val(t);
        for (u32 ai = 0; ai < arity; ai++) {
            Term c = heap_read(ctx, loc + ai);
            if (term_tag(c) != TAG_DP0 && term_tag(c) != TAG_DP1) continue;
            u64 dl = term_val(c);
            // Record edge
            if (n_dup_edges < HDOT_DUP_EDGE_MAX)
                dup_edges[n_dup_edges++] = (DupEdge){dl, loc, term_tag(c) == TAG_DP1};
            // Record unique location
            int found = 0;
            for (u32 di = 0; di < n_dup_locs; di++) if (dup_locs[di] == dl) { found = 1; break; }
            if (!found && n_dup_locs < HDOT_DUP_MAX) dup_locs[n_dup_locs++] = dl;
        }
    }
    // Emit ALL DUP triangles — every DP0/DP1 on the heap is a real IC node.
    // Tensors and ops must have single output; DUP is required for sharing.
    for (u32 di = 0; di < n_dup_locs; di++) {
        u64 dl = dup_locs[di];
        Term shared = heap_read(ctx, dl);
        u8 stag = term_tag(shared);
        fprintf(f, "  dup%llu [label=\"DUP\", shape=invtriangle, fillcolor=\"#d4b8e8\", fontsize=9, width=0.7, height=0.5];\n", dl);
        // Input edge: shared value → DUP
        if (stag == TAG_TOP)
            fprintf(f, "  n%llu -> dup%llu;\n", term_val(shared), dl);
        else if (stag == TAG_TEN)
            fprintf(f, "  t%u -> dup%llu;\n", (u32)term_val(shared), dl);
        else if (stag == TAG_DP0 || stag == TAG_DP1)
            fprintf(f, "  dup%llu -> dup%llu [label=\"%s\"];\n", term_val(shared), dl,
                    stag==TAG_DP1?"dp1":"dp0");
        else if (stag == TAG_NUM) {
            f32 fv; u32 bv=(u32)term_val(shared); memcpy(&fv,&bv,4);
            fprintf(f, "  num_dup%llu [label=\"%.4g\",shape=plaintext,fontsize=8];\n", dl, fv);
            fprintf(f, "  num_dup%llu -> dup%llu;\n", dl, dl);
        }
        // Output edges: dp0 + dp1
        int has_dp0 = 0, has_dp1 = 0;
        for (u32 ei = 0; ei < n_dup_edges; ei++) {
            if (dup_edges[ei].dp_loc != dl) continue;
            const char *lbl = dup_edges[ei].is_dp1 ? "dp1" : "dp0";
            fprintf(f, "  dup%llu -> n%llu [label=\"%s\"];\n", dl, dup_edges[ei].consumer_loc, lbl);
            if (dup_edges[ei].is_dp1) has_dp1 = 1; else has_dp0 = 1;
        }
        // Dangling stubs for missing ports (structural error)
        if (!has_dp0) {
            fprintf(f, "  dang%llu_0 [label=\"?\",shape=plain,fontsize=8,fontcolor=red];\n", dl);
            fprintf(f, "  dup%llu -> dang%llu_0 [label=\"dp0\",style=dotted,color=red];\n", dl, dl);
        }
        if (!has_dp1) {
            fprintf(f, "  dang%llu_1 [label=\"?\",shape=plain,fontsize=8,fontcolor=red];\n", dl);
            fprintf(f, "  dup%llu -> dang%llu_1 [label=\"dp1\",style=dotted,color=red];\n", dl, dl);
        }
    }
    // Helper: check if a DP reference is handled by a DUP triangle
    #define IS_DUP_HANDLED(dp_loc) ({ \
        int _h = 0; for (u32 _di = 0; _di < n_dup_locs; _di++) \
            if (dup_locs[_di] == (dp_loc)) { _h = 1; break; } _h; })

    // Emit all nodes + child edges
    for (u32 ni = 0; ni < nn; ni++) {
        Term t = nodes[ni].term;
        u8 tag = term_tag(t); u32 ext = term_ext(t); u64 loc = term_val(t);

        // --- Node label + color ---
        char label[128]; const char *color = "#f0f0f0"; const char *nshape = "box";
        if (tag == TAG_TOP) {
            const char *opn = (ext < UOP_COUNT) ? uop_names[ext] : "?";
            const View *v = st_get(loc);
            char sh[64] = "";
            if (v) { int p = 0; for (u32 d = 0; d < v->shape.rank && p < 50; d++)
                p += snprintf(sh+p, sizeof(sh)-p, "%s%u", d?",":"", v->shape.dims[d]); }
            if (ext == UOP_FUSING) {
                Term kid_t = heap_read(ctx, loc + 1);
                u32 kid = (term_tag(kid_t) == TAG_NUM) ? (u32)term_val(kid_t) : 0;
                extern KernelEntry sched_kernels[];
                KernelEntry *ke = &sched_kernels[kid];
                char ops_s[64] = ""; int p = 0;
                if (ke->has_reduce) { const char *rn = ke->has_reduce==UOP_SUM?"SUM":"RMAX";
                    p += snprintf(ops_s+p,sizeof(ops_s)-p,"%s",rn);
                    if (ke->n_ops) p += snprintf(ops_s+p,sizeof(ops_s)-p,"+"); }
                u8 seen[UOP_COUNT]={0};
                for (u32 oi=0; oi<ke->n_ops && p<50; oi++) {
                    u32 u=ke->ops[oi].uop;
                    if (u<UOP_COUNT && !seen[u]) { seen[u]=1;
                        if (p>0 && ops_s[p-1]!='+') p+=snprintf(ops_s+p,sizeof(ops_s)-p,"+");
                        p+=snprintf(ops_s+p,sizeof(ops_s)-p,"%s",uop_names[u]); }}
                snprintf(label,sizeof(label),"K%u: %s\\n[%s]\\nops=%u lv=%u",kid,ops_s,sh,ke->n_ops,ke->n_leaves);
                color = "#ccffcc";
            } else {
                snprintf(label, sizeof(label), "%s\\n[%s]", opn, sh);
                if (ext == UOP_ASSIGN) color = "#ffd700";
                else if (is_elementwise(ext)) color = "#cce5ff";
                else if (ext == UOP_SUM || ext == UOP_RMAX) color = "#ffcccc";
                else if (is_view_op(ext)) color = "#fff3cd";
                else if (ext == UOP_MM) color = "#ffccff";
            }
        } else if (tag == TAG_APP) { snprintf(label,sizeof(label),"APP"); color="#fce4d6"; }
        else if (tag == TAG_LAM) { snprintf(label,sizeof(label),"LAM"); color="#d6fce4"; }
        else if (tag == TAG_SUP) { snprintf(label,sizeof(label),"SUP #%u",ext); color="#e4d6fc"; nshape="hexagon"; }
        else if (tag == TAG_CTR) { snprintf(label,sizeof(label),"CTR/%u",ext); color="#d6fcd6"; nshape="octagon"; }
        else if (tag == TAG_OP2) { snprintf(label,sizeof(label),"OP2"); color="#d6e4fc"; }
        else if (tag == TAG_REF) { snprintf(label,sizeof(label),"REF"); color="#f0e0e0"; nshape="ellipse"; }
        else if (tag == TAG_BRI) { snprintf(label,sizeof(label),"BRI"); color="#e0f0e0"; }
        else if (tag == TAG_ANN) { snprintf(label,sizeof(label),"ANN"); color="#f0f0e0"; }
        else snprintf(label,sizeof(label),"tag%u/%u",tag,ext);

        fprintf(f, "  n%llu [label=\"%s\", shape=%s, fillcolor=\"%s\"];\n", loc, label, nshape, color);

        // Debug: print children tags for nodes with no drawn input edges
        if (getenv("THVM_GRAPH_DEBUG")) {
            u32 _ar = tag_arity(tag, ext);
            fprintf(stderr, "  n%llu [%s] arity=%u:", loc, label, _ar);
            for (u32 _ai = 0; _ai < _ar; _ai++) {
                Term _c = heap_read(ctx, loc + _ai);
                fprintf(stderr, " c%u=tag%u/ext%u/val%llu", _ai, term_tag(_c), term_ext(_c), term_val(_c));
            }
            fprintf(stderr, "\n");
        }

        // --- Child edges with labels ---
        u32 arity = tag_arity(tag, ext);
        if (tag == TAG_TOP && ext == UOP_FUSING) arity = 0;
        // Unary ops: only show first input (second is ERA padding)
        if (tag == TAG_TOP && !is_binary(ext) && is_elementwise(ext)) arity = 1;
        for (u32 ai = 0; ai < arity; ai++) {
            Term child = heap_read(ctx, loc + ai);
            u8 ctag = term_tag(child); u64 cval = term_val(child);
            // Edge label based on parent op + child index
            const char *elbl = "";
            if (tag == TAG_TOP) {
                if (ext == UOP_ASSIGN) elbl = ai==0 ? "tgt" : "src";
                else if (ext >= UOP_RESHAPE && ext <= UOP_PAD) elbl = ai==0 ? "in" : "shape";
                else if (ext == UOP_SUM || ext == UOP_RMAX) elbl = ai==0 ? "in" : "axes";
                else if (ext == UOP_GRAD) elbl = ai==0 ? "y" : ai==1 ? "gy" : "x";
                else if (is_binary(ext)) elbl = ai==0 ? "a" : "b";
            } else if (tag == TAG_APP) elbl = ai==0 ? "fn" : "arg";
            else if (tag == TAG_LAM) elbl = ai==0 ? "var" : "body";

            if (ctag == TAG_DP0 || ctag == TAG_DP1) {
                continue; // handled by DUP triangle pass
            } else if (ctag == TAG_TEN) {
                char _sh[64]=""; int _p=0;
                if (cval < ctx->tensor_count) { TensorMeta *_m=&ctx->tensors[cval];
                    for (u32 _d=0;_d<_m->view.shape.rank;_d++)
                        _p+=snprintf(_sh+_p,sizeof(_sh)-_p,"%s%u",_d?",":"",_m->view.shape.dims[_d]); }
                fprintf(f, "  t%u [label=\"t%u\\n[%s]\",shape=triangle,fillcolor=\"#e0e0e0\"];\n",(u32)cval,(u32)cval,_sh);
                fprintf(f, "  t%u -> n%llu [label=\"%s\"];\n", (u32)cval, loc, elbl);
            } else if (ctag == TAG_ERA) {
                fprintf(f, "  era%llu_%u [label=\"\",shape=point,width=0.1];\n", loc, ai);
                fprintf(f, "  era%llu_%u -> n%llu [label=\"%s\"];\n", loc, ai, loc, elbl);
            } else if (ctag == TAG_NUM) {
                f32 fv; u32 bv=(u32)cval; memcpy(&fv,&bv,4);
                fprintf(f, "  num%llu_%u [label=\"%.4g\",shape=plaintext,fontsize=8];\n", loc, ai, fv);
                fprintf(f, "  num%llu_%u -> n%llu [label=\"%s\"];\n", loc, ai, loc, elbl);
            } else if (ctag == TAG_VAR) {
                fprintf(f, "  var%llu [label=\"VAR\",shape=circle,fillcolor=\"#ffffcc\",width=0.3,fontsize=8];\n", cval);
                fprintf(f, "  var%llu -> n%llu [label=\"%s\"];\n", cval, loc, elbl);
            } else {
                fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n", cval, loc, elbl);
            }
        }
    }

    // FUSING kernel leaf edges (these are the data dependencies, not heap children)
    {
        extern KernelEntry sched_kernels[];
        extern u32 sched_kernel_count;
        for (u32 ni = 0; ni < nn; ni++) {
            Term t = nodes[ni].term;
            if (term_tag(t) != TAG_TOP || term_ext(t) != UOP_FUSING) continue;
            u64 loc = term_val(t);
            Term kid_t = heap_read(ctx, loc + 1);
            if (term_tag(kid_t) != TAG_NUM) continue;
            u32 kid = (u32)term_val(kid_t);
            if (kid >= sched_kernel_count) continue;
            KernelEntry *ke = &sched_kernels[kid];
            for (u32 li = 0; li < ke->n_leaves; li++) {
                Term lt = ke->leaf_terms[li];
                // Resolve through views/DP
                for (int _r = 0; _r < 10; _r++) {
                    if (term_tag(lt)==TAG_DP0||term_tag(lt)==TAG_DP1)
                        { lt = heap_read(ctx, term_val(lt)); continue; }
                    if (term_tag(lt)==TAG_TOP && is_view_op(term_ext(lt)))
                        { lt = heap_read(ctx, term_val(lt)); continue; }
                    break;
                }
                if (term_tag(lt) == TAG_TOP)
                    fprintf(f, "  n%llu -> n%llu [style=dashed, color=\"#009900\"];\n", term_val(lt), loc);
                else if (term_tag(lt) == TAG_TEN)
                    fprintf(f, "  t%u -> n%llu [style=dashed, color=\"#009900\"];\n", (u32)term_val(lt), loc);
            }
        }
    }

    fprintf(f, "}\n");
    fclose(f);
    fprintf(stderr, "heap_dot: %u nodes → %s\n", nn, path);
    #undef HDOT_MAX
    #undef HDOT_DUP_MAX
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
