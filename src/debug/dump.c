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
static void thvm_heap_dot(TinyHVM *ctx, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "heap_dot: can't open %s\n", path); return; }
    fprintf(f, "digraph G {\n");
    fprintf(f, "  rankdir=BT;\n");
    fprintf(f, "  node [fontname=\"Helvetica\", fontsize=10, style=filled, shape=box, margin=\"0.1,0.05\"];\n");
    fprintf(f, "  edge [fontsize=8, fontname=\"Helvetica\"];\n\n");

    // Collect unique terms on the heap (skip duplicates from multi-consumer propagation)
    #define HDOT_MAX 4096
    typedef struct { Term term; u64 loc; } HNode;
    HNode nodes[HDOT_MAX]; u32 nn = 0;
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term t = ctx->heap[h];
        u8 tag = term_tag(t);
        if (tag != TAG_TOP) continue;
        u32 ext = term_ext(t);
        if (ext == UOP_ASSIGN || ext == UOP_GRAD) continue;
        // Dedup: skip if same term already recorded
        int dup = 0;
        for (u32 i = 0; i < nn; i++) if (nodes[i].term == t) { dup = 1; break; }
        if (dup) continue;
        if (nn < HDOT_MAX) nodes[nn++] = (HNode){t, h};
    }

    // Helper: emit a tensor leaf node (deduped by tid)
    #define EMIT_TEN(tid) do { \
        char _sh[64] = ""; int _p = 0; \
        if ((tid) < ctx->tensor_count) { \
            TensorMeta *_m = &ctx->tensors[tid]; \
            for (u32 _d = 0; _d < _m->view.shape.rank; _d++) \
                _p += snprintf(_sh+_p, sizeof(_sh)-_p, "%s%u", _d?",":"", _m->view.shape.dims[_d]); \
        } \
        fprintf(f, "  t%u [label=\"t%u\\n[%s]\", shape=ellipse, fillcolor=\"#e0e0e0\"];\n", (tid), (tid), _sh); \
    } while(0)

    // Helper: get the node ID string for a term (resolves to n<loc>, t<tid>, or dup<loc>)
    // Returns the node id in buf. Emits the node definition if needed.
    // For DP0/DP1: emits a single DUP diamond at the shared location.
    #define HDOT_DUP_MAX 512
    u64 dup_emitted[HDOT_DUP_MAX]; u32 n_dup_emitted = 0;

    // Emit nodes
    for (u32 ni = 0; ni < nn; ni++) {
        Term t = nodes[ni].term;
        u64 loc = term_val(t);
        u32 ext = term_ext(t);
        const char *opn = (ext < UOP_COUNT) ? uop_names[ext] : "?";

        // Shape from shape table
        const View *v = st_get(loc);
        char shape[80] = "";
        if (v) {
            int p = 0;
            for (u32 d = 0; d < v->shape.rank && p < 70; d++)
                p += snprintf(shape+p, sizeof(shape)-p, "%s%u", d?",":"", v->shape.dims[d]);
        }

        // Color by op type
        const char *color;
        if (is_elementwise(ext))                    color = "#cce5ff"; // blue
        else if (ext == UOP_SUM || ext == UOP_RMAX) color = "#ffcccc"; // red
        else if (ext == UOP_FUSING)                 color = "#ccffcc"; // green
        else if (is_view_op(ext))                   color = "#fff3cd"; // yellow
        else if (ext == UOP_MM)                     color = "#ffccff"; // pink
        else                                        color = "#f0f0f0"; // gray

        fprintf(f, "  n%llu [label=\"%s\\n[%s]\", fillcolor=\"%s\"];\n",
                loc, opn, shape, color);

        // Edges: walk children, render DUP as single 1-in-2-out diamond
        u32 arity = (ext == UOP_WHERE) ? 3 : 2;
        for (u32 ai = 0; ai < arity; ai++) {
            Term child = heap_read(ctx, loc + ai);

            if (term_tag(child) == TAG_DP0 || term_tag(child) == TAG_DP1) {
                u64 dp_loc = term_val(child);
                const char *port = (term_tag(child) == TAG_DP1) ? "dp1" : "dp0";

                // Emit single DUP node at dp_loc (once)
                int already = 0;
                for (u32 di = 0; di < n_dup_emitted; di++)
                    if (dup_emitted[di] == dp_loc) { already = 1; break; }
                if (!already && n_dup_emitted < HDOT_DUP_MAX) {
                    dup_emitted[n_dup_emitted++] = dp_loc;
                    fprintf(f, "  dup%llu [label=\"DUP\", shape=triangle, fillcolor=\"#d4b8e8\", "
                            "fontsize=9, width=0.6, height=0.5];\n", dp_loc);
                    // Input edge: shared value → DUP
                    Term shared = heap_read(ctx, dp_loc);
                    if (term_tag(shared) == TAG_TOP)
                        fprintf(f, "  n%llu -> dup%llu;\n", term_val(shared), dp_loc);
                    else if (term_tag(shared) == TAG_TEN) {
                        EMIT_TEN((u32)term_val(shared));
                        fprintf(f, "  t%u -> dup%llu;\n", (u32)term_val(shared), dp_loc);
                    } else if (term_tag(shared) == TAG_DP0 || term_tag(shared) == TAG_DP1) {
                        // Nested DUP: shared value is itself a DP ref
                        u64 inner_loc = term_val(shared);
                        fprintf(f, "  dup%llu -> dup%llu;\n", inner_loc, dp_loc);
                    }
                    // Scan heap for BOTH dp0 and dp1 consumers to ensure 2 outputs
                    // (the current node provides one; find the other)
                    for (u64 _ch = 1; _ch < ctx->heap_pos; _ch++) {
                        Term _ct = ctx->heap[_ch];
                        if (term_tag(_ct) != TAG_TOP) continue;
                        u32 _ce = term_ext(_ct);
                        if (_ce == UOP_ASSIGN || _ce == UOP_GRAD) continue;
                        u64 _cl = term_val(_ct);
                        u32 _ca = (_ce == UOP_WHERE) ? 3 : 2;
                        for (u32 _ai = 0; _ai < _ca; _ai++) {
                            Term _cc = heap_read(ctx, _cl + _ai);
                            if ((term_tag(_cc) == TAG_DP0 || term_tag(_cc) == TAG_DP1) &&
                                term_val(_cc) == dp_loc && _ct != nodes[ni].term) {
                                // Found the other consumer
                                const char *_op = (term_tag(_cc) == TAG_DP1) ? "dp1" : "dp0";
                                fprintf(f, "  dup%llu -> n%llu [label=\"%s\"];\n",
                                        dp_loc, _cl, _op);
                            }
                        }
                    }
                }
                // Output edge: DUP → consumer (this node), labeled dp0/dp1
                fprintf(f, "  dup%llu -> n%llu [label=\"%s\"];\n", dp_loc, term_val(nodes[ni].term), port);
            } else if (term_tag(child) == TAG_TOP) {
                fprintf(f, "  n%llu -> n%llu;\n", term_val(child), term_val(nodes[ni].term));
            } else if (term_tag(child) == TAG_TEN) {
                EMIT_TEN((u32)term_val(child));
                fprintf(f, "  t%u -> n%llu;\n", (u32)term_val(child), term_val(nodes[ni].term));
            }
        }
    }
    #undef EMIT_TEN

    fprintf(f, "}\n");
    fclose(f);
    fprintf(stderr, "heap_dot: %u nodes → %s\n", nn, path);
    #undef HDOT_MAX
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
