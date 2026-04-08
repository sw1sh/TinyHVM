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

    // Collect unique terms on the heap — all combinators, not just TAG_TOP
    #define HDOT_MAX 4096
    typedef struct { Term term; u64 loc; } HNode;
    HNode nodes[HDOT_MAX]; u32 nn = 0;
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term t = ctx->heap[h];
        u8 tag = term_tag(t);
        // Skip pure leaves and already-visited terms
        if (tag == TAG_ERA || tag == TAG_NUM || tag == TAG_VAR || tag == TAG_ANY) continue;
        if (tag == TAG_TEN) continue; // emitted inline when referenced
        if (tag == TAG_DP0 || tag == TAG_DP1) continue; // handled by DUP pass
        if (tag == TAG_TOP && term_ext(t) == UOP_GRAD) continue;
        // Need a heap location to walk children
        u32 arity = tag_arity(tag, term_ext(t));
        if (arity == 0) continue;
        // Dedup
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

    // Pass 1: collect all DUP locations and their consumers
    #define HDOT_DUP_MAX 512
    typedef struct { u64 loc; u64 dp0_consumer; u64 dp1_consumer; } DupInfo;
    DupInfo dups[HDOT_DUP_MAX]; u32 n_dups = 0;

    for (u32 ni = 0; ni < nn; ni++) {
        u64 loc = term_val(nodes[ni].term);
        u32 ext = term_ext(nodes[ni].term);
        u32 arity = (ext == UOP_WHERE) ? 3 : 2;
        for (u32 ai = 0; ai < arity; ai++) {
            Term child = heap_read(ctx, loc + ai);
            if (term_tag(child) != TAG_DP0 && term_tag(child) != TAG_DP1) continue;
            u64 dp_loc = term_val(child);
            int is_dp1 = (term_tag(child) == TAG_DP1);
            // Find or create DupInfo
            u32 di;
            for (di = 0; di < n_dups; di++) if (dups[di].loc == dp_loc) break;
            if (di == n_dups && n_dups < HDOT_DUP_MAX) {
                dups[n_dups++] = (DupInfo){dp_loc, 0, 0};
            }
            if (di < n_dups) {
                if (is_dp1 && !dups[di].dp1_consumer) dups[di].dp1_consumer = loc;
                if (!is_dp1 && !dups[di].dp0_consumer) dups[di].dp0_consumer = loc;
            }
        }
    }

    // Pass 2: emit DUP triangles — only those with BOTH dp0 and dp1 consumers
    for (u32 di = 0; di < n_dups; di++) {
        DupInfo *d = &dups[di];
        if (!d->dp0_consumer || !d->dp1_consumer) continue; // skip dead/single-output DUPs
        fprintf(f, "  dup%llu [label=\"DUP\", shape=invtriangle, fillcolor=\"#d4b8e8\", "
                "fontsize=9, width=0.7, height=0.5];\n", d->loc);
        // Input: shared value → DUP
        Term shared = heap_read(ctx, d->loc);
        if (term_tag(shared) == TAG_TOP)
            fprintf(f, "  n%llu -> dup%llu;\n", term_val(shared), d->loc);
        else if (term_tag(shared) == TAG_TEN) {
            EMIT_TEN((u32)term_val(shared));
            fprintf(f, "  t%u -> dup%llu;\n", (u32)term_val(shared), d->loc);
        } else if (term_tag(shared) == TAG_DP0 || term_tag(shared) == TAG_DP1) {
            // Nested: input comes from another DUP
            u64 inner_loc = term_val(shared);
            fprintf(f, "  dup%llu -> dup%llu [label=\"%s\"];\n", inner_loc, d->loc,
                    term_tag(shared)==TAG_DP1 ? "dp1" : "dp0");
        }
        // Output dp0
        if (d->dp0_consumer)
            fprintf(f, "  dup%llu -> n%llu [label=\"dp0\"];\n", d->loc, d->dp0_consumer);
        // Output dp1
        if (d->dp1_consumer)
            fprintf(f, "  dup%llu -> n%llu [label=\"dp1\"];\n", d->loc, d->dp1_consumer);
    }

    // Pass 3: emit compute nodes + direct edges (non-DUP children)
    for (u32 ni = 0; ni < nn; ni++) {
        Term t = nodes[ni].term;
        u8 tag = term_tag(t);
        u64 loc = term_val(t);
        u32 ext = term_ext(t);
        // Label and color by tag
        char label[96]; const char *color; const char *nshape = "box";
        if (tag == TAG_TOP) {
            const char *opn = (ext < UOP_COUNT) ? uop_names[ext] : "?";
            const View *v = st_get(loc);
            char sh[64] = "";
            if (v) { int p = 0; for (u32 d = 0; d < v->shape.rank && p < 50; d++)
                p += snprintf(sh+p, sizeof(sh)-p, "%s%u", d?",":"", v->shape.dims[d]); }
            if (ext == UOP_FUSING) {
                // Show kernel ID + fused op summary
                Term kid_t = heap_read(ctx, loc + 1);
                u32 kid = (term_tag(kid_t) == TAG_NUM) ? (u32)term_val(kid_t) : 0;
                extern KernelEntry sched_kernels[];
                KernelEntry *ke = &sched_kernels[kid];
                char ops_str[64] = "";
                if (ke->n_ops > 0 || ke->has_reduce) {
                    int p = 0;
                    if (ke->has_reduce) {
                        const char *rn = (ke->has_reduce == UOP_SUM) ? "SUM" :
                                         (ke->has_reduce == UOP_RMAX) ? "RMAX" : "?";
                        p += snprintf(ops_str+p, sizeof(ops_str)-p, "%s", rn);
                        if (ke->n_ops > 0) p += snprintf(ops_str+p, sizeof(ops_str)-p, "+");
                    }
                    // Unique op names in the kernel
                    u8 seen[UOP_COUNT] = {0};
                    for (u32 oi = 0; oi < ke->n_ops && p < 50; oi++) {
                        u32 u = ke->ops[oi].uop;
                        if (u < UOP_COUNT && !seen[u]) {
                            seen[u] = 1;
                            if (p > 0 && ops_str[p-1] != '+')
                                p += snprintf(ops_str+p, sizeof(ops_str)-p, "+");
                            p += snprintf(ops_str+p, sizeof(ops_str)-p, "%s", uop_names[u]);
                        }
                    }
                }
                snprintf(label, sizeof(label), "K%u: %s\\n[%s]\\nops=%u lv=%u",
                         kid, ops_str, sh, ke->n_ops, ke->n_leaves);
                color = "#ccffcc"; nshape = "box";
            } else {
                snprintf(label, sizeof(label), "%s\\n[%s]", opn, sh);
                if (ext == UOP_ASSIGN)                      color = "#ffd700";
                else if (is_elementwise(ext))               color = "#cce5ff";
                else if (ext == UOP_SUM || ext == UOP_RMAX) color = "#ffcccc";
                else if (is_view_op(ext))                   color = "#fff3cd";
                else if (ext == UOP_MM)                     color = "#ffccff";
                else                                        color = "#f0f0f0";
            }
        } else if (tag == TAG_APP) {
            snprintf(label, sizeof(label), "APP"); color = "#fce4d6"; // orange
        } else if (tag == TAG_LAM) {
            snprintf(label, sizeof(label), "LAM"); color = "#d6fce4"; // mint
        } else if (tag == TAG_SUP) {
            snprintf(label, sizeof(label), "SUP\\n#%u", ext); color = "#e4d6fc"; // lavender
            nshape = "hexagon";
        } else if (tag == TAG_OP2) {
            snprintf(label, sizeof(label), "OP2"); color = "#d6e4fc";
        } else if (tag == TAG_USP) {
            snprintf(label, sizeof(label), "USP\\n#%u", ext); color = "#fcd6e4";
        } else if (tag == TAG_MAT) {
            snprintf(label, sizeof(label), "MAT"); color = "#e4fcd6";
        } else if (tag == TAG_CTR) {
            snprintf(label, sizeof(label), "CTR/%u", ext); color = "#d6fcd6";
        } else {
            snprintf(label, sizeof(label), "tag%u", tag); color = "#f0f0f0";
        }

        fprintf(f, "  n%llu [label=\"%s\", shape=%s, fillcolor=\"%s\"];\n",
                loc, label, nshape, color);

        // Edges to children (skip FUSING internal metadata)
        u32 arity = tag_arity(tag, ext);
        if (tag == TAG_TOP && ext == UOP_FUSING) arity = 0; // FUSING children are metadata, not data
        for (u32 ai = 0; ai < arity; ai++) {
            Term child = heap_read(ctx, loc + ai);
            u8 ctag = term_tag(child);
            u64 cval = term_val(child);
            if (ctag == TAG_ERA) continue; // skip ERA children (clutter)
            if (ctag == TAG_DP0 || ctag == TAG_DP1) {
                // Check if DUP triangle exists (both consumers present)
                u64 dp_loc = cval;
                int has_both = 0;
                for (u32 di = 0; di < n_dups; di++)
                    if (dups[di].loc == dp_loc && dups[di].dp0_consumer && dups[di].dp1_consumer)
                        { has_both = 1; break; }
                if (has_both) continue; // handled by DUP pass
                // Single-output DUP: draw direct edge from shared value
                Term shared = heap_read(ctx, dp_loc);
                u8 stag = term_tag(shared);
                if (stag == TAG_TEN) {
                    EMIT_TEN((u32)term_val(shared));
                    fprintf(f, "  t%u -> n%llu;\n", (u32)term_val(shared), loc);
                } else if (stag != TAG_ERA && stag != TAG_NUM)
                    fprintf(f, "  n%llu -> n%llu;\n", term_val(shared), loc);
            } else if (ctag == TAG_TEN) {
                EMIT_TEN((u32)cval);
                fprintf(f, "  t%u -> n%llu;\n", (u32)cval, loc);
            } else if (ctag == TAG_ERA) {
                fprintf(f, "  era%llu_%u [label=\"ERA\", shape=point, width=0.15];\n", loc, ai);
                fprintf(f, "  era%llu_%u -> n%llu;\n", loc, ai, loc);
            } else if (ctag == TAG_NUM) {
                continue; // skip scalar constants (clutter)
            } else if (ctag == TAG_VAR) {
                fprintf(f, "  var%llu [label=\"VAR\", shape=circle, fillcolor=\"#ffffcc\", width=0.3, fontsize=8];\n", cval);
                fprintf(f, "  var%llu -> n%llu;\n", cval, loc);
            } else {
                // Other combinator: link by heap location
                fprintf(f, "  n%llu -> n%llu;\n", cval, loc);
            }
        }
    }
    // Pass 4: FUSING leaf edges — connect each FUSING kernel to its data dependencies
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
                // Resolve through views/DP to find the source node
                for (int _r = 0; _r < 10; _r++) {
                    if (term_tag(lt)==TAG_DP0||term_tag(lt)==TAG_DP1)
                        { lt = heap_read(ctx, term_val(lt)); continue; }
                    if (term_tag(lt)==TAG_TOP && is_view_op(term_ext(lt)))
                        { lt = heap_read(ctx, term_val(lt)); continue; }
                    break;
                }
                if (term_tag(lt) == TAG_TOP) {
                    fprintf(f, "  n%llu -> n%llu;\n", term_val(lt), loc);
                } else if (term_tag(lt) == TAG_TEN) {
                    u32 tid = (u32)term_val(lt);
                    EMIT_TEN(tid);
                    fprintf(f, "  t%u -> n%llu;\n", tid, loc);
                }
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
