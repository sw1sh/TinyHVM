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

// Dump heap as DOT — flat walk, no BFS. Every combinator shown faithfully.
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root);
static void thvm_heap_dot(TinyHVM *ctx, const char *path) {
    thvm_heap_dot_root(ctx, path, term_era());
}
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "heap_dot: can't open %s\n", path); return; }
    fprintf(f, "digraph G {\n");
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
            fprintf(f,"  t%u [label=\"t%u\\n[%s]\\n%s %s%s\",shape=triangle,fillcolor=\"%s\"];\n", \
                    (tid),(tid),_sh,_dt,_bk,_m->requires_grad?" grad":"",_fc); \
        } } while(0)

    u32 nn = 0; // node count for stats
    // Track emitted node locations (dedup TAG_TOP by val)
    #define NODE_DEDUP_MAX 1024
    u64 node_emitted[NODE_DEDUP_MAX]; u32 n_node_emitted = 0;
    #define NODE_SEEN(v) ({ int _s=0; for(u32 _i=0;_i<n_node_emitted;_i++) \
        if(node_emitted[_i]==(v)){_s=1;break;} _s; })
    #define NODE_MARK(v) do { if(n_node_emitted<NODE_DEDUP_MAX) node_emitted[n_node_emitted++]=(v); } while(0)

    // Reachability: BFS from GRAD/ASSIGN/CTR to mark live node locations.
    // Unreachable nodes are being consumed by ERA — shown dimmed.
    #define REACH_MAX 4096
    u8 reach_live[REACH_MAX]; memset(reach_live, 0, sizeof(reach_live));
    {
        u64 rq[REACH_MAX]; u32 rh = 0, rt = 0;
        #define R_MARK(p) do { if((p)<REACH_MAX && !reach_live[p]) { reach_live[p]=1; rq[rt++]=p; } } while(0)
        for (u64 h = 1; h < ctx->heap_pos && h < REACH_MAX; h++) {
            Term gt = ctx->heap[h]; u8 gtag = term_tag(gt);
            if (gtag == TAG_TOP && (term_ext(gt)==UOP_GRAD||term_ext(gt)==UOP_ASSIGN)) R_MARK(h);
            else if (gtag == TAG_CTR || gtag == TAG_SUP) R_MARK(h);
        }
        while (rh < rt) {
            u64 pos = rq[rh++]; Term gt = ctx->heap[pos]; u8 gtag = term_tag(gt);
            if (gtag == TAG_CTR) {
                u64 cl = term_val(gt); R_MARK(cl);
                Term nt = ctx->heap[cl]; u32 n = (term_tag(nt)==TAG_NUM)?(u32)term_val(nt):0;
                for (u32 gi=0; gi<n && gi<16; gi++) { R_MARK(cl+1+2*gi); R_MARK(cl+1+2*gi+1); }
                continue;
            }
            u32 ar = 0;
            if (gtag == TAG_TOP) ar = (term_ext(gt)==UOP_GRAD||term_ext(gt)==UOP_WHERE)?3:2;
            else if (gtag == TAG_SUP || gtag == TAG_APP) ar = 2;
            else continue;
            u64 base = term_val(gt);
            for (u32 ci = 0; ci < ar; ci++) {
                u64 cp = base + ci; R_MARK(cp); Term ct = ctx->heap[cp];
                if (term_tag(ct)==TAG_DP0||term_tag(ct)==TAG_DP1) R_MARK(term_val(ct));
                else if (term_tag(ct)==TAG_TOP||term_tag(ct)==TAG_SUP||
                         term_tag(ct)==TAG_APP||term_tag(ct)==TAG_CTR) {
                    u64 cb = term_val(ct);
                    u32 ca = (term_tag(ct)==TAG_TOP)?((term_ext(ct)==UOP_GRAD||term_ext(ct)==UOP_WHERE)?3:2):2;
                    for (u32 cj = 0; cj < ca; cj++) R_MARK(cb + cj);
                }
            }
        }
        #undef R_MARK
    }
    // Check if a node at val is reachable (any of its child positions marked)
    #define IS_LIVE(v, ar) ({ int _l=0; for(u32 _i=0;_i<(ar)&&(v)+_i<REACH_MAX;_i++) \
        if(reach_live[(v)+_i]){_l=1;break;} _l; })

    // Flat walk: every heap position
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term t = ctx->heap[h];
        u8 tag = term_tag(t); u32 ext = term_ext(t); u64 val = term_val(t);

        // --- ERA: active (val!=0) shown as node with edge to target ---
        if (tag == TAG_ERA) {
            if (val == 0) continue;
            fprintf(f, "  era_h%llu [label=\"ERA\",shape=circle,width=0.3,fillcolor=\"#ff4444\","
                    "fontsize=8,fontcolor=white];\n", h);
            if (val < ctx->heap_pos) {
                Term tgt = ctx->heap[val];
                // Follow DP chains to find actual target
                for (int _dp=0; _dp<20 && (term_tag(tgt)==TAG_DP0||term_tag(tgt)==TAG_DP1); _dp++)
                    tgt = ctx->heap[term_val(tgt)];
                if (term_tag(tgt) == TAG_TOP)
                    fprintf(f, "  n%llu -> era_h%llu [color=\"#cc0000\",penwidth=2];\n", term_val(tgt), h);
                else if (term_tag(tgt) == TAG_TEN) {
                    EMIT_TEN((u32)term_val(tgt));
                    fprintf(f, "  t%u -> era_h%llu [color=\"#cc0000\",penwidth=2];\n", (u32)term_val(tgt), h);
                }
            }
            nn++;
            continue;
        }

        // --- TAG_TEN: only shown when referenced as child of a node ---
        if (tag == TAG_TEN) {
            // Don't emit standalone — EMIT_TEN is called when a parent renders children
            continue;
        }

        // --- DUP (DP0/DP1 references) ---
        if (tag == TAG_DP0 || tag == TAG_DP1) {
            // DUP nodes are emitted separately when referenced by parent nodes.
            // We collect them below.
            continue;
        }

        // --- TAG_TOP: op nodes (dedup by val) ---
        if (tag == TAG_TOP) {
            if (NODE_SEEN(val)) continue;
            NODE_MARK(val);
            char label[128]; const char *color = "#f0f0f0"; const char *nshape = "box";
            const char *opn = (ext < UOP_COUNT) ? uop_names[ext] : "?";
            const View *v = st_get(val);
            char sh[64] = "";
            if (v) { int p = 0; for (u32 d = 0; d < v->shape.rank && p < 50; d++)
                p += snprintf(sh+p, sizeof(sh)-p, "%s%u", d?",":"", v->shape.dims[d]); }
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
                Term gx = heap_read(ctx, val + 2);
                for (int _dp=0; _dp<10 && (term_tag(gx)==TAG_DP0||term_tag(gx)==TAG_DP1); _dp++)
                    gx = heap_read(ctx, term_val(gx));
                char tgt[32] = "?";
                if (term_tag(gx) == TAG_CTR) {
                    u64 cl = term_val(gx); u32 nt = term_as_u32(heap_read(ctx, cl)); int p = 0;
                    for (u32 gi = 0; gi < nt && p < 24; gi++) {
                        Term pt = heap_read(ctx, cl + 1 + 2*gi);
                        if (term_tag(pt) == TAG_TEN) p += snprintf(tgt+p, sizeof(tgt)-p, "%st%u", gi?",":"", (u32)term_val(pt));
                    }
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

            // Child edges: GRAD has 2 ports (y=input, gy=output).
            // x is a node parameter (not shown).
            u32 arity = 2;
            if (ext == UOP_GRAD) arity = 2; // y + gy
            else if (ext == UOP_WHERE) arity = 3;
            else if (ext == UOP_FUSING) arity = 0;
            else if (!is_binary(ext) && is_elementwise(ext)) arity = 1;
            // Orphaned nodes rendered normally — their children are still intact.
            // ERA will consume them in the next ERA step.
            for (u32 ai = 0; ai < arity; ai++) {
                Term child = heap_read(ctx, val + ai);
                u8 ctag = term_tag(child); u64 cval = term_val(child);
                const char *elbl = "";
                if (ext == UOP_ASSIGN) elbl = ai==0 ? "tgt" : "src";
                else if (ext >= UOP_RESHAPE && ext <= UOP_PAD) elbl = ai==0 ? "in" : "shape";
                else if (ext == UOP_SUM || ext == UOP_RMAX) elbl = ai==0 ? "in" : "axes";
                else if (ext == UOP_GRAD) elbl = ai==0 ? "y" : ai==1 ? "gy" : "x";
                else if (is_binary(ext)) elbl = ai==0 ? "a" : "b";

                // GRAD gy port (ai=1): reverse edge direction (output goes up)
                int rev = (ext == UOP_GRAD && ai == 1);

                if (ctag == TAG_DP0 || ctag == TAG_DP1) {
                    // Always show full DUP triangle with both ports
                    u64 dl = cval;
                    int dup_new = !NODE_SEEN(dl + 0x100000);
                    if (dup_new) {
                        NODE_MARK(dl + 0x100000);
                        fprintf(f, "  dup%llu [label=\"DUP\", shape=invtriangle, fillcolor=\"#d4b8e8\", fontsize=9, width=0.7, height=0.5];\n", dl);
                        Term shared = heap_read(ctx, dl);
                        u8 stag = term_tag(shared);
                        if (stag == TAG_TOP) fprintf(f, "  n%llu -> dup%llu;\n", term_val(shared), dl);
                        else if (stag == TAG_TEN) { EMIT_TEN((u32)term_val(shared)); fprintf(f, "  t%u -> dup%llu;\n", (u32)term_val(shared), dl); }
                        else if (stag == TAG_DP0 || stag == TAG_DP1) fprintf(f, "  dup%llu -> dup%llu;\n", term_val(shared), dl);
                        else if (stag == TAG_ERA) {
                            fprintf(f, "  era_s%llu [label=\"ERA\",shape=circle,width=0.2,fillcolor=\"#ff9999\",fontsize=7];\n", dl);
                            fprintf(f, "  era_s%llu -> dup%llu;\n", dl, dl);
                        }
                        else if (stag == TAG_NUM) { f32 fv; u32 bv=(u32)term_val(shared); memcpy(&fv,&bv,4);
                            fprintf(f, "  num_d%llu [label=\"%.4g\",shape=plaintext,fontsize=8];\n", dl, fv);
                            fprintf(f, "  num_d%llu -> dup%llu;\n", dl, dl); }
                    }
                    fprintf(f, "  dup%llu -> n%llu [label=\"%s%s\"];\n", dl, val, elbl,
                            ctag==TAG_DP1?" (dp1)":" (dp0)");
                } else if (ctag == TAG_TEN) {
                    EMIT_TEN((u32)cval);
                    if (rev) fprintf(f, "  n%llu -> t%u [label=\"%s\"];\n", val, (u32)cval, elbl);
                    else     fprintf(f, "  t%u -> n%llu [label=\"%s\"];\n", (u32)cval, val, elbl);
                } else if (ctag == TAG_ERA) {
                    fprintf(f, "  era%llu_%u [label=\"ERA\",shape=circle,width=0.2,fillcolor=\"#ff9999\",fontsize=7];\n", val, ai);
                    if (rev) fprintf(f, "  n%llu -> era%llu_%u [label=\"%s\",color=\"#cc0000\"];\n", val, val, ai, elbl);
                    else     fprintf(f, "  era%llu_%u -> n%llu [label=\"%s\",color=\"#cc0000\"];\n", val, ai, val, elbl);
                } else if (ctag == TAG_NUM) {
                    f32 fv; u32 bv=(u32)cval; memcpy(&fv,&bv,4);
                    fprintf(f, "  num%llu_%u [label=\"%.4g\",shape=plaintext,fontsize=8];\n", val, ai, fv);
                    fprintf(f, "  num%llu_%u -> n%llu [label=\"%s\"];\n", val, ai, val, elbl);
                } else if (ctag == TAG_TOP) {
                    if (rev) fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n", val, cval, elbl);
                    else     fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n", cval, val, elbl);
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

    // Orphaned nodes: unreachable positions not yet shown — about to be ERA'd
    for (u64 h = 1; h < ctx->heap_pos && h < REACH_MAX; h++) {
        if (reach_live[h]) continue;
        Term t = ctx->heap[h];
        u8 otag = term_tag(t);
        // Orphaned tensors
        if (otag == TAG_TEN) {
            u32 tid = (u32)term_val(t);
            if (tid < 256 && !ten_emitted[tid]) { EMIT_TEN(tid); nn++; }
        }
        // Orphaned DP refs: show ERA edge to their DUP (DUP port about to be consumed)
        if ((otag == TAG_DP0 || otag == TAG_DP1) && NODE_SEEN(term_val(t) + 0x100000)) {
            u64 dl = term_val(t);
            fprintf(f, "  era_dp%llu [label=\"ERA\",shape=circle,width=0.2,fillcolor=\"#ff9999\",fontsize=7];\n", h);
            fprintf(f, "  era_dp%llu -> dup%llu [label=\"%s\",color=\"#cc0000\",style=dashed];\n",
                    h, dl, otag==TAG_DP0?"(dp0)":"(dp1)");
            nn++;
        }
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
    fprintf(stderr, "heap_dot: %u nodes → %s\n", nn, path);
    #undef EMIT_TEN
    #undef REACH_MAX
    #undef IS_LIVE
    #undef NODE_DEDUP_MAX
    #undef NODE_SEEN
    #undef NODE_MARK
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
