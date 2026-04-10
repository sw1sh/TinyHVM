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
    #define EMIT_DUP_STUB(dloc) do { \
        if (!NODE_SEEN((dloc) + 0x100000)) { \
            NODE_MARK((dloc) + 0x100000); \
            fprintf(f, "  dup%llu [label=\"DUP\", shape=invtriangle, fillcolor=\"#d4b8e8\", fontsize=9, width=0.7, height=0.5];\n", (unsigned long long)(dloc)); \
        } \
    } while(0)
    #define ERA_DEDUP_MAX 1024
    u64 era_emitted[ERA_DEDUP_MAX]; u32 n_era_emitted = 0;
    #define ERA_SEEN(v) ({ int _s=0; for(u32 _i=0;_i<n_era_emitted;_i++) \
        if(era_emitted[_i]==(v)){_s=1;break;} _s; })
    #define ERA_MARK(v) do { if(n_era_emitted<ERA_DEDUP_MAX) era_emitted[n_era_emitted++]=(v); } while(0)
    #define EMIT_ERA_NODE(epos, eterm) do { \
        if (!ERA_SEEN(epos)) { \
            ERA_MARK(epos); \
            fprintf(f, "  era%llu [label=\"ERA\",shape=circle,width=0.2,fillcolor=\"#ff9999\",fontsize=7];\n", (unsigned long long)(epos)); \
            Term _et = (eterm); \
            u64 _ev = term_val(_et); \
            if (_ev == 0) break; \
            if (_ev < ctx->heap_pos) { \
                Term _src = heap_read(ctx, _ev); \
                u8 _st = term_tag(_src); \
                if (_st == TAG_TOP) fprintf(f, "  n%llu -> era%llu [color=\"#cc0000\"];\n", term_val(_src), (unsigned long long)(epos)); \
                else if (_st == TAG_TEN) { EMIT_TEN((u32)term_val(_src)); fprintf(f, "  t%u -> era%llu [color=\"#cc0000\"];\n", (u32)term_val(_src), (unsigned long long)(epos)); } \
                else if (_st == TAG_NUM) { f32 _fv; u32 _bv=(u32)term_val(_src); memcpy(&_fv,&_bv,4); \
                    fprintf(f, "  num_e%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", (unsigned long long)(epos), _fv); \
                    fprintf(f, "  num_e%llu -> era%llu [color=\"#cc0000\"];\n", (unsigned long long)(epos), (unsigned long long)(epos)); } \
                else if (_st == TAG_ANY) { \
                    EMIT_ANY_NODE(_ev); \
                    fprintf(f, "  any%llu -> era%llu [color=\"#cc0000\"];\n", (unsigned long long)_ev, (unsigned long long)(epos)); \
                } \
                else if (_st == TAG_DP0 || _st == TAG_DP1) { \
                    u64 _dl = term_val(_src); \
                    Term _shared = heap_read(ctx, _dl); \
                    u8 _sg = term_tag(_shared); \
                    int _share_is_known = (_sg == TAG_TOP || _sg == TAG_TEN || _sg == TAG_DP0 || _sg == TAG_DP1 || _sg == TAG_NUM); \
                    if (!_share_is_known) { \
                        EMIT_RAW_NODE(_dl, _shared); \
                        fprintf(f, "  h%llu -> era%llu [color=\"#cc0000\"];\n", (unsigned long long)_dl, (unsigned long long)(epos)); \
                    } else { \
                        int _dup_new = !NODE_SEEN(_dl + 0x100000); \
                        if (_dup_new) { \
                            NODE_MARK(_dl + 0x100000); \
                            fprintf(f, "  dup%llu [label=\"DUP\", shape=invtriangle, fillcolor=\"#d4b8e8\", fontsize=9, width=0.7, height=0.5];\n", _dl); \
                            if (_sg == TAG_TOP) fprintf(f, "  n%llu -> dup%llu;\n", term_val(_shared), _dl); \
                            else if (_sg == TAG_TEN) { EMIT_TEN((u32)term_val(_shared)); fprintf(f, "  t%u -> dup%llu;\n", (u32)term_val(_shared), _dl); } \
                            else if (_sg == TAG_DP0 || _sg == TAG_DP1) { EMIT_DUP_STUB(term_val(_shared)); fprintf(f, "  dup%llu -> dup%llu;\n", term_val(_shared), _dl); } \
                            else if (_sg == TAG_NUM) { f32 _sv; u32 _sb=(u32)term_val(_shared); memcpy(&_sv,&_sb,4); \
                                fprintf(f, "  num_d%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", _dl, _sv); \
                                fprintf(f, "  num_d%llu -> dup%llu;\n", _dl, _dl); } \
                            else if (_sg == TAG_ANY) { EMIT_ANY_NODE(_dl); fprintf(f, "  any%llu -> dup%llu;\n", _dl, _dl); } \
                        } \
                        fprintf(f, "  dup%llu -> era%llu [color=\"#cc0000\"];\n", _dl, (unsigned long long)(epos)); \
                    } \
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
            if (val == 0) continue;
            EMIT_ERA_NODE(h, t);
            nn++;
            continue;
        }

        // --- TAG_TEN: only shown when referenced as child of a node ---
        if (tag == TAG_TEN) {
            // Keep standalone heap tensors visible so step-to-step disappearance
            // can be attributed explicitly to ERA interactions.
            EMIT_TEN((u32)val);
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
                const char *elbl = "";
                if (ext == UOP_ASSIGN) elbl = ai==0 ? "tgt" : "src";
                else if (ext >= UOP_RESHAPE && ext <= UOP_PAD) elbl = ai==0 ? "in" : "shape";
                else if (ext == UOP_SUM || ext == UOP_RMAX) elbl = ai==0 ? "in" : "axes";
                else if (ext == UOP_GRAD) elbl = ai==0 ? "y" : "gy";
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
                        else if (stag == TAG_DP0 || stag == TAG_DP1) { EMIT_DUP_STUB(term_val(shared)); fprintf(f, "  dup%llu -> dup%llu;\n", term_val(shared), dl); }
                        else if (stag == TAG_CTR) {
                            u64 cl = term_val(shared);
                            EMIT_CTR_NODE(cl);
                            fprintf(f, "  ctr%llu -> dup%llu;\n", cl, dl);
                            if (cl < ctx->heap_pos) {
                                Term nt = heap_read(ctx, cl);
                                u32 n_tgt = (term_tag(nt) == TAG_NUM) ? (u32)term_val(nt) : 0;
                                if (n_tgt > 64) n_tgt = 64;
                                for (u32 gi = 0; gi < n_tgt; gi++) {
                                    Term pterm = heap_read(ctx, cl + 1 + 2*gi);
                                    Term sterm = heap_read(ctx, cl + 1 + 2*gi + 1);
                                    char plbl[16], slbl[16];
                                    snprintf(plbl, sizeof(plbl), "p%u", gi);
                                    snprintf(slbl, sizeof(slbl), "slot%u", gi);
                                    if (term_tag(pterm) == TAG_TEN) {
                                        EMIT_TEN((u32)term_val(pterm));
                                        fprintf(f, "  t%u -> ctr%llu [label=\"%s\"];\n", (u32)term_val(pterm), cl, plbl);
                                    } else if (term_tag(pterm) == TAG_TOP) {
                                        fprintf(f, "  n%llu -> ctr%llu [label=\"%s\"];\n", term_val(pterm), cl, plbl);
                                    } else if (term_tag(pterm) == TAG_DP0 || term_tag(pterm) == TAG_DP1) {
                                        fprintf(f, "  dup%llu -> ctr%llu [label=\"%s%s\"];\n", term_val(pterm), cl, plbl,
                                                term_tag(pterm)==TAG_DP1 ? " (dp1)" : " (dp0)");
                                    }
                                    if (term_tag(sterm) == TAG_TEN) {
                                        EMIT_TEN((u32)term_val(sterm));
                                        fprintf(f, "  t%u -> ctr%llu [label=\"%s\"];\n", (u32)term_val(sterm), cl, slbl);
                                    } else if (term_tag(sterm) == TAG_TOP) {
                                        fprintf(f, "  n%llu -> ctr%llu [label=\"%s\"];\n", term_val(sterm), cl, slbl);
                                    } else if (term_tag(sterm) == TAG_DP0 || term_tag(sterm) == TAG_DP1) {
                                        fprintf(f, "  dup%llu -> ctr%llu [label=\"%s%s\"];\n", term_val(sterm), cl, slbl,
                                                term_tag(sterm)==TAG_DP1 ? " (dp1)" : " (dp0)");
                                    }
                                }
                            }
                        }
                        else if (stag == TAG_ERA) {
                            EMIT_RAW_NODE(dl, shared);
                            fprintf(f, "  h%llu -> dup%llu;\n", (unsigned long long)dl, dl);
                        }
                        else if (stag == TAG_NUM) { f32 fv; u32 bv=(u32)term_val(shared); memcpy(&fv,&bv,4);
                            fprintf(f, "  num_d%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", dl, fv);
                            fprintf(f, "  num_d%llu -> dup%llu;\n", dl, dl); }
                        else { EMIT_RAW_NODE(dl, shared); fprintf(f, "  h%llu -> dup%llu;\n", (unsigned long long)dl, dl); }
                    }
                    fprintf(f, "  dup%llu -> n%llu [label=\"%s%s\"];\n", dl, val, elbl,
                            ctag==TAG_DP1?" (dp1)":" (dp0)");
                } else if (ctag == TAG_TEN) {
                    EMIT_TEN((u32)cval);
                    if (rev) fprintf(f, "  n%llu -> t%u [label=\"%s\"];\n", val, (u32)cval, elbl);
                    else     fprintf(f, "  t%u -> n%llu [label=\"%s\"];\n", (u32)cval, val, elbl);
                } else if (ctag == TAG_CTR) {
                    EMIT_CTR_NODE(cval);
                    if (rev) fprintf(f, "  n%llu -> ctr%llu [label=\"%s\"];\n", val, cval, elbl);
                    else     fprintf(f, "  ctr%llu -> n%llu [label=\"%s\"];\n", cval, val, elbl);

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
                                    int _dup_new = !NODE_SEEN(_dl + 0x100000); \
                                    if (_dup_new) { \
                                        NODE_MARK(_dl + 0x100000); \
                                        fprintf(f, "  dup%llu [label=\"DUP\", shape=invtriangle, fillcolor=\"#d4b8e8\", fontsize=9, width=0.7, height=0.5];\n", _dl); \
                                        Term _shared = heap_read(ctx, _dl); \
                                        u8 _stag = term_tag(_shared); \
                                        if (_stag == TAG_TOP) fprintf(f, "  n%llu -> dup%llu;\n", term_val(_shared), _dl); \
                                        else if (_stag == TAG_TEN) { EMIT_TEN((u32)term_val(_shared)); fprintf(f, "  t%u -> dup%llu;\n", (u32)term_val(_shared), _dl); } \
                                        else if (_stag == TAG_DP0 || _stag == TAG_DP1) fprintf(f, "  dup%llu -> dup%llu;\n", term_val(_shared), _dl); \
                                        else if (_stag == TAG_ERA) { EMIT_RAW_NODE(_dl, _shared); fprintf(f, "  h%llu -> dup%llu;\n", (unsigned long long)_dl, _dl); } \
                                        else if (_stag == TAG_NUM) { f32 _fv; u32 _bv=(u32)term_val(_shared); memcpy(&_fv,&_bv,4); \
                                            fprintf(f, "  num_d%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", _dl, _fv); \
                                            fprintf(f, "  num_d%llu -> dup%llu;\n", _dl, _dl); } \
                                        else { EMIT_RAW_NODE(_dl, _shared); fprintf(f, "  h%llu -> dup%llu;\n", (unsigned long long)_dl, _dl); } \
                                    } \
                                    fprintf(f, "  dup%llu -> ctr%llu [label=\"%s%s\"];\n", _dl, cval, (clbl), _ct==TAG_DP1 ? " (dp1)" : " (dp0)"); \
                                } else if (_ct == TAG_NUM) { \
                                    f32 _fv; u32 _bv=(u32)_cv; memcpy(&_fv,&_bv,4); \
                                    fprintf(f, "  num_ctr%llu_%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", cval, (unsigned long long)(cpos), _fv); \
                                    fprintf(f, "  num_ctr%llu_%llu -> ctr%llu [label=\"%s\"];\n", cval, (unsigned long long)(cpos), cval, (clbl)); \
                                } else if (_ct == TAG_ERA) { \
                                    if (_cv != 0) EMIT_ERA_NODE((cpos), (cterm)); \
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
                    if (rev) fprintf(f, "  n%llu -> any%llu [label=\"%s\"];\n", val, cval, elbl);
                    else     fprintf(f, "  any%llu -> n%llu [label=\"%s\"];\n", cval, val, elbl);
                } else if (ctag == TAG_ERA) {
                    if (cval == 0) continue; // inert ERA is hidden
                    u64 epos = val + ai;
                    EMIT_ERA_NODE(epos, child);
                    (void)rev; (void)elbl;
                } else if (ctag == TAG_NUM) {
                    f32 fv; u32 bv=(u32)cval; memcpy(&fv,&bv,4);
                    fprintf(f, "  num%llu_%u [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", val, ai, fv);
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
    #undef EMIT_RAW_NODE
    #undef RAW_NODE_KEY
    #undef EMIT_CTR_NODE
    #undef CTR_NODE_KEY
    #undef EMIT_ANY_NODE
    #undef ANY_NODE_KEY
    #undef EMIT_DUP_STUB
    #undef EMIT_ERA_NODE
    #undef ERA_DEDUP_MAX
    #undef ERA_SEEN
    #undef ERA_MARK
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
