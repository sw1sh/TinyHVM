// debug/dump.c — Dump tensor provenance graph as DOT (graphviz) or JSON
// Usage: thvm_dump_dot(ctx, "graph.dot") or thvm_dump_json(ctx, "graph.json")
// Visualize: dot -Tpng graph.dot -o graph.png

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Step-graph overlay: highlight one predicted next interaction edge.
static int heap_dot_hl_on = 0;
static u64 heap_dot_hl_slot = 0;
static Term heap_dot_hl_term = 0;
static int heap_dot_hl_hit = 0;
static char heap_dot_prev_name[96] = {0};
static char heap_dot_next_name[96] = {0};
static int heap_dot_include_sched_kernels = 0;
// When set, the heap dumper seeds the walk from every non-ERA heap slot
// in addition to `root`, so off-root scaffolding created by the wnf
// stack machine (DUP cells, intermediate compute TOPs pending
// substitution) shows up in per-step graphs.
static int heap_dot_include_all_slots = 0;
static void thvm_heap_dot_set_include_all(int enabled) {
    heap_dot_include_all_slots = enabled ? 1 : 0;
}
#define DOT_KERNEL_FALLBACK_MAX 1024
static u64 heap_dot_kernel_fallback_locs[DOT_KERNEL_FALLBACK_MAX];
static u32 heap_dot_kernel_fallback_n = 0;
static u32 heap_dot_kernel_fallback_base = 0;
// heap_dot_node_hl and heap_dot_root_only declared in graph.c (included before dump.c)
static void thvm_heap_dot_set_highlight(u64 slot, Term term) {
    heap_dot_hl_on = (slot != 0);
    heap_dot_hl_slot = slot;
    heap_dot_hl_term = term;
    heap_dot_hl_hit = 0;
}

// GRAD cursor overlay: draws a dotted edge from the node at
// heap[grad_slot] (a GRAD) to the node at heap[cursor_loc] (the
// currently-active forward TOP being differentiated).  Used by the
// step-graph hook to show the VJP descent cursor without mutating
// heap.  Both 0 disables.
static u64 heap_dot_grad_cursor_grad  = 0;
static u64 heap_dot_grad_cursor_loc   = 0;
static void thvm_heap_dot_set_grad_cursor(u64 grad_slot, u64 cursor_loc) {
    heap_dot_grad_cursor_grad = grad_slot;
    heap_dot_grad_cursor_loc  = cursor_loc;
}
static void thvm_heap_dot_set_step_meta(const char *prev_name, const char *next_name) {
    snprintf(heap_dot_prev_name, sizeof(heap_dot_prev_name), "%s", prev_name ? prev_name : "");
    snprintf(heap_dot_next_name, sizeof(heap_dot_next_name), "%s", next_name ? next_name : "");
}
static void thvm_heap_dot_set_sched_kernels(int enabled) {
    heap_dot_include_sched_kernels = enabled ? 1 : 0;
}

static void thvm_heap_dot_reset_kernel_fallback_ids(void) {
    extern u32 sched_kernel_count;
    heap_dot_kernel_fallback_n = 0;
    heap_dot_kernel_fallback_base = sched_kernel_count;
}

static u32 dot_kernel_display_kid(TinyHVM *ctx, Term kernel) {
    extern u32 sched_kernel_count;
    if (!ctx || term_tag(kernel) != TAG_TOP || term_ext(kernel) != UOP_KERNEL)
        return ~0u;
    u64 loc = term_val(kernel);
    u32 kid = 0;
    if (thvm_kernel_lookup_kid(loc, &kid) && kid < sched_kernel_count)
        return kid;
    for (u32 i = 0; i < heap_dot_kernel_fallback_n; i++) {
        if (heap_dot_kernel_fallback_locs[i] == loc)
            return heap_dot_kernel_fallback_base + i;
    }
    if (heap_dot_kernel_fallback_n < DOT_KERNEL_FALLBACK_MAX) {
        heap_dot_kernel_fallback_locs[heap_dot_kernel_fallback_n] = loc;
        return heap_dot_kernel_fallback_base + heap_dot_kernel_fallback_n++;
    }
    return ~0u;
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
    if (uop == UOP_GRAD || uop == UOP_GRAD_FWD) return argi == 0 ? "y" : "tgt";
    if (uop == UOP_GRAD_PIN) return argi == 0 ? "y" : "tgt";
    if (is_view_op(uop)) return argi == 0 ? "in" : "shape";
    if (is_binary(uop) || uop == UOP_CMP || uop == UOP_MM) return argi == 0 ? "a" : "b";
    if (uop == UOP_LOG_PRINT) return "in";
    if (uop == UOP_DETACH) return "in";
    return "in";
}

// Follow SUB redirects through commuted GRAD cells.  When a GRAD cell's
// slot 0 has the SUB bit set, the cell is "dead" — BW already fired and
// stored its sibling redirect.  Edges to such cells render as edges to
// the actual destination:
//   TAG_TOP(UOP_GRAD, loc)     → heap[loc+1]  (bwd_wrapper, ∂v target)
//   TAG_TOP(UOP_GRAD_PIN, loc) → strip(heap[loc+0])  (reparented forward)
// Loops through chained commutes.
static Term dot_deref_commuted_grad(TinyHVM *ctx, Term t) {
    for (u32 hop = 0; hop < 32 && term_tag(t) == TAG_TOP; hop++) {
        u32 e = term_ext(t);
        u64 l = term_val(t);
        if (l == 0 || l + 1 >= ctx->heap_pos) break;
        Term s0 = heap_read(ctx, l + 0);
        if (!term_is_sub(s0)) break;
        if (e == UOP_GRAD || e == UOP_GRAD_FWD) {
            t = heap_read(ctx, l + 1);
            continue;
        }
        if (e == UOP_GRAD_PIN) {
            t = term_strip_sub(s0);
            continue;
        }
        break;
    }
    return t;
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
        case TAG_SEQ:
        case TAG_CTR:
        case TAG_INC:
            return 1;
        case TAG_ALO:
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
        case TAG_SEQ: return "SEQ";
        case TAG_ALO: return "ALO";
        case TAG_CTR: return "CTR";
        case TAG_REF: return "REF";
        case TAG_VAR: return "VAR";
        case TAG_INC: return "INC";
        case TAG_ERA: return "ERA";
        default: return "?";
    }
}

static const char *dot_term_name_short(Term t, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return "";
    buf[0] = '\0';
    switch (term_tag(t)) {
        case TAG_TOP:
            if (term_ext(t) < UOP_COUNT) snprintf(buf, nbuf, "%s", uop_names[term_ext(t)]);
            else snprintf(buf, nbuf, "TOP");
            break;
        case TAG_TEN:
            snprintf(buf, nbuf, "TEN");
            break;
        case TAG_NUM:
            snprintf(buf, nbuf, "NUM");
            break;
        case TAG_ANY:
            snprintf(buf, nbuf, "ANY");
            break;
        default:
            snprintf(buf, nbuf, "%s", dot_heap_tag_name(term_tag(t)));
            break;
    }
    return buf;
}

static const char *dot_heap_node_shape(u8 tag) {
    switch (tag) {
        case TAG_LAM: return "triangle";
        case TAG_APP: return "invtriangle";
        case TAG_SUP:
        case TAG_USP:
        case TAG_CTR: return "hexagon";
        case TAG_REF:
        case TAG_VAR: return "oval";
        case TAG_ALO: return "box3d";
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
        case TAG_ALO: return "#e8f7ff";
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
        case TAG_SEQ: return idx == 0 ? "first" : "then";
        case TAG_ALO: return idx == 0 ? "book" : "env";
        default: return idx == 0 ? "a" : "b";
    }
}

static const char *dot_tensor_edge_slot_attrs(u64 slot, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return "";
    buf[0] = '\0';
    if (slot == 0) return "";
    snprintf(buf, nbuf,
             ",taillabel=\"@%llu\",labelfontsize=7,fontcolor=\"#666666\"",
             (unsigned long long)slot);
    return buf;
}

typedef struct {
    int semantic;
    int step;
} DotLayers;

static int dot_env_flag_enabled(const char *name, int default_value) {
    const char *v = getenv(name);
    if (!v || !v[0]) return default_value;
    return !(v[0] == '0' && v[1] == '\0');
}

static DotLayers dot_layers_from_env(void) {
    int raw_only = dot_env_flag_enabled("THVM_HEAP_DOT_RAW_ONLY", 0);
    DotLayers layers = {
        .semantic = raw_only ? 0 : dot_env_flag_enabled("THVM_HEAP_DOT_SEMANTIC", 1),
        .step = raw_only ? 0 : dot_env_flag_enabled("THVM_HEAP_DOT_STEP", 1),
    };
    return layers;
}

static void dot_raw_slot_term_label(TinyHVM *ctx, Term t, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return;
    buf[0] = '\0';
    (void)ctx;
    u8 tag = term_tag(t);
    u32 ext = term_ext(t);
    u64 val = term_val(t);
    switch (tag) {
        case TAG_TOP:
            if (ext < UOP_COUNT) snprintf(buf, nbuf, "%s\\n@%llu", uop_names[ext], (unsigned long long)val);
            else snprintf(buf, nbuf, "TOP\\n@%llu", (unsigned long long)val);
            return;
        case TAG_DP0:
            snprintf(buf, nbuf, "DP0\\n@%llu", (unsigned long long)val);
            return;
        case TAG_DP1:
            snprintf(buf, nbuf, "DP1\\n@%llu", (unsigned long long)val);
            return;
        case TAG_LAM:
        case TAG_BRI:
            snprintf(buf, nbuf, "%s\\n#%u@%llu", dot_heap_tag_name(tag), ext, (unsigned long long)val);
            return;
        case TAG_TEN:
            snprintf(buf, nbuf, "TEN\\nt%llu", (unsigned long long)val);
            return;
        case TAG_ANY:
            snprintf(buf, nbuf, "ANY\\n@%llu", (unsigned long long)val);
            return;
        case TAG_ERA:
            snprintf(buf, nbuf, "ERA\\n@%llu", (unsigned long long)val);
            return;
        case TAG_REF:
        case TAG_MAT:
            snprintf(buf, nbuf, "%s\\n#%u@%llu", dot_heap_tag_name(tag), ext, (unsigned long long)val);
            return;
        case TAG_SUP:
        case TAG_USP:
            snprintf(buf, nbuf, "%s #%u@%llu", dot_heap_tag_name(tag), ext, (unsigned long long)val);
            return;
        case TAG_NUM: {
            f32 fv;
            u32 bv = (u32)val;
            memcpy(&fv, &bv, 4);
            snprintf(buf, nbuf, "NUM\\n%.4g", (double)fv);
            return;
        }
        default: {
            const char *name = dot_heap_tag_name(tag);
            if (name[0] == '?' && name[1] == '\0')
                snprintf(buf, nbuf, "tag=%u/%u\\n@%llu", (u32)tag, ext, (unsigned long long)val);
            else
                snprintf(buf, nbuf, "%s\\n@%llu", name, (unsigned long long)val);
            return;
        }
    }
}

static u32 dot_book_struct_arity(Term t);
static Term dot_book_to_dynamic_term(TinyHVM *ctx, Term target);

static const char *dot_alo_env_bind_label_fallback(const AloState *st, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return "";
    buf[0] = '\0';
    if (!st || st->bind_book == 0) return "";
    switch (st->bind_tag) {
        case TAG_LAM:
        case TAG_BRI:
            snprintf(buf, nbuf, "%s@%llu/var",
                     dot_heap_tag_name(st->bind_tag),
                     (unsigned long long)st->bind_book);
            return buf;
        case TAG_DP0:
            snprintf(buf, nbuf, "DUP@%llu/dp0@%llu",
                     (unsigned long long)st->bind_book,
                     (unsigned long long)st->bind_book);
            return buf;
        case TAG_DP1:
            snprintf(buf, nbuf, "DUP@%llu/dp1@%llu",
                     (unsigned long long)st->bind_book,
                     (unsigned long long)st->bind_book);
            return buf;
        default:
            snprintf(buf, nbuf, "h%llu", (unsigned long long)st->bind_book);
            return buf;
    }
}

static const char *dot_dp_port_label(char *buf, size_t nbuf, const char *prefix, u8 tag, u64 slot) {
    if (!buf || nbuf == 0) return "";
    const char *dp = tag == TAG_DP1 ? "dp1" : "dp0";
    if (prefix && prefix[0]) {
        snprintf(buf, nbuf, "%s (%s@%llu)", prefix, dp, (unsigned long long)slot);
    } else {
        snprintf(buf, nbuf, "%s@%llu", dp, (unsigned long long)slot);
    }
    return buf;
}

static int dot_book_find_bind_use_r(TinyHVM *ctx, Term book_term, u64 bind_book,
                                    Term *out_parent, u32 *out_port, u32 depth) {
    if (!ctx || book_term == 0 || bind_book == 0 || depth > 256) return 0;
    u32 ar = dot_book_struct_arity(book_term);
    if (ar == 0) return 0;
    u64 loc = term_val(book_term);
    if (loc == 0 || loc + ar > ctx->book_heap_pos) return 0;
    for (u32 i = 0; i < ar; i++) {
        Term child = ctx->book_heap[loc + i];
        u8 ctag = term_tag(child);
        if ((ctag == TAG_VAR || ctag == TAG_DP0 || ctag == TAG_DP1) &&
            term_val(child) == bind_book) {
            if (out_parent) *out_parent = book_term;
            if (out_port) *out_port = i;
            return 1;
        }
        if (dot_book_find_bind_use_r(ctx, child, bind_book, out_parent, out_port, depth + 1))
            return 1;
    }
    return 0;
}

static const char *dot_alo_env_bind_label(TinyHVM *ctx, Term current_book_term,
                                          const AloState *st, char *buf, size_t nbuf) {
    if (!buf || nbuf == 0) return "";
    buf[0] = '\0';
    if (ctx && current_book_term != 0 && st && st->bind_book != 0) {
        Term parent_book = 0;
        u32 parent_port = 0;
        if (dot_book_find_bind_use_r(ctx, current_book_term, st->bind_book,
                                     &parent_book, &parent_port, 0)) {
            Term shown_parent = dot_book_to_dynamic_term(ctx, parent_book);
            if (shown_parent == 0) shown_parent = parent_book;
            char name[32];
            dot_term_name_short(shown_parent, name, sizeof(name));
            const char *port = term_tag(parent_book) == TAG_TOP
                ? dot_uop_port_name(term_ext(parent_book), parent_port)
                : dot_heap_port_name(term_tag(parent_book), parent_port);
            if (name[0] && port && port[0]) {
                snprintf(buf, nbuf, "%s@%llu/%s",
                         name,
                         (unsigned long long)term_val(shown_parent),
                         port);
                return buf;
            }
        }
        Term shown_book = dot_book_to_dynamic_term(ctx, current_book_term);
        if (shown_book != 0 && term_val(shown_book) == st->bind_book) {
            char name[32];
            dot_term_name_short(shown_book, name, sizeof(name));
            const char *port = term_tag(shown_book) == TAG_TOP
                ? dot_uop_port_name(term_ext(shown_book), 0)
                : dot_heap_port_name(term_tag(shown_book), 0);
            if (name[0] && port && port[0]) {
                snprintf(buf, nbuf, "%s@%llu/%s",
                         name,
                         (unsigned long long)term_val(shown_book),
                         port);
                return buf;
            }
        }
    }
    return dot_alo_env_bind_label_fallback(st, buf, nbuf);
}

static const char *dot_fuse_payload_label_r(TinyHVM *ctx, Term payload, char *buf,
                                            size_t nbuf, u32 depth) {
    if (!buf || nbuf == 0) return "";
    if (!ctx || payload == 0 || depth > 16) {
        snprintf(buf, nbuf, "NULL");
        return buf;
    }
    if (term_tag(payload) == TAG_ALO) {
        u64 loc = term_val(payload);
        if (loc > 0 && loc + 1 < ctx->heap_pos)
            return dot_fuse_payload_label_r(ctx, heap_read(ctx, loc + 0), buf, nbuf, depth + 1);
    }
    if (term_tag(payload) == TAG_TOP) {
        u32 uop = term_ext(payload);
        if (uop == UOP_KERNEL) {
            u32 kop = thvm_kernel_root_uop(ctx, payload);
            if (kop < UOP_COUNT) {
                snprintf(buf, nbuf, "%s", uop_names[kop]);
                return buf;
            }
        }
        if (uop < UOP_COUNT &&
            uop != UOP_FUSE) {
            snprintf(buf, nbuf, "%s", uop_names[uop]);
            return buf;
        }
    }
    snprintf(buf, nbuf, "NULL");
    return buf;
}


static u32 dot_book_struct_arity(Term t) {
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
        case TAG_DSU:
        case TAG_DDU:
            return 3;
        case TAG_INC:
        case TAG_UDP:
            return 1;
        case TAG_CTR:
            return ext;
        default:
            return 0;
    }
}

static int dot_book_find_dynamic_r(TinyHVM *ctx, Term dyn_term, Term book_term, Term target,
                                   u32 depth, Term *out_dyn) {
    if (target == 0) return 0;
    if (book_term == target) {
        if (out_dyn) *out_dyn = dyn_term;
        return 1;
    }
    if (depth > 4096) return 0;
    if (term_tag(book_term) != term_tag(dyn_term) || term_ext(book_term) != term_ext(dyn_term))
        return 0;
    u32 ar = dot_book_struct_arity(book_term);
    if (ar == 0) return 0;
    u64 book_loc = term_val(book_term);
    u64 dyn_loc = term_val(dyn_term);
    if (book_loc == 0 || dyn_loc == 0) return 0;
    for (u32 i = 0; i < ar; i++) {
        if (book_loc + i >= ctx->book_heap_pos || dyn_loc + i >= ctx->heap_pos) break;
        if (dot_book_find_dynamic_r(ctx, heap_read(ctx, dyn_loc + i), ctx->book_heap[book_loc + i],
                                    target, depth + 1, out_dyn))
            return 1;
    }
    return 0;
}

static Term dot_book_to_dynamic_term(TinyHVM *ctx, Term target) {
    if (!ctx || target == 0) return 0;
    for (u32 i = 0; i < ctx->def_count; i++) {
        Term book_root = ctx->def_books[i];
        Term dyn_root = ctx->defs[i];
        if (book_root == 0 || dyn_root == 0 || term_tag(dyn_root) == TAG_ERA) continue;
        Term out = 0;
        if (dot_book_find_dynamic_r(ctx, dyn_root, book_root, target, 0, &out))
            return out;
    }
    return 0;
}

static Term dot_resolve_var_term(TinyHVM *ctx, u64 var_loc, u64 *out_cell) {
    u64 cell = var_loc;
    Term resolved = (ctx && cell < ctx->heap_pos) ? heap_read(ctx, cell) : term_era();
    for (u32 depth = 0; depth < 16 && term_tag(resolved) == TAG_VAR; depth++) {
        u64 next = term_val(resolved);
        if (next == 0 || next >= ctx->heap_pos) break;
        cell = next;
        resolved = heap_read(ctx, cell);
    }
    if (out_cell) *out_cell = cell;
    return resolved;
}

static int dot_term_node_id(TinyHVM *ctx, Term t, char *buf, size_t nbuf) {
    if (nbuf == 0) return 0;
    u8 tag = term_tag(t);
    u64 val = term_val(t);
    if (tag == TAG_DP0 || tag == TAG_DP1) {
        snprintf(buf, nbuf, "dup%llu", (unsigned long long)val);
        return 1;
    }
    if (tag == TAG_VAR) {
        u64 cell = val;
        Term resolved = dot_resolve_var_term(ctx, val, &cell);
        u8 rtag = term_tag(resolved);
        u64 rval = term_val(resolved);
        if (rtag == TAG_DP0 || rtag == TAG_DP1) {
            snprintf(buf, nbuf, "dup%llu", (unsigned long long)rval);
            return 1;
        }
        if (rtag == TAG_CTR) {
            snprintf(buf, nbuf, "ctr%llu", (unsigned long long)rval);
            return 1;
        }
        if (rtag == TAG_VAR) {
            snprintf(buf, nbuf, "n%llu", (unsigned long long)rval);
            return 1;
        }
        if (rtag == TAG_TOP || rtag == TAG_APP || dot_visible_heap_loc_tag(rtag)) {
            snprintf(buf, nbuf, "n%llu", (unsigned long long)rval);
            return 1;
        }
        return 0;
    }
    if (tag == TAG_CTR) {
        snprintf(buf, nbuf, "ctr%llu", (unsigned long long)val);
        return 1;
    }
    if (tag == TAG_TOP || tag == TAG_APP || dot_visible_heap_loc_tag(tag)) {
        snprintf(buf, nbuf, "n%llu", (unsigned long long)val);
        return 1;
    }
    return 0;
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

int dot_kernel_entry_for_term(TinyHVM *ctx, Term kernel,
                                     KernelEntry *scratch, KernelEntry **out_ke, u32 *out_kid);
void dot_kernel_ops_summary(const KernelEntry *ke, u32 fallback_uop,
                                   char *buf, size_t bufsz);

void thvm_kernel_op_chain(TinyHVM *ctx, Term kernel, char *buf, size_t bufsz) {
    if (!buf || bufsz == 0) return;
    buf[0] = '\0';
    if (term_tag(kernel) != TAG_TOP || term_ext(kernel) != UOP_KERNEL) return;
    KernelEntry scratch;
    KernelEntry *ke = NULL;
    u32 kid = ~0u;
    u32 kop = thvm_kernel_root_uop(ctx, kernel);
    int have = dot_kernel_entry_for_term(ctx, kernel, &scratch, &ke, &kid);
    if (have) dot_kernel_ops_summary(ke, kop, buf, bufsz);
    else if (kop < UOP_COUNT) snprintf(buf, bufsz, "%s", uop_names[kop]);
}

int dot_kernel_entry_for_term(TinyHVM *ctx, Term kernel,
                                     KernelEntry *scratch, KernelEntry **out_ke, u32 *out_kid) {
    extern KernelEntry sched_kernels[];
    extern u32 sched_kernel_count;
    if (out_ke) *out_ke = NULL;
    if (out_kid) *out_kid = ~0u;
    if (!ctx || term_tag(kernel) != TAG_TOP || term_ext(kernel) != UOP_KERNEL)
        return 0;
    u64 loc = term_val(kernel);
    u32 kid = 0;
    if (thvm_kernel_lookup_kid(loc, &kid) && kid < sched_kernel_count) {
        if (out_ke) *out_ke = &sched_kernels[kid];
        if (out_kid) *out_kid = kid;
        return 1;
    }
    if (!scratch || !thvm_kernel_is_monolithic(ctx, kernel))
        return 0;
    memset(scratch, 0, sizeof(*scratch));
    Term compute = term_era();
    if (!thvm_kernel_to_compute(ctx, kernel, &compute, 0))
        return 0;
    if (term_tag(compute) != TAG_TOP)
        return 0;
    fuse_set_schedule_boundaries(NULL, NULL, NULL, 0, term_val(compute));
    int ok = fuse_build_kernel(ctx, compute, scratch);
    fuse_clear_schedule_boundaries();
    if (!ok)
        return 0;
    scratch->original_term = compute;
    if (out_ke) *out_ke = scratch;
    if (out_kid) *out_kid = dot_kernel_display_kid(ctx, kernel);
    return 1;
}

void dot_kernel_ops_summary(const KernelEntry *ke, u32 fallback_uop,
                                   char *buf, size_t bufsz) {
    if (bufsz == 0) return;
    buf[0] = '\0';
    if (!ke || ke->n_ops == 0) {
        snprintf(buf, bufsz, "%s", fallback_uop < UOP_COUNT ? uop_names[fallback_uop] : "KERNEL");
        return;
    }
    size_t used = 0;
    u32 shown = 0;
    for (u32 i = 0; i < ke->n_ops && shown < 4; i++) {
        u32 u = ke->ops[i].uop;
        const char *name = u < UOP_COUNT ? uop_names[u] : "?";
        int wrote = snprintf(buf + used, bufsz > used ? bufsz - used : 0,
                             "%s%s", shown ? "+" : "", name);
        if (wrote < 0) break;
        if ((size_t)wrote >= bufsz - used) {
            used = bufsz - 1;
            break;
        }
        used += (size_t)wrote;
        shown++;
    }
    if (ke->has_reduce && shown < 4) {
        const char *rname = ke->has_reduce < UOP_COUNT ? uop_names[ke->has_reduce] : "REDUCE";
        int wrote = snprintf(buf + used, bufsz > used ? bufsz - used : 0,
                             "%s%s", shown ? "+" : "", rname);
        if (wrote > 0 && (size_t)wrote < bufsz - used) {
            used += (size_t)wrote;
            shown++;
        }
    }
    if ((ke->n_ops > shown) || (ke->has_reduce && shown == 4))
        snprintf(buf + used, bufsz > used ? bufsz - used : 0, "+...");
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
    Term a = term_era();
    Term b = term_era();
    if (uop == UOP_FUSE) {
        a = heap_read(ctx, loc + 0);
    } else if (uop == UOP_KERNEL) {
        Term kernel = term_new(TAG_TOP, UOP_KERNEL, loc);
        if (thvm_kernel_is_monolithic(ctx, kernel))
            return dot_term_shape(ctx, thvm_kernel_monolithic_payload(ctx, kernel), out);
        a = heap_read(ctx, loc + 0);
        b = heap_read(ctx, loc + 1);
        u32 kop = thvm_kernel_root_uop(ctx, term_new(TAG_TOP, UOP_KERNEL, loc));
        Shape sa = SHAPE(1), sb = SHAPE(1);
        int has_a = dot_term_shape(ctx, a, &sa);
        int has_b = dot_term_shape(ctx, b, &sb);
        if (kop == UOP_RESHAPE || kop == UOP_EXPAND) {
            if (dot_meta_shape_from_tensor(ctx, b, out)) return 1;
            if (has_a) { *out = sa; return 1; }
            return 0;
        }
        if (kop == UOP_SUM || kop == UOP_RMAX) {
            if (has_a) { *out = sa; return 1; }
            return 0;
        }
        if (kop == UOP_COUNT) {
            if (has_b) { *out = sb; return 1; }
            if (has_a) { *out = sa; return 1; }
            return 0;
        }
        if (is_binary(kop) || kop == UOP_CMP) {
            if (has_a && has_b && dot_shape_broadcast(sa, sb, out)) return 1;
            if (has_a) { *out = sa; return 1; }
            if (has_b) { *out = sb; return 1; }
            return 0;
        }
        if (has_a) { *out = sa; return 1; }
        return 0;
    } else {
        a = heap_read(ctx, loc + 0);
        b = heap_read(ctx, loc + 1);
    }
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


static u32 dot_term_arity(TinyHVM *ctx, Term t) {
    u8 tag = term_tag(t);
    u32 ext = term_ext(t);
    switch (tag) {
        case TAG_TOP:
            if (ctx && ext == UOP_KERNEL && thvm_kernel_is_monolithic(ctx, t))
                return 0;
            return thvm_uop_storage_arity(ext);
        case TAG_APP:
        case TAG_LAM:
        case TAG_BRI:
        case TAG_SUP:
        case TAG_USP:
        case TAG_OP2:
        case TAG_EQL:
        case TAG_AND:
        case TAG_OR:
        case TAG_SEQ:
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



// Dump heap as DOT — flat walk, no BFS. Every combinator shown faithfully.
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root);
static void thvm_heap_dot(TinyHVM *ctx, const char *path) {
    thvm_heap_dot_root(ctx, path, term_era());
}
static void thvm_heap_dot_root(TinyHVM *ctx, const char *path, Term root) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "heap_dot: can't open %s\n", path); return; }
    DotLayers layers = dot_layers_from_env();
    thvm_heap_dot_reset_kernel_fallback_ids();
    // Save heap_pos: dump-only helpers (e.g. thvm_kernel_to_compute
    // allocating a temporary MUL payload to flatten a growing kernel)
    // expand the heap.  Without a restore those cells stick around as
    // "live" TOPs on subsequent renders (e.g. the stray MUL@138 that
    // used to appear in later step frames).  Restoring at dump-exit
    // discards dumper-local scratch without affecting reducer state.
    u64 _dot_heap_pos_save = ctx->heap_pos;
    fprintf(f, "digraph G {\n");
    if (heap_dot_prev_name[0]) fprintf(f, "  // PREV_INTERACTION: %s\n", heap_dot_prev_name);
    if (heap_dot_next_name[0]) fprintf(f, "  // NEXT_INTERACTION: %s\n", heap_dot_next_name);
    fprintf(f, "  rankdir=BT;\n");
    fprintf(f, "  node [fontname=\"Helvetica\", fontsize=10, style=filled, shape=box, margin=\"0.1,0.05\"];\n");
    fprintf(f, "  edge [fontsize=8, fontname=\"Helvetica\"];\n\n");

    // Track emitted tensor nodes (avoid duplicates)
    u8 ten_emitted[256]; memset(ten_emitted, 0, sizeof(ten_emitted));
    int dot_show_tensor_vals = (getenv("THVM_STEP_GRAPH_TENSOR_VALUES") != NULL) ||
                               (getenv("THVM_GRAPH_TENSOR_VALUES") != NULL);
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
            char _vals[192]=""; \
            if (_pslot) snprintf(_slot, sizeof(_slot), "\\nslot%u%s", _pslot, _planned ? " planned" : ""); \
            if (dot_show_tensor_vals && _m->buf_id && _m->view.numel <= 16 && _m->view.contiguous) { \
                u32 _esz = dtype_size(_m->dtype); \
                u8 _raw[64]; \
                if (_esz > 0 && (_m->view.numel * _esz) <= sizeof(_raw)) \
                    _m->backend->buf_read(_m->buf_id, _raw, _m->view.numel * _esz); \
                int _vp = snprintf(_vals, sizeof(_vals), "\\nvals=["); \
                for (u32 _i = 0; _i < _m->view.numel && _vp > 0 && _vp < (int)sizeof(_vals) - 2; _i++) { \
                    if (_m->dtype == DTYPE_I32) { \
                        i32 _v = ((i32 *)_raw)[_i]; \
                        _vp += snprintf(_vals + _vp, sizeof(_vals) - (size_t)_vp, "%s%d", _i ? "," : "", _v); \
                    } else if (_m->dtype == DTYPE_U32) { \
                        u32 _v = ((u32 *)_raw)[_i]; \
                        _vp += snprintf(_vals + _vp, sizeof(_vals) - (size_t)_vp, "%s%u", _i ? "," : "", _v); \
                    } else if (_m->dtype == DTYPE_F16) { \
                        f32 _v = f16_bits_to_f32(((u16 *)_raw)[_i]); \
                        _vp += snprintf(_vals + _vp, sizeof(_vals) - (size_t)_vp, "%s%.6g", _i ? "," : "", _v); \
                    } else { \
                        f32 _v = ((f32 *)_raw)[_i]; \
                        _vp += snprintf(_vals + _vp, sizeof(_vals) - (size_t)_vp, "%s%.6g", _i ? "," : "", _v); \
                    } \
                } \
                if (_vp > 0 && _vp < (int)sizeof(_vals)) snprintf(_vals + _vp, sizeof(_vals) - (size_t)_vp, "]"); \
            } \
            fprintf(f,"  t%u [label=\"t%u\\n[%s]\\n%s %s%s%s%s\",shape=box,fillcolor=\"%s\"%s];\n", \
                    (tid),(tid),_sh,_dt,_bk,_m->requires_grad?" grad":"",_slot,_vals,_fc, \
                    (term_tag(root) == TAG_TEN && (u32)term_val(root) == (u32)(tid)) ? ",color=\"#1f78ff\",penwidth=2.2" : ""); \
        } } while(0)

    u32 nn = 0; // node count for stats
    // Track emitted node locations (dedup TAG_TOP by val)
    #define NODE_DEDUP_MAX 1024
    u64 node_emitted[NODE_DEDUP_MAX]; u32 n_node_emitted = 0;
    #define NODE_SEEN(v) ({ int _s=0; for(u32 _i=0;_i<n_node_emitted;_i++) \
        if(node_emitted[_i]==(v)){_s=1;break;} _s; })
    #define NODE_MARK(v) do { if(n_node_emitted<NODE_DEDUP_MAX) node_emitted[n_node_emitted++]=(v); } while(0)
    // Track liveness buffer size separately: dot_kernel_entry_for_term
    // (called later while rendering KERNEL labels) allocates scratch
    // via thvm_kernel_to_compute, which grows ctx->heap_pos mid-render.
    // Subsequent top_live[val] reads for val in the new region would be
    // out-of-bounds against this calloc buffer — fatally reading stale
    // heap bytes that make orphan MULs appear "live".
    u64 _dot_live_cap = ctx->heap_pos;
    u8 *dp_slot_emitted = (u8 *)calloc((size_t)_dot_live_cap, 1);
    u8 *top_live = (u8 *)calloc((size_t)_dot_live_cap, 1);
    u8 *slot_live = (u8 *)calloc((size_t)_dot_live_cap, 1);
    u8 *loc_live = (u8 *)calloc((size_t)_dot_live_cap, 1);
    u8 *ctr_children_emitted = (u8 *)calloc((size_t)_dot_live_cap, 1);
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
            const char *_tn = dot_heap_tag_name(term_tag(_rt)); \
            if (_tn[0] == '?') \
                fprintf(f, "  h%llu [label=\"h%llu\\ntag=%u\", shape=box, fillcolor=\"#dddddd\", fontsize=8%s];\n", \
                        (unsigned long long)(hid), (unsigned long long)(hid), (u32)term_tag(_rt), \
                        ROOT_NODE_ATTRS_TERM(_rt)); \
            else \
                fprintf(f, "  h%llu [label=\"h%llu\\n%s\", shape=box, fillcolor=\"#dddddd\", fontsize=8%s];\n", \
                        (unsigned long long)(hid), (unsigned long long)(hid), _tn, \
                        ROOT_NODE_ATTRS_TERM(_rt)); \
        } \
    } while(0)
    #define EMIT_CTR_NODE(cloc) do { \
        if (!NODE_SEEN(CTR_NODE_KEY(cloc))) { \
            NODE_MARK(CTR_NODE_KEY(cloc)); \
            fprintf(f, "  ctr%llu [label=\"CTR\\n@%llu\", shape=hexagon, fillcolor=\"#f3f3f3\", fontsize=9%s];\n", \
                    (unsigned long long)(cloc), \
                    (unsigned long long)(cloc), ROOT_NODE_ATTRS_TAG_VAL(TAG_CTR, cloc)); \
        } \
    } while(0)
    #define EMIT_ANY_NODE(apos) do { \
        if (!NODE_SEEN(ANY_NODE_KEY(apos))) { \
            NODE_MARK(ANY_NODE_KEY(apos)); \
            fprintf(f, "  any%llu [label=\"ANY\\n@%llu\", shape=oval, fillcolor=\"#eeeeee\", fontsize=8%s%s];\n", \
                    (unsigned long long)(apos), (unsigned long long)(apos), NODE_HL_ATTRS(apos), \
                    ROOT_NODE_ATTRS_TAG_VAL(TAG_ANY, apos)); \
        } \
    } while(0)
    #define EMIT_REF_NODE(slot, name) do { \
        if (!NODE_SEEN(REF_NODE_KEY(slot))) { \
            NODE_MARK(REF_NODE_KEY(slot)); \
            fprintf(f, "  ref%llu [label=\"REF\\n#%u@%llu\", shape=oval, fillcolor=\"#eeeeee\", fontsize=8%s%s];\n", \
                    (unsigned long long)(slot), (u32)(name), (unsigned long long)(slot), NODE_HL_ATTRS(slot), \
                    (term_tag(root) == TAG_REF && step_root_slot == (u64)(slot)) ? ",color=\"#1f78ff\",penwidth=2.2" : ""); \
        } \
    } while(0)
    #define EMIT_VAR_NODE(slot, loc, is_sub) do { \
        (void)(slot); \
        (void)(loc); \
        (void)(is_sub); \
    } while(0)
    #define SLOT_LIVE(pos) (((pos) < ctx->heap_pos) ? slot_live[(pos)] : 0)
    // In whole-heap mode (root_only=0), there's no privileged root to
    // traverse from — every cell in the heap is treated as live.
    // root_only=1 (coarse-phase dumps) keeps the reach-walk-based
    // filtering to trim noise from the entire history of allocations.
    #define LOC_LIVE(pos)  (heap_dot_root_only ? \
        (((pos) < ctx->heap_pos) ? loc_live[(pos)] : 0) : \
        ((pos) > 0 && (pos) < ctx->heap_pos))
    // Node highlight: red border when edge highlight failed
    #define NODE_HL_ATTRS(loc) ((heap_dot_node_hl > 0 && (u64)(loc) == heap_dot_node_hl) ? \
        ",color=\"#cc0000\",penwidth=2.0" : "")
    #define ROOT_NODE_ATTRS_TERM(tt) ((term_tag(root) == term_tag((tt)) && \
                                       term_ext(root) == term_ext((tt)) && \
                                       term_val(root) == term_val((tt))) ? \
        ",color=\"#1f78ff\",penwidth=2.2" : "")
    #define ROOT_NODE_ATTRS_TAG_VAL(tag, val) ((term_tag(root) == (tag) && \
                                                term_val(root) == (u64)(val)) ? \
        ",color=\"#1f78ff\",penwidth=2.2" : "")
    #define IS_INLINE_UNARY_CTR(tt) ({ \
        Term _tt = (tt); \
        u64 _tv = term_val(_tt); \
        term_tag(_tt) == TAG_CTR && term_ext(_tt) == 1 && _tv > 0 && _tv < ctx->heap_pos; \
    })
    #define INLINE_UNARY_CTR_CHILD(tt) heap_read(ctx, term_val((tt)))
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
        //
        // In step-graph mode (root_only), skip this ERA seeding: an ERA at
        // slot @X with val=Y means "erased reference to @Y", and pulling Y's
        // subtree in drags orphan-chain MUL/ADD scratch into the reach set.
        // The step-graph narrative is about what's reachable from the root
        // term; detached ERA remnants are noise.
        for (u64 h = 1; h < ctx->heap_pos; h++) {
            Term ht = ctx->heap[h];
            if (term_tag(ht) == TAG_ERA && term_val(ht) != 0 && !heap_dot_root_only) {
                slot_live[h] = 1;
                PUSH_TERM(ht);
            }
            if (!heap_dot_root_only) {
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
            if (heap_dot_include_all_slots &&
                !(term_tag(ht) == TAG_ERA && term_val(ht) == 0)) {
                // Force every non-inert slot live so wnf scaffolding
                // (DUP cells, intermediate TOPs, ALO stubs, …) shows up
                // in each per-step snapshot.
                slot_live[h] = 1;
                PUSH_TERM(ht);
            }
        }

        // Seed definition bodies so step graphs retain the dashed REF -> def
        // context; otherwise recursive substitutions appear out of nowhere.
        for (u32 _di = 0; _di < ctx->def_count; _di++) {
            Term _dt = ctx->defs[_di];
            if (term_tag(_dt) != TAG_ERA) PUSH_TERM(_dt);
        }

        while (work && wp > 0) {
            Term tt = work[--wp];
            u8 tg = term_tag(tt);
            u64 tv = term_val(tt);
            if (dot_visible_heap_loc_tag(tg) && tv > 0 && tv < ctx->heap_pos)
                loc_live[tv] = 1;
            if (tg == TAG_TOP) {
                if (tv == 0 || tv >= _dot_live_cap || top_live[tv]) continue;
                top_live[tv] = 1;
                // Monolithic kernels are self-contained: their inputs
                // are rendered via kernel-leaf edges (t1 -> KERNEL[MUL:a]
                // etc.).  Don't descend into their inline compute
                // structure or we'll surface the internal MUL/SUM/ADD
                // cells as standalone orphan nodes.
                if (term_ext(tt) == UOP_KERNEL &&
                    thvm_kernel_is_monolithic(ctx, tt))
                    continue;
                u32 ar = dot_term_arity(ctx, tt);
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
                if (!seen_slot || !seen_slot[dl]) {
                    if (seen_slot) seen_slot[dl] = 1;
                    if (slot_live) slot_live[dl] = 1;
                }
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
            u32 ar = dot_term_arity(ctx, tt);
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
    #define DOT_HL_SLOT_MATCH(pos) ({ \
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
        _m; \
    })
    #define DOT_HL_MATCH_TERM(pos, term) ({ \
        int _m = DOT_HL_SLOT_MATCH(pos); \
        (void)(term); \
        if (_m) heap_dot_hl_hit = 1; \
        _m; \
    })
    #define DOT_HL_MATCH(pos) DOT_HL_MATCH_TERM(pos, 0)
    #define EDGE_HL_ONLY(pos) (DOT_HL_MATCH(pos) ? " [color=\"#cc0000\",penwidth=2.0]" : "")
    #define EDGE_HL_LABEL(pos) (DOT_HL_MATCH(pos) ? ",color=\"#cc0000\",penwidth=2.0" : "")
    #define EDGE_HL_LABEL_TERM(pos, term) (DOT_HL_MATCH_TERM(pos, term) ? ",color=\"#cc0000\",penwidth=2.0" : "")
    #define HL_SLOT_TERM() ((heap_dot_hl_on && heap_dot_hl_slot < ctx->heap_pos) ? heap_read(ctx, heap_dot_hl_slot) : 0)
    #define HL_SLOT_IS_DP() ({ Term _ht = HL_SLOT_TERM(); term_tag(_ht) == TAG_DP0 || term_tag(_ht) == TAG_DP1; })
    #define DUP_PORT_HAS_VISIBLE_CONSUMER(_dloc, _ptag) ({ \
        int _has = 0; \
        for (u64 _hh = 1; _hh < ctx->heap_pos && !_has; _hh++) { \
            if (!SLOT_LIVE(_hh)) continue; \
            Term _pp = ctx->heap[_hh]; \
            if (term_tag(_pp) == (_ptag) && term_val(_pp) == (_dloc)) { _has = 1; break; } \
            u64 _pl = term_val(_pp); \
            u32 _pa = dot_term_arity(ctx, _pp); \
            if (_pa == 0 || _pl == 0 || _pl + _pa > ctx->heap_pos) continue; \
            for (u32 _pi = 0; _pi < _pa; _pi++) { \
                Term _ch = heap_read(ctx, _pl + _pi); \
                if (term_tag(_ch) != (_ptag)) continue; \
                u64 _dl = term_val(_ch); \
                if (_dl == 0 || _dl >= ctx->heap_pos) continue; \
                if (_dl == (_dloc)) { _has = 1; break; } \
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
                fprintf(f, "  dup%llu [label=\"\", shape=circle, width=0.12, height=0.12, fixedsize=true, fillcolor=\"#d4b8e8\", color=\"#8a63b7\", fontsize=1%s%s];\n", \
                        (unsigned long long)_cur, _hl_principal ? ",penwidth=2.0" : "", \
                        ((term_tag(root) == TAG_DP0 || term_tag(root) == TAG_DP1) && term_val(root) == _cur) ? ",color=\"#1f78ff\",penwidth=2.2" : ""); \
            } \
            if (!NODE_SEEN(_cur + 0x180000)) { \
                NODE_MARK(_cur + 0x180000); \
                if (!DUP_PORT_HAS_VISIBLE_CONSUMER(_cur, TAG_DP0)) { \
                    fprintf(f, "  freedup%llu_0 [label=\"\",shape=circle,width=0.14,height=0.14,fixedsize=true,fillcolor=\"#ffffff\",color=\"#888888\",fontsize=1];\n", \
                            (unsigned long long)_cur); \
                    char _dp_lbl0[32]; \
                    fprintf(f, "  dup%llu -> freedup%llu_0 [label=\"%s\",style=dotted,color=\"#999999\"];\n", \
                            (unsigned long long)_cur, (unsigned long long)_cur, \
                            dot_dp_port_label(_dp_lbl0, sizeof(_dp_lbl0), "", TAG_DP0, _cur)); \
                } \
                if (!DUP_PORT_HAS_VISIBLE_CONSUMER(_cur, TAG_DP1)) { \
                    fprintf(f, "  freedup%llu_1 [label=\"\",shape=circle,width=0.14,height=0.14,fixedsize=true,fillcolor=\"#ffffff\",color=\"#888888\",fontsize=1];\n", \
                            (unsigned long long)_cur); \
                    char _dp_lbl1[32]; \
                    fprintf(f, "  dup%llu -> freedup%llu_1 [label=\"%s\",style=dotted,color=\"#999999\"];\n", \
                            (unsigned long long)_cur, (unsigned long long)_cur, \
                            dot_dp_port_label(_dp_lbl1, sizeof(_dp_lbl1), "", TAG_DP1, _cur)); \
                } \
            } \
            Term _shared = heap_read(ctx, _cur); \
            u8 _stag = term_tag(_shared); \
            if (_stag == TAG_DP0 || _stag == TAG_DP1) { \
                u64 _up = term_val(_shared); \
                if (_up == _cur) break; \
                if (_cur_new) { \
                    char _dp_chain_lbl[32]; \
                    fprintf(f, "  dup%llu -> dup%llu [label=\"%s\"];\n", \
                        (unsigned long long)_up, (unsigned long long)_cur, \
                        dot_dp_port_label(_dp_chain_lbl, sizeof(_dp_chain_lbl), "", _stag, _cur)); \
                } \
                _cur = _up; \
                continue; \
            } \
            if (_cur_new) { \
                if (_stag == TAG_TOP) fprintf(f, "  n%llu -> dup%llu%s;\n", (unsigned long long)term_val(_shared), (unsigned long long)_cur, _hl_principal ? " [color=\"#cc0000\",penwidth=2.0]" : ""); \
                else if (_stag == TAG_TEN) { \
                    EMIT_TEN((u32)term_val(_shared)); \
                    char _slot_attrs[96]; \
                    dot_tensor_edge_slot_attrs(_cur, _slot_attrs, sizeof(_slot_attrs)); \
                    fprintf(f, "  t%u -> dup%llu[label=\"\"%s%s];\n", \
                            (u32)term_val(_shared), (unsigned long long)_cur, _slot_attrs, \
                            _hl_principal ? ",color=\"#cc0000\",penwidth=2.0" : ""); \
                } \
                else if (_stag == TAG_CTR) { EMIT_CTR_NODE(term_val(_shared)); fprintf(f, "  ctr%llu -> dup%llu%s;\n", (unsigned long long)term_val(_shared), (unsigned long long)_cur, _hl_principal ? " [color=\"#cc0000\",penwidth=2.0]" : ""); } \
                else if (_stag == TAG_REF) { EMIT_REF_NODE(_cur, term_ext(_shared)); fprintf(f, "  ref%llu -> dup%llu%s;\n", (unsigned long long)_cur, (unsigned long long)_cur, _hl_principal ? " [color=\"#cc0000\",penwidth=2.0]" : ""); } \
                else if (_stag == TAG_VAR) { EMIT_VAR_OR_RESOLVED_TO_TARGET(_shared, term_val(_shared), "dup", _cur, "", _hl_principal, ""); } \
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

    // EMIT_GRAD_CELL: render a GRAD cell (the heap slot holding y for a
    // (x,dx) = GRAD(y) pair). Drawn as an upside-down triangle with fwd
    // and bwd aux ports. Idempotent: first caller emits the node + fwd+bwd
    // free-port stubs if they have no visible consumer; subsequent callers
    // just wire their edge.
    #define EMIT_GRAD_CELL(gloc) do { \
        u64 _gcur = (gloc); \
        if (_gcur > 0 && _gcur < ctx->heap_pos) { \
            if (!NODE_SEEN(_gcur + 0x200000)) { \
                NODE_MARK(_gcur + 0x200000); \
                fprintf(f, "  grad%llu [label=\"GRAD\\n@%llu\", shape=invtriangle, fillcolor=\"#e8d0ff\"];\n", \
                        (unsigned long long)_gcur, (unsigned long long)_gcur); \
                /* Wire body (y) INTO the cell — the principal port. */ \
                Term _gy = heap_read(ctx, _gcur); \
                u8 _gyt = term_tag(_gy); \
                if (_gyt == TAG_TOP) \
                    fprintf(f, "  n%llu -> grad%llu [label=\"y\",color=\"#cc0000\",penwidth=2.0];\n", \
                            (unsigned long long)term_val(_gy), (unsigned long long)_gcur); \
                else if (_gyt == TAG_TEN) { \
                    EMIT_TEN((u32)term_val(_gy)); \
                    fprintf(f, "  t%u -> grad%llu [label=\"y\",color=\"#cc0000\",penwidth=2.0];\n", \
                            (u32)term_val(_gy), (unsigned long long)_gcur); \
                } else if (_gyt == TAG_CTR) { \
                    EMIT_CTR_NODE(term_val(_gy)); \
                    fprintf(f, "  ctr%llu -> grad%llu [label=\"y\",color=\"#cc0000\",penwidth=2.0];\n", \
                            (unsigned long long)term_val(_gy), (unsigned long long)_gcur); \
                } \
            } \
        } \
    } while (0)
    #define ERA_DEDUP_MAX 1024
    u64 era_emitted[ERA_DEDUP_MAX]; u32 n_era_emitted = 0;
    #define ERA_SEEN(v) ({ int _s=0; for(u32 _i=0;_i<n_era_emitted;_i++) \
        if(era_emitted[_i]==(v)){_s=1;break;} _s; })
    #define ERA_MARK(v) do { if(n_era_emitted<ERA_DEDUP_MAX) era_emitted[n_era_emitted++]=(v); } while(0)
    #define EMIT_ERA_NODE(epos, eterm) do { \
        if (!ERA_SEEN(epos)) { \
            ERA_MARK(epos); \
            fprintf(f, "  era%llu [label=\"ERA\\n@%llu\",shape=circle,width=0.38,height=0.38,fixedsize=true,fillcolor=\"#ffb8b8\",fontsize=7%s];\n", \
                    (unsigned long long)(epos), (unsigned long long)(epos), \
                    ROOT_NODE_ATTRS_TAG_VAL(TAG_ERA, epos)); \
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
                    char _dp_era_lbl[32]; \
                    fprintf(f, "  dup%llu -> era%llu [label=\"%s\"%s];\n", _dl, (unsigned long long)(epos), \
                            dot_dp_port_label(_dp_era_lbl, sizeof(_dp_era_lbl), "", _st, _ev), EDGE_HL_LABEL_TERM(epos, _et)); \
                } else if (_st == TAG_TOP) { \
                    fprintf(f, "  n%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)term_val(_src), (unsigned long long)(epos), EDGE_HL_LABEL_TERM(epos, _et)); \
                } else if (_st == TAG_TEN) { \
                    EMIT_TEN((u32)term_val(_src)); \
                    char _slot_attrs[96]; \
                    dot_tensor_edge_slot_attrs(_ev, _slot_attrs, sizeof(_slot_attrs)); \
                    fprintf(f, "  t%u -> era%llu [label=\"p\"%s%s];\n", (u32)term_val(_src), (unsigned long long)(epos), _slot_attrs, EDGE_HL_LABEL_TERM(epos, _et)); \
                } else if (_st == TAG_CTR) { \
                    EMIT_CTR_NODE(term_val(_src)); \
                    fprintf(f, "  ctr%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)term_val(_src), (unsigned long long)(epos), EDGE_HL_LABEL_TERM(epos, _et)); \
                } else if (_st == TAG_REF) { \
                    EMIT_REF_NODE(_ev, term_ext(_src)); \
                    fprintf(f, "  ref%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)_ev, (unsigned long long)(epos), EDGE_HL_LABEL_TERM(epos, _et)); \
                } else if (_st == TAG_VAR) { \
                    EMIT_VAR_OR_RESOLVED_TO_TARGET(_src, term_val(_src), "era", (epos), "p", DOT_HL_MATCH_TERM(epos, _et), ""); \
                } else if (dot_visible_heap_loc_tag(_st)) { \
                    fprintf(f, "  n%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)term_val(_src), (unsigned long long)(epos), EDGE_HL_LABEL_TERM(epos, _et)); \
                } else if (_st == TAG_NUM) { \
                    f32 _fv; u32 _bv=(u32)term_val(_src); memcpy(&_fv,&_bv,4); \
                    fprintf(f, "  num_era%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", (unsigned long long)(epos), (double)_fv); \
                    fprintf(f, "  num_era%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)(epos), (unsigned long long)(epos), EDGE_HL_LABEL_TERM(epos, _et)); \
                } else if (_st == TAG_ANY) { \
                    EMIT_ANY_NODE(_ev); \
                    fprintf(f, "  any%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)_ev, (unsigned long long)(epos), EDGE_HL_LABEL_TERM(epos, _et)); \
                } else if (_st == TAG_ERA) { \
                    /* If the payload cell already contains inert ERA(0), don't draw \
                       a second floating ERA node just to point back into the active \
                       eraser. The outer ERA node already fully represents the state. */ \
                    if (term_val(_src) != 0) { \
                        if (!ERA_SEEN(_ev)) { \
                            ERA_MARK(_ev); \
                            fprintf(f, "  era%llu [label=\"ERA\\n@%llu\",shape=circle,width=0.38,height=0.38,fixedsize=true,fillcolor=\"#ffb8b8\",fontsize=7];\n", \
                                    (unsigned long long)_ev, (unsigned long long)_ev); \
                        } \
                        fprintf(f, "  era%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)_ev, (unsigned long long)(epos), EDGE_HL_LABEL_TERM(epos, _et)); \
                    } \
                } else { \
                    EMIT_RAW_NODE(_ev, _src); \
                    fprintf(f, "  h%llu -> era%llu [label=\"p\"%s];\n", (unsigned long long)_ev, (unsigned long long)(epos), EDGE_HL_LABEL_TERM(epos, _et)); \
                } \
            } \
        } \
    } while(0)
    #define EMIT_FREE_PORT(parent_loc, port_idx) do { \
        fprintf(f, "  free%llu_%u [label=\"\",shape=circle,width=0.14,height=0.14,fixedsize=true,fillcolor=\"#ffffff\",color=\"#888888\",fontsize=1];\n", \
                (unsigned long long)(parent_loc), (u32)(port_idx)); \
    } while(0)
    #define TERM_HAS_PARENT_REF(tref) ({ \
        int _hasp = 0; \
        for (u64 _hh = 1; _hh < ctx->heap_pos && !_hasp; _hh++) { \
            Term _pp = ctx->heap[_hh]; \
            u8 _pt = term_tag(_pp); \
            u64 _pv = term_val(_pp); \
            /* Whole-heap mode (root_only=0): treat every in-bounds slot \
               as live — there's no reach-walk seed to populate liveness \
               arrays.  Otherwise require top_live/loc_live from the \
               root-anchored walk. */ \
            if (_pt == TAG_TOP) { \
                if (heap_dot_root_only) { \
                    if (!top_live || _pv == 0 || _pv >= ctx->heap_pos || !top_live[_pv]) continue; \
                } else { \
                    if (_pv == 0 || _pv >= ctx->heap_pos) continue; \
                } \
            } else if (dot_visible_heap_loc_tag(_pt)) { \
                if (_pv == 0 || _pv >= ctx->heap_pos || !LOC_LIVE(_pv)) continue; \
            } \
            u32 _pa = dot_term_arity(ctx, _pp); \
            u64 _pl = term_val(_pp); \
            if (_pa == 0 || _pl == 0 || _pl + _pa > ctx->heap_pos) continue; \
            for (u32 _pi = 0; _pi < _pa; _pi++) { \
                Term _prt = heap_read(ctx, _pl + _pi); \
                if (term_is_sub(_prt)) _prt = term_strip_sub(_prt); \
                if (_prt == (tref)) { _hasp = 1; break; } \
            } \
        } \
        for (u32 _di = 0; _di < ctx->def_count && !_hasp; _di++) { \
            Term _pp = ctx->defs[_di]; \
            u32 _pa = dot_term_arity(ctx, _pp); \
            u64 _pl = term_val(_pp); \
            if (_pa == 0 || _pl == 0 || _pl + _pa > ctx->heap_pos) continue; \
            for (u32 _pi = 0; _pi < _pa; _pi++) { \
                Term _prt = heap_read(ctx, _pl + _pi); \
                if (term_is_sub(_prt)) _prt = term_strip_sub(_prt); \
                if (_prt == (tref)) { _hasp = 1; break; } \
            } \
        } \
        _hasp; \
    })
    #define TERM_HAS_PARENT_APP_ARG(tref) ({ \
        int _hasp = 0; \
        for (u64 _hh = 1; _hh < ctx->heap_pos && !_hasp; _hh++) { \
            if (!SLOT_LIVE(_hh)) continue; \
            Term _pp = ctx->heap[_hh]; \
            if (term_tag(_pp) != TAG_APP) continue; \
            u64 _pl = term_val(_pp); \
            if (_pl == 0 || _pl + 1 >= ctx->heap_pos) continue; \
            if (heap_read(ctx, _pl + 1) == (tref)) _hasp = 1; \
        } \
        _hasp; \
    })
    #define TERM_IS_DEF_ROOT(tref) ({ \
        int _is = 0; \
        for (u32 _di = 0; _di < ctx->def_count; _di++) { \
            if (ctx->defs[_di] == (tref)) { _is = 1; break; } \
        } \
        _is; \
    })
    #define REF_TARGETS_LOC(locv) ({ \
        int _hit = 0; \
        for (u64 _hh = 1; _hh < ctx->heap_pos && !_hit; _hh++) { \
            if (!SLOT_LIVE(_hh)) continue; \
            Term _rt = ctx->heap[_hh]; \
            if (term_tag(_rt) != TAG_REF) continue; \
            u32 _ext = term_ext(_rt); \
            if (_ext >= ctx->def_count) continue; \
            Term _d = ctx->defs[_ext]; \
            if (dot_visible_heap_loc_tag(term_tag(_d)) && term_val(_d) == (locv)) \
                _hit = 1; \
        } \
        _hit; \
    })
    #define ERA_HAS_VISIBLE_VAR_CONSUMER(epos) ({ \
        int _hv = 0; \
        if (term_tag(root) == TAG_VAR && term_val(root) == (epos)) _hv = 1; \
        for (u64 _vh = 1; _vh < ctx->heap_pos && !_hv; _vh++) { \
            if (!SLOT_LIVE(_vh)) continue; \
            Term _vv = heap_read(ctx, _vh); \
            if (term_tag(_vv) == TAG_VAR && term_val(_vv) == (epos)) _hv = 1; \
        } \
        _hv; \
    })
    #define EMIT_VAR_OR_RESOLVED_TO_TARGET(var_term, var_loc, dst_prefix, dst_loc, elbl, edge_hl, extra_attrs) do { \
        (void)(var_term); \
        int _edge_hl = (edge_hl) || (heap_dot_hl_on && heap_dot_hl_slot == (var_loc) && \
                         term_tag(heap_dot_hl_term) == TAG_VAR && term_val(heap_dot_hl_term) == (var_loc)); \
        const char *_extra_attrs = (extra_attrs); \
        const char *_hl_attrs = _edge_hl ? ",color=\"#cc0000\",penwidth=2.0" : ""; \
        u64 _cell = (var_loc); \
        Term _rv = dot_resolve_var_term(ctx, (var_loc), &_cell); \
        u8 _rt = term_tag(_rv); \
        u64 _rvv = term_val(_rv); \
        if (_rt == TAG_TEN) { \
            EMIT_TEN((u32)_rvv); \
            char _slot_attrs[96]; \
            dot_tensor_edge_slot_attrs(_cell, _slot_attrs, sizeof(_slot_attrs)); \
            fprintf(f, "  t%u -> " dst_prefix "%llu [label=\"%s\"%s%s%s];\n", \
                    (u32)_rvv, (unsigned long long)(dst_loc), (elbl), _slot_attrs, _extra_attrs, _hl_attrs); \
        } else if (_rt == TAG_NUM) { \
            f32 _fv; u32 _bv = (u32)_rvv; memcpy(&_fv, &_bv, 4); \
            fprintf(f, "  num_vsrc%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", \
                    (unsigned long long)(var_loc), (double)_fv); \
            fprintf(f, "  num_vsrc%llu -> " dst_prefix "%llu [label=\"%s\"%s%s];\n", \
                    (unsigned long long)(var_loc), (unsigned long long)(dst_loc), (elbl), _extra_attrs, _hl_attrs); \
        } else if (_rt == TAG_REF) { \
            EMIT_REF_NODE(_cell, term_ext(_rv)); \
            fprintf(f, "  ref%llu -> " dst_prefix "%llu [label=\"%s\"%s%s];\n", \
                    (unsigned long long)_cell, (unsigned long long)(dst_loc), (elbl), _extra_attrs, _hl_attrs); \
        } else if (_rt == TAG_ERA && _rvv != 0) { \
            if (!ERA_SEEN(_cell)) { \
                ERA_MARK(_cell); \
                fprintf(f, "  era%llu [label=\"ERA@%llu\",shape=circle,fillcolor=\"#fff0f0\"];\n", \
                        (unsigned long long)_cell, (unsigned long long)_cell); \
            } \
            fprintf(f, "  era%llu -> " dst_prefix "%llu [label=\"%s\"%s%s];\n", \
                    (unsigned long long)_cell, (unsigned long long)(dst_loc), (elbl), _extra_attrs, _hl_attrs); \
        } else if (_rt == TAG_DP0 || _rt == TAG_DP1) { \
            if (!NODE_SEEN(_rvv + 0x100000)) { \
                NODE_MARK(_rvv + 0x100000); \
                fprintf(f, "  dup%llu [label=\"\", shape=circle, width=0.12, height=0.12, fixedsize=true, fillcolor=\"#d4b8e8\", color=\"#8a63b7\", fontsize=1];\n", \
                        (unsigned long long)_rvv); \
            } \
            fprintf(f, "  dup%llu -> " dst_prefix "%llu [label=\"%s\"%s%s];\n", \
                    (unsigned long long)_rvv, (unsigned long long)(dst_loc), (elbl), _extra_attrs, _hl_attrs); \
        } else if (_rt == TAG_CTR) { \
            EMIT_CTR_NODE(_rvv); \
            fprintf(f, "  ctr%llu -> " dst_prefix "%llu [label=\"%s\"%s%s];\n", \
                    (unsigned long long)_rvv, (unsigned long long)(dst_loc), (elbl), _extra_attrs, _hl_attrs); \
        } else if (_rt == TAG_ANY) { \
            EMIT_ANY_NODE(_cell); \
            fprintf(f, "  any%llu -> " dst_prefix "%llu [label=\"%s\"%s%s];\n", \
                    (unsigned long long)_cell, (unsigned long long)(dst_loc), (elbl), _extra_attrs, _hl_attrs); \
        } else if (_rt == TAG_VAR) { \
            fprintf(f, "  n%llu -> " dst_prefix "%llu [label=\"%s\"%s%s];\n", \
                    (unsigned long long)_rvv, (unsigned long long)(dst_loc), (elbl), _extra_attrs, _hl_attrs); \
        } else if (_rt == TAG_TOP || dot_visible_heap_loc_tag(_rt)) { \
            if (!NODE_SEEN(_rvv)) { \
                NODE_MARK(_rvv); \
                char _rlabel[96]; \
                dot_raw_slot_term_label(ctx, _rv, _rlabel, sizeof(_rlabel)); \
                fprintf(f, "  n%llu [label=\"%s\", shape=%s, fillcolor=\"%s\"];\n", \
                        (unsigned long long)_rvv, _rlabel, \
                        dot_heap_node_shape(_rt), dot_heap_node_color(_rt)); \
            } \
            fprintf(f, "  n%llu -> " dst_prefix "%llu [label=\"%s\"%s%s];\n", \
                    (unsigned long long)_rvv, (unsigned long long)(dst_loc), (elbl), _extra_attrs, _hl_attrs); \
        } else { \
            EMIT_RAW_NODE(_cell, _rv); \
            fprintf(f, "  h%llu -> " dst_prefix "%llu [label=\"%s\"%s%s];\n", \
                    (unsigned long long)_cell, (unsigned long long)(dst_loc), (elbl), _extra_attrs, _hl_attrs); \
        } \
    } while(0)
    #define EMIT_VAR_OR_RESOLVED_TO_NODE(var_term, var_loc, dst_loc, elbl, edge_hl) do { \
        EMIT_VAR_OR_RESOLVED_TO_TARGET((var_term), (var_loc), "n", (dst_loc), (elbl), (edge_hl), ""); \
    } while(0)
    // REF definitions live in ctx->defs rather than ctx->heap[h], so emit the
    // structural root explicitly when we draw REF -> def edges.
    #define EMIT_REF_DEF_ROOT(def_term) do { \
        Term _dr = (def_term); \
        u8 _dtag = term_tag(_dr); \
        u32 _dext = term_ext(_dr); \
        u64 _dval = term_val(_dr); \
        if (!dot_visible_heap_loc_tag(_dtag) || _dval == 0 || _dval >= ctx->heap_pos) break; \
        if (!NODE_SEEN(_dval)) { \
            NODE_MARK(_dval); \
            char _dlabel[96]; \
            if (_dtag == TAG_SUP || _dtag == TAG_USP) { \
                snprintf(_dlabel, sizeof(_dlabel), "%s #%u@%llu", \
                         dot_heap_tag_name(_dtag), _dext, (unsigned long long)_dval); \
            } else if (_dtag == TAG_LAM || _dtag == TAG_BRI) { \
                snprintf(_dlabel, sizeof(_dlabel), "%s\\n#%u@%llu", \
                         dot_heap_tag_name(_dtag), _dext, (unsigned long long)_dval); \
            } else if (_dtag == TAG_MAT || _dtag == TAG_REF) { \
                snprintf(_dlabel, sizeof(_dlabel), "%s\\n#%u@%llu", \
                         dot_heap_tag_name(_dtag), _dext, (unsigned long long)_dval); \
            } else { \
                snprintf(_dlabel, sizeof(_dlabel), "%s\\n@%llu", \
                         dot_heap_tag_name(_dtag), (unsigned long long)_dval); \
            } \
            fprintf(f, "  n%llu [label=\"%s\", shape=%s, fillcolor=\"%s\"];\n", \
                    (unsigned long long)_dval, _dlabel, \
                    dot_heap_node_shape(_dtag), dot_heap_node_color(_dtag)); \
            u32 _dar = dot_term_arity(ctx, _dr); \
            for (u32 _dai = 0; _dai < _dar && _dval + _dai < ctx->heap_pos; _dai++) { \
                Term _child = heap_read(ctx, _dval + _dai); \
                u8 _ctag = term_tag(_child); \
                u64 _cval = term_val(_child); \
                u64 _cpos = _dval + _dai; \
                const char *_elbl = dot_heap_port_name(_dtag, _dai); \
                char _elbl_buf[32]; \
                snprintf(_elbl_buf, sizeof(_elbl_buf), "%s", _elbl); \
                _elbl = _elbl_buf; \
                if (_ctag == TAG_DP0 || _ctag == TAG_DP1) { \
                    DP_SLOT_MARK(_cpos); \
                    EMIT_DUP_CHAIN(_cval); \
                    char _dp_def_lbl[64]; \
                    fprintf(f, "  dup%llu -> n%llu [label=\"%s\"];\n", \
                            (unsigned long long)_cval, (unsigned long long)_dval, \
                            dot_dp_port_label(_dp_def_lbl, sizeof(_dp_def_lbl), _elbl, _ctag, _cpos)); \
                } else if (_ctag == TAG_TEN) { \
                    EMIT_TEN((u32)_cval); \
                    char _slot_attrs[96]; \
                    dot_tensor_edge_slot_attrs(_cpos, _slot_attrs, sizeof(_slot_attrs)); \
                    fprintf(f, "  t%u -> n%llu [label=\"%s\"%s];\n", \
                            (u32)_cval, (unsigned long long)_dval, _elbl, _slot_attrs); \
                } else if (_ctag == TAG_NUM) { \
                    f32 _fv; u32 _bv = (u32)_cval; memcpy(&_fv, &_bv, 4); \
                    fprintf(f, "  num_refdef%llu_%u [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n", \
                            (unsigned long long)_dval, _dai, (double)_fv); \
                    fprintf(f, "  num_refdef%llu_%u -> n%llu [label=\"%s\"];\n", \
                            (unsigned long long)_dval, _dai, (unsigned long long)_dval, _elbl); \
                } else if (_ctag == TAG_ERA) { \
                    if (_cval != 0) { \
                        EMIT_ERA_NODE(_cpos, _child); \
                        fprintf(f, "  era%llu -> n%llu [label=\"%s\"];\n", \
                                (unsigned long long)_cpos, (unsigned long long)_dval, _elbl); \
                    } else { \
                        EMIT_FREE_PORT(_dval, _dai); \
                        fprintf(f, "  free%llu_%u -> n%llu [label=\"%s\"];\n", \
                                (unsigned long long)_dval, _dai, (unsigned long long)_dval, _elbl); \
                    } \
                } else if (_ctag == TAG_ANY) { \
                    EMIT_ANY_NODE(_cval); \
                    fprintf(f, "  any%llu -> n%llu [label=\"%s\"];\n", \
                            (unsigned long long)_cval, (unsigned long long)_dval, _elbl); \
                } else if (_ctag == TAG_CTR) { \
                    EMIT_CTR_NODE(_cval); \
                    fprintf(f, "  ctr%llu -> n%llu [label=\"%s\"];\n", \
                            (unsigned long long)_cval, (unsigned long long)_dval, _elbl); \
                } else if (_ctag == TAG_REF) { \
                    EMIT_REF_NODE(_cpos, term_ext(_child)); \
                    fprintf(f, "  ref%llu -> n%llu [label=\"%s\"];\n", \
                            (unsigned long long)_cpos, (unsigned long long)_dval, _elbl); \
                } else if (_ctag == TAG_VAR) { \
                    if (!((_dtag == TAG_LAM || _dtag == TAG_BRI) && _dai == 0)) { \
                        EMIT_VAR_OR_RESOLVED_TO_NODE(_child, _cval, _dval, _elbl, 0); \
                    } \
                } else if (_ctag == TAG_TOP || dot_visible_heap_loc_tag(_ctag)) { \
                    fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n", \
                            (unsigned long long)_cval, (unsigned long long)_dval, _elbl); \
                } else { \
                    EMIT_RAW_NODE(_cpos, _child); \
                    fprintf(f, "  h%llu -> n%llu [label=\"%s\"];\n", \
                            (unsigned long long)_cpos, (unsigned long long)_dval, _elbl); \
                } \
            } \
        } \
    } while(0)

    if (term_tag(root) == TAG_TEN) {
        EMIT_TEN((u32)term_val(root));
    } else if (term_tag(root) == TAG_NUM) {
        f32 _fv; u32 _bv = (u32)term_val(root); memcpy(&_fv, &_bv, 4);
        fprintf(f, "  num_root [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8%s];\n",
                (double)_fv,
                term_tag(root) == TAG_NUM ? ",color=\"#1f78ff\",penwidth=2.2" : "");
    }
    // Flat walk: every heap position
    for (u64 h = 1; h < ctx->heap_pos; h++) {
        Term t = ctx->heap[h];
        u8 tag = term_tag(t); u32 ext = term_ext(t); u64 val = term_val(t);

        // --- ERA ---
        if (tag == TAG_ERA) {
            if (val != 0 && val < ctx->heap_pos) {
                Term payload_cell = heap_read(ctx, val);
                if (term_tag(payload_cell) == TAG_ERA && term_val(payload_cell) == 0 &&
                    !ERA_HAS_VISIBLE_VAR_CONSUMER(h)) {
                    continue;
                }
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
            // In root-only mode (coarse phase dumps), the reach walk
            // populates top_live from a privileged root.  Without a
            // root — or in explicit whole-heap mode (root_only=0) — we
            // render EVERY live TAG_TOP we encounter; the net's visible
            // structure comes from the heap state itself, not from a
            // root-anchored walk.
            if (heap_dot_root_only) {
                if (top_live && (val >= _dot_live_cap || !top_live[val])) continue;
            }
            // Dead-cell skip: IC-consumed cells are marked by the rule
            // that consumed them.  Two conventions used here:
            //   (a) All slots ERA(0) — pure ERA wipe.
            //   (b) Slot 0 is SUB(X) — redirect to the cell that took
            //       this cell's structural role (e.g. KERNEL-CTR slide:
            //       outer empty KERNEL's slot 0 = SUB(CTR)).  Anything
            //       still referencing this cell dereferences through SUB
            //       to the live replacement.
            // In either case, we skip rendering the dead cell.
            if (val + 0 < ctx->heap_pos) {
                u32 ar = thvm_uop_storage_arity(ext);
                int all_era = 1;
                for (u32 _i = 0; _i < ar; _i++) {
                    if (val + _i >= ctx->heap_pos) { all_era = 0; break; }
                    Term _s = heap_read(ctx, val + _i);
                    if (!(term_tag(_s) == TAG_ERA && term_val(_s) == 0)) {
                        all_era = 0; break;
                    }
                }
                if (ar > 0 && all_era) continue;
                Term s0 = heap_read(ctx, val + 0);
                if (term_is_sub(s0)) continue;
            }
            // Commuted GRAD cell (slot 0 has SUB-bit) is "dead"; its edges
            // are dereferenced through dot_dereference_commuted_grad below.
            // Don't emit a node for it — spec's "single GRAD" view.
            if ((ext == UOP_GRAD || ext == UOP_GRAD_FWD || ext == UOP_GRAD_PIN)
                && val + 0 < ctx->heap_pos
                && term_is_sub(heap_read(ctx, val + 0))) {
                continue;
            }
            // (Orphan filtering handled by the reach walk via
            // monolithic-kernel-stop + no-ERA-seeding — see earlier.
            // No render-time filtering: if top_live[val] is set, we
            // trust the walk and emit.)
            if (NODE_SEEN(val)) continue;
            NODE_MARK(val);
            char label[128]; const char *color = "#f0f0f0"; const char *nshape = "box";
            const char *opn = (ext < UOP_COUNT) ? uop_names[ext] : "?";
            // Heap span: TAG_TOP cells occupy val..val+storage_arity-1.
            // Show the full range so the reader can see which heap slots
            // belong to this node (e.g. MUL at @1 owns @1 and @2;
            // KERNEL at @9 owns @9, @10, @11).
            u32 _span_ar = thvm_uop_storage_arity(ext);
            char span[32];
            if (_span_ar > 1)
                snprintf(span, sizeof(span), "@%llu..%llu",
                         (unsigned long long)val,
                         (unsigned long long)(val + _span_ar - 1));
            else
                snprintf(span, sizeof(span), "@%llu", (unsigned long long)val);
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
                u32 kop = thvm_kernel_root_uop(ctx, t);
                KernelEntry ke_tmp;
                KernelEntry *dot_ke = NULL;
                u32 dot_kid = ~0u;
                int have_ke = dot_kernel_entry_for_term(ctx, t, &ke_tmp, &dot_ke, &dot_kid);
                char op_label[96] = "NULL";
                if (have_ke) dot_kernel_ops_summary(dot_ke, kop, op_label, sizeof(op_label));
                else if (kop < UOP_COUNT) snprintf(op_label, sizeof(op_label), "%s", uop_names[kop]);
                else if (kop == UOP_COUNT) snprintf(op_label, sizeof(op_label), "SEQ");
                // Empty KERNEL (pending-fusion marker): 3-slot
                // [payload, ERA, NUM(UOP_COUNT)].  Render with FUSE-style
                // invhouse shape + KERNEL green color — "same shape as
                // FUSE, but green" per spec.  The node itself IS a kernel,
                // just one without a populated op yet.
                int empty_kernel = (kop == UOP_COUNT && val + 1 < ctx->heap_pos &&
                                    term_tag(heap_read(ctx, val + 1)) == TAG_ERA);
                if (empty_kernel) {
                    snprintf(label, sizeof(label), "KERNEL\\n(empty)\\n%s", span);
                    color = "#ccffcc";
                    nshape = "invhouse";
                } else if (kop == UOP_COUNT) {
                    if (has_shape)
                        snprintf(label,sizeof(label),"SEQ\\n[%s]\\n%s", sh, span);
                    else
                        snprintf(label,sizeof(label),"SEQ\\n%s", span);
                } else {
                    u32 label_kid = (dot_kid != ~0u) ? dot_kid : dot_kernel_display_kid(ctx, t);
                    if (have_ke && label_kid != ~0u) {
                        if (has_shape)
                            snprintf(label,sizeof(label),"KERNEL\\n%s\\nkid=%u %s\\n[%s]\\n%s",
                                     op_label, label_kid, dot_kernel_backend(ctx, dot_ke), sh, span);
                        else
                            snprintf(label,sizeof(label),"KERNEL\\n%s\\nkid=%u %s\\n%s",
                                     op_label, label_kid, dot_kernel_backend(ctx, dot_ke), span);
                    } else if (label_kid != ~0u) {
                        if (has_shape)
                            snprintf(label,sizeof(label),"KERNEL\\n%s\\nkid=%u\\n[%s]\\n%s",
                                     op_label, label_kid, sh, span);
                        else
                            snprintf(label,sizeof(label),"KERNEL\\n%s\\nkid=%u\\n%s",
                                     op_label, label_kid, span);
                    } else if (has_shape) {
                        snprintf(label,sizeof(label),"KERNEL\\n%s\\n[%s]\\n%s",
                                 op_label, sh, span);
                    } else {
                        snprintf(label,sizeof(label),"KERNEL\\n%s\\n%s", op_label, span);
                    }
                    color = "#ccffcc";
                }
            } else if (ext == UOP_EXEC) {
                Term kid_term = heap_read(ctx, val + 0);
                u32 kid = term_as_u32(kid_term);
                snprintf(label,sizeof(label),"EXEC\\n#%u\\n%s", kid, span);
                color = "#ccccff";  // light blue for exec triggers
            } else if (ext == UOP_GRAD || ext == UOP_GRAD_FWD || ext == UOP_GRAD_PIN) {
                // 2-slot cell [body, target]; PIN and BW share it.  After
                // commute, slot 1 gets overwritten with bwd_wrapper — we
                // look up the original target from the side-table
                // populated at GRAD construction.
                extern Term thvm_grad_target_get(u64 loc);
                Term tgt = thvm_grad_target_get(val);
                char tgt_desc[64] = "?";
                if (term_tag(tgt) == TAG_TEN)
                    snprintf(tgt_desc, sizeof(tgt_desc), "t%u", (u32)term_val(tgt));
                else if (term_tag(tgt) == TAG_DP0 || term_tag(tgt) == TAG_DP1)
                    snprintf(tgt_desc, sizeof(tgt_desc), "dp@%llu",
                             (unsigned long long)term_val(tgt));
                snprintf(label, sizeof(label), "GRAD\\nd/d(%s)\\n%s", tgt_desc, span);
                color = "#e8d0ff";
                nshape = "box";
            } else {
                if (ext == UOP_FUSE) {
                    snprintf(label, sizeof(label), "%s\\n%s", opn, span);
                    color = "#b3e6ff";
                    nshape = "invhouse";
                } else {
                // Pure combinators (IFZ, DETACH, etc.) don't carry tensor shapes
                    int is_combinator = (ext == UOP_IFZ || ext == UOP_DETACH ||
                                        ext == UOP_LOG_PRINT || ext == UOP_FUSE);
                    if (is_combinator || !has_shape)
                        snprintf(label, sizeof(label), "%s\\n%s", opn, span);
                    else
                        snprintf(label, sizeof(label), "%s\\n[%s]\\n%s", opn, sh, span);
                }
                if (ext == UOP_ASSIGN) color = "#ffd700";
                else if (is_elementwise(ext)) color = "#cce5ff";
                else if (ext == UOP_SUM || ext == UOP_RMAX) color = "#ffcccc";
                else if (is_view_op(ext)) color = "#fff3cd";
                else if (ext == UOP_MM) color = "#ffccff";
            }
            fprintf(f, "  n%llu [label=\"%s\", shape=%s, fillcolor=\"%s\"%s%s];\n",
                    val, label, nshape, color, NODE_HL_ATTRS(val), ROOT_NODE_ATTRS_TERM(t));

            // Child edges: GRAD has 2 heap ports (y + gy). target pattern is metadata.
            KernelEntry ke_tmp;
            KernelEntry *dot_ke = NULL;
            u32 dot_kid = ~0u;
            int monolithic_kernel = (ext == UOP_KERNEL && thvm_kernel_is_monolithic(ctx, t));
            int have_ke = (ext == UOP_KERNEL)
                        ? dot_kernel_entry_for_term(ctx, t, &ke_tmp, &dot_ke, &dot_kid)
                        : 0;
            u32 arity = 2;
            // GRAD / GRAD_FWD: only render the y operand (slot 0).  The
            // target (slot 1) is metadata, already shown in the node label
            // as d/d(tN).  The GRAD's own outgoing edge carries fw/bw
            // semantics and is drawn by EMIT_FREE_OUT below.
            if (ext == UOP_GRAD || ext == UOP_GRAD_FWD || ext == UOP_GRAD_PIN) {
                // 2-slot cell.  Before commute: slot 0 = body (y), slot 1 = target.
                //   Render only slot 0 as incoming y edge.
                // After commute: slot 0 = SUB(body_with_reparent), slot 1 = bwd_wrapper.
                //   v_pass edge (GRAD → slot-0-SUB-target) + ∂v edge (GRAD → slot-1) rendered
                //   outside the arity loop.  Arity stays 1 so the loop renders slot 0.
                arity = 1;
            }
            else if (ext == UOP_WHERE || ext == UOP_IFZ) arity = 3;
            else if (ext == UOP_KERNEL) {
                // Empty kernels have a single meaningful slot (payload);
                // monolithic kernels show inputs via leaves; growing
                // kernels have left + right input slots.
                int _empty = (thvm_kernel_root_uop(ctx, t) == UOP_COUNT &&
                              val + 1 < ctx->heap_pos &&
                              term_tag(heap_read(ctx, val + 1)) == TAG_ERA);
                arity = _empty ? 1 : (monolithic_kernel ? 0 : 2);
            }
            else if (ext == UOP_EXEC) arity = 2;  // kid + deps (flags hidden)
            else if (ext == UOP_FUSE) arity = 1;
            else if (ext == UOP_LOG_PRINT) arity = 1;
            else if (ext == UOP_DETACH) arity = 1;
            else if (!is_binary(ext) && is_elementwise(ext)) arity = 1;
            for (u32 ai = 0; ai < arity; ai++) {
                Term child_raw = heap_read(ctx, val + ai);
                Term child = dot_deref_commuted_grad(ctx, child_raw);
                u64 cpos = val + ai;
                u8 ctag = term_tag(child); u64 cval = term_val(child);
                int edge_hl = heap_dot_hl_on && cpos == heap_dot_hl_slot;
                if (edge_hl) heap_dot_hl_hit = 1;
                const char *elbl = "";
                if (ext == UOP_ASSIGN) elbl = ai==0 ? "tgt" : "src";
                else if (ext == UOP_IFZ) elbl = ai==0 ? "cond" : (ai==1 ? "then" : "else");
                else if (ext == UOP_KERNEL) {
                    u32 kop = thvm_kernel_root_uop(ctx, t);
                    if (ai == 1 && term_tag(child) == TAG_ERA &&
                        !is_binary(kop) && is_elementwise(kop))
                        continue;
                    // Empty KERNEL (pending fusion): slot 0 holds the
                    // payload waiting to be kernelised.  Label "in".
                    int _empty_fuse = (kop == UOP_COUNT &&
                                       val + 1 < ctx->heap_pos &&
                                       term_tag(heap_read(ctx, val + 1)) == TAG_ERA);
                    if (_empty_fuse) elbl = "in";
                    else if (kop == UOP_COUNT) elbl = ai==0 ? "eff" : "next";
                    else if (kop >= UOP_RESHAPE && kop <= UOP_PAD) elbl = ai==0 ? "in" : "shape";
                    else if (kop == UOP_SUM || kop == UOP_RMAX) elbl = ai==0 ? "in" : "axes";
                    else if (is_binary(kop)) elbl = ai==0 ? "a" : "b";
                    else elbl = ai==0 ? "in" : "";
                }
                else if (ext == UOP_FUSE) {
                    // Semantic port label based on what's wrapped: PIN
                    // tag view → "v_pass", BW tag view → "∂v", else "in".
                    // Display-only; FUSE itself is still a single uop.
                    u8 rt = term_tag(child_raw);
                    if (rt == TAG_TOP) {
                        u32 re = term_ext(child_raw);
                        if (re == UOP_GRAD_PIN) elbl = "v_pass";
                        else if (re == UOP_GRAD || re == UOP_GRAD_FWD) elbl = "\u2202v";
                        else elbl = "in";
                    } else elbl = "in";
                }
                else if (ext >= UOP_RESHAPE && ext <= UOP_PAD) {
                    // EXPAND wrapping the Leibniz ADD renders as ∂v
                    // to match spec's backward-chain labelling.  Check
                    // dereferenced child (post-commute redirect).
                    if (ai == 0 && ext == UOP_EXPAND) {
                        u32 ce = (ctag == TAG_TOP) ? term_ext(child) : UOP_COUNT;
                        if (ce == UOP_ADD) elbl = "\u2202v";
                        else elbl = "in";
                    } else {
                        elbl = ai==0 ? "in" : "shape";
                    }
                }
                else if (ext == UOP_SUM || ext == UOP_RMAX) elbl = ai==0 ? "in" : "axes";
                else if (ext == UOP_GRAD || ext == UOP_GRAD_FWD || ext == UOP_GRAD_PIN) elbl = ai==0 ? "y" : "tgt";
                else if (ext == UOP_DETACH) elbl = "in";
                else if (is_binary(ext)) {
                    // Spec-match: chain-MUL's cotangent input gets ∂a/∂b.
                    // Detected via chain-MUL side table (set by VJP_BINARY
                    // when the chain MUL is allocated).  Works even after
                    // TEN-match replaces the sub-GRAD with a concrete
                    // ones tensor in the same slot.
                    extern int thvm_chain_mul_partial_slot(u64 loc);
                    int partial = (ext == UOP_MUL) ? thvm_chain_mul_partial_slot(val) : -1;
                    if (partial >= 0 && (u32)partial == ai) {
                        elbl = ai==0 ? "\u2202a" : "\u2202b";
                    } else {
                        elbl = ai==0 ? "a" : "b";
                    }
                }

                // GRAD-output override: when the child_raw is a commuted
                // GRAD tag view (GRAD_PIN = v_pass output; GRAD/GRAD_FWD =
                // \u2202v output), label the edge by GRAD's output port rather
                // than the consumer's input port.  Matches spec's
                // `GRAD \u2192 SUM [v_pass]` and `GRAD \u2192 EXPAND [\u2202v]`
                // labelling in step_001+.  Skip inside the GRAD cell's own
                // arity loop (y/tgt labels), inside FUSE (already handled
                // above with its own tag-view check), and on chain-MUL
                // partial slots (those keep their \u2202a/\u2202b labels).
                if (ext != UOP_GRAD && ext != UOP_GRAD_FWD &&
                    ext != UOP_GRAD_PIN && ext != UOP_FUSE &&
                    term_tag(child_raw) == TAG_TOP &&
                    term_tag(child) == TAG_TOP) {
                    // Only label as GRAD-output if deref still lands on a
                    // GRAD cell \u2014 i.e., sub-GRAD hasn't yet been resolved
                    // to a leaf by TEN-match.  Once v_pass aliases to
                    // t1/ones directly, fall back to the parent's port
                    // label (a/b/in) matching spec step_003+.
                    extern int thvm_chain_mul_partial_slot(u64 loc);
                    int _partial_slot = (ext == UOP_MUL)
                        ? thvm_chain_mul_partial_slot(val) : -1;
                    int _is_chain_partial = (_partial_slot >= 0 &&
                                             (u32)_partial_slot == ai);
                    if (!_is_chain_partial) {
                        u32 cre = term_ext(child_raw);
                        if (cre == UOP_GRAD_PIN) elbl = "v_pass";
                        else if (cre == UOP_GRAD || cre == UOP_GRAD_FWD) elbl = "\u2202v";
                    }
                }

                if (ctag == TAG_DP0 || ctag == TAG_DP1) {
                    u64 dl = cval;
                    DP_SLOT_MARK(cpos);
                    EMIT_DUP_CHAIN(dl);
                    char dp_lbl[64];
                    fprintf(f, "  dup%llu -> n%llu [label=\"%s\"];\n", dl, val,
                            dot_dp_port_label(dp_lbl, sizeof(dp_lbl), elbl, ctag, cpos));
                } else if (ctag == TAG_TEN) {
                    EMIT_TEN((u32)cval);
                    char slot_attrs[96];
                    dot_tensor_edge_slot_attrs(cpos, slot_attrs, sizeof(slot_attrs));
                    if (edge_hl) fprintf(f, "  t%u -> n%llu [label=\"%s\"%s,color=\"#cc0000\",penwidth=2.0];\n", (u32)cval, val, elbl, slot_attrs);
                    else         fprintf(f, "  t%u -> n%llu [label=\"%s\"%s];\n", (u32)cval, val, elbl, slot_attrs);
                } else if (ctag == TAG_CTR) {
                    EMIT_CTR_NODE(cval);
                    if (edge_hl) fprintf(f, "  ctr%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", cval, val, elbl);
                    else         fprintf(f, "  ctr%llu -> n%llu [label=\"%s\"];\n", cval, val, elbl);
                } else if (ctag == TAG_ANY) {
                    if (ext == UOP_GRAD) continue;
                    EMIT_ANY_NODE(cval);
                    if (edge_hl) fprintf(f, "  any%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", cval, val, elbl);
                    else         fprintf(f, "  any%llu -> n%llu [label=\"%s\"];\n", cval, val, elbl);
                } else if (ctag == TAG_ERA) {
                    u64 epos = cpos;
                    if (cval != 0) {
                        EMIT_ERA_NODE(epos, child);
                        if (edge_hl) fprintf(f, "  era%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", (unsigned long long)epos, val, elbl);
                        else         fprintf(f, "  era%llu -> n%llu [label=\"%s\"];\n", (unsigned long long)epos, val, elbl);
                    } else {
                        EMIT_FREE_PORT(val, ai);
                        if (edge_hl) fprintf(f, "  free%llu_%u -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)val, ai, (unsigned long long)val, elbl);
                        else         fprintf(f, "  free%llu_%u -> n%llu [label=\"%s\"];\n",
                                             (unsigned long long)val, ai, (unsigned long long)val, elbl);
                    }
                } else if (ctag == TAG_NUM) {
                    char num_label[32];
                    f32 fv; u32 bv=(u32)cval; memcpy(&fv,&bv,4);
                    snprintf(num_label, sizeof(num_label), "%.4g", (double)fv);
                    fprintf(f, "  num%llu_%u [label=\"%s\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n",
                            (unsigned long long)val, ai, num_label);
                    if (edge_hl) fprintf(f, "  num%llu_%u -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", val, ai, val, elbl);
                    else         fprintf(f, "  num%llu_%u -> n%llu [label=\"%s\"];\n", val, ai, val, elbl);
                } else if (ctag == TAG_REF) {
                    EMIT_REF_NODE(cpos, term_ext(child));
                    if (edge_hl) fprintf(f, "  ref%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                    else         fprintf(f, "  ref%llu -> n%llu [label=\"%s\"];\n",
                                         (unsigned long long)cpos, (unsigned long long)val, elbl);
                } else if (ctag == TAG_VAR) {
                    EMIT_VAR_OR_RESOLVED_TO_NODE(child, cval, val, elbl, edge_hl);
                } else if (ctag == TAG_TOP) {
                    if (edge_hl) fprintf(f, "  n%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n", cval, val, elbl);
                    else         fprintf(f, "  n%llu -> n%llu [label=\"%s\"];\n", cval, val, elbl);
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
            if (layers.semantic && ext == UOP_KERNEL && monolithic_kernel && have_ke) {
                // Emit one edge per op-arg pair so a leaf used as both
                // arg_a and arg_b (e.g. MUL(t1, t1)) shows BOTH edges.
                // Leaves are deduped by tid during kernel-build, so the
                // previous leaf-iteration approach only drew one edge.
                for (u32 oi = 0; oi < dot_ke->n_ops; oi++) {
                    u32 op_uop = dot_ke->ops[oi].uop;
                    for (u32 port = 0; port < 2; port++) {
                        u32 leaf_idx = (port == 0) ? dot_ke->ops[oi].arg_a
                                                   : dot_ke->ops[oi].arg_b;
                        // arg_b == arg_a for unary ops; skip the duplicate.
                        if (port == 1 && !is_binary(op_uop)) continue;
                        if (leaf_idx >= dot_ke->n_leaves) continue;
                        char edge_lbl[48];
                        snprintf(edge_lbl, sizeof(edge_lbl), "%s:%s",
                                 (op_uop < UOP_COUNT) ? uop_names[op_uop] : "?",
                                 dot_uop_port_name(op_uop, port));
                        if (dot_ke->leaf_kinds[leaf_idx] == KERNEL_LEAF_TENSOR) {
                            EMIT_TEN(dot_ke->leaf_ids[leaf_idx]);
                            fprintf(f, "  t%u -> n%llu [style=dashed,color=\"#009900\",fontcolor=\"#006600\",label=\"%s\"];\n",
                                    dot_ke->leaf_ids[leaf_idx],
                                    (unsigned long long)val, edge_lbl);
                        } else if (dot_ke->leaf_kinds[leaf_idx] == KERNEL_LEAF_NUM) {
                            fprintf(f, "  nummono%llu_%u_%u [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n",
                                    (unsigned long long)val, oi, port,
                                    (double)dot_ke->leaf_nums[leaf_idx]);
                            fprintf(f, "  nummono%llu_%u_%u -> n%llu [style=dashed,color=\"#009900\",fontcolor=\"#006600\",label=\"%s\"];\n",
                                    (unsigned long long)val, oi, port,
                                    (unsigned long long)val, edge_lbl);
                        }
                    }
                }
                // Reduce axes: SUM/RMAX kernels carry axes as metadata
                // on the sum_term cell.  Emit as an explicit "axes" edge
                // so downstream readers see which tensor controls the
                // reduction.
                if (dot_ke->has_reduce && term_tag(dot_ke->sum_term) == TAG_TOP) {
                    u64 sloc = term_val(dot_ke->sum_term);
                    if (sloc + 1 < ctx->heap_pos) {
                        Term axes = heap_read(ctx, sloc + 1);
                        if (term_tag(axes) == TAG_DP0 || term_tag(axes) == TAG_DP1)
                            axes = heap_read(ctx, term_val(axes));
                        if (term_tag(axes) == TAG_TEN) {
                            EMIT_TEN((u32)term_val(axes));
                            fprintf(f, "  t%u -> n%llu [style=dashed,color=\"#009900\",fontcolor=\"#006600\",label=\"%s:axes\"];\n",
                                    (u32)term_val(axes), (unsigned long long)val,
                                    dot_ke->has_reduce == UOP_SUM ? "SUM" : "RMAX");
                        }
                    }
                }
            }
            if (layers.semantic && ext == UOP_KERNEL && monolithic_kernel) {
                Term payload = thvm_kernel_monolithic_payload(ctx, t);
                u32 kop = thvm_kernel_root_uop(ctx, t);
                if (term_tag(payload) == TAG_TOP && is_view_op(kop)) {
                    u64 ploc = term_val(payload);
                    u32 parity = thvm_uop_visible_arity(kop);
                    for (u32 ai = 0; ai < parity; ai++) {
                        if (ploc + ai >= ctx->heap_pos) break;
                        Term pchild = heap_read(ctx, ploc + ai);
                        const char *elbl = dot_uop_port_name(kop, ai);
                        if (term_tag(pchild) == TAG_TEN) {
                            int already_leaf = 0;
                            if (have_ke) {
                                for (u32 li = 0; li < dot_ke->n_leaves; li++) {
                                    if (dot_ke->leaf_kinds[li] == KERNEL_LEAF_TENSOR &&
                                        dot_ke->leaf_ids[li] == (u32)term_val(pchild)) {
                                        already_leaf = 1;
                                        break;
                                    }
                                }
                            }
                            if (already_leaf) continue;
                            EMIT_TEN((u32)term_val(pchild));
                            fprintf(f, "  t%u -> n%llu [style=dashed,color=\"#009900\",fontcolor=\"#006600\",label=\"%s\"];\n",
                                    (u32)term_val(pchild), (unsigned long long)val, elbl);
                        } else if (term_tag(pchild) == TAG_NUM) {
                            int already_leaf = 0;
                            f32 pfv = term_ext(pchild) == NUM_U32 ? (f32)term_as_u32(pchild) : term_as_f32(pchild);
                            if (have_ke) {
                                for (u32 li = 0; li < dot_ke->n_leaves; li++) {
                                    if (dot_ke->leaf_kinds[li] != KERNEL_LEAF_NUM) continue;
                                    if (dot_ke->leaf_nums[li] == pfv) {
                                        already_leaf = 1;
                                        break;
                                    }
                                }
                            }
                            if (already_leaf) continue;
                            fprintf(f, "  numview%llu_%u [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n",
                                    (unsigned long long)val, ai, (double)pfv);
                            fprintf(f, "  numview%llu_%u -> n%llu [style=dashed,color=\"#009900\",fontcolor=\"#006600\",label=\"%s\"];\n",
                                    (unsigned long long)val, ai, (unsigned long long)val, elbl);
                        }
                    }
                }
            }
            {
                Term _self = term_new(TAG_TOP, ext, val);
                int has_parent = TERM_HAS_PARENT_REF(_self) || TERM_IS_DEF_ROOT(_self) || REF_TARGETS_LOC(val);
                if (ext == UOP_GRAD || ext == UOP_GRAD_FWD || ext == UOP_GRAD_PIN) {
                    // Post-commute: slot 0 SUB-bit, slot 1 = bwd_wrapper.
                    //   v_pass edge → stripped slot-0 target (reparented forward)
                    //   ∂v edge     → slot-1 bwd_wrapper
                    Term body = (val + 0 < ctx->heap_pos) ? heap_read(ctx, val + 0) : term_era();
                    if (term_is_sub(body)) {
                        Term stripped = term_strip_sub(body);
                        u8 st = term_tag(stripped);
                        u64 sv = term_val(stripped);
                        if (st == TAG_TOP && sv != 0 && sv < ctx->heap_pos && sv != val) {
                            fprintf(f, "  n%llu -> n%llu [label=\"v_pass\"];\n",
                                    (unsigned long long)val, (unsigned long long)sv);
                        } else if (st == TAG_TEN && sv > 0) {
                            EMIT_TEN((u32)sv);
                            fprintf(f, "  n%llu -> t%u [label=\"v_pass\"];\n",
                                    (unsigned long long)val, (u32)sv);
                        }
                        Term bwd = (val + 1 < ctx->heap_pos) ? heap_read(ctx, val + 1) : term_era();
                        u8 bt = term_tag(bwd);
                        u64 bv = term_val(bwd);
                        if (bt == TAG_TOP && bv != 0 && bv < ctx->heap_pos && bv != val) {
                            fprintf(f, "  n%llu -> n%llu [label=\"\u2202v\"];\n",
                                    (unsigned long long)val, (unsigned long long)bv);
                        } else if (bt == TAG_TEN && bv > 0) {
                            EMIT_TEN((u32)bv);
                            fprintf(f, "  n%llu -> t%u [label=\"\u2202v\"];\n",
                                    (unsigned long long)val, (u32)bv);
                        }
                    }
                } else if (!has_parent) {
                    EMIT_FREE_PORT(val, 1000u);
                    fprintf(f, "  n%llu -> free%llu_%u [label=\"out\"];\n",
                            (unsigned long long)val, (unsigned long long)val, 1000u);
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
            fprintf(f, "  n%llu [label=\"APP\\n#%u@%llu\", shape=invtriangle, fillcolor=\"#f3f3f3\"%s];\n",
                    (unsigned long long)val, ext, (unsigned long long)val, NODE_HL_ATTRS(val));
            for (u32 ai = 0; ai < 2; ai++) {
                Term child = heap_read(ctx, val + ai);
                if (IS_INLINE_UNARY_CTR(child))
                    child = INLINE_UNARY_CTR_CHILD(child);
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
                    char dp_lbl[64];
                    fprintf(f, "  dup%llu -> n%llu [label=\"%s\"%s];\n",
                            (unsigned long long)dl, (unsigned long long)val,
                            dot_dp_port_label(dp_lbl, sizeof(dp_lbl), elbl, ctag, cpos),
                            edge_hl ? " [color=\"#cc0000\",penwidth=2.0]" : "");
                } else if (ctag == TAG_TEN) {
                    EMIT_TEN((u32)cval);
                    char slot_attrs[96];
                    dot_tensor_edge_slot_attrs(cpos, slot_attrs, sizeof(slot_attrs));
                    if (edge_hl) fprintf(f, "  t%u -> n%llu [label=\"%s\"%s,color=\"#cc0000\",penwidth=2.0];\n",
                                         (u32)cval, (unsigned long long)val, elbl, slot_attrs);
                    else         fprintf(f, "  t%u -> n%llu [label=\"%s\"%s];\n",
                                         (u32)cval, (unsigned long long)val, elbl, slot_attrs);
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
                    if (cval != 0) {
                        EMIT_ERA_NODE(epos, child);
                        if (edge_hl) fprintf(f, "  era%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)epos, (unsigned long long)val, elbl);
                        else         fprintf(f, "  era%llu -> n%llu [label=\"%s\"];\n",
                                             (unsigned long long)epos, (unsigned long long)val, elbl);
                    } else {
                        EMIT_FREE_PORT(val, ai);
                        if (edge_hl) fprintf(f, "  free%llu_%u -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)val, ai, (unsigned long long)val, elbl);
                        else         fprintf(f, "  free%llu_%u -> n%llu [label=\"%s\"];\n",
                                             (unsigned long long)val, ai, (unsigned long long)val, elbl);
                    }
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
                    EMIT_VAR_OR_RESOLVED_TO_NODE(child, cval, val, elbl, edge_hl);
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
            {
                Term _self = term_new(TAG_APP, ext, val);
                if (!TERM_HAS_PARENT_REF(_self) && !TERM_IS_DEF_ROOT(_self) && !REF_TARGETS_LOC(val)) {
                    EMIT_FREE_PORT(val, 1000u);
                    fprintf(f, "  n%llu -> free%llu_%u [label=\"out\"];\n",
                            (unsigned long long)val, (unsigned long long)val, 1000u);
                }
            }
            nn++;
            continue;
        }

        // --- CTR ---
        if (tag == TAG_CTR) {
            if (ext == 1 && TERM_HAS_PARENT_APP_ARG(t)) continue;
            if (!LOC_LIVE(val)) continue;
            if (!NODE_SEEN(CTR_NODE_KEY(val))) {
                NODE_MARK(CTR_NODE_KEY(val));
                // CTR owns heap slots val..val+ext-1 (one per arm).
                char _ctr_span[32];
                if (ext > 1)
                    snprintf(_ctr_span, sizeof(_ctr_span), "@%llu..%llu",
                             (unsigned long long)val,
                             (unsigned long long)(val + ext - 1));
                else
                    snprintf(_ctr_span, sizeof(_ctr_span), "@%llu",
                             (unsigned long long)val);
                fprintf(f, "  ctr%llu [label=\"CTR\\nN=%u\\n%s\", shape=hexagon, fillcolor=\"#f3f3f3\", fontsize=9];\n",
                        (unsigned long long)val, ext, _ctr_span);
                // Output port: only emit a free-port sink circle when
                // the CTR has NO outgoing consumer on the heap.  If any
                // cell holds TAG_CTR with val=this_loc, the CTR's output
                // flows into that consumer via a normal edge — drawing
                // an extra free port on top would leave a dangling
                // orphan circle alongside the real edge.
                int ctr_has_consumer = 0;
                for (u64 _hh = 1; !ctr_has_consumer && _hh < _dot_live_cap; _hh++) {
                    Term _pt = ctx->heap[_hh];
                    if (term_tag(_pt) == TAG_CTR && term_val(_pt) == val) {
                        ctr_has_consumer = 1;
                    }
                }
                if (!ctr_has_consumer) {
                    fprintf(f, "  ctr_out%llu [label=\"\",shape=circle,width=0.16,height=0.16,fixedsize=true,fillcolor=\"#ffffff\",color=\"#888888\"];\n",
                            (unsigned long long)val);
                    fprintf(f, "  ctr%llu -> ctr_out%llu [label=\"out\"];\n",
                            (unsigned long long)val, (unsigned long long)val);
                }
            }
            if (val < ctx->heap_pos && (!ctr_children_emitted || !ctr_children_emitted[val])) {
                if (ctr_children_emitted) ctr_children_emitted[val] = 1;
                for (u32 gi = 0; gi < ext; gi++) {
                    Term cterm_raw = heap_read(ctx, val + gi);
                    // Deref commuted GRAD tag views: after a GRAD-X
                    // commute, the old outer GRAD cell carries SUB on
                    // slot 0 and the real bwd/fwd target on slot 1.
                    // Rendering the raw GRAD tag here would produce a
                    // gray-box ghost with no shape.  Follow the SUB
                    // redirect to the reparented target so the edge
                    // lands on the correct live node (EXPAND / SUM /
                    // the Leibniz ADD / etc.).
                    Term cterm = dot_deref_commuted_grad(ctx, cterm_raw);
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
                        char dp_lbl[64];
                        if (edge_hl) fprintf(f, "  dup%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)dl, (unsigned long long)val,
                                             dot_dp_port_label(dp_lbl, sizeof(dp_lbl), clbl, ctag, cpos));
                        else         fprintf(f, "  dup%llu -> ctr%llu [label=\"%s\"];\n",
                                             (unsigned long long)dl, (unsigned long long)val,
                                             dot_dp_port_label(dp_lbl, sizeof(dp_lbl), clbl, ctag, cpos));
                    } else if (ctag == TAG_NUM) {
                        f32 fv; u32 bv = (u32)cval; memcpy(&fv, &bv, 4);
                        fprintf(f, "  num_ctr%llu_%llu [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n",
                                (unsigned long long)val, (unsigned long long)cpos, (double)fv);
                        if (edge_hl) fprintf(f, "  num_ctr%llu_%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)val, (unsigned long long)cpos, (unsigned long long)val, clbl);
                        else         fprintf(f, "  num_ctr%llu_%llu -> ctr%llu [label=\"%s\"];\n",
                                             (unsigned long long)val, (unsigned long long)cpos, (unsigned long long)val, clbl);
                    } else if (ctag == TAG_ERA) {
                        if (cval != 0) {
                            EMIT_ERA_NODE(cpos, cterm);
                            if (edge_hl) fprintf(f, "  era%llu -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                                 (unsigned long long)cpos, (unsigned long long)val, clbl);
                            else         fprintf(f, "  era%llu -> ctr%llu [label=\"%s\"];\n",
                                                 (unsigned long long)cpos, (unsigned long long)val, clbl);
                        } else {
                            EMIT_FREE_PORT(val, gi);
                            if (edge_hl) fprintf(f, "  free%llu_%u -> ctr%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                                 (unsigned long long)val, gi, (unsigned long long)val, clbl);
                            else         fprintf(f, "  free%llu_%u -> ctr%llu [label=\"%s\"];\n",
                                                 (unsigned long long)val, gi, (unsigned long long)val, clbl);
                        }
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
                        EMIT_VAR_OR_RESOLVED_TO_TARGET(cterm, cval, "ctr", val, clbl, edge_hl, "");
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
            // #region agent log
            do {
                static u32 ref_root_only_dbg_count = 0;
                if (ref_root_only_dbg_count >= 8) break;
                if (!heap_dot_root_only || ext >= ctx->def_count) break;
                ref_root_only_dbg_count++;
                Term _d = ctx->defs[ext];
                char _dbg[256];
                snprintf(_dbg, sizeof(_dbg),
                         "{\"slot\":%llu,\"name\":%u,\"heap_dot_root_only\":%d,"
                         "\"def_tag\":%u,\"def_ext\":%u,\"def_val\":%llu}",
                         (unsigned long long)h, ext, heap_dot_root_only ? 1 : 0,
                         (u32)term_tag(_d), (u32)term_ext(_d), (unsigned long long)term_val(_d));
                thvm_agent_debug_log("pre-fix", "H1", "src/debug/dump.c:1567",
                                     "ref_def_edge_drawn", _dbg);
            } while (0);
            // #endregion
            // Draw dashed edge REF → definition root in step graphs too so
            // unfolding stays visually anchored to the visible book body.
            if (ext < ctx->def_count) {
                Term _d = ctx->defs[ext];
                u8 _dt = term_tag(_d);
                u64 _dl = term_val(_d);
                if (_dl > 0 && _dl < ctx->heap_pos) {
                    if (_dt == TAG_REF) {
                        EMIT_REF_NODE(_dl, term_ext(_d));
                        fprintf(f, "  ref%llu -> ref%llu [style=dashed, color=\"#999999\", label=\"def\"];\n",
                                (unsigned long long)h, (unsigned long long)_dl);
                    } else if (_dt == TAG_VAR) {
                        char _var_dst_id[64];
                        if (dot_term_node_id(ctx, _d, _var_dst_id, sizeof(_var_dst_id))) {
                            fprintf(f, "  ref%llu -> %s [style=dashed, color=\"#999999\", label=\"def\"];\n",
                                    (unsigned long long)h, _var_dst_id);
                        }
                    } else if (_dt == TAG_ANY) {
                        EMIT_ANY_NODE(_dl);
                        fprintf(f, "  ref%llu -> any%llu [style=dashed, color=\"#999999\", label=\"def\"];\n",
                                (unsigned long long)h, (unsigned long long)_dl);
                    } else if (dot_visible_heap_loc_tag(_dt)) {
                        EMIT_REF_DEF_ROOT(_d);
                        fprintf(f, "  ref%llu -> n%llu [style=dashed, color=\"#999999\", label=\"def\"];\n",
                                (unsigned long long)h, (unsigned long long)_dl);
                    }
                }
            }
            nn++;
            continue;
        }

        if (tag == TAG_VAR) {
            // Standalone VARs are only emitted if they have a live parent
            // edge (i.e., some rendered node references this heap slot).
            // Otherwise they float disconnected and add noise.
            // VARs inline in parent edges (APP, TOP, etc.) are emitted
            // by EMIT_VAR_NODE in those handlers, not here.
            continue;
        }

        if (dot_visible_heap_loc_tag(tag) && tag != TAG_TOP && tag != TAG_APP && tag != TAG_CTR) {
            if (!LOC_LIVE(val)) continue;
            if (NODE_SEEN(val)) continue;
            NODE_MARK(val);
            char label[96];
            if (tag == TAG_SUP || tag == TAG_USP) {
                snprintf(label, sizeof(label), "%s #%u@%llu", dot_heap_tag_name(tag), ext, (unsigned long long)val);
            } else if (tag == TAG_LAM || tag == TAG_BRI) {
                snprintf(label, sizeof(label), "%s\\n#%u@%llu", dot_heap_tag_name(tag), ext, (unsigned long long)val);
            } else if (tag == TAG_MAT || tag == TAG_REF) {
                snprintf(label, sizeof(label), "%s\\n#%u@%llu", dot_heap_tag_name(tag), ext, (unsigned long long)val);
            } else if (tag == TAG_ALO && val + 1 < ctx->heap_pos) {
                Term book_term = heap_read(ctx, val + 0);
                snprintf(label, sizeof(label), "ALO\\n@%llu", (unsigned long long)val);
                // #region agent log
                do {
                    static u32 alo_label_dbg_count = 0;
                    if (alo_label_dbg_count >= 12) break;
                    alo_label_dbg_count++;
                    Term sid_term = heap_read(ctx, val + 1);
                    u32 sid = term_tag(sid_term) == TAG_NUM ? term_as_u32(sid_term) : 0;
                    char _dbg[320];
                    snprintf(_dbg, sizeof(_dbg),
                             "{\"alo_loc\":%llu,\"book_tag\":%u,\"book_ext\":%u,\"book_val\":%llu,\"state_id\":%u}",
                             (unsigned long long)val,
                             (u32)term_tag(book_term), (u32)term_ext(book_term),
                             (unsigned long long)term_val(book_term), sid);
                    thvm_agent_debug_log("pre-fix", "H13", "src/debug/dump.c:1650",
                                         "alo_label_context", _dbg);
                } while (0);
                // #endregion
            } else {
                snprintf(label, sizeof(label), "%s\\n@%llu", dot_heap_tag_name(tag), (unsigned long long)val);
            }
            fprintf(f, "  n%llu [label=\"%s\", shape=%s, fillcolor=\"%s\"%s];\n",
                    (unsigned long long)val, label, dot_heap_node_shape(tag), dot_heap_node_color(tag),
                    ROOT_NODE_ATTRS_TERM(t));
            if (tag == TAG_ALO && val + 1 < ctx->heap_pos) {
                int book_hl = heap_dot_hl_on && (val + 0) == heap_dot_hl_slot;
                int env_hl = heap_dot_hl_on && (val + 1) == heap_dot_hl_slot;
                if (book_hl || env_hl) heap_dot_hl_hit = 1;
            }
            if (layers.semantic && tag == TAG_ALO && val + 1 < ctx->heap_pos) {
                Term book_term = heap_read(ctx, val + 0);
                Term dyn_book = dot_book_to_dynamic_term(ctx, book_term);
                char dst_id[64];
                if (dyn_book != 0 && dot_term_node_id(ctx, dyn_book, dst_id, sizeof(dst_id))) {
                    fprintf(f, "  n%llu -> %s [style=dashed,color=\"#7a7a7a\",fontcolor=\"#5a5a5a\",label=\"book\"];\n",
                            (unsigned long long)val, dst_id);
                    // #region agent log
                    do {
                        static u32 alo_book_edge_dbg_count = 0;
                        if (alo_book_edge_dbg_count >= 16) break;
                        alo_book_edge_dbg_count++;
                        char _dbg[320];
                        snprintf(_dbg, sizeof(_dbg),
                                 "{\"ghost\":0,\"alo_loc\":%llu,\"book_tag\":%u,\"book_ext\":%u,\"book_val\":%llu,"
                                 "\"target_tag\":%u,\"target_ext\":%u,\"target_val\":%llu}",
                                 (unsigned long long)val,
                                 (u32)term_tag(book_term), (u32)term_ext(book_term), (unsigned long long)term_val(book_term),
                                 (u32)term_tag(dyn_book), (u32)term_ext(dyn_book), (unsigned long long)term_val(dyn_book));
                        thvm_agent_debug_log("pre-fix", "H14", "src/debug/dump.c:1710",
                                             "alo_book_edge_rendered", _dbg);
                    } while (0);
                    // #endregion
                }
            }
            if (layers.semantic && tag == TAG_ALO && val + 1 < ctx->heap_pos) {
                Term alo_book_term = heap_read(ctx, val + 0);
                Term sid_term = heap_read(ctx, val + 1);
                if (term_tag(sid_term) == TAG_NUM) {
                    u32 sid = term_as_u32(sid_term);
                    for (u32 walk = sid; walk != 0 && walk < ctx->alo_state_count; walk = ctx->alo_states[walk].parent) {
                        AloState st = ctx->alo_states[walk];
                        if (st.kind != 1 || st.bind_dyn == 0 || st.bind_dyn >= ctx->heap_pos) continue;
                        char env_label[64];
                        const char *env_lbl = dot_alo_env_bind_label(ctx, alo_book_term, &st, env_label, sizeof(env_label));
                        if (!env_lbl[0]) env_lbl = "env";
                        Term bound = heap_read(ctx, st.bind_dyn);
                        if (term_is_sub(bound)) {
                            // #region agent log
                            do {
                                static u32 alo_binder_skip_dbg_count = 0;
                                if (alo_binder_skip_dbg_count >= 16) break;
                                alo_binder_skip_dbg_count++;
                                char _dbg[320];
                                snprintf(_dbg, sizeof(_dbg),
                                         "{\"alo_loc\":%llu,\"state_id\":%u,\"bind_book\":%llu,\"bind_dyn\":%llu,\"bound_tag\":%u,\"bound_ext\":%u,\"bound_val\":%llu}",
                                         (unsigned long long)val, sid,
                                         (unsigned long long)st.bind_book,
                                         (unsigned long long)st.bind_dyn,
                                         (u32)term_tag(bound), (u32)term_ext(bound),
                                         (unsigned long long)term_val(bound));
                                thvm_agent_debug_log("pre-fix", "H17", "src/debug/dump.c:1702",
                                                     "alo_binder_env_skipped", _dbg);
                            } while (0);
                            // #endregion
                            u64 bval = term_val(bound);
                            EMIT_VAR_OR_RESOLVED_TO_TARGET(bound, bval, "n", val, env_lbl, 0,
                                                           ",style=dotted,color=\"#7a7a7a\"");
                            // #region agent log
                            do {
                                static u32 alo_binder_render_dbg_count = 0;
                                if (alo_binder_render_dbg_count >= 16) break;
                                alo_binder_render_dbg_count++;
                                char _dbg[224];
                                snprintf(_dbg, sizeof(_dbg),
                                         "{\"alo_loc\":%llu,\"state_id\":%u,\"bind_book\":%llu,\"bind_dyn\":%llu,\"var_loc\":%llu}",
                                         (unsigned long long)val, sid,
                                         (unsigned long long)st.bind_book,
                                         (unsigned long long)st.bind_dyn,
                                         (unsigned long long)bval);
                                thvm_agent_debug_log("post-fix", "H19", "src/debug/dump.c:1715",
                                                     "alo_binder_env_rendered", _dbg);
                            } while (0);
                            // #endregion
                            continue;
                        }
                        u8 btag = term_tag(bound);
                        u64 bval = term_val(bound);
                        if (btag == TAG_DP0 || btag == TAG_DP1) {
                            DP_SLOT_MARK(st.bind_dyn);
                            EMIT_DUP_CHAIN(bval);
                            char env_dup_label[96];
                            dot_dp_port_label(env_dup_label, sizeof(env_dup_label), env_lbl, btag, st.bind_dyn);
                            fprintf(f, "  dup%llu -> n%llu [label=\"%s\",style=dotted,color=\"#7a7a7a\"];\n",
                                    (unsigned long long)bval, (unsigned long long)val, env_dup_label);
                        } else if (btag == TAG_TEN) {
                            EMIT_TEN((u32)bval);
                            char slot_attrs[96];
                            dot_tensor_edge_slot_attrs(st.bind_dyn, slot_attrs, sizeof(slot_attrs));
                            fprintf(f, "  t%u -> n%llu [label=\"%s\"%s,style=dotted,color=\"#7a7a7a\"];\n",
                                    (u32)bval, (unsigned long long)val, env_lbl, slot_attrs);
                        } else if (btag == TAG_VAR) {
                            EMIT_VAR_OR_RESOLVED_TO_TARGET(bound, bval, "n", val, env_lbl, 0,
                                                           ",style=dotted,color=\"#7a7a7a\"");
                        } else if (btag == TAG_REF) {
                            EMIT_REF_NODE(st.bind_dyn, term_ext(bound));
                            fprintf(f, "  ref%llu -> n%llu [label=\"%s\",style=dotted,color=\"#7a7a7a\"];\n",
                                    (unsigned long long)st.bind_dyn, (unsigned long long)val, env_lbl);
                        } else if (btag == TAG_NUM) {
                            f32 fv; u32 bv = (u32)bval; memcpy(&fv, &bv, 4);
                            fprintf(f, "  num_alo_env%llu_%u [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n",
                                    (unsigned long long)val, walk, (double)fv);
                            fprintf(f, "  num_alo_env%llu_%u -> n%llu [label=\"%s\",style=dotted,color=\"#7a7a7a\"];\n",
                                    (unsigned long long)val, walk, (unsigned long long)val, env_lbl);
                        } else if (btag == TAG_ANY) {
                            EMIT_ANY_NODE(st.bind_dyn);
                            fprintf(f, "  any%llu -> n%llu [label=\"%s\",style=dotted,color=\"#7a7a7a\"];\n",
                                    (unsigned long long)st.bind_dyn, (unsigned long long)val, env_lbl);
                        } else if (btag == TAG_CTR) {
                            EMIT_CTR_NODE(bval);
                            fprintf(f, "  ctr%llu -> n%llu [label=\"%s\",style=dotted,color=\"#7a7a7a\"];\n",
                                    (unsigned long long)bval, (unsigned long long)val, env_lbl);
                        } else if (btag == TAG_ALO) {
                            if (!NODE_SEEN(bval)) {
                                NODE_MARK(bval);
                                fprintf(f, "  n%llu [label=\"ALO\\n@%llu\", shape=%s, fillcolor=\"%s\"];\n",
                                        (unsigned long long)bval, (unsigned long long)bval,
                                        dot_heap_node_shape(btag), dot_heap_node_color(btag));
                            }
                            fprintf(f, "  n%llu -> n%llu [label=\"%s\",style=dotted,color=\"#7a7a7a\"];\n",
                                    (unsigned long long)bval, (unsigned long long)val, env_lbl);
                        } else if (btag == TAG_TOP || dot_visible_heap_loc_tag(btag)) {
                            if (!NODE_SEEN(bval)) {
                                NODE_MARK(bval);
                                char blabel[96];
                                dot_raw_slot_term_label(ctx, bound, blabel, sizeof(blabel));
                                fprintf(f, "  n%llu [label=\"%s\", shape=%s, fillcolor=\"%s\"];\n",
                                        (unsigned long long)bval, blabel,
                                        dot_heap_node_shape(btag), dot_heap_node_color(btag));
                            }
                            fprintf(f, "  n%llu -> n%llu [label=\"%s\",style=dotted,color=\"#7a7a7a\"];\n",
                                    (unsigned long long)bval, (unsigned long long)val, env_lbl);
                        }
                    }
                }
            }
            u32 ar = dot_term_arity(ctx, t);
            for (u32 ai = 0; ai < ar; ai++) {
                Term child = heap_read(ctx, val + ai);
                u8 ctag = term_tag(child);
                u64 cval = term_val(child);
                u64 cpos = val + ai;
                int edge_hl = heap_dot_hl_on && cpos == heap_dot_hl_slot;
                if (edge_hl) heap_dot_hl_hit = 1;
                const char *elbl = dot_heap_port_name(tag, ai);
                char elbl_buf[32];
                snprintf(elbl_buf, sizeof(elbl_buf), "%s", elbl);
                elbl = elbl_buf;
                if (ctag == TAG_DP0 || ctag == TAG_DP1) {
                    u64 dl = cval;
                    DP_SLOT_MARK(cpos);
                    EMIT_DUP_CHAIN(dl);
                    char dp_lbl[64];
                    fprintf(f, "  dup%llu -> n%llu [label=\"%s\"%s];\n",
                            (unsigned long long)dl, (unsigned long long)val,
                            dot_dp_port_label(dp_lbl, sizeof(dp_lbl), elbl, ctag, cpos),
                            edge_hl ? " [color=\"#cc0000\",penwidth=2.0]" : "");
                } else if (ctag == TAG_TEN) {
                    EMIT_TEN((u32)cval);
                    char slot_attrs[96];
                    dot_tensor_edge_slot_attrs(cpos, slot_attrs, sizeof(slot_attrs));
                    if (edge_hl) fprintf(f, "  t%u -> n%llu [label=\"%s\"%s,color=\"#cc0000\",penwidth=2.0];\n",
                                         (u32)cval, (unsigned long long)val, elbl, slot_attrs);
                    else         fprintf(f, "  t%u -> n%llu [label=\"%s\"%s];\n",
                                         (u32)cval, (unsigned long long)val, elbl, slot_attrs);
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
                    if (cval != 0) {
                        EMIT_ERA_NODE(epos, child);
                        if (edge_hl) fprintf(f, "  era%llu -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)epos, (unsigned long long)val, elbl);
                        else         fprintf(f, "  era%llu -> n%llu [label=\"%s\"];\n",
                                             (unsigned long long)epos, (unsigned long long)val, elbl);
                    } else {
                        EMIT_FREE_PORT(val, ai);
                        if (edge_hl) fprintf(f, "  free%llu_%u -> n%llu [label=\"%s\",color=\"#cc0000\",penwidth=2.0];\n",
                                             (unsigned long long)val, ai, (unsigned long long)val, elbl);
                        else         fprintf(f, "  free%llu_%u -> n%llu [label=\"%s\"];\n",
                                             (unsigned long long)val, ai, (unsigned long long)val, elbl);
                    }
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
                    if (!((tag == TAG_LAM || tag == TAG_BRI) && ai == 0)) {
                        EMIT_VAR_OR_RESOLVED_TO_NODE(child, cval, val, elbl, edge_hl);
                    }
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
            {
                Term _self = term_new(tag, ext, val);
                if (!TERM_HAS_PARENT_REF(_self) && !TERM_IS_DEF_ROOT(_self) && !REF_TARGETS_LOC(val)) {
                    EMIT_FREE_PORT(val, 1000u);
                    fprintf(f, "  n%llu -> free%llu_%u [label=\"out\"];\n",
                            (unsigned long long)val, (unsigned long long)val, 1000u);
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
                u32 kop2 = thvm_kernel_root_uop(ctx, term_new(TAG_TOP, UOP_KERNEL, (_loc))); \
                char op_label2[32] = "NULL"; \
                if (kop2 < UOP_COUNT) snprintf(op_label2, sizeof(op_label2), "%s", uop_names[kop2]); \
                else if (kop2 == UOP_COUNT) snprintf(op_label2, sizeof(op_label2), "SEQ"); \
                if (kop2 == UOP_COUNT) { \
                    snprintf(label, sizeof(label), "SEQ\\n[%s]", sh); \
                } else { \
                    KernelEntry _ke2_tmp; \
                    KernelEntry *_dke2 = NULL; \
                    u32 kid2 = ~0u; \
                    int _have_ke2 = dot_kernel_entry_for_term(ctx, term_new(TAG_TOP, UOP_KERNEL, (_loc)), &_ke2_tmp, &_dke2, &kid2); \
                    if (kid2 == ~0u) kid2 = dot_kernel_display_kid(ctx, term_new(TAG_TOP, UOP_KERNEL, (_loc))); \
                    if (_have_ke2 && kid2 != ~0u) { \
                        snprintf(label, sizeof(label), "KERNEL\\n%s\\nkid=%u %s\\n[%s]", \
                                 op_label2, kid2, dot_kernel_backend(ctx, _dke2), sh); \
                    } else if (kid2 != ~0u) { \
                        snprintf(label, sizeof(label), "KERNEL\\n%s\\nkid=%u\\n[%s]", \
                                 op_label2, kid2, sh); \
                    } else { \
                        snprintf(label, sizeof(label), "KERNEL\\n%s\\n[%s]", op_label2, sh); \
                    } \
                    color = "#ccffcc"; \
                } \
            } else if ((_ext) == UOP_GRAD) { \
                snprintf(label, sizeof(label), "GRAD"); \
                color = "#e8d0ff"; \
            } else if ((_ext) == UOP_GRAD_PIN) { \
                snprintf(label, sizeof(label), "PIN"); \
                color = "#b3e6ff"; \
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
            u32 kid = 0;
            if (!thvm_kernel_lookup_kid(loc, &kid) || kid >= sched_kernel_count) continue;
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

    // Show producer kernels for visible tensors even after the kernel term has
    // already dispatched and vanished from the reachable heap.
    if (!heap_dot_include_sched_kernels && !heap_dot_root_only) {
        extern KernelEntry sched_kernels[];
        extern u32 sched_kernel_count;
        u8 kernel_leaf_drawn[SCHED_MAX_KERNELS];
        memset(kernel_leaf_drawn, 0, sizeof(kernel_leaf_drawn));
        for (u32 tid = 0; tid < ctx->tensor_count && tid < 256; tid++) {
            if (!ten_emitted[tid]) continue;
            TensorMeta *m = &ctx->tensors[tid];
            if (m->creator_op != UOP_KERNEL || m->fusing_loc == 0 || m->fusing_loc + 2 >= ctx->heap_pos)
                continue;
            u64 loc = m->fusing_loc;
            u32 kid = 0;
            if (!thvm_kernel_lookup_kid(loc, &kid) || kid >= sched_kernel_count) continue;
            KernelEntry *ke = &sched_kernels[kid];
            EMIT_TOP_NODE_LABEL(loc, UOP_KERNEL);
            if (!kernel_leaf_drawn[kid]) {
                kernel_leaf_drawn[kid] = 1;
                for (u32 li = 0; li < ke->n_leaves; li++) {
                    char leaf_lbl[48];
                    dot_kernel_leaf_label(ctx, ke, li, leaf_lbl, sizeof(leaf_lbl));
                    if (ke->leaf_kinds[li] == KERNEL_LEAF_TENSOR) {
                        EMIT_TEN(ke->leaf_ids[li]);
                        fprintf(f, "  t%u -> n%llu [style=dashed,color=\"#009900\",fontcolor=\"#006600\",label=\"%s\"];\n",
                                ke->leaf_ids[li], (unsigned long long)loc, leaf_lbl);
                    } else if (ke->leaf_kinds[li] == KERNEL_LEAF_NUM) {
                        f32 fv = ke->leaf_nums[li];
                        fprintf(f, "  numkg%llu_%u [label=\"%.4g\",shape=triangle,fillcolor=\"#fff2cc\",fontsize=8];\n",
                                (unsigned long long)loc, li, (double)fv);
                        fprintf(f, "  numkg%llu_%u -> n%llu [style=dashed,color=\"#009900\",fontcolor=\"#006600\",label=\"%s\"];\n",
                                (unsigned long long)loc, li, (unsigned long long)loc, leaf_lbl);
                    }
                }
            }
            char out_lbl[48];
            dot_kernel_output_label(ke, out_lbl, sizeof(out_lbl));
            fprintf(f, "  n%llu -> t%u [style=dashed,color=\"#009900\",fontcolor=\"#006600\",label=\"%s\"];\n",
                    (unsigned long long)loc, tid, out_lbl);
        }
    }

    #undef EMIT_FUSING_LEAF_CHILD
    #undef EMIT_TOP_NODE_LABEL

    if (layers.step && heap_dot_root_only && ctx->step_alo_subst_count > 0) {
        for (u32 si = 0; si < ctx->step_alo_subst_count; si++) {
            StepGraphAloSubst subst = ctx->step_alo_substs[si];
            if (subst.old_alo_loc == 0 || subst.out == 0) continue;
            u8 out_tag = term_tag(subst.out);
            u64 out_val = term_val(subst.out);
            int emitted = dot_visible_heap_loc_tag(out_tag) &&
                          out_val > 0 && out_val < ctx->heap_pos &&
                          NODE_SEEN(out_val);
            if (!emitted) continue;
            fprintf(f, "  subalo%llu [label=\"ALO\\n@%llu\", shape=box3d, style=\"filled,dashed\", fillcolor=\"#f4fbff\", color=\"#7a7a7a\", fontcolor=\"#5a5a5a\"];\n",
                    (unsigned long long)subst.old_alo_loc, (unsigned long long)subst.old_alo_loc);
            Term dyn_book = dot_book_to_dynamic_term(ctx, subst.book_term);
            char book_dst_id[64];
            if (dyn_book != 0 && dot_term_node_id(ctx, dyn_book, book_dst_id, sizeof(book_dst_id))) {
                fprintf(f, "  subalo%llu -> %s [style=dashed,color=\"#7a7a7a\",fontcolor=\"#5a5a5a\",label=\"book\"];\n",
                        (unsigned long long)subst.old_alo_loc, book_dst_id);
                // #region agent log
                do {
                    static u32 subalo_book_edge_dbg_count = 0;
                    if (subalo_book_edge_dbg_count >= 16) break;
                    subalo_book_edge_dbg_count++;
                    char _dbg[320];
                    snprintf(_dbg, sizeof(_dbg),
                             "{\"ghost\":1,\"alo_loc\":%llu,\"book_tag\":%u,\"book_ext\":%u,\"book_val\":%llu,"
                             "\"target_tag\":%u,\"target_ext\":%u,\"target_val\":%llu}",
                             (unsigned long long)subst.old_alo_loc,
                             (u32)term_tag(subst.book_term), (u32)term_ext(subst.book_term),
                             (unsigned long long)term_val(subst.book_term),
                             (u32)term_tag(dyn_book), (u32)term_ext(dyn_book),
                             (unsigned long long)term_val(dyn_book));
                    thvm_agent_debug_log("pre-fix", "H14", "src/debug/dump.c:2088",
                                         "alo_book_edge_rendered", _dbg);
                } while (0);
                // #endregion
            }
            fprintf(f, "  subalo%llu -> n%llu [style=dashed,color=\"#7a7a7a\",fontcolor=\"#5a5a5a\",label=\"subst\"];\n",
                    (unsigned long long)subst.old_alo_loc, (unsigned long long)out_val);
            // #region agent log
            do {
                static u32 alo_overlay_dbg_count = 0;
                if (alo_overlay_dbg_count >= 12) break;
                alo_overlay_dbg_count++;
                char _dbg[224];
                snprintf(_dbg, sizeof(_dbg),
                         "{\"old_alo_loc\":%llu,\"book_tag\":%u,\"book_ext\":%u,\"book_val\":%llu,\"out_tag\":%u,\"out_val\":%llu,\"emitted\":1}",
                         (unsigned long long)subst.old_alo_loc,
                         (u32)term_tag(subst.book_term), (u32)term_ext(subst.book_term),
                         (unsigned long long)term_val(subst.book_term),
                         out_tag,
                         (unsigned long long)out_val);
                thvm_agent_debug_log("pre-fix", "H12", "src/debug/dump.c:2006",
                                     "alo_subst_overlay_rendered", _dbg);
            } while (0);
            // #endregion
        }
    }

    if (term_tag(root) == TAG_TEN) {
        u32 root_tid = (u32)term_val(root);
        EMIT_TEN(root_tid);
        fprintf(f, "  rootout_t%u [label=\"\",shape=circle,width=0.14,height=0.14,fixedsize=true,fillcolor=\"#ffffff\",color=\"#888888\",fontsize=1];\n",
                root_tid);
        fprintf(f, "  t%u -> rootout_t%u [label=\"out\"];\n", root_tid, root_tid);
    }

    // Step-graph GRAD-cursor overlay: if both the GRAD slot and the
    // current forward TOP are live & rendered as n<val> nodes, draw a
    // dashed edge from the GRAD to the cursor TOP.  This is the
    // "slide" visualization — without mutating the heap — that shows
    // which forward node the VJP is currently differentiating.
    if (heap_dot_grad_cursor_grad != 0 && heap_dot_grad_cursor_loc != 0 &&
        heap_dot_grad_cursor_grad < ctx->heap_pos &&
        heap_dot_grad_cursor_loc < ctx->heap_pos) {
        fprintf(f,
            "  n%llu -> n%llu [label=\"cursor\",style=dashed,color=\"#cc0000\","
            "penwidth=1.5,fontcolor=\"#cc0000\",constraint=false];\n",
            (unsigned long long)heap_dot_grad_cursor_loc,
            (unsigned long long)heap_dot_grad_cursor_grad);
    }

    // Emit all ctx tensors that the heap walk didn't already surface —
    // these are vestigial tensors absorbed into kernels (e.g. cotangent
    // seeds from VJP) or materialized side-tensors that have no direct
    // heap reference.  Render with a DASHED outline so they're visually
    // distinguished from tensors currently live on the graph edge set.
    for (u32 _tid = 1; _tid < ctx->tensor_count && _tid < 256; _tid++) {
        if (ten_emitted[_tid]) continue;
        TensorMeta *_m = &ctx->tensors[_tid];
        char _sh[64] = ""; int _p = 0;
        for (u32 _d = 0; _d < _m->view.shape.rank; _d++)
            _p += snprintf(_sh + _p, sizeof(_sh) - _p, "%s%u",
                           _d ? "," : "", _m->view.shape.dims[_d]);
        const char *_dt = dtype_name(_m->dtype);
        fprintf(f, "  t%u [label=\"t%u\\n[%s]\\n%s (absorbed)\",shape=box,"
                   "fillcolor=\"#f0f0f0\",style=\"dashed,filled\","
                   "color=\"#888888\",fontcolor=\"#666666\"];\n",
                _tid, _tid, _sh, _dt);
    }

    fprintf(f, "}\n");
    fclose(f);
    // Discard scratch allocations (e.g. kernel-to-compute temp payloads).
    if (ctx->heap_pos > _dot_heap_pos_save)
        ctx->heap_pos = _dot_heap_pos_save;
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
    #undef EMIT_VAR_OR_RESOLVED_TO_TARGET
    #undef EMIT_VAR_OR_RESOLVED_TO_NODE
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

char *thvm_heap_dot_string(TinyHVM *ctx, Term root) {
    char template[] = "/tmp/thvm_heap_dot_XXXXXX.dot";
    int fd = mkstemps(template, 4);
    char *buf = NULL;
    FILE *f = NULL;
    long len = 0;
    int old_hl_on = heap_dot_hl_on;
    u64 old_hl_slot = heap_dot_hl_slot;
    Term old_hl_term = heap_dot_hl_term;
    int old_hl_hit = heap_dot_hl_hit;
    int old_root_only = heap_dot_root_only;
    int old_sched_kernels = heap_dot_include_sched_kernels;
    char old_prev_name[sizeof(heap_dot_prev_name)];
    char old_next_name[sizeof(heap_dot_next_name)];

    if (fd < 0)
        return NULL;

    close(fd);

    memcpy(old_prev_name, heap_dot_prev_name, sizeof(heap_dot_prev_name));
    memcpy(old_next_name, heap_dot_next_name, sizeof(heap_dot_next_name));

    thvm_heap_dot_set_highlight(0, 0);
    thvm_heap_dot_set_step_meta("", "");
    thvm_heap_dot_set_sched_kernels(0);
    heap_dot_root_only = 1;

    thvm_heap_dot_root(ctx, template, root);

    f = fopen(template, "rb");
    if (!f)
        goto cleanup;

    if (fseek(f, 0, SEEK_END) != 0)
        goto cleanup;

    len = ftell(f);
    if (len < 0)
        goto cleanup;

    if (fseek(f, 0, SEEK_SET) != 0)
        goto cleanup;

    buf = (char *)malloc((size_t)len + 1);
    if (!buf)
        goto cleanup;

    if (fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        buf = NULL;
        goto cleanup;
    }

    buf[len] = '\0';

cleanup:
    if (f)
        fclose(f);

    unlink(template);

    heap_dot_hl_on = old_hl_on;
    heap_dot_hl_slot = old_hl_slot;
    heap_dot_hl_term = old_hl_term;
    heap_dot_hl_hit = old_hl_hit;
    heap_dot_root_only = old_root_only;
    heap_dot_include_sched_kernels = old_sched_kernels;
    memcpy(heap_dot_prev_name, old_prev_name, sizeof(heap_dot_prev_name));
    memcpy(heap_dot_next_name, old_next_name, sizeof(heap_dot_next_name));

    return buf;
}

