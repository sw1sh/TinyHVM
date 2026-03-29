// inet/_.c — Interaction combinator term constructors

Term thvm_lam(TinyHVM *ctx, Term *var_out, Term body) {
    u64 loc = heap_alloc(ctx, 2);
    Term var = term_new(TAG_VAR, 0, loc);
    heap_set(ctx, loc,     term_set_sub(var));
    heap_set(ctx, loc + 1, body);
    if (var_out) *var_out = var;
    return term_new(TAG_LAM, 0, loc);
}

Term thvm_app(TinyHVM *ctx, Term fun, Term arg) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     fun);
    heap_set(ctx, loc + 1, arg);
    return term_new(TAG_APP, 0, loc);
}

Term thvm_sup(TinyHVM *ctx, u32 label, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     a);
    heap_set(ctx, loc + 1, b);
    return term_new(TAG_SUP, label, loc);
}

// DUP: split a term into two projections (DP0, DP1) with a label.
// heap[loc] = value. DP0/DP1 reduce the value, then fire DUP interaction rules.
// Label determines annihilation (same label SUP) vs commutation (different label).
void thvm_dup(TinyHVM *ctx, u32 label, Term z, Term *out0, Term *out1) {
    u64 loc = heap_alloc(ctx, 1);
    heap_set(ctx, loc, z);
    *out0 = term_new(TAG_DP0, label, loc);
    *out1 = term_new(TAG_DP1, label, loc);
}

// Allocate a fresh label (monotonic counter). Only call at search-space construction
// time — interaction rules propagate existing labels, never create fresh ones.
u32 thvm_fresh_label(TinyHVM *ctx) {
    return ctx->next_sup_label++;
}

u32 thvm_define(TinyHVM *ctx, Term body) {
    assert(ctx->def_count < 256);
    u32 name = ctx->def_count++;
    ctx->defs[name] = body;
    return name;
}

Term thvm_ref(TinyHVM *ctx, u32 name) {
    (void)ctx;
    return term_new(TAG_REF, name, 0);
}

Term thvm_where(TinyHVM *ctx, Term cond, Term then_t, Term else_t) {
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     cond);
    heap_set(ctx, loc + 1, then_t);
    heap_set(ctx, loc + 2, else_t);
    return term_new(TAG_TOP, UOP_WHERE, loc);
}

Term thvm_assign(TinyHVM *ctx, Term dst, Term src) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     dst);
    heap_set(ctx, loc + 1, src);
    return term_new(TAG_TOP, UOP_ASSIGN, loc);
}

Term thvm_ifz(TinyHVM *ctx, Term counter, Term zero_case, Term succ_lam) {
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     counter);
    heap_set(ctx, loc + 1, zero_case);
    heap_set(ctx, loc + 2, succ_lam);
    return term_new(TAG_TOP, UOP_IFZ, loc);
}

Term thvm_log_print(TinyHVM *ctx, Term tensor) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     tensor);
    heap_set(ctx, loc + 1, term_era());
    return term_new(TAG_TOP, UOP_LOG_PRINT, loc);
}

// BRI: bridge (dual of lambda — contra-variant binding for ICC types)
Term thvm_bri(TinyHVM *ctx, Term *var_out, Term body) {
    u64 loc = heap_alloc(ctx, 2);
    Term var = term_new(TAG_VAR, 0, loc);
    heap_set(ctx, loc,     term_set_sub(var));
    heap_set(ctx, loc + 1, body);
    if (var_out) *var_out = var;
    return term_new(TAG_BRI, 0, loc);
}

// ANN: annotation {term : type}
Term thvm_ann(TinyHVM *ctx, Term term, Term type) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     term);
    heap_set(ctx, loc + 1, type);
    return term_new(TAG_ANN, 0, loc);
}

// DSU: dynamic superposition — label is an expression reduced at interaction time
Term thvm_dsu(TinyHVM *ctx, Term label_expr, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     label_expr);
    heap_set(ctx, loc + 1, a);
    heap_set(ctx, loc + 2, b);
    return term_new(TAG_DSU, 0, loc);
}

// DDU: dynamic dup — label is an expression, reduces then clones val
Term thvm_ddu(TinyHVM *ctx, Term label_expr, Term val, Term bod) {
    u64 loc = heap_alloc(ctx, 3);
    heap_set(ctx, loc,     label_expr);
    heap_set(ctx, loc + 1, val);
    heap_set(ctx, loc + 2, bod);
    return term_new(TAG_DDU, 0, loc);
}

// INC: priority wrapper — transparent to reduce, lower priority in collapse
Term thvm_inc(TinyHVM *ctx, Term term) {
    u64 loc = heap_alloc(ctx, 1);
    heap_set(ctx, loc, term);
    return term_new(TAG_INC, 0, loc);
}

// Collapse: DFS that extracts all SUP branches into a flat array
static void collapse_dfs(TinyHVM *ctx, Term t, CollapseResult *cr) {
    Term reduced = thvm_reduce(ctx, t);
    if (term_tag(reduced) == TAG_SUP) {
        u64 loc = term_val(reduced);
        collapse_dfs(ctx, heap_read(ctx, loc + 0), cr);
        collapse_dfs(ctx, heap_read(ctx, loc + 1), cr);
    } else {
        if (cr->count >= cr->cap) {
            cr->cap = cr->cap ? cr->cap * 2 : 16;
            cr->terms = realloc(cr->terms, cr->cap * sizeof(Term));
        }
        cr->terms[cr->count++] = reduced;
    }
}

CollapseResult thvm_collapse(TinyHVM *ctx, Term t) {
    CollapseResult cr = {NULL, 0, 0};
    collapse_dfs(ctx, t, &cr);
    return cr;
}

void thvm_collapse_free(CollapseResult *cr) {
    if (cr->terms) { free(cr->terms); cr->terms = NULL; }
    cr->count = cr->cap = 0;
}

// Grouped collapse: DFS that tracks (label, branch) path through SUPs
static void grouped_dfs(TinyHVM *ctx, Term t, GroupedCollapseResult *gr,
                        LabelChoice *path, u32 path_len, u32 path_cap) {
    Term reduced = thvm_reduce(ctx, t);
    if (term_tag(reduced) == TAG_SUP) {
        u64 loc = term_val(reduced);
        u32 label = term_ext(reduced);
        // Grow path buffer if needed
        if (path_len >= path_cap) {
            path_cap = path_cap ? path_cap * 2 : 16;
            path = realloc(path, path_cap * sizeof(LabelChoice));
        }
        // Left branch: branch=0
        LabelChoice *left_path = malloc(path_cap * sizeof(LabelChoice));
        memcpy(left_path, path, path_len * sizeof(LabelChoice));
        left_path[path_len] = (LabelChoice){label, 0};
        grouped_dfs(ctx, heap_read(ctx, loc + 0), gr, left_path, path_len + 1, path_cap);
        // Right branch: branch=1
        LabelChoice *right_path = malloc(path_cap * sizeof(LabelChoice));
        memcpy(right_path, path, path_len * sizeof(LabelChoice));
        right_path[path_len] = (LabelChoice){label, 1};
        grouped_dfs(ctx, heap_read(ctx, loc + 1), gr, right_path, path_len + 1, path_cap);
        free(path);  // this copy no longer needed
    } else {
        // Leaf: store term + path
        if (gr->count >= gr->cap) {
            gr->cap = gr->cap ? gr->cap * 2 : 16;
            gr->leaves = realloc(gr->leaves, gr->cap * sizeof(GroupedLeaf));
        }
        gr->leaves[gr->count++] = (GroupedLeaf){reduced, path, path_len};
    }
}

GroupedCollapseResult thvm_collapse_grouped(TinyHVM *ctx, Term t) {
    GroupedCollapseResult gr = {NULL, 0, 0};
    LabelChoice *path = malloc(16 * sizeof(LabelChoice));
    grouped_dfs(ctx, t, &gr, path, 0, 16);
    return gr;
}

void thvm_collapse_grouped_free(GroupedCollapseResult *gr) {
    for (u32 i = 0; i < gr->count; i++)
        if (gr->leaves[i].path) free(gr->leaves[i].path);
    if (gr->leaves) free(gr->leaves);
    gr->leaves = NULL;
    gr->count = gr->cap = 0;
}

// Parallel collapse: distribute SUP branches across threads
#include <pthread.h>

typedef struct {
    TinyHVM *ctx;
    Term *tasks;
    u32 task_start, task_end;
    u32 tid;
    CollapseResult result;
} CollapseWorkerArg;

static void *collapse_worker_fn(void *arg) {
    CollapseWorkerArg *w = (CollapseWorkerArg *)arg;
    tl_thread_id = w->tid;
    w->result = (CollapseResult){NULL, 0, 0};
    for (u32 i = w->task_start; i < w->task_end; i++)
        collapse_dfs(w->ctx, w->tasks[i], &w->result);
    return NULL;
}

// Collect top-level SUP branches (non-reducing walk of already-reduced SUP tree)
static void collect_sup_branches(TinyHVM *ctx, Term t, Term **out, u32 *count, u32 *cap, u32 depth) {
    if (depth > 0 && term_tag(t) == TAG_SUP) {
        u64 loc = term_val(t);
        collect_sup_branches(ctx, heap_read(ctx, loc + 0), out, count, cap, depth - 1);
        collect_sup_branches(ctx, heap_read(ctx, loc + 1), out, count, cap, depth - 1);
    } else {
        if (*count >= *cap) {
            *cap = *cap ? *cap * 2 : 16;
            *out = realloc(*out, *cap * sizeof(Term));
        }
        (*out)[(*count)++] = t;
    }
}

CollapseResult thvm_collapse_par(TinyHVM *ctx, Term t, u32 n_threads) {
    if (n_threads <= 1) return thvm_collapse(ctx, t);

    // First reduce root to expose top-level SUP tree
    Term reduced = thvm_reduce(ctx, t);
    if (term_tag(reduced) != TAG_SUP)  {
        CollapseResult cr = {malloc(sizeof(Term)), 1, 1};
        cr.terms[0] = reduced;
        return cr;
    }

    // Collect branches from the top-level SUP tree (depth up to ~14 levels = 16K tasks max)
    Term *tasks = NULL;
    u32 task_count = 0, task_cap = 0;
    u32 target_depth = 1;
    while ((1u << target_depth) < n_threads * 4 && target_depth < 14) target_depth++;
    collect_sup_branches(ctx, reduced, &tasks, &task_count, &task_cap, target_depth);

    if (task_count <= 1 || n_threads <= 1) {
        // Not enough work to parallelize — fall back to serial
        free(tasks);
        CollapseResult cr = {NULL, 0, 0};
        collapse_dfs(ctx, reduced, &cr);
        return cr;
    }

    // Set up per-thread heap banks for safe concurrent allocation
    u32 nt = (n_threads < 16) ? n_threads : 16;
    if (nt > task_count) nt = task_count;

    // Init thread banks
    u64 heap_per = (HEAP_CAP - ctx->heap_pos) / nt;
    u64 hbase = ctx->heap_pos;
    u32 tensors_per = (MAX_TENSORS - ctx->tensor_count) / nt;
    u32 tbase = ctx->tensor_count;
    for (u32 i = 0; i < nt; i++) {
        ThvmThread *ts = &ctx->threads[i];
        ts->tid = i;
        ts->bank_start = hbase + i * heap_per;
        ts->bank_next  = ts->bank_start;
        ts->bank_end   = (i == nt - 1) ? HEAP_CAP : ts->bank_start + heap_per;
        ts->tensor_start = tbase + i * tensors_per;
        ts->tensor_next  = ts->tensor_start;
        ts->tensor_end   = (i == nt - 1) ? MAX_TENSORS : ts->tensor_start + tensors_per;
        ts->itrs = 0;
    }
    ctx->n_threads = nt;

    // Distribute tasks across workers
    u32 per_worker = task_count / nt;
    u32 leftover = task_count % nt;
    CollapseWorkerArg *args = calloc(nt, sizeof(CollapseWorkerArg));
    pthread_t *threads = calloc(nt, sizeof(pthread_t));
    u32 pos = 0;
    for (u32 i = 0; i < nt; i++) {
        args[i].ctx = ctx;
        args[i].tasks = tasks;
        args[i].tid = i;
        args[i].task_start = pos;
        u32 chunk = per_worker + (i < leftover ? 1 : 0);
        args[i].task_end = pos + chunk;
        pos += chunk;
    }

    // Spawn workers (main thread does worker 0's share)
    for (u32 i = 1; i < nt; i++)
        pthread_create(&threads[i], NULL, collapse_worker_fn, &args[i]);

    // Main thread = worker 0
    tl_thread_id = 0;
    args[0].result = (CollapseResult){NULL, 0, 0};
    for (u32 i = args[0].task_start; i < args[0].task_end; i++)
        collapse_dfs(ctx, tasks[i], &args[0].result);

    for (u32 i = 1; i < nt; i++)
        pthread_join(threads[i], NULL);

    // Finalize banks: update global watermarks
    u64 max_heap = ctx->heap_pos;
    u32 max_tensor = ctx->tensor_count;
    for (u32 i = 0; i < nt; i++) {
        if (ctx->threads[i].bank_next > max_heap)
            max_heap = ctx->threads[i].bank_next;
        if (ctx->threads[i].tensor_next > max_tensor)
            max_tensor = ctx->threads[i].tensor_next;
    }
    ctx->heap_pos = max_heap;
    ctx->tensor_count = max_tensor;
    ctx->n_threads = 0;

    // Merge results
    u32 total = 0;
    for (u32 i = 0; i < nt; i++) total += args[i].result.count;
    CollapseResult cr = {malloc(total * sizeof(Term)), 0, total};
    for (u32 i = 0; i < nt; i++) {
        if (args[i].result.count > 0) {
            memcpy(cr.terms + cr.count, args[i].result.terms,
                   args[i].result.count * sizeof(Term));
            cr.count += args[i].result.count;
        }
        free(args[i].result.terms);
    }

    free(tasks);
    free(args);
    free(threads);
    return cr;
}

// OP2: binary integer operation on TAG_NUM values.
// opr: 0=add, 1=sub, 2=mul, 3=div, 4=eq, 5=mod
Term thvm_op2(TinyHVM *ctx, u32 opr, Term x, Term y) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     x);
    heap_set(ctx, loc + 1, y);
    return term_new(TAG_OP2, opr, loc);
}
