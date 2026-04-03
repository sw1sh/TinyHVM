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

static inline u32 hc_hash_pair(Term f, Term x) {
    u64 h = f * 0x9E3779B97F4A7C15ULL + x;
    h ^= h >> 30;
    h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27;
    return (u32)(h & (HC_TABLE_CAP - 1));
}

Term thvm_app_hc(TinyHVM *ctx, Term fun, Term arg) {
    if (!ctx->hc_table)
        ctx->hc_table = calloc(HC_TABLE_CAP, sizeof(HCSlot));
    u32 h = hc_hash_pair(fun, arg);
    for (;;) {
        HCSlot *s = &ctx->hc_table[h];
        if (s->result == 0) {
            u64 loc = heap_alloc(ctx, 2);
            heap_set(ctx, loc,     fun);
            heap_set(ctx, loc + 1, arg);
            Term t = term_new(TAG_APP, 0, loc);
            s->fun = fun;
            s->arg = arg;
            s->result = t;
            return t;
        }
        if (s->fun == fun && s->arg == arg)
            return s->result;
        h = (h + 1) & (HC_TABLE_CAP - 1);
    }
}

void thvm_hc_clear(TinyHVM *ctx) {
    if (ctx->hc_table)
        memset(ctx->hc_table, 0, HC_TABLE_CAP * sizeof(HCSlot));
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

// EQL: structural equality — heap [a, b]
Term thvm_eql(TinyHVM *ctx, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     a);
    heap_set(ctx, loc + 1, b);
    return term_new(TAG_EQL, 0, loc);
}

// AND: short-circuit AND — heap [a, b]
Term thvm_and(TinyHVM *ctx, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     a);
    heap_set(ctx, loc + 1, b);
    return term_new(TAG_AND, 0, loc);
}

// OR: short-circuit OR — heap [a, b]
Term thvm_or(TinyHVM *ctx, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     a);
    heap_set(ctx, loc + 1, b);
    return term_new(TAG_OR, 0, loc);
}

// MAT: pattern match — heap [handler, fallback], EXT = match_tag
Term thvm_mat(TinyHVM *ctx, u32 match_tag, Term handler, Term fallback) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     handler);
    heap_set(ctx, loc + 1, fallback);
    return term_new(TAG_MAT, (u8)match_tag, loc);
}

// USP: unordered SUP — heap [a, b], EXT = label
Term thvm_usp(TinyHVM *ctx, u32 label, Term a, Term b) {
    u64 loc = heap_alloc(ctx, 2);
    heap_set(ctx, loc,     a);
    heap_set(ctx, loc + 1, b);
    return term_new(TAG_USP, (u8)label, loc);
}

// UDP: unordered DUP — heap [val], EXT = label, single output port
Term thvm_udp(TinyHVM *ctx, u32 label, Term val) {
    u64 loc = heap_alloc(ctx, 1);
    heap_set(ctx, loc, val);
    return term_new(TAG_UDP, (u8)label, loc);
}

// INC: priority wrapper — transparent to reduce, lower priority in collapse
Term thvm_inc(TinyHVM *ctx, Term term) {
    u64 loc = heap_alloc(ctx, 1);
    heap_set(ctx, loc, term);
    return term_new(TAG_INC, 0, loc);
}

// Collapse: DFS that extracts all SUP/USP branches into a flat array
static void collapse_dfs(TinyHVM *ctx, Term t, CollapseResult *cr) {
    Term reduced = thvm_reduce(ctx, t);
    if (term_tag(reduced) == TAG_SUP || term_tag(reduced) == TAG_USP) {
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

// Priority-aware collapse: 64-bucket priority queue.
// INC decreases key (explore sooner), SUP/USP increases key (explore later).
#define COLLAPSE_BUCKETS 64

typedef struct { Term *tasks; u32 head, count, cap; } CollapseBucket;

static void bucket_push(CollapseBucket *b, Term t) {
    if (b->count >= b->cap) {
        u32 new_cap = b->cap ? b->cap * 2 : 16;
        Term *new_tasks = (Term *)malloc(new_cap * sizeof(Term));
        for (u32 i = 0; i < b->count; i++)
            new_tasks[i] = b->tasks[(b->head + i) % b->cap];
        free(b->tasks);
        b->tasks = new_tasks;
        b->head = 0;
        b->cap = new_cap;
    }
    u32 tail = (b->head + b->count) % b->cap;
    b->tasks[tail] = t;
    b->count++;
}

static Term bucket_pop(CollapseBucket *b) {
    Term t = b->tasks[b->head];
    b->head = (b->head + 1) % b->cap;
    b->count--;
    return t;
}

static void cr_push(CollapseResult *cr, Term t) {
    if (cr->count >= cr->cap) {
        cr->cap = cr->cap ? cr->cap * 2 : 16;
        cr->terms = realloc(cr->terms, cr->cap * sizeof(Term));
    }
    cr->terms[cr->count++] = t;
}

CollapseResult thvm_collapse_ordered(TinyHVM *ctx, Term t, u32 limit) {
    CollapseBucket buckets[COLLAPSE_BUCKETS];
    memset(buckets, 0, sizeof(buckets));
    CollapseResult cr = {NULL, 0, 0};

    // Seed at middle priority
    bucket_push(&buckets[32], t);

    while (cr.count < limit) {
        // Find lowest non-empty bucket
        int best = -1;
        for (int i = 0; i < COLLAPSE_BUCKETS; i++) {
            if (buckets[i].count > 0) { best = i; break; }
        }
        if (best < 0) break;

        Term task = bucket_pop(&buckets[best]);
        // Peel INC wrappers before reducing (thvm_reduce strips INC transparently).
        // Re-enqueue at adjusted priority so ordering is respected.
        if (term_tag(task) == TAG_INC) {
            while (term_tag(task) == TAG_INC) {
                best = (best > 0) ? best - 1 : 0;
                task = heap_read(ctx, term_val(task));
            }
            bucket_push(&buckets[best], task);
            continue;
        }
        Term reduced = thvm_reduce(ctx, task);
        u8 tag = term_tag(reduced);

        if (tag == TAG_SUP || tag == TAG_USP) {
            // SUP/USP: increase key (lower priority = explored later)
            u32 key = (best < COLLAPSE_BUCKETS - 1) ? (u32)(best + 1) : (u32)(COLLAPSE_BUCKETS - 1);
            u64 loc = term_val(reduced);
            bucket_push(&buckets[key], heap_read(ctx, loc + 0));
            bucket_push(&buckets[key], heap_read(ctx, loc + 1));
        } else if (tag == TAG_ERA) {
            continue;  // skip erased branches
        } else {
            cr_push(&cr, reduced);
        }
    }

    for (int i = 0; i < COLLAPSE_BUCKETS; i++) free(buckets[i].tasks);
    return cr;
}

// Parallel collapse via Chase-Lev work-stealing deques.
// Each worker owns a deque: push SUP children (LIFO for locality),
// idle workers steal from others (FIFO for load balance).
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>

typedef struct {
    TinyHVM *ctx;
    WsDeque  deque;
    u32      tid;
    u32      n_workers;
    CollapseResult result;
} WSCollapseWorker;

static WSCollapseWorker ws_cw[16];
static _Atomic(i32)     ws_remaining;  // tasks in-flight + in deques

static inline void ws_store_leaf(WSCollapseWorker *w, Term t) {
    if (w->result.count >= w->result.cap) {
        w->result.cap = w->result.cap ? w->result.cap * 2 : 256;
        w->result.terms = realloc(w->result.terms, w->result.cap * sizeof(Term));
    }
    w->result.terms[w->result.count++] = t;
}

static inline void ws_process(WSCollapseWorker *w, Term task) {
    Term reduced = thvm_reduce(w->ctx, task);
    if (term_tag(reduced) == TAG_SUP) {
        u64 loc = term_val(reduced);
        // +2 children -1 self = net +1
        atomic_fetch_add_explicit(&ws_remaining, 1, memory_order_relaxed);
        ws_push(&w->deque, heap_read(w->ctx, loc + 0));
        ws_push(&w->deque, heap_read(w->ctx, loc + 1));
    } else {
        ws_store_leaf(w, reduced);
        atomic_fetch_sub_explicit(&ws_remaining, 1, memory_order_release);
    }
}

static void *ws_collapse_fn(void *arg) {
    WSCollapseWorker *w = (WSCollapseWorker *)arg;
    tl_thread_id = w->tid;
    u32 idle_spins = 0;

    for (;;) {
        u64 task;
        // Try own deque (LIFO — cache-friendly depth-first)
        if (ws_pop(&w->deque, &task)) {
            ws_process(w, (Term)task);
            idle_spins = 0;
            continue;
        }
        // Try stealing from others (FIFO — breadth-first load balance)
        int found = 0;
        for (u32 i = 1; i < w->n_workers; i++) {
            u32 victim = (w->tid + i) % w->n_workers;
            if (ws_steal(&ws_cw[victim].deque, &task)) {
                ws_process(w, (Term)task);
                found = 1;
                idle_spins = 0;
                break;
            }
        }
        if (found) continue;
        // No work — check termination
        if (atomic_load_explicit(&ws_remaining, memory_order_acquire) <= 0) break;
        if (++idle_spins > 64) { sched_yield(); idle_spins = 0; }
    }
    return NULL;
}

CollapseResult thvm_collapse_par(TinyHVM *ctx, Term t, u32 n_threads) {
    if (n_threads <= 1) return thvm_collapse(ctx, t);

    // Phase 1: Serial reduce-and-collect.
    // collapse_dfs fires ALL interaction rules (serial — required because branches
    // share heap state via DUP nodes). After this, cr contains flat WHNF leaves.
    CollapseResult cr = {NULL, 0, 0};
    collapse_dfs(ctx, t, &cr);
    if (cr.count == 0) return cr;

    // Phase 2: Parallel final-reduce on collected leaves.
    // Leaves are typically already NUMs/TENs (WHNF atoms), so this is trivially
    // fast. But if any leaf is a complex term (e.g., from grouped collapse that
    // stopped early), threads can reduce them independently since they no longer
    // share DUP structure after collapse_dfs resolved all SUPs.
    u32 nt = (n_threads < 16) ? n_threads : 16;
    if (cr.count < nt * 64) return cr;  // not enough work to justify threading

    // Init per-thread heap banks
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

    // Init work-stealing workers and seed with collected leaves
    for (u32 i = 0; i < nt; i++) {
        ws_init(&ws_cw[i].deque);
        ws_cw[i].ctx = ctx;
        ws_cw[i].tid = i;
        ws_cw[i].n_workers = nt;
        ws_cw[i].result = (CollapseResult){NULL, 0, 0};
    }

    // Distribute leaves round-robin across worker deques
    u32 seeded = 0;
    for (u32 i = 0; i < cr.count; i++) {
        ws_push(&ws_cw[i % nt].deque, cr.terms[i]);
        seeded++;
    }
    atomic_store_explicit(&ws_remaining, (i32)seeded, memory_order_release);

    // Spawn workers (main thread = worker 0)
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 32 * 1024 * 1024);
    pthread_t threads[16];
    for (u32 i = 1; i < nt; i++)
        pthread_create(&threads[i], &attr, ws_collapse_fn, &ws_cw[i]);
    pthread_attr_destroy(&attr);

    tl_thread_id = 0;
    ws_collapse_fn(&ws_cw[0]);

    for (u32 i = 1; i < nt; i++)
        pthread_join(threads[i], NULL);

    // Finalize banks
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

    // Free the serial-phase results and merge parallel results
    free(cr.terms);
    u32 total = 0;
    for (u32 i = 0; i < nt; i++) total += ws_cw[i].result.count;
    cr = (CollapseResult){malloc(total * sizeof(Term)), 0, total};
    for (u32 i = 0; i < nt; i++) {
        if (ws_cw[i].result.count > 0) {
            memcpy(cr.terms + cr.count, ws_cw[i].result.terms,
                   ws_cw[i].result.count * sizeof(Term));
            cr.count += ws_cw[i].result.count;
        }
        free(ws_cw[i].result.terms);
    }

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
