/* Saved from gist (Victor Taelin, 2026-04-13):
 * https://gist.github.com/VictorTaelin/f2c4440cc366f68fd9d3fcfb6ff426d5
 * Title: par_tree_sum.bend compiled to C (Bend2 / experimental codegen).
 */

//# Tree sum: allocates a complete binary tree with U32 leaves 0,1,2,...
//# then sums all values in parallel. Tests heap-heavy parallel workloads.
//
//type Tree() {
//  leaf{U32}
//  node{Tree, Tree}
//}
//
//def build(d: U32, x: U32) -> Tree:
//  match d:
//    case 0: leaf{x}
//    case d:
//      l & r = build(d - 1, x * 2) & build(d - 1, x * 2 + 1)
//      node{l, r}
//
//def sum(t: Tree) -> U32:
//  match t:
//    case leaf{v}: v
//    case node{l, r}:
//      a & b = sum(l) & sum(r)
//      a + b
//
//def main() -> U32:
//  sum(build(27, 0))

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>

/* ── Platform ─────────────────────────────────────────────────────── */

#if defined(__aarch64__)
  #define SPIN_PAUSE() __asm__ volatile("yield")
#elif defined(__x86_64__) || defined(_M_X64)
  #define SPIN_PAUSE() __asm__ volatile("pause")
#else
  #define SPIN_PAUSE() ((void)0)
#endif

/* ── Types & constants ────────────────────────────────────────────── */

typedef uint64_t Term;

#define CACHE_LINE   128
#define HEAP_CHUNK   4194304
#define MAX_THREADS  256
#define TASK_BUF_CAP (MAX_THREADS * 8)
#define MAX_ARGS     4
#define SEQ_STK_CAP  4096

#define R_VALUE  0
#define R_SPLIT  1
#define R_CALL   2

#define ROOT_RET  0xFFFFFFFFu
#define SEQ_VALUE 0xFFFFFFFFu

/* Function / continuation IDs */
#define FN_BUILD    0
#define FN_SUM      1
#define FN_MAIN     2
#define CONT_BUILD  3
#define CONT_SUM    4
#define CONT_MAIN   5

/* ── Heap ─────────────────────────────────────────────────────────── */

static uint8_t  *heap_T;
static uint32_t *heap_L;
static uint32_t  heap_cap;

static uint32_t g_heap_bump __attribute__((aligned(CACHE_LINE)));
static __thread uint32_t tl_hp = 0, tl_he = 0;

static inline uint8_t  ctr_tag(Term t) { return (uint8_t)(t >> 32); }
static inline uint32_t ctr_loc(Term t) { return (uint32_t)t; }

static inline Term make_ctr(uint8_t tag, uint32_t loc) {
  return ((uint64_t)tag << 32) | (uint64_t)loc;
}

static inline Term heap_get(uint32_t i) {
  return ((uint64_t)heap_T[i] << 32) | heap_L[i];
}

static inline void heap_set(uint32_t i, Term v) {
  heap_T[i] = (uint8_t)(v >> 32);
  heap_L[i] = (uint32_t)v;
}

static inline Term ctr_get(Term t, uint32_t i) {
  return heap_get(ctr_loc(t) + i);
}

static inline uint32_t alloc(uint32_t n) {
  if (__builtin_expect(tl_hp + n > tl_he, 0)) {
    tl_hp = __atomic_fetch_add(&g_heap_bump, HEAP_CHUNK, __ATOMIC_RELAXED);
    tl_he = tl_hp + HEAP_CHUNK;
  }
  uint32_t p = tl_hp;
  tl_hp += n;
  return p;
}

/* ── Tasks & results ──────────────────────────────────────────────── */

struct Task   { uint32_t fn, ret; Term args[MAX_ARGS]; };
struct Result { uint32_t tag; Term val; struct Task t0, t1; };

static inline struct Task make_task(uint32_t fn, uint32_t ret,
                                    Term a0, Term a1, Term a2, Term a3) {
  struct Task t;
  memset(&t, 0, sizeof(t));
  t.fn = fn; t.ret = ret;
  t.args[0] = a0; t.args[1] = a1; t.args[2] = a2; t.args[3] = a3;
  return t;
}

static inline struct Result make_value(Term v) {
  struct Result r; r.tag = R_VALUE; r.val = v; return r;
}
static inline struct Result make_split(struct Task a, struct Task b) {
  struct Result r; r.tag = R_SPLIT; r.t0 = a; r.t1 = b; return r;
}
static inline struct Result make_call(struct Task a) {
  struct Result r; r.tag = R_CALL; r.t0 = a; return r;
}

static inline uint32_t enc_ret(uint32_t dp, uint32_t slot) {
  return (dp << 1) | slot;
}

/* ── Barrier ──────────────────────────────────────────────────────── */

typedef struct {
  uint32_t cnt __attribute__((aligned(CACHE_LINE)));
  uint32_t gen __attribute__((aligned(CACHE_LINE)));
  int tot;
  pthread_mutex_t m;
  pthread_cond_t  c;
} Barrier __attribute__((aligned(CACHE_LINE)));

static Barrier g_barrier;
static int     g_num_threads;
static int     g_spin_only;

static void barrier_init(Barrier *b, int n) {
  __atomic_store_n(&b->cnt, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&b->gen, 0, __ATOMIC_RELAXED);
  b->tot = n;
  pthread_mutex_init(&b->m, NULL);
  pthread_cond_init(&b->c, NULL);
}

static void barrier_wait(Barrier *b) {
  uint32_t g = __atomic_load_n(&b->gen, __ATOMIC_ACQUIRE);
  if (__atomic_fetch_add(&b->cnt, 1, __ATOMIC_ACQ_REL) + 1 == (uint32_t)b->tot) {
    __atomic_store_n(&b->cnt, 0, __ATOMIC_RELAXED);
    __atomic_fetch_add(&b->gen, 1, __ATOMIC_RELEASE);
    if (!g_spin_only) {
      pthread_mutex_lock(&b->m);
      pthread_cond_broadcast(&b->c);
      pthread_mutex_unlock(&b->m);
    }
  } else if (g_spin_only) {
    while (__atomic_load_n(&b->gen, __ATOMIC_ACQUIRE) == g)
      SPIN_PAUSE();
  } else {
    for (int i = 0; i < 128; i++) {
      if (__atomic_load_n(&b->gen, __ATOMIC_ACQUIRE) != g) return;
      SPIN_PAUSE();
    }
    pthread_mutex_lock(&b->m);
    while (__atomic_load_n(&b->gen, __ATOMIC_ACQUIRE) == g)
      pthread_cond_wait(&b->c, &b->m);
    pthread_mutex_unlock(&b->m);
  }
}

static void barrier_destroy(Barrier *b) {
  pthread_mutex_destroy(&b->m);
  pthread_cond_destroy(&b->c);
}

/* ── Global parallel state ────────────────────────────────────────── */

static struct Task *g_task_buf;
static uint32_t     g_task_cnt;
static uint32_t     g_task_cnt_new __attribute__((aligned(CACHE_LINE)));
static uint32_t     g_work_idx    __attribute__((aligned(CACHE_LINE)));
static uint32_t     g_done        __attribute__((aligned(CACHE_LINE)));
static Term         g_result      __attribute__((aligned(CACHE_LINE)));

/* ── Continuation fire / write ────────────────────────────────────── */

static inline void write_slot(uint32_t ret, Term val) {
  uint32_t dp = ret >> 1, slot = ret & 1;
  __atomic_fetch_or(&heap_L[dp + 2],
                    (uint32_t)ctr_tag(val) << (slot * 8), __ATOMIC_RELAXED);
  heap_L[dp + 3 + slot] = ctr_loc(val);
}

static struct Task fire_cont(uint32_t dp) {
  uint32_t hdr    = heap_L[dp];
  uint32_t fn_id  = hdr >> 12;
  uint32_t n_saved = (hdr >> 2) & 0x3FFu;
  struct Task t = make_task(fn_id, heap_L[dp + 1], 0, 0, 0, 0);
  uint32_t tags = heap_L[dp + 2];
  t.args[0] = make_ctr((uint8_t)(tags >> 0), heap_L[dp + 3]);
  t.args[1] = make_ctr((uint8_t)(tags >> 8), heap_L[dp + 4]);
  for (uint32_t i = 0; i < n_saved; i++)
    t.args[2 + i] = make_ctr(heap_T[dp + 5 + i], heap_L[dp + 5 + i]);
  return t;
}

/* ── Parallel entry points ────────────────────────────────────────── */

static inline Term make_leaf(Term v) {
  uint32_t loc = alloc(1);
  heap_set(loc, v);
  return make_ctr(1, loc);
}

static inline Term make_node(Term l, Term r) {
  uint32_t loc = alloc(2);
  heap_set(loc, l);
  heap_set(loc + 1, r);
  return make_ctr(2, loc);
}

static inline void alloc_dp(uint32_t cont_id, uint32_t ret, uint32_t *dp_out) {
  uint32_t dp = alloc(5);
  heap_L[dp]     = (cont_id << 12) | (0u << 2) | 2u;
  heap_L[dp + 1] = ret;
  heap_L[dp + 2] = 0;
  heap_L[dp + 3] = 0;
  heap_L[dp + 4] = 0;
  *dp_out = dp;
}

static inline void alloc_dp_call(uint32_t cont_id, uint32_t ret, uint32_t *dp_out) {
  uint32_t dp = alloc(5);
  heap_L[dp]     = (cont_id << 12) | (0u << 2) | 1u;
  heap_L[dp + 1] = ret;
  heap_L[dp + 2] = 0;
  heap_L[dp + 3] = 0;
  heap_L[dp + 4] = 0;
  *dp_out = dp;
}

/* par build: entry */
static struct Result par_build(uint32_t ret, Term *args) {
  Term d = args[0], x = args[1];
  if (d == 0)
    return make_value(make_leaf(x));

  uint32_t dp;
  alloc_dp(CONT_BUILD, ret, &dp);
  Term d1 = (d - 1) & 0xFFFFFFFFULL;
  Term x2 = (x * 2) & 0xFFFFFFFFULL;
  return make_split(
    make_task(FN_BUILD, enc_ret(dp, 0), d1, x2, 0, 0),
    make_task(FN_BUILD, enc_ret(dp, 1), d1, (x2 + 1) & 0xFFFFFFFFULL, 0, 0));
}

/* par build: continuation (both children done) */
static struct Result par_build_cont(uint32_t ret, Term *args) {
  return make_value(make_node(args[0], args[1]));
}

/* par sum: entry */
static struct Result par_sum(uint32_t ret, Term *args) {
  Term t = args[0];
  switch (ctr_tag(t)) {
    case 1: return make_value(ctr_get(t, 0));
    case 2: {
      uint32_t dp;
      alloc_dp(CONT_SUM, ret, &dp);
      return make_split(
        make_task(FN_SUM, enc_ret(dp, 0), ctr_get(t, 0), 0, 0, 0),
        make_task(FN_SUM, enc_ret(dp, 1), ctr_get(t, 1), 0, 0, 0));
    }
    default: return make_value(0);
  }
}

/* par sum: continuation */
static struct Result par_sum_cont(uint32_t ret, Term *args) {
  return make_value(((args[0]) + (args[1])) & 0xFFFFFFFFULL);
}

/* par main: entry */
static struct Result par_main(uint32_t ret, Term *args) {
  uint32_t dp;
  alloc_dp_call(CONT_MAIN, ret, &dp);
  return make_call(make_task(FN_BUILD, enc_ret(dp, 0), 27, 0, 0, 0));
}

/* par main: continuation */
static struct Result par_main_cont(uint32_t ret, Term *args) {
  return make_call(make_task(FN_SUM, ret, args[0], 0, 0, 0));
}

/* ── Parallel dispatch ────────────────────────────────────────────── */

static struct Result dispatch(struct Task *task) {
  switch (task->fn) {
    case FN_BUILD:   return par_build(task->ret, task->args);
    case FN_SUM:     return par_sum(task->ret, task->args);
    case FN_MAIN:    return par_main(task->ret, task->args);
    case CONT_BUILD: return par_build_cont(task->ret, task->args);
    case CONT_SUM:   return par_sum_cont(task->ret, task->args);
    case CONT_MAIN:  return par_main_cont(task->ret, task->args);
    default:         return make_value(0);
  }
}

/* ── Sequential CPS (managed stack) ──────────────────────────────── */

/* CPS continuation labels */
#define CPS_BUILD_AFTER_LEFT  6
#define CPS_BUILD_AFTER_RIGHT 7
#define CPS_SUM_AFTER_LEFT    8
#define CPS_SUM_AFTER_RIGHT   9
#define CPS_MAIN_AFTER_BUILD  10

static inline uint32_t seq_build_entry(Term *a, Term *stk, uint32_t *sp) {
  Term d = a[0], x = a[1];
  if (d == 0) {
    a[0] = make_leaf(x);
    return SEQ_VALUE;
  }
  Term d1 = (d - 1) & 0xFFFFFFFFULL;
  Term x2 = (x * 2) & 0xFFFFFFFFULL;
  /* save args for second recursive call */
  stk[(*sp)++] = d1;
  stk[(*sp)++] = (x2 + 1) & 0xFFFFFFFFULL;
  stk[(*sp)++] = CPS_BUILD_AFTER_LEFT;
  a[0] = d1; a[1] = x2;
  return FN_BUILD;
}

static inline uint32_t seq_build_after_left(Term *a, Term *stk, uint32_t *sp) {
  *sp -= 3;
  Term left = a[0];
  Term d2   = stk[*sp];
  Term x2p1 = stk[*sp + 1];
  stk[(*sp)++] = left;
  stk[(*sp)++] = CPS_BUILD_AFTER_RIGHT;
  a[0] = d2; a[1] = x2p1;
  return FN_BUILD;
}

static inline uint32_t seq_build_after_right(Term *a, Term *stk, uint32_t *sp) {
  *sp -= 2;
  Term left  = stk[*sp];
  Term right = a[0];
  a[0] = make_node(left, right);
  return SEQ_VALUE;
}

static inline uint32_t seq_sum_entry(Term *a, Term *stk, uint32_t *sp) {
  Term t = a[0];
  switch (ctr_tag(t)) {
    case 1:
      a[0] = ctr_get(t, 0);
      return SEQ_VALUE;
    case 2:
      stk[(*sp)++] = ctr_get(t, 1);
      stk[(*sp)++] = CPS_SUM_AFTER_LEFT;
      a[0] = ctr_get(t, 0);
      return FN_SUM;
    default:
      a[0] = 0;
      return SEQ_VALUE;
  }
}

static inline uint32_t seq_sum_after_left(Term *a, Term *stk, uint32_t *sp) {
  *sp -= 2;
  Term left_val = a[0];
  Term right    = stk[*sp];
  stk[(*sp)++] = left_val;
  stk[(*sp)++] = CPS_SUM_AFTER_RIGHT;
  a[0] = right;
  return FN_SUM;
}

static inline uint32_t seq_sum_after_right(Term *a, Term *stk, uint32_t *sp) {
  *sp -= 2;
  Term left_val = stk[*sp];
  a[0] = (left_val + a[0]) & 0xFFFFFFFFULL;
  return SEQ_VALUE;
}

static inline uint32_t seq_main_entry(Term *a, Term *stk, uint32_t *sp) {
  stk[(*sp)++] = CPS_MAIN_AFTER_BUILD;
  a[0] = 27; a[1] = 0;
  return FN_BUILD;
}

static inline uint32_t seq_main_after_build(Term *a, Term *stk, uint32_t *sp) {
  *sp -= 1;
  /* a[0] already holds the tree */
  return FN_SUM;
}

/* ── Sequential evaluators (trampoline per function) ──────────────── */

static Term eval_build(Term *args) {
  Term stk[SEQ_STK_CAP];
  uint32_t sp = 0;

entry:
  { uint32_t r = seq_build_entry(args, stk, &sp);
    if (r != SEQ_VALUE) goto entry;
    goto pop; }

pop:
  if (sp == 0) return args[0];
  { uint32_t cont = (uint32_t)stk[sp - 1];
    if (cont == CPS_BUILD_AFTER_LEFT) {
      uint32_t r = seq_build_after_left(args, stk, &sp);
      if (r != SEQ_VALUE) goto entry;
      goto pop;
    }
    /* CPS_BUILD_AFTER_RIGHT */
    seq_build_after_right(args, stk, &sp);
    goto pop;
  }
}

static Term eval_sum(Term *args) {
  Term stk[SEQ_STK_CAP];
  uint32_t sp = 0;

entry:
  { uint32_t r = seq_sum_entry(args, stk, &sp);
    if (r != SEQ_VALUE) goto entry;
    goto pop; }

pop:
  if (sp == 0) return args[0];
  { uint32_t cont = (uint32_t)stk[sp - 1];
    if (cont == CPS_SUM_AFTER_LEFT) {
      uint32_t r = seq_sum_after_left(args, stk, &sp);
      if (r != SEQ_VALUE) goto entry;
      goto pop;
    }
    /* CPS_SUM_AFTER_RIGHT */
    seq_sum_after_right(args, stk, &sp);
    goto pop;
  }
}

static Term eval_main(Term *args) {
  Term stk[SEQ_STK_CAP];
  uint32_t sp = 0;

  { seq_main_entry(args, stk, &sp);
    args[0] = eval_build(args);
    goto pop; }

pop:
  if (sp == 0) return args[0];
  { seq_main_after_build(args, stk, &sp);
    args[0] = eval_sum(args);
    goto pop; }
}

static Term eval(uint32_t fn, Term *args) {
  switch (fn) {
    case FN_BUILD: return eval_build(args);
    case FN_SUM:   return eval_sum(args);
    case FN_MAIN:  return eval_main(args);
    default:       return 0;
  }
}

/* ── Resolve: deliver values up the continuation tree ─────────────── */

static void resolve(uint32_t ret, Term val,
                    struct Task *out, uint32_t *out_n) {
  for (;;) {
    if (ret == ROOT_RET) {
      g_result = val;
      __atomic_store_n(&g_done, 1, __ATOMIC_RELEASE);
      return;
    }
    write_slot(ret, val);
    uint32_t dp  = ret >> 1;
    uint32_t old = __atomic_fetch_sub(&heap_L[dp], 1, __ATOMIC_ACQ_REL);
    if ((old & 3u) != 1u) return;          /* other branch not done yet */

    struct Task t = fire_cont(dp);
    struct Result r = dispatch(&t);
    if (r.tag == R_VALUE) { val = r.val; ret = t.ret; continue; }
    if (r.tag == R_SPLIT) {
      uint32_t i = __atomic_fetch_add(out_n, 2, __ATOMIC_RELAXED);
      out[i] = r.t0; out[i + 1] = r.t1;
    } else if (r.tag == R_CALL) {
      uint32_t i = __atomic_fetch_add(out_n, 1, __ATOMIC_RELAXED);
      out[i] = r.t0;
    }
    return;
  }
}

/* ── Worker thread ────────────────────────────────────────────────── */

static void *worker(void *arg) {
  int tid = (int)(intptr_t)arg;
  tl_hp = __atomic_fetch_add(&g_heap_bump, HEAP_CHUNK, __ATOMIC_RELAXED);
  tl_he = tl_hp + HEAP_CHUNK;

  for (;;) {
    barrier_wait(&g_barrier);
    if (g_task_cnt == 0 || __atomic_load_n(&g_done, __ATOMIC_ACQUIRE))
      break;

    /* SEED: split tasks until we have enough parallelism */
    uint32_t seed_target = (uint32_t)g_num_threads * 2;
    for (int iter = 0; iter < 64 && g_task_cnt > 0 && g_task_cnt < seed_target;
         iter++) {
      uint32_t stc = g_task_cnt;
      if (tid == 0) {
        __atomic_store_n(&g_task_cnt_new, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&g_work_idx, 0, __ATOMIC_RELAXED);
      }
      barrier_wait(&g_barrier);

      for (;;) {
        uint32_t si = __atomic_fetch_add(&g_work_idx, 1, __ATOMIC_RELAXED);
        if (si >= stc) break;
        struct Task my = g_task_buf[si];
        struct Result r = dispatch(&my);
        if (r.tag == R_SPLIT) {
          uint32_t j = __atomic_fetch_add(&g_task_cnt_new, 2, __ATOMIC_RELAXED);
          g_task_buf[stc + j]     = r.t0;
          g_task_buf[stc + j + 1] = r.t1;
        } else if (r.tag == R_CALL) {
          uint32_t j = __atomic_fetch_add(&g_task_cnt_new, 1, __ATOMIC_RELAXED);
          g_task_buf[stc + j] = r.t0;
        } else {
          resolve(my.ret, r.val, g_task_buf + stc, &g_task_cnt_new);
        }
      }

      barrier_wait(&g_barrier);
      if (tid == 0) {
        uint32_t nc = __atomic_load_n(&g_task_cnt_new, __ATOMIC_RELAXED);
        for (uint32_t i = 0; i < nc; i++)
          g_task_buf[i] = g_task_buf[stc + i];
        g_task_cnt = nc;
      }
      barrier_wait(&g_barrier);
    }

    /* WORK: each thread grabs tasks via atomic index, evaluates sequentially */
    {
      uint32_t tc = g_task_cnt;
      if (tid == 0) {
        __atomic_store_n(&g_task_cnt_new, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&g_work_idx, 0, __ATOMIC_RELAXED);
      }
      barrier_wait(&g_barrier);

      for (;;) {
        uint32_t i = __atomic_fetch_add(&g_work_idx, 1, __ATOMIC_RELAXED);
        if (i >= tc) break;
        struct Task my = g_task_buf[i];
        Term ea[MAX_ARGS];
        for (int j = 0; j < MAX_ARGS; j++) ea[j] = my.args[j];
        Term val = eval(my.fn, ea);
        resolve(my.ret, val, g_task_buf + tc, &g_task_cnt_new);
      }

      barrier_wait(&g_barrier);
      uint32_t nc = __atomic_load_n(&g_task_cnt_new, __ATOMIC_RELAXED);
      for (uint32_t i = (uint32_t)tid; i < nc; i += (uint32_t)g_num_threads)
        g_task_buf[i] = g_task_buf[tc + i];
      barrier_wait(&g_barrier);
      if (tid == 0)
        g_task_cnt = nc;
    }
  }
  return NULL;
}

/* ── Main ─────────────────────────────────────────────────────────── */

int main(void) {
  heap_cap = 1u << 31;
  heap_T = (uint8_t *)mmap(NULL, heap_cap,
                           PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  heap_L = (uint32_t *)mmap(NULL, (size_t)heap_cap * sizeof(uint32_t),
                            PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
  if (heap_T == MAP_FAILED || heap_L == MAP_FAILED) {
    fprintf(stderr, "heap mmap failed\n");
    return 1;
  }

  g_task_buf = (struct Task *)malloc(TASK_BUF_CAP * sizeof(struct Task));
  if (!g_task_buf) { fprintf(stderr, "alloc failed\n"); return 1; }

  char *nt = getenv("NUM_THREADS");
  g_num_threads = nt ? atoi(nt) : 1;
  if (g_num_threads < 1)           g_num_threads = 1;
  if (g_num_threads > MAX_THREADS) g_num_threads = MAX_THREADS;
  long ncpu  = sysconf(_SC_NPROCESSORS_ONLN);
  g_spin_only = (ncpu > 0 && g_num_threads <= ncpu);

  Term result;
  if (g_num_threads <= 1) {
    Term args[MAX_ARGS] = {0};
    result = eval(FN_MAIN, args);
  } else {
    g_task_buf[0] = make_task(FN_MAIN, ROOT_RET, 0, 0, 0, 0);
    g_task_cnt = 1;
    g_done = 0;
    g_heap_bump = 0;

    barrier_init(&g_barrier, g_num_threads);
    pthread_t threads[MAX_THREADS];
    for (int i = 1; i < g_num_threads; i++)
      pthread_create(&threads[i], NULL, worker, (void *)(intptr_t)i);
    worker((void *)0);
    for (int i = 1; i < g_num_threads; i++)
      pthread_join(threads[i], NULL);
    barrier_destroy(&g_barrier);
    result = g_result;
  }

  printf("%llu\n", (unsigned long long)result);
  return 0;
}
