// interact/_.c — interaction handler dispatch
// Split into sub-files for readability:
//   grad.c        — UOP_GRAD (backward chain rule)
//   tensor_ops.c  — ASSIGN, TODEVICE, WHERE, IFZ, LOG_PRINT, ew/reduce/view ops
//   combinators.c — APP, LAM, REF, SUP, DP0/DP1, OP2, VAR, etc.

// Read small metadata (axes, shapes, pad specs) without GPU flush.
#define META_READ(be, buf_id, out, bytes) \
    ((be)->buf_read_nosync ? \
     (be)->buf_read_nosync((buf_id), (out), (bytes)) : \
     (be)->buf_read((buf_id), (out), (bytes)))

// Forward declarations (defined in rewrite/_.c, included after interact)
static int is_view_op(u32 uop);
static int is_elementwise(u32 uop);

static Term thvm_era_payload(TinyHVM *ctx, Term item) {
    while (term_tag(item) == TAG_ERA) {
        u64 el = term_val(item);
        if (el == 0 || el >= ctx->heap_pos) return term_era();
        item = heap_read(ctx, el);
    }
    return item;
}

static Term thvm_make_active_era(TinyHVM *ctx, Term item) {
    item = thvm_era_payload(ctx, item);
    if (term_tag(item) == TAG_ERA && term_val(item) == 0) return term_era();
    u64 el = heap_alloc(ctx, 1);
    heap_set(ctx, el, item);
    return term_era_at(el);
}

static void thvm_spawn_detached_era(TinyHVM *ctx, Term item) {
    Term era = thvm_make_active_era(ctx, item);
    if (term_tag(era) == TAG_ERA && term_val(era) != 0) {
        u64 slot = heap_alloc(ctx, 1);
        heap_set(ctx, slot, era);
    }
}

static Term thvm_interact(TinyHVM *ctx, Term t) {
    // Return result directly — the trampoline handles TAG_TOP results
    // via `next = r; goto enter;` (no need to force-reduce here).
    // Return directly — trampoline handles TAG_TOP via goto enter
    #define RETURN_REDUCED(result) do { return (result); } while(0)
    // One interact() call performs exactly one local rewrite.
    // Continued reduction is the reducer trampoline's job.
    #define INET_RECURSE() do { return t; } while (0)
    // No GRAD_STEP — all GRAD sub-terms are placed on heap iteratively

    u32 tag;
inet_step:
    tag = term_tag(t);

    switch (tag) {
        case TAG_TOP: {
            u32 uop = term_ext(t);
            u64 loc = term_val(t);

            // === UOP_GRAD ===
#include "grad.c"

            // === Tensor ops (ASSIGN, ew, reduce, view, etc.) ===
#include "tensor_ops.c"
        } // end case TAG_TOP

        // === Combinators (APP, LAM, SUP, DP0/DP1, etc.) ===
#include "combinators.c"

        default:
            #undef INET_RECURSE
            return t;
    }
    #undef INET_RECURSE
    return t;
}

// ============================================================
// Trampoline frame tags (used by reduce/_.c)
// ============================================================
#define TAG_TOP1  0x7E  // TAG_TOP arg0 done, entering arg1
#define TAG_TOP2  0x7F  // TAG_TOP arg1 done, entering arg2 (GRAD/WHERE/IFZ)
#define TAG_OP2_1 0x7D  // OP2 arg0 done (non-SUP), entering arg1. EXT=opr, VAL=loc
#define TAG_EQL_1 0x7C  // EQL arg0 done (non-SUP), entering arg1. VAL=loc
#define TAG_MAT_1 0x7B  // APP fun=MAT, entering arg. VAL=app_loc

#define FRAME_CAP 65536
