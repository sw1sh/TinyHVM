// grad/_.c — minimal GRAD bundle accessor.
// Full gradient logic lives in src/interact/grad.c (UOP_GRAD rule)
// and src/inet/_.c (builders thvm_grad, thvm_grad_bundle).

// Cache of the most recently evaluated bundle (term → eval'd CTR) so
// the common pattern of calling thvm_grad_bundle_get in a loop over
// indices avoids re-evaluating the bundle each time.  Re-evaluation of
// an already-evaluated CTR corrupts sibling fields because the phase-2
// sweep's reachability walk marks them unreachable the second time
// around and ERAs them out.  Cache is keyed on (ctx, bundle) since
// thvm_free/thvm_init can reuse heap positions, causing term-value
// collisions across contexts.
static TinyHVM *g_grad_bundle_cache_ctx = NULL;
static Term     g_grad_bundle_cache_in  = 0;
static Term     g_grad_bundle_cache_out = 0;

void thvm_grad_bundle_cache_reset(TinyHVM *ctx) {
    if (ctx == NULL || g_grad_bundle_cache_ctx == ctx) {
        g_grad_bundle_cache_ctx = NULL;
        g_grad_bundle_cache_in  = 0;
        g_grad_bundle_cache_out = 0;
    }
}

Term thvm_grad_bundle_get(TinyHVM *ctx, Term bundle, u32 index) {
    Term ev;
    if (ctx == g_grad_bundle_cache_ctx && bundle == g_grad_bundle_cache_in &&
        g_grad_bundle_cache_out != 0) {
        ev = g_grad_bundle_cache_out;
    } else {
        ev = thvm_eval(ctx, bundle);
        g_grad_bundle_cache_ctx = ctx;
        g_grad_bundle_cache_in  = bundle;
        g_grad_bundle_cache_out = ev;
    }
    if (term_tag(ev) != TAG_CTR) return index == 0 ? ev : term_era();
    u64 loc = term_val(ev);
    u32 arity = (u32)term_ext(ev);
    if (index >= arity) return term_era();
    if (loc == 0 || loc + index >= ctx->heap_pos) return term_era();
    Term raw = heap_read(ctx, loc + index);
    u8 tag = term_tag(raw);
    if (tag == TAG_TEN || tag == TAG_NUM || tag == TAG_ERA || tag == TAG_ANY)
        return raw;
    return thvm_eval(ctx, raw);
}

u32 thvm_grad_bundle_count(TinyHVM *ctx, Term bundle) {
    bundle = thvm_eval(ctx, bundle);
    if (term_tag(bundle) == TAG_CTR) return (u32)term_ext(bundle);
    if (term_tag(bundle) == TAG_ERA && term_val(bundle) == 0) return 0;
    return 1;
}
