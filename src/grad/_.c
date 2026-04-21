// grad/_.c — minimal GRAD bundle accessor.
// Full gradient logic lives in src/interact/grad.c (UOP_GRAD rule)
// and src/inet/_.c (builders thvm_grad, thvm_grad_bundle).

Term thvm_grad_bundle_get(TinyHVM *ctx, Term bundle, u32 index) {
    bundle = thvm_eval(ctx, bundle);
    if (term_tag(bundle) != TAG_CTR) return index == 0 ? bundle : term_era();
    u64 loc = term_val(bundle);
    u32 arity = (u32)term_ext(bundle);
    if (index >= arity) return term_era();
    if (loc == 0 || loc + index >= ctx->heap_pos) return term_era();
    Term raw = heap_read(ctx, loc + index);
    return thvm_eval(ctx, raw);
}

u32 thvm_grad_bundle_count(TinyHVM *ctx, Term bundle) {
    bundle = thvm_eval(ctx, bundle);
    if (term_tag(bundle) == TAG_CTR) return (u32)term_ext(bundle);
    if (term_tag(bundle) == TAG_ERA && term_val(bundle) == 0) return 0;
    return 1;
}
