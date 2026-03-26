// schedule/_.c — Passthrough (fusion handled by rewrite rules)
// Range-based fusion happens in the trampoline via rewrite_apply():
// elementwise ops with matching output ranges fuse automatically.
// No separate scheduler pass needed.

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_reduce(ctx, t);
}
