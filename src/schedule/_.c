// schedule/_.c — Single reduce with lazy compute TAG_TOPs
// Compute TAG_TOPs are WNF (trampoline skips them).
// GRAD fires on TAG_TOP y → creates backward TAG_TOPs.
// ASSIGN blocks (src is TAG_TOP, not TAG_TEN).
// After reduce: heap has TAG_TOP DAG wired to blocked ASSIGNs.
// TODO: scheduler walks TAG_TOPs, rewrites with UOP_FUSING, fires ASSIGNs.

Term thvm_eval(TinyHVM *ctx, Term t) {
    // For now: just reduce. Compute TAG_TOPs stay lazy.
    // ASSIGN/GRAD/IFZ fire normally.
    return thvm_reduce(ctx, t);
}

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_eval(ctx, t);
}
