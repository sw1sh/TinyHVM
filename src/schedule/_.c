// schedule/_.c — Lazy graph compiler
// The inet IS the lazy graph. View ops stay as TAG_TOP nodes.
// The trampoline's fusion walks through them, composing views
// into leaf index expressions. No separate scheduler pass needed —
// the inet interactions handle everything.

Term thvm_schedule(TinyHVM *ctx, Term t) {
    return thvm_reduce(ctx, t);
}
