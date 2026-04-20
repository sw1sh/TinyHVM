// term/sub.c — term_set_sub() / term_is_sub() / term_strip_sub()
//
// The SUB bit marks a heap slot's content as "already substituted."
// - On a VAR binder slot: SUB = placeholder, clear = resolved value.
// - On a DUP body slot:   SUB = sibling's clone/identity pre-written,
//                         clear = body not yet fired.
// Both readings are guarded by the slot's role at the call site.

static inline Term term_set_sub(Term t)   { return t | (1ULL << SUB_SHIFT); }
static inline int  term_is_sub(Term t)    { return (t >> SUB_SHIFT) & 1; }
static inline Term term_strip_sub(Term t) { return t & ~(1ULL << SUB_SHIFT); }
