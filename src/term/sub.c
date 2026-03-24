// term/sub.c — term_set_sub() / term_is_sub()
// The SUB bit marks a heap slot as "already substituted" (used by TAG_VAR).

static inline Term term_set_sub(Term t) { return t | (1ULL << SUB_SHIFT); }
static inline int  term_is_sub(Term t)  { return (t >> SUB_SHIFT) & 1; }
