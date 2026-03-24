// term/val.c — term_val(): extract val field
static inline u64 term_val(Term t) { return t & VAL_MASK; }
