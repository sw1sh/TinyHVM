// term/ext.c — term_ext(): extract ext field
static inline u32 term_ext(Term t) { return (u32)((t >> EXT_SHIFT) & EXT_MASK); }
