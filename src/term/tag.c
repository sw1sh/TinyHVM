// term/tag.c — term_tag(): extract tag field
static inline u32 term_tag(Term t) { return (u32)((t >> TAG_SHIFT) & TAG_MASK); }
