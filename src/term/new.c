// term/new.c — term_new(): pack tag + ext + val into a 64-bit term word
static inline Term term_new(u32 tag, u32 ext, u64 val) {
  return ((u64)(tag & TAG_MASK) << TAG_SHIFT)
       | ((u64)(ext & EXT_MASK) << EXT_SHIFT)
       | (val & VAL_MASK);
}
