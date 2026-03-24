// term/con.c — term constructors: era, var, ten, top
// TAG_NUM has been removed. Use thvm_scalar(ctx, val) for numeric constants.

static inline Term term_era(void)              { return term_new(TAG_ERA, 0,  0); }
static inline Term term_var(u64 loc)           { return term_new(TAG_VAR, 0,  loc); }
static inline Term term_ten(u32 tid, u32 dt)   { return term_new(TAG_TEN, dt, (u64)tid); }
static inline Term term_top(u32 uop, u64 loc)  { return term_new(TAG_TOP, uop, loc); }
