// test_supgen.m — Test APP-SUP distribution for SupGen
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif

int main(void) {
    TinyHVM *ctx = thvm_init("metal");

    // Test 1: OP2-SUP — SUP(3,7) + 10 = SUP(13, 17)
    printf("=== Test 1: OP2-SUP ===\n");
    {
        Term s = thvm_sup(ctx, term_num_u32(3), term_num_u32(7));
        Term r = thvm_op2(ctx, 0, s, term_num_u32(10));  // ADD
        Term result = thvm_reduce(ctx, r);
        thvm_print_term(ctx, result);
        printf("\n");
        // Should print SUP(NUM(13), NUM(17))
    }

    // Test 2: APP-SUP (arg position) — double(SUP(3, 7)) = SUP(6, 14)
    printf("\n=== Test 2: APP-SUP (arg) ===\n");
    {
        // Build: double = λx. x + x
        Term var;
        Term body_x1 = term_new(TAG_VAR, 0, 0);  // placeholder
        Term lam = thvm_lam(ctx, &var, term_era());  // placeholder body
        // Now build body using var
        Term add = thvm_op2(ctx, 0, var, var);  // OP2(ADD, var, var)
        // Set body
        u64 lam_loc = term_val(lam);
        heap_set(ctx, lam_loc + 1, add);  // overwrite body

        Term input = thvm_sup(ctx, term_num_u32(3), term_num_u32(7));
        Term app = thvm_app(ctx, lam, input);

        printf("Before reduce: ");
        thvm_print_term(ctx, app);
        printf("\n");

        Term result = thvm_reduce(ctx, app);
        printf("After reduce:  ");
        thvm_print_term(ctx, result);
        printf("\n");

        // Extract branches
        if (term_tag(result) == TAG_SUP) {
            u64 sloc = term_val(result);
            Term b0 = thvm_reduce(ctx, heap_read(ctx, sloc));
            Term b1 = thvm_reduce(ctx, heap_read(ctx, sloc + 1));
            printf("Branch 0: ");
            thvm_print_term(ctx, b0);
            printf("\nBranch 1: ");
            thvm_print_term(ctx, b1);
            printf("\n");
        }
    }

    // Test 3: Simple APP-SUP — id(SUP(10, 20)) = SUP(10, 20)
    printf("\n=== Test 3: APP-SUP (identity) ===\n");
    {
        Term var;
        Term lam = thvm_lam(ctx, &var, var);  // λx. x
        Term input = thvm_sup(ctx, term_num_u32(10), term_num_u32(20));
        Term app = thvm_app(ctx, lam, input);
        Term result = thvm_reduce(ctx, app);
        printf("Result: ");
        thvm_print_term(ctx, result);
        printf("\n");
    }

    printf("\nInteractions: %llu\n", (unsigned long long)ctx->itrs);
    thvm_free(ctx);
    return 0;
}
