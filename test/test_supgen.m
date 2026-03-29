// test_supgen.m — Test labeled SUP/DUP, APP-SUP, OP2-SUP distribution
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif

int main(void) {
    TinyHVM *ctx = thvm_init("cpu");

    // Test 1: OP2-SUP — SUP(3,7) + 10 = SUP(13, 17)
    printf("=== Test 1: OP2-SUP ===\n");
    {
        u32 lab = thvm_fresh_label(ctx);
        Term s = thvm_sup(ctx, lab, term_num_u32(3), term_num_u32(7));
        Term r = thvm_op2(ctx, 0, s, term_num_u32(10));  // ADD
        Term result = thvm_reduce(ctx, r);
        thvm_print_term(ctx, result);
        printf("\n");
        // Should print SUP(NUM(13), NUM(17))
    }

    // Test 2: APP-SUP — double(SUP(3, 7)) = SUP(6, 14)
    // HVM4 approach: beta reduces first, then OP2-SUP + DUP-SUP(same label) handles it
    printf("\n=== Test 2: APP-SUP (double) ===\n");
    {
        // Build: double = λx. x + x
        Term var;
        Term lam = thvm_lam(ctx, &var, term_era());  // placeholder body
        Term add = thvm_op2(ctx, 0, var, var);        // OP2(ADD, var, var)
        u64 lam_loc = term_val(lam);
        heap_set(ctx, lam_loc + 1, add);              // overwrite body

        u32 lab = thvm_fresh_label(ctx);
        Term input = thvm_sup(ctx, lab, term_num_u32(3), term_num_u32(7));
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
            // Should be 6 and 14
        }
    }

    // Test 3: APP-SUP (fun position) — SUP(id, double)(5) = SUP(5, 10)
    printf("\n=== Test 3: APP-SUP (fun) ===\n");
    {
        // id = λx. x
        Term var_id;
        Term id = thvm_lam(ctx, &var_id, term_era());  // placeholder body
        heap_set(ctx, term_val(id) + 1, var_id);        // overwrite body = var

        // double = λx. x + x
        Term var_dbl;
        Term dbl = thvm_lam(ctx, &var_dbl, term_era());
        Term add = thvm_op2(ctx, 0, var_dbl, var_dbl);
        heap_set(ctx, term_val(dbl) + 1, add);

        u32 lab = thvm_fresh_label(ctx);
        Term fun_sup = thvm_sup(ctx, lab, id, dbl);
        Term app = thvm_app(ctx, fun_sup, term_num_u32(5));

        Term result = thvm_reduce(ctx, app);
        printf("Result: ");
        thvm_print_term(ctx, result);
        printf("\n");

        if (term_tag(result) == TAG_SUP) {
            u64 sloc = term_val(result);
            Term b0 = thvm_reduce(ctx, heap_read(ctx, sloc));
            Term b1 = thvm_reduce(ctx, heap_read(ctx, sloc + 1));
            printf("Branch 0: ");
            thvm_print_term(ctx, b0);
            printf("\nBranch 1: ");
            thvm_print_term(ctx, b1);
            printf("\n");
            // Should be 5 and 10
        }
    }

    // Test 4: DUP-SUP annihilation (same label)
    printf("\n=== Test 4: DUP-SUP annihilation ===\n");
    {
        u32 lab = thvm_fresh_label(ctx);
        Term s = thvm_sup(ctx, lab, term_num_u32(42), term_num_u32(99));
        Term dp0, dp1;
        thvm_dup(ctx, lab, s, &dp0, &dp1);
        Term r0 = thvm_reduce(ctx, dp0);
        Term r1 = thvm_reduce(ctx, dp1);
        printf("DP0: "); thvm_print_term(ctx, r0);
        printf("\nDP1: "); thvm_print_term(ctx, r1);
        printf("\n");
        // Should be 42 and 99
    }

    // Test 5: DUP-SUP commutation (different labels)
    printf("\n=== Test 5: DUP-SUP commutation ===\n");
    {
        u32 lab_sup = thvm_fresh_label(ctx);
        u32 lab_dup = thvm_fresh_label(ctx);
        Term s = thvm_sup(ctx, lab_sup, term_num_u32(10), term_num_u32(20));
        Term dp0, dp1;
        thvm_dup(ctx, lab_dup, s, &dp0, &dp1);
        // Each projection should be a SUP with lab_sup
        Term r0 = thvm_reduce(ctx, dp0);
        Term r1 = thvm_reduce(ctx, dp1);
        printf("DP0: "); thvm_print_term(ctx, r0);
        printf("\nDP1: "); thvm_print_term(ctx, r1);
        printf("\n");
        // DP0 = SUP_lab_sup(DP0_lab_dup(10), DP0_lab_dup(20)) → SUP(10, 20)
        // DP1 = SUP_lab_sup(DP1_lab_dup(10), DP1_lab_dup(20)) → SUP(10, 20)
        // Both should be SUP(10, 20) since atoms just copy through DUP
        if (term_tag(r0) == TAG_SUP) {
            u64 sloc = term_val(r0);
            Term a = thvm_reduce(ctx, heap_read(ctx, sloc));
            Term b = thvm_reduce(ctx, heap_read(ctx, sloc + 1));
            printf("  DP0 branches: ");
            thvm_print_term(ctx, a); printf(", ");
            thvm_print_term(ctx, b); printf("\n");
        }
    }

    // Test 6: Nested SUPs with different labels (independent dimensions)
    printf("\n=== Test 6: Nested SUPs ===\n");
    {
        u32 lab0 = thvm_fresh_label(ctx);
        u32 lab1 = thvm_fresh_label(ctx);
        // x ∈ {1, 2}, y ∈ {10, 20}
        Term x = thvm_sup(ctx, lab0, term_num_u32(1), term_num_u32(2));
        Term y = thvm_sup(ctx, lab1, term_num_u32(10), term_num_u32(20));
        // x + y should give SUP over 4 values: {11, 21, 12, 22}
        Term sum = thvm_op2(ctx, 0, x, y);
        Term result = thvm_reduce(ctx, sum);
        printf("x + y = ");
        thvm_print_term(ctx, result);
        printf("\n");
    }

    printf("\nInteractions: %llu\n", (unsigned long long)ctx->itrs);
    thvm_free(ctx);
    return 0;
}
