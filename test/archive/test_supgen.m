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

    // ============================================================
    // Phase 3: Collapse
    // ============================================================

    // Test 7: thvm_collapse — flatten nested SUPs
    printf("\n=== Test 7: Collapse ===\n");
    {
        u32 l0 = thvm_fresh_label(ctx);
        u32 l1 = thvm_fresh_label(ctx);
        Term inner = thvm_sup(ctx, l1, term_num_u32(10), term_num_u32(20));
        Term outer = thvm_sup(ctx, l0, term_num_u32(1), inner);
        CollapseResult cr = thvm_collapse(ctx, outer);
        printf("Collapse count: %u\n", cr.count);
        for (u32 i = 0; i < cr.count; i++) {
            printf("  [%u]: ", i); thvm_print_term(ctx, cr.terms[i]); printf("\n");
        }
        // Should be 3 leaves: NUM(1), NUM(10), NUM(20)
        assert(cr.count == 3);
        assert(term_as_u32(cr.terms[0]) == 1);
        assert(term_as_u32(cr.terms[1]) == 10);
        assert(term_as_u32(cr.terms[2]) == 20);
        thvm_collapse_free(&cr);
    }

    // Test 7b: collapse with computation inside SUPs
    printf("\n=== Test 7b: Collapse with OP2 ===\n");
    {
        u32 lab = thvm_fresh_label(ctx);
        Term s = thvm_sup(ctx, lab,
            thvm_op2(ctx, 0, term_num_u32(3), term_num_u32(4)),   // 3+4=7
            thvm_op2(ctx, 2, term_num_u32(5), term_num_u32(6)));  // 5*6=30
        CollapseResult cr = thvm_collapse(ctx, s);
        printf("Collapse count: %u\n", cr.count);
        for (u32 i = 0; i < cr.count; i++) {
            printf("  [%u]: ", i); thvm_print_term(ctx, cr.terms[i]); printf("\n");
        }
        assert(cr.count == 2);
        assert(term_as_u32(cr.terms[0]) == 7);
        assert(term_as_u32(cr.terms[1]) == 30);
        thvm_collapse_free(&cr);
    }

    // ============================================================
    // Phase 4: BRI / ANN
    // ============================================================

    // Test 8: BRI — theta abstraction (same beta rule as LAM)
    printf("\n=== Test 8: BRI (theta) ===\n");
    {
        Term var;
        Term bri = thvm_bri(ctx, &var, term_era());
        Term body = thvm_op2(ctx, 0, var, term_num_u32(100));
        heap_set(ctx, term_val(bri) + 1, body);
        Term app = thvm_app(ctx, bri, term_num_u32(5));
        Term result = thvm_reduce(ctx, app);
        printf("BRI(5)+100 = "); thvm_print_term(ctx, result); printf("\n");
        assert(term_tag(result) == TAG_NUM && term_as_u32(result) == 105);
    }

    // Test 9: ANN — transparent (strip annotation)
    printf("\n=== Test 9: ANN (transparent) ===\n");
    {
        Term ann = thvm_ann(ctx, term_num_u32(42), term_num_u32(0));
        Term result = thvm_reduce(ctx, ann);
        printf("ANN result = "); thvm_print_term(ctx, result); printf("\n");
        assert(term_tag(result) == TAG_NUM && term_as_u32(result) == 42);
    }

    // Test 10: DUP-BRI commutation
    printf("\n=== Test 10: DUP-BRI ===\n");
    {
        // θx.x  (identity bridge)
        Term var;
        Term bri = thvm_bri(ctx, &var, term_era());
        heap_set(ctx, term_val(bri) + 1, var);  // body = var
        u32 lab = thvm_fresh_label(ctx);
        Term dp0, dp1;
        thvm_dup(ctx, lab, bri, &dp0, &dp1);
        Term r0 = thvm_reduce(ctx, thvm_app(ctx, dp0, term_num_u32(10)));
        Term r1 = thvm_reduce(ctx, thvm_app(ctx, dp1, term_num_u32(20)));
        printf("DUP-BRI dp0(10)="); thvm_print_term(ctx, r0);
        printf(", dp1(20)="); thvm_print_term(ctx, r1); printf("\n");
        assert(term_tag(r0) == TAG_NUM && term_as_u32(r0) == 10);
        assert(term_tag(r1) == TAG_NUM && term_as_u32(r1) == 20);
    }

    // Test 11: DUP-ANN — dup through annotation
    printf("\n=== Test 11: DUP-ANN ===\n");
    {
        Term ann = thvm_ann(ctx, term_num_u32(55), term_num_u32(0));
        u32 lab = thvm_fresh_label(ctx);
        Term dp0, dp1;
        thvm_dup(ctx, lab, ann, &dp0, &dp1);
        Term r0 = thvm_reduce(ctx, dp0);
        Term r1 = thvm_reduce(ctx, dp1);
        printf("DUP-ANN dp0="); thvm_print_term(ctx, r0);
        printf(", dp1="); thvm_print_term(ctx, r1); printf("\n");
        // Both should reduce to 55 (ANN is transparent, NUM copies through DUP)
        assert(term_tag(r0) == TAG_NUM && term_as_u32(r0) == 55);
        assert(term_tag(r1) == TAG_NUM && term_as_u32(r1) == 55);
    }

    // ============================================================
    // Phase 5: DSU / DDU / INC
    // ============================================================

    // Test 12: DSU — dynamic superposition
    printf("\n=== Test 12: DSU ===\n");
    {
        Term dsu = thvm_dsu(ctx, term_num_u32(42), term_num_u32(1), term_num_u32(2));
        Term result = thvm_reduce(ctx, dsu);
        printf("DSU result = "); thvm_print_term(ctx, result); printf("\n");
        assert(term_tag(result) == TAG_SUP && term_ext(result) == 42);
        u64 sloc = term_val(result);
        Term a = thvm_reduce(ctx, heap_read(ctx, sloc));
        Term b = thvm_reduce(ctx, heap_read(ctx, sloc + 1));
        assert(term_as_u32(a) == 1 && term_as_u32(b) == 2);
    }

    // Test 12b: DSU + collapse
    printf("\n=== Test 12b: DSU + collapse ===\n");
    {
        Term dsu = thvm_dsu(ctx, term_num_u32(7),
            thvm_op2(ctx, 0, term_num_u32(10), term_num_u32(1)),  // 11
            thvm_op2(ctx, 0, term_num_u32(20), term_num_u32(2))); // 22
        CollapseResult cr = thvm_collapse(ctx, dsu);
        printf("DSU+collapse count: %u\n", cr.count);
        for (u32 i = 0; i < cr.count; i++) {
            printf("  [%u]: ", i); thvm_print_term(ctx, cr.terms[i]); printf("\n");
        }
        assert(cr.count == 2);
        assert(term_as_u32(cr.terms[0]) == 11);
        assert(term_as_u32(cr.terms[1]) == 22);
        thvm_collapse_free(&cr);
    }

    // Test 13: DDU — dynamic dup
    // DDU applies bod to both copies: APP(APP(bod, dp0), dp1)
    // bod must be a 2-arg curried function
    printf("\n=== Test 13: DDU ===\n");
    {
        // bod = λa.λb. a + b
        Term var_a, var_b;
        Term lam_b = thvm_lam(ctx, &var_b, term_era());  // placeholder
        Term lam_a = thvm_lam(ctx, &var_a, lam_b);
        Term add = thvm_op2(ctx, 0, var_a, var_b);       // a + b
        heap_set(ctx, term_val(lam_b) + 1, add);         // body of inner lam
        // DDU(7, 99, λa.λb.a+b) → APP(APP(λa.λb.a+b, dp0), dp1) → dp0+dp1 → 99+99
        Term ddu = thvm_ddu(ctx, term_num_u32(7), term_num_u32(99), lam_a);
        Term result = thvm_reduce(ctx, ddu);
        printf("DDU result = "); thvm_print_term(ctx, result); printf("\n");
        assert(term_tag(result) == TAG_NUM && term_as_u32(result) == 198);
    }

    // Test 14: INC — transparent unwrap
    printf("\n=== Test 14: INC ===\n");
    {
        Term inc = thvm_inc(ctx, term_num_u32(77));
        Term result = thvm_reduce(ctx, inc);
        printf("INC result = "); thvm_print_term(ctx, result); printf("\n");
        assert(term_tag(result) == TAG_NUM && term_as_u32(result) == 77);
    }

    // Test 14b: INC inside SUP — collapse should still find leaves
    printf("\n=== Test 14b: INC + collapse ===\n");
    {
        u32 lab = thvm_fresh_label(ctx);
        Term s = thvm_sup(ctx, lab,
            term_num_u32(1),
            thvm_inc(ctx, term_num_u32(2)));
        CollapseResult cr = thvm_collapse(ctx, s);
        printf("INC+collapse count: %u\n", cr.count);
        for (u32 i = 0; i < cr.count; i++) {
            printf("  [%u]: ", i); thvm_print_term(ctx, cr.terms[i]); printf("\n");
        }
        assert(cr.count == 2);
        assert(term_as_u32(cr.terms[0]) == 1);
        assert(term_as_u32(cr.terms[1]) == 2);
        thvm_collapse_free(&cr);
    }

    printf("\nAll tests passed! Interactions: %llu\n", (unsigned long long)ctx->itrs);
    thvm_free(ctx);
    return 0;
}
