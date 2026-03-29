// Interaction Calculus of Constructions
// =====================================

// Type
// ----

data Term
  = (Lam bod)
  | (App fun arg)
  | (Val bod)
  | (Ann val typ)
  | (Var idx)

// Evaluation
// ----------

(APP fun arg) = match fun {
  Lam: (fun.bod arg)
  Val: (Val λx(APP (fun.bod (Lam λ$k(x))) (ANN $k arg)))
  //Fix: (APP (fun.bod (Fix λx(fun.bod x))) arg)
  var: (App var arg)
}

(ANN val typ) = match typ {
  Lam: (Lam λx(ANN (APP val $k) (typ.bod (Val λ$k(x)))))
  Val: (typ.bod val)
  var: (Ann val var)
}

// Equality
// --------

(Equal a b sym dep) = match a {
  Lam: (Equal (a.bod (Var dep)) (APP b (Var dep)) sym (+ dep 1))
  App: match b {
    App: (& (Equal a.fun b.fun sym dep) (Equal a.arg b.arg sym dep))
    var: 0
  }
  Val: (Equal (a.bod (Var dep)) (ANN (Var dep) b) sym (+ dep 1))
  Ann: (Equal a.val b sym dep)
  Var: match b {
    Var: (== a.idx b.idx)
    var: match sym {
      0: (Equal b a 1 dep)
      +: 0
    }
  }
}

// Checking
// --------

(Check a dep) = match a {
  Lam: (Check (a.bod (Var dep)) (+ dep 1))
  App: match a.fun {
    Ann: 0
    fun: (& (Check fun dep) (Check a.arg dep))
  }
  Val: (Check (a.bod (Var dep)) (+ dep 1))
  Ann: match a.val {
    Ann: (& (Equal a.val.typ a.typ 0 dep) (Check a.val dep))
    var: (Check var dep)
  }
  Var: 1
}

// Encodings
// ---------

// Set ::= free-var
Set = (Var (- 0 1))

// Any ::= θv v
Any = (Val λx(x))

// (A -> B) ::= θf λx {(f {x: A}): B}
(Arr A B) = (Val λf(Lam λx(ANN (APP f (ANN x A)) B)))

// (Π(x: A). B[x]) ::= θf λx {(f {x: A}): (B x)}
(All A B) = (Val λf(Lam λx(ANN (APP f (ANN x A)) (B x))))

// (Πf(x: A). B[x]) ::= θf λx {(f {x: A}): (B x f)}
(Ind A B) = (Val λf(Lam λx(ANN (APP f (ANN x A)) (B f x))))

// Stringification
// ---------------

(Concat SNil         ys) = ys
(Concat (SCons x xs) ys) = (SCons x (Concat xs ys))

(Join LNil)         = ""
(Join (LCons x xs)) = (Concat x (Join xs))

(U60.show n) = (U60.show.go n SNil)
(U60.show.go n res) = match k = (< n 10) {
  0: (U60.show.go (/ n 10) (SCons (+ '0' (% n 10)) res))
  +: (SCons (+ '0' n) res)
}

(Show term dep) = match term {
  Lam: (Join ["λx" (U60.show dep) " " (Show (term.bod (Var dep)) (+ dep 1))])
  App: (Join ["(" (Show term.fun dep) " " (Show term.arg dep) ")"])
  Val: (Join ["θx" (U60.show dep) " " (Show (term.bod (Var dep)) (+ dep 1))])
  Ann: (Join ["{" (Show term.val dep) " : " (Show term.typ dep) "}"])
  Var: match k=(== term.idx (- 0 1)) { 0: (Join ["x" (U60.show term.idx)]); +: "*" }
}

// Tests
// -----

// μNat. ∀(P: *) (Nat -> P) -> P -> P
Nat = (Fix λNat(All Set λP(All (Arr Nat P) λt (All P λf(P)))))

// ...
zero = (Lam λP(Lam λs(Lam λz(z))))

// ...
succ = (Lam λn(Lam λP(Lam λs(Lam λz(APP s n)))))

// ∀(P: *) ∀(t: P) ∀(f: P) P
Bool = (All Set λP(All P λt (All P λf(P))))

// λP λt λf t
true = (Lam λP(Lam λt(Lam λf(t))))

// λP λt λf f
false = (Lam λP(Lam λt(Lam λf(f))))

// λb (b false true)
not = (Lam λb(APP (APP (APP b Bool) false) true))

// λP λs λz z
zero = (Lam λP(Lam λs(Lam λz(z))))

// λP λs λz (s λP λs λz (z))
one = (Lam λP(Lam λs(Lam λz(APP s (Lam λP(Lam λs(Lam λz(z))))))))

// λn λP λs λz (s n)
succ = (Lam λn(Lam λP(Lam λs(Lam λz(APP s n)))))

// ∀(P: *) ∀(x: P) P
Unit = (All Set λP(All P λu(P)))

// λP λx x
unit = (Lam λP(Lam λx(x)))

// ∀(A: *) ∀(B: *) ∀(A -> B) ∀(B -> A) ∀A A
Test = (All Set λA(All Set λB(Arr (Arr A B) (Arr (Arr B A) (Arr A A)))))

// λA λB λab λba λa a
test = (Lam λA(Lam λB(Lam λab(Lam λba(Lam λa(APP ba (APP ab a))))))) 

// Main
Main = (Check (ANN test Test) 0)

// (λx x) : (Nat -> Nat)
// -----------------------------------------------------------------------------
// λn λP λt λf {{(n P λp{{(t {{p: Nat}: Nat}): P}: {P: *}} {{f: P}: P}) : P}: P}
// -----------------------------------------------------------------------------
// would expand {p: Nat} infinitely...
