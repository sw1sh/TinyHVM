# Chapter 2: Combinators — LAM, APP, ERA

Interaction nets have just a few node types. In this chapter we meet the three most fundamental: **lambda** (constructor), **application** (destructor), and **eraser** (garbage collector).

---

## Lambda: `λx. body`

A lambda creates a **binding** — it introduces a variable and a body that can reference it.

```wolfram
TInit[];
{lam, var} = TLam[x |-> x];     (* identity: λx. x *)
```

`TLam` returns **two** terms:
- `lam` — the lambda node itself (TAG=LAM, VAL points to heap location)
- `var` — the variable (TAG=VAR, also pointing to the same heap location)

The body function `x |-> x` receives `var` as `x` and returns it as the body. Internally, two heap words are allocated:

```
heap[loc]     = VAR (with SUB bit set — marks it as the binding site)
heap[loc + 1] = body (the term returned by the body function)
```

Let's inspect:

```wolfram
TTermTag[lam]         (* "Lam" *)
TINetGraph[lam]       (* a single LAM node — body is its own VAR *)
```

A more interesting lambda — one that uses its variable:

```wolfram
TInit[];
double = TLam[x |-> TOp2["Add", x, x]];
TINetGraph[double]
```

Now the graph shows the LAM node connected to an OP2 node, with the variable wired to both inputs.

---

## Application: `(f x)`

Application creates a node that connects a function to an argument:

```wolfram
TInit[];
id = TLam[x |-> x];
app = TApp[id, TNum[42]];
TINetGraph[app]
```

Before reduction, you see: APP connected to LAM (the function) and NUM (the argument). The APP node allocates 2 heap words: `heap[loc] = fun`, `heap[loc+1] = arg`.

Now reduce:

```wolfram
result = TReduce[app]
TGet[result]         (* {42.} — the number passes through *)
```

The APP-LAM interaction (beta reduction) substituted the argument into the body.

---

## Eraser: ERA

The eraser appears when a value is **not needed**. If you create a lambda that ignores its variable:

```wolfram
TInit[];
konst = TLam[x |-> TNum[7]];     (* λx. 7 — x is unused *)
TINetGraph[konst]
```

The variable `x` is connected to ERA (automatically inserted by TLam when the variable is unused). ERA propagates through any structure it meets, erasing everything.

```wolfram
result = TReduce[TApp[konst, TNum[999]]];
TGet[result]         (* {7.} — the 999 was erased *)
```

---

## Heap Structure in Detail

Let's watch exactly what happens on the heap:

```wolfram
TInit[];
h0 = THeapSnapshot[];

lam = TLam[x |-> TOp2["Add", x, TNum[1]]];
h1 = THeapSnapshot[];

app = TApp[lam, TNum[41]];
h2 = THeapSnapshot[];

{h0["HeapSize"], h1["HeapSize"], h2["HeapSize"]}
```

Each operation grows the heap. Read the raw words:

```wolfram
(* Read the first few heap locations *)
Table[THeapRead[i], {i, 0, h2["HeapSize"] - 1}]
```

You'll see the TAG/EXT/VAL triples for every allocated word — VAR nodes, body terms, the OP2 node, NUM nodes, and the APP node.

---

## Visualizing Before and After

```wolfram
TInit[];
term = TApp[TLam[x |-> x], TNum[42]];

(* Before: APP connected to LAM connected to NUM *)
TINetGraph[term]

(* After: just the NUM *)
TINetGraph[TReduce[term]]
```

The interaction net went from 3 nodes + edges to 1 node. That's what "reduction" means — nodes interact and cancel.

---

## Summary

| Node | Tag | Heap Words | Purpose |
|------|-----|-----------|---------|
| LAM | 1 | 2 (var + body) | Introduces a binding |
| APP | 0 | 2 (fun + arg) | Applies function to argument |
| ERA | 6 | 0 (leaf) | Erases unused values |

- `TLam[x |-> body]` returns `{lam, var}` — the lambda and its bound variable
- `TApp[fun, arg]` creates an application node
- ERA appears automatically for unused variables
- APP-LAM is **beta reduction** — the fundamental computation step
