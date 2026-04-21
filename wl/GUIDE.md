# Wolfram Language Style Guide

This guide describes the WL style used in this repository. It is close to the style in `QuantumFramework` and `DiagrammaticComputation`, but the rules below are the source of truth.

## Indentation and spacing

- Indent multiline code with 4 spaces. Never indent with 2 spaces.
- Put spaces around infix operators and pattern tests: write `t_ ? NumericQ`, not `t_?NumericQ`.
- Prefer structural indentation over column alignment.
- In long forms such as `Which`, `Switch`, `Association`, `Table`, and option lists, put one semantic unit per line.
- Do not make variable lists tall by default. Prefer wider local lists grouped semantically.

## Definition layout

Prefer `Block` for local workspaces unless unique symbols are actually required.

Short definitions:

```wolfram
f[x_] := x + 1
```

Definitions with local state:

```wolfram
function[args] := Block[{
    x = ...,
    y
},
    body
]
```

If the left-hand side is long, split the arguments one per line:

```wolfram
f[
    arg1,
    arg2,
    arg3
] := Block[{
    local
},
    body
]
```

Body indentation inside `Block` should be one indentation level. Do not add extra indentation layers just because `Block` appears inside another construct.

Leave at least one empty line between top-level definitions.

Do not end `SetDelayed` definitions with a trailing semicolon.

## Naming

- Public symbols use descriptive CamelCase names.
- Internal helpers use lowerCamelCase names that begin with a lowercase letter.
- Do not use the `i...` prefix for internal helper names.
- Local names may be short, but they should still read clearly in context.

## Control flow

- Prefer `Block` over `Module` unless you specifically need unique symbols.
- Use `With` for fixed structural values.
- Prefer extracted helper functions over repeating nontrivial parsing, layout, or geometry logic.
- Keep imperative flow readable. If a function is stateful, make the local variables explicit in the `Block` variable list.

## File structure

When practical, organize files in this order:

1. Short file comment if needed.
2. `ClearAll[...]` or package declarations.
3. Small general helpers.
4. Domain-specific helpers.
5. Main entry points near the end.

## Comments

- Comment non-obvious behavior, quirks, or external format constraints.
- Do not narrate obvious code.
- Prefer short section comments over many tiny inline comments.

## Refactoring preference

- Preserve behavior while improving readability.
- Normalize code toward this style when touching it.
- Remove duplicated helper logic before adding new abstraction.
- If a refactor is only cosmetic, it should still make the code more uniform and easier to scan.

