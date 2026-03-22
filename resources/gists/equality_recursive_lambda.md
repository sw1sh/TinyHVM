A tricky problem in Type Theory has a short satisfactory solution. To check if two potentially recursive expressions are equal, first we take the weak head normal form of both sides, without unrolling fixed points. Then, we do:

```haskell
(F . G) == (G . F)
------------------- Fix=Fix case: apply T6's Theorem
μk(F k) == μk(G k)

(F Y) == Y
------------- Fix=Val case: apply T6's Theorem
μk(F k) == Y

X == (G μk(G k))
---------------- Val=Fix case: unroll the right side
X == μk(G k)

-- Otherwise: just do case analysis and recurse, as usual. 
```


This might give false negatives (we believe?) but never false positives. It is elegant, efficient and covers most practical cases. You can use it to check equality on arbitrary type-level recursive expressions, which aren't allowed in conventional proof assistants. The first 2 cases use the theorem discovered by T6. Explanation and proofs in our Discord:

https://discord.com/channels/912426566838013994/1007033260662071306/1328426881111953420