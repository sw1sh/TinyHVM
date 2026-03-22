module Main where

import Data.List (nub)

-- Pattern Matching Flattener Algorithm
-- ====================================
-- 
-- This algorithm converts pattern matching expressions with multiple 
-- scrutinees into nested case trees with single scrutinees.
--
-- Example transformation:
--   match x y                =>    match x:
--   | Z     Z         = X0          case Z:
--   | (S x) Z         = X1            match y:
--   | Z     (S y)     = X2              case Z: X0
--   | (S x) (S Z)     = X3              case S y: X2
--   | (S x) (S (S z)) = X4            case S x:
--                                       match y:
--                                         case Z: X1
--                                         case S y_:
--                                           match y_:
--                                             case Z: X3
--                                             case S z: X4
--
-- The algorithm works column-by-column from left to right:
-- 1. Take the first scrutinee (x) and its column of patterns [Z, S x, Z, S x]
-- 2. Group rules by constructor (Z and S)
-- 3. For each constructor, extract its arguments and continue with remaining columns
-- 4. Handle variable patterns as default cases

-- Data Types
-- ----------

-- Expr represents expressions in our pattern matching language
-- Examples:
--   Ctr "Z" []           represents the zero constructor: Z
--   Ctr "S" [Var "x"]    represents successor: (S x)
--   Var "x"              represents a variable: x
--   Mat [e1,e2] rules    represents a pattern match on expressions e1,e2
data Expr = Ctr String [Expr] | Var String | Mat [Expr] [([Expr], Expr)] deriving (Show, Eq)

-- Utilities
-- ---------

-- Check if expression is a variable
-- Examples:
--   isVar (Var "x") = True
--   isVar (Ctr "Z" []) = False
isVar :: Expr -> Bool
isVar (Var _) = True
isVar _       = False

-- Extract constructor names from column
-- Example:
--   getCtrs [Ctr "Z" [], Var "x", Ctr "S" [Var "y"], Ctr "Z" []] 
--   = ["Z", "S"]
getCtrs :: [Expr] -> [String]
getCtrs col = nub [c | Ctr c _ <- col]

-- Get arity of constructor in column
-- Example:
--   getArity "S" [Ctr "Z" [], Ctr "S" [Var "x"], Var "y"]
--   = 1 (because S takes 1 argument)
getArity :: String -> [Expr] -> Int
getArity c col = maximum (0 : [length args | Ctr c' args <- col, c == c'])

-- Flattener
-- ---------

-- Convert a Mat expression to a flattened Mat (case tree representation)
-- The algorithm processes one scrutinee at a time, from left to right
-- Example transformation:
--   Mat [x, y] [([Z, Z], X0), ([S x, Z], X1), ([Z, S y], X2), ([S x, S y], X3)]
--   becomes a nested case tree on x then y
flatten :: Expr -> Expr
flatten e = flattenWithDepth 0 e

flattenWithDepth :: Int -> Expr -> Expr
flattenWithDepth d (Mat (x:xs) rules) = processColWithDepth d x (splitCol rules) xs
flattenWithDepth d (Mat [] [([], rhs)]) = rhs  -- No patterns left, return result
flattenWithDepth d (Mat [] _) = error "Incomplete pattern match"
flattenWithDepth d e = e  -- Non-Mat expressions are already flat

-- Split rules into first column of patterns and remaining rules
-- Example:
--   splitCol [([Z, Z], X0), ([S x, Z], X1), ([Z, S y], X2)]
--   = ([Z, S x, Z], [([Z], X0), ([Z], X1), ([S y], X2)])
--   First element: patterns from first column [Z, S x, Z]
--   Second element: remaining patterns and results
splitCol :: [([Expr], Expr)] -> ([Expr], [([Expr], Expr)])
splitCol rules = unzip [(p, (ps, rhs)) | (p:ps, rhs) <- rules]

-- Process a single column of patterns
-- This is the core of the algorithm: it decides how to handle the current column
-- Example with col = [Z, S x, Z] (mixed constructors):
--   Creates a Mat that switches on 'scrut' with cases for Z and S
processCol :: Expr -> ([Expr], [([Expr], Expr)]) -> [Expr] -> Expr
processCol = processColWithDepth 0

processColWithDepth :: Int -> Expr -> ([Expr], [([Expr], Expr)]) -> [Expr] -> Expr
processColWithDepth d scrut (col, rules) scruts
  | all isVar col = flattenWithDepth d (Mat scruts rules)       -- Skip variable-only columns
  | otherwise     = Mat [scrut] (ctrRules++varRules) -- Build case tree as Mat
  where ctrRules = makeCtrRulesWithDepth d col rules scruts
        varRules = makeVarRulesWithDepth d col rules scruts

-- Build constructor rules for case tree
-- For each constructor in the column, create a case branch
-- Example: if col has Z and S constructors, create two branches
makeCtrRules :: [Expr] -> [([Expr], Expr)] -> [Expr] -> [([Expr], Expr)]
makeCtrRules = makeCtrRulesWithDepth 0

makeCtrRulesWithDepth :: Int -> [Expr] -> [([Expr], Expr)] -> [Expr] -> [([Expr], Expr)]
makeCtrRulesWithDepth d col rules scruts = concatMap makeRule (getCtrs col) where

  -- Create a case branch for constructor c
  -- Example: for constructor "S" with patterns like (S x) and (S (S y)):
  --   fieldPats might be [Var "x"] or [Ctr "S" [Var "y"]]
  --   fieldVars uses existing names when available, fresh vars otherwise
  --   The result is a case branch: ([S x], <recursive match on x>)
  makeRule c = [([Ctr c fieldVars], subExpr) | not (null specialized)] where
    fieldPats   = getFieldPats c col rules
    -- Use existing variable names when available, otherwise create fresh ones
    fieldVars   = [case p of 
                     Var v -> Var v  -- Keep user-provided names
                     _     -> Var ("x" ++ show (d * 10 + i))
                  | (p, i) <- zip fieldPats [0..]]
    newScruts   = fieldVars ++ scruts  -- Always use variables as scrutinees
    specialized = specializeRules c col rules
    subExpr     = flattenWithDepth (d+1) (Mat newScruts specialized)

  -- Extract subpatterns from a constructor match
  -- Example: getFieldPats "S" [Z, S x, S (S y)] rules
  --   finds first S pattern and returns its arguments: [Var "x"]
  --   For nested patterns like (S (S y)), returns: [Ctr "S" [Var "y"]]
  --   If no S found, returns wildcards based on arity
  getFieldPats :: String -> [Expr] -> [([Expr], Expr)] -> [Expr]
  getFieldPats c col rules = 
    case [(args, rule) | (Ctr c' args, rule) <- zip col rules, c == c'] of
      ((args, _):_) -> args
      []            -> replicate (getArity c col) (Var "_")

  -- Specialize rules for a specific constructor
  -- Example: specializeRules "S" [Z, S x, Var y] [([Z], X0), ([Z], X1), ([S y], X2)]
  --   Row 1: Z doesn't match S, skip
  --   Row 2: (S x) matches S, extract x and prepend: ([x, Z], X1)
  --   Row 3: Variable matches anything, add wildcards: ([_, S y], X2)
  specializeRules :: String -> [Expr] -> [([Expr], Expr)] -> [([Expr], Expr)]
  specializeRules c col rules = concat (zipWith specialize col rules) where
    specialize (Ctr c' args) (ps, rhs) | c == c' = [(args ++ ps, rhs)]
    specialize (Var _)       (ps, rhs)           = [(replicate (getArity c col) (Var "_") ++ ps, rhs)]
    specialize _             _                   = []

-- Build variable/default rules for case tree
-- Creates a default case for patterns with variables
-- Example: if col = [Z, Var x, S y] and we've handled Z and S,
--   the variable case handles the middle rule
makeVarRules :: [Expr] -> [([Expr], Expr)] -> [Expr] -> [([Expr], Expr)]
makeVarRules = makeVarRulesWithDepth 0

makeVarRulesWithDepth :: Int -> [Expr] -> [([Expr], Expr)] -> [Expr] -> [([Expr], Expr)]
makeVarRulesWithDepth d col rules scruts
  | hasVars   = [([Var "_"], flattenWithDepth d (Mat scruts varRules))]
  | otherwise = []
  where
    hasVars  = any isVar col
    varRules = selectVarRules col rules

    -- Select rules that match on variables
    -- Example: selectVarRules [Z, Var x, S y] rules
    --   returns only the second rule (the one with Var x)
    selectVarRules :: [Expr] -> [([Expr], Expr)] -> [([Expr], Expr)]
    selectVarRules col rules = concat (zipWith select col rules) where
      select (Var _) clause = [clause]
      select _ _ = []

-- Pretty Printing
-- ---------------

-- Generate indentation
indent :: Int -> String
indent n = replicate n ' '

-- Pretty print expressions
ppExpr :: Expr -> String
ppExpr (Var v) = v
ppExpr (Ctr c []) = "(" ++ c ++ ")"
ppExpr (Ctr c args) = "(" ++ c ++ " " ++ unwords (map ppExpr args) ++ ")"
ppExpr (Mat scruts rules) = ppMat 0 (Mat scruts rules)

-- Print field variables if any
ppFields :: [String] -> String
ppFields [] = ""
ppFields fs = " " ++ unwords fs

-- Pretty print Mat expressions
ppMat :: Int -> Expr -> String
ppMat ind (Mat [] [([], e)]) = indent ind ++ ppExpr e
ppMat ind (Mat scruts rules) = 
  indent ind ++ "match " ++ ppScruts scruts ++ ":\n" ++
  concatMap (ppRule (ind + 2)) rules
ppMat ind e = indent ind ++ ppExpr e

-- Pretty print scrutinees (showing <pattern> for non-variables)
ppScruts :: [Expr] -> String  
ppScruts scruts = unwords (map ppScrut scruts)
  where ppScrut (Var v) = v
        ppScrut e       = "<" ++ ppExpr e ++ ">"

-- Print a rule
ppRule :: Int -> ([Expr], Expr) -> String
ppRule ind (pats, body) = 
  indent ind ++ "case " ++ ppPats pats ++ ":\n" ++
  ppBody (ind + 2) body ++ "\n"

-- Print patterns
ppPats :: [Expr] -> String
ppPats [Ctr c args] = c ++ ppFields [n | Var n <- args]
ppPats [Var v] = v
ppPats pats = unwords (map ppExpr pats)

-- Print rule body
ppBody :: Int -> Expr -> String
ppBody ind e@(Mat _ _) = ppMat ind e
ppBody ind e = indent ind ++ ppExpr e

-- Main
-- ----

-- Test the flattener
main :: IO ()
main = do
  let example = makeExample
      result  = flatten example
  
  putStrLn "Input:\nmatch x y"
  putStrLn "| Z     Z         = X0"
  putStrLn "| (S x) Z         = X1"
  putStrLn "| Z     (S y)     = X2"
  putStrLn "| (S x) (S Z)     = X3"
  putStrLn "| (S x) (S (S z)) = X4\n"
  putStrLn "Output:"
  putStrLn $ ppMat 0 result
  putStrLn "\nStep-by-step explanation:"
  putStrLn "1. First column has [Z, S x, Z, S x, S x] - mixed constructors"
  putStrLn "2. Create case on x with branches for Z and S"
  putStrLn "3. For Z branch: remaining patterns are [Z, S y] on y"
  putStrLn "4. For S branch: remaining patterns are [Z, S Z, S (S z)] on y"
  putStrLn "5. The nested pattern (S Z) and (S (S z)) create another level"
  putStrLn "6. User names (x, y, z) are preserved; fresh vars (x0, x1) only when needed"

-- Build example match expression
-- This represents:
--   match x y
--   | Z     Z         = X0
--   | (S x) Z         = X1  
--   | Z     (S y)     = X2
--   | (S x) (S Z)     = X3
--   | (S x) (S (S z)) = X4
makeExample :: Expr
makeExample = Mat [Var "x", Var "y"] 
  [ ([zero, zero], Var "X0")
  , ([suc (Var "x"), zero], Var "X1")
  , ([zero, suc (Var "y")], Var "X2")
  , ([suc (Var "x"), suc zero], Var "X3")
  , ([suc (Var "x"), suc (suc (Var "z"))], Var "X4")
  ]
  where
    zero = Ctr "Z" []
    suc x = Ctr "S" [x]
