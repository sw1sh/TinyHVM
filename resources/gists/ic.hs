import Control.Monad (when)
import Data.Char (chr, ord)
import Data.IORef
import Data.Word
import Debug.Trace
import System.IO.Unsafe (unsafePerformIO)
import Text.Parsec hiding (State)
import qualified Data.IntMap.Strict as IntMap
import qualified Data.Map as Map
import qualified Text.Parsec as Parsec
import Control.Monad.IO.Class (liftIO)

type Name = Word64

data Term
  = Var Name                     -- Name
  | Let Name Term Term           -- "! " Name " = " Term "; " Term
  | Era                          -- "*"
  | Sup Name Term Term           -- "&" Name "{" Term "," Term "}"
  | Dup Name Name Name Term Term -- "! &" Name "{" Name "," Name "}" "=" Term ";" Term
  | Lam Name Term                -- "λ" Name "." Term
  | App Term Term                -- "(" Term " " Term ")"

-- Globals
-- -------

{-# NOINLINE gSUBST #-}
gSUBST :: IORef (IntMap.IntMap Term)
gSUBST = unsafePerformIO $ newIORef IntMap.empty

{-# NOINLINE gFRESH #-}
gFRESH :: IORef Name
gFRESH = unsafePerformIO $ newIORef 0

{-# NOINLINE gINTERS #-}
gINTERS :: IORef Word64
gINTERS = unsafePerformIO $ newIORef 0

-- Evaluator
-- ---------

-- Helper functions for global substitution
set :: Name -> Term -> IO ()
set name term = do
  subMap <- readIORef gSUBST
  writeIORef gSUBST (IntMap.insert (fromIntegral name) term subMap)

get :: Name -> IO (Maybe Term)
get name = do
  subMap <- readIORef gSUBST
  return $ IntMap.lookup (fromIntegral name) subMap

fresh :: IO Name
fresh = do
  n <- readIORef gFRESH
  writeIORef gFRESH (n + 1)
  return n

incInters :: IO ()
incInters = do
  n <- readIORef gINTERS
  writeIORef gINTERS (n + 1)

app_era :: Term -> Term -> IO Term
app_era Era _ = do
  incInters
  return Era
app_era _ _ = error "app_era: expected Era as first argument"

app_lam :: Term -> Term -> IO Term
app_lam (Lam nam bod) arg = do
  incInters
  set nam arg
  whnf bod
app_lam _ _ = error "app_lam: expected Lam as first argument"

app_sup :: Term -> Term -> IO Term
app_sup (Sup lab lft rgt) arg = do
  incInters
  c0 <- fresh
  c1 <- fresh
  let a0 = App lft (Var c0)
  let a1 = App rgt (Var c1)
  whnf (Dup lab c0 c1 arg (Sup lab a0 a1))
app_sup _ _ = error "app_sup: expected Sup as first argument"

app_dup :: Term -> IO Term
app_dup (App f (Dup lab x y val bod)) = do
  incInters
  whnf (Dup lab x y val (App f bod))
app_dup term = error "app_dup: expected App with Dup"

dup_era :: Term -> Term -> IO Term
dup_era (Dup lab r s _ k) Era = do
  incInters
  set r Era
  set s Era
  whnf k
dup_era _ _ = error "dup_era: expected Dup and Era"

dup_lam :: Term -> Term -> IO Term
dup_lam (Dup lab r s _ k) (Lam x f) = do
  incInters
  x0 <- fresh
  x1 <- fresh
  f0 <- fresh
  f1 <- fresh
  set r (Lam x0 (Var f0))
  set s (Lam x1 (Var f1))
  set x (Sup lab (Var x0) (Var x1))
  whnf (Dup lab f0 f1 f k)
dup_lam _ _ = error "dup_lam: expected Dup and Lam"

dup_sup :: Term -> Term -> IO Term
dup_sup (Dup dupLab x y _ k) (Sup supLab a b) = do
  incInters
  if dupLab == supLab then do
    set x a
    set y b
    whnf k
  else do
    a0 <- fresh
    a1 <- fresh
    b0 <- fresh
    b1 <- fresh
    set x (Sup supLab (Var a0) (Var b0))
    set y (Sup supLab (Var a1) (Var b1))
    whnf (Dup dupLab a0 a1 a (Dup dupLab b0 b1 b k))
dup_sup _ _ = error "dup_sup: expected Dup and Sup"

dup_dup :: Term -> Term -> IO Term
dup_dup (Dup labL x0 x1 _ t) (Dup labR y0 y1 y x) = do
  -- incInters
  whnf (Dup labL x0 x1 x (Dup labL y0 y1 y t))
dup_dup _ _ = error "dup_dup: expected Dup with inner Dup"

whnf :: Term -> IO Term
whnf term = do
  case term of
    Var n -> do
      sub <- get n
      case sub of
        Just s  -> whnf s
        Nothing -> return (Var n)
    Let x v b -> do
      v' <- whnf v
      set x v'
      whnf b
    App f a -> do
      f' <- whnf f
      case f' of
        Lam _ _       -> app_lam f' a
        Sup _ _ _     -> app_sup f' a
        Era           -> app_era f' a
        Dup _ _ _ _ _ -> app_dup (App f' a)
        _             -> return (App f' a)
    Dup l x y v b -> do
      v' <- whnf v
      case v' of
        Lam _ _       -> dup_lam (Dup l x y v' b) v'
        Sup _ _ _     -> dup_sup (Dup l x y v' b) v'
        Era           -> dup_era (Dup l x y v' b) v'
        Dup _ _ _ _ _ -> dup_dup (Dup l x y v' b) v'
        _             -> return $ Dup l x y v' b
    _ -> return term

-- Normalization
-- -------------

normal :: Term -> IO Term
normal term = do
  term' <- whnf term
  case term' of
    Lam x b -> do
      b' <- normal b
      return $ Lam x b'
    App f a -> do
      f' <- normal f
      a' <- normal a
      return $ App f' a'
    Sup l a b -> do
      a' <- normal a
      b' <- normal b
      return $ Sup l a' b'
    Dup l x y v b -> do
      v' <- normal v
      b' <- normal b
      return $ Dup l x y v' b'
    _ -> return term'

-- Stringifier
-- -----------

name :: Name -> String
name k = all !! fromIntegral (k+1) where
  all :: [String]
  all = [""] ++ concatMap (\str -> map (: str) ['a'..'z']) all

instance Show Term where
  show (Var n)           = name n
  show (Let x t1 t2)     = "! " ++ name x ++ " = " ++ show t1 ++ "; " ++ show t2
  show Era               = "*"
  show (Sup l t1 t2)     = "&" ++ show (fromIntegral l :: Int) ++ "{" ++ show t1 ++ "," ++ show t2 ++ "}"
  show (Dup l x y t1 t2) = "! &" ++ show (fromIntegral l :: Int) ++ "{" ++ name x ++ "," ++ name y ++ "} = " ++ show t1 ++ "; " ++ show t2
  show (Lam x t)         = "λ" ++ name x ++ "." ++ show t
  show (App t1 t2)       = "(" ++ show t1 ++ " " ++ show t2 ++ ")"

-- Parser
-- ------

type ParserST = Map.Map String Name
type LocalCtx = Map.Map String Name
type Parser a = ParsecT String ParserST IO a

whiteSpace :: Parser ()
whiteSpace = skipMany (space <|> comment) where
  comment = do 
    try (string "//")
    skipMany (noneOf "\n\r")
    (newline <|> (eof >> return '\n'))

lexeme :: Parser a -> Parser a
lexeme p = p <* whiteSpace

symbol :: String -> Parser String
symbol s = lexeme (string s)

parseNatural :: Parser Integer
parseNatural = lexeme $ read <$> many1 digit

isGlobal :: String -> Bool
isGlobal name = take 1 name == "$"

getGlobalName :: String -> Parser Name
getGlobalName gname = do
  globalMap <- getState
  case Map.lookup gname globalMap of
    Just n  -> return n
    Nothing -> do
      n <- liftIO fresh
      putState (Map.insert gname n globalMap)
      return n

bindVar :: String -> LocalCtx -> Parser (Name, LocalCtx)
bindVar name ctx
  | isGlobal name = do
      n <- getGlobalName name
      return (n, ctx)
  | otherwise = do
      n <- liftIO fresh
      let ctx' = Map.insert name n ctx
      return (n, ctx')

getVar :: String -> LocalCtx -> Parser Name
getVar name ctx
  | isGlobal name = getGlobalName name
  | otherwise = case Map.lookup name ctx of
      Just n  -> return n
      Nothing -> fail $ "Unbound local variable: " ++ name

parseVarName :: Parser String
parseVarName = lexeme $ try (do
  char '$'
  name <- many1 (alphaNum <|> char '_')
  return ("$" ++ name)
 ) <|> many1 (alphaNum <|> char '_')

-- Term parsers
parseTerm :: LocalCtx -> Parser Term
parseTerm ctx
  =   try (parseApp ctx)
  <|> try (parseLet ctx)
  <|> try (parseLam ctx)
  <|> try (parseSup ctx)
  <|> try (parseDup ctx)
  <|> parseSimpleTerm ctx

parseSimpleTerm :: LocalCtx -> Parser Term
parseSimpleTerm ctx
  = parseVar ctx
  <|> parseEra
  <|> between (symbol "(") (symbol ")") (parseTerm ctx)

parseVar :: LocalCtx -> Parser Term
parseVar ctx = do
  name <- parseVarName
  n    <- getVar name ctx
  return $ Var n

parseLam :: LocalCtx -> Parser Term
parseLam ctx = do
  symbol "λ"
  name      <- parseVarName
  (n, ctx') <- bindVar name ctx
  symbol "."
  body      <- parseTerm ctx'
  return $ Lam n body

parseApp :: LocalCtx -> Parser Term
parseApp ctx = between (symbol "(") (symbol ")") $ do
  f <- parseTerm ctx
  whiteSpace
  a <- parseTerm ctx
  return $ App f a

parseSup :: LocalCtx -> Parser Term
parseSup ctx = do
  symbol "&"
  l <- fromIntegral <$> parseNatural
  (a, b) <- between (symbol "{") (symbol "}") $ do
    a <- parseTerm ctx
    symbol ","
    b <- parseTerm ctx
    return (a, b)
  return $ Sup l a b

parseDup :: LocalCtx -> Parser Term
parseDup ctx = do
  symbol "!"
  symbol "&"
  l <- fromIntegral <$> parseNatural
  (name1, name2) <- between (symbol "{") (symbol "}") $ do
    a <- parseVarName
    symbol ","
    b <- parseVarName
    return (a, b)
  symbol "="
  val <- parseTerm ctx
  symbol ";"
  (n1, ctx') <- bindVar name1 ctx
  (n2, ctx'') <- bindVar name2 ctx'
  body <- parseTerm ctx''
  return $ Dup l n1 n2 val body

parseLet :: LocalCtx -> Parser Term
parseLet ctx = do
  symbol "!"
  name      <- parseVarName
  symbol "="
  t1        <- parseTerm ctx
  symbol ";"
  (n, ctx') <- bindVar name ctx
  t2        <- parseTerm ctx'
  return $ Let n t1 t2

parseEra :: Parser Term
parseEra = do
  symbol "*"
  return Era

parseIC :: String -> IO (Either ParseError (Term, Map.Map String Name))
parseIC input = runParserT parser Map.empty "" input where
  parser = do
    whiteSpace
    term <- parseTerm Map.empty
    state <- getState
    return (term, state)

doParseIC :: String -> IO Term
doParseIC input = do
  result <- parseIC input
  case result of
    Left err        -> error $ show err
    Right (term, _) -> return term

-- Tests
-- -----

test :: String -> IO ()
test input = do
  term <- doParseIC input
  norm <- normal term
  print term
  print norm

main :: IO ()
main = do
  test $ unlines
    [ "!F = λf."
    , "  !&0{f0,f1} = f;"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  !&0{f0,f1} = λx.(f0 (f1 x));"
    , "  λx.(f0 (f1 x));"
    , "((F λnx.((nx λt0.λf0.f0) λt1.λf1.t1)) λT.λF.T)"
    ]
  inters <- readIORef gINTERS
  putStrLn $ "- WORK: " ++ show inters
