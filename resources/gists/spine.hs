module Main where

import Unsafe.Coerce (unsafeCoerce)

data Term
  = App Term Term
  | Var String
  | Ct0
  | Ct1 Term
  | Ct2 Term Term
  deriving Show

data Spine = Spine Int ([Term] -> Term)

instance Show Spine where
  show (Spine k f) = show (f (replicate k (Var "_")))

spine_new :: Spine
spine_new = Spine 0 (\_ -> Var "FN")

spine_app :: Int -> a -> [Term] -> Term
spine_app 0 c []     = unsafeCoerce c
spine_app n f (x:xs) = spine_app (n-1) (unsafeCoerce f x) xs

spine_ctr :: Spine -> Int -> a -> Spine
spine_ctr (Spine 0 f) n c = Spine n $ \hs -> App (f []) (spine_app n c hs)
spine_ctr (Spine k f) n c = Spine (k+n-1) $ \hs -> let (h,t) = splitAt n hs in f (spine_app n c h : t)

main :: IO ()
main = do
  let spn_0 = spine_new
  let spn_1 = spine_ctr spn_0 1 (\x -> Ct1 x)
  let spn_2 = spine_ctr spn_1 2 (\x y -> Ct2 x y)
  let spn_3 = spine_ctr spn_2 0 Ct0
  let spn_4 = spine_ctr spn_3 0 Ct0
  print spn_0
  print spn_1
  print spn_2
  print spn_3
  print spn_4
