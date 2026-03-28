# basic self-contained tests — ported from tinygrad's test_tiny.py
# zero dependencies beyond tinyhvm itself
import unittest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tinyhvm import Tensor

class TestTiny(unittest.TestCase):

  # *** basic arithmetic ***

  def test_plus(self):
    out = Tensor([1.,2,3]) + Tensor([4.,5,6])
    self.assertListEqual(out.tolist(), [5.0, 7.0, 9.0])

  def test_plus_scalar(self):
    out = Tensor([1.,2,3]) + 10
    self.assertListEqual(out.tolist(), [11.0, 12.0, 13.0])

  def test_plus_big(self):
    out = Tensor.ones(16).contiguous() + Tensor.ones(16).contiguous()
    self.assertListEqual(out.tolist(), [2.0]*16)

  def test_sub(self):
    out = Tensor([10.,20,30]) - Tensor([1.,2,3])
    self.assertListEqual(out.tolist(), [9.0, 18.0, 27.0])

  def test_mul(self):
    out = Tensor([2.,3,4]) * Tensor([5.,6,7])
    self.assertListEqual(out.tolist(), [10.0, 18.0, 28.0])

  def test_div(self):
    out = Tensor([10.,20,30]) / Tensor([2.,4,5])
    self.assertListEqual(out.tolist(), [5.0, 5.0, 6.0])

  def test_neg(self):
    out = -Tensor([1.,-2,3])
    self.assertListEqual(out.tolist(), [-1.0, 2.0, -3.0])

  # *** scalar ops ***

  def test_radd(self):
    out = 5 + Tensor([1.,2,3])
    self.assertListEqual(out.tolist(), [6.0, 7.0, 8.0])

  def test_rsub(self):
    out = 10 - Tensor([1.,2,3])
    self.assertListEqual(out.tolist(), [9.0, 8.0, 7.0])

  def test_rmul(self):
    out = 3 * Tensor([1.,2,3])
    self.assertListEqual(out.tolist(), [3.0, 6.0, 9.0])

  # *** unary ops ***

  def test_relu(self):
    out = Tensor([-2., -1., 0., 1., 2.]).relu()
    self.assertListEqual(out.tolist(), [0.0, 0.0, 0.0, 1.0, 2.0])

  def test_exp(self):
    out = Tensor([0.0]).exp()
    self.assertAlmostEqual(out.item(), 1.0, places=5)

  def test_log(self):
    out = Tensor([1.0]).log()
    self.assertAlmostEqual(out.item(), 0.0, places=5)

  def test_sqrt(self):
    out = Tensor([4.0, 9.0, 16.0]).sqrt()
    self.assertListEqual(out.tolist(), [2.0, 3.0, 4.0])

  # *** reduction ***

  def test_sum(self):
    out = Tensor.ones(256).contiguous().sum()
    self.assertEqual(out.item(), 256.0)

  def test_sum_axis(self):
    t = Tensor([[1.,2,3],[4.,5,6]])
    self.assertListEqual(t.sum(axis=0).tolist(), [5.0, 7.0, 9.0])
    self.assertListEqual(t.sum(axis=1).tolist(), [6.0, 15.0])

  def test_max(self):
    out = Tensor([3., 1., 4., 1., 5., 9., 2., 6.]).max()
    self.assertEqual(out.item(), 9.0)

  def test_mean(self):
    out = Tensor([2., 4., 6., 8.]).mean()
    self.assertEqual(out.item(), 5.0)

  # *** matmul ***

  def test_gemm(self):
    N = 64
    a = Tensor.ones(N, N).contiguous()
    b = Tensor.eye(N).contiguous()
    lst = (a @ b).tolist()
    for y in range(N):
      for x in range(N):
        self.assertEqual(lst[y][x], 1.0, msg=f"mismatch at ({y},{x})")

  def test_gemv(self):
    N = 64
    a = Tensor.ones(1, N).contiguous()
    b = Tensor.eye(N).contiguous()
    lst = (a @ b).tolist()
    for x in range(N):
      self.assertEqual(lst[0][x], 1.0, msg=f"mismatch at {x}")

  def test_matmul_small(self):
    a = Tensor([[1.,2.],[3.,4.]])
    b = Tensor([[5.,6.],[7.,8.]])
    out = (a @ b).tolist()
    self.assertListEqual(out, [[19.0, 22.0], [43.0, 50.0]])

  # *** movement ***

  def test_reshape(self):
    t = Tensor([1.,2,3,4,5,6])
    self.assertEqual(t.reshape(2, 3).shape, (2, 3))
    self.assertEqual(t.reshape(3, 2).shape, (3, 2))

  def test_reshape_neg1(self):
    t = Tensor.ones(2, 3, 4)
    self.assertEqual(t.reshape(-1).shape, (24,))
    self.assertEqual(t.reshape(2, -1).shape, (2, 12))

  def test_permute(self):
    t = Tensor.ones(2, 3, 4)
    self.assertEqual(t.permute(2, 0, 1).shape, (4, 2, 3))

  def test_transpose(self):
    t = Tensor([[1.,2,3],[4.,5,6]])
    self.assertEqual(t.T.shape, (3, 2))
    self.assertListEqual(t.T.tolist(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

  def test_flatten(self):
    t = Tensor.ones(2, 3, 4)
    self.assertEqual(t.flatten().shape, (24,))
    self.assertEqual(t.flatten(start_dim=1).shape, (2, 12))

  def test_unsqueeze(self):
    t = Tensor([1., 2., 3.])
    self.assertEqual(t.unsqueeze(0).shape, (1, 3))
    self.assertEqual(t.unsqueeze(1).shape, (3, 1))

  def test_squeeze(self):
    t = Tensor.ones(1, 3, 1)
    self.assertEqual(t.squeeze().shape, (3,))
    self.assertEqual(t.squeeze(0).shape, (3, 1))

  def test_expand(self):
    t = Tensor.ones(1, 3)
    out = t.expand(4, 3)
    self.assertEqual(out.shape, (4, 3))
    self.assertEqual(out.sum().item(), 12.0)

  # *** factory ***

  def test_zeros(self):
    self.assertListEqual(Tensor.zeros(4).tolist(), [0.0]*4)

  def test_ones(self):
    self.assertListEqual(Tensor.ones(4).tolist(), [1.0]*4)

  def test_eye(self):
    self.assertListEqual(Tensor.eye(3).tolist(),
                         [[1,0,0],[0,1,0],[0,0,1]])

  def test_full(self):
    self.assertListEqual(Tensor.full((3,), 7.0).tolist(), [7.0, 7.0, 7.0])

  def test_arange(self):
    self.assertListEqual(Tensor.arange(5).tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])
    self.assertListEqual(Tensor.arange(2, 5).tolist(), [2.0, 3.0, 4.0])

  # *** randomness ***

  def test_random(self):
    out = Tensor.rand(10)
    for x in out.tolist():
      self.assertGreaterEqual(x, 0.0)
      self.assertLessEqual(x, 1.0)

  def test_randn_shape(self):
    self.assertEqual(Tensor.randn(3, 4, 5).shape, (3, 4, 5))

  # *** compound ops ***

  def test_pow_square(self):
    out = Tensor([2., 3., 4.]) ** 2
    self.assertListEqual(out.tolist(), [4.0, 9.0, 16.0])

  def test_pow_sqrt(self):
    out = Tensor([4., 9., 16.]) ** 0.5
    self.assertListEqual(out.tolist(), [2.0, 3.0, 4.0])

  def test_sigmoid(self):
    out = Tensor([0.0]).sigmoid()
    self.assertAlmostEqual(out.item(), 0.5, places=4)

  def test_softmax(self):
    out = Tensor([1., 2., 3.]).softmax(axis=0)
    s = sum(out.tolist())
    self.assertAlmostEqual(s, 1.0, places=4)

  # *** 2d nested list ***

  def test_2d_creation(self):
    t = Tensor([[1., 2.], [3., 4.]])
    self.assertEqual(t.shape, (2, 2))
    self.assertListEqual(t.tolist(), [[1.0, 2.0], [3.0, 4.0]])

  def test_3d_creation(self):
    t = Tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    self.assertEqual(t.shape, (2, 2, 2))

  # *** misc properties ***

  def test_len(self):
    self.assertEqual(len(Tensor.ones(5, 3)), 5)

  def test_numel(self):
    self.assertEqual(Tensor.ones(2, 3, 4).numel, 24)

  def test_ndim(self):
    self.assertEqual(Tensor.ones(2, 3).ndim, 2)

if __name__ == '__main__':
  unittest.main()
