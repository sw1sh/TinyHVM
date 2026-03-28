# operation-level tests — ported from tinygrad's test_ops.py (subset)
# zero dependencies beyond tinyhvm itself
import unittest, sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tinyhvm import Tensor

def approx(a, b, atol=1e-4):
  """Compare two floats or nested lists with tolerance."""
  if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
    assert len(a) == len(b), f"length mismatch: {len(a)} vs {len(b)}"
    for x, y in zip(a, b):
      approx(x, y, atol)
  else:
    assert abs(float(a) - float(b)) < atol, f"{a} != {b} (atol={atol})"

class TestUnary(unittest.TestCase):

  def test_neg(self):
    approx((-Tensor([1., -2., 3.])).tolist(), [-1., 2., -3.])

  def test_relu(self):
    approx(Tensor([-3., -1., 0., 1., 5.]).relu().tolist(), [0., 0., 0., 1., 5.])

  def test_exp(self):
    approx(Tensor([0., 1., 2.]).exp().tolist(), [1., math.e, math.e**2], atol=1e-3)

  def test_log(self):
    approx(Tensor([1., math.e, math.e**2]).log().tolist(), [0., 1., 2.], atol=1e-3)

  def test_sqrt(self):
    approx(Tensor([1., 4., 9., 25.]).sqrt().tolist(), [1., 2., 3., 5.])

  def test_abs(self):
    approx(Tensor([-3., -1., 0., 2., 5.]).abs().tolist(), [3., 1., 0., 2., 5.])

  def test_sigmoid_bounds(self):
    out = Tensor([-10., 0., 10.]).sigmoid().tolist()
    self.assertAlmostEqual(out[1], 0.5, places=4)
    self.assertGreater(out[2], 0.99)
    self.assertLess(out[0], 0.01)

  def test_tanh_bounds(self):
    out = Tensor([-10., 0., 10.]).tanh().tolist()
    self.assertAlmostEqual(out[1], 0.0, places=4)
    self.assertGreater(out[2], 0.99)
    self.assertLess(out[0], -0.99)

class TestBinary(unittest.TestCase):

  def test_add(self):
    approx((Tensor([1.,2.,3.]) + Tensor([4.,5.,6.])).tolist(), [5., 7., 9.])

  def test_sub(self):
    approx((Tensor([5.,7.,9.]) - Tensor([1.,2.,3.])).tolist(), [4., 5., 6.])

  def test_mul(self):
    approx((Tensor([2.,3.,4.]) * Tensor([5.,6.,7.])).tolist(), [10., 18., 28.])

  def test_div(self):
    approx((Tensor([10.,20.,30.]) / Tensor([2.,5.,6.])).tolist(), [5., 4., 5.])

  def test_maximum(self):
    a = Tensor([1., 5., 3.])
    b = Tensor([4., 2., 6.])
    approx(a.maximum(b).tolist(), [4., 5., 6.])

  def test_scalar_add(self):
    approx((Tensor([1.,2.,3.]) + 10).tolist(), [11., 12., 13.])

  def test_scalar_mul(self):
    approx((Tensor([1.,2.,3.]) * 0.5).tolist(), [0.5, 1.0, 1.5])

  def test_scalar_radd(self):
    approx((100 + Tensor([1.,2.,3.])).tolist(), [101., 102., 103.])

  def test_scalar_rsub(self):
    approx((10 - Tensor([1.,2.,3.])).tolist(), [9., 8., 7.])

  def test_pow_int(self):
    approx((Tensor([2., 3.]) ** 3).tolist(), [8., 27.], atol=1e-2)

class TestReduce(unittest.TestCase):

  def test_sum_1d(self):
    self.assertAlmostEqual(Tensor([1., 2., 3., 4.]).sum().item(), 10.0)

  def test_sum_2d_axis0(self):
    t = Tensor([[1., 2.], [3., 4.]])
    approx(t.sum(axis=0).tolist(), [4., 6.])

  def test_sum_2d_axis1(self):
    t = Tensor([[1., 2.], [3., 4.]])
    approx(t.sum(axis=1).tolist(), [3., 7.])

  def test_sum_keepdim(self):
    t = Tensor([[1., 2.], [3., 4.]])
    out = t.sum(axis=1, keepdim=True)
    self.assertEqual(out.shape, (2, 1))
    approx(out.tolist(), [[3.], [7.]])

  def test_max_1d(self):
    self.assertEqual(Tensor([3., 1., 4., 1., 5.]).max().item(), 5.0)

  def test_max_axis(self):
    t = Tensor([[1., 5.], [3., 2.]])
    approx(t.max(axis=0).tolist(), [3., 5.])
    approx(t.max(axis=1).tolist(), [5., 3.])

  def test_mean(self):
    self.assertAlmostEqual(Tensor([1., 2., 3., 4.]).mean().item(), 2.5)

  def test_mean_axis(self):
    t = Tensor([[1., 3.], [5., 7.]])
    approx(t.mean(axis=0).tolist(), [3., 5.])
    approx(t.mean(axis=1).tolist(), [2., 6.])

class TestMovement(unittest.TestCase):

  def test_reshape(self):
    t = Tensor.arange(12)
    self.assertEqual(t.reshape(3, 4).shape, (3, 4))
    self.assertEqual(t.reshape(2, 2, 3).shape, (2, 2, 3))

  def test_reshape_neg1(self):
    t = Tensor.arange(24).reshape(2, 3, 4)
    self.assertEqual(t.reshape(-1).shape, (24,))
    self.assertEqual(t.reshape(6, -1).shape, (6, 4))

  def test_permute_data(self):
    t = Tensor([[1., 2., 3.], [4., 5., 6.]])  # (2, 3)
    out = t.permute(1, 0)  # (3, 2)
    self.assertEqual(out.shape, (3, 2))
    approx(out.tolist(), [[1., 4.], [2., 5.], [3., 6.]])

  def test_expand_broadcast(self):
    t = Tensor([[1.], [2.], [3.]])  # (3, 1)
    out = t.expand(3, 4)
    self.assertEqual(out.shape, (3, 4))
    approx(out.tolist(), [[1.,1.,1.,1.], [2.,2.,2.,2.], [3.,3.,3.,3.]])

  def test_unsqueeze_squeeze(self):
    t = Tensor([1., 2., 3.])
    self.assertEqual(t.unsqueeze(0).shape, (1, 3))
    self.assertEqual(t.unsqueeze(0).squeeze(0).shape, (3,))

  def test_flatten(self):
    t = Tensor.ones(2, 3, 4)
    self.assertEqual(t.flatten().shape, (24,))
    self.assertEqual(t.flatten(1).shape, (2, 12))

  def test_T_roundtrip(self):
    t = Tensor([[1., 2.], [3., 4.]])
    approx(t.T.T.tolist(), t.tolist())

class TestMatmul(unittest.TestCase):

  def test_2x2(self):
    a = Tensor([[1., 2.], [3., 4.]])
    b = Tensor([[5., 6.], [7., 8.]])
    approx((a @ b).tolist(), [[19., 22.], [43., 50.]])

  def test_identity(self):
    a = Tensor([[1., 2., 3.], [4., 5., 6.]])
    eye = Tensor.eye(3)
    approx((a @ eye).tolist(), a.tolist())

  def test_matmul_broadcast(self):
    """Batch of vectors times matrix."""
    a = Tensor.ones(1, 4)
    b = Tensor.full((4, 2), 0.5)
    out = a @ b
    self.assertEqual(out.shape, (1, 2))
    approx(out.tolist(), [[2.0, 2.0]])

class TestCompound(unittest.TestCase):

  def test_softmax_sums_to_1(self):
    out = Tensor([[1., 2., 3., 4.]]).softmax(axis=-1).tolist()[0]
    self.assertAlmostEqual(sum(out), 1.0, places=4)

  def test_softmax_monotonic(self):
    out = Tensor([[1., 2., 3.]]).softmax(axis=-1).tolist()[0]
    self.assertLess(out[0], out[1])
    self.assertLess(out[1], out[2])

  def test_log_softmax_sum_exp(self):
    """exp(log_softmax(x)).sum() == 1"""
    out = Tensor([1., 2., 3.]).log_softmax(axis=0).exp()
    self.assertAlmostEqual(out.sum().item(), 1.0, places=3)

  def test_mse_loss_zero(self):
    a = Tensor([1., 2., 3.])
    self.assertAlmostEqual(a.mse_loss(a).item(), 0.0, places=5)

  def test_mse_loss_value(self):
    a = Tensor([1., 2., 3.])
    b = Tensor([4., 5., 6.])
    # MSE = mean((3,3,3)^2) = mean(9,9,9) = 9
    self.assertAlmostEqual(a.mse_loss(b).item(), 9.0, places=3)

  def test_reciprocal(self):
    approx(Tensor([2., 4., 5.]).reciprocal().tolist(), [0.5, 0.25, 0.2], atol=1e-3)

class TestFactory(unittest.TestCase):

  def test_arange(self):
    approx(Tensor.arange(5).tolist(), [0., 1., 2., 3., 4.])

  def test_arange_start_stop(self):
    approx(Tensor.arange(2, 6).tolist(), [2., 3., 4., 5.])

  def test_arange_step(self):
    approx(Tensor.arange(0, 10, 3).tolist(), [0., 3., 6., 9.])

  def test_eye_matmul(self):
    """eye(n) is the identity for matmul."""
    a = Tensor([[1., 2.], [3., 4.]])
    approx((a @ Tensor.eye(2)).tolist(), a.tolist())

  def test_full(self):
    approx(Tensor.full((2, 3), 5.0).tolist(), [[5., 5., 5.], [5., 5., 5.]])

  def test_zeros_ones(self):
    self.assertEqual(Tensor.zeros(3).sum().item(), 0.0)
    self.assertEqual(Tensor.ones(3).sum().item(), 3.0)

  def test_rand_range(self):
    vals = Tensor.rand(100).tolist()
    self.assertTrue(all(0.0 <= v <= 1.0 for v in vals))

  def test_randn_shape(self):
    self.assertEqual(Tensor.randn(5, 3, 7).shape, (5, 3, 7))

class TestGrad(unittest.TestCase):

  def test_grad_square(self):
    """d/dx(sum(x^2)) = 2x"""
    x = Tensor([2., 3.], requires_grad=True)
    y = (x * x).sum()
    g = Tensor.grad_fn(y, x).realize()
    approx(g.tolist(), [4., 6.])

  def test_grad_linear(self):
    """d/dx(sum(3x + 1)) = 3"""
    x = Tensor([1., 2., 3.], requires_grad=True)
    y = (x * 3 + 1).sum()
    g = Tensor.grad_fn(y, x).realize()
    approx(g.tolist(), [3., 3., 3.])

  def test_grad_matmul(self):
    """d/dW(sum(x @ W)) with x=ones, W=anything → gradient = x^T broadcast."""
    W = Tensor([[1., 2.], [3., 4.]], requires_grad=True)
    x = Tensor.ones(1, 2)
    y = (x @ W).sum()
    g = Tensor.grad_fn(y, W).realize()
    approx(g.tolist(), [[1., 1.], [1., 1.]])

if __name__ == '__main__':
  unittest.main()
