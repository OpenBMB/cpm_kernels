import cpm_kernels.torch as ct
import torch
import unittest


class TestGeLU(unittest.TestCase):
    def test_gelu(self):
        with torch.cuda.device(1):
            x = torch.randn(4, 16, 512, 1024, device="cuda").half()
            x1 = x.clone().requires_grad_()
            x2 = x.clone().requires_grad_()
            del x
            out = ct.gelu(x1)
            ans = ct.geluTH(x2)
            diff = torch.abs(out - ans).max()
            self.assertLess(diff, 1e-2)

            gradient_start = torch.randn(4, 16, 512, 1024, device="cuda").half()
            out.backward(gradient=gradient_start)
            ans.backward(gradient=gradient_start)

            diff = torch.abs(x1.grad - x2.grad).max()
            self.assertLess(diff, 1e-2)

