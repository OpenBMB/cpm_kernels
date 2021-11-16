import cpm_kernels.torch as ct
import torch
import unittest


class TestGeLU(unittest.TestCase):
    def test_gelu(self):
        with torch.cuda.device(2):
            x = torch.randn(4, 16, 512, 1024, device="cuda").half()
            x1 = x.clone().requires_grad_()
            x2 = x.clone().requires_grad_()
            del x
            out = ct.gelu(x1)
            ans = ct.geluTH(x2)
            self.assertTrue(torch.isclose(out, ans, 1e-2, 1e-2).all())

            gradient_start = torch.randn(4, 16, 512, 1024, device="cuda").half()
            out.backward(gradient=gradient_start)
            ans.backward(gradient=gradient_start)

            self.assertTrue(torch.isclose(x1.grad, x2.grad, 1e-2, 1e-2).all())

    def test_gelu_inplace(self):
        with torch.cuda.device(0):
            x = torch.randn(4, 1237, device="cuda").half()
            ans = ct.geluTH(x)
            ct.gelu_inplace(x)

            self.assertTrue(torch.isclose(x, ans, 1e-2, 1e-2).all())