import cpm_kernels.torch as ct
import torch
import unittest

class TestInplace(unittest.TestCase):
    def test_add(self):
        with torch.cuda.device(3):
            x = torch.randn(3, 5, 7, 9, 123, 321, device="cuda").half()
            y = torch.randn(3, 5, 7, 9, 123, 321, device="cuda").half()

            ans = x + y
            ct.inplace_add(x, y)
            diff = torch.abs(x - ans).max()
            self.assertLess(diff, 1e-5)
    
    def test_mask_inf(self):
        with torch.cuda.device(3):
            x = torch.randn(3, 5, 7, 9, 123, 321, device="cuda").half()
            mask = torch.randn(3, 5, 7, 9, 123, 321, device="cuda") < 0
            value = float("-inf")
            
            ans = torch.where(
                mask,
                x,
                torch.scalar_tensor(value, device=x.device, dtype=x.dtype),
            )
            ct.inplace_mask(x, mask, value)
            self.assertTrue(torch.isclose(x, ans, 1e-5).all())

    def test_mask_zero(self):
        with torch.cuda.device(3):
            x = torch.randn(3, 5, 7, 9, 123, 321, device="cuda").half()
            mask = torch.randn(3, 5, 7, 9, 123, 321, device="cuda") < 0
            value = 0
            
            ans = torch.where(
                mask,
                x,
                torch.scalar_tensor(value, device=x.device, dtype=x.dtype),
            )
            ct.inplace_mask(x, mask, value)

            diff = torch.abs(x - ans).max()
            self.assertLess(diff, 1e-5)
    
    def test_scale_add(self):
        with torch.cuda.device(3):
            x = torch.randn(32, 4096, 1024, device="cuda").half()
            alpha = (torch.randn(4096, device="cuda").half() + 10) / 5
            beta = torch.randn(4096, device="cuda").half()

            ans = ct.inplace_mul_addTH(x, alpha, beta)
            ct.inplace_mul_add(x, alpha, beta)
            diff = torch.abs(x - ans).max()
            self.assertLess(diff, 1e-2)

            ans = ct.inplace_sub_divTH(x, alpha, beta)
            ct.inplace_sub_div(x, alpha, beta)
            
            diff = torch.abs(x - ans).max()
            self.assertLess(diff, 1e-2)
    
    def test_add_2(self):
        with torch.cuda.device(3):
            x = torch.randn(32, 4096, 1024, device="cuda").half()
            alpha = torch.ones(4096, device="cuda").half()
            beta = torch.randn(4096, device="cuda").half()

            ans = ct.inplace_mul_addTH(x, alpha, beta)
            ct.inplace_mul_add(x, alpha, beta)
            diff = torch.abs(x - ans).max()
            self.assertLess(diff, 1e-5)

            ans = ct.inplace_sub_divTH(x, alpha, beta)
            ct.inplace_sub_div(x, alpha, beta)
            
            diff = torch.abs(x - ans).max()
            self.assertLess(diff, 1e-5)
    
    def test_scale(self):
        with torch.cuda.device(3):
            x = torch.randn(32, 4096, 1024, device="cuda").half()
            alpha = torch.randn(4096, device="cuda").half() + 10

            ans = ct.inplace_mulTH(x, alpha)
            ct.inplace_mul(x, alpha)
            diff = torch.abs(x - ans).max()
            
            self.assertLess(diff, 1e-5)

            ans = ct.inplace_divTH(x, alpha)
            ct.inplace_div(x, alpha)

            diff = torch.abs(x - ans).max()
            self.assertLess(diff, 1e-5)
    
    def test_batched_add(self):
        with torch.cuda.device(3):
            for shape in [
                (3, 5, 6),
                (17, 32, 128),
                (32, 1024, 4096),
                (33, 777, 1231),
                (31, 123, 567),
                (3, 4, 5, 6, 7),
                (21, 66, 5, 3, 2),
                (11, 3, 5, 7, 9)
            ]:
                batch = shape[0]
                x = torch.randn(shape, device="cuda").half()
                y = torch.randn(shape[1:], device="cuda").half()

                x1 = x.clone().requires_grad_()
                y1 = y.clone().requires_grad_()
                x2 = x.clone().requires_grad_()
                y2 = y.clone().requires_grad_()

                out = ct.batched_add(x1, y1)
                ans = ct.batched_addTH(x2, y2)
                diff = torch.abs(out - ans).max() / batch

                self.assertLess(diff, 1e-5)

                gradient_start = torch.randn(shape, device="cuda").half()
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)
                
                diff = torch.abs(x1.grad - x2.grad).max()
                self.assertLess(diff, 1e-3)

                diff = torch.abs(y1.grad - y2.grad).max() / batch
                self.assertLess(diff, 1e-3)
    
