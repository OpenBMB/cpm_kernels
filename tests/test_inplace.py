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
    
