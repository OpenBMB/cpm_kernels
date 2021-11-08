import cpm_kernels.torch as ct
import torch
import unittest

class TestArith(unittest.TestCase):
    def test_element_add(self):
        with torch.cuda.device(3):
            for shape in [
                (3, 5, 6),
                (17, 32, 128),
                (32, 1024, 4096),
                (33, 777, 1232),
                (31, 123, 566),
                (3, 4, 5, 6, 8),
                (21, 66, 5, 3, 2),
                (11, 3, 5, 7, 10)
            ]:
                x = torch.randn(*shape, device="cuda").half()
                y = torch.randn(*shape, device="cuda").half()

                x1 = x.clone().requires_grad_()
                x2 = x.clone().requires_grad_()
                y1 = y.clone().requires_grad_()
                y2 = y.clone().requires_grad_()
                
                out = ct.element_add(x1, y1)
                ans = ct.element_addTH(x2, y2)
                diff = torch.abs(out - ans).max()
                self.assertLess(diff, 1e-5)

                gradient_start = torch.randn_like(out)
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                diff = torch.abs(x1.grad - x2.grad).max()
                self.assertLess(diff, 1e-5)

                diff = torch.abs(y1.grad - y2.grad).max()
                self.assertLess(diff, 1e-5)

                ct.element_add_inplace(x, y)
                diff = torch.abs(x - ans).max()
                self.assertLess(diff, 1e-5)

    def test_element_mul(self):
        with torch.cuda.device(3):
            for shape in [
                (3, 5, 6),
                (17, 32, 128),
                (32, 1024, 4096),
                (33, 778, 1231),
                (31, 124, 321),
                (3, 4, 5, 6, 37),
                (21, 66, 5, 3, 2),
                (11, 3, 4, 7, 11)
            ]:
                x = torch.randn(*shape, device="cuda").half()
                y = torch.randn(*shape, device="cuda").half()

                x1 = x.clone().requires_grad_()
                y1 = y.clone().requires_grad_()
                x2 = x.clone().requires_grad_()
                y2 = y.clone().requires_grad_()
                
                out = ct.element_mul(x1, y1)
                ans = ct.element_mulTH(x2, y2)
                diff = torch.abs(out - ans).max()
                self.assertLess(diff, 5e-3)

                gradient_start = torch.randn_like(out)
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                diff = torch.abs(x1.grad - x2.grad).max()
                self.assertLess(diff, 1e-2)

                diff = torch.abs(y1.grad - y2.grad).max()
                self.assertLess(diff, 1e-2)

                ct.element_mul_inplace(x, y)
                diff = torch.abs(x - ans).max()
                self.assertLess(diff, 1e-2)
    
    def test_mask_inf(self):
        with torch.cuda.device(3):
            for shape in [
                (3, 5, 6),
                (17, 32, 128),
                (32, 1024, 4096),
                (33, 777, 1232),
                (31, 123, 566),
                (3, 4, 5, 6, 8),
                (21, 66, 5, 3, 2),
                (11, 3, 5, 7, 10)
            ]:
                x = torch.randn(*shape, device="cuda").half()
                mask = torch.randn((shape[0], shape[2]), device="cuda") < 0
                value = float("-inf")
                
                x1 = x.clone().requires_grad_()
                x2 = x.clone().requires_grad_()

                ans = ct.maskTH(x1, mask, value)
                out = ct.mask(x2, mask, value)
                self.assertTrue(torch.isclose(out, ans, 1e-5).all())

                gradient_start = torch.randn_like(out)
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                diff = torch.abs(x1.grad - x2.grad).max()
                self.assertLess(diff, 1e-5)

                ct.mask_inplace(x, mask, value)
                diff = torch.abs(x - ans).max()

                self.assertLess(diff, 1e-5)

    def test_mask_inf(self):
        with torch.cuda.device(3):
            for shape in [
                (3, 5, 6),
                (17, 32, 128),
                (32, 1024, 4096),
                (33, 777, 1232),
                (31, 123, 566),
                (3, 5, 8),
                (21, 66, 2),
                (11, 3, 10)
            ]:
                x = torch.randn(*shape, device="cuda").half()
                mask = torch.randn((shape[0], shape[2]), device="cuda") < 0
                value = 0
                
                x1 = x.clone().requires_grad_()
                x2 = x.clone().requires_grad_()

                ans = ct.maskTH(x1, mask, value)
                out = ct.mask(x2, mask, value)
                self.assertTrue(torch.isclose(out, ans, 1e-5).all())

                gradient_start = torch.randn_like(out)
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                diff = torch.abs(x1.grad - x2.grad).max()
                self.assertLess(diff, 1e-5)

                ct.mask_inplace(x, mask, value)
                diff = torch.abs(x - ans).max()

                self.assertLess(diff, 1e-5)
        
    def test_batched_add(self):
        with torch.cuda.device(3):
            for shape in [
                (3, 5, 6),
                (17, 32, 128),
                (32, 1024, 4096),
                (33, 777, 1232),
                (31, 123, 566),
                (3, 5, 8),
                (21, 66, 2),
                (11, 3, 10)
            ]:
                x = torch.randn(*shape, device="cuda").half()
                y = torch.randn(x.size()[1:], device="cuda").half()

                x1 = x.clone().requires_grad_()
                y1 = y.clone().requires_grad_()
                x2 = x.clone().requires_grad_()
                y2 = y.clone().requires_grad_()

                out = ct.batched_add(x1, y1)
                ans = ct.batched_addTH(x2, y2)
                diff = torch.abs(out - ans).max()
                self.assertLess(diff, 1e-5)

                gradient_start = torch.randn_like(out) / shape[0]
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                diff = torch.abs(x1.grad - x2.grad).max()
                self.assertLess(diff, 1e-2)

                diff = torch.abs(y1.grad - y2.grad).max()
                self.assertLess(diff, 1e-2)

                ct.batched_add_inplace(x, y)
                diff = torch.abs(x - ans).max()
                self.assertLess(diff, 1e-5)
    
    def test_ln_mul_add(self):
        with torch.cuda.device(3):
            for shape in [
                (3, 5, 6),
                (17, 32, 128),
                (32, 1024, 4096),
                (33, 777, 1232),
                (31, 123, 566),
                (3, 5, 8),
                (21, 66, 2),
                (11, 3, 10)
            ]:
                x = torch.randn(*shape, device="cuda").half()
                alpha = (torch.randn(shape[1], device="cuda").half() + 10) / 5
                beta = torch.randn(shape[1], device="cuda").half()

                x1 = x.clone().requires_grad_()
                alpha1 = alpha.clone().requires_grad_()
                beta1 = beta.clone().requires_grad_()

                x2 = x.clone().requires_grad_()
                alpha2 = alpha.clone().requires_grad_()
                beta2 = beta.clone().requires_grad_()

                out = ct.ln_mul_add(x1, alpha1, beta1)
                ans = ct.ln_mul_addTH(x2, alpha2, beta2)
                diff = torch.abs(out - ans).max()
                self.assertLess(diff, 1e-2)

                gradient_start = torch.randn_like(out) / shape[0]
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                diff = torch.abs(x1.grad - x2.grad).max()
                self.assertLess(diff, 5e-2)

                diff = torch.abs(alpha1.grad - alpha2.grad).max()
                self.assertLess(diff, 5e-2)

                diff = torch.abs(beta1.grad - beta2.grad).max()
                self.assertLess(diff, 5e-2)

                ct.ln_mul_add_inplace(x, alpha, beta)
                diff = torch.abs(x - ans).max()

                self.assertLess(diff, 1e-2)
    
    def test_ln_mul(self):
        with torch.cuda.device(3):
            for shape in [
                (3, 5, 6),
                (17, 32, 128),
                (32, 1024, 4096),
                (33, 777, 1232),
                (31, 123, 566),
                (3, 5, 8),
                (21, 66, 2),
                (11, 3, 10)
            ]:
                x = torch.randn(*shape, device="cuda").half()
                alpha = (torch.randn(shape[1], device="cuda").half() + 10) / 5

                x1 = x.clone().requires_grad_()
                alpha1 = alpha.clone().requires_grad_()

                x2 = x.clone().requires_grad_()
                alpha2 = alpha.clone().requires_grad_()

                out = ct.ln_mul(x1, alpha1)
                ans = ct.ln_mulTH(x2, alpha2)
                diff = torch.abs(out - ans).max()
                self.assertLess(diff, 5e-3)

                gradient_start = torch.randn_like(out) / shape[0]
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                diff = torch.abs(x1.grad - x2.grad).max()
                self.assertLess(diff, 5e-2)

                diff = torch.abs(alpha1.grad - alpha2.grad).max()
                self.assertLess(diff, 5e-2)

                ct.ln_mul_inplace(x, alpha)
                diff = torch.abs(x - ans).max()

                self.assertLess(diff, 5e-3)
    
    def test_ln_sub_div(self):
        with torch.cuda.device(3):
            for shape in [
                (3, 5, 6),
                (17, 32, 128),
                (32, 1024, 4096),
                (33, 777, 1232),
                (31, 123, 566),
                (3, 5, 8),
                (21, 66, 2),
                (11, 3, 10)
            ]:
                x = torch.randn(*shape, device="cuda").half()
                alpha = (torch.randn(shape[1], device="cuda").half() + 10) / 5
                beta = torch.randn(shape[1], device="cuda").half()

                out = ct.ln_sub_div(x, alpha, beta)
                ans = ct.ln_sub_divTH(x, alpha, beta)
                diff = torch.abs(out - ans).max()
                self.assertLess(diff, 5e-3)

                ct.ln_sub_div_inplace(x, alpha, beta)
                diff = torch.abs(x - ans).max()
                self.assertLess(diff, 5e-3)

    def test_ln_div(self):
        with torch.cuda.device(3):
            for shape in [
                (3, 5, 6),
                (17, 32, 128),
                (32, 1024, 4096),
                (33, 777, 1232),
                (31, 123, 566),
                (3, 5, 8),
                (21, 66, 2),
                (11, 3, 10)
            ]:
                x = torch.randn(*shape, device="cuda").half()
                alpha = (torch.randn(shape[1], device="cuda").half() + 10) / 5

                out = ct.ln_div(x, alpha)
                ans = ct.ln_divTH(x, alpha)
                diff = torch.abs(out - ans).max()
                self.assertLess(diff, 5e-3)

                ct.ln_div_inplace(x, alpha)
                diff = torch.abs(x - ans).max()
                self.assertLess(diff, 5e-3)
    
    def test_ln_add(self):
        with torch.cuda.device(3):
            for shape in [
                (3, 5, 6),
                (17, 32, 128),
                (32, 1024, 4096),
                (33, 777, 1232),
                (31, 123, 566),
                (3, 5, 8),
                (21, 66, 2),
                (11, 3, 10)
            ]:
                x = torch.randn(*shape, device="cuda").half()
                alpha = (torch.randn(shape[1], device="cuda").half() + 10) / 5

                x1 = x.clone().requires_grad_()
                alpha1 = alpha.clone().requires_grad_()

                x2 = x.clone().requires_grad_()
                alpha2 = alpha.clone().requires_grad_()

                out = ct.ln_add(x1, alpha1)
                ans = ct.ln_addTH(x2, alpha2)
                diff = torch.abs(out - ans).max()
                self.assertLess(diff, 5e-3)

                gradient_start = torch.randn_like(out) / shape[0]
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                diff = torch.abs(x1.grad - x2.grad).max()
                self.assertLess(diff, 5e-2)

                diff = torch.abs(alpha1.grad - alpha2.grad).max()
                self.assertLess(diff, 5e-2)

                ct.ln_add_inplace(x, alpha)
                diff = torch.abs(x - ans).max()

                self.assertLess(diff, 5e-3)