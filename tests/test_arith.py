import cpm_kernels.torch as ct
import cpm_kernels.kernels as ck
import torch
import unittest
import random

class TestArith(unittest.TestCase):
    def test_global_scale(self):
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
                scale = random.random() * 10

                x1 = x.clone().requires_grad_()
                x2 = x.clone().requires_grad_()
                
                out = ct.global_scale(x1, scale)
                ans = ct.global_scaleTH(x2, scale)
                self.assertTrue(torch.isclose(out, ans, 1e-2, 1e-2).all())

                gradient_start = torch.randn_like(out)
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                self.assertTrue(torch.isclose(x1.grad, x2.grad, 1e-2, 1e-2).all())

                ct.global_scale_inplace(x, scale)
                self.assertTrue(torch.isclose(x, ans, 1e-2, 1e-2).all())

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
                self.assertTrue(torch.isclose(out, ans, 1e-2, 1e-2).all())

                gradient_start = torch.randn_like(out)
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                self.assertTrue(torch.isclose(x1.grad, x2.grad, 1e-2, 1e-2).all())

                self.assertTrue(torch.isclose(y1.grad, y2.grad, 1e-2, 1e-2).all())

                ct.element_add_inplace(x, y)
                self.assertTrue(torch.isclose(x, ans, 1e-2, 1e-2).all())

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
                self.assertTrue(torch.isclose(out, ans, 1e-2, 1e-2).all())

                gradient_start = torch.randn_like(out)
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                self.assertTrue(torch.isclose(x1.grad, x2.grad, 1e-2, 1e-2).all())

                self.assertTrue(torch.isclose(y1.grad, y2.grad, 1e-2, 1e-2).all())

                ct.element_mul_inplace(x, y)
                self.assertTrue(torch.isclose(x, ans, 1e-2, 1e-2).all())
    
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
                self.assertTrue(torch.isclose(out, ans, 1e-2, 1e-2).all())

                gradient_start = torch.randn_like(out)
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                self.assertTrue(torch.isclose(x1.grad, x2.grad, 1e-2, 1e-2).all())

                ct.mask_inplace(x, mask, value)
                self.assertTrue(torch.isclose(x, ans, 1e-2, 1e-2).all())

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
                self.assertTrue(torch.isclose(out, ans, 1e-2).all())

                gradient_start = torch.randn_like(out)
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                self.assertTrue(torch.isclose(x1.grad, x2.grad, 1e-2, 1e-2).all())

                ct.mask_inplace(x, mask, value)
                self.assertTrue(torch.isclose(x, ans, 1e-2, 1e-2).all())
        
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
                self.assertTrue(torch.isclose(out, ans, 1e-2, 1e-2).all())

                gradient_start = torch.randn_like(out) / shape[0]
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                self.assertTrue(torch.isclose(x1.grad, x2.grad, 1e-2, 1e-2).all())

                self.assertTrue(torch.isclose(y1.grad, y2.grad, 1e-2, 1e-2).all())

                ct.batched_add_inplace(x, y)
                self.assertTrue(torch.isclose(x, ans, 1e-2, 1e-2).all())
    
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
                alpha = torch.randn(shape[1], device="cuda").half()
                beta = torch.randn(shape[1], device="cuda").half()

                x1 = x.clone().requires_grad_()
                alpha1 = alpha.clone().requires_grad_()
                beta1 = beta.clone().requires_grad_()

                x2 = x.clone().requires_grad_()
                alpha2 = alpha.clone().requires_grad_()
                beta2 = beta.clone().requires_grad_()

                out = ct.ln_mul_add(x1, alpha1, beta1)
                ans = ct.ln_mul_addTH(x2, alpha2, beta2)
                self.assertTrue(torch.isclose(out, ans, 1e-2, 1e-2).all())

                gradient_start = torch.randn_like(out) / shape[0]
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)
                self.assertTrue(torch.isclose(x1.grad, x2.grad, 1e-2, 1e-2).all())
                self.assertTrue(torch.isclose(alpha1.grad, alpha2.grad, 1e-2, 1e-2).all())

                self.assertTrue(torch.isclose(beta1.grad, beta2.grad, 1e-2, 1e-2).all())

                ct.ln_mul_add_inplace(x, alpha, beta)
                self.assertTrue(torch.isclose(x, ans, 1e-2, 1e-2).all())
    
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
                self.assertTrue(torch.isclose(out, ans, 1e-2, 1e-2).all())

                gradient_start = torch.randn_like(out) / shape[0]
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                self.assertTrue(torch.isclose(x1.grad, x2.grad, 1e-2, 1e-2).all())

                self.assertTrue(torch.isclose(alpha1.grad, alpha2.grad, 1e-2, 1e-2).all())

                ct.ln_mul_inplace(x, alpha)
                self.assertTrue(torch.isclose(x, ans, 1e-2, 1e-2).all())
    
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
                self.assertTrue(torch.isclose(out, ans, 1e-2, 1e-2).all())

                ct.ln_sub_div_inplace(x, alpha, beta)
                self.assertTrue(torch.isclose(x, ans, 1e-2, 1e-2).all())

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
                self.assertTrue(torch.isclose(out, ans, 1e-2, 1e-2).all())

                ct.ln_div_inplace(x, alpha)
                self.assertTrue(torch.isclose(x, ans, 1e-2, 1e-2).all())
    
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
                self.assertTrue(torch.isclose(out, ans, 1e-2, 1e-2).all())

                gradient_start = torch.randn_like(out) / shape[0]
                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                self.assertTrue(torch.isclose(x1.grad, x2.grad, 1e-2, 1e-2).all())

                self.assertTrue(torch.isclose(alpha1.grad, alpha2.grad, 1e-2, 1e-2).all())

                ct.ln_add_inplace(x, alpha)
                self.assertTrue(torch.isclose(x, ans, 1e-2, 1e-2).all())
    
    def test_batched_mul_add(self):
        with torch.cuda.device(3):
            for shape in [
                (3 * 5, 6),
                (17 * 32, 128),
                (32 * 1024, 4096),
                (33 * 777, 1232),
                (31 * 123, 566),
                (3 * 5, 8),
                (21 * 66, 2),
                (11 * 3, 10)
            ]:
                x = torch.randn(*shape, 2, device="cuda").half()
                alpha = (torch.randn(shape[1], device="cuda").half() + 10) / 5
                beta = torch.randn(shape[1], device="cuda").half()

                ans = torch.empty(shape + (2,), dtype=torch.half, device="cuda")
                ck.arith_ln_mul_add(
                    shape[0], shape[1], 2,
                    x.data_ptr(),
                    alpha.data_ptr(),
                    beta.data_ptr(),
                    ans.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )

                x_0, x_1 = x[:, :, 0].contiguous(), x[:, :, 1].contiguous()
                
                out = torch.empty( shape, dtype=torch.half, device="cuda")
                ck.arith_batch_mul_add(
                    shape[0], shape[1],
                    x_0.data_ptr(),
                    alpha.data_ptr(),
                    beta.data_ptr(),
                    out.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                self.assertTrue(torch.isclose(out, ans[:, :, 0], 1e-5, 1e-5).all())

                ck.arith_batch_mul_add(
                    shape[0], shape[1],
                    x_1.data_ptr(),
                    alpha.data_ptr(),
                    beta.data_ptr(),
                    out.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                self.assertTrue(torch.isclose(out, ans[:, :, 1], 1e-5, 1e-5).all())
    
    def test_batched_mul(self):
        with torch.cuda.device(3):
            for shape in [
                (3 * 5, 6),
                (17 * 32, 128),
                (32 * 1024, 4096),
                (33 * 777, 1232),
                (31 * 123, 566),
                (3 * 5, 8),
                (21 * 66, 2),
                (11 * 3, 10)
            ]:
                x = torch.randn(*shape, 2, device="cuda").half()
                alpha = torch.randn(shape[1], device="cuda").half()

                ans = torch.empty(shape + (2,), dtype=torch.half, device="cuda")
                ck.arith_ln_mul(
                    shape[0], shape[1], 2,
                    x.data_ptr(),
                    alpha.data_ptr(),
                    ans.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )

                x_0, x_1 = x[:, :, 0].contiguous(), x[:, :, 1].contiguous()

                out = torch.empty( shape, dtype=torch.half, device="cuda")
                ck.arith_batch_mul(
                    shape[0], shape[1],
                    x_0.data_ptr(),
                    alpha.data_ptr(),
                    out.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                self.assertTrue(torch.isclose(out, ans[:, :, 0], 1e-5, 1e-5).all())
                ck.arith_batch_mul(
                    shape[0], shape[1],
                    x_1.data_ptr(),
                    alpha.data_ptr(),
                    out.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                self.assertTrue(torch.isclose(out, ans[:, :, 1], 1e-5, 1e-5).all())