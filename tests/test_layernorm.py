import cpm_kernels.torch as ct
import cpm_kernels.kernels as ck
import torch
import unittest

def normalize_stepTH(x : torch.Tensor, eps : float, rd_mean : bool) -> torch.Tensor:
    old_dtype = x.dtype
    x = x.to(torch.float32)
    var = (x**2).mean(axis=-1, keepdim=True)
    if rd_mean:
        mean = x.mean(axis=-1, keepdim=True)
        var = var - (mean**2)
        x = (x - mean) * torch.rsqrt(var + eps)
    else:
        x = x * torch.rsqrt(var + eps)
    return x.to(old_dtype)

class TestLayerNorm(unittest.TestCase):
    def test_layernorm_unbias(self):
        with torch.cuda.device(4):
            for shape, eps in [
                (768, 1e-5),
                (768, 1e-6),
                (1024, 1e-3),
                (1024, 1e-6)
            ]:
                l1 = ct.LayerNormTH(shape, eps, False)
                l2 = ct.LayerNorm(shape, eps, False)
                state_dict = {
                    "weight": torch.randn(shape) * 0.1 + 1,
                }
                l1.load_state_dict(state_dict)
                l2.load_state_dict(state_dict)
                                
                l1 = l1.to("cuda").half()
                l2 = l2.to("cuda").half()

                for _ in range(16):
                    x_raw = torch.randn((128, shape, 512), device="cuda").half()
                    x1 = x_raw.clone().requires_grad_()
                    x2 = x_raw.requires_grad_()
                    y1 = l1(x1)
                    y2 = l2(x2)

                    self.assertTrue(torch.isclose(y1, y2, 1e-2, 1e-2).all())

                    rd = torch.randn( x_raw.size(), device="cuda").half()
                    y1.backward(gradient=rd)
                    y2.backward(gradient=rd)
                    
                    self.assertTrue(torch.isclose(x1.grad, x2.grad, 1e-2, 1e-2).all())
                    self.assertTrue(torch.isclose(l1.weight.grad, l2.weight.grad, 1e-1, 5e-1).all())
                    
                    l1.weight.grad.zero_()
                    l2.weight.grad.zero_()
    
    def test_layernorm_bias(self):
        with torch.cuda.device(4):
            for shape, eps in [
                (768, 1e-5),
                (768, 1e-6),
                (1024, 1e-3),
                (1024, 1e-6)
            ]:
                l1 = ct.LayerNormTH(shape, eps, True)
                l2 = ct.LayerNorm(shape, eps, True)
                state_dict = {
                    "weight": torch.randn(shape) * 0.1 + 1,
                    "bias": torch.randn(shape),
                }
                l1.load_state_dict(state_dict)
                l2.load_state_dict(state_dict)
                                
                l1 = l1.to("cuda").half()
                l2 = l2.to("cuda").half()

                for _ in range(16):
                    x_raw = torch.randn((128, shape, 512), device="cuda").half()
                    x1 = x_raw.clone().requires_grad_()
                    x2 = x_raw.requires_grad_()
                    y1 = l1(x1)
                    y2 = l2(x2)

                    self.assertTrue(torch.isclose(y1, y2, 1e-2, 1e-2).all())

                    rd = torch.randn( x_raw.size(), device="cuda").half()
                    y1.backward(gradient=rd)
                    y2.backward(gradient=rd)
                    
                    self.assertTrue(torch.isclose(x1.grad, x2.grad, 1e-2, 1e-2).all())

                    self.assertTrue(torch.isclose(l1.weight.grad, l2.weight.grad, 1e-1, 5e-1).all())
                    self.assertTrue(torch.isclose(l1.bias.grad, l2.bias.grad, 1e-2, 1e-2).all())
                    
                    l1.weight.grad.zero_()
                    l2.weight.grad.zero_()
                    l1.bias.grad.zero_()
                    l2.bias.grad.zero_()
    
    def test_normalize(self):
        with torch.cuda.device(4):
            for shape, eps in [
                (768, 1e-5),
                (768, 1e-6),
                (1024, 1e-3),
                (1024, 1e-6)
            ]:
                for i in range(16):
                    x = torch.randn((128, shape, 512), device="cuda").half()
                    ans = ct.normalizeTH(x, eps, i < 8)
                    ct.normalize_inplace(x, eps, i < 8)

                    self.assertTrue(torch.isclose(ans, x, 5e-3, 5e-3).all())
    
    def test_normalize_step(self):
        with torch.cuda.device(4):
            for shape, eps in [
                (768, 1e-5),
                (768, 1e-6),
                (1024, 1e-3),
                (1024, 1e-6)
            ]:
                for i in range(16):
                    x = torch.randn((128, shape), device="cuda").half()
                    ans = torch.empty(128, shape, device="cuda", dtype=torch.half)
                    ck.layernorm_forward(
                        128, shape, 1,
                        x.data_ptr(),
                        ans.data_ptr(),
                        eps,
                        i < 8,
                        torch.cuda.current_stream().cuda_stream
                    )
                    out = torch.empty(128, shape, device="cuda", dtype=torch.half)
                    ck.layernorm_step(
                        128, shape,
                        x.data_ptr(),
                        out.data_ptr(),
                        eps,
                        i < 8,
                        torch.cuda.current_stream().cuda_stream
                    )
                    self.assertTrue(torch.isclose(ans, out, 1e-5, 1e-5).all())

                    ck.layernorm_step_inplace(
                        128, shape,
                        x.data_ptr(),
                        eps,
                        i < 8,
                        torch.cuda.current_stream().cuda_stream
                    )
                    self.assertTrue(torch.isclose(ans, x, 1e-5, 1e-5).all())
