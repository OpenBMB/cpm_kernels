import cpm_kernels.torch as ct
import torch
import unittest

class TestNormalize(unittest.TestCase):
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
                    "weight": torch.randn(shape) * 0.05 + 1,
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

                    diff = (y1 - y2).abs().max()
                    self.assertLess(diff, 1e-1)

                    rd = torch.randn( x_raw.size(), device="cuda").half()
                    y1.backward(gradient=rd)
                    y2.backward(gradient=rd)
                    
                    diff_1 = (x1.grad - x2.grad).abs().max() / 128
                    diff_2 = (l1.weight.grad - l2.weight.grad).abs().max() / 512

                    self.assertLess(diff_1, 1e-2)
                    self.assertLess(diff_2, 1e-2)
                    
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
                    "weight": torch.randn(shape) * 0.05 + 1,
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

                    diff = (y1 - y2).abs().max()
                    self.assertLess(diff, 1e-1)

                    rd = torch.randn( x_raw.size(), device="cuda").half()
                    y1.backward(gradient=rd)
                    y2.backward(gradient=rd)
                    
                    diff_1 = (x1.grad - x2.grad).abs().max() / 128
                    diff_2 = (l1.weight.grad - l2.weight.grad).abs().max() / 512
                    diff_3 = (l1.bias.grad - l2.bias.grad).abs().max() / 512

                    self.assertLess(diff_1, 1e-1)
                    self.assertLess(diff_2, 1e-1)
                    self.assertLess(diff_3, 1e-1)
                    
                    l1.weight.grad.zero_()
                    l2.weight.grad.zero_()
                    l1.bias.grad.zero_()
                    l2.bias.grad.zero_()