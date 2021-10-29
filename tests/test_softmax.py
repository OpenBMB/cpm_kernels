import unittest
import torch
import cpm_kernels.torch as ct

class TestSoftmax(unittest.TestCase):
    def test_softmax(self):
        with torch.cuda.device(6):
            for shape in [
                (1, 2, 32),
                (4, 128, 128),
                (16, 128, 32),
                (4, 16, 128),
                (123, 512, 321),
                (123, 768, 321),
                (233, 1024, 321),
                (4, 123, 16),
                (4, 321, 16),
            ]:
                x = torch.randn(*shape, device="cuda").half()
                x1 = x.clone().requires_grad_()
                x2 = x.requires_grad_()
                y1 = ct.softmaxTH(x1)
                y2 = ct.softmax(x2)
                diff = (y1 - y2).abs().max()
                self.assertLess(diff, 5e-3)

                rd = torch.randn( *shape, device="cuda").half()
                y1.backward(gradient=rd)
                y2.backward(gradient=rd)
                
                diff_grad = (x1.grad - x2.grad).abs().max()
                self.assertLess(diff_grad, 5e-3)
