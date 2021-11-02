import cpm_kernels.torch as ct
import torch
import unittest

TEST_CASE = [
    (32, 12, 128, False),
    (32, 12, 128, True),
    (32, 24, 256, True),
    (128, 64, 128, True),
    (16, 64, 256, False),
    (16, 16, 512, True),
]

class TestPositionEmbedding(unittest.TestCase):
    def test_position_embedding(self):
        with torch.cuda.device(5):
            for num_buckets, num_heads, max_distance, bidi in TEST_CASE:
                p1 = ct.PositionEmbedding(num_heads, num_buckets, max_distance, bidi)
                p2 = ct.PositionEmbeddingTH(num_heads, num_buckets, max_distance, bidi)
                state_dict = {
                    "weight": torch.randn(num_heads, num_buckets, device="cuda").half()
                }
                p1.load_state_dict(state_dict)
                p2.load_state_dict(state_dict)

                p1 = p1.cuda().half()
                p2 = p2.cuda().half()

                out = p1(128, 128)
                ans = p2(128, 128)

                diff = torch.abs(out - ans).max()
                self.assertLess(diff, 1e-5)

                gradient_start = torch.randn(out.size(), device="cuda").half()
                if not bidi:
                    mask = torch.arange(128, device="cuda")[:, None] <= torch.arange(128, device="cuda")[None, :]
                    ct.inplace_mask(gradient_start, mask[None, :, :].repeat(num_heads, 1, 1), 0)

                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)
                diff = torch.abs(p1.weight.grad - p2.weight.grad).max()
                self.assertLess(diff, 1e-5)

                