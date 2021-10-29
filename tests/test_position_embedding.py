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
                    "embedding.weight": torch.randn(num_buckets, num_heads, device="cuda").half()
                }
                p1.load_state_dict(state_dict)
                p2.load_state_dict(state_dict)

                p1 = p1.cuda().half()
                p2 = p2.cuda().half()

                out = p1(32, 128, 128)
                ans = p2(32, 128, 128)

                diff = torch.abs(out - ans).max()
                self.assertLess(diff, 1e-5)

                gradient_start = torch.randn(out.size(), device="cuda").half() / 32

                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)

                diff = torch.abs(p1.embedding.weight.grad - p2.embedding.weight.grad).max()
                self.assertLess(diff, 1e-1)

                