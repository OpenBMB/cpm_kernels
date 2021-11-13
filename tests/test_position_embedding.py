import cpm_kernels.torch as ct
import cpm_kernels.kernels as ck
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

                self.assertTrue(torch.isclose(out, ans, 1e-4, 1e-4).all())

                gradient_start = torch.randn(out.size(), device="cuda").half()
                if not bidi:
                    mask = torch.arange(128, device="cuda")[:, None] <= torch.arange(128, device="cuda")[None, :]
                    gradient_start = torch.where(
                        mask[None, :, :].repeat(num_heads, 1, 1),
                        gradient_start,
                        torch.zeros_like(gradient_start),
                    )

                out.backward(gradient=gradient_start)
                ans.backward(gradient=gradient_start)
                self.assertTrue(torch.isclose(p1.weight.grad, p2.weight.grad, 1e-3, 1e-3).all())

    def test_position_embedding_step(self):
        with torch.cuda.device(5):
            for num_buckets, num_heads, max_distance, bidi in TEST_CASE:
                weight = torch.randn(num_heads, num_buckets, device="cuda", dtype=torch.half)
                p2 = ct.PositionEmbeddingTH(num_heads, num_buckets, max_distance, bidi)
                p2.load_state_dict({"weight": weight })
                p2 = p2.cuda().half()

                ans = p2(128, 128)

                mapping = torch.empty(max_distance, dtype=torch.int32, device="cuda")
                ck.position_embedding_init(
                    num_buckets, max_distance, 
                    mapping.data_ptr(),
                    bidi,
                    torch.cuda.current_stream().cuda_stream
                )
                for i in range(128):
                    out = torch.empty(num_heads, 128, dtype=torch.half, device="cuda")
                    ck.position_embedding_step(
                        i, 128, num_buckets, max_distance, num_heads,
                        mapping.data_ptr(),
                        weight.data_ptr(),
                        out.data_ptr(),
                        bidi,
                        torch.cuda.current_stream().cuda_stream
                    )
                    self.assertTrue(torch.isclose(out, ans[:, :, i], 1e-4, 1e-4).all())
