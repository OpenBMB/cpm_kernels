import cpm_kernels.torch as ct
import cpm_kernels.kernels as ck
import torch
import unittest


TEST_CASE = [
    (10, 3, 1, 5),
    (5000, 1024, 16, 128),
    (999, 888, 77, 6),
    (123, 321, 123, 312),
    (25012, 4096, 32, 512)
]

class TestEmbedding(unittest.TestCase):
    def test_embedding(self):
        with torch.cuda.device(2):
            # Test the embedding layer.
            for args in TEST_CASE:
                vocab_size, hidden_size, batch, seq_len = args
                cpm_emb = ct.Embedding(vocab_size, hidden_size)
                pth_emb = ct.EmbeddingTH(vocab_size, hidden_size)
                state_dict = {
                    'weight': torch.randn(vocab_size, hidden_size, dtype=torch.half),
                }
                cpm_emb.load_state_dict(state_dict)
                pth_emb.load_state_dict(state_dict)

                cpm_emb = cpm_emb.to("cuda")
                pth_emb = pth_emb.to("cuda")

                ipt = torch.randint(0, vocab_size, (batch, seq_len), dtype=torch.long).to("cuda")
                out = cpm_emb(ipt.to(torch.int32))
                ans = pth_emb(ipt)

                self.assertTrue(torch.isclose(out, ans, 1e-3, 1e-3).all())

                graident_start = torch.randn(batch, hidden_size, seq_len, dtype=torch.half).to("cuda") / batch

                out.backward(gradient=graident_start)
                ans.backward(gradient=graident_start)

                self.assertTrue(torch.isclose(cpm_emb.weight.grad, pth_emb.weight.grad, 1e-3, 1e-3).all())
                
    def test_embedding_step(self):
        with torch.cuda.device(2):
            for args in TEST_CASE:
                vocab_size, hidden_size, batch, _ = args
                weight = torch.randn(vocab_size, hidden_size, dtype=torch.half, device="cuda")
                ipt = torch.randint(0, vocab_size, (batch,), dtype=torch.int32).to("cuda")

                ans = torch.empty((batch, hidden_size), dtype=torch.half, device="cuda")
                ck.embedding_forward(
                    batch, hidden_size, 1,
                    ipt.data_ptr(),
                    weight.data_ptr(),
                    ans.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                out = torch.empty((batch, hidden_size), dtype=torch.half, device="cuda")
                ck.embedding_step(
                    batch, hidden_size,
                    ipt.data_ptr(),
                    weight.data_ptr(),
                    out.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                self.assertTrue(torch.isclose(out, ans, 1e-5, 1e-5).all())
                