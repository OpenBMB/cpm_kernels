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
        with torch.cuda.device(0):
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

                diff = torch.abs(out - ans).max()
                self.assertLess(diff, 1e-5)

                graident_start = torch.randn(batch, hidden_size, seq_len, dtype=torch.half).to("cuda") / batch

                out.backward(gradient=graident_start)
                ans.backward(gradient=graident_start)

                diff = torch.abs(cpm_emb.weight.grad - pth_emb.weight.grad).max()
                self.assertLess(diff, 1e-3)
                
    def test_embedding_step(self):
        with torch.cuda.device(0):
            for args in TEST_CASE:
                vocab_size, hidden_size, batch, _ = args
                weight = torch.randn(vocab_size, hidden_size, dtype=torch.half, device="cuda")
                ipt = torch.randint(0, vocab_size, (batch,), dtype=torch.long).to("cuda")

                ans = torch.embedding(
                    weight,
                    ipt
                )
                out = torch.empty((batch, hidden_size), dtype=torch.half, device="cuda")
                ck.embedding_step(
                    batch, hidden_size,
                    ipt.to(torch.int32).data_ptr(),
                    weight.data_ptr(),
                    out.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                diff = torch.abs(out - ans).max()
                self.assertLess(diff, 1e-5)
                