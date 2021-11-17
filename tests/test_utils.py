from torch._C import dtype
import cpm_kernels.kernels as ck
import torch
import unittest
import math
import random

class TestUtils(unittest.TestCase):
    def test_gemv_int8(self):
        with torch.cuda.device(2):
            a = torch.randn(33, 55, 58, dtype=torch.half, device="cuda")
            vec = []
            for i in range(55):
                b = torch.randn(33, 58, dtype=torch.half, device="cuda")
                ck.copy_data_to_kv(
                    33, 55, 58,
                    b.data_ptr(),
                    a.data_ptr(),
                    i,
                    torch.cuda.current_stream().cuda_stream
                )
                vec.append(b)
            ans = torch.stack(vec, dim=1)
            diff = torch.abs(a - ans).max()
            self.assertLess(diff, 1e-5)

    def test_array_add(self):
        with torch.cuda.device(2):
            a = torch.randint(0, 10, (128,), dtype=torch.int32, device="cuda")

            for _ in range(128):
                pos = random.randint(0, a.size(0) - 1)
                val = random.randint(-10, 10)
                old_val = a[pos].item()
                ck.utils.array_add(a.data_ptr(), pos, val, torch.cuda.current_stream().cuda_stream)

                self.assertEqual(a[pos], old_val + val)
    
    def test_justify_logits(self):
        with torch.cuda.device(2):
            for batch, n in [
                (3, 128),
                (16, 333),
                (1, 2341),
                (15, 2341),
            ]:
                freq = torch.randint(0, 1, (batch, n), dtype=torch.int32, device="cuda")
                logits = torch.randn(batch, n, dtype=torch.float16, device="cuda")
                
                temp = (random.random() + 1) / 2
                freq_p = random.random()
                prec_p = random.random()

                ans = logits / temp - freq_p * freq.half() - prec_p * (freq > 0).half()

                ck.utils.adjustify_logits(
                    batch, n,
                    logits.data_ptr(),
                    temp, freq_p, prec_p,
                    freq.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )

                self.assertTrue(torch.isclose(logits, ans, 1e-3, 1e-3).all())

    def test_extend_buffer(self):
        with torch.cuda.device(2):
            for batch, old_size, nw_size in [
                (3, 128, 256),
                (16, 333, 334),
                (1, 2341, 4567),
                (15, 2341, 3451),
            ]:
                x = torch.randn(batch, old_size, dtype=torch.float16, device="cuda")
                nw_buf = torch.empty(batch, nw_size, dtype=torch.float16, device="cuda")
                ck.utils.copy_extend_buffer(
                    batch, old_size, nw_size,
                    x.data_ptr(),
                    nw_buf.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                self.assertTrue(torch.isclose(x, nw_buf[:, :old_size], 1e-5, 1e-5).all())

        
