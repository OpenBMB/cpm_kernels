from torch._C import dtype
import cpm_kernels.torch as ct
import cpm_kernels.kernels as ck
import torch
import unittest
import math

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

