import cpm_kernels.torch as ct
import cpm_kernels.kernels as ck
import torch
import unittest
import math

class TestGemv(unittest.TestCase):
    def test_gemv_int8(self):
        with torch.cuda.device(2):
            for _ in range(10):
                BATCH = 16
                N = 4444
                M = 8888
                ssk = math.sqrt(math.sqrt(M))
                mat = torch.randn(N, M, dtype=torch.half, device="cuda")
                vec = torch.randn(BATCH, M, dtype=torch.half, device="cuda")

                mat_scale = torch.empty(N, dtype=torch.half, device="cuda")
                mat_quant = torch.empty(N, M, dtype=torch.int8, device="cuda")
                ck.gemm_calc_scale(
                    1, N, M,
                    mat.data_ptr(),
                    mat_scale.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                ck.gemm_round(
                    1, N, M,
                    mat.data_ptr(),
                    mat_scale.data_ptr(),
                    mat_quant.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                vec_scale = torch.empty(BATCH, dtype=torch.half, device="cuda")
                ck.gemv_calc_scale(
                    BATCH, M,
                    vec.data_ptr(),
                    vec_scale.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                vec_quant = torch.empty(BATCH, M, dtype=torch.int8, device="cuda")
                ck.gemv_round(
                    BATCH, M,
                    vec.data_ptr(),
                    vec_scale.data_ptr(),
                    vec_quant.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                out = torch.empty(BATCH, N, dtype=torch.half, device="cuda")
                ck.gemv_broadcast_mat_int8(
                    BATCH, N, M,
                    mat_scale.data_ptr(),
                    mat_quant.data_ptr(),
                    vec_scale.data_ptr(),
                    vec_quant.data_ptr(),
                    out.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                ans = ct.bmm( vec.unsqueeze(0), False, mat.unsqueeze(0), True , int8=True)
                self.assertTrue(torch.isclose(out, ans, 1e-3, 1e-3).all())

    def test_gemv_fp16(self):
        with torch.cuda.device(2):
            for _ in range(10):
                BATCH = 16
                N = 2222
                M = 128
                ssk = math.sqrt(math.sqrt(M))
                mat = torch.randn(BATCH, N, M, dtype=torch.half, device="cuda") / ssk
                vec = torch.randn(BATCH, M, 2, dtype=torch.half, device="cuda") / ssk
                vec_0 = vec[:, :, 0].clone()

                out = torch.empty(BATCH, N, dtype=torch.half, device="cuda")
                ck.gemv_fp16(
                    BATCH, N, M,
                    mat.data_ptr(),
                    vec_0.data_ptr(),
                    out.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )
                ans = ct.bmm( mat, False, vec, False , int8=False)[:, :, 0]
                self.assertTrue(torch.isclose(out, ans, 5e-2, 5e-2).all())

    def test_gemv_fp16_transpose(self):
        with torch.cuda.device(2):
            for _ in range(10):
                BATCH = 16
                N = 128
                M = 2222
                ssk = math.sqrt(math.sqrt(M))
                mat = torch.randn(BATCH, M, N, dtype=torch.half, device="cuda") / ssk
                vec = torch.randn(BATCH, M, 2, dtype=torch.half, device="cuda") / ssk
                vec_0 = vec[:, :, 0].clone()

                out = torch.zeros(BATCH, N, dtype=torch.half, device="cuda")
                ck.gemv_fp16_transpose(
                    BATCH, N, M,
                    mat.data_ptr(),
                    vec_0.data_ptr(),
                    out.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )

                ans = ct.bmm( mat, True, vec, False , int8=False)[:, :, 0]
                self.assertTrue(torch.isclose(out, ans, 5e-2, 5e-2).all())

    def test_gemv_logits(self):
        with torch.cuda.device(2):
            for _ in range(10):
                BATCH = 16
                N = 22222
                M = 4444
                ssk = math.sqrt(math.sqrt(M))
                mat = torch.randn(N, M, dtype=torch.half, device="cuda") / ssk
                vec = torch.randn(BATCH, M, dtype=torch.half, device="cuda") / ssk

            
                out = torch.empty(BATCH, N, dtype=torch.half, device="cuda")
                ck.gemv_broadcast_mat_fp16(
                    BATCH, N, M,
                    mat.data_ptr(),
                    vec.data_ptr(),
                    out.data_ptr(),
                    torch.cuda.current_stream().cuda_stream
                )

                ans = ct.bmm( vec.unsqueeze(0), False, mat.unsqueeze(0), True , int8=False)
                self.assertTrue(torch.isclose(out, ans, 5e-2, 5e-2).all())