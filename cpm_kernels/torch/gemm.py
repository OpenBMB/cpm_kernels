import torch
from ..kernels import gemm_calc_scale, gemm_calc_scale_transpose, gemm_round, gemm_round_transpose, gemm_scale, gemm_fp16, gemm_int8

def calc_scale(mat : torch.Tensor, transpose : bool):
    assert mat.is_contiguous() and mat.is_cuda
    if transpose:
        out = torch.empty((mat.size(0), mat.size(2)), dtype=torch.half, device=mat.device)
        gemm_calc_scale_transpose(
            mat.size(0), mat.size(1), mat.size(2),
            mat.data_ptr(), out.data_ptr(), torch.cuda.current_stream().cuda_stream
        )
    else:
        out = torch.empty((mat.size(0), mat.size(1)), dtype=torch.half, device=mat.device)
        gemm_calc_scale(
            mat.size(0), mat.size(1), mat.size(2),
            mat.data_ptr(), out.data_ptr(), torch.cuda.current_stream().cuda_stream
        )
    return out

def round_i8(mat : torch.Tensor, scale : torch.Tensor, transpose : bool):
    assert mat.is_contiguous() and mat.is_cuda
    assert scale.is_contiguous() and scale.is_cuda
    if transpose:
        out = torch.empty(mat.size(), dtype=torch.int8, device=mat.device)
        gemm_round_transpose(
            mat.size(0), mat.size(1), mat.size(2),
            mat.data_ptr(), scale.data_ptr(), out.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
    else:
        out = torch.empty(mat.size(), dtype=torch.int8, device=mat.device)
        gemm_round(
            mat.size(0), mat.size(1), mat.size(2),
            mat.data_ptr(), scale.data_ptr(), out.data_ptr(),
            torch.cuda.current_stream().cuda_stream
        )
    return out

def gemm_and_scale(quantA : torch.Tensor, scaleA : torch.Tensor, quantB : torch.Tensor, scaleB : torch.Tensor, aT, bT) -> torch.Tensor:
    M = quantA.size(2) if aT else quantA.size(1)
    K = quantA.size(1) if aT else quantA.size(2)
    N = quantB.size(1) if bT else quantB.size(2)
    result_i32 = torch.empty((max(quantA.size(0), quantB.size(0)), M, N), dtype=torch.int32, device=quantA.device)
    gemm_int8 (
        N, K, M,
        quantB.size(0), quantA.size(0),
        bT, aT,
        quantB.data_ptr(), quantA.data_ptr(),
        result_i32.data_ptr(),
        torch.cuda.current_stream().cuda_stream
    )
    result_fp = torch.empty((max(quantA.size(0), quantB.size(0)), M, N), dtype=torch.float16, device=quantA.device)
    gemm_scale(
        result_i32.size(0), M, N, 
        result_i32.data_ptr(), 
        scaleA.data_ptr(), scaleB.data_ptr(), 
        result_fp.data_ptr(),
        quantA.size(0) == 1,
        quantB.size(0) == 1,
        torch.cuda.current_stream().cuda_stream
    )
    return result_fp

class GEMMInt8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A : torch.Tensor, aT : bool, B : torch.Tensor, bT : bool):
        """
        Input:
          - A:  (batchA, M, K)
          - B:  (batchB, K, N)
        Output:
          - C:  (batch, M, N)
        """
        assert A.is_cuda and B.is_cuda and A.device == B.device

        scale_A = calc_scale(A, aT)
        scale_B = calc_scale(B, not bT)

        quantized_A = round_i8(A, scale_A, aT)
        quantized_B = round_i8(B, scale_B, not bT)

        result = gemm_and_scale(quantized_A, scale_A, quantized_B, scale_B, aT, bT)

        # save backward
        bw_scale_A = calc_scale(A, not aT)
        bw_scale_B = calc_scale(B, bT)
        bw_quantized_A = round_i8(A, bw_scale_A, not aT)
        bw_quantized_B = round_i8(B, bw_scale_B, bT)
        ctx.save_for_backward(
            bw_scale_A, bw_quantized_A,
            bw_scale_B, bw_quantized_B
        )
        ctx.aT = aT
        ctx.bT = bT
        return result

    @staticmethod
    def backward(ctx, grad_f : torch.Tensor):
        assert grad_f.is_contiguous() and grad_f.is_cuda

        bw_scale_A, bw_quantized_A, bw_scale_B, bw_quantized_B = ctx.saved_tensors
        aT, bT = ctx.aT, ctx.bT
        
        grad_scale_r = calc_scale(grad_f, False)
        grad_quantized_r = round_i8(grad_f, grad_scale_r, False)

        if aT:
            grad_A = gemm_and_scale(bw_quantized_B, bw_scale_B, grad_quantized_r, grad_scale_r, bT, True)
        else:
            grad_A = gemm_and_scale(grad_quantized_r, grad_scale_r, bw_quantized_B, bw_scale_B, False, not bT)


        grad_scale_c = calc_scale(grad_f, True)
        grad_quantized_c = round_i8(grad_f, grad_scale_c, True)

        if bT:
            grad_B = gemm_and_scale(grad_quantized_c, grad_scale_c, bw_quantized_A, bw_scale_A, True, aT)
        else:
            grad_B = gemm_and_scale(bw_quantized_A, bw_scale_A, grad_quantized_c, grad_scale_c, not aT, False)
        
        if bw_scale_A.size(0) == 1 and grad_A.size(0) > 1:
            grad_A = grad_A.sum(dim=0, keepdim=True)
        if bw_scale_B.size(0) == 1 and grad_B.size(0) > 1:
            grad_B = grad_B.sum(dim=0, keepdim=True)
    
        return grad_A, None, grad_B, None 
        

def gemm_pth_fp16(A : torch.Tensor, aT : bool, B : torch.Tensor,bT : bool) -> torch.Tensor:
    M = A.size(2) if aT else A.size(1)
    K = A.size(1) if aT else A.size(2)
    N = B.size(1) if bT else B.size(2)
    out = torch.empty((max(A.size(0), B.size(0)), M, N), dtype=torch.float16, device=A.device)
    gemm_fp16(
        N, K, M,
        B.size(0), A.size(0),
        bT, aT,
        B.data_ptr(), A.data_ptr(),
        out.data_ptr(),
        torch.cuda.current_stream().cuda_stream
    )
    return out

class GEMMFloat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, aT, B, bT):
        ctx.save_for_backward(A, B)
        ctx.aT = aT
        ctx.bT = bT

        return gemm_pth_fp16(A, aT, B, bT)

    @staticmethod
    def backward(ctx, grad_f):
        aT = ctx.aT
        bT = ctx.bT
        A, B = ctx.saved_tensors
        if aT:
            grad_A = gemm_pth_fp16(B, bT, grad_f, True)
        else:
            grad_A = gemm_pth_fp16(grad_f, False, B, not bT)
        
        if bT:
            grad_B = gemm_pth_fp16(grad_f, True, A, aT)
        else:
            grad_B = gemm_pth_fp16(A, not aT, grad_f, False)
        
        if A.size(0) == 1 and grad_A.size(0) > 1:
            grad_A = grad_A.sum(dim=0, keepdim=True)
        if B.size(0) == 1 and grad_B.size(0) > 1:
            grad_B = grad_B.sum(dim=0, keepdim=True)

        return grad_A, None, grad_B, None 

def bmm(A : torch.Tensor, aT : bool, B : torch.Tensor, bT : bool, int8 : bool =False) -> torch.Tensor:
    assert A.ndim == 3
    assert B.ndim == 3
    if int8:
        return GEMMInt8.apply(A, aT, B, bT)
    else:
        return GEMMFloat.apply(A, aT, B, bT)
