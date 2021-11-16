#include "reduce.cuh"
#include "common.h"
#include <cuda_fp16.h>

// block <batch_size, n // 1024>   thread<1024>
CPM_KERNEL_EXPORT void cu_gelu_forward(
    int32_t batch, int32_t n,
    const half *mat,    // (batch, n)
    half *out           // (batch, n)
) {
    int32_t col_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (col_idx < n) {
        float x = __half2float(mat[blockIdx.x * n + col_idx]);
        x = 0.5 * x * (1.0 + tanhf(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)));
        out[blockIdx.x * n + col_idx] = __float2half(x);
    }
}

// block <batch_size, n // 1024>   thread<1024>
CPM_KERNEL_EXPORT void cu_gelu_backward(
    int32_t batch, int32_t n,
    const half *grad_out,   // (batch, n)
    const half *mat,        // (batch, n)
    half *grad              // (batch, n)
) {
    int32_t col_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int32_t offset = blockIdx.x * n + col_idx;
    if (col_idx < n) {
        float v = __half2float( grad_out[offset] );
        float x = __half2float( mat[offset] );
        float gelu_grad;
        
        if (-5 < x && x < 5) {
            float x3 = x * x * x;
            float sech2 = 1.0 / coshf(0.797885 * x + 0.0356774 * x3);
            sech2 = sech2 * sech2;

            gelu_grad = 0.5 + (0.398942 * x + 0.0535161 * x3) * sech2 + 0.5 * tanhf(0.797885 * x + 0.0356774 * x3);
        }
        else {
            gelu_grad = x < 0 ? 0 : 1;
        }
        grad[offset] = __float2half(gelu_grad * v);
    }
}
