#include "reduce.cuh"
#include "common.h"
#include <cuda_fp16.h>

// block <batch_size>,  thread<min(n, 1024)>
CPM_KERNEL_EXPORT void cu_inplace_add(
    int32_t batch, int32_t n,
    half *x,        // (batch, n)
    const half *y   // (batch, n)
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;
    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            x[base_idx + i] = __hadd(x[base_idx + i], y[base_idx + i]);;
        }
    }
}

// block <batch_size>,  thread<min(n, 1024)>
CPM_KERNEL_EXPORT void cu_batch_add_forward(
    int32_t batch, int32_t n,
    const half *x,        // (batch, n)
    const half *y,        // (n)
    half *out             // (batch, n)
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;
    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            out[base_idx + i] = __hadd(x[base_idx + i], __ldg(y + i + threadIdx.x));
        }
    }
}

// block <n / WARP_SZ>,  thread<WARP_SZ, WARP_SZ>
CPM_KERNEL_EXPORT void cu_batch_add_backward(
    int32_t batch, int32_t n,
    const half *grad_out,   // (batch, n)
    half *grad              // (n)
) {
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int i = 0; i < batch; i += blockDim.y) {
        if (i + threadIdx.y < batch && col < n) {
            sum += __half2float(grad_out[(i + threadIdx.y) * n + col]);
        }
    }
    sum = transposeReduceSum(sum);
    if (threadIdx.y == 0) {
        grad[col] = __float2half(sum);
    }
}