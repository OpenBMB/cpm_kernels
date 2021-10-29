#include "reduce.cuh"
#include <cuda_fp16.h>
#include "common.h"

// block <batch, n>    thread<min(m, 1024)>
CPM_KERNEL_EXPORT void cu_inplace_mul_add(
    int32_t batch, int32_t n, int32_t m,
    half *x,    // (batch, n, m)
    const half *alpha,  // (n) 
    const half *beta    // (n)
) {
    int32_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + threadIdx.x;
    float alpha_v = __half2float(__ldg(alpha + blockIdx.y));
    float beta_v = __half2float(__ldg(beta + blockIdx.y));
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            float v = x[base_x_idx + i];
            v =  v * alpha_v + beta_v;
            x[base_x_idx + i] = __float2half(v);
        }
    }
}

// block <batch, n>    thread<min(m, 1024)>
CPM_KERNEL_EXPORT void cu_inplace_mul(
    int32_t batch, int32_t n, int32_t m,
    half *x,            // (batch, n, m)
    const half *alpha   // (n)
) {
    int32_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + threadIdx.x;
    float alpha_v = __half2float(__ldg(alpha + blockIdx.y));
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            float v = x[base_x_idx + i];
            x[base_x_idx + i] = __float2half(v * alpha_v);
        }
    }
}


// block <batch, n>    thread<min(m, 1024)>
CPM_KERNEL_EXPORT void cu_inplace_div(
    int32_t batch, int32_t n, int32_t m,
    half *x,            // (batch, n, m)
    const half *alpha   // (n)
) {
    int32_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + threadIdx.x;
    float alpha_v = __half2float(__ldg(alpha + blockIdx.y));
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            float v = x[base_x_idx + i];
            x[base_x_idx + i] = __float2half(v / alpha_v);
        }
    }
}

// block <batch, n>    thread<min(m, 1024)>
CPM_KERNEL_EXPORT void cu_inplace_sub_div(
    int32_t batch, int32_t n, int32_t m,
    half *x,            // (batch, n, m)
    const half *alpha,  // (n)
    const half *beta    // (n)
) {
    int32_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + threadIdx.x;
    float alpha_v = __half2float(__ldg(alpha + blockIdx.y));
    float beta_v = __half2float(__ldg(beta + blockIdx.y));
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            float v = x[base_x_idx + i];
            v =  (v - beta_v) / alpha_v;
            x[base_x_idx + i] = __float2half(v);
        }
    }
}



// block <n>    thread<32, 32>
CPM_KERNEL_EXPORT void cu_inplace_mul_backward(
    int32_t batch, int32_t n, int32_t m,
    const half *x,              // (batch, n, m)
    const half *grad_out,       // (batch, n, m)
    half *grad                  // (n)
) {
    float sum = 0;
    for (int b = 0; b < batch; b += blockDim.x) {
        int batch_idx = b + threadIdx.y;
        int base_idx = batch_idx * n * m + blockIdx.x * m + threadIdx.x;
        for (int i = 0; i < m; i += blockDim.y) {
            if (batch_idx < batch && i + threadIdx.x < m) {
                sum += (float)__ldg(grad_out + base_idx + i) * (float)__ldg(x + base_idx + i);
            }
        }
    }
    sum = warpReduceSum(sum);
    __shared__ float shared[32];
    if (threadIdx.x == 0) {
        shared[threadIdx.y] = sum;
    }
    __syncthreads();
    sum = warpReduceSum(shared[threadIdx.x]);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        grad[blockIdx.x] = __float2half(sum);
    }
}


// block <n>    thread<32, 32>
CPM_KERNEL_EXPORT void cu_inplace_add_backward(
    int32_t batch, int32_t n, int32_t m,
    const half *grad_out,       // (batch, n, m)
    half *grad                  // (n)
) {
    float sum = 0;
    for (int b = 0; b < batch; b += blockDim.x) {
        int batch_idx = b + threadIdx.y;
        int base_idx = batch_idx * n * m + blockIdx.x * m + threadIdx.x;
        for (int i = 0; i < m; i += blockDim.y) {
            if (batch_idx < batch && i + threadIdx.x < m) {
                sum += (float)__ldg(grad_out + base_idx + i);
            }
        }
    }
    sum = warpReduceSum(sum);
    __shared__ float shared[32];
    if (threadIdx.x == 0) {
        shared[threadIdx.y] = sum;
    }
    __syncthreads();
    sum = warpReduceSum(shared[threadIdx.x]);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        grad[blockIdx.x] = __float2half(sum);
    }
}


