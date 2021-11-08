#include "reduce.cuh"
#include "common.h"
#include <cuda_fp16.h>

// block <batch_size>,  thread<min(n, 1024)>,  half n
CPM_KERNEL_EXPORT void cu_arith_element_add (
    int32_t batch, int32_t n,
    const half2 *x,         // (batch, n)
    const half2 *y,         // (batch, n)
    half2 *out
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;
    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            out[base_idx + i] = __hadd2(x[base_idx + i], y[base_idx + i]);;
        }
    }
}

// block <batch_size>,  thread<min(n, 1024)>,  half n
CPM_KERNEL_EXPORT void cu_arith_element_mul (
    int32_t batch, int32_t n,
    const half2 *x,         // (batch, n)
    const half2 *y,         // (batch, n)
    half2 *out
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;
    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            out[base_idx + i] = __hmul2(x[base_idx + i], y[base_idx + i]);;
        }
    }
}

// block <batch_size>,  thread<min(n, 1024)>,   half n
CPM_KERNEL_EXPORT void cu_arith_batch_add_forward(
    int32_t batch, int32_t n,
    const half2 *x,         // (batch, n)
    const half2 *y,         // (n)
    half2 *out              // (batch, n)
) {
    int32_t base_idx = blockIdx.x * n + threadIdx.x;
    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            out[base_idx + i] = __hadd2(x[base_idx + i], __ldg(y + i + threadIdx.x));
        }
    }
}

// block <n / WARP_SZ>,  thread<WARP_SZ, WARP_SZ>
CPM_KERNEL_EXPORT void cu_arith_batch_add_backward(
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
    sum = transposeReduceSum(sum);   // does not support half2
    if (threadIdx.y == 0) {
        grad[col] = __float2half(sum);
    }
}

// block <batch, n>    thread<min(m, 1024)>,  half m
CPM_KERNEL_EXPORT void cu_arith_ln_add(
    int32_t batch, int32_t n, int32_t m,
    const half2 *x,         // (batch, n, m)
    const half *beta,      // (n)
    half2 *out              // (batch, n, m)
) {
    int32_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + threadIdx.x;
    half2 beta_v = __half2half2(__ldg(beta + blockIdx.y));
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            out[base_x_idx + i] = __hadd2(x[base_x_idx + i], beta_v);
        }
    }
}


// block <batch, n>    thread<min(m, 1024)>,  half m
CPM_KERNEL_EXPORT void cu_arith_ln_mul_add(
    int32_t batch, int32_t n, int32_t m,
    const half2 *x,         // (batch, n, m)
    const half *alpha,     // (n) 
    const half *beta,      // (n)
    half2 *out              // (batch, n, m)
) {
    int32_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + threadIdx.x;
    half2 alpha_v = __half2half2(__ldg(alpha + blockIdx.y));
    half2 beta_v = __half2half2(__ldg(beta + blockIdx.y));
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            out[base_x_idx + i] = __hfma2(x[base_x_idx + i], alpha_v, beta_v);
        }
    }
}

// block <batch, n>    thread<min(m, 1024)>,    half m
CPM_KERNEL_EXPORT void cu_arith_ln_mul(
    int32_t batch, int32_t n, int32_t m,
    const half2 *x,         // (batch, n, m)
    const half *alpha,      // (n)
    half2 *out
) {
    int32_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + threadIdx.x;
    half2 alpha_v = __half2half2(__ldg(alpha + blockIdx.y));
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            out[base_x_idx + i] =  __hmul2(x[base_x_idx + i], alpha_v);
        }
    }
}


// block <batch, n>    thread<min(m, 1024)>,   half m
CPM_KERNEL_EXPORT void cu_arith_ln_div(
    int32_t batch, int32_t n, int32_t m,
    const half2 *x,         // (batch, n, m)
    const half *alpha,      // (n)
    half2 *out
) {
    int32_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + threadIdx.x;
    half2 alpha_v = __half2half2(__hdiv(__float2half(1.0), __ldg(alpha + blockIdx.y)));
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            out[base_x_idx + i] = __hmul2(x[base_x_idx + i], alpha_v);
        }
    }
}

// block <batch, n>    thread<min(m, 1024)>,    half m
CPM_KERNEL_EXPORT void cu_arith_ln_sub_div(
    int32_t batch, int32_t n, int32_t m,
    const half2 *x,         // (batch, n, m)
    const half *alpha,      // (n)
    const half *beta,       // (n)
    half2* out
) {
    int32_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + threadIdx.x;
    float rev_alpha = 1.0 / (float)(__ldg(alpha + blockIdx.y));
    float neg_beta = - (float)(__ldg(beta + blockIdx.y)) * rev_alpha;
    
    half2 alpha_v = __float2half2_rn(rev_alpha);   // 1 / alpha
    half2 beta_v = __float2half2_rn(neg_beta);   // - beta / alpha
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            out[base_x_idx + i] = __hfma2(x[base_x_idx + i], alpha_v, beta_v);
        }
    }
}



// block <n>    thread<32, 32>
CPM_KERNEL_EXPORT void cu_arith_ln_mul_backward(
    int32_t batch, int32_t n, int32_t m,
    const half *x,              // (batch, n, m)
    const half *grad_out,       // (batch, n, m)
    half *grad                  // (n)
) {
    /*
    reduce_sum(x * grad_out)
    */
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
CPM_KERNEL_EXPORT void cu_arith_ln_add_backward(
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


