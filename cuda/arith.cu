#include "reduce.cuh"
#include "common.h"
#include <cuda_fp16.h>

// block <n // 1024>,   thread <min(n, 1024)>
CPM_KERNEL_EXPORT void cu_arith_global_scale(
    int64_t n,
    const half *inp,    // (n,)
    float scale,
    half *out           // (n,)
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__half2float(inp[idx]) * scale);
    }
}

// block <batch_size，n // 1024>,  thread<min(n, 1024)>,  half n
CPM_KERNEL_EXPORT void cu_arith_element_add (
    int64_t batch, int64_t n,
    const half2 *x,         // (batch, n)
    const half2 *y,         // (batch, n)
    half2 *out
) {
    int64_t col = threadIdx.x + blockIdx.y * blockDim.x;
    int64_t pos = blockIdx.x * n + col;
    if (col < n) {
        out[pos] = __hadd2(x[pos], y[pos]);
    }
}

// block <batch_size, n // 1024>,  thread<min(n, 1024)>,  half n
CPM_KERNEL_EXPORT void cu_arith_element_mul (
    int64_t batch, int64_t n,
    const half2 *x,         // (batch, n)
    const half2 *y,         // (batch, n)
    half2 *out
) {
    int64_t col = threadIdx.x + blockIdx.y * blockDim.x;
    int64_t pos = blockIdx.x * n + col;
    if (col < n) {
        out[pos] = __hmul2(x[pos], y[pos]);
    }
}

// block <batch_size, n // 1024>,  thread<min(n, 1024)>,   half n
CPM_KERNEL_EXPORT void cu_arith_batch_add_forward(
    int64_t batch, int64_t n,
    const half2 *x,         // (batch, n)
    const half2 *y,         // (n)
    half2 *out              // (batch, n)
) {
    int64_t col = threadIdx.x + blockIdx.y * blockDim.x;
    int64_t pos = blockIdx.x * n + col;
    if (col < n) {
        out[pos] = __hadd2(x[pos], __ldg(y + col));
    }
}

// block <n / WARP_SZ>,  thread<WARP_SZ, WARP_SZ>
CPM_KERNEL_EXPORT void cu_arith_batch_add_backward(
    int64_t batch, int64_t n,
    const half *grad_out,   // (batch, n)
    half *grad              // (n)
) {
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
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

// block <batch, n, m // 1024>    thread<min(m, 1024)>,  half m
CPM_KERNEL_EXPORT void cu_arith_ln_add(
    int64_t batch, int64_t n, int64_t m,
    const half2 *x,         // (batch, n, m)
    const half *beta,      // (n)
    half2 *out              // (batch, n, m)
) {
    int64_t col = threadIdx.x + blockIdx.z * blockDim.x;
    int64_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + col;
    half2 beta_v = __half2half2(__ldg(beta + blockIdx.y));

    if (col < m) {
        out[base_x_idx] = __hadd2(x[base_x_idx], beta_v);
    }
}


// block <batch, n, m // 1024>    thread<min(m, 1024)>,  half m
CPM_KERNEL_EXPORT void cu_arith_ln_mul_add(
    int64_t batch, int64_t n, int64_t m,
    const half2 *x,         // (batch, n, m)
    const half *alpha,     // (n) 
    const half *beta,      // (n)
    half2 *out              // (batch, n, m)
) {
    int64_t col = threadIdx.x + blockIdx.z * blockDim.x;
    int64_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + col;
    half2 alpha_v = __half2half2(__ldg(alpha + blockIdx.y));
    half2 beta_v = __half2half2(__ldg(beta + blockIdx.y));

    if (col < m) {
        out[base_x_idx] = __hfma2(x[base_x_idx], alpha_v, beta_v);
    }
}

// block <batch, n, m // 1024>    thread<min(m, 1024)>,    half m
CPM_KERNEL_EXPORT void cu_arith_ln_mul(
    int64_t batch, int64_t n, int64_t m,
    const half2 *x,         // (batch, n, m)
    const half *alpha,      // (n)
    half2 *out
) {
    int64_t col = threadIdx.x + blockIdx.z * blockDim.x;
    int64_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + col;
    half2 alpha_v = __half2half2(__ldg(alpha + blockIdx.y));
    if (col < m) {
        out[base_x_idx] =  __hmul2(x[base_x_idx], alpha_v);
    }
}


// block <batch, n, m // 1024>    thread<min(m, 1024)>,   half m
CPM_KERNEL_EXPORT void cu_arith_ln_div(
    int64_t batch, int64_t n, int64_t m,
    const half2 *x,         // (batch, n, m)
    const half *alpha,      // (n)
    half2 *out
) {
    int64_t col = threadIdx.x + blockIdx.z * blockDim.x;
    int64_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + col;
    half2 alpha_v = __half2half2(__hdiv(__float2half(1.0), __ldg(alpha + blockIdx.y)));
    if (col < m) {
        out[base_x_idx] = __hmul2(x[base_x_idx], alpha_v);
    }
}

// block <batch, n, m // 1024>    thread<min(m, 1024)>,    half m
CPM_KERNEL_EXPORT void cu_arith_ln_sub_div(
    int64_t batch, int64_t n, int64_t m,
    const half2 *x,         // (batch, n, m)
    const half *alpha,      // (n)
    const half *beta,       // (n)
    half2* out
) {
    int64_t col = threadIdx.x + blockIdx.z * blockDim.x;
    int64_t base_x_idx = (blockIdx.x * n + blockIdx.y) * m + col;
    float rev_alpha = 1.0 / (float)(__ldg(alpha + blockIdx.y));
    float neg_beta = - (float)(__ldg(beta + blockIdx.y)) * rev_alpha;
    
    half2 alpha_v = __float2half2_rn(rev_alpha);   // 1 / alpha
    half2 beta_v = __float2half2_rn(neg_beta);   // - beta / alpha
    if (col < m) {
        out[base_x_idx] = __hfma2(x[base_x_idx], alpha_v, beta_v);
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
    float local_sum = 0;
    for (int b = 0; b < batch; b += WARP_SZ * WARP_SZ) {
        float inner_sum = 0;
        for (int inner_b = 0; inner_b < WARP_SZ * WARP_SZ && inner_b + b < batch; inner_b += WARP_SZ) {
            int batch_idx = b + inner_b + threadIdx.y;
            int base_idx = batch_idx * n * m + blockIdx.x * m + threadIdx.x;

            float batch_sum = 0;
            for (int i = 0; i < m; i += WARP_SZ * WARP_SZ) {
                float inner_v = 0;
                for (int j = 0; j < WARP_SZ * WARP_SZ && i + j < m; j += WARP_SZ) {
                    float v = 0;
                    if (batch_idx < batch && i + j + threadIdx.x < m) {
                        v = (float)grad_out[base_idx + i + j] * (float)x[base_idx + i + j];
                    }
                    v = warpReduceSum(v);           // sum of 32 elements
                    v = __shfl_sync(0xFFFFFFFF, v, 0); // broadcast to all threads in warp
                    if (threadIdx.x * WARP_SZ == j) inner_v = v;
                }
                inner_v = warpReduceSum(inner_v);   // sum of 1024 elements

                // stores the sum of batch (b + inner_b + threadIdx.y) in (0, threadIdx.y)
                batch_sum += inner_v;   // sum of all elements in batch
            }
            
            batch_sum = transposeReduceSum(batch_sum);  // sum of 32 batches
            if (threadIdx.y * WARP_SZ == inner_b) inner_sum = batch_sum;
        }
        inner_sum = transposeReduceSum(inner_sum);  // sum of 1024 batches    
        local_sum += inner_sum; // sum of all batches
    }


    if (threadIdx.x == 0 && threadIdx.y == 0) {
        grad[blockIdx.x] = __float2half(local_sum);
    }
}


// block <n>    thread<32, 32>
CPM_KERNEL_EXPORT void cu_arith_ln_add_backward(
    int64_t batch, int64_t n, int64_t m,
    const half *grad_out,       // (batch, n, m)
    half *grad                  // (n)
) {

    float local_sum = 0;
    for (int b = 0; b < batch; b += WARP_SZ * WARP_SZ) {
        float inner_sum = 0;
        for (int inner_b = 0; inner_b < WARP_SZ * WARP_SZ && inner_b + b < batch; inner_b += WARP_SZ) {
            int batch_idx = b + inner_b + threadIdx.y;
            int base_idx = batch_idx * n * m + blockIdx.x * m + threadIdx.x;

            float batch_sum = 0;
            for (int i = 0; i < m; i += WARP_SZ * WARP_SZ) {
                float inner_v = 0;
                for (int j = 0; j < WARP_SZ * WARP_SZ && i + j < m; j += WARP_SZ) {
                    float v = 0;
                    if (batch_idx < batch && i + j + threadIdx.x < m) {
                        v = (float)grad_out[base_idx + i + j];
                    }
                    v = warpReduceSum(v);           // sum of 32 elements
                    v = __shfl_sync(0xFFFFFFFF, v, 0); // broadcast to all threads in warp
                    if (threadIdx.x * WARP_SZ == j) inner_v = v;
                }
                inner_v = warpReduceSum(inner_v);   // sum of 1024 elements

                // stores the sum of batch (b + inner_b + threadIdx.y) in (0, threadIdx.y)
                batch_sum += inner_v;   // sum of all elements in batch
            }
            
            batch_sum = transposeReduceSum(batch_sum);  // sum of 32 batches
            if (threadIdx.y * WARP_SZ == inner_b) inner_sum = batch_sum;
        }
        inner_sum = transposeReduceSum(inner_sum);  // sum of 1024 batches    
        local_sum += inner_sum; // sum of all batches
    }


    if (threadIdx.x == 0 && threadIdx.y == 0) {
        grad[blockIdx.x] = __float2half(local_sum);
    }
}



// block <batch, n // 1024>    thread<min(n, 1024)>,  half n
CPM_KERNEL_EXPORT void cu_arith_batch_mul_add(
    int64_t batch, int64_t n,
    const half2 *x,         // (batch, n)
    const half2 *alpha,     // (n) 
    const half2 *beta,      // (n)
    half2 *out              // (batch, n)
) {
    int64_t col = threadIdx.x + blockIdx.y * blockDim.x;
    if (col < n) {
        out[blockIdx.x * n + col] = __hfma2(x[blockIdx.x * n + col], __ldg(alpha + col), __ldg(beta + col));
    }
}

// block <batch>    thread<min(n, 1024)>,  half n
CPM_KERNEL_EXPORT void cu_arith_batch_mul(
    int64_t batch, int64_t n,
    const half2 *x,         // (batch, n)
    const half2 *alpha,     // (n) 
    half2 *out              // (batch, n)
) {
    int64_t col = threadIdx.x + blockIdx.y * blockDim.x;
    if (col < n) {
        out[blockIdx.x * n + col] = __hmul2(x[blockIdx.x * n + col], __ldg(alpha + col));
    }
}