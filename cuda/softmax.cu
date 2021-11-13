#include <cuda_fp16.h>
#include "common.h"
#include "reduce.cuh"

// grid <batch, m / 32>,    thread <32, 32>
CPM_KERNEL_EXPORT  void cu_softmax_forward(
    int32_t batch, int32_t n, int32_t m,
    const half *in,    // batch, n, m
    half *out          // batch, n, m
) {
    float local_max = -INFINITY;

    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;
    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            local_max = fmaxf((float)in[base_mat_idx + i * m], local_max);
        }
    }

    local_max = fmaxf(transposeReduceMax(local_max), -1e6);
    
    float local_sum = 0;
    for (int i = 0; i < n; i += WARP_SZ * WARP_SZ) {
        float inner_sum = 0;
        for (int j = 0; j < WARP_SZ * WARP_SZ && i + j < n; j += WARP_SZ) {
            float v = 0;
            if (col_idx < m && i + j + threadIdx.y < n) {
                v = expf((float)in[base_mat_idx + (i + j) * m] - local_max);
            }
            v = transposeReduceSum(v);
            if (threadIdx.y * WARP_SZ == j) inner_sum = v;
        }
        local_sum += transposeReduceSum(inner_sum);
    }
    local_sum += 1e-10; // avoid nan
    
    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            out[base_mat_idx + i * m] = __float2half( expf((float)in[base_mat_idx + i * m] - local_max) / local_sum );
        }
    }
}

// grid <batch, m / 32>,    thread <32, 32>
CPM_KERNEL_EXPORT  void cu_softmax_inplace_forward(
    int32_t batch, int32_t n, int32_t m,
    half *x             // batch, n, m
) {
    float local_max = -INFINITY;

    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;
    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            local_max = fmaxf((float)x[base_mat_idx + i * m], local_max);
        }
    }

    local_max = fmaxf(transposeReduceMax(local_max), -1e6);
    
    float local_sum = 0;
    for (int i = 0; i < n; i += WARP_SZ * WARP_SZ) {
        float inner_sum = 0;
        for (int j = 0; j < WARP_SZ * WARP_SZ && i + j < n; j += WARP_SZ) {
            float v = 0;
            if (col_idx < m && i + j + threadIdx.y < n) {
                v = expf((float)x[base_mat_idx + (i + j) * m] - local_max);
            }
            v = transposeReduceSum(v);
            if (threadIdx.y * WARP_SZ == j) inner_sum = v;
        }
        local_sum += transposeReduceSum(inner_sum);
    }
    local_sum += 1e-10; // avoid nan
    
    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            x[base_mat_idx + i * m] = __float2half( expf((float)x[base_mat_idx + i * m] - local_max) / local_sum );
        }
    }
}


// grid <batch, m / 32>,    thread <32, 32>
CPM_KERNEL_EXPORT  void cu_softmax_backward(
    int32_t batch, int32_t n, int32_t m,
    const half *out,       // batch, n, m 
    const half *grad_in,   // batch, n, m
    half *grad_out         // batch, n, m
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;

    float local_sum = 0;
    for (int i = 0; i < n; i += WARP_SZ * WARP_SZ) {
        float inner_sum = 0;
        for (int j = 0; j < WARP_SZ * WARP_SZ && i + j < n; j += WARP_SZ) {
            float v = 0;
            if (col_idx < m && i + j + threadIdx.y < n) {
                v = (float)out[base_mat_idx + (i + j) * m] * (float)grad_in[base_mat_idx + (i + j) * m];
            }
            v = transposeReduceSum(v);
            if (threadIdx.y * WARP_SZ == j) inner_sum = v;
        }
        local_sum += transposeReduceSum(inner_sum);
    }
    
    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            grad_out[base_mat_idx + i * m] = __float2half((float)__ldg(out + base_mat_idx + i * m) * ((float)__ldg(grad_in + base_mat_idx + i * m) - local_sum ) );
        }
    }
}

// grid <batch>,    thread <min(1024, round_up(n, 32))>
CPM_KERNEL_EXPORT void cu_softmax_step_inplace(
    int32_t batch, int32_t n,
    half *x         // batch, n
) {
    int32_t base_x_idx = blockIdx.x * n + threadIdx.x;

    float local_max = -INFINITY;
    __shared__ float global_max;

    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            local_max = fmaxf(local_max, x[base_x_idx + i]);
        }
    }

    local_max = blockReduceMax(local_max);
    if (threadIdx.x == 0) {
        global_max = fmaxf(local_max, -1e6);
    }
    __syncthreads();

    local_max = global_max;
    float local_sum = 0;
    __shared__ float global_sum;

    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            local_sum += expf((float)x[base_x_idx + i] - local_max);
        }
    }
    local_sum = blockReduceSum(local_sum);
    if (threadIdx.x == 0) {
        global_sum = local_sum + 1e-10;      // avoid nan
    }
    __syncthreads();
    local_sum = global_sum;

    for (int i = 0; i < n; i += blockDim.x) {
        if (i + threadIdx.x < n) {
            x[base_x_idx + i] = __float2half(expf((float)x[base_x_idx + i] - local_max) / local_sum);
        }
    }
}