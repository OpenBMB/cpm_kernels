#include "reduce.cuh"
#include <cuda_fp16.h>
#include "common.h"

// block <batch, n> thread <min(m, 1024)>
CPM_KERNEL_EXPORT void cu_gemm_round(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,       // b, n, m
    const half *scale,     // b, n
    int8_t *out
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;   // mat[batch][n][m], scale[batch][n]
    half local_scale = scale[blockIdx.x * n + blockIdx.y];

    for (int32_t i = threadIdx.x; i < m; i += blockDim.x) {
        out[base_idx + i] = (int8_t)nearbyintf((float)(__ldg(mat + base_idx + i) / local_scale)); 
    }
}


// block <batch, n> thread <min(m, 1024)>
CPM_KERNEL_EXPORT void cu_gemm_round_transpose(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,       // b, n, m
    const half *scale,     // b, m
    int8_t *out
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;   // mat[batch][n][m], scale[batch][m]

    for (int32_t i = threadIdx.x; i < m; i += blockDim.x) {
        out[base_idx + i] = (int8_t)nearbyintf((float)(mat[base_idx + i] / __ldg(scale + blockIdx.x * m + i)));
    }
}


// grid <batch, n>  thread <min(m, 1024)>
CPM_KERNEL_EXPORT void cu_gemm_scale(
    int32_t batch, int32_t n, int32_t m,
    const int32_t *mat,        // b, n, m
    const half *scale_x,   // b, n
    const half *scale_y,   // b, m
    half *out,
    bool broad_cast_x, bool broad_cast_y
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;
    float scale_x_value = 0;
    if (broad_cast_x) {
        scale_x_value = (float)__ldg(scale_x + blockIdx.y);
    } else {
        scale_x_value = (float)__ldg(scale_x + blockIdx.x * n + blockIdx.y);
    }

    for (int32_t i = threadIdx.x; i < m; i += blockDim.x){
        if (broad_cast_y) {
            out[base_idx + i] = __float2half((float)(mat[base_idx + i]) * scale_x_value * (float)__ldg(scale_y + i));
        }
        else {
            out[base_idx + i] = __float2half((float)(mat[base_idx + i]) * scale_x_value * (float)__ldg(scale_y + blockIdx.x * m + i));
        }
    }
}

// grid <batch, n>  thread <min(round_up(m, 32), 1024)>
CPM_KERNEL_EXPORT void cu_gemm_calc_scale(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,        // b, n, m
    half *out  // b, n
) {
    float local_max = 0;

    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m;
    for (int32_t i = 0; i < m; i += blockDim.x){
        int32_t offset = threadIdx.x + i;
        float v = 0;
        if (offset < m) {
            v = fabsf((float)(mat[base_idx + offset]));
        }
        local_max = fmaxf(v, local_max);
    }
    local_max = blockReduceMax(local_max);

    if (threadIdx.x == 0) {
        out[ blockIdx.x * n + blockIdx.y ] = __float2half(local_max / 127.0);
    }
}

// grid <batch, round_up(m, WARP_SZ) / WARP_SZ>  thread <WARP_SZ, WARP_SZ>
CPM_KERNEL_EXPORT void cu_gemm_calc_scale_transpose(
    int32_t batch, int32_t n, int32_t m,
    const half *in,        // b, n, m
    half *out  // b, m
) {
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t base_idx = (blockIdx.x * n + threadIdx.y) * m + col_idx;
    
    float local_max = 0.0;
    for (int32_t i = 0; i < n; i += WARP_SZ) {
        // put & transpose
        if (i + threadIdx.y < n && col_idx < m) {
            local_max = fmaxf(fabsf((float)(in[base_idx + i * m])), local_max);
        }
    }
    local_max = transposeReduceMax(local_max);
    if (threadIdx.y == 0 && col_idx < m) {
        out[blockIdx.x * m + col_idx] = __float2half(local_max / 127.0);
    }
}

// Backward

// grid <batch, n>,   thread <min(round_up(m, 32), 1024)>
CPM_KERNEL_EXPORT void cu_gemm_backward_round_scale(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,        // (batch, n, m)
    const half *scale_y,    // (batch, m)   or  (1, m) if broadcast_y
    int8_t *out,            // (batch, n, m)
    half *scale_x,          // (batch, n)
    bool broad_cast_y
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m + threadIdx.x;
    int32_t base_m_idx = blockIdx.x * m + threadIdx.x;
    if (broad_cast_y) base_m_idx = threadIdx.x;

    float local_max = 0;
    __shared__ float global_max;
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            local_max = fmaxf(fabsf((float)(mat[base_idx + i]) * (float)__ldg(scale_y + base_m_idx + i)), local_max);
        }
    }
    local_max = blockReduceMax(local_max) / 127.0;
    if (threadIdx.x == 0) {
        global_max = local_max;
        scale_x[blockIdx.x * n + blockIdx.y] = __float2half(local_max);
    }
    __syncthreads();
    local_max = global_max;
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            out[base_idx + i] = (int8_t)nearbyintf((float)mat[base_idx + i] * (float)__ldg(scale_y + base_m_idx + i) / local_max);
        }
    }
}

// grid <batch, m / WARP_SZ>,   thread <WARP_SZ, WARP_SZ>
CPM_KERNEL_EXPORT void cu_gemm_backward_scale_round(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,        // (batch, n, m)
    const half *scale_x,    // (batch, n)   or  (1, n)    if broad_cast_x
    int8_t *out,            // (batch, n, m)
    half *scale_y,          // (batch, m)
    bool broad_cast_x
) {
    int32_t col = blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t base_idx = (blockIdx.x * n + threadIdx.y) * m + col;
    int32_t base_n_idx = blockIdx.x * n + threadIdx.y;
    if (broad_cast_x) base_n_idx = threadIdx.y;
    
    float local_max = 0;

    if (col < m) {
        for (int i = 0; i < n; i += blockDim.y) {
            if (i + threadIdx.y < n) {
                local_max = fmaxf(fabsf( (float)mat[base_idx + i * m] * (float)__ldg(scale_x + base_n_idx + i) ), local_max);
            }
        }
    }
    local_max = transposeReduceMax(local_max);  // reduce max along y
    local_max = local_max / 127.0;
    if (threadIdx.y == 0 && col < m) {
        scale_y[blockIdx.x * m + col] = __float2half(local_max);
    }
    
    if (col < m) {
        for (int i = 0; i < n; i += blockDim.y) {
            if (i + threadIdx.y < n) {
                out[base_idx + i * m] = (int8_t)nearbyintf((float)mat[base_idx + i * m] * (float)__ldg(scale_x + base_n_idx + i) / local_max);
            }
        }
    }
}


// block <batch, n>,    thread <min(1024, m)>
CPM_KERNEL_EXPORT void cu_gemm_scale_x (
    int32_t batch, int32_t n, int32_t m,
    const int32_t *mat,     // (batch, n, m)
    const half *scale_x,    // (batch, n)
    half *out               // (batch, n, m)
) {
    float scale = scale_x[blockIdx.x * n + blockIdx.y];
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m + threadIdx.x;
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            out[base_idx + i] = __float2half(scale * (float)mat[base_idx + i]);
        }
    }
}

// block <batch, n>,    thread <min(1024, m)>
CPM_KERNEL_EXPORT void cu_gemm_scale_y (
    int32_t batch, int32_t n, int32_t m,
    const int32_t *mat,     // (batch, n, m)
    const half *scale_y,    // (batch, m)
    half *out               // (batch, n, m)
) {
    int32_t base_idx = (blockIdx.x * n + blockIdx.y) * m + threadIdx.x;
    int32_t base_m_idx = blockIdx.x * m + threadIdx.x;
    for (int i = 0; i < m; i += blockDim.x) {
        if (i + threadIdx.x < m) {
            out[base_idx + i] = __float2half((float)mat[base_idx + i] * (float)__ldg(scale_y + base_m_idx + i));
        }
    }
}