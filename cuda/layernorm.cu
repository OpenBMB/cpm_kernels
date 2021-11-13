#include "reduce.cuh"
#include <cuda_fp16.h>
#include "common.h"

// block <batch_idx, m/32>   thread <32, 32>
CPM_KERNEL_EXPORT void cu_layernorm_forward(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,    // (batch, n, m)
    half *out,          // (batch, n, m)
    float eps,
    bool rd_mean
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;

    float local_total_v = 0.0;
    float local_total_v2 = 0.0;

    // nested reduce loops to achieve higher precision
    for (int i = 0; i < n; i += WARP_SZ * WARP_SZ) {
        
        // inner loop reduce
        float inner_v = 0.0;
        float inner_v2 = 0.0;
        for (int j = 0; j < WARP_SZ * WARP_SZ && i + j < n; j += WARP_SZ) {
            // reduce per block
            float v = 0, tv = 0, tv2 = 0;
            if (col_idx < m && i + j + threadIdx.y < n) {
                v = (float)mat[base_mat_idx + (i + j) * m];
            }
            if (rd_mean) {
                tv = transposeReduceSum(v);
            }
            tv2 = transposeReduceSum(v * v);
            if (threadIdx.y * WARP_SZ == j) {
                inner_v = tv;
                inner_v2 = tv2;
            }
        }

        // outter loop reduce, reduce per 32 blocks
        if (rd_mean) {
            local_total_v += transposeReduceSum(inner_v);
        }
        local_total_v2 += transposeReduceSum(inner_v2);
    }

    local_total_v2 = local_total_v2 / (float)n;
    if (rd_mean) {
        local_total_v = local_total_v / (float)n;
        local_total_v2 -= local_total_v * local_total_v;
    }

    local_total_v2 = rsqrtf(local_total_v2 + eps);

    float local_mean =  local_total_v;
    float local_var = local_total_v2;
    if (rd_mean) {
        for (int i = 0; i < n; i += WARP_SZ) {
            if (col_idx < m && i + threadIdx.y < n) {
                out[base_mat_idx + i * m] = __float2half(((float)mat[base_mat_idx + i * m] - local_mean) * local_var);
            }
        }
    } else {
        for (int i = 0; i < n; i += blockDim.y) {
            if (col_idx < m && i + threadIdx.y < n) {
                out[base_mat_idx + i * m] = __float2half((float)mat[base_mat_idx + i * m] * local_var);
            }
        }
    }
}

// block <batch_idx, m/32>   thread <32, 32>
// WARNING: this function is the same as cu_layernorm_forward now.
CPM_KERNEL_EXPORT void cu_layernorm_inplace_forward(
    int32_t batch, int32_t n, int32_t m,
    half *mat,    // (batch, n, m)
    float eps,
    bool rd_mean
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;

    float local_total_v = 0.0;
    float local_total_v2 = 0.0;

    // nested reduce loops to achieve higher precision
    for (int i = 0; i < n; i += WARP_SZ * WARP_SZ) {
        
        // inner loop reduce
        float inner_v = 0.0;
        float inner_v2 = 0.0;
        for (int j = 0; j < WARP_SZ * WARP_SZ && i + j < n; j += WARP_SZ) {
            // reduce per block
            float v = 0, tv = 0, tv2 = 0;
            if (col_idx < m && i + j + threadIdx.y < n) {
                v = (float)mat[base_mat_idx + (i + j) * m];
            }
            if (rd_mean) {
                tv = transposeReduceSum(v);
            }
            tv2 = transposeReduceSum(v * v);
            if (threadIdx.y * WARP_SZ == j) {
                inner_v = tv;
                inner_v2 = tv2;
            }
        }

        // outter loop reduce, reduce per 32 blocks
        if (rd_mean) {
            local_total_v += transposeReduceSum(inner_v);
        }
        local_total_v2 += transposeReduceSum(inner_v2);
    }

    local_total_v2 = local_total_v2 / (float)n;
    if (rd_mean) {
        local_total_v = local_total_v / (float)n;
        local_total_v2 -= local_total_v * local_total_v;
    }

    local_total_v2 = rsqrtf(local_total_v2 + eps);

    float local_mean =  local_total_v;
    float local_var = local_total_v2;
    if (rd_mean) {
        for (int i = 0; i < n; i += WARP_SZ) {
            if (col_idx < m && i + threadIdx.y < n) {
                mat[base_mat_idx + i * m] = __float2half(((float)(mat[base_mat_idx + i * m]) - local_mean) * local_var);
            }
        }
    } else {
        for (int i = 0; i < n; i += blockDim.y) {
            if (col_idx < m && i + threadIdx.y < n) {
                mat[base_mat_idx + i * m] = __float2half((float)(mat[base_mat_idx + i * m]) * local_var);
            }
        }
    }
}

// block <batch_idx, offset_m/32>   thread <32, 32>
CPM_KERNEL_EXPORT void cu_layernorm_forward_v(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,    // (batch, n, m)
    half *out,          // (batch, n, m)
    half *out_var,      // (batch, m)
    float eps
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;

    float local_total_v2 = 0.0;

    // nested reduce loops to achieve higher precision
    for (int i = 0; i < n; i += WARP_SZ * WARP_SZ) {
        
        // inner loop reduce
        float inner_v2 = 0.0;
        for (int j = 0; j < WARP_SZ * WARP_SZ && i + j < n; j += WARP_SZ) {
            // reduce per block
            float v = 0, tv2 = 0;
            if (col_idx < m && i + j + threadIdx.y < n) {
                v = (float)mat[base_mat_idx + (i + j) * m];
            }
            tv2 = transposeReduceSum(v * v);
            if (threadIdx.y * WARP_SZ == j) {
                inner_v2 = tv2;
            }
        }

        // outter loop reduce, reduce per 32 blocks
        local_total_v2 += transposeReduceSum(inner_v2);
    }

    float local_var = rsqrtf(local_total_v2 / (float)n + eps);

    if (threadIdx.y == 0 && col_idx < m) out_var[blockIdx.x * m + col_idx] = __float2half(local_var);

    for (int i = 0; i < n; i += blockDim.y) {
        if (col_idx < m && i + threadIdx.y < n) {
            out[base_mat_idx + i * m] = __float2half((float)mat[base_mat_idx + i * m] * local_var);
        }
    }
}

// block <batch_idx, offset_m/32>   thread <32, 32>
CPM_KERNEL_EXPORT void cu_layernorm_forward_mv(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,    // (batch, n, m)
    half *out,
    half *out_mean,
    half *out_var,
    float eps
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;

    float local_total_v = 0.0;
    float local_total_v2 = 0.0;

    // nested reduce loops to achieve higher precision
    for (int i = 0; i < n; i += WARP_SZ * WARP_SZ) {
        
        // inner loop reduce
        float inner_v = 0.0;
        float inner_v2 = 0.0;
        for (int j = 0; j < WARP_SZ * WARP_SZ && i + j < n; j += WARP_SZ) {
            // reduce per block
            float v = 0, tv = 0, tv2 = 0;
            if (col_idx < m && i + j + threadIdx.y < n) {
                v = (float)mat[base_mat_idx + (i + j) * m];
            }
            tv = transposeReduceSum(v);
            tv2 = transposeReduceSum(v * v);
            if (threadIdx.y * WARP_SZ == j) {
                inner_v = tv;
                inner_v2 = tv2;
            }
        }

        // outter loop reduce, reduce per 32 blocks
        local_total_v += transposeReduceSum(inner_v);
        local_total_v2 += transposeReduceSum(inner_v2);
    }

    local_total_v = local_total_v / (float)n;
    local_total_v2 = rsqrtf(local_total_v2 / (float)n - local_total_v * local_total_v + eps);

    if (threadIdx.y == 0 && col_idx < m) {
        out_var[blockIdx.x * m + col_idx] = __float2half(local_total_v2);
        out_mean[blockIdx.x * m + col_idx] = __float2half(local_total_v);
    }

    float local_mean =  local_total_v;
    float local_var = local_total_v2;
    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            out[base_mat_idx + i * m] = __float2half(((float)(mat[base_mat_idx + i * m]) - local_mean) * local_var);
        }
    }
}

// block <batch_idx, offset_m/32>   thread <32, 32>
CPM_KERNEL_EXPORT void cu_layernorm_backward_v(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,        // (batch, n, m)
    const half *grad_in,    // (batch, n, m)
    const half *var,        // (batch, m) 
    half *grad_out
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;

    float local_grad_var = 0;

    float local_var = col_idx < m ? (float)__ldg(var + blockIdx.x * m + col_idx) : 0.0;
    float n_half_rsqrt_v3 = -0.5 * local_var * local_var * local_var;

    for (int i = 0; i < n; i += WARP_SZ * WARP_SZ) {
        float inner_v = 0;
        for (int j = 0; j < WARP_SZ * WARP_SZ && i + j < n; j += WARP_SZ) {
            float v = 0;
            if (col_idx < m && i + j + threadIdx.y < n) {
                v = (float)grad_in[base_mat_idx + (i + j) * m] * 
                                    n_half_rsqrt_v3 * 
                                    ((float)mat[base_mat_idx + (i + j) * m]);
            }
            v = transposeReduceSum(v);
            if (threadIdx.y * WARP_SZ == j) inner_v = v;
        }
        local_grad_var += transposeReduceSum(inner_v);
    }

    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            grad_out[base_mat_idx + i * m] = __float2half(
                (float)grad_in[base_mat_idx + i * m] * local_var +
                ((local_grad_var * (float)mat[base_mat_idx + i * m] * 2) / (float)n)
            );
        }
    }
}

// block <batch_idx, offset_m/32>   thread <32, 32>
CPM_KERNEL_EXPORT void cu_layernorm_backward_mv(
    int32_t batch, int32_t n, int32_t m,
    const half *mat,        // (batch, n, m)
    const half *grad_in,    // (batch, n, m)
    const half *mean,       // (batch, m)
    const half *var,        // (batch, m) 
    half *grad_out
) {
    int32_t base_mat_idx = (blockIdx.x * n + threadIdx.y) * m + blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t col_idx = blockIdx.y * WARP_SZ + threadIdx.x;


    float local_grad_var = 0;
    float local_grad_mean = 0;

    float local_mean =  col_idx < m ? (float)__ldg(mean + blockIdx.x * m + col_idx) : 0.0;
    float local_var = col_idx < m ? (float)__ldg(var + blockIdx.x * m + col_idx) : 0.0;

    float n_half_rsqrt_v3 = -0.5 * local_var * local_var * local_var;


    for (int i = 0; i < n; i += WARP_SZ * WARP_SZ) {
        float inner_m = 0;
        float inner_v = 0;

        for (int j = 0; j < WARP_SZ * WARP_SZ && i + j < n; j += WARP_SZ) {
            float vv = 0, vm = 0;
            if (col_idx < m && i + j + threadIdx.y < n) {
                float gi = (float)grad_in[base_mat_idx + (i + j) * m];
                vv = gi * n_half_rsqrt_v3 * ((float)mat[base_mat_idx + (i + j) * m] - local_mean);
                vm = -gi * local_var;
            }
            vv = transposeReduceSum(vv);
            vm = transposeReduceSum(vm);
            if (threadIdx.y * WARP_SZ == j) {
                inner_v = vv;
                inner_m = vm;
            }
        }

        local_grad_var += transposeReduceSum(inner_v);
        local_grad_mean += transposeReduceSum(inner_m);
    }

    local_grad_mean -= 2 * local_grad_var * local_mean;

    for (int i = 0; i < n; i += WARP_SZ) {
        if (col_idx < m && i + threadIdx.y < n) {
            grad_out[base_mat_idx + i * m] = __float2half(
                (float)grad_in[base_mat_idx + i * m] * local_var +
                ((local_grad_mean + local_grad_var * (float)mat[base_mat_idx + i * m] * 2) / (float)n)
            );
        }
    }
}


// block <batch>    thread <min(round_up(n, 32), 1024)>
CPM_KERNEL_EXPORT void cu_layernorm_step(
    int32_t batch, int32_t n,
    const half *mat,    // (batch, n)
    half *out,          // (batch, n)
    float eps,
    bool rd_mean
) {
    int32_t base_mat_idx = blockIdx.x * n;

    float local_total_v = 0;
    float local_total_v2 = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v =  mat[base_mat_idx + i];
        if (rd_mean) local_total_v += v;
        local_total_v2 += v * v;
    }

    __shared__ float global_mean;
    __shared__ float global_var;

    if (rd_mean) local_total_v = blockReduceSum(local_total_v);
    local_total_v2 = blockReduceSum(local_total_v2);
    if (threadIdx.x == 0) {
        if (rd_mean) {
            global_mean = local_total_v / (float)n;
            global_var = local_total_v2 / (float)n - global_mean * global_mean;
        } else {
            global_var = local_total_v2 / (float)n;
            global_mean = 0;
        }
    }
    __syncthreads();
    local_total_v2 = rsqrtf(global_var + eps);
    local_total_v = global_mean;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out[base_mat_idx + i] = __float2half((__half2float(mat[base_mat_idx + i]) - local_total_v) * local_total_v2);
    }
}

// block <batch>    thread <min(round_up(n, 32), 1024)>
CPM_KERNEL_EXPORT void cu_layernorm_step_inplace(
    int32_t batch, int32_t n,
    half *mat,    // (batch, n)
    float eps,
    bool rd_mean
) {
    int32_t base_mat_idx = blockIdx.x * n;

    float local_total_v = 0;
    float local_total_v2 = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = mat[base_mat_idx + i];
        if (rd_mean) local_total_v += v;
        local_total_v2 += v * v;
    }

    __shared__ float global_mean;
    __shared__ float global_var;

    if (rd_mean) local_total_v = blockReduceSum(local_total_v);
    local_total_v2 = blockReduceSum(local_total_v2);
    if (threadIdx.x == 0) {
        if (rd_mean) {
            global_mean = local_total_v / (float)n;
            global_var = local_total_v2 / (float)n - global_mean * global_mean;
        } else {
            global_var = local_total_v2 / (float)n;
            global_mean = 0;
        }
    }
    __syncthreads();
    local_total_v2 = rsqrtf(global_var + eps);
    local_total_v = global_mean;
    
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        mat[base_mat_idx + i] = __float2half((__half2float(mat[base_mat_idx + i]) - local_total_v) * local_total_v2);
    }
}