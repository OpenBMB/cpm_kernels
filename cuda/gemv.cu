#include "reduce.cuh"
#include <cuda_fp16.h>
#include "common.h"
#include <sm_61_intrinsics.h>


__inline__ __device__ int32_t warpReduceSumInt32(int32_t x) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) 
        x += __shfl_down_sync(0xFFFFFFFF, x, offset);
    return x;
}

__inline__ __device__ int32_t blockReduceSumInt32(int32_t x) {
    static __shared__ int32_t shared[WARP_SZ]; // blockDim.x / warpSize
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    x = warpReduceSumInt32(x);
    if (lane == 0) shared[wid] = x;
    __syncthreads();
    x = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) x = warpReduceSumInt32(x);
    return x;
}

// block <batch>,   thread <min(1024, round_up(n, 32))>
CPM_KERNEL_EXPORT void cu_gemv_calc_scale(
    int32_t batch, int32_t n,
    const half *vec,                   // <batch, n>
    half *out                           // <batch>
) {
    int32_t base_vec = blockIdx.x * n;
    float local_max = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        local_max = fmaxf( local_max, fabsf(vec[base_vec + i]) );
    }
    local_max = blockReduceMax(local_max);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = __float2half(local_max / 127.0);
    }
}

// block <batch, n // 1024>     thread <min(1024, n)>
CPM_KERNEL_EXPORT void cu_gemv_round(
    int32_t batch, int32_t n,
    const half *vec,            // (batch, n)
    const half* scale,          // (batch,)
    int8_t* out                 // (batch, n)
) {
    int32_t col_idx = blockIdx.y * blockDim.x + threadIdx.x;
    half v_scale = __ldg(scale + blockIdx.x);
    if (col_idx < n) {
        out[blockIdx.x * n + col_idx] = (int8_t) nearbyintf((float)vec[blockIdx.x * n + col_idx] / (float)v_scale);
    }
}

// block <batch, dim_out>       thread <min(1024, dim_in // 4)>         dim_in % 4 == 0
CPM_KERNEL_EXPORT void cu_gemv_broadcast_mat_int8(
    int32_t batch, int32_t dim_out, int32_t dim_in,
    const half *scale_mat,              // <dim_out>
    const char4 *mat,                   // <dim_out, dim_in>
    const half *scale_vec,              // <batch,>
    const char4 *vec,                   // <batch, dim_in>
    half *out                           // <batch, dim_out>
) {
    int32_t quarter_dim_in = dim_in >> 2;
    const char4* base_mat = mat + blockIdx.y * quarter_dim_in;
    const char4* base_vec = vec + blockIdx.x * quarter_dim_in;
    int32_t local_sum = 0;
    for (int32_t i = threadIdx.x; i < quarter_dim_in; i += blockDim.x) {
        local_sum = __dp4a(base_mat[i], base_vec[i], local_sum);
    }
    local_sum = blockReduceSumInt32(local_sum);
    if (threadIdx.x == 0) {
        out[blockIdx.x * dim_out + blockIdx.y] = __float2half((float)local_sum * (float)__ldg(scale_vec + blockIdx.x) * (float)__ldg(scale_mat + blockIdx.y));
    }
}

// block <batch, dim_out>,   thread <min(1024, round_up(dim_in // 2, 32))>
CPM_KERNEL_EXPORT void cu_gemv_fp16(
    int32_t batch, int32_t dim_out, int32_t dim_in,
    const half2 *mat,                   //  <batch, dim_out, dim_in>
    const half2 *vec,                   //  <batch, dim_in>
    half *out                           //  <batch, dim_out>
) {
    int32_t half_dim_in = dim_in >> 1;
    int32_t base_v = blockIdx.x * half_dim_in;
    int32_t base_mat = (blockIdx.x * dim_out + blockIdx.y) * half_dim_in;

#if __CUDA_ARCH__ >= 620 || !defined(__CUDA_ARCH__)
    half2 sum = __float2half2_rn(0);
    for (int i = threadIdx.x; i  < half_dim_in; i += blockDim.x) {
        sum = __hfma2(vec[base_v + i], mat[base_mat + i], sum);
    }
    float v = (float)sum.x + (float)sum.y;
#else
    // fallback to fp32
    float v = 0;
    for (int i = threadIdx.x; i < half_dim_in; i += blockDim.x) {
        v += (float)vec[base_v + i].x * (float)mat[base_mat + i].x + (float)vec[base_v + i].y * (float)mat[base_mat + i].y;
    }
#endif
    v = blockReduceSum(v);
    if (threadIdx.x == 0) {
        out[blockIdx.x * dim_out + blockIdx.y] = __float2half(v);
    }
}

// block <batch, dim_out // WARP_SZ>,   thread <WARP_SZ, WARP_SZ>
CPM_KERNEL_EXPORT void cu_gemv_fp16_transpose(
    int32_t batch, int32_t dim_out, int32_t dim_in,
    const half *mat,                   //  <batch, dim_in, dim_out>
    const half *vec,                   //  <batch, dim_in>
    half *out                          //  <batch, dim_out>
) {
    int32_t col = blockIdx.y * WARP_SZ + threadIdx.x;
    int32_t base_idx = blockIdx.x * dim_in + threadIdx.y;
    float sum = 0;
    for (int i = 0; i < dim_in; i += WARP_SZ * WARP_SZ) { // warp * warp blocks
        float local_sum = 0;
        for (int j = 0; j < WARP_SZ * WARP_SZ && i + j < dim_in; j += WARP_SZ) {    // warp block
            float v = 0;
            if (i + j + threadIdx.y < dim_in && col < dim_out) v = (float)vec[base_idx + i + j] * (float)mat[(base_idx + i + j) * dim_out + col];
            v = transposeReduceSum(v);
            if (threadIdx.y * WARP_SZ == j) {
                local_sum = v;
            }
        }
        local_sum = transposeReduceSum(local_sum);
        sum += local_sum;
    }
    
    if (threadIdx.y == 0 && col < dim_out) {
        out[blockIdx.x * dim_out + col] = sum;
    }
}

// block <batch, dim_out>,      thread <min(1024, round_up(dim_in // 2, 32))>
CPM_KERNEL_EXPORT void cu_gemv_broadcast_mat_fp16(
    int32_t batch, int32_t dim_out, int32_t dim_in,
    const half2 *mat,                  //  <dim_out, dim_in>
    const half2 *vec,                  //  <batch, dim_in>
    half *out                          //  <batch, dim_out>
) {
    int32_t half_dim_in = dim_in >> 1;
    int32_t base_vec_idx = blockIdx.x * half_dim_in;
    int32_t base_mat = blockIdx.y * half_dim_in;

#if __CUDA_ARCH__ >= 620 || !defined(__CUDA_ARCH__)
    half2 sum = __float2half2_rn(0);
    for (int i = threadIdx.x; i < half_dim_in; i += blockDim.x) {
        sum = __hfma2(vec[base_vec_idx + i], mat[base_mat + i], sum);
    }
    float v = (float)sum.x + (float)sum.y;
#else
    float v = 0;
    for (int i = threadIdx.x; i < half_dim_in; i += blockDim.x) {
        v += (float)vec[base_vec_idx + i].x * (float)mat[base_mat + i].x + (float)vec[base_vec_idx + i].y * (float)mat[base_mat + i].y;
    }
#endif
    v = blockReduceSum(v);
    if (threadIdx.x == 0) {
        out[blockIdx.x * dim_out + blockIdx.y] = __float2half(v);
    }
}