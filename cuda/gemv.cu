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

// block <batch>,   thread <min(1024, round_up(dim_in // 4, 32))>
CPM_KERNEL_EXPORT void cu_gemv_broadcast_mat_int8(
    int32_t batch, int32_t dim_out, int32_t dim_in,
    const half *scale_mat,              //  <dim_out>
    const char4 *mat,                   //  <dim_out, dim_in>
    const half2 *vec,                   //  <batch, dim_in>
    half *out                           //  <batch, dim_out>
) {
    int32_t base_v = blockIdx.x * (dim_in / 2) + threadIdx.x * 2;
    
    __shared__ float global_max;
    float local_max = 0;
    
    for (int i = 0; (i + threadIdx.x) * 4 < dim_in; i += blockDim.x) {
        const half2 *ptr = vec + (base_v + i * 2);
        half2 a = __habs2(__ldg(ptr));
        half2 b = __habs2(__ldg(ptr + 1));
        local_max = fmaxf(local_max, fmaxf(a.x, a.y));
        local_max = fmaxf(local_max, fmaxf(b.x, b.y));
    }
    local_max = blockReduceMax(local_max);
    if (threadIdx.x == 0) {
        global_max = local_max / 127.0;
    }
    __syncthreads();

    local_max = global_max;

    half2 scale_vec = __float2half2_rn(local_max);
    for (int j = 0; j < dim_out; j ++) {
        const char4 *base_mat = mat + j * (dim_in / 4)  + threadIdx.x;
        int sum = 0;
        for (int i = 0; (i + threadIdx.x) * 4 < dim_in; i += blockDim.x) {
            const half2 *ptr = vec + (base_v + i * 2);
            half2 a = __ldg(ptr);
            half2 b = __ldg(ptr + 1);
            
            a = __h2div(a, scale_vec);
            b = __h2div(b, scale_vec);

            char4 vec_x;
            vec_x.x = (int8_t)nearbyintf(a.x);
            vec_x.y = (int8_t)nearbyintf(a.y);
            vec_x.z = (int8_t)nearbyintf(b.x);
            vec_x.w = (int8_t)nearbyintf(b.y);

            sum = __dp4a(vec_x, __ldg(base_mat + i), sum);
        }
        sum = blockReduceSumInt32(sum);
        if (threadIdx.x == 0) {
             out[blockIdx.x * dim_out + j] = __float2half((float)sum * local_max * (float)__ldg(scale_mat + j));
        }
    }
}

// block <batch>,   thread <min(1024, round_up(dim_in // 2, 32))>
CPM_KERNEL_EXPORT void cu_gemv_fp16(
    int32_t batch, int32_t dim_out, int32_t dim_in,
    const half2 *mat,                   //  <batch, dim_out, dim_in>
    const half2 *vec,                   //  <batch, dim_in>
    half *out                           //  <batch, dim_out>
) {
    int32_t base_v = blockIdx.x * (dim_in / 2) + threadIdx.x;

    for (int j = 0; j < dim_out; j ++) {
        int32_t base_mat = (blockIdx.x * dim_out + j) * dim_in / 2 + threadIdx.x;
        half2 sum = __float2half2_rn(0);

        for (int i = 0; (i + threadIdx.x) * 2 < dim_in; i += blockDim.x) {
            sum = __hfma2(__ldg(vec + base_v + i), mat[base_mat + i], sum);
        }
        float v = sum.x + sum.y;
        v = blockReduceSum(v);
        if (threadIdx.x == 0) {
            out[blockIdx.x * dim_out + j] = __float2half(v);
        }
    }
}

