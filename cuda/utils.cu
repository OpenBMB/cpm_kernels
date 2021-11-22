
#include "reduce.cuh"
#include <cuda_fp16.h>
#include "common.h"

__inline__ __device__ bool isnan_(half v) {
#if __CUDA_ARCH__ >= 700 || __CUDA_ARCH__ == 600
    return __hisnan(v);
#else
    return v != v;
#endif
}

// grid <batch>,    thread<min(1024, n // 2)>
CPM_KERNEL_EXPORT void copy_data_to_kv(
    int32_t batch, int32_t buffer_len, int32_t n,
    const half2 *in,        // (batch, n)
    half2 *out,             // (batch, buffer_len, n)
    int32_t pos
) {
    int32_t half_n = n >> 1;
    int32_t base_in_idx = blockIdx.x * half_n + threadIdx.x;
    int32_t base_out_idx = (blockIdx.x * buffer_len + pos) * half_n + threadIdx.x;
    for (int i = 0; i < half_n; i += blockDim.x) {
        if (threadIdx.x + i < half_n) {
            out[base_out_idx + i] = in[base_in_idx + i];
        }
    }
}

// grid<1>,     thread<1>
CPM_KERNEL_EXPORT void cu_array_add(
    int32_t *arr, int32_t pos, int32_t val
) {
    if (threadIdx.x == 0) arr[pos] += val;
}

// grid<batch, n // 1024>,     thread<min(n, 1024)>
CPM_KERNEL_EXPORT void cu_adjustify_logits(
    int32_t batch, int32_t n,
    half *logits,                   // (batch, n)
    float temperature,
    float frequency_penalty,
    float presence_penalty,
    int32_t *frequency              // (batch, n)
) {
    int32_t col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < n) {
        float v = __half2float(logits[ blockIdx.x * n + col ]);
        int32_t freq = frequency[ blockIdx.x * n + col ];
        v /= temperature;
        v -= frequency_penalty * (float)freq;
        v -= presence_penalty * (freq > 0 ? 1.0f : 0.0f);
        logits[ blockIdx.x * n + col ] = __float2half(v);
    }
}

// grid <batch, old_size // 1024>   block <min(old_size, 1024)>
CPM_KERNEL_EXPORT void cu_copy_extend_buffer(
    int32_t batch, int32_t old_size, int32_t nw_size,
    const half* old_buf,    // (batch, old_size)
    half* nw_buf            // (batch, nw_size)
) {
    int32_t col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < old_size) {
        nw_buf[ blockIdx.x * nw_size + col ] = old_buf[ blockIdx.x * old_size + col ];
    }
}

// grid <1>,        thread<min(round_up(n, 32), 1024)>
CPM_KERNEL_EXPORT void cu_has_nan_inf(
    int32_t n,
    const half* inp,    // (n,) 
    int8_t* out
) {
    float r = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        half v = inp[i];
        if (__hisinf(v) || isnan_(v)) {
            r = 10;
            break;
        }
    }
    r = blockReduceSum(r);
    if (threadIdx.x == 0 && r > 1) {
        out[0] = 1;
    }
}