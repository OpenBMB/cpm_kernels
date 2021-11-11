
#include "reduce.cuh"
#include <cuda_fp16.h>
#include "common.h"

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

