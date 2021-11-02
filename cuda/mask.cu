#include "reduce.cuh"
#include <cuda_fp16.h>
#include "common.h"

// block <batch_size, m / 1024>,  thread <min(m, 1024)>
CPM_KERNEL_EXPORT void cu_mask(
    int32_t batch, int32_t n, int32_t m,
    const half *x,          // (batch, n, m)
    const int8_t *mask,     // (batch, m)
    float value,
    half *out               // (batch, n, m)
) {
    int32_t col_idx = threadIdx.x + blockIdx.y * blockDim.x;
    int32_t base_x_idx = blockIdx.x * n * m + col_idx;
    half half_value = __float2half(value);

    if (col_idx < m) {
        int8_t mask_val = mask[blockIdx.x * m + col_idx];
        for (int i = 0; i < n; i ++) {
            out[base_x_idx + i * m] = (mask_val == 0) ? half_value : x[base_x_idx + i * m];
        }
    }
}