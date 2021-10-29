#include "common.h"

// block <key_len>    <min(query_len, 1024)>
CPM_KERNEL_EXPORT void cu_position_bucket(
    int32_t query_len,
    int32_t key_len,
    int32_t num_buckets,
    int32_t max_distance,
    int32_t *out,   // (key_len, query_len)
    bool bidirectional
) {
    int32_t max_exact = num_buckets / (bidirectional ? 4 : 2);
    float v = logf(max_distance / max_exact);

    for (int i = threadIdx.x; i < query_len; i += blockDim.x) {
        int32_t relative_position = i - blockIdx.x;
        int32_t bucket_offset = 0;
        if (relative_position < 0) {
            if (bidirectional) {
                bucket_offset = num_buckets / 2;
                relative_position = -relative_position;
            }  else {
                relative_position = 0;
            }
        }
        if (relative_position > max_distance) relative_position = max_distance;

        if (relative_position < max_exact) {
            out[i + blockIdx.x * query_len] = relative_position + bucket_offset;
        } else {
            bucket_offset += max_exact;
            out[i + blockIdx.x * query_len] = logf((float)relative_position / (float)max_exact) / v * max_exact + bucket_offset;
        }
    }
}