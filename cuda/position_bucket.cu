#include "reduce.cuh"
#include <cuda_fp16.h>
#include "common.h"

// block <1>,   thread <min(max_distance, 1024)>
CPM_KERNEL_EXPORT void cu_init_position_mapping(
    int32_t num_buckets,
    int32_t max_distance,
    int32_t *out,   // (max_distance)
    bool bidirectional
) {
    int32_t part_buckets = num_buckets / (bidirectional ? 2 : 1);
    int32_t exact_buckets = part_buckets / 2;
    int32_t log_buckets = part_buckets - exact_buckets;

    float v = logf(max_distance / exact_buckets);
    for (int i = threadIdx.x; i < max_distance; i++) {
        if (i < exact_buckets) out[i] = i;
        else out[i] = (int32_t)(logf((float)i / (float)exact_buckets) / v * log_buckets) + exact_buckets;
    }
}


// block <key_len>    <min(query_len, 1024)>
CPM_KERNEL_EXPORT void cu_position_embedding_forward(
    int32_t query_len,
    int32_t key_len,
    int32_t num_buckets,
    int32_t max_distance,
    int32_t num_head,
    const int32_t *position_mapping,    // (max_distance)
    const half *weight,                 // (num_head, num_bucket)
    half *out,                          // (num_head, key_len, query_len)
    bool bidirectional
) {
    int32_t total_len = key_len * query_len;
    for (int i = threadIdx.x; i < query_len; i += blockDim.x) {
        int32_t relative_position = i - blockIdx.x;
        int32_t bucket_offset = 0;
        if (relative_position < 0) {
            if (bidirectional) {
                relative_position = -relative_position;
                bucket_offset = num_buckets / 2;
            } else {
                relative_position = 0;
            }
        }
        if (relative_position >= max_distance) relative_position = max_distance - 1;
        int32_t bucket = __ldg(position_mapping + relative_position) + bucket_offset;
        for (int j = 0; j < num_head; j++){
            out[j * total_len + blockIdx.x * query_len + i] = __ldg(weight + j * num_buckets + bucket);
        }
    }
}

// block <num_buckets>,     thread <1024>
CPM_KERNEL_EXPORT void cu_position_embedding_backward(
    int32_t query_len,
    int32_t key_len,
    int32_t num_buckets,
    int32_t max_distance,
    int32_t num_heads,          // no more than 1024 heads
    const int32_t *position_mapping,    // (max_distance)
    const half *grad_out,               // (num_head, key_len, query_len) 
    half *grad,                         // (num_head, num_bucket)
    bool bidirectional
) {
    __shared__ float sum[1024];

    int32_t total_len = key_len * query_len;

    sum[threadIdx.x] = 0;

    for (int i = 0; i < total_len; i += blockDim.x) {
        int32_t bucket = -1;
        if (i + threadIdx.x < total_len) {
            int32_t relative_position = ((i + threadIdx.x) % query_len) - ((i + threadIdx.x) / query_len);
            int32_t bucket_offset = 0;
            if (relative_position < 0) {
                if (bidirectional) {
                    relative_position = -relative_position;
                    bucket_offset = num_buckets / 2;
                } else {
                    relative_position = 0;
                }
            }
            if (relative_position >= max_distance) relative_position = max_distance - 1;
            bucket = __ldg(position_mapping + relative_position) + bucket_offset;
        }

        for (int j = 0; j < num_heads; j ++) {
            float v = 0;
            if (bucket == blockIdx.x) v = (float)__ldg(grad_out + j * total_len + i + threadIdx.x);
            v = blockReduceSum(v);  // synchronized here
            if (threadIdx.x == 0) sum[j] += v;
        }
    }
    __syncthreads();
    if (threadIdx.x < num_heads) {
        grad[ threadIdx.x * num_buckets + blockIdx.x ] = sum[threadIdx.x];
    }
}

// block <1>    <min(key_len, 1024)>
CPM_KERNEL_EXPORT void cu_position_embedding_step(
    int32_t query_pos,
    int32_t key_len,
    int32_t num_buckets,
    int32_t max_distance,
    int32_t num_head,
    const int32_t *position_mapping,    // (max_distance)
    const half *weight,                 // (num_head, num_bucket)
    half *out,                          // (num_head, key_len)
    bool bidirectional
) {
    for (int i = threadIdx.x; i < key_len; i += blockDim.x) {
        int32_t relative_position = query_pos - i;
        int32_t bucket_offset = 0;
        if (relative_position < 0) {
            if (bidirectional) {
                relative_position = -relative_position;
                bucket_offset = num_buckets / 2;
            } else {
                relative_position = 0;
            }
        }
        if (relative_position >= max_distance) relative_position = max_distance - 1;
        int32_t bucket = __ldg(position_mapping + relative_position) + bucket_offset;
        for (int j = 0; j < num_head; j++){
            out[j * key_len + i] = __ldg(weight + j * num_buckets + bucket);
        }
    }
}