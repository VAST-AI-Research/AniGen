// Host-side GPU parallel primitives using Metal compute shaders
#pragma once

#import "metal_context.h"
#import <Metal/Metal.h>
#import <vector>

namespace mtlmesh {

class MetalPrimitives {
public:
    static MetalPrimitives& instance();

    // Exclusive prefix sum: output[i] = sum(input[0..i-1])
    // In-place capable (input == output ok)
    void exclusive_sum(id<MTLBuffer> data, int N);

    // Inclusive prefix sum: output[i] = sum(input[0..i])
    void inclusive_sum(id<MTLBuffer> data, int N);

    // Stream compaction: select elements where flags[i] == 1
    // Returns count of selected elements
    int select_flagged_int(id<MTLBuffer> input, id<MTLBuffer> flags,
                           id<MTLBuffer> output, int N);
    int select_flagged_int3(id<MTLBuffer> input, id<MTLBuffer> flags,
                            id<MTLBuffer> output, int N);

    // Radix sort keys (uint32)
    void sort_keys_uint32(id<MTLBuffer> keys, int N);

    // Radix sort key-value pairs
    void sort_pairs_int(id<MTLBuffer> keys, id<MTLBuffer> values, int N);
    void sort_pairs_uint64(id<MTLBuffer> keys, id<MTLBuffer> values, int N);

    // Run-length encode: given sorted keys, find unique keys and their counts
    // Returns number of unique keys
    int run_length_encode(id<MTLBuffer> sorted_keys, id<MTLBuffer> unique_keys,
                          id<MTLBuffer> counts, int N);

    // Segmented reduce (sum)
    void segmented_sum_float(id<MTLBuffer> input, id<MTLBuffer> output,
                             id<MTLBuffer> offsets, int num_segments);
    void segmented_max_float(id<MTLBuffer> input, id<MTLBuffer> output,
                             id<MTLBuffer> offsets, int num_segments);
    void segmented_sum_float3(id<MTLBuffer> input, id<MTLBuffer> output,
                              id<MTLBuffer> offsets, int num_segments);

    // Reduce by key (sum)
    int reduce_by_key_float(id<MTLBuffer> sorted_keys, id<MTLBuffer> sorted_values,
                            id<MTLBuffer> unique_keys, id<MTLBuffer> reduced_values,
                            int N);

    // Global reduce sum
    int reduce_sum_int(id<MTLBuffer> input, int N);

private:
    MetalPrimitives() = default;
    id<MTLBuffer> temp_buffer(size_t bytes);

    // Temp buffers for multi-pass algorithms
    std::vector<id<MTLBuffer>> temp_buffers_;
};

} // namespace mtlmesh
