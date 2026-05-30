// Metal GPU parallel primitives implementation
// Pure Metal — no PyTorch, no MLX dependencies
#import "metal_primitives.h"

namespace mtlmesh {

MetalPrimitives& MetalPrimitives::instance() {
    static MetalPrimitives p;
    return p;
}

id<MTLBuffer> MetalPrimitives::temp_buffer(size_t bytes) {
    auto& ctx = MetalContext::instance();
    return [ctx.device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

// ========== Exclusive Prefix Sum ==========
// GPU Blelloch scan for N > 1024, CPU for small arrays.
// Multi-block: scan blocks → scan block sums → add offsets.

void MetalPrimitives::exclusive_sum(id<MTLBuffer> data, int N) {
    if (N <= 0) return;

    auto& ctx = MetalContext::instance();
    int block_size = 1024;

    if (N <= block_size) {
        // Single threadgroup GPU Blelloch scan
        auto block_sums = temp_buffer(sizeof(int));
        memset([block_sums contents], 0, sizeof(int));

        auto pso = ctx.pipeline("prefix_sum_threadgroup_int");
        id<MTLCommandBuffer> cmdBuf = [ctx.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:data offset:0 atIndex:0];
        [enc setBuffer:block_sums offset:0 atIndex:1];
        int n_val = N;
        [enc setBytes:&n_val length:sizeof(int) atIndex:2];
        // Round up to next power of 2 for Blelloch scan
        NSUInteger po2 = 1;
        while (po2 < (NSUInteger)N) po2 <<= 1;
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(po2, 1, 1)];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        return;
    }

    // Multi-block: scan each block, scan block sums recursively, propagate offsets
    int num_blocks = (N + block_size - 1) / block_size;

    auto block_sums_buf = temp_buffer(num_blocks * sizeof(int));
    memset([block_sums_buf contents], 0, num_blocks * sizeof(int));

    // Step 1: per-block prefix scan (each block produces its local exclusive sum + block total)
    auto pso = ctx.pipeline("prefix_sum_threadgroup_int");
    int n_val = N;
    id<MTLCommandBuffer> cmdBuf = [ctx.queue() commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:data offset:0 atIndex:0];
    [enc setBuffer:block_sums_buf offset:0 atIndex:1];
    [enc setBytes:&n_val length:sizeof(int) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // Step 2: recursively scan block sums
    exclusive_sum(block_sums_buf, num_blocks);

    // Step 3: add block offsets to each element
    // MUST use same threadgroup size as the scan (block_size) so gid matches
    auto pso2 = ctx.pipeline("add_block_offsets_int");
    cmdBuf = [ctx.queue() commandBuffer];
    enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pso2];
    [enc setBuffer:data offset:0 atIndex:0];
    [enc setBuffer:block_sums_buf offset:0 atIndex:1];
    [enc setBytes:&n_val length:sizeof(int) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(num_blocks, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(block_size, 1, 1)];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
}

void MetalPrimitives::inclusive_sum(id<MTLBuffer> data, int N) {
    if (N <= 0) return;
    // Save original values, run GPU exclusive sum, add originals back
    auto backup = temp_buffer(N * sizeof(int));
    memcpy([backup contents], [data contents], N * sizeof(int));
    exclusive_sum(data, N);
    // inclusive[i] = exclusive[i] + original[i]
    // CPU add — small cost relative to the scan itself
    int* d = (int*)[data contents];
    int* b = (int*)[backup contents];
    for (int i = 0; i < N; i++) d[i] += b[i];
}

// ========== Stream Compaction ==========

int MetalPrimitives::select_flagged_int(id<MTLBuffer> input, id<MTLBuffer> flags,
                                         id<MTLBuffer> output, int N) {
    if (N <= 0) return 0;

    // Compute prefix sum of flags to get output positions
    auto offsets = temp_buffer((N + 1) * sizeof(int));
    memcpy([offsets contents], [flags contents], N * sizeof(int));
    exclusive_sum(offsets, N);

    // Get total count
    int* off_ptr = (int*)[offsets contents];
    int* flag_ptr = (int*)[flags contents];
    int count = off_ptr[N - 1] + flag_ptr[N - 1];

    // Scatter selected elements
    auto& ctx = MetalContext::instance();
    ctx.dispatch("scatter_flagged_int", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:input offset:0 atIndex:0];
        [enc setBuffer:flags offset:0 atIndex:1];
        [enc setBuffer:offsets offset:0 atIndex:2];
        [enc setBytes:&N length:sizeof(int) atIndex:3];
        [enc setBuffer:output offset:0 atIndex:4];
    }, N);

    return count;
}

int MetalPrimitives::select_flagged_int3(id<MTLBuffer> input, id<MTLBuffer> flags,
                                          id<MTLBuffer> output, int N) {
    if (N <= 0) return 0;

    auto offsets = temp_buffer((N + 1) * sizeof(int));
    memcpy([offsets contents], [flags contents], N * sizeof(int));
    exclusive_sum(offsets, N);

    int* off_ptr = (int*)[offsets contents];
    int* flag_ptr = (int*)[flags contents];
    int count = off_ptr[N - 1] + flag_ptr[N - 1];

    auto& ctx = MetalContext::instance();
    ctx.dispatch("scatter_flagged_int3", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:input offset:0 atIndex:0];
        [enc setBuffer:flags offset:0 atIndex:1];
        [enc setBuffer:offsets offset:0 atIndex:2];
        [enc setBytes:&N length:sizeof(int) atIndex:3];
        [enc setBuffer:output offset:0 atIndex:4];
    }, N);

    return count;
}

// ========== Radix Sort ==========
// 8-pass radix sort with 4-bit radix (16 buckets)
// Uses shared memory histograms + prefix sums

void MetalPrimitives::sort_pairs_int(id<MTLBuffer> keys, id<MTLBuffer> values, int N) {
    if (N <= 1) return;
    auto& ctx = MetalContext::instance();

    auto keys_temp = temp_buffer(N * sizeof(uint32_t));
    auto vals_temp = temp_buffer(N * sizeof(int));
    auto histogram = temp_buffer(16 * sizeof(int));
    auto counters = temp_buffer(16 * sizeof(int));

    id<MTLBuffer> k_in = keys, v_in = values;
    id<MTLBuffer> k_out = keys_temp, v_out = vals_temp;

    for (int shift = 0; shift < 32; shift += 4) {
        // Zero histogram
        memset([histogram contents], 0, 16 * sizeof(int));

        // Count occurrences of each digit (CPU fallback for simplicity/correctness)
        uint32_t* k = (uint32_t*)[k_in contents];
        int* hist = (int*)[histogram contents];
        for (int i = 0; i < N; i++) {
            uint32_t digit = (k[i] >> shift) & 0xF;
            hist[digit]++;
        }

        // Exclusive prefix sum on histogram
        int prefix[16];
        prefix[0] = 0;
        for (int i = 1; i < 16; i++) prefix[i] = prefix[i-1] + hist[i-1];

        // Scatter (CPU for guaranteed correctness — GPU version for perf later)
        uint32_t* k_o = (uint32_t*)[k_out contents];
        int* v_i = (int*)[v_in contents];
        int* v_o = (int*)[v_out contents];
        int count[16] = {};
        for (int i = 0; i < N; i++) {
            uint32_t digit = (k[i] >> shift) & 0xF;
            int pos = prefix[digit] + count[digit]++;
            k_o[pos] = k[i];
            v_o[pos] = v_i[i];
        }

        // Swap buffers
        std::swap(k_in, k_out);
        std::swap(v_in, v_out);
    }

    // If result ended up in temp buffers, copy back
    if (k_in != keys) {
        memcpy([keys contents], [k_in contents], N * sizeof(uint32_t));
        memcpy([values contents], [v_in contents], N * sizeof(int));
    }
}

void MetalPrimitives::sort_pairs_uint64(id<MTLBuffer> keys, id<MTLBuffer> values, int N) {
    if (N <= 1) return;

    auto keys_temp = temp_buffer(N * sizeof(uint64_t));
    auto vals_temp = temp_buffer(N * sizeof(int));

    id<MTLBuffer> k_in = keys, v_in = values;
    id<MTLBuffer> k_out = keys_temp, v_out = vals_temp;

    // 16 passes for 64-bit keys
    for (int shift = 0; shift < 64; shift += 4) {
        int hist[16] = {};
        uint64_t* k = (uint64_t*)[k_in contents];
        for (int i = 0; i < N; i++) {
            hist[(k[i] >> shift) & 0xF]++;
        }

        int prefix[16]; prefix[0] = 0;
        for (int i = 1; i < 16; i++) prefix[i] = prefix[i-1] + hist[i-1];

        uint64_t* k_o = (uint64_t*)[k_out contents];
        int* v_i = (int*)[v_in contents];
        int* v_o = (int*)[v_out contents];
        int count[16] = {};
        for (int i = 0; i < N; i++) {
            int digit = (k[i] >> shift) & 0xF;
            int pos = prefix[digit] + count[digit]++;
            k_o[pos] = k[i];
            v_o[pos] = v_i[i];
        }

        std::swap(k_in, k_out);
        std::swap(v_in, v_out);
    }

    if (k_in != keys) {
        memcpy([keys contents], [k_in contents], N * sizeof(uint64_t));
        memcpy([values contents], [v_in contents], N * sizeof(int));
    }
}

void MetalPrimitives::sort_keys_uint32(id<MTLBuffer> keys, int N) {
    if (N <= 1) return;
    auto keys_temp = temp_buffer(N * sizeof(uint32_t));
    id<MTLBuffer> k_in = keys, k_out = keys_temp;

    for (int shift = 0; shift < 32; shift += 4) {
        int hist[16] = {};
        uint32_t* k = (uint32_t*)[k_in contents];
        for (int i = 0; i < N; i++) hist[(k[i] >> shift) & 0xF]++;

        int prefix[16]; prefix[0] = 0;
        for (int i = 1; i < 16; i++) prefix[i] = prefix[i-1] + hist[i-1];

        uint32_t* k_o = (uint32_t*)[k_out contents];
        int count[16] = {};
        for (int i = 0; i < N; i++) {
            int d = (k[i] >> shift) & 0xF;
            k_o[prefix[d] + count[d]++] = k[i];
        }
        std::swap(k_in, k_out);
    }
    if (k_in != keys) memcpy([keys contents], [k_in contents], N * sizeof(uint32_t));
}

// ========== Run-Length Encode ==========

int MetalPrimitives::run_length_encode(id<MTLBuffer> sorted_keys, id<MTLBuffer> unique_keys,
                                        id<MTLBuffer> counts, int N) {
    if (N <= 0) return 0;

    // CPU path for tiny arrays — avoids GPU dispatch overhead
    if (N <= 256) {
        int* sk = (int*)[sorted_keys contents];
        int* uk = (int*)[unique_keys contents];
        int* cnt = (int*)[counts contents];
        int num_runs = 0;
        for (int i = 0; i < N; ) {
            int key = sk[i];
            int run_len = 1;
            while (i + run_len < N && sk[i + run_len] == key) run_len++;
            uk[num_runs] = key;
            cnt[num_runs] = run_len;
            num_runs++;
            i += run_len;
        }
        return num_runs;
    }

    auto& ctx = MetalContext::instance();

    // Step 1: mark transitions (is_transition[i] = 1 if key[i] != key[i-1], 0 for first element)
    auto is_transition = temp_buffer(N * sizeof(int));
    ctx.dispatch("rle_mark_transitions", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:sorted_keys offset:0 atIndex:0];
        int n_val = N;
        [enc setBytes:&n_val length:sizeof(int) atIndex:1];
        [enc setBuffer:is_transition offset:0 atIndex:2];
    }, N);

    // Step 2: inclusive prefix sum → run IDs (0-indexed)
    // inclusive_sum([0,0,0,1,0,1]) = [0,0,0,1,1,2] — correct run IDs
    auto run_ids = temp_buffer(N * sizeof(int));
    memcpy([run_ids contents], [is_transition contents], N * sizeof(int));
    inclusive_sum(run_ids, N);

    // Number of runs = last run ID + 1
    int* run_ids_ptr = (int*)[run_ids contents];
    int num_runs = run_ids_ptr[N - 1] + 1;

    // Step 3: extract unique keys and accumulate run lengths (atomic)
    memset([counts contents], 0, num_runs * sizeof(int));
    ctx.dispatch("rle_extract", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:sorted_keys offset:0 atIndex:0];
        [enc setBuffer:run_ids offset:0 atIndex:1];
        int n_val = N;
        [enc setBytes:&n_val length:sizeof(int) atIndex:2];
        [enc setBuffer:unique_keys offset:0 atIndex:3];
        [enc setBuffer:counts offset:0 atIndex:4];
    }, N);

    return num_runs;
}

// ========== Segmented Reduce ==========

void MetalPrimitives::segmented_sum_float(id<MTLBuffer> input, id<MTLBuffer> output,
                                           id<MTLBuffer> offsets, int num_segments) {
    auto& ctx = MetalContext::instance();
    ctx.dispatch("segmented_reduce_sum_float", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:input offset:0 atIndex:0];
        [enc setBuffer:output offset:0 atIndex:1];
        [enc setBuffer:offsets offset:0 atIndex:2];
        [enc setBytes:&num_segments length:sizeof(int) atIndex:3];
    }, num_segments);
}

void MetalPrimitives::segmented_max_float(id<MTLBuffer> input, id<MTLBuffer> output,
                                           id<MTLBuffer> offsets, int num_segments) {
    auto& ctx = MetalContext::instance();
    ctx.dispatch("segmented_reduce_max_float", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:input offset:0 atIndex:0];
        [enc setBuffer:output offset:0 atIndex:1];
        [enc setBuffer:offsets offset:0 atIndex:2];
        [enc setBytes:&num_segments length:sizeof(int) atIndex:3];
    }, num_segments);
}

void MetalPrimitives::segmented_sum_float3(id<MTLBuffer> input, id<MTLBuffer> output,
                                            id<MTLBuffer> offsets, int num_segments) {
    auto& ctx = MetalContext::instance();
    ctx.dispatch("segmented_reduce_sum_float3", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:input offset:0 atIndex:0];
        [enc setBuffer:output offset:0 atIndex:1];
        [enc setBuffer:offsets offset:0 atIndex:2];
        [enc setBytes:&num_segments length:sizeof(int) atIndex:3];
    }, num_segments);
}

// ========== Reduce By Key ==========

int MetalPrimitives::reduce_by_key_float(id<MTLBuffer> sorted_keys, id<MTLBuffer> sorted_values,
                                          id<MTLBuffer> unique_keys, id<MTLBuffer> reduced_values,
                                          int N) {
    if (N <= 0) return 0;

    uint64_t* sk = (uint64_t*)[sorted_keys contents];
    float* sv = (float*)[sorted_values contents];
    uint64_t* uk = (uint64_t*)[unique_keys contents];
    float* rv = (float*)[reduced_values contents];

    int num_keys = 0;
    for (int i = 0; i < N; ) {
        uint64_t key = sk[i];
        float sum = 0;
        while (i < N && sk[i] == key) { sum += sv[i]; i++; }
        uk[num_keys] = key;
        rv[num_keys] = sum;
        num_keys++;
    }
    return num_keys;
}

// ========== Global Reduce ==========

int MetalPrimitives::reduce_sum_int(id<MTLBuffer> input, int N) {
    int* data = (int*)[input contents];
    int sum = 0;
    for (int i = 0; i < N; i++) sum += data[i];
    return sum;
}

} // namespace mtlmesh
