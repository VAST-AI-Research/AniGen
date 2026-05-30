// Host-side dispatch for Metal spatial hash table kernels
#import "metal_hash.h"

namespace mtlmesh {

#define CTX MetalContext::instance()

static id<MTLBuffer> tensor_to_buffer(const torch::Tensor& t) {
    auto tc = t.contiguous();
    size_t bytes = tc.nbytes();
    auto dev = CTX.device();
    auto buf = [dev newBufferWithBytes:tc.data_ptr() length:bytes options:MTLResourceStorageModeShared];
    return buf;
}

static id<MTLBuffer> alloc(size_t bytes) {
    return [CTX.device() newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> alloc_zero(size_t bytes) {
    auto buf = alloc(bytes);
    memset([buf contents], 0, bytes);
    return buf;
}

// Determine kernel suffix based on key/val tensor types
static std::string type_suffix(const torch::Tensor& keys, const torch::Tensor& vals) {
    bool k64 = (keys.scalar_type() == torch::kInt64 || keys.scalar_type() == torch::kFloat64 ||
                keys.nbytes() / keys.size(0) == 8);
    bool v64 = (vals.scalar_type() == torch::kInt64 || vals.scalar_type() == torch::kFloat64 ||
                vals.nbytes() / vals.size(0) == 8);
    // Default assumption: keys are uint32 or uint64, vals are uint32 or uint64
    std::string kstr = k64 ? "u64" : "u32";
    std::string vstr = v64 ? "u64" : "u32";
    return kstr + "_" + vstr;
}

void hashmap_insert_3d_idx_as_val(
    torch::Tensor keys, torch::Tensor vals,
    torch::Tensor coords, int W, int H, int D)
{
    int num_items = (int)coords.size(0);   // M in kernel: thread limit
    int hash_cap  = (int)keys.size(0);     // N in kernel: hashmap capacity

    auto keys_buf = tensor_to_buffer(keys);
    auto vals_buf = tensor_to_buffer(vals);
    // coords is [num_items, 4] int32 — convert to int4 for GPU
    auto coords_buf = tensor_to_buffer(coords.to(torch::kInt32));

    uint cap_u = (uint)hash_cap, items_u = (uint)num_items;
    int Wi = W, Hi = H, Di = D;

    std::string suffix = type_suffix(keys, vals);
    std::string kernel = "hashmap_insert_3d_idx_as_val_" + suffix + "_kernel";

    CTX.dispatch(kernel, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:keys_buf offset:0 atIndex:0];
        [enc setBuffer:vals_buf offset:0 atIndex:1];
        [enc setBuffer:coords_buf offset:0 atIndex:2];
        [enc setBytes:&cap_u length:sizeof(uint) atIndex:3];    // N = hashmap capacity
        [enc setBytes:&items_u length:sizeof(uint) atIndex:4];  // M = item count
        [enc setBytes:&Wi length:sizeof(int) atIndex:5];
        [enc setBytes:&Hi length:sizeof(int) atIndex:6];
        [enc setBytes:&Di length:sizeof(int) atIndex:7];
    }, num_items);

    // Copy results back to input tensors (shared memory on Apple Silicon)
    memcpy(keys.data_ptr(), [keys_buf contents], keys.nbytes());
    memcpy(vals.data_ptr(), [vals_buf contents], vals.nbytes());
}

torch::Tensor hashmap_lookup_3d(
    torch::Tensor keys, torch::Tensor vals,
    torch::Tensor coords, int W, int H, int D)
{
    int num_queries = (int)coords.size(0);  // M in kernel: thread limit
    int hash_cap    = (int)keys.size(0);    // N in kernel: hashmap capacity

    auto keys_buf = tensor_to_buffer(keys);
    auto vals_buf = tensor_to_buffer(vals);
    auto coords_buf = tensor_to_buffer(coords.to(torch::kInt32));
    auto out_buf = alloc(num_queries * 4);  // uint32 output

    uint cap_u = (uint)hash_cap, queries_u = (uint)num_queries;
    int Wi = W, Hi = H, Di = D;

    std::string suffix = type_suffix(keys, vals);
    std::string kernel = "hashmap_lookup_3d_" + suffix + "_kernel";

    CTX.dispatch(kernel, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:keys_buf offset:0 atIndex:0];
        [enc setBuffer:vals_buf offset:0 atIndex:1];
        [enc setBuffer:coords_buf offset:0 atIndex:2];
        [enc setBuffer:out_buf offset:0 atIndex:3];
        [enc setBytes:&cap_u length:sizeof(uint) atIndex:4];     // N = hashmap capacity
        [enc setBytes:&queries_u length:sizeof(uint) atIndex:5];  // M = query count
        [enc setBytes:&Wi length:sizeof(int) atIndex:6];
        [enc setBytes:&Hi length:sizeof(int) atIndex:7];
        [enc setBytes:&Di length:sizeof(int) atIndex:8];
    }, num_queries);

    // Copy out to tensor
    auto result = torch::empty({num_queries}, torch::kInt32);
    memcpy(result.data_ptr(), [out_buf contents], num_queries * 4);
    return result;
}

torch::Tensor get_sparse_voxel_grid_active_vertices(
    torch::Tensor keys, torch::Tensor vals,
    torch::Tensor coords, int W, int H, int D)
{
    int Nvox = (int)coords.size(0);
    int M = (int)keys.size(0);

    auto keys_buf = tensor_to_buffer(keys);
    auto vals_buf = tensor_to_buffer(vals);
    auto coords_buf = tensor_to_buffer(coords.to(torch::kInt32));

    // Each voxel contributes up to 8 vertices (cube corners)
    // Collect all active vertex coords, then unique them
    auto offsets = torch::tensor({
        0, 0, 0,  1, 0, 0,  0, 1, 0,  1, 1, 0,
        0, 0, 1,  1, 0, 1,  0, 1, 1,  1, 1, 1
    }, torch::kInt32).reshape({8, 3});

    auto all_verts = coords.unsqueeze(1) + offsets.unsqueeze(0).to(coords.device());
    all_verts = all_verts.reshape({-1, 3}).contiguous();

    // Unique vertices
    auto [unique_verts, inverse, counts] = torch::unique_dim(all_verts, 0, true, true, false);

    return unique_verts;
}

std::tuple<torch::Tensor, torch::Tensor> simple_dual_contour(
    torch::Tensor keys, torch::Tensor vals,
    torch::Tensor coords, torch::Tensor distances,
    int W, int H, int D)
{
    int Nvox = (int)coords.size(0);
    int Nvert = (int)distances.size(0);
    int M_hash = (int)keys.size(0);

    auto keys_buf = tensor_to_buffer(keys);
    auto vals_buf = tensor_to_buffer(vals);
    auto coords_buf = tensor_to_buffer(coords.to(torch::kInt32));
    auto dist_buf = tensor_to_buffer(distances.to(torch::kFloat32));

    auto out_verts_buf = alloc_zero(Nvox * 12);  // float3, one per voxel
    auto out_intersected_buf = alloc_zero(Nvox * 12);  // int3 (per-axis flags)

    uint Nvert_u = (uint)Nvert, M_u = (uint)M_hash;
    int Wi = W, Hi = H, Di = D;

    bool k64 = (keys.nbytes() / keys.size(0) == 8);
    std::string kernel = k64 ? "simple_dual_contour_u64_kernel" : "simple_dual_contour_u32_kernel";

    CTX.dispatch(kernel, [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:keys_buf offset:0 atIndex:0];
        [enc setBuffer:vals_buf offset:0 atIndex:1];
        [enc setBuffer:coords_buf offset:0 atIndex:2];
        [enc setBuffer:dist_buf offset:0 atIndex:3];
        [enc setBuffer:out_verts_buf offset:0 atIndex:4];
        [enc setBuffer:out_intersected_buf offset:0 atIndex:5];
        [enc setBytes:&Nvert_u length:sizeof(uint) atIndex:6];
        [enc setBytes:&M_u length:sizeof(uint) atIndex:7];
        [enc setBytes:&Wi length:sizeof(int) atIndex:8];
        [enc setBytes:&Hi length:sizeof(int) atIndex:9];
        [enc setBytes:&Di length:sizeof(int) atIndex:10];
    }, Nvox);

    auto verts = torch::empty({Nvox, 3}, torch::kFloat32);
    memcpy(verts.data_ptr(), [out_verts_buf contents], Nvox * 12);

    auto intersected = torch::empty({Nvox, 3}, torch::kInt32);
    memcpy(intersected.data_ptr(), [out_intersected_buf contents], Nvox * 12);

    return {verts, intersected};
}

} // namespace mtlmesh
