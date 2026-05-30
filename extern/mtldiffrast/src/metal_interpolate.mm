// Metal interpolation implementation — zero-copy MPS path.
#import "metal_interpolate.h"
#import "metal_utils.h"

namespace mtldiffrast {

std::tuple<torch::Tensor, torch::Tensor> interpolate(
    const torch::Tensor& attr,
    const torch::Tensor& rast,
    const torch::Tensor& tri,
    const torch::Tensor& rast_db,
    bool enable_da
) {
    TORCH_CHECK(attr.dim() == 2, "attr must be [V, A]");
    TORCH_CHECK(rast.dim() == 3 && rast.size(2) == 4, "rast must be [H, W, 4]");
    TORCH_CHECK(tri.dim() == 2 && tri.size(1) == 3, "tri must be [T, 3]");

    int H = rast.size(0);
    int W = rast.size(1);
    int A = attr.size(1);
    int V = attr.size(0);
    int T = tri.size(0);

    auto attr_c = attr.contiguous().to(torch::kFloat32);
    auto rast_c = rast.contiguous().to(torch::kFloat32);
    auto tri_c = tri.contiguous().to(torch::kInt32).view({-1});

    // Output on same device as input
    auto output = make_output_tensor({H, W, A}, torch::kFloat32, attr);
    torch::Tensor output_da;

    struct {
        int num_triangles;
        int num_vertices;
        int num_attr;
        int width;
        int height;
    } params = {T, V, A, W, H};

    auto dev = mtl_get_device();
    auto queue = mtl_get_queue();

    if (any_tensor_on_mps(attr, rast, tri)) mps_sync();

    auto tri_ref = tensor_to_mtl_buffer(tri_c);
    auto attr_ref = tensor_to_mtl_buffer(attr_c);
    auto rast_ref = tensor_to_mtl_buffer(rast_c);
    auto out_ref = tensor_to_mtl_buffer(output);

    bool on_mps = output.device().is_mps();

    if (enable_da && rast_db.defined()) {
        output_da = make_output_tensor({H, W, A, 2}, torch::kFloat32, attr);
        auto rast_db_c = rast_db.contiguous().to(torch::kFloat32);
        auto db_ref = tensor_to_mtl_buffer(rast_db_c);
        auto da_ref = tensor_to_mtl_buffer(output_da);

        auto pso = mtl_get_pipeline("interpolate_fwd_da_kernel");
        dispatch_kernel(on_mps, ^(id<MTLCommandBuffer> cmdBuf, id<MTLComputeCommandEncoder> enc) {
            (void)cmdBuf;
            [enc setComputePipelineState:pso];
            [enc setBuffer:tri_ref.buffer  offset:tri_ref.offset  atIndex:0];
            [enc setBuffer:attr_ref.buffer offset:attr_ref.offset atIndex:1];
            [enc setBuffer:rast_ref.buffer offset:rast_ref.offset atIndex:2];
            [enc setBuffer:db_ref.buffer   offset:db_ref.offset   atIndex:3];
            [enc setBytes:&params length:sizeof(params) atIndex:4];
            [enc setBuffer:out_ref.buffer  offset:out_ref.offset  atIndex:5];
            [enc setBuffer:da_ref.buffer   offset:da_ref.offset   atIndex:6];
            [enc dispatchThreads:MTLSizeMake(W, H, 1)
                threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        });

        output_da = output_da.view({H, W, 2 * A});
    } else {
        auto pso = mtl_get_pipeline("interpolate_fwd_kernel");
        dispatch_kernel(on_mps, ^(id<MTLCommandBuffer> cmdBuf, id<MTLComputeCommandEncoder> enc) {
            (void)cmdBuf;
            [enc setComputePipelineState:pso];
            [enc setBuffer:tri_ref.buffer  offset:tri_ref.offset  atIndex:0];
            [enc setBuffer:attr_ref.buffer offset:attr_ref.offset atIndex:1];
            [enc setBuffer:rast_ref.buffer offset:rast_ref.offset atIndex:2];
            [enc setBytes:&params length:sizeof(params) atIndex:3];
            [enc setBuffer:out_ref.buffer  offset:out_ref.offset  atIndex:4];
            [enc dispatchThreads:MTLSizeMake(W, H, 1)
                threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        });

        output_da = torch::Tensor();
    }

    return {output, output_da};
}

std::tuple<torch::Tensor, torch::Tensor> interpolate_grad(
    const torch::Tensor& attr,
    const torch::Tensor& rast,
    const torch::Tensor& tri,
    const torch::Tensor& dy
) {
    TORCH_CHECK(attr.dim() == 2, "attr must be [V, A]");
    TORCH_CHECK(rast.dim() == 3 && rast.size(2) == 4, "rast must be [H, W, 4]");
    TORCH_CHECK(tri.dim() == 2 && tri.size(1) == 3, "tri must be [T, 3]");

    int H = rast.size(0), W = rast.size(1);
    int A = attr.size(1), V = attr.size(0), T = tri.size(0);

    auto attr_c = attr.contiguous().to(torch::kFloat32);
    auto rast_c = rast.contiguous().to(torch::kFloat32);
    auto tri_c = tri.contiguous().to(torch::kInt32).view({-1});
    auto dy_c = dy.contiguous().to(torch::kFloat32);

    auto grad_attr = make_output_tensor({V, A}, torch::kFloat32, attr);
    auto grad_rast = make_output_tensor({H, W, 4}, torch::kFloat32, rast);

    struct { int num_triangles; int num_vertices; int num_attr; int width; int height; } params =
        {T, V, A, W, H};

    if (any_tensor_on_mps(attr, rast, tri, dy)) mps_sync();

    auto tri_ref = tensor_to_mtl_buffer(tri_c);
    auto attr_ref = tensor_to_mtl_buffer(attr_c);
    auto rast_ref = tensor_to_mtl_buffer(rast_c);
    auto dy_ref = tensor_to_mtl_buffer(dy_c);
    auto ga_ref = tensor_to_mtl_buffer(grad_attr);
    auto gr_ref = tensor_to_mtl_buffer(grad_rast);

    bool on_mps = grad_attr.device().is_mps();
    auto pso = mtl_get_pipeline("interpolate_grad_kernel");
    dispatch_kernel(on_mps, ^(id<MTLCommandBuffer> cmdBuf, id<MTLComputeCommandEncoder> enc) {
        (void)cmdBuf;
        [enc setComputePipelineState:pso];
        [enc setBuffer:tri_ref.buffer  offset:tri_ref.offset  atIndex:0];
        [enc setBuffer:attr_ref.buffer offset:attr_ref.offset atIndex:1];
        [enc setBuffer:rast_ref.buffer offset:rast_ref.offset atIndex:2];
        [enc setBuffer:dy_ref.buffer   offset:dy_ref.offset   atIndex:3];
        [enc setBuffer:ga_ref.buffer   offset:ga_ref.offset   atIndex:4];
        [enc setBuffer:gr_ref.buffer   offset:gr_ref.offset   atIndex:5];
        [enc setBytes:&params length:sizeof(params) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(W, H, 1)
            threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    });

    return {grad_attr, grad_rast};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> interpolate_grad_da(
    const torch::Tensor& attr,
    const torch::Tensor& rast,
    const torch::Tensor& tri,
    const torch::Tensor& dy,
    const torch::Tensor& rast_db,
    const torch::Tensor& dda
) {
    TORCH_CHECK(attr.dim() == 2, "attr must be [V, A]");
    TORCH_CHECK(rast.dim() == 3 && rast.size(2) == 4, "rast must be [H, W, 4]");

    int H = rast.size(0), W = rast.size(1);
    int A = attr.size(1), V = attr.size(0), T = tri.size(0);

    auto attr_c = attr.contiguous().to(torch::kFloat32);
    auto rast_c = rast.contiguous().to(torch::kFloat32);
    auto tri_c = tri.contiguous().to(torch::kInt32).view({-1});
    auto dy_c = dy.contiguous().to(torch::kFloat32);
    auto db_c = rast_db.contiguous().to(torch::kFloat32);
    auto dda_c = dda.contiguous().to(torch::kFloat32);

    auto grad_attr = make_output_tensor({V, A}, torch::kFloat32, attr);
    auto grad_rast = make_output_tensor({H, W, 4}, torch::kFloat32, rast);
    auto grad_rast_db = make_output_tensor({H, W, 4}, torch::kFloat32, rast);

    struct { int num_triangles; int num_vertices; int num_attr; int width; int height; } params =
        {T, V, A, W, H};

    if (any_tensor_on_mps(attr, rast, tri, dy, rast_db, dda)) mps_sync();

    auto tri_ref = tensor_to_mtl_buffer(tri_c);
    auto attr_ref = tensor_to_mtl_buffer(attr_c);
    auto rast_ref = tensor_to_mtl_buffer(rast_c);
    auto db_ref = tensor_to_mtl_buffer(db_c);
    auto dy_ref = tensor_to_mtl_buffer(dy_c);
    auto dda_ref = tensor_to_mtl_buffer(dda_c);
    auto ga_ref = tensor_to_mtl_buffer(grad_attr);
    auto gr_ref = tensor_to_mtl_buffer(grad_rast);
    auto grdb_ref = tensor_to_mtl_buffer(grad_rast_db);

    bool on_mps = grad_attr.device().is_mps();
    auto pso = mtl_get_pipeline("interpolate_grad_da_kernel");
    dispatch_kernel(on_mps, ^(id<MTLCommandBuffer> cmdBuf, id<MTLComputeCommandEncoder> enc) {
        (void)cmdBuf;
        [enc setComputePipelineState:pso];
        [enc setBuffer:tri_ref.buffer  offset:tri_ref.offset  atIndex:0];
        [enc setBuffer:attr_ref.buffer offset:attr_ref.offset atIndex:1];
        [enc setBuffer:rast_ref.buffer offset:rast_ref.offset atIndex:2];
        [enc setBuffer:db_ref.buffer   offset:db_ref.offset   atIndex:3];
        [enc setBuffer:dy_ref.buffer   offset:dy_ref.offset   atIndex:4];
        [enc setBuffer:dda_ref.buffer  offset:dda_ref.offset  atIndex:5];
        [enc setBuffer:ga_ref.buffer   offset:ga_ref.offset   atIndex:6];
        [enc setBuffer:gr_ref.buffer   offset:gr_ref.offset   atIndex:7];
        [enc setBuffer:grdb_ref.buffer offset:grdb_ref.offset atIndex:8];
        [enc setBytes:&params length:sizeof(params) atIndex:9];
        [enc dispatchThreads:MTLSizeMake(W, H, 1)
            threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    });

    return {grad_attr, grad_rast, grad_rast_db};
}

} // namespace mtldiffrast
