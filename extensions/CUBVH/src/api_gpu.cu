#include <gpu/api_gpu.h>
#include <gpu/common.h>
#include <gpu/bvh.cuh>
#include <gpu/floodfill.cuh>
#include <gpu/spcumc.cuh>
#include <gpu/hashtable.cuh>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iterator>
#include <stack>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace cubvh {

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> pack_state(
    const std::vector<Triangle>& triangles_cpu,
    const std::vector<TriangleBvhNode>& nodes_cpu) {

    auto float_opts = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
    auto long_opts = at::TensorOptions().dtype(at::kLong).device(at::kCPU);
    auto int_opts = at::TensorOptions().dtype(at::kInt).device(at::kCPU);

    const int64_t n_triangles = static_cast<int64_t>(triangles_cpu.size());
    at::Tensor triangle_vertices = at::empty({n_triangles, 3, 3}, float_opts);
    at::Tensor triangle_ids = at::empty({n_triangles}, long_opts);

    auto verts_acc = triangle_vertices.accessor<float, 3>();
    auto ids_acc = triangle_ids.accessor<int64_t, 1>();

    for (int64_t i = 0; i < n_triangles; ++i) {
        const Triangle& tri = triangles_cpu[i];
        ids_acc[i] = tri.id;
        const Eigen::Vector3f verts[3] = {tri.a, tri.b, tri.c};
        for (int v = 0; v < 3; ++v) {
            verts_acc[i][v][0] = verts[v].x();
            verts_acc[i][v][1] = verts[v].y();
            verts_acc[i][v][2] = verts[v].z();
        }
    }

    const int64_t n_nodes = static_cast<int64_t>(nodes_cpu.size());
    at::Tensor node_mins = at::empty({n_nodes, 3}, float_opts);
    at::Tensor node_maxs = at::empty({n_nodes, 3}, float_opts);
    at::Tensor node_children = at::empty({n_nodes, 3}, int_opts);

    auto mins_acc = node_mins.accessor<float, 2>();
    auto maxs_acc = node_maxs.accessor<float, 2>();
    auto child_acc = node_children.accessor<int, 2>();

    for (int64_t i = 0; i < n_nodes; ++i) {
        const TriangleBvhNode& node = nodes_cpu[i];
        mins_acc[i][0] = node.bb.min.x();
        mins_acc[i][1] = node.bb.min.y();
        mins_acc[i][2] = node.bb.min.z();

        maxs_acc[i][0] = node.bb.max.x();
        maxs_acc[i][1] = node.bb.max.y();
        maxs_acc[i][2] = node.bb.max.z();

        child_acc[i][0] = node.left_idx;
        child_acc[i][1] = node.right_idx;
        child_acc[i][2] = node.escape_idx;
    }

    return {triangle_vertices, triangle_ids, node_mins, node_maxs, node_children};
}

constexpr uint32_t kBranchingFactor = 4;

std::vector<TriangleBvhNode> build_nodes_cpu(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) {
    std::vector<TriangleBvhNode> nodes;
    if (triangles.empty()) {
        return nodes;
    }

    nodes.emplace_back();
    nodes.front().bb = BoundingBox(std::begin(triangles), std::end(triangles));

    struct BuildNode {
        int node_idx;
        std::vector<Triangle>::iterator begin;
        std::vector<Triangle>::iterator end;
    };

    std::stack<BuildNode> build_stack;
    build_stack.push({0, std::begin(triangles), std::end(triangles)});

    while (!build_stack.empty()) {
        BuildNode curr = build_stack.top();
        build_stack.pop();

        std::array<BuildNode, kBranchingFactor> children;
        children[0].begin = curr.begin;
        children[0].end = curr.end;

        int n_children = 1;
        while (n_children < static_cast<int>(kBranchingFactor)) {
            for (int i = n_children - 1; i >= 0; --i) {
                auto& child = children[i];

                const auto span = std::distance(child.begin, child.end);
                if (span <= 0) {
                    continue;
                }
                const float inv_count = 1.0f / static_cast<float>(span);

                Vector3f mean = Vector3f::Zero();
                for (auto it = child.begin; it != child.end; ++it) {
                    mean += it->centroid();
                }
                mean *= inv_count;

                Vector3f var = Vector3f::Zero();
                for (auto it = child.begin; it != child.end; ++it) {
                    Vector3f diff = it->centroid() - mean;
                    var += diff.cwiseProduct(diff);
                }
                var *= inv_count;

                Vector3f::Index axis;
                var.maxCoeff(&axis);

                auto mid = child.begin + span / 2;
                std::nth_element(child.begin, mid, child.end, [axis](const Triangle& tri1, const Triangle& tri2) {
                    return tri1.centroid(axis) < tri2.centroid(axis);
                });

                children[i * 2].begin = child.begin;
                children[i * 2].end = mid;
                children[i * 2 + 1].begin = mid;
                children[i * 2 + 1].end = child.end;
            }
            n_children *= 2;
        }

        nodes[curr.node_idx].left_idx = static_cast<int>(nodes.size());
        for (uint32_t i = 0; i < kBranchingFactor; ++i) {
            auto& child = children[i];
            if (child.begin == child.end) {
                continue;
            }

            child.node_idx = static_cast<int>(nodes.size());
            nodes.emplace_back();
            nodes.back().bb = BoundingBox(child.begin, child.end);

            const auto span = std::distance(child.begin, child.end);
            if (span <= static_cast<int64_t>(n_primitives_per_leaf)) {
                nodes.back().left_idx = -static_cast<int>(std::distance(std::begin(triangles), child.begin)) - 1;
                nodes.back().right_idx = -static_cast<int>(std::distance(std::begin(triangles), child.end)) - 1;
            } else {
                build_stack.push(child);
            }
        }
        nodes[curr.node_idx].right_idx = static_cast<int>(nodes.size());
    }

    for (auto& node : nodes) {
        node.escape_idx = -1;
    }

    std::function<void(int, int)> thread_bvh = [&](int node_idx, int escape_idx) {
        TriangleBvhNode& node = nodes[node_idx];
        node.escape_idx = escape_idx;
        if (node.left_idx < 0) {
            return;
        }
        int first_child = node.left_idx;
        int end_child = node.right_idx;
        for (int c = first_child; c < end_child; ++c) {
            int next_escape = (c + 1 < end_child) ? (c + 1) : escape_idx;
            thread_bvh(c, next_escape);
        }
    };

    if (!nodes.empty()) {
        thread_bvh(0, -1);
    }

    return nodes;
}

} // namespace

class cuBVHImpl : public cuBVH {
public:

    // accept numpy array (cpu) to init 
    cuBVHImpl(Ref<const Verts> vertices, Ref<const Trigs> triangles) : cuBVH() {

        const size_t n_triangles = triangles.rows();

        triangles_cpu.resize(n_triangles);

        for (size_t i = 0; i < n_triangles; i++) {
            triangles_cpu[i] = {vertices.row(triangles(i, 0)), vertices.row(triangles(i, 1)), vertices.row(triangles(i, 2)), (int64_t)i};
        }

        triangle_bvh = TriangleBvh::make();
        triangle_bvh->build(triangles_cpu, 8);
        nodes_cpu = triangle_bvh->host_nodes();

        triangles_gpu.resize_and_copy_from_host(triangles_cpu);

        // TODO: need OPTIX
        // triangle_bvh->build_optix(triangles_gpu, m_inference_stream);

    }

    cuBVHImpl(std::vector<Triangle>&& triangles, std::vector<TriangleBvhNode>&& nodes) : cuBVH() {
        triangles_cpu = std::move(triangles);
        nodes_cpu = std::move(nodes);

        triangle_bvh = TriangleBvh::make();
        triangle_bvh->set_nodes(nodes_cpu);
        nodes_cpu = triangle_bvh->host_nodes();

        triangles_gpu.resize_and_copy_from_host(triangles_cpu);
    }

    void ray_trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor face_id, at::Tensor depth) {

        const uint32_t n_elements = rays_o.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->ray_trace_gpu(n_elements, rays_o.data_ptr<float>(), rays_d.data_ptr<float>(), positions.data_ptr<float>(), face_id.data_ptr<int64_t>(), depth.data_ptr<float>(), triangles_gpu.data(), stream);
    }

    void unsigned_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw) {

        const uint32_t n_elements = positions.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->unsigned_distance_gpu(
            n_elements,
            positions.data_ptr<float>(),
            distances.data_ptr<float>(),
            face_id.data_ptr<int64_t>(),
            uvw.has_value() ? uvw.value().data_ptr<float>() : nullptr,
            triangles_gpu.data(),
            static_cast<uint32_t>(triangles_gpu.size()),
            stream
        );

    }

    void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) {

        const uint32_t n_elements = positions.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->signed_distance_gpu(
            n_elements,
            mode,
            positions.data_ptr<float>(),
            distances.data_ptr<float>(),
            face_id.data_ptr<int64_t>(),
            uvw.has_value() ? uvw.value().data_ptr<float>() : nullptr,
            triangles_gpu.data(),
            static_cast<uint32_t>(triangles_gpu.size()),
            stream
        );
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> export_state() const override {
        return pack_state(triangles_cpu, nodes_cpu);
    }

    std::vector<Triangle> triangles_cpu;
    std::vector<TriangleBvhNode> nodes_cpu;
    GPUMemory<Triangle> triangles_gpu;
    std::shared_ptr<TriangleBvh> triangle_bvh;
};

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> build_cuBVH_state(
    Ref<const Verts> vertices,
    Ref<const Trigs> triangles) {

    const size_t n_triangles = triangles.rows();
    std::vector<Triangle> triangles_cpu;
    triangles_cpu.resize(n_triangles);

    for (size_t i = 0; i < n_triangles; ++i) {
        triangles_cpu[i] = {
            vertices.row(triangles(i, 0)),
            vertices.row(triangles(i, 1)),
            vertices.row(triangles(i, 2)),
            static_cast<int64_t>(i)
        };
    }

    auto nodes_cpu = build_nodes_cpu(triangles_cpu, 8);

    return pack_state(triangles_cpu, nodes_cpu);
}
    
cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles) {
    return new cuBVHImpl{vertices, triangles};
}

cuBVH* create_cuBVH_from_state(
    at::Tensor triangle_vertices,
    at::Tensor triangle_ids,
    at::Tensor node_mins,
    at::Tensor node_maxs,
    at::Tensor node_children) {

    TORCH_CHECK(!triangle_vertices.is_cuda(), "triangle_vertices must reside on CPU");
    TORCH_CHECK(!triangle_ids.is_cuda(), "triangle_ids must reside on CPU");
    TORCH_CHECK(!node_mins.is_cuda(), "node_mins must reside on CPU");
    TORCH_CHECK(!node_maxs.is_cuda(), "node_maxs must reside on CPU");
    TORCH_CHECK(!node_children.is_cuda(), "node_children must reside on CPU");

    TORCH_CHECK(triangle_vertices.dtype() == at::kFloat, "triangle_vertices must be float32");
    TORCH_CHECK(triangle_vertices.dim() == 3 && triangle_vertices.size(1) == 3 && triangle_vertices.size(2) == 3, "triangle_vertices must have shape [N,3,3]");

    TORCH_CHECK(triangle_ids.dtype() == at::kLong, "triangle_ids must be int64");
    TORCH_CHECK(triangle_ids.dim() == 1 && triangle_ids.size(0) == triangle_vertices.size(0), "triangle_ids must have shape [N]");

    TORCH_CHECK(node_mins.dtype() == at::kFloat && node_mins.dim() == 2 && node_mins.size(1) == 3, "node_mins must have shape [M,3]");
    TORCH_CHECK(node_maxs.dtype() == at::kFloat && node_maxs.dim() == 2 && node_maxs.size(1) == 3, "node_maxs must have shape [M,3]");
    TORCH_CHECK(node_children.dtype() == at::kInt && node_children.dim() == 2 && node_children.size(1) == 3, "node_children must have shape [M,3]");
    TORCH_CHECK(node_mins.size(0) == node_maxs.size(0) && node_mins.size(0) == node_children.size(0), "node tensor shapes must match");

    triangle_vertices = triangle_vertices.contiguous();
    triangle_ids = triangle_ids.contiguous();
    node_mins = node_mins.contiguous();
    node_maxs = node_maxs.contiguous();
    node_children = node_children.contiguous();

    const int64_t n_triangles = triangle_vertices.size(0);
    std::vector<Triangle> triangles;
    triangles.resize(n_triangles);

    auto verts_acc = triangle_vertices.accessor<float, 3>();
    auto ids_acc = triangle_ids.accessor<int64_t, 1>();

    for (int64_t i = 0; i < n_triangles; ++i) {
        Triangle tri;
        tri.a = Eigen::Vector3f(verts_acc[i][0][0], verts_acc[i][0][1], verts_acc[i][0][2]);
        tri.b = Eigen::Vector3f(verts_acc[i][1][0], verts_acc[i][1][1], verts_acc[i][1][2]);
        tri.c = Eigen::Vector3f(verts_acc[i][2][0], verts_acc[i][2][1], verts_acc[i][2][2]);
        tri.id = ids_acc[i];
        triangles[i] = tri;
    }

    const int64_t n_nodes = node_mins.size(0);
    auto mins_acc = node_mins.accessor<float, 2>();
    auto maxs_acc = node_maxs.accessor<float, 2>();
    auto child_acc = node_children.accessor<int, 2>();

    std::vector<TriangleBvhNode> nodes;
    nodes.resize(n_nodes);

    for (int64_t i = 0; i < n_nodes; ++i) {
        TriangleBvhNode node;
        node.bb.min = Eigen::Vector3f(mins_acc[i][0], mins_acc[i][1], mins_acc[i][2]);
        node.bb.max = Eigen::Vector3f(maxs_acc[i][0], maxs_acc[i][1], maxs_acc[i][2]);
        node.left_idx = child_acc[i][0];
        node.right_idx = child_acc[i][1];
        node.escape_idx = child_acc[i][2];
        nodes[i] = node;
    }

    return new cuBVHImpl{std::move(triangles), std::move(nodes)};
}

at::Tensor floodfill(at::Tensor grid) {

    // assert grid is uint8_t
    assert(grid.dtype() == at::ScalarType::Bool);

    const int B = grid.size(0);
    const int H = grid.size(1);
    const int W = grid.size(2);
    const int D = grid.size(3);

    // allocate mask
    at::Tensor mask = at::zeros({B, H, W, D}, at::device(grid.device()).dtype(at::ScalarType::Int));

    _floodfill_batch(grid.data_ptr<bool>(), B, H, W, D, mask.data_ptr<int32_t>());

    return mask;
}

std::tuple<at::Tensor, at::Tensor> sparse_marching_cubes(
    at::Tensor coords,        // [N,3] int32, cuda
    at::Tensor corners,       // [N,8] float32, cuda
    double iso_d,             // (PyTorch passes double ⇒ cast to float)
    bool ensure_consistency)  // whether to ensure corner consistency
{
    TORCH_CHECK(coords.is_cuda(),  "coords must reside on CUDA");
    TORCH_CHECK(corners.is_cuda(), "corners must reside on CUDA");
    TORCH_CHECK(coords.dtype()  == at::kInt,   "coords must be int32");
    TORCH_CHECK(corners.dtype() == at::kFloat, "corners must be float32");
    TORCH_CHECK(coords.sizes().size()  == 2 && coords.size(1)  == 3,
                "coords must be of shape [N,3]");
    TORCH_CHECK(corners.sizes().size() == 2 && corners.size(1) == 8,
                "corners must be of shape [N,8]");
    TORCH_CHECK(coords.size(0) == corners.size(0),
                "coords and corners must have the same first-dim (N)");

    // Ensure contiguous memory - PyTorch extensions expect this.
    coords  = coords.contiguous();
    corners = corners.contiguous();
    const int    N   = static_cast<int>(coords.size(0));
    const int   *d_coords  = coords.data_ptr<int>();
    const float *d_corners = corners.data_ptr<float>();
    const float  iso       = static_cast<float>(iso_d);

    // Use the current PyTorch CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // --- call the CUDA sparse MC core (header we wrote earlier) -------------------
    auto mesh = _sparse_marching_cubes(d_coords, d_corners, N, iso, ensure_consistency, stream);
    thrust::device_vector<V3f> &verts_vec = mesh.first;
    thrust::device_vector<Tri> &tris_vec  = mesh.second;
    const int64_t M = static_cast<int64_t>(verts_vec.size());
    const int64_t T = static_cast<int64_t>(tris_vec.size());

    // --- create output tensors ----------------------------------------------------
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(coords.device());
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(coords.device());

    at::Tensor verts = at::empty({M, 3}, opts_f);
    at::Tensor tris  = at::empty({T, 3}, opts_i);

    // Copy GPU→GPU (same stream ⇒ async & cheap)
    cudaMemcpyAsync(verts.data_ptr<float>(),
                    thrust::raw_pointer_cast(verts_vec.data()),
                    M * 3 * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    cudaMemcpyAsync(tris.data_ptr<int>(),
                    thrust::raw_pointer_cast(tris_vec.data()),
                    T * 3 * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);

    // Make sure copies finish before we free device_vectors
    cudaStreamSynchronize(stream);

    return {verts, tris};
}

// ------------------------ GPU Hash Table bindings (virtual pattern) ----------

class cuHashTableImpl : public cuHashTable {
public:
    cuHashTableImpl() {}
    ~cuHashTableImpl() override {}

    void set_num_dims(int d) override {
        ht.set_num_dims(d);
    }

    int get_num_dims() const override {
        return ht.num_dims;
    }

    void resize(int capacity) override {
        ht.resize(capacity);
    }

    void prepare() override {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ht.prepare(stream);
    }

    void insert(at::Tensor coords) override {
        TORCH_CHECK(coords.is_cuda(),  "coords must reside on CUDA");
        TORCH_CHECK(coords.dtype()  == at::kInt,   "coords must be int32");
        TORCH_CHECK(coords.dim() == 2, "coords must be 2D [N,D]");
        coords = coords.contiguous();
        const int N = (int)coords.size(0);
        const int D = (int)coords.size(1);
        ht.set_num_dims(D);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ht.insert(coords.data_ptr<int>(), N, stream);
    }

    void build(at::Tensor coords) override {
        TORCH_CHECK(coords.is_cuda(),  "coords must reside on CUDA");
        TORCH_CHECK(coords.dtype()  == at::kInt,   "coords must be int32");
        TORCH_CHECK(coords.dim() == 2, "coords must be 2D [N,D]");
        coords = coords.contiguous();
        const int N = (int)coords.size(0);
        const int D = (int)coords.size(1);
        ht.set_num_dims(D);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ht.build(coords.data_ptr<int>(), N, stream);
    }

    at::Tensor search(at::Tensor queries) const override {
        TORCH_CHECK(queries.is_cuda(),  "queries must reside on CUDA");
        TORCH_CHECK(queries.dtype()  == at::kInt,   "queries must be int32");
        TORCH_CHECK(queries.dim() == 2, "queries must be 2D [M,D]");
        TORCH_CHECK(ht.capacity > 0, "hash table is not built");
        at::Tensor q = queries.contiguous();
        const int M = (int)q.size(0);
        auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
        at::Tensor out = at::empty({M}, opts_i);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ht.search(q.data_ptr<int>(), M, out.data_ptr<int>(), stream);
        return out;
    }

private:
    HashTableInt ht;
};

cuHashTable* create_cuHashTable() {
    return new cuHashTableImpl{};
}

} // namespace cubvh