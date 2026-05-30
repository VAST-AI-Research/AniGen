#include "cumesh.h"


namespace cumesh {


template <typename T>
struct TorchTypeMapping;

template <> struct TorchTypeMapping<int> {
    static constexpr auto scalar_type = torch::kInt32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 1;
};

template <> struct TorchTypeMapping<char> {
    static constexpr auto scalar_type = torch::kInt8;
    static constexpr int sizeof_scalar = 1;
    static constexpr int channels = 1;
};

template <> struct TorchTypeMapping<uint8_t> {
    static constexpr auto scalar_type = torch::kUInt8;
    static constexpr int sizeof_scalar = 1;
    static constexpr int channels = 1;
};

template <> struct TorchTypeMapping<float> {
    static constexpr auto scalar_type = torch::kFloat32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 1;
};

template <> struct TorchTypeMapping<uint64_t> {
    static constexpr auto scalar_type = torch::kInt32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 2;
};

template <> struct TorchTypeMapping<int2> {
    static constexpr auto scalar_type = torch::kInt32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 2;
};

template <> struct TorchTypeMapping<int3> {
    static constexpr auto scalar_type = torch::kInt32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 3;
};

template <> struct TorchTypeMapping<float2> {
    static constexpr auto scalar_type = torch::kFloat32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 2;
};

template <> struct TorchTypeMapping<float3> {
    static constexpr auto scalar_type = torch::kFloat32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 3;
};

template <> struct TorchTypeMapping<float4> {
    static constexpr auto scalar_type = torch::kFloat32;
    static constexpr int sizeof_scalar = 4;
    static constexpr int channels = 4;
};


template<typename T>
static torch::Tensor buffer_to_tensor(const Buffer<T> buffer) {
    using Mapping = TorchTypeMapping<T>;

    int64_t count = static_cast<int64_t>(buffer.size);
    std::vector<int64_t> shape;
    if (Mapping::channels == 1) {
        shape = {count}; // 1D Tensor
    } else {
        shape = {count, Mapping::channels}; // 2D Tensor [N, C]
    }

    auto options = torch::dtype(Mapping::scalar_type).device(torch::kCUDA);
    auto tensor = torch::empty(shape, options);

    static constexpr int dst_bytes = Mapping::sizeof_scalar * Mapping::channels;
    if (sizeof(T) == dst_bytes) {
        CUDA_CHECK(cudaMemcpy(
            tensor.data_ptr(),
            buffer.ptr,
            count * sizeof(T),
            cudaMemcpyDeviceToDevice
        ));
    } else {
        CUDA_CHECK(cudaMemcpy2D(
            tensor.data_ptr(),
            dst_bytes,
            buffer.ptr,
            sizeof(T),
            dst_bytes,
            count,
            cudaMemcpyDeviceToDevice
        ));
    }

    return tensor;
}


void CuMesh::init(const torch::Tensor& vertices, const torch::Tensor& faces) {
    size_t num_vertices = vertices.size(0);
    size_t num_faces = faces.size(0);
    this->vertices.resize(num_vertices);
    this->faces.resize(num_faces);
    CUDA_CHECK(cudaMemcpy2D(
        this->vertices.ptr,
        sizeof(float3),
        vertices.data_ptr<float>(),
        sizeof(float) * 3,
        sizeof(float) * 3,
        num_vertices,
        cudaMemcpyDeviceToDevice
    ));
    CUDA_CHECK(cudaMemcpy2D(
        this->faces.ptr,
        sizeof(int3),
        faces.data_ptr<int>(),
        sizeof(int) * 3,
        sizeof(int) * 3,
        num_faces,
        cudaMemcpyDeviceToDevice
    ));
}


std::tuple<torch::Tensor, torch::Tensor> CuMesh::read() {
    auto vertices = buffer_to_tensor(this->vertices);
    auto faces = buffer_to_tensor(this->faces);
    return std::make_tuple(vertices, faces);
}


torch::Tensor CuMesh::read_face_normals() {
    auto face_normals = buffer_to_tensor(this->face_normals);
    return face_normals;
}


torch::Tensor CuMesh::read_vertex_normals() {
    auto vertex_normals = buffer_to_tensor(this->vertex_normals);
    return vertex_normals;
}


torch::Tensor CuMesh::read_edges() {
    auto edges = buffer_to_tensor(this->edges);
    return edges;
}


torch::Tensor CuMesh::read_boundaries() {
    auto boundaries = buffer_to_tensor(this->boundaries);
    return boundaries;
}


torch::Tensor CuMesh::read_manifold_face_adjacency() {
    auto manifold_face_adj = buffer_to_tensor(this->manifold_face_adj);
    return manifold_face_adj;
}


torch::Tensor CuMesh::read_manifold_boundary_adjacency() {
    auto manifold_bound_adj = buffer_to_tensor(this->manifold_bound_adj);
    return manifold_bound_adj;
}

  
std::tuple<int, torch::Tensor> CuMesh::read_connected_components() {
    auto conn_comp_ids_tensor = buffer_to_tensor(this->conn_comp_ids);
    return std::make_tuple(this->num_conn_comps, conn_comp_ids_tensor);
}


std::tuple<int, torch::Tensor> CuMesh::read_boundary_connected_components() {
    auto bound_conn_comp_ids_tensor = buffer_to_tensor(this->bound_conn_comp_ids);
    return std::make_tuple(this->num_bound_conn_comps, bound_conn_comp_ids_tensor);
}


std::tuple<int, torch::Tensor, torch::Tensor> CuMesh::read_boundary_loops() {
    auto loop_boundaries_tensor = buffer_to_tensor(this->loop_boundaries);
    auto loop_boundaries_offset_tensor = buffer_to_tensor(this->loop_boundaries_offset);
    return std::make_tuple(this->num_bound_loops, loop_boundaries_tensor, loop_boundaries_offset_tensor);
}


std::unordered_map<std::string, torch::Tensor> CuMesh::read_all_cache() {
    std::unordered_map<std::string, torch::Tensor> cache;
    cache["face_areas"] = buffer_to_tensor(this->face_areas);
    cache["face_normals"] = buffer_to_tensor(this->face_normals);
    cache["vertex_normals"] = buffer_to_tensor(this->vertex_normals);
    cache["edges"] = buffer_to_tensor(this->edges);
    cache["boundaries"] = buffer_to_tensor(this->boundaries);
    cache["vert_is_boundary"] = buffer_to_tensor(this->vert_is_boundary);
    cache["vert_is_manifold"] = buffer_to_tensor(this->vert_is_manifold);
    cache["vert2edge"] = buffer_to_tensor(this->vert2edge);
    cache["vert2edge_cnt"] = buffer_to_tensor(this->vert2edge_cnt);
    cache["vert2edge_offset"] = buffer_to_tensor(this->vert2edge_offset);
    cache["vert2bound"] = buffer_to_tensor(this->vert2bound);
    cache["vert2bound_cnt"] = buffer_to_tensor(this->vert2bound_cnt);
    cache["vert2bound_offset"] = buffer_to_tensor(this->vert2bound_offset);
    cache["edge2face"] = buffer_to_tensor(this->edge2face);
    cache["edge2face_cnt"] = buffer_to_tensor(this->edge2face_cnt);
    cache["edge2face_offset"] = buffer_to_tensor(this->edge2face_offset);
    cache["face2edge"] = buffer_to_tensor(this->face2edge);
    cache["vert2face"] = buffer_to_tensor(this->vert2face);
    cache["vert2face_cnt"] = buffer_to_tensor(this->vert2face_cnt);
    cache["vert2face_offset"] = buffer_to_tensor(this->vert2face_offset);
    cache["manifold_face_adj"] = buffer_to_tensor(this->manifold_face_adj);
    cache["manifold_bound_adj"] = buffer_to_tensor(this->manifold_bound_adj);
    cache["conn_comp_ids"] = buffer_to_tensor(this->conn_comp_ids);
    cache["bound_conn_comp_ids"] = buffer_to_tensor(this->bound_conn_comp_ids);
    cache["loop_boundaries"] = buffer_to_tensor(this->loop_boundaries);
    cache["loop_boundaries_offset"] = buffer_to_tensor(this->loop_boundaries_offset);
    cache["vertices_map"] = buffer_to_tensor(this->vertices_map);
    cache["faces_map"] = buffer_to_tensor(this->faces_map);
    cache["edge_collapse_costs"] = buffer_to_tensor(this->edge_collapse_costs);
    cache["propagated_costs"] = buffer_to_tensor(this->propagated_costs);
    cache["atlas_chart_ids"] = buffer_to_tensor(this->atlas_chart_ids);
    cache["atlas_chart_vertex_map"] = buffer_to_tensor(this->atlas_chart_vertex_map);
    cache["atlas_chart_faces"] = buffer_to_tensor(this->atlas_chart_faces);
    cache["atlas_chart_faces_offset"] = buffer_to_tensor(this->atlas_chart_faces_offset);
    cache["atlas_chart_vertex_offset"] = buffer_to_tensor(this->atlas_chart_vertex_offset);
    cache["atlas_chart_uvs"] = buffer_to_tensor(this->atlas_chart_uvs);
    cache["atlas_chart_normal_cones"] = buffer_to_tensor(this->atlas_chart_normal_cones);
    cache["atlas_chart_adj"] = buffer_to_tensor(this->atlas_chart_adj);
    cache["atlas_chart_adj_length"] = buffer_to_tensor(this->atlas_chart_adj_length);
    cache["atlas_chart_perims"] = buffer_to_tensor(this->atlas_chart_perims);
    cache["atlas_chart_areas"] = buffer_to_tensor(this->atlas_chart_areas);
    cache["atlas_chart2edge"] = buffer_to_tensor(this->atlas_chart2edge);
    cache["atlas_chart2edge_cnt"] = buffer_to_tensor(this->atlas_chart2edge_cnt);
    cache["atlas_chart2edge_offset"] = buffer_to_tensor(this->atlas_chart2edge_offset);
    cache["temp_storage"] = buffer_to_tensor(this->temp_storage);
    cache["cub_temp_storage"] = buffer_to_tensor(this->cub_temp_storage);
    return cache;
}


} // namespace cumesh
