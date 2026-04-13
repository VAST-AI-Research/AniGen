#pragma once

#include <Eigen/Dense>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include <memory>
#include <tuple>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace cubvh {

// abstract class of raytracer
class cuBVH {
public:
    cuBVH() {}
    virtual ~cuBVH() {}

    virtual void ray_trace(at::Tensor rays_o, at::Tensor rays_d, at::Tensor positions, at::Tensor face_id, at::Tensor depth) = 0;
    virtual void unsigned_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw) = 0;
    virtual void signed_distance(at::Tensor positions, at::Tensor distances, at::Tensor face_id, at::optional<at::Tensor> uvw, uint32_t mode) = 0;
    virtual std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> export_state() const = 0;
};

// function to create an implementation of cuBVH
cuBVH* create_cuBVH(Ref<const Verts> vertices, Ref<const Trigs> triangles);
cuBVH* create_cuBVH_from_state(at::Tensor triangle_vertices, at::Tensor triangle_ids, at::Tensor node_mins, at::Tensor node_maxs, at::Tensor node_children);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> build_cuBVH_state(Ref<const Verts> vertices, Ref<const Trigs> triangles);
// floodfill
at::Tensor floodfill(at::Tensor grid);

// sparse marching cubes
std::tuple<at::Tensor, at::Tensor> sparse_marching_cubes(at::Tensor coords, at::Tensor corners, double iso_d, bool ensure_consistency = false);

// GPU ND integer hash table (default D=3) - abstract interface
class cuHashTable {
public:
    cuHashTable() {}
    virtual ~cuHashTable() {}
    virtual void set_num_dims(int d) = 0;
    virtual int get_num_dims() const = 0;
    virtual void resize(int capacity) = 0;
    virtual void prepare() = 0;
    virtual void insert(at::Tensor coords) = 0;
    virtual void build(at::Tensor coords) = 0;
    virtual at::Tensor search(at::Tensor queries) const = 0;
};

// Factory to create an implementation of cuHashTable
cuHashTable* create_cuHashTable();

} // namespace cubvh