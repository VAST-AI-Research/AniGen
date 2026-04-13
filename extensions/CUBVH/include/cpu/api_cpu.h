#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <cpu/fill_holes.h>
#include <cpu/merge_vertices.h>
// CPU sparse marching cubes
#include <cpu/hashtable.h>
#include <cpu/spmc.h>
// CPU mesh decimator
#include <cpu/decimation.h>

#include <vector>
#include <cstring>

namespace py = pybind11;

namespace cubvh {

static py::array_t<int> fill_holes(
    py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<int,   py::array::c_style | py::array::forcecast> faces,
    bool return_added,
    bool check_containment,
    double eps_d,
    bool verbose) {
    // Validate shapes
    auto vbuf = vertices.request();
    auto fbuf = faces.request();
    if (!(vbuf.ndim == 2 && vbuf.shape[1] == 3)) {
        throw std::runtime_error("vertices must be of shape [N,3]");
    }
    if (!(fbuf.ndim == 2 && fbuf.shape[1] == 3)) {
        throw std::runtime_error("faces must be of shape [M,3]");
    }

    const size_t N = static_cast<size_t>(vbuf.shape[0]);
    const size_t M = static_cast<size_t>(fbuf.shape[0]);

    const float* vptr = static_cast<const float*>(vbuf.ptr);
    const int*   fptr = static_cast<const int*>(fbuf.ptr);

    std::vector<Eigen::Vector3f> V(N);
    for (size_t i = 0; i < N; ++i) {
        V[i] = Eigen::Vector3f(vptr[3*i+0], vptr[3*i+1], vptr[3*i+2]);
    }

    std::vector<Eigen::Vector3i> F(M);
    for (size_t i = 0; i < M; ++i) {
        F[i] = Eigen::Vector3i(fptr[3*i+0], fptr[3*i+1], fptr[3*i+2]);
    }

    cubvh::cpu::HoleFillOptions opt;
    opt.checkContainment = check_containment;
    opt.eps = static_cast<float>(eps_d);
    opt.verbose = verbose;

    if (return_added) {
        auto added = cubvh::cpu::fill_holes(V, F, opt);
        py::array_t<int> out({(py::ssize_t)added.size(), (py::ssize_t)3});
        auto obuf = out.request();
        int* optr = static_cast<int*>(obuf.ptr);
        for (size_t i = 0; i < added.size(); ++i) {
            optr[3*i+0] = added[i][0];
            optr[3*i+1] = added[i][1];
            optr[3*i+2] = added[i][2];
        }
        return out;
    } else {
        cubvh::cpu::fill_holes_inplace(V, F, opt);
        py::array_t<int> out({(py::ssize_t)F.size(), (py::ssize_t)3});
        auto obuf = out.request();
        int* optr = static_cast<int*>(obuf.ptr);
        for (size_t i = 0; i < F.size(); ++i) {
            optr[3*i+0] = F[i][0];
            optr[3*i+1] = F[i][1];
            optr[3*i+2] = F[i][2];
        }
        return out;
    }
}

// merge_vertices binding: returns (vertices, faces) after merge
static std::pair<py::array_t<float>, py::array_t<int>> merge_vertices(
    py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<int,   py::array::c_style | py::array::forcecast> faces,
    double threshold_d) {
    auto vbuf = vertices.request();
    auto fbuf = faces.request();
    if (!(vbuf.ndim == 2 && vbuf.shape[1] == 3)) {
        throw std::runtime_error("vertices must be of shape [N,3]");
    }
    if (!(fbuf.ndim == 2 && fbuf.shape[1] == 3)) {
        throw std::runtime_error("faces must be of shape [M,3]");
    }
    const size_t N = static_cast<size_t>(vbuf.shape[0]);
    const size_t M = static_cast<size_t>(fbuf.shape[0]);
    const float* vptr = static_cast<const float*>(vbuf.ptr);
    const int* fptr = static_cast<const int*>(fbuf.ptr);

    std::vector<Eigen::Vector3f> V(N);
    for (size_t i=0;i<N;++i) {
        V[i] = Eigen::Vector3f(vptr[3*i+0], vptr[3*i+1], vptr[3*i+2]);
    }
    std::vector<Eigen::Vector3i> F(M);
    for (size_t i=0;i<M;++i) {
        F[i] = Eigen::Vector3i(fptr[3*i+0], fptr[3*i+1], fptr[3*i+2]);
    }

    std::vector<Eigen::Vector3f> V_out; std::vector<Eigen::Vector3i> F_out;
    cubvh::cpu::merge_vertices(V, F, static_cast<float>(threshold_d), V_out, F_out);

    py::array_t<float> v_out({(py::ssize_t)V_out.size(), (py::ssize_t)3});
    py::array_t<int> f_out({(py::ssize_t)F_out.size(), (py::ssize_t)3});
    auto vObuf = v_out.request(); auto fObuf = f_out.request();
    float* vO = static_cast<float*>(vObuf.ptr);
    int* fO = static_cast<int*>(fObuf.ptr);
    for (size_t i=0;i<V_out.size();++i) {
        vO[3*i+0] = V_out[i][0]; vO[3*i+1] = V_out[i][1]; vO[3*i+2] = V_out[i][2];
    }
    for (size_t i=0;i<F_out.size();++i) {
        fO[3*i+0] = F_out[i][0]; fO[3*i+1] = F_out[i][1]; fO[3*i+2] = F_out[i][2];
    }
    return {v_out, f_out};
}

// sparse marching cubes (CPU): returns (vertices [M,3] float32, faces [T,3] int32)
static std::pair<py::array_t<float>, py::array_t<int>> sparse_marching_cubes_cpu(
    py::array_t<int,   py::array::c_style | py::array::forcecast> coords,
    py::array_t<float, py::array::c_style | py::array::forcecast> corners,
    double iso_d,
    bool ensure_consistency = false) {

    auto cbuf = coords.request();
    auto vbuf = corners.request();
    if (!(cbuf.ndim == 2 && cbuf.shape[1] == 3)) {
        throw std::runtime_error("coords must be of shape [N,3] (int32)");
    }
    if (!(vbuf.ndim == 2 && vbuf.shape[1] == 8)) {
        throw std::runtime_error("corners must be of shape [N,8] (float32)");
    }
    if (cbuf.shape[0] != vbuf.shape[0]) {
        throw std::runtime_error("coords and corners must have the same first dimension N");
    }

    const int N = static_cast<int>(cbuf.shape[0]);
    const int*   cptr = static_cast<const int*>(cbuf.ptr);
    const float* fptr = static_cast<const float*>(vbuf.ptr);
    const float iso = static_cast<float>(iso_d);

    auto mesh = cubvh::cpu::sparse_marching_cubes(cptr, fptr, N, iso, ensure_consistency);
    const auto& V = mesh.first;
    const auto& F = mesh.second;

    // Allocate outputs
    py::array_t<float> v_out({(py::ssize_t)V.size(), (py::ssize_t)3});
    py::array_t<int>   f_out({(py::ssize_t)F.size(), (py::ssize_t)3});
    auto vObuf = v_out.request();
    auto fObuf = f_out.request();
    float* vO = static_cast<float*>(vObuf.ptr);
    int*   fO = static_cast<int*>(fObuf.ptr);

    for (size_t i = 0; i < V.size(); ++i) {
        vO[3*i+0] = V[i].x;
        vO[3*i+1] = V[i].y;
        vO[3*i+2] = V[i].z;
    }
    for (size_t i = 0; i < F.size(); ++i) {
        fO[3*i+0] = F[i].v0;
        fO[3*i+1] = F[i].v1;
        fO[3*i+2] = F[i].v2;
    }

    return {v_out, f_out};
}

// CPU decimator bindings
template <typename T>
static cubvh::cpu::qd::MeshT<T> _mesh_from_numpy_typed(
    py::array_t<T, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<int, py::array::c_style | py::array::forcecast> faces)
{
    if (vertices.ndim() != 2 || vertices.shape(1) != 3)
        throw std::runtime_error("vertices must be (N,3) array");
    if (faces.ndim() != 2 || faces.shape(1) != 3)
        throw std::runtime_error("faces must be (M,3) int32 array");

    cubvh::cpu::qd::MeshT<T> m;
    m.vertices.reserve(vertices.shape(0));
    auto vbuf = vertices.template unchecked<2>();
    for (ssize_t i = 0; i < vertices.shape(0); ++i) {
        m.vertices.emplace_back(vbuf(i,0), vbuf(i,1), vbuf(i,2));
    }
    m.faces.reserve(faces.shape(0));
    auto fbuf = faces.template unchecked<2>();
    for (ssize_t i = 0; i < faces.shape(0); ++i) {
        m.faces.push_back({ fbuf(i,0), fbuf(i,1), fbuf(i,2) });
    }
    return m;
}

template <typename T>
static std::pair<py::array_t<T>, py::array_t<int>> _mesh_to_numpy_typed(const cubvh::cpu::qd::MeshT<T>& m)
{
    py::array_t<T> V({ (ssize_t)m.vertices.size(), (ssize_t)3 });
    py::array_t<int> F({ (ssize_t)m.faces.size(), (ssize_t)3 });
    auto vbuf = V.template mutable_unchecked<2>();
    for (ssize_t i = 0; i < (ssize_t)m.vertices.size(); ++i) {
        vbuf(i,0) = m.vertices[i].x;
        vbuf(i,1) = m.vertices[i].y;
        vbuf(i,2) = m.vertices[i].z;
    }
    auto fbuf = F.template mutable_unchecked<2>();
    for (ssize_t i = 0; i < (ssize_t)m.faces.size(); ++i) {
        fbuf(i,0) = m.faces[i][0];
        fbuf(i,1) = m.faces[i][1];
        fbuf(i,2) = m.faces[i][2];
    }
    return { V, F };
}

static py::tuple decimate(py::array vertices,
                          py::array faces,
                          int target_vertices)
{
    py::dtype dt = vertices.dtype();
    if (dt.is(py::dtype::of<float>())){
        auto v = vertices.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        auto f = faces.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        auto mesh = _mesh_from_numpy_typed<float>(v, f);
        cubvh::cpu::qd::DecimatorT<float> dec(mesh);
        dec.decimate(target_vertices);
        auto out = _mesh_to_numpy_typed<float>(dec.mesh());
        return py::make_tuple(out.first, out.second);
    } else if (dt.is(py::dtype::of<double>())){
        auto v = vertices.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto f = faces.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        auto mesh = _mesh_from_numpy_typed<double>(v, f);
        cubvh::cpu::qd::DecimatorT<double> dec(mesh);
        dec.decimate(target_vertices);
        auto out = _mesh_to_numpy_typed<double>(dec.mesh());
        return py::make_tuple(out.first, out.second);
    } else {
        throw std::runtime_error("vertices must be float32 or float64 array");
    }
}

static py::tuple parallel_decimate(py::array vertices,
                                   py::array faces,
                                   int target_vertices)
{
    py::dtype dt = vertices.dtype();
    if (dt.is(py::dtype::of<float>())){
        auto v = vertices.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        auto f = faces.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        auto mesh = _mesh_from_numpy_typed<float>(v, f);
        cubvh::cpu::qd::DecimatorT<float> dec(mesh);
        dec.parallelDecimate(target_vertices);
        auto out = _mesh_to_numpy_typed<float>(dec.mesh());
        return py::make_tuple(out.first, out.second);
    } else if (dt.is(py::dtype::of<double>())){
        auto v = vertices.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto f = faces.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
        auto mesh = _mesh_from_numpy_typed<double>(v, f);
        cubvh::cpu::qd::DecimatorT<double> dec(mesh);
        dec.parallelDecimate(target_vertices);
        auto out = _mesh_to_numpy_typed<double>(dec.mesh());
        return py::make_tuple(out.first, out.second);
    } else {
        throw std::runtime_error("vertices must be float32 or float64 array");
    }
}

// CPU Hash Table bindings
class HashTable {
public:
    HashTable() {}

    void set_num_dims(int d) { ht.set_num_dims(d); }
    int get_num_dims() const { return ht.get_num_dims(); }
    void resize(int capacity) { ht.resize(capacity); }
    void prepare() { ht.prepare(); }

    void insert(at::Tensor coords) {
        TORCH_CHECK(!coords.is_cuda(),  "coords must reside on CPU");
        TORCH_CHECK(coords.dtype() == at::kInt, "coords must be int32");
        TORCH_CHECK(coords.dim() == 2, "coords must be 2D [N,D]");
        coords_ref_ = coords.contiguous();
        const int N = (int)coords_ref_.size(0);
        const int D = (int)coords_ref_.size(1);
        ht.set_num_dims(D);
        ht.insert(coords_ref_.data_ptr<int>(), N);
    }

    void build(at::Tensor coords) {
        TORCH_CHECK(!coords.is_cuda(),  "coords must reside on CPU");
        TORCH_CHECK(coords.dtype() == at::kInt, "coords must be int32");
        TORCH_CHECK(coords.dim() == 2, "coords must be 2D [N,D]");
        coords_ref_ = coords.contiguous();
        const int N = (int)coords_ref_.size(0);
        const int D = (int)coords_ref_.size(1);
        ht.set_num_dims(D);
        ht.build(coords_ref_.data_ptr<int>(), N);
    }

    at::Tensor search(at::Tensor queries) const {
        TORCH_CHECK(!queries.is_cuda(),  "queries must reside on CPU");
        TORCH_CHECK(queries.dtype() == at::kInt, "queries must be int32");
        TORCH_CHECK(queries.dim() == 2, "queries must be 2D [M,D]");
        TORCH_CHECK(ht.capacity > 0, "hash table is not built");
        at::Tensor q = queries.contiguous();
        const int M = (int)q.size(0);
        auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(q.device());
        at::Tensor out = at::empty({M}, opts_i);
        ht.search(q.data_ptr<int>(), M, out.data_ptr<int>());
        return out;
    }

private:
    mutable HashTableIntCPU ht;
    at::Tensor coords_ref_;
};

} // namespace cubvh