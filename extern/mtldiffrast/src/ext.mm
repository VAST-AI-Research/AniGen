// PyBind11 bindings for mtldiffrast
// PyTorch extension bindings for mtldiffrast.
#import <torch/extension.h>
#import "metal_rasterize.h"
#import "metal_interpolate.h"
#import "metal_texture.h"
#import "metal_antialias.h"

// Topology hash state wrapper for antialias persistence across calls
struct TopologyHashWrapper {
    torch::Tensor ev_hash;
    int num_triangles;

    TopologyHashWrapper() : num_triangles(0) {}
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Rasterize context
    py::class_<mtldiffrast::MtlRasterizeContext>(m, "MtlRasterizeContext")
        .def(py::init<>());

    // Rasterize function
    m.def("rasterize_fwd", [](mtldiffrast::MtlRasterizeContext& ctx,
                               const torch::Tensor& pos,
                               const torch::Tensor& tri,
                               const std::vector<int>& resolution) {
        return ctx.rasterize(pos, tri, resolution);
    }, "Forward rasterization using Metal");

    // Rasterize backward
    m.def("rasterize_grad", [](mtldiffrast::MtlRasterizeContext& ctx,
                                const torch::Tensor& pos,
                                const torch::Tensor& tri,
                                const torch::Tensor& rast_out,
                                const torch::Tensor& dy,
                                const torch::Tensor& ddb) {
        return ctx.rasterize_grad(pos, tri, rast_out, dy, ddb);
    }, py::arg("ctx"), py::arg("pos"), py::arg("tri"),
       py::arg("rast_out"), py::arg("dy"), py::arg("ddb"),
       "Backward rasterization using Metal");

    // Interpolate function
    m.def("interpolate_fwd", [](const torch::Tensor& attr,
                                 const torch::Tensor& rast,
                                 const torch::Tensor& tri,
                                 const torch::Tensor& rast_db,
                                 bool enable_da) {
        return mtldiffrast::interpolate(attr, rast, tri, rast_db, enable_da);
    }, py::arg("attr"), py::arg("rast"), py::arg("tri"),
       py::arg("rast_db") = torch::Tensor(), py::arg("enable_da") = false,
       "Forward interpolation using Metal");

    // Interpolate backward (no DA)
    m.def("interpolate_grad", [](const torch::Tensor& attr,
                                  const torch::Tensor& rast,
                                  const torch::Tensor& tri,
                                  const torch::Tensor& dy) {
        return mtldiffrast::interpolate_grad(attr, rast, tri, dy);
    }, py::arg("attr"), py::arg("rast"), py::arg("tri"), py::arg("dy"),
       "Backward interpolation using Metal");

    // Interpolate backward (with DA)
    m.def("interpolate_grad_da", [](const torch::Tensor& attr,
                                     const torch::Tensor& rast,
                                     const torch::Tensor& tri,
                                     const torch::Tensor& dy,
                                     const torch::Tensor& rast_db,
                                     const torch::Tensor& dda) {
        return mtldiffrast::interpolate_grad_da(attr, rast, tri, dy, rast_db, dda);
    }, py::arg("attr"), py::arg("rast"), py::arg("tri"),
       py::arg("dy"), py::arg("rast_db"), py::arg("dda"),
       "Backward interpolation with derivatives using Metal");

    // Texture: construct mip chain
    m.def("texture_construct_mip", [](const torch::Tensor& tex,
                                       int max_mip_level,
                                       bool cube_mode) {
        return mtldiffrast::texture_construct_mip(tex, max_mip_level, cube_mode);
    }, py::arg("tex"), py::arg("max_mip_level"), py::arg("cube_mode") = false,
       "Build mipmap chain from base texture");

    // Texture: forward lookup (no mips)
    m.def("texture_fwd", [](const torch::Tensor& tex,
                             const torch::Tensor& uv,
                             int filter_mode,
                             int boundary_mode) {
        return mtldiffrast::texture_fwd(tex, uv, filter_mode, boundary_mode);
    }, py::arg("tex"), py::arg("uv"),
       py::arg("filter_mode") = 1, py::arg("boundary_mode") = 1,
       "Forward texture lookup using Metal");

    // Texture: forward lookup with mips
    m.def("texture_fwd_mip", [](const torch::Tensor& tex,
                                 const torch::Tensor& uv,
                                 const torch::Tensor& uv_da,
                                 const torch::Tensor& mip_level_bias,
                                 mtldiffrast::TextureMipWrapper& mip_wrapper,
                                 const std::vector<torch::Tensor>& mip_stack,
                                 int filter_mode,
                                 int boundary_mode) {
        return mtldiffrast::texture_fwd_mip(tex, uv, uv_da, mip_level_bias,
                                             mip_wrapper, mip_stack,
                                             filter_mode, boundary_mode);
    }, py::arg("tex"), py::arg("uv"), py::arg("uv_da"), py::arg("mip_level_bias"),
       py::arg("mip_wrapper"), py::arg("mip_stack"),
       py::arg("filter_mode"), py::arg("boundary_mode"),
       "Forward texture lookup with mipmap support");

    // TextureMipWrapper for mip state
    py::class_<mtldiffrast::TextureMipWrapper>(m, "TextureMipWrapper")
        .def(py::init<>())
        .def_readwrite("mip", &mtldiffrast::TextureMipWrapper::mip)
        .def_readwrite("max_mip_level", &mtldiffrast::TextureMipWrapper::max_mip_level)
        .def_readwrite("cube_mode", &mtldiffrast::TextureMipWrapper::cube_mode);

    // Antialias: topology hash wrapper
    py::class_<TopologyHashWrapper>(m, "TopologyHashWrapper")
        .def(py::init<>())
        .def_readwrite("ev_hash", &TopologyHashWrapper::ev_hash)
        .def_readwrite("num_triangles", &TopologyHashWrapper::num_triangles);

    // Antialias: construct topology hash
    m.def("antialias_construct_topology_hash", [](const torch::Tensor& tri,
                                                    TopologyHashWrapper& wrapper) {
        wrapper.ev_hash = mtldiffrast::antialias_construct_topology_hash(tri);
        wrapper.num_triangles = (int)tri.size(0);
    }, py::arg("tri"), py::arg("wrapper"),
       "Build edge-vertex topology hash for antialias");

    // Antialias: forward pass
    m.def("antialias_fwd", [](const torch::Tensor& color,
                               const torch::Tensor& rast,
                               const torch::Tensor& pos,
                               const torch::Tensor& tri,
                               const TopologyHashWrapper& wrapper) {
        return mtldiffrast::antialias_fwd(color, rast, pos, tri, wrapper.ev_hash);
    }, py::arg("color"), py::arg("rast"), py::arg("pos"),
       py::arg("tri"), py::arg("topology_hash"),
       "Forward antialias pass using Metal");

    // Antialias: backward pass
    m.def("antialias_grad", [](const torch::Tensor& color,
                                const torch::Tensor& rast,
                                const torch::Tensor& pos,
                                const torch::Tensor& tri,
                                const torch::Tensor& dy,
                                const torch::Tensor& work_buffer) {
        return mtldiffrast::antialias_grad(color, rast, pos, tri, dy, work_buffer);
    }, py::arg("color"), py::arg("rast"), py::arg("pos"),
       py::arg("tri"), py::arg("dy"), py::arg("work_buffer"),
       "Backward antialias pass using Metal");

    // Texture backward
    m.def("texture_grad", [](const torch::Tensor& tex,
                              const torch::Tensor& uv,
                              const torch::Tensor& dy,
                              const torch::Tensor& uv_da,
                              const torch::Tensor& mip_level_bias,
                              mtldiffrast::TextureMipWrapper& mip_wrapper,
                              const std::vector<torch::Tensor>& mip_stack,
                              int filter_mode,
                              int boundary_mode) {
        return mtldiffrast::texture_grad(tex, uv, dy, uv_da, mip_level_bias,
                                         mip_wrapper, mip_stack,
                                         filter_mode, boundary_mode);
    }, py::arg("tex"), py::arg("uv"), py::arg("dy"),
       py::arg("uv_da"), py::arg("mip_level_bias"),
       py::arg("mip_wrapper"), py::arg("mip_stack"),
       py::arg("filter_mode"), py::arg("boundary_mode"),
       "Backward texture lookup using Metal");

    // Texture mode constants
    m.attr("TEX_MODE_NEAREST") = py::int_(0);
    m.attr("TEX_MODE_LINEAR") = py::int_(1);
    m.attr("TEX_MODE_LINEAR_MIPMAP_NEAREST") = py::int_(2);
    m.attr("TEX_MODE_LINEAR_MIPMAP_LINEAR") = py::int_(3);
    m.attr("TEX_BOUNDARY_MODE_CUBE") = py::int_(0);
    m.attr("TEX_BOUNDARY_MODE_WRAP") = py::int_(1);
    m.attr("TEX_BOUNDARY_MODE_CLAMP") = py::int_(2);
    m.attr("TEX_BOUNDARY_MODE_ZERO") = py::int_(3);
}
