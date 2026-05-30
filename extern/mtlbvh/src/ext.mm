// PyBind11 bindings for mtlbvh
#import <torch/extension.h>
#import "metal_bvh.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<MtlBVHImpl>(m, "MtlBVH")
        .def(py::init<const torch::Tensor&, const torch::Tensor&>(),
             py::arg("vertices"), py::arg("triangles"))
        .def("unsigned_distance", &MtlBVHImpl::unsigned_distance,
             py::arg("positions"), py::arg("return_uvw") = false,
             py::arg("out_dtype") = 0)
        .def("signed_distance", &MtlBVHImpl::signed_distance,
             py::arg("positions"), py::arg("return_uvw") = false,
             py::arg("mode") = 0, py::arg("out_dtype") = 0)
        .def("ray_trace", &MtlBVHImpl::ray_trace,
             py::arg("rays_o"), py::arg("rays_d"),
             py::arg("out_dtype") = 0);
}
