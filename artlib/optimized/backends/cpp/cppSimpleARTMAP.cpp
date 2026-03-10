#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <vector>

#include "artlib_cpp/simple_artmap_core.hpp"

namespace py = pybind11;

namespace {

using IArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

py::array_t<int> MapSimpleARTMAPLabels(IArray labels_a, IArray cluster_labels) {
    auto la = labels_a.request();
    auto cl = cluster_labels.request();
    if (la.ndim != 1) throw std::runtime_error("labels_a must be 1-D");
    if (cl.ndim != 1) throw std::runtime_error("cluster_labels must be 1-D");

    const auto* la_ptr = static_cast<const int*>(la.ptr);
    const auto* cl_ptr = static_cast<const int*>(cl.ptr);
    std::vector<int> out = artlib_cpp::MapSimpleARTMAPLabels(
        la_ptr,
        static_cast<std::size_t>(la.shape[0]),
        cl_ptr,
        static_cast<std::size_t>(cl.shape[0])
    );

    py::array_t<int> arr(out.size());
    std::memcpy(arr.mutable_data(), out.data(), out.size() * sizeof(int));
    return arr;
}

}  // namespace

PYBIND11_MODULE(cppSimpleARTMAP, m) {
    m.def(
        "MapSimpleARTMAPLabels",
        &MapSimpleARTMAPLabels,
        py::arg("labels_a"),
        py::arg("cluster_labels")
    );
}
