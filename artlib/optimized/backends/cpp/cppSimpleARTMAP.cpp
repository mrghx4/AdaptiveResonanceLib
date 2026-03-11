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

py::array_t<double> GatherClusterCenters(IArray labels, py::array_t<double> centers) {
    auto lb = labels.request();
    auto cb = centers.request();
    if (lb.ndim != 1) throw std::runtime_error("labels must be 1-D");
    if (cb.ndim != 2) throw std::runtime_error("centers must be 2-D");

    const auto* lb_ptr = static_cast<const int*>(lb.ptr);
    const auto* cb_ptr = static_cast<const double*>(cb.ptr);
    const std::size_t n_labels = static_cast<std::size_t>(lb.shape[0]);
    const std::size_t n_centers = static_cast<std::size_t>(cb.shape[0]);
    const std::size_t center_dim = static_cast<std::size_t>(cb.shape[1]);

    std::vector<double> out = artlib_cpp::GatherClusterCenters(
        lb_ptr, n_labels, cb_ptr, n_centers, center_dim
    );

    py::array_t<double> arr({lb.shape[0], cb.shape[1]});
    std::memcpy(arr.mutable_data(), out.data(), out.size() * sizeof(double));
    return arr;
}

py::array_t<int> MapSimpleARTMAPLabelsChain(IArray labels, py::list map_chain) {
    auto lb = labels.request();
    if (lb.ndim != 1) throw std::runtime_error("labels must be 1-D");
    const auto* lb_ptr = static_cast<const int*>(lb.ptr);
    const std::size_t n_labels = static_cast<std::size_t>(lb.shape[0]);

    std::vector<std::vector<int>> chain;
    chain.reserve(static_cast<std::size_t>(py::len(map_chain)));
    for (py::handle item : map_chain) {
        IArray map_arr = py::cast<IArray>(item);
        auto mb = map_arr.request();
        if (mb.ndim != 1) throw std::runtime_error("each map entry must be 1-D");
        const auto* mp = static_cast<const int*>(mb.ptr);
        const std::size_t n = static_cast<std::size_t>(mb.shape[0]);
        chain.emplace_back(mp, mp + n);
    }

    std::vector<int> out = artlib_cpp::MapSimpleARTMAPLabelsChain(lb_ptr, n_labels, chain);
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
    m.def(
        "GatherClusterCenters",
        &GatherClusterCenters,
        py::arg("labels"),
        py::arg("centers")
    );
    m.def(
        "MapSimpleARTMAPLabelsChain",
        &MapSimpleARTMAPLabelsChain,
        py::arg("labels"),
        py::arg("map_chain")
    );
}
