#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "artlib_cpp/bartmap_metrics_core.hpp"

namespace py = pybind11;

namespace {

using XArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using LabelArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

double AveragePearsonCorr(
    XArray X,
    std::size_t k,
    int c_b,
    LabelArray column_labels
) {
    auto xb = X.request();
    auto lb = column_labels.request();

    if (xb.ndim != 2) throw std::runtime_error("X must be 2-D");
    if (lb.ndim != 1) throw std::runtime_error("column_labels must be 1-D");

    return artlib_cpp::average_pearson_corr(
        static_cast<const double*>(xb.ptr),
        static_cast<std::size_t>(xb.shape[0]),
        static_cast<std::size_t>(xb.shape[1]),
        static_cast<const int*>(lb.ptr),
        static_cast<std::size_t>(lb.shape[0]),
        k,
        c_b
    );
}

bool AnyClusterMatch(
    XArray X,
    std::size_t k,
    std::size_t n_clusters_b,
    double eta,
    LabelArray column_labels
) {
    auto xb = X.request();
    auto lb = column_labels.request();

    if (xb.ndim != 2) throw std::runtime_error("X must be 2-D");
    if (lb.ndim != 1) throw std::runtime_error("column_labels must be 1-D");

    return artlib_cpp::any_cluster_match(
        static_cast<const double*>(xb.ptr),
        static_cast<std::size_t>(xb.shape[0]),
        static_cast<std::size_t>(xb.shape[1]),
        static_cast<const int*>(lb.ptr),
        static_cast<std::size_t>(lb.shape[0]),
        k,
        n_clusters_b,
        eta
    );
}

}  // namespace

PYBIND11_MODULE(cppBARTMAPMetrics, m) {
    m.def(
        "AveragePearsonCorr",
        &AveragePearsonCorr,
        py::arg("X"),
        py::arg("k"),
        py::arg("c_b"),
        py::arg("column_labels")
    );
    m.def(
        "AnyClusterMatch",
        &AnyClusterMatch,
        py::arg("X"),
        py::arg("k"),
        py::arg("n_clusters_b"),
        py::arg("eta"),
        py::arg("column_labels")
    );
}
