#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "artlib_cpp/cvi_metrics_core.hpp"

namespace py = pybind11;

namespace {

using XArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using LabelArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

double EvaluateCVI(XArray X, LabelArray labels, int validity) {
    auto xb = X.request();
    auto yb = labels.request();

    if (xb.ndim != 2) throw std::runtime_error("X must be 2-D");
    if (yb.ndim != 1) throw std::runtime_error("labels must be 1-D");
    if (static_cast<std::size_t>(yb.shape[0]) != static_cast<std::size_t>(xb.shape[0])) {
        throw std::runtime_error("labels length must match number of rows in X");
    }

    const auto rows = static_cast<std::size_t>(xb.shape[0]);
    const auto cols = static_cast<std::size_t>(xb.shape[1]);

    return artlib_cpp::evaluate_cvi(
        static_cast<const double*>(xb.ptr),
        rows,
        cols,
        static_cast<const int*>(yb.ptr),
        static_cast<artlib_cpp::CVIType>(validity)
    );
}

}  // namespace

PYBIND11_MODULE(cppCVIMetrics, m) {
    m.def("EvaluateCVI", &EvaluateCVI, py::arg("X"), py::arg("labels"), py::arg("validity"));
}
