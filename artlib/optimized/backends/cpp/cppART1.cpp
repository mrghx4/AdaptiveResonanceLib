#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "artlib_cpp/art1_core.hpp"

namespace py = pybind11;

namespace {

using MatrixArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

artlib_cpp::MatrixView to_matrix_view(const MatrixArray& x) {
    const auto req = x.request();
    if (req.ndim != 2) {
        throw std::runtime_error("X must be 2-D [n_samples, n_features]");
    }
    return artlib_cpp::MatrixView{
        static_cast<const double*>(req.ptr),
        static_cast<std::size_t>(req.shape[0]),
        static_cast<std::size_t>(req.shape[1]),
    };
}

std::vector<std::vector<double>> parse_weights(const py::object& weights) {
    if (weights.is_none()) {
        return {};
    }

    std::vector<std::vector<double>> out;

    if (py::isinstance<py::array>(weights)) {
        const MatrixArray arr = weights.cast<MatrixArray>();
        const auto req = arr.request();
        if (req.ndim != 2) {
            throw std::runtime_error("weights ndarray must be 2-D");
        }
        const auto rows = static_cast<std::size_t>(req.shape[0]);
        const auto cols = static_cast<std::size_t>(req.shape[1]);
        const auto* ptr = static_cast<const double*>(req.ptr);
        out.reserve(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            out.emplace_back(ptr + r * cols, ptr + (r + 1) * cols);
        }
        return out;
    }

    if (!py::isinstance<py::list>(weights)) {
        throw std::runtime_error("weights must be None, list, or ndarray");
    }

    py::list w_list = weights.cast<py::list>();
    out.reserve(w_list.size());
    for (py::handle item : w_list) {
        py::array_t<double, py::array::c_style | py::array::forcecast> w_arr =
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(item);
        const auto w_req = w_arr.request();
        if (w_req.ndim != 1) {
            throw std::runtime_error("each weight must be 1-D");
        }
        const auto* ptr = static_cast<const double*>(w_req.ptr);
        out.emplace_back(ptr, ptr + static_cast<std::size_t>(w_req.shape[0]));
    }
    return out;
}

std::vector<py::array_t<double>> to_py_weights(
    const std::vector<std::vector<double>>& weights
) {
    std::vector<py::array_t<double>> out;
    out.reserve(weights.size());
    for (const auto& w_vec : weights) {
        py::array_t<double> w_arr(w_vec.size());
        std::memcpy(w_arr.mutable_data(), w_vec.data(), sizeof(double) * w_vec.size());
        out.push_back(std::move(w_arr));
    }
    return out;
}

class cppART1 {
public:
    cppART1(double rho, double L, py::object weights = py::none()) : params_{rho, L} {
        initial_weights_ = parse_weights(weights);
    }

    std::tuple<py::array_t<int>, std::vector<py::array_t<double>>> fit(const MatrixArray& x) {
        artlib_cpp::ART1Core core(params_);
        core.set_weights(initial_weights_);
        core.fit(to_matrix_view(x));

        const auto& labels = core.labels();
        py::array_t<int> labels_out(labels.size());
        std::memcpy(labels_out.mutable_data(), labels.data(), sizeof(int) * labels.size());
        return {labels_out, to_py_weights(core.weights())};
    }

    py::array_t<int> predict(const MatrixArray& x) {
        artlib_cpp::ART1Core core(params_);
        core.set_weights(initial_weights_);
        auto labels = core.predict(to_matrix_view(x));

        py::array_t<int> out(labels.size());
        std::memcpy(out.mutable_data(), labels.data(), sizeof(int) * labels.size());
        return out;
    }

private:
    artlib_cpp::ART1Params params_;
    std::vector<std::vector<double>> initial_weights_;
};

auto FitART1(
    const MatrixArray& x,
    double rho,
    double L,
    py::object weights = py::none()
) {
    cppART1 model(rho, L, weights);
    return model.fit(x);
}

auto PredictART1(
    const MatrixArray& x,
    double rho,
    double L,
    py::object weights = py::none()
) {
    cppART1 model(rho, L, weights);
    return model.predict(x);
}

}  // namespace

PYBIND11_MODULE(cppART1, m) {
    py::class_<cppART1>(m, "cppART1")
        .def(
            py::init<double, double, py::object>(),
            py::arg("rho"),
            py::arg("L"),
            py::arg("weights") = py::none()
        )
        .def("fit", &cppART1::fit, py::arg("X"))
        .def("predict", &cppART1::predict, py::arg("X"))
        .def("__repr__", [](const cppART1&) { return "<cppART1>"; });

    m.def(
        "FitART1",
        &FitART1,
        py::arg("X"),
        py::arg("rho"),
        py::arg("L"),
        py::arg("weights") = py::none()
    );

    m.def(
        "PredictART1",
        &PredictART1,
        py::arg("X"),
        py::arg("rho"),
        py::arg("L"),
        py::arg("weights") = py::none()
    );
}
