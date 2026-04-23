#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "artlib_cpp/binary_fuzzy_art_core.hpp"

namespace py = pybind11;

namespace {

using U8Array = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>;

U8Array require_int_or_bool_and_cast_u8(const py::handle& obj, const char* name) {
    py::array arr = py::array::ensure(obj);
    if (!arr) {
        throw std::runtime_error(std::string(name) + " must be a numpy array (or array-like).");
    }
    py::dtype dt = arr.dtype();
    std::string kind_s = py::str(dt.attr("kind"));
    const char kind = kind_s.empty() ? '\0' : kind_s[0];
    if (!(kind == 'b' || kind == 'i' || kind == 'u')) {
        throw std::runtime_error(std::string(name) + " must have bool or integer dtype.");
    }
    return U8Array(arr);
}

std::vector<std::vector<uint32_t>> parse_weights(const py::object& weights) {
    if (weights.is_none()) {
        return {};
    }
    if (!py::isinstance<py::list>(weights) && !py::isinstance<py::array>(weights)) {
        throw std::runtime_error("weights must be None, list, or ndarray");
    }

    std::vector<std::vector<uint32_t>> out;

    if (py::isinstance<py::array>(weights)) {
        auto arr = require_int_or_bool_and_cast_u8(weights, "weights");
        auto req = arr.request();
        if (req.ndim != 2) {
            throw std::runtime_error("weights ndarray must be 2-D");
        }
        const auto rows = static_cast<std::size_t>(req.shape[0]);
        const auto cols = static_cast<std::size_t>(req.shape[1]);
        const auto* ptr = static_cast<const std::uint8_t*>(req.ptr);
        out.reserve(rows);
        for (std::size_t r = 0; r < rows; ++r) {
            std::vector<uint32_t> w(cols, 0u);
            for (std::size_t c = 0; c < cols; ++c) {
                w[c] = (ptr[r * cols + c] != 0u) ? 1u : 0u;
            }
            out.push_back(std::move(w));
        }
        return out;
    }

    py::list w_list = weights.cast<py::list>();
    out.reserve(w_list.size());
    for (py::handle item : w_list) {
        auto w_u8 = require_int_or_bool_and_cast_u8(item, "weights[i]");
        auto req = w_u8.request();
        if (req.ndim != 1) {
            throw std::runtime_error("Each weight array must be 1D.");
        }
        const auto len = static_cast<std::size_t>(req.shape[0]);
        const auto* ptr = static_cast<const std::uint8_t*>(req.ptr);
        std::vector<uint32_t> w(len, 0u);
        for (std::size_t j = 0; j < len; ++j) {
            w[j] = (ptr[j] != 0u) ? 1u : 0u;
        }
        out.push_back(std::move(w));
    }
    return out;
}

std::vector<py::array_t<int>> to_py_weights(
    const std::vector<std::vector<uint32_t>>& weights
) {
    std::vector<py::array_t<int>> out;
    out.reserve(weights.size());
    for (const auto& w_vec : weights) {
        py::array_t<int> arr(w_vec.size());
        int* out_ptr = arr.mutable_data();
        for (std::size_t j = 0; j < w_vec.size(); ++j) {
            out_ptr[j] = static_cast<int>(w_vec[j]);
        }
        out.push_back(std::move(arr));
    }
    return out;
}

class cppBinaryFuzzyART {
public:
    cppBinaryFuzzyART(double rho, py::object weights = py::none())
        : rho_(rho), initial_weights_(parse_weights(weights)) {}

    std::tuple<py::array_t<int>, std::vector<py::array_t<int>>> fit(py::object X_obj) {
        auto x = require_int_or_bool_and_cast_u8(X_obj, "X");
        auto req = x.request();
        if (req.ndim != 2) {
            throw std::runtime_error("X must be a 2D array.");
        }
        const auto rows = static_cast<std::size_t>(req.shape[0]);
        const auto cols = static_cast<std::size_t>(req.shape[1]);
        const auto* ptr = static_cast<const std::uint8_t*>(req.ptr);

        artlib_cpp::BinaryFuzzyARTCore core(rho_);
        core.set_weights(initial_weights_);
        core.fit(ptr, rows, cols);

        const auto& labels = core.labels();
        py::array_t<int> labels_out(labels.size());
        std::memcpy(labels_out.mutable_data(), labels.data(), sizeof(int) * labels.size());
        return {labels_out, to_py_weights(core.weights())};
    }

    py::array_t<int> predict(py::object X_obj) {
        auto x = require_int_or_bool_and_cast_u8(X_obj, "X");
        auto req = x.request();
        if (req.ndim != 2) {
            throw std::runtime_error("X must be a 2D array.");
        }
        const auto rows = static_cast<std::size_t>(req.shape[0]);
        const auto cols = static_cast<std::size_t>(req.shape[1]);
        const auto* ptr = static_cast<const std::uint8_t*>(req.ptr);

        artlib_cpp::BinaryFuzzyARTCore core(rho_);
        core.set_weights(initial_weights_);
        auto labels = core.predict(ptr, rows, cols);

        py::array_t<int> pred(labels.size());
        std::memcpy(pred.mutable_data(), labels.data(), sizeof(int) * labels.size());
        return pred;
    }

private:
    double rho_;
    std::vector<std::vector<uint32_t>> initial_weights_;
};

std::tuple<py::array_t<int>, std::vector<py::array_t<int>>> FitBinaryFuzzyART(
    py::object X_obj, double rho, py::object weights = py::none()
) {
    cppBinaryFuzzyART model(rho, weights);
    return model.fit(X_obj);
}

py::array_t<int> PredictBinaryFuzzyART(
    py::object X_obj, double rho, py::object weights = py::none()
) {
    cppBinaryFuzzyART model(rho, weights);
    return model.predict(X_obj);
}

}  // namespace

PYBIND11_MODULE(cppBinaryFuzzyART, m) {
    py::class_<cppBinaryFuzzyART>(m, "cppBinaryFuzzyART")
        .def(py::init<double, py::object>(), py::arg("rho"), py::arg("weights") = py::none())
        .def("fit", &cppBinaryFuzzyART::fit, py::arg("X"))
        .def("predict", &cppBinaryFuzzyART::predict, py::arg("X"))
        .def("__repr__", [](const cppBinaryFuzzyART&) { return "<cppBinaryFuzzyART model>"; });

    m.def(
        "FitBinaryFuzzyART",
        &FitBinaryFuzzyART,
        py::arg("X"),
        py::arg("rho"),
        py::arg("weights") = py::none()
    );

    m.def(
        "PredictBinaryFuzzyART",
        &PredictBinaryFuzzyART,
        py::arg("X"),
        py::arg("rho"),
        py::arg("weights") = py::none()
    );
}
