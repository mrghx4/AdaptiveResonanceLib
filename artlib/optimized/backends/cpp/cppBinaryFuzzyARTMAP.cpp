#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "artlib_cpp/binary_fuzzy_artmap_core.hpp"

namespace py = pybind11;

namespace {

using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

std::vector<std::vector<uint32_t>> parse_weights(const py::object& weights) {
    if (weights.is_none()) return {};

    if (py::isinstance<py::array>(weights)) {
        IntArray arr = weights.cast<IntArray>();
        auto req = arr.request();
        if (req.ndim != 2) throw std::runtime_error("weights ndarray must be 2-D");
        const auto rows = static_cast<std::size_t>(req.shape[0]);
        const auto cols = static_cast<std::size_t>(req.shape[1]);
        const int* ptr = static_cast<const int*>(req.ptr);
        std::vector<std::vector<uint32_t>> out(rows, std::vector<uint32_t>(cols, 0u));
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::size_t c = 0; c < cols; ++c) {
                out[r][c] = (ptr[r * cols + c] != 0) ? 1u : 0u;
            }
        }
        return out;
    }

    if (!py::isinstance<py::list>(weights)) {
        throw std::runtime_error("weights must be None, list, or ndarray");
    }

    py::list w_list = weights.cast<py::list>();
    std::vector<std::vector<uint32_t>> out;
    out.reserve(w_list.size());
    for (py::handle item : w_list) {
        IntArray w_arr = py::cast<IntArray>(item);
        auto req = w_arr.request();
        if (req.ndim != 1) throw std::runtime_error("Each weight array must be 1D.");
        const auto len = static_cast<std::size_t>(req.shape[0]);
        const int* ptr = static_cast<const int*>(req.ptr);
        std::vector<uint32_t> w(len, 0u);
        for (std::size_t j = 0; j < len; ++j) w[j] = (ptr[j] != 0) ? 1u : 0u;
        out.push_back(std::move(w));
    }
    return out;
}

std::vector<int> parse_cluster_labels(const py::object& cluster_labels) {
    if (cluster_labels.is_none()) return {};
    IntArray c_arr = cluster_labels.cast<IntArray>();
    auto req = c_arr.request();
    if (req.ndim != 1) throw std::runtime_error("cluster_labels must be a 1D array.");
    const int* ptr = static_cast<const int*>(req.ptr);
    return {ptr, ptr + static_cast<std::size_t>(req.shape[0])};
}

std::vector<py::array_t<int>> to_py_weights(
    const std::vector<std::vector<uint32_t>>& weights
) {
    std::vector<py::array_t<int>> out;
    out.reserve(weights.size());
    for (const auto& w_vec : weights) {
        py::array_t<int> arr(w_vec.size());
        int* p = arr.mutable_data();
        for (std::size_t j = 0; j < w_vec.size(); ++j) p[j] = static_cast<int>(w_vec[j]);
        out.push_back(std::move(arr));
    }
    return out;
}

class cppBinaryFuzzyARTMAP {
public:
    cppBinaryFuzzyARTMAP(
        double rho,
        std::string MT,
        uint32_t epsilon,
        py::object weights = py::none(),
        py::object cluster_labels = py::none()
    ) : params_{rho, std::move(MT), epsilon} {
        const bool have_w = !weights.is_none();
        const bool have_cl = !cluster_labels.is_none();
        if (have_w != have_cl) {
            throw std::invalid_argument(
                "You must provide BOTH 'weights' and 'cluster_labels' OR neither."
            );
        }
        if (have_w) {
            initial_weights_ = parse_weights(weights);
            initial_cluster_labels_ = parse_cluster_labels(cluster_labels);
        }
    }

    std::tuple<py::array_t<int>, std::vector<py::array_t<int>>, py::array_t<int>>
    fit(IntArray X, IntArray y) {
        auto xb = X.request();
        auto yb = y.request();
        if (xb.ndim != 2) throw std::runtime_error("X must be a 2D array.");
        if (yb.ndim != 1) throw std::runtime_error("y must be a 1D array.");

        artlib_cpp::BinaryFuzzyARTMAPCore core(params_);
        core.set_state(initial_weights_, initial_cluster_labels_);
        core.fit(
            static_cast<const int*>(xb.ptr),
            static_cast<std::size_t>(xb.shape[0]),
            static_cast<std::size_t>(xb.shape[1]),
            static_cast<const int*>(yb.ptr),
            static_cast<std::size_t>(yb.shape[0])
        );

        const auto& labels = core.labels_a();
        py::array_t<int> labels_py(labels.size());
        std::memcpy(labels_py.mutable_data(), labels.data(), labels.size() * sizeof(int));

        auto cl = core.cluster_labels();
        py::array_t<int> cl_py(cl.size());
        std::memcpy(cl_py.mutable_data(), cl.data(), cl.size() * sizeof(int));

        return {labels_py, to_py_weights(core.weights()), cl_py};
    }

    std::tuple<py::array_t<int>, py::array_t<int>> predict(IntArray X) {
        auto xb = X.request();
        if (xb.ndim != 2) throw std::runtime_error("X must be a 2D array.");

        artlib_cpp::BinaryFuzzyARTMAPCore core(params_);
        core.set_state(initial_weights_, initial_cluster_labels_);
        auto [a, b] = core.predict(
            static_cast<const int*>(xb.ptr),
            static_cast<std::size_t>(xb.shape[0]),
            static_cast<std::size_t>(xb.shape[1])
        );

        py::array_t<int> a_py(a.size()), b_py(b.size());
        std::memcpy(a_py.mutable_data(), a.data(), a.size() * sizeof(int));
        std::memcpy(b_py.mutable_data(), b.data(), b.size() * sizeof(int));
        return {a_py, b_py};
    }

private:
    artlib_cpp::BinaryFuzzyARTMAPParams params_;
    std::vector<std::vector<uint32_t>> initial_weights_;
    std::vector<int> initial_cluster_labels_;
};

std::tuple<py::array_t<int>, std::vector<py::array_t<int>>, py::array_t<int>>
FitBinaryFuzzyARTMAP(
    IntArray X,
    IntArray y,
    double rho,
    std::string MT,
    uint32_t epsilon,
    py::object weights = py::none(),
    py::object cluster_labels = py::none()
) {
    cppBinaryFuzzyARTMAP model(rho, std::move(MT), epsilon, weights, cluster_labels);
    return model.fit(X, y);
}

std::tuple<py::array_t<int>, py::array_t<int>>
PredictBinaryFuzzyARTMAP(
    IntArray X,
    double rho,
    std::string MT,
    uint32_t epsilon,
    py::object weights = py::none(),
    py::object cluster_labels = py::none()
) {
    cppBinaryFuzzyARTMAP model(rho, std::move(MT), epsilon, weights, cluster_labels);
    return model.predict(X);
}

}  // namespace

PYBIND11_MODULE(cppBinaryFuzzyARTMAP, m) {
    py::class_<cppBinaryFuzzyARTMAP>(m, "cppBinaryFuzzyARTMAP")
        .def(
            py::init<double, std::string, uint32_t, py::object, py::object>(),
            py::arg("rho"),
            py::arg("MT"),
            py::arg("epsilon"),
            py::arg("weights") = py::none(),
            py::arg("cluster_labels") = py::none()
        )
        .def("fit", &cppBinaryFuzzyARTMAP::fit, py::arg("X"), py::arg("y"))
        .def("predict", &cppBinaryFuzzyARTMAP::predict, py::arg("X"))
        .def("__repr__", [](const cppBinaryFuzzyARTMAP&) { return "<cppBinaryFuzzyARTMAP model>"; });

    m.def(
        "FitBinaryFuzzyARTMAP",
        &FitBinaryFuzzyARTMAP,
        py::arg("X"),
        py::arg("y"),
        py::arg("rho"),
        py::arg("MT"),
        py::arg("epsilon"),
        py::arg("weights") = py::none(),
        py::arg("cluster_labels") = py::none()
    );

    m.def(
        "PredictBinaryFuzzyARTMAP",
        &PredictBinaryFuzzyARTMAP,
        py::arg("X"),
        py::arg("rho"),
        py::arg("MT"),
        py::arg("epsilon"),
        py::arg("weights") = py::none(),
        py::arg("cluster_labels") = py::none()
    );
}
