#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "artlib_cpp/fuzzy_artmap_core.hpp"

namespace py = pybind11;

namespace {

using XArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using YArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

std::vector<std::vector<double>> parse_weights(const py::object& weights) {
    if (weights.is_none()) return {};

    if (py::isinstance<py::array>(weights)) {
        XArray arr = weights.cast<XArray>();
        auto req = arr.request();
        if (req.ndim != 2) throw std::runtime_error("weights ndarray must be 2-D");
        const auto rows = static_cast<std::size_t>(req.shape[0]);
        const auto cols = static_cast<std::size_t>(req.shape[1]);
        const auto* ptr = static_cast<const double*>(req.ptr);
        std::vector<std::vector<double>> out(rows, std::vector<double>(cols, 0.0));
        for (std::size_t r = 0; r < rows; ++r) {
            std::memcpy(out[r].data(), ptr + r * cols, cols * sizeof(double));
        }
        return out;
    }

    if (!py::isinstance<py::list>(weights)) {
        throw std::runtime_error("weights must be None, list, or ndarray");
    }

    py::list w_list = weights.cast<py::list>();
    std::vector<std::vector<double>> out;
    out.reserve(w_list.size());
    for (py::handle item : w_list) {
        XArray w_arr = py::cast<XArray>(item);
        auto req = w_arr.request();
        if (req.ndim != 1) throw std::runtime_error("each weight must be 1-D");
        const auto len = static_cast<std::size_t>(req.shape[0]);
        const auto* ptr = static_cast<const double*>(req.ptr);
        std::vector<double> w(len, 0.0);
        std::memcpy(w.data(), ptr, len * sizeof(double));
        out.push_back(std::move(w));
    }
    return out;
}

std::vector<int> parse_cluster_labels(const py::object& labels) {
    if (labels.is_none()) return {};
    YArray arr = labels.cast<YArray>();
    auto req = arr.request();
    if (req.ndim != 1) throw std::runtime_error("cluster_labels must be 1-D");
    const int* ptr = static_cast<const int*>(req.ptr);
    return {ptr, ptr + static_cast<std::size_t>(req.shape[0])};
}

std::vector<py::array_t<double>> to_py_weights(
    const std::vector<std::vector<double>>& weights
) {
    std::vector<py::array_t<double>> out;
    out.reserve(weights.size());
    for (const auto& w_vec : weights) {
        py::array_t<double> arr(w_vec.size());
        std::memcpy(arr.mutable_data(), w_vec.data(), w_vec.size() * sizeof(double));
        out.push_back(std::move(arr));
    }
    return out;
}

class cppFuzzyARTMAP {
public:
    cppFuzzyARTMAP(
        double rho,
        double alpha,
        double beta,
        std::string MT,
        double epsilon,
        py::object weights = py::none(),
        py::object cluster_labels = py::none()
    ) : params_{rho, alpha, beta, std::move(MT), epsilon} {
        const bool have_w = !weights.is_none();
        const bool have_cl = !cluster_labels.is_none();
        if (have_w != have_cl) {
            throw std::invalid_argument("Provide BOTH 'weights' and 'cluster_labels' or neither.");
        }
        if (have_w) {
            initial_weights_ = parse_weights(weights);
            initial_cluster_labels_ = parse_cluster_labels(cluster_labels);
        }
    }

    std::tuple<py::array_t<int>, std::vector<py::array_t<double>>, py::array_t<int>>
    fit(XArray X, YArray y) {
        auto xb = X.request();
        auto yb = y.request();
        if (xb.ndim != 2 || yb.ndim != 1) {
            throw std::runtime_error("X must be 2-D and y must be 1-D");
        }

        artlib_cpp::FuzzyARTMAPCore core(params_);
        core.set_state(initial_weights_, initial_cluster_labels_);
        core.fit(
            static_cast<const double*>(xb.ptr),
            static_cast<std::size_t>(xb.shape[0]),
            static_cast<std::size_t>(xb.shape[1]),
            static_cast<const int*>(yb.ptr),
            static_cast<std::size_t>(yb.shape[0])
        );

        const auto& labels = core.labels_a();
        py::array_t<int> labels_out(labels.size());
        std::memcpy(labels_out.mutable_data(), labels.data(), labels.size() * sizeof(int));

        auto cl = core.cluster_labels();
        py::array_t<int> cl_out(cl.size());
        std::memcpy(cl_out.mutable_data(), cl.data(), cl.size() * sizeof(int));

        return {labels_out, to_py_weights(core.weights()), cl_out};
    }

    std::tuple<py::array_t<int>, py::array_t<int>> predict(XArray X) {
        auto xb = X.request();
        if (xb.ndim != 2) throw std::runtime_error("X must be 2-D");

        artlib_cpp::FuzzyARTMAPCore core(params_);
        core.set_state(initial_weights_, initial_cluster_labels_);
        auto [ya, yb] = core.predict(
            static_cast<const double*>(xb.ptr),
            static_cast<std::size_t>(xb.shape[0]),
            static_cast<std::size_t>(xb.shape[1])
        );

        py::array_t<int> ya_out(ya.size()), yb_out(yb.size());
        std::memcpy(ya_out.mutable_data(), ya.data(), ya.size() * sizeof(int));
        std::memcpy(yb_out.mutable_data(), yb.data(), yb.size() * sizeof(int));
        return {ya_out, yb_out};
    }

private:
    artlib_cpp::FuzzyARTMAPParams params_;
    std::vector<std::vector<double>> initial_weights_;
    std::vector<int> initial_cluster_labels_;
};

auto FitFuzzyARTMAP(
    XArray X,
    YArray y,
    double rho,
    double alpha,
    double beta,
    const std::string& MT,
    double epsilon,
    py::object weights = py::none(),
    py::object cluster_labels = py::none()
) {
    cppFuzzyARTMAP model(rho, alpha, beta, MT, epsilon, weights, cluster_labels);
    return model.fit(X, y);
}

auto PredictFuzzyARTMAP(
    XArray X,
    double rho,
    double alpha,
    double beta,
    const std::string& MT,
    double epsilon,
    py::object weights = py::none(),
    py::object cluster_labels = py::none()
) {
    cppFuzzyARTMAP model(rho, alpha, beta, MT, epsilon, weights, cluster_labels);
    return model.predict(X);
}

}  // namespace

PYBIND11_MODULE(cppFuzzyARTMAP, m) {
    py::class_<cppFuzzyARTMAP>(m, "cppFuzzyARTMAP")
        .def(
            py::init<double, double, double, std::string, double, py::object, py::object>(),
            py::arg("rho"),
            py::arg("alpha"),
            py::arg("beta"),
            py::arg("MT"),
            py::arg("epsilon"),
            py::arg("weights") = py::none(),
            py::arg("cluster_labels") = py::none()
        )
        .def("fit", &cppFuzzyARTMAP::fit, py::arg("X"), py::arg("y"))
        .def("predict", &cppFuzzyARTMAP::predict, py::arg("X"))
        .def("__repr__", [](const cppFuzzyARTMAP&) { return "<cppFuzzyARTMAP>"; });

    m.def(
        "FitFuzzyARTMAP",
        &FitFuzzyARTMAP,
        py::arg("X"),
        py::arg("y"),
        py::arg("rho"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("MT"),
        py::arg("epsilon"),
        py::arg("weights") = py::none(),
        py::arg("cluster_labels") = py::none()
    );

    m.def(
        "PredictFuzzyARTMAP",
        &PredictFuzzyARTMAP,
        py::arg("X"),
        py::arg("rho"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("MT"),
        py::arg("epsilon"),
        py::arg("weights") = py::none(),
        py::arg("cluster_labels") = py::none()
    );
}
