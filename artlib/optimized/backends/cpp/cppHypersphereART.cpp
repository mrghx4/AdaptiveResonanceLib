#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "artlib_cpp/hypersphere_art_core.hpp"

namespace py = pybind11;

namespace {

using XArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

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

class cppHypersphereART {
public:
    cppHypersphereART(
        double rho,
        double alpha,
        double beta,
        double r_hat,
        py::object weights = py::none()
    ) : params_{rho, alpha, beta, r_hat} {
        if (!weights.is_none()) {
            initial_weights_ = parse_weights(weights);
        }
    }

    std::tuple<py::array_t<int>, std::vector<py::array_t<double>>> fit(XArray X) {
        auto xb = X.request();
        if (xb.ndim != 2) throw std::runtime_error("X must be 2-D");

        artlib_cpp::HypersphereARTCore core(params_);
        core.set_weights(initial_weights_);
        core.fit(
            static_cast<const double*>(xb.ptr),
            static_cast<std::size_t>(xb.shape[0]),
            static_cast<std::size_t>(xb.shape[1])
        );

        const auto& labels = core.labels();
        py::array_t<int> labels_out(labels.size());
        std::memcpy(labels_out.mutable_data(), labels.data(), labels.size() * sizeof(int));
        return {labels_out, to_py_weights(core.weights())};
    }

    py::array_t<int> predict(XArray X) {
        auto xb = X.request();
        if (xb.ndim != 2) throw std::runtime_error("X must be 2-D");

        artlib_cpp::HypersphereARTCore core(params_);
        core.set_weights(initial_weights_);
        auto y = core.predict(
            static_cast<const double*>(xb.ptr),
            static_cast<std::size_t>(xb.shape[0]),
            static_cast<std::size_t>(xb.shape[1])
        );

        py::array_t<int> y_out(y.size());
        std::memcpy(y_out.mutable_data(), y.data(), y.size() * sizeof(int));
        return y_out;
    }

private:
    artlib_cpp::HypersphereARTParams params_;
    std::vector<std::vector<double>> initial_weights_;
};

auto FitHypersphereART(
    XArray X,
    double rho,
    double alpha,
    double beta,
    double r_hat,
    py::object weights = py::none()
) {
    cppHypersphereART model(rho, alpha, beta, r_hat, weights);
    return model.fit(X);
}

auto PredictHypersphereART(
    XArray X,
    double rho,
    double alpha,
    double beta,
    double r_hat,
    py::object weights = py::none()
) {
    cppHypersphereART model(rho, alpha, beta, r_hat, weights);
    return model.predict(X);
}

}  // namespace

PYBIND11_MODULE(cppHypersphereART, m) {
    py::class_<cppHypersphereART>(m, "cppHypersphereART")
        .def(
            py::init<double, double, double, double, py::object>(),
            py::arg("rho"),
            py::arg("alpha"),
            py::arg("beta"),
            py::arg("r_hat"),
            py::arg("weights") = py::none()
        )
        .def("fit", &cppHypersphereART::fit, py::arg("X"))
        .def("predict", &cppHypersphereART::predict, py::arg("X"))
        .def("__repr__", [](const cppHypersphereART&) { return "<cppHypersphereART>"; });

    m.def(
        "FitHypersphereART",
        &FitHypersphereART,
        py::arg("X"),
        py::arg("rho"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("r_hat"),
        py::arg("weights") = py::none()
    );

    m.def(
        "PredictHypersphereART",
        &PredictHypersphereART,
        py::arg("X"),
        py::arg("rho"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("r_hat"),
        py::arg("weights") = py::none()
    );
}
