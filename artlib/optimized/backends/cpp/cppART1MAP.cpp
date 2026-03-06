#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "artlib_cpp/art1map_core.hpp"

namespace py = pybind11;

namespace {

using XArray = py::array_t<std::int16_t, py::array::c_style | py::array::forcecast>;
using YArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

struct XYView {
    const std::int16_t* x;
    std::size_t rows;
    std::size_t cols;
};

XYView parse_x(const XArray& x) {
    const auto req = x.request();
    if (req.ndim != 2) {
        throw std::runtime_error("X must be 2-D");
    }
    return {
        static_cast<const std::int16_t*>(req.ptr),
        static_cast<std::size_t>(req.shape[0]),
        static_cast<std::size_t>(req.shape[1]),
    };
}

std::pair<const int*, std::size_t> parse_y(const YArray& y) {
    const auto req = y.request();
    if (req.ndim != 1) {
        throw std::runtime_error("y must be 1-D");
    }
    return {
        static_cast<const int*>(req.ptr),
        static_cast<std::size_t>(req.shape[0]),
    };
}

std::vector<std::vector<double>> parse_weights(const py::object& weights) {
    if (weights.is_none()) {
        return {};
    }

    std::vector<std::vector<double>> out;
    if (py::isinstance<py::array>(weights)) {
        py::array_t<double, py::array::c_style | py::array::forcecast> arr =
            weights.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
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

std::vector<int> parse_cluster_labels(const py::object& labels) {
    if (labels.is_none()) {
        return {};
    }
    YArray arr = labels.cast<YArray>();
    const auto req = arr.request();
    if (req.ndim != 1) {
        throw std::runtime_error("cluster_labels must be 1-D");
    }
    const auto* ptr = static_cast<const int*>(req.ptr);
    return {ptr, ptr + static_cast<std::size_t>(req.shape[0])};
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

class cppART1MAP {
public:
    cppART1MAP(
        double rho,
        double L,
        std::string MT,
        double epsilon,
        py::object weights = py::none(),
        py::object cluster_labels = py::none()
    )
        : params_{rho, L, std::move(MT), epsilon} {
        const bool have_w = !weights.is_none();
        const bool have_cl = !cluster_labels.is_none();
        if (have_w != have_cl) {
            throw std::invalid_argument(
                "Provide BOTH 'weights' and 'cluster_labels' or neither."
            );
        }
        if (have_w) {
            initial_weights_ = parse_weights(weights);
            initial_cluster_labels_ = parse_cluster_labels(cluster_labels);
        }
    }

    std::tuple<py::array_t<int>, std::vector<py::array_t<double>>, py::array_t<int>> fit(
        const XArray& x,
        const YArray& y
    ) {
        artlib_cpp::ART1MAPCore core(params_);
        core.set_state(initial_weights_, initial_cluster_labels_);

        const auto xv = parse_x(x);
        const auto yv = parse_y(y);
        core.fit(xv.x, xv.rows, xv.cols, yv.first, yv.second);

        const auto& labels = core.labels_a();
        py::array_t<int> labels_out(labels.size());
        std::memcpy(labels_out.mutable_data(), labels.data(), sizeof(int) * labels.size());

        auto cluster_labels = core.cluster_labels();
        py::array_t<int> cl_out(cluster_labels.size());
        std::memcpy(cl_out.mutable_data(), cluster_labels.data(), sizeof(int) * cluster_labels.size());

        return {labels_out, to_py_weights(core.weights()), cl_out};
    }

    std::tuple<py::array_t<int>, py::array_t<int>> predict(const XArray& x) {
        artlib_cpp::ART1MAPCore core(params_);
        core.set_state(initial_weights_, initial_cluster_labels_);

        const auto xv = parse_x(x);
        auto [ya, yb] = core.predict(xv.x, xv.rows, xv.cols);

        py::array_t<int> ya_out(ya.size()), yb_out(yb.size());
        std::memcpy(ya_out.mutable_data(), ya.data(), sizeof(int) * ya.size());
        std::memcpy(yb_out.mutable_data(), yb.data(), sizeof(int) * yb.size());
        return {ya_out, yb_out};
    }

private:
    artlib_cpp::ART1MAPParams params_;
    std::vector<std::vector<double>> initial_weights_;
    std::vector<int> initial_cluster_labels_;
};

auto FitART1MAP(
    const XArray& x,
    const YArray& y,
    double rho,
    double L,
    const std::string& MT,
    double epsilon,
    py::object weights = py::none(),
    py::object cluster_labels = py::none()
) {
    cppART1MAP model(rho, L, MT, epsilon, weights, cluster_labels);
    return model.fit(x, y);
}

auto PredictART1MAP(
    const XArray& x,
    double rho,
    double L,
    const std::string& MT,
    double epsilon,
    py::object weights = py::none(),
    py::object cluster_labels = py::none()
) {
    cppART1MAP model(rho, L, MT, epsilon, weights, cluster_labels);
    return model.predict(x);
}

}  // namespace

PYBIND11_MODULE(cppART1MAP, m) {
    py::class_<cppART1MAP>(m, "cppART1MAP")
        .def(
            py::init<double, double, std::string, double, py::object, py::object>(),
            py::arg("rho"),
            py::arg("L"),
            py::arg("MT"),
            py::arg("epsilon"),
            py::arg("weights") = py::none(),
            py::arg("cluster_labels") = py::none()
        )
        .def("fit", &cppART1MAP::fit, py::arg("X"), py::arg("y"))
        .def("predict", &cppART1MAP::predict, py::arg("X"))
        .def("__repr__", [](const cppART1MAP&) { return "<cppART1MAP>"; });

    m.def(
        "FitART1MAP",
        &FitART1MAP,
        py::arg("X"),
        py::arg("y"),
        py::arg("rho"),
        py::arg("L"),
        py::arg("MT"),
        py::arg("epsilon"),
        py::arg("weights") = py::none(),
        py::arg("cluster_labels") = py::none()
    );

    m.def(
        "PredictART1MAP",
        &PredictART1MAP,
        py::arg("X"),
        py::arg("rho"),
        py::arg("L"),
        py::arg("MT"),
        py::arg("epsilon"),
        py::arg("weights") = py::none(),
        py::arg("cluster_labels") = py::none()
    );
}
