#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "artlib_cpp/fusion_core.hpp"

namespace py = pybind11;

namespace {

int ArgmaxWeightedActivations(
    py::array_t<double, py::array::c_style | py::array::forcecast> activations,
    py::array_t<double, py::array::c_style | py::array::forcecast> gamma_values,
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> skip_mask
) {
    auto ab = activations.request();
    auto gb = gamma_values.request();
    auto sb = skip_mask.request();
    if (ab.ndim != 2) throw std::runtime_error("activations must be 2-D");
    if (gb.ndim != 1) throw std::runtime_error("gamma_values must be 1-D");
    if (sb.ndim != 1) throw std::runtime_error("skip_mask must be 1-D");
    if (gb.shape[0] != ab.shape[1]) {
        throw std::runtime_error("gamma_values length must match n_channels");
    }
    if (sb.shape[0] != ab.shape[1]) {
        throw std::runtime_error("skip_mask length must match n_channels");
    }

    return artlib_cpp::ArgmaxWeightedActivations(
        static_cast<const double*>(ab.ptr),
        static_cast<std::size_t>(ab.shape[0]),
        static_cast<std::size_t>(ab.shape[1]),
        static_cast<const double*>(gb.ptr),
        static_cast<const unsigned char*>(sb.ptr)
    );
}

int ArgmaxWeightedChannelActivations(
    py::array_t<double, py::array::c_style | py::array::forcecast> channel_activations,
    py::array_t<double, py::array::c_style | py::array::forcecast> gamma_values,
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> skip_mask
) {
    auto ab = channel_activations.request();
    auto gb = gamma_values.request();
    auto sb = skip_mask.request();
    if (ab.ndim != 2) throw std::runtime_error("channel_activations must be 2-D");
    if (gb.ndim != 1) throw std::runtime_error("gamma_values must be 1-D");
    if (sb.ndim != 1) throw std::runtime_error("skip_mask must be 1-D");
    if (gb.shape[0] != ab.shape[0]) {
        throw std::runtime_error("gamma_values length must match n_channels");
    }
    if (sb.shape[0] != ab.shape[0]) {
        throw std::runtime_error("skip_mask length must match n_channels");
    }

    return artlib_cpp::ArgmaxWeightedChannelActivations(
        static_cast<const double*>(ab.ptr),
        static_cast<std::size_t>(ab.shape[0]),
        static_cast<std::size_t>(ab.shape[1]),
        static_cast<const double*>(gb.ptr),
        static_cast<const unsigned char*>(sb.ptr)
    );
}

}  // namespace

PYBIND11_MODULE(cppFusionUtils, m) {
    m.def(
        "ArgmaxWeightedActivations",
        &ArgmaxWeightedActivations,
        py::arg("activations"),
        py::arg("gamma_values"),
        py::arg("skip_mask")
    );
    m.def(
        "ArgmaxWeightedChannelActivations",
        &ArgmaxWeightedChannelActivations,
        py::arg("channel_activations"),
        py::arg("gamma_values"),
        py::arg("skip_mask")
    );
}
