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

py::array_t<double> BuildStateActionRewardQuery(
    py::array_t<double, py::array::c_style | py::array::forcecast> state,
    py::array_t<double, py::array::c_style | py::array::forcecast> actions,
    std::size_t reward_dim,
    double fill_value
) {
    auto sb = state.request();
    auto ab = actions.request();
    if (sb.ndim != 1) throw std::runtime_error("state must be 1-D");
    if (ab.ndim != 2) throw std::runtime_error("actions must be 2-D");
    if (ab.shape[0] == 0) throw std::runtime_error("actions must have at least one row");
    if (reward_dim == 0) throw std::runtime_error("reward_dim must be > 0");

    const auto state_dim = static_cast<std::size_t>(sb.shape[0]);
    const auto n_actions = static_cast<std::size_t>(ab.shape[0]);
    const auto action_dim = static_cast<std::size_t>(ab.shape[1]);
    const auto row_dim = state_dim + action_dim + reward_dim;

    py::array_t<double> out({ab.shape[0], static_cast<py::ssize_t>(row_dim)});
    auto ob = out.request();

    artlib_cpp::BuildStateActionRewardQuery(
        static_cast<const double*>(sb.ptr),
        state_dim,
        static_cast<const double*>(ab.ptr),
        n_actions,
        action_dim,
        reward_dim,
        fill_value,
        static_cast<double*>(ob.ptr)
    );
    return out;
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
    m.def(
        "BuildStateActionRewardQuery",
        &BuildStateActionRewardQuery,
        py::arg("state"),
        py::arg("actions"),
        py::arg("reward_dim"),
        py::arg("fill_value") = 0.5
    );
}
