#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>
#include <vector>

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
    py::gil_scoped_release release;
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

    py::gil_scoped_release release;
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

    {
        py::gil_scoped_release release;
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
    }
    return out;
}

py::array_t<double> JoinChannelsWithFill(
    py::list channel_data,
    py::array_t<long long, py::array::c_style | py::array::forcecast> channel_widths,
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> present_mask,
    double fill_value
) {
    auto wb = channel_widths.request();
    auto mb = present_mask.request();
    if (wb.ndim != 1) throw std::runtime_error("channel_widths must be 1-D");
    if (mb.ndim != 1) throw std::runtime_error("present_mask must be 1-D");
    if (wb.shape[0] != mb.shape[0]) {
        throw std::runtime_error("channel_widths and present_mask lengths must match");
    }
    const auto n_total_channels = static_cast<std::size_t>(wb.shape[0]);
    if (n_total_channels == 0) {
        throw std::runtime_error("at least one channel is required");
    }

    auto* widths_ptr = static_cast<const long long*>(wb.ptr);
    auto* mask_ptr = static_cast<const unsigned char*>(mb.ptr);
    std::vector<std::size_t> widths(n_total_channels);
    std::vector<const double*> channel_ptrs;
    channel_ptrs.reserve(static_cast<std::size_t>(py::len(channel_data)));

    std::size_t n_present_channels = 0;
    std::size_t n_samples = 0;
    for (std::size_t k = 0; k < n_total_channels; ++k) {
        if (widths_ptr[k] <= 0) {
            throw std::runtime_error("channel widths must be > 0");
        }
        widths[k] = static_cast<std::size_t>(widths_ptr[k]);
        if (mask_ptr[k]) {
            ++n_present_channels;
        }
    }
    if (static_cast<std::size_t>(py::len(channel_data)) != n_present_channels) {
        throw std::runtime_error("channel_data length must match number of present channels");
    }

    std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> arrays;
    arrays.reserve(n_present_channels);
    std::size_t input_idx = 0;
    for (std::size_t k = 0; k < n_total_channels; ++k) {
        if (!mask_ptr[k]) {
            continue;
        }
        py::array_t<double, py::array::c_style | py::array::forcecast> arr =
            py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(
                channel_data[input_idx]
            );
        auto ab = arr.request();
        if (ab.ndim != 2) throw std::runtime_error("each channel must be 2-D");
        if (static_cast<std::size_t>(ab.shape[1]) != widths[k]) {
            throw std::runtime_error("channel width does not match channel_widths");
        }
        if (input_idx == 0) {
            n_samples = static_cast<std::size_t>(ab.shape[0]);
            if (n_samples == 0) {
                throw std::runtime_error("channels must have at least one row");
            }
        } else if (static_cast<std::size_t>(ab.shape[0]) != n_samples) {
            throw std::runtime_error("all channels must have the same number of rows");
        }
        arrays.push_back(arr);
        channel_ptrs.push_back(static_cast<const double*>(ab.ptr));
        ++input_idx;
    }

    std::size_t total_width = 0;
    for (std::size_t width : widths) {
        total_width += width;
    }

    py::array_t<double> out({static_cast<py::ssize_t>(n_samples), static_cast<py::ssize_t>(total_width)});
    auto ob = out.request();
    {
        py::gil_scoped_release release;
        artlib_cpp::JoinChannelsWithFill(
            channel_ptrs.data(),
            widths.data(),
            static_cast<const unsigned char*>(mb.ptr),
            n_total_channels,
            n_samples,
            fill_value,
            static_cast<double*>(ob.ptr)
        );
    }
    return out;
}

py::list ExtractPresentChannels(
    py::array_t<double, py::array::c_style | py::array::forcecast> joined_data,
    py::array_t<long long, py::array::c_style | py::array::forcecast> channel_widths,
    py::array_t<unsigned char, py::array::c_style | py::array::forcecast> present_mask
) {
    auto jb = joined_data.request();
    auto wb = channel_widths.request();
    auto mb = present_mask.request();
    if (jb.ndim != 2) throw std::runtime_error("joined_data must be 2-D");
    if (wb.ndim != 1) throw std::runtime_error("channel_widths must be 1-D");
    if (mb.ndim != 1) throw std::runtime_error("present_mask must be 1-D");
    if (wb.shape[0] != mb.shape[0]) {
        throw std::runtime_error("channel_widths and present_mask lengths must match");
    }

    const auto n_total_channels = static_cast<std::size_t>(wb.shape[0]);
    if (n_total_channels == 0) {
        throw std::runtime_error("at least one channel is required");
    }

    auto* widths_ptr = static_cast<const long long*>(wb.ptr);
    auto* mask_ptr = static_cast<const unsigned char*>(mb.ptr);
    std::vector<std::size_t> widths(n_total_channels);
    std::size_t total_width = 0;
    std::size_t n_present_channels = 0;
    for (std::size_t k = 0; k < n_total_channels; ++k) {
        if (widths_ptr[k] <= 0) {
            throw std::runtime_error("channel widths must be > 0");
        }
        widths[k] = static_cast<std::size_t>(widths_ptr[k]);
        total_width += widths[k];
        if (mask_ptr[k]) {
            ++n_present_channels;
        }
    }
    if (static_cast<std::size_t>(jb.shape[1]) != total_width) {
        throw std::runtime_error("joined_data width does not match channel_widths");
    }
    if (jb.shape[0] == 0) {
        throw std::runtime_error("joined_data must have at least one row");
    }

    std::vector<py::array_t<double>> outputs;
    outputs.reserve(n_present_channels);
    std::vector<double*> output_ptrs;
    output_ptrs.reserve(n_present_channels);
    for (std::size_t k = 0; k < n_total_channels; ++k) {
        if (!mask_ptr[k]) {
            continue;
        }
        py::array_t<double> out(
            {jb.shape[0], static_cast<py::ssize_t>(widths[k])}
        );
        auto ob = out.request();
        outputs.push_back(out);
        output_ptrs.push_back(static_cast<double*>(ob.ptr));
    }

    {
        py::gil_scoped_release release;
        artlib_cpp::ExtractPresentChannels(
            static_cast<const double*>(jb.ptr),
            static_cast<std::size_t>(jb.shape[0]),
            widths.data(),
            static_cast<const unsigned char*>(mb.ptr),
            n_total_channels,
            output_ptrs.data()
        );
    }

    py::list result;
    for (auto& out : outputs) {
        result.append(out);
    }
    return result;
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
    m.def(
        "JoinChannelsWithFill",
        &JoinChannelsWithFill,
        py::arg("channel_data"),
        py::arg("channel_widths"),
        py::arg("present_mask"),
        py::arg("fill_value") = 0.5
    );
    m.def(
        "ExtractPresentChannels",
        &ExtractPresentChannels,
        py::arg("joined_data"),
        py::arg("channel_widths"),
        py::arg("present_mask")
    );
}
