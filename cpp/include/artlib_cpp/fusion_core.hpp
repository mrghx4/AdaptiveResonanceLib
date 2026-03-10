#pragma once

#include <cstddef>

namespace artlib_cpp {

int ArgmaxWeightedActivations(
    const double* activations,
    std::size_t n_categories,
    std::size_t n_channels,
    const double* gamma_values,
    const unsigned char* skip_mask
);

int ArgmaxWeightedChannelActivations(
    const double* channel_activations,
    std::size_t n_channels,
    std::size_t n_categories,
    const double* gamma_values,
    const unsigned char* skip_mask
);

}  // namespace artlib_cpp
