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

void BuildStateActionRewardQuery(
    const double* state,
    std::size_t state_dim,
    const double* actions,
    std::size_t n_actions,
    std::size_t action_dim,
    std::size_t reward_dim,
    double fill_value,
    double* out
);

}  // namespace artlib_cpp
