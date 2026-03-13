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

void JoinChannelsWithFill(
    const double* const* channels,
    const std::size_t* widths,
    const unsigned char* present_mask,
    std::size_t n_total_channels,
    std::size_t n_samples,
    double fill_value,
    double* out
);

void ExtractPresentChannels(
    const double* joined_data,
    std::size_t n_samples,
    const std::size_t* widths,
    const unsigned char* present_mask,
    std::size_t n_total_channels,
    double** outputs
);

}  // namespace artlib_cpp
