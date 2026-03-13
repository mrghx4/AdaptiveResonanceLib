#include "artlib_cpp/fusion_core.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace artlib_cpp {

int ArgmaxWeightedActivations(
    const double* activations,
    std::size_t n_categories,
    std::size_t n_channels,
    const double* gamma_values,
    const unsigned char* skip_mask
) {
    if (activations == nullptr) {
        throw std::invalid_argument("activations cannot be null");
    }
    if (gamma_values == nullptr) {
        throw std::invalid_argument("gamma_values cannot be null");
    }
    if (skip_mask == nullptr) {
        throw std::invalid_argument("skip_mask cannot be null");
    }
    if (n_categories == 0) {
        throw std::invalid_argument("n_categories must be > 0");
    }
    if (n_channels == 0) {
        throw std::invalid_argument("n_channels must be > 0");
    }

    int best_idx = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < n_categories; ++c) {
        const std::size_t row_off = c * n_channels;
        double score = 0.0;
        for (std::size_t k = 0; k < n_channels; ++k) {
            if (skip_mask[k]) {
                continue;
            }
            score += activations[row_off + k] * gamma_values[k];
        }
        if (score > best_score) {
            best_score = score;
            best_idx = static_cast<int>(c);
        }
    }
    return best_idx;
}

int ArgmaxWeightedChannelActivations(
    const double* channel_activations,
    std::size_t n_channels,
    std::size_t n_categories,
    const double* gamma_values,
    const unsigned char* skip_mask
) {
    if (channel_activations == nullptr) {
        throw std::invalid_argument("channel_activations cannot be null");
    }
    if (gamma_values == nullptr) {
        throw std::invalid_argument("gamma_values cannot be null");
    }
    if (skip_mask == nullptr) {
        throw std::invalid_argument("skip_mask cannot be null");
    }
    if (n_categories == 0) {
        throw std::invalid_argument("n_categories must be > 0");
    }
    if (n_channels == 0) {
        throw std::invalid_argument("n_channels must be > 0");
    }

    int best_idx = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < n_categories; ++c) {
        double score = 0.0;
        for (std::size_t k = 0; k < n_channels; ++k) {
            if (skip_mask[k]) {
                continue;
            }
            const std::size_t off = k * n_categories + c;
            score += channel_activations[off] * gamma_values[k];
        }
        if (score > best_score) {
            best_score = score;
            best_idx = static_cast<int>(c);
        }
    }
    return best_idx;
}

void BuildStateActionRewardQuery(
    const double* state,
    std::size_t state_dim,
    const double* actions,
    std::size_t n_actions,
    std::size_t action_dim,
    std::size_t reward_dim,
    double fill_value,
    double* out
) {
    if (state == nullptr) {
        throw std::invalid_argument("state cannot be null");
    }
    if (actions == nullptr) {
        throw std::invalid_argument("actions cannot be null");
    }
    if (out == nullptr) {
        throw std::invalid_argument("out cannot be null");
    }
    if (n_actions == 0) {
        throw std::invalid_argument("n_actions must be > 0");
    }
    if (reward_dim == 0) {
        throw std::invalid_argument("reward_dim must be > 0");
    }

    const std::size_t row_dim = state_dim + action_dim + reward_dim;
    for (std::size_t i = 0; i < n_actions; ++i) {
        double* row = out + (i * row_dim);
        std::copy(state, state + state_dim, row);
        std::copy(
            actions + (i * action_dim),
            actions + ((i + 1) * action_dim),
            row + state_dim
        );
        std::fill(
            row + state_dim + action_dim,
            row + row_dim,
            fill_value
        );
    }
}

void JoinChannelsWithFill(
    const double* const* channels,
    const std::size_t* widths,
    const unsigned char* present_mask,
    std::size_t n_total_channels,
    std::size_t n_samples,
    double fill_value,
    double* out
) {
    if (channels == nullptr) {
        throw std::invalid_argument("channels cannot be null");
    }
    if (widths == nullptr) {
        throw std::invalid_argument("widths cannot be null");
    }
    if (present_mask == nullptr) {
        throw std::invalid_argument("present_mask cannot be null");
    }
    if (out == nullptr) {
        throw std::invalid_argument("out cannot be null");
    }
    if (n_total_channels == 0) {
        throw std::invalid_argument("n_total_channels must be > 0");
    }
    if (n_samples == 0) {
        throw std::invalid_argument("n_samples must be > 0");
    }

    std::size_t total_width = 0;
    for (std::size_t k = 0; k < n_total_channels; ++k) {
        total_width += widths[k];
    }

    std::size_t out_offset = 0;
    std::size_t input_idx = 0;
    for (std::size_t k = 0; k < n_total_channels; ++k) {
        const std::size_t width = widths[k];
        if (present_mask[k]) {
            const double* channel = channels[input_idx];
            if (channel == nullptr) {
                throw std::invalid_argument("present channel pointer cannot be null");
            }
            for (std::size_t row = 0; row < n_samples; ++row) {
                const double* src = channel + (row * width);
                double* dst = out + (row * total_width) + out_offset;
                std::copy(src, src + width, dst);
            }
            ++input_idx;
        } else {
            for (std::size_t row = 0; row < n_samples; ++row) {
                double* dst = out + (row * total_width) + out_offset;
                std::fill(dst, dst + width, fill_value);
            }
        }
        out_offset += width;
    }
}

void ExtractPresentChannels(
    const double* joined_data,
    std::size_t n_samples,
    const std::size_t* widths,
    const unsigned char* present_mask,
    std::size_t n_total_channels,
    double** outputs
) {
    if (joined_data == nullptr) {
        throw std::invalid_argument("joined_data cannot be null");
    }
    if (widths == nullptr) {
        throw std::invalid_argument("widths cannot be null");
    }
    if (present_mask == nullptr) {
        throw std::invalid_argument("present_mask cannot be null");
    }
    if (outputs == nullptr) {
        throw std::invalid_argument("outputs cannot be null");
    }
    if (n_total_channels == 0) {
        throw std::invalid_argument("n_total_channels must be > 0");
    }
    if (n_samples == 0) {
        throw std::invalid_argument("n_samples must be > 0");
    }

    std::size_t total_width = 0;
    for (std::size_t k = 0; k < n_total_channels; ++k) {
        total_width += widths[k];
    }

    std::size_t input_idx = 0;
    std::size_t joined_offset = 0;
    for (std::size_t k = 0; k < n_total_channels; ++k) {
        const std::size_t width = widths[k];
        if (present_mask[k]) {
            double* out = outputs[input_idx];
            if (out == nullptr) {
                throw std::invalid_argument("output channel pointer cannot be null");
            }
            for (std::size_t row = 0; row < n_samples; ++row) {
                const double* src = joined_data + (row * total_width) + joined_offset;
                double* dst = out + (row * width);
                std::copy(src, src + width, dst);
            }
            ++input_idx;
        }
        joined_offset += width;
    }
}

}  // namespace artlib_cpp
