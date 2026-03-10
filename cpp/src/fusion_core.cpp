#include "artlib_cpp/fusion_core.hpp"

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

}  // namespace artlib_cpp
