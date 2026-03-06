#include "artlib_cpp/binary_fuzzy_art_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "artlib_cpp/fraction_sort_core.hpp"

namespace artlib_cpp {

BinaryFuzzyARTCore::BinaryFuzzyARTCore(double rho)
    : rho_(rho), dim_original_(0), rho_int_(0) {
    if (rho_ < 0.0 || rho_ > 1.0) {
        throw std::invalid_argument("rho must be in [0,1]");
    }
}

void BinaryFuzzyARTCore::set_weights(const std::vector<std::vector<uint32_t>>& weights) {
    if (weights.empty()) {
        weights_.clear();
        w_count_cache_.clear();
        dim_original_ = 0;
        rho_int_ = 0;
        return;
    }

    const std::size_t expected_len = weights.front().size();
    if (expected_len == 0 || (expected_len % 2) != 0) {
        throw std::invalid_argument("Weight length must be even (2*dim_original).");
    }
    for (const auto& w : weights) {
        if (w.size() != expected_len) {
            throw std::invalid_argument("All weight vectors must have the same length.");
        }
    }

    const std::size_t inferred_dim = expected_len / 2;
    if (dim_original_ != 0 && inferred_dim != dim_original_) {
        throw std::invalid_argument("Weight dimensionality mismatch with model state.");
    }

    weights_ = weights;
    w_count_cache_.assign(weights_.size(), 0u);
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        w_count_cache_[i] = ones_count(weights_[i]);
    }

    dim_original_ = inferred_dim;
    rho_int_ = static_cast<uint32_t>(std::ceil(rho_ * static_cast<double>(dim_original_)));
}

void BinaryFuzzyARTCore::fit(const std::uint8_t* x, std::size_t rows, std::size_t cols) {
    validate_matrix(x, rows, cols);
    if (dim_original_ == 0) {
        dim_original_ = cols / 2;
        rho_int_ = static_cast<uint32_t>(std::ceil(rho_ * static_cast<double>(dim_original_)));
    }

    labels_.assign(rows, 0);
    std::vector<uint32_t> sample(cols);
    for (std::size_t i = 0; i < rows; ++i) {
        const auto* row = x + i * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            sample[j] = (row[j] != 0u) ? 1u : 0u;
        }
        labels_[i] = static_cast<int>(step_fit(sample));
    }
}

std::vector<int> BinaryFuzzyARTCore::predict(
    const std::uint8_t* x, std::size_t rows, std::size_t cols
) const {
    validate_matrix(x, rows, cols);
    if (weights_.empty()) {
        throw std::runtime_error("Model has no clusters");
    }

    std::vector<int> pred(rows, 0);
    std::vector<uint32_t> sample(cols);
    std::vector<fracsort::Item<uint32_t>> items(weights_.size());

    for (std::size_t i = 0; i < rows; ++i) {
        const auto* row = x + i * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            sample[j] = (row[j] != 0u) ? 1u : 0u;
        }

        for (std::size_t c = 0; c < weights_.size(); ++c) {
            const uint32_t iw_count = intersection_count(sample, weights_[c]);
            const uint32_t den = std::max<uint32_t>(1u, w_count_cache_[c]);
            items[c] = fracsort::Item<uint32_t>{iw_count, den, 0, 1, c};
        }
        const std::size_t best_cluster =
            fracsort::fracargmax_items<uint32_t>(items.data(), items.size());
        pred[i] = static_cast<int>(best_cluster);
    }
    return pred;
}

const std::vector<std::vector<uint32_t>>& BinaryFuzzyARTCore::weights() const {
    return weights_;
}

const std::vector<int>& BinaryFuzzyARTCore::labels() const {
    return labels_;
}

uint32_t BinaryFuzzyARTCore::intersection_count(
    const std::vector<uint32_t>& i, const std::vector<uint32_t>& w
) {
    uint32_t c = 0u;
    for (std::size_t j = 0; j < i.size(); ++j) {
        c += (i[j] & w[j]);
    }
    return c;
}

uint32_t BinaryFuzzyARTCore::ones_count(const std::vector<uint32_t>& v) {
    uint32_t c = 0u;
    for (uint32_t x : v) {
        c += x;
    }
    return c;
}

void BinaryFuzzyARTCore::update_inplace(std::vector<uint32_t>& w, const std::vector<uint32_t>& i) {
    for (std::size_t j = 0; j < w.size(); ++j) {
        w[j] = (w[j] & i[j]);
    }
}

uint32_t BinaryFuzzyARTCore::step_fit(const std::vector<uint32_t>& sample) {
    if (weights_.empty()) {
        weights_.push_back(sample);
        w_count_cache_.push_back(ones_count(sample));
        return 0u;
    }

    const std::size_t n_clusters = weights_.size();
    std::vector<fracsort::Item<uint32_t>> items;
    items.reserve(n_clusters);
    std::vector<uint32_t> iw_counts(n_clusters, 0u);

    for (std::size_t c = 0; c < n_clusters; ++c) {
        const uint32_t iw = intersection_count(sample, weights_[c]);
        iw_counts[c] = iw;
        if (iw < rho_int_) {
            continue;
        }
        const uint32_t den = std::max<uint32_t>(1u, w_count_cache_[c]);
        items.push_back(fracsort::Item<uint32_t>{iw, den, 0, 1, c});
    }

    if (!items.empty()) {
        const uint32_t idx = static_cast<uint32_t>(
            fracsort::fracargmax_items<uint32_t>(items.data(), items.size())
        );
        if (idx >= n_clusters) {
            throw std::runtime_error("idx out of range from fracargmax_items");
        }
        const uint32_t iw_count = iw_counts[idx];
        update_inplace(weights_[idx], sample);
        w_count_cache_[idx] = iw_count;
        return idx;
    }

    const uint32_t new_id = static_cast<uint32_t>(weights_.size());
    weights_.push_back(sample);
    w_count_cache_.push_back(ones_count(sample));
    return new_id;
}

void BinaryFuzzyARTCore::validate_matrix(
    const std::uint8_t* x, std::size_t /*rows*/, std::size_t cols
) const {
    if (x == nullptr) {
        throw std::invalid_argument("X cannot be null");
    }
    if (cols == 0 || (cols % 2u) != 0u) {
        throw std::invalid_argument("Number of features must be even (2*dim_original).");
    }
    if (dim_original_ != 0 && cols != 2u * dim_original_) {
        throw std::invalid_argument("Number of features does not match existing weights.");
    }
}

}  // namespace artlib_cpp
