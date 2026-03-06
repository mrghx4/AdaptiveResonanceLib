#include "artlib_cpp/binary_fuzzy_artmap_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "artlib_cpp/fraction_sort_core.hpp"

namespace artlib_cpp {

BinaryFuzzyARTMAPCore::BinaryFuzzyARTMAPCore(BinaryFuzzyARTMAPParams params)
    : params_(std::move(params)), dim_original_(0), rho_w1_(0), rho_int_(0) {
    if (params_.rho < 0.0 || params_.rho > 1.0) {
        throw std::invalid_argument("rho must be in [0,1]");
    }
}

void BinaryFuzzyARTMAPCore::set_state(
    const std::vector<std::vector<uint32_t>>& weights,
    const std::vector<int>& cluster_labels
) {
    if (weights.empty() && cluster_labels.empty()) {
        weights_.clear();
        w_count_cache_.clear();
        cluster_map_.clear();
        dim_original_ = 0;
        rho_w1_ = 0;
        rho_int_ = 0;
        return;
    }
    if (weights.size() != cluster_labels.size()) {
        throw std::invalid_argument("weights / cluster_labels size mismatch");
    }

    const uint32_t expected_len = static_cast<uint32_t>(weights.front().size());
    if (expected_len == 0u || (expected_len % 2u) != 0u) {
        throw std::invalid_argument("Weight length must be even (2*dim_original).");
    }
    for (const auto& w : weights) {
        if (w.size() != expected_len) {
            throw std::invalid_argument("All weight vectors must have the same length.");
        }
    }
    const uint32_t inferred_dim = expected_len / 2u;
    if (dim_original_ != 0u && inferred_dim != dim_original_) {
        throw std::invalid_argument("Weight dimensionality mismatch with model state.");
    }

    weights_ = weights;
    w_count_cache_.assign(weights_.size(), 0u);
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        w_count_cache_[i] = ones_count(weights_[i]);
    }
    cluster_map_.clear();
    for (std::size_t i = 0; i < cluster_labels.size(); ++i) {
        cluster_map_[static_cast<uint32_t>(i)] = static_cast<uint32_t>(cluster_labels[i]);
    }

    dim_original_ = inferred_dim;
    rho_w1_ = static_cast<uint32_t>(std::ceil(params_.rho * static_cast<double>(dim_original_)));
    rho_int_ = rho_w1_;
}

void BinaryFuzzyARTMAPCore::fit(
    const int* x,
    std::size_t rows,
    std::size_t cols,
    const int* y,
    std::size_t y_len
) {
    if (y == nullptr || rows != y_len) {
        throw std::invalid_argument("X/y size mismatch");
    }
    validate_x(x, rows, cols);
    if (dim_original_ == 0u) {
        dim_original_ = static_cast<uint32_t>(cols / 2u);
        rho_w1_ = static_cast<uint32_t>(std::ceil(params_.rho * static_cast<double>(dim_original_)));
        rho_int_ = rho_w1_;
    }

    labels_a_.assign(rows, 0);
    std::vector<uint32_t> sample(cols, 0u);
    for (std::size_t i = 0; i < rows; ++i) {
        const int* row = x + i * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            sample[j] = (row[j] != 0) ? 1u : 0u;
        }
        labels_a_[i] = static_cast<int>(step_fit(sample, static_cast<uint32_t>(y[i])));
    }
}

std::pair<std::vector<int>, std::vector<int>> BinaryFuzzyARTMAPCore::predict(
    const int* x, std::size_t rows, std::size_t cols
) {
    if (weights_.empty()) {
        throw std::runtime_error("Model has no clusters");
    }
    validate_x(x, rows, cols);

    std::vector<int> pred_a(rows), pred_b(rows);
    std::vector<uint32_t> sample(cols, 0u);
    std::vector<fracsort::Item<uint32_t>> items(weights_.size());

    for (std::size_t i = 0; i < rows; ++i) {
        const int* row = x + i * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            sample[j] = (row[j] != 0) ? 1u : 0u;
        }
        for (std::size_t c = 0; c < weights_.size(); ++c) {
            const uint32_t iw_count = intersection_count(sample, weights_[c]);
            const uint32_t den = std::max<uint32_t>(1u, w_count_cache_[c]);
            items[c] = fracsort::Item<uint32_t>{iw_count, den, 0, 1, c};
        }
        const std::size_t best_cluster =
            fracsort::fracargmax_items<uint32_t>(items.data(), items.size());
        pred_a[i] = static_cast<int>(best_cluster);
        pred_b[i] = static_cast<int>(cluster_map_.at(static_cast<uint32_t>(best_cluster)));
    }
    return {pred_a, pred_b};
}

const std::vector<int>& BinaryFuzzyARTMAPCore::labels_a() const { return labels_a_; }
const std::vector<std::vector<uint32_t>>& BinaryFuzzyARTMAPCore::weights() const { return weights_; }

std::vector<int> BinaryFuzzyARTMAPCore::cluster_labels() const {
    std::vector<int> out(weights_.size(), 0);
    for (const auto& kv : cluster_map_) {
        if (kv.first < out.size()) out[kv.first] = static_cast<int>(kv.second);
    }
    return out;
}

void BinaryFuzzyARTMAPCore::reset_rho() { rho_int_ = rho_w1_; }

uint32_t BinaryFuzzyARTMAPCore::intersection_count(
    const std::vector<uint32_t>& i, const std::vector<uint32_t>& w
) {
    uint32_t c = 0u;
    for (std::size_t j = 0; j < i.size(); ++j) c += (i[j] & w[j]);
    return c;
}

uint32_t BinaryFuzzyARTMAPCore::ones_count(const std::vector<uint32_t>& v) {
    uint32_t c = 0u;
    for (uint32_t x : v) c += x;
    return c;
}

bool BinaryFuzzyARTMAPCore::validate_hypothesis(uint32_t cluster_id, uint32_t c_b) const {
    auto it = cluster_map_.find(cluster_id);
    return (it == cluster_map_.end()) || (it->second == c_b);
}

bool BinaryFuzzyARTMAPCore::match_operator(const std::string& mt, uint32_t a, uint32_t b) {
    if (mt == "MT+" || mt == "MT-" || mt == "MT1") return a >= b;
    if (mt == "MT0" || mt == "MT~") return a > b;
    throw std::invalid_argument("Invalid Match Tracking Method: " + mt);
}

bool BinaryFuzzyARTMAPCore::match_tracking_integer(uint32_t m_int) {
    if (params_.mt == "MT+") {
        rho_int_ = m_int + params_.epsilon;
    } else if (params_.mt == "MT-") {
        rho_int_ = (m_int > params_.epsilon) ? (m_int - params_.epsilon) : 0u;
    } else if (params_.mt == "MT0") {
        rho_int_ = m_int;
    } else if (params_.mt == "MT1") {
        rho_int_ = dim_original_ + 1u;
    } else if (params_.mt == "MT~") {
        // no-op
    } else {
        throw std::invalid_argument("Invalid Match Tracking Method: " + params_.mt);
    }
    if (params_.mt == "MT1" || rho_int_ > dim_original_) return false;
    return true;
}

void BinaryFuzzyARTMAPCore::update_inplace(std::vector<uint32_t>& w, const std::vector<uint32_t>& i) {
    for (std::size_t j = 0; j < w.size(); ++j) w[j] = (w[j] & i[j]);
}

uint32_t BinaryFuzzyARTMAPCore::step_fit(const std::vector<uint32_t>& sample, uint32_t c_b) {
    reset_rho();
    if (weights_.empty()) {
        weights_.push_back(sample);
        w_count_cache_.push_back(ones_count(sample));
        cluster_map_[0u] = c_b;
        return 0u;
    }

    const std::size_t n_clusters = weights_.size();
    std::vector<fracsort::Item<uint32_t>> items;
    items.reserve(n_clusters);
    std::vector<uint32_t> iw_counts(n_clusters, 0u);

    for (std::size_t c = 0; c < n_clusters; ++c) {
        const uint32_t iw = intersection_count(sample, weights_[c]);
        iw_counts[c] = iw;
        bool mt_status = true;
        if (params_.mt != "MT-") mt_status = match_operator(params_.mt, iw, rho_int_);
        if (!mt_status) continue;
        const uint32_t den = std::max<uint32_t>(1u, w_count_cache_[c]);
        items.push_back(fracsort::Item<uint32_t>{iw, den, 0, 1, c});
    }

    if (!items.empty()) fracsort::argsort_items_inplace<uint32_t>(items.data(), items.size());

    for (const auto& it : items) {
        const uint32_t idx = static_cast<uint32_t>(it.idx);
        const uint32_t iw_count = iw_counts[it.idx];
        if (!match_operator(params_.mt, iw_count, rho_int_)) continue;

        if (validate_hypothesis(idx, c_b)) {
            update_inplace(weights_[idx], sample);
            w_count_cache_[idx] = iw_count;
            cluster_map_[idx] = c_b;
            return idx;
        }

        const bool keep_searching = match_tracking_integer(iw_count);
        if (!keep_searching) break;
    }

    const uint32_t new_cluster_id = static_cast<uint32_t>(weights_.size());
    weights_.push_back(sample);
    w_count_cache_.push_back(ones_count(sample));
    cluster_map_[new_cluster_id] = c_b;
    return new_cluster_id;
}

void BinaryFuzzyARTMAPCore::validate_x(const int* x, std::size_t rows, std::size_t cols) const {
    if (x == nullptr) throw std::invalid_argument("X cannot be null");
    if (cols == 0 || (cols % 2u) != 0u) {
        throw std::invalid_argument("Number of features must be even (2*dim_original).");
    }
    if (dim_original_ != 0u && cols != 2u * dim_original_) {
        throw std::invalid_argument("Number of features do not match existing weights.");
    }
    // Rows content can be non-binary int; normalize with !=0 in fit/predict.
    (void)rows;
}

}  // namespace artlib_cpp
