#include "artlib_cpp/fuzzy_artmap_core.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace artlib_cpp {

FuzzyARTMAPCore::FuzzyARTMAPCore(FuzzyARTMAPParams params)
    : params_(std::move(params)), dim_original_(0), rho_runtime_(params_.rho) {}

void FuzzyARTMAPCore::set_state(
    const std::vector<std::vector<double>>& weights,
    const std::vector<int>& cluster_labels
) {
    if (weights.empty() && cluster_labels.empty()) {
        weights_.clear();
        weight_l1_.clear();
        cluster_map_.clear();
        dim_original_ = 0;
        rho_runtime_ = params_.rho;
        return;
    }
    if (weights.size() != cluster_labels.size()) {
        throw std::invalid_argument("weights / cluster_labels size mismatch");
    }

    const std::size_t expected_len = weights.front().size();
    if (expected_len == 0 || (expected_len % 2) != 0) {
        throw std::invalid_argument("weight length must be even (2*dim)");
    }
    for (const auto& w : weights) {
        if (w.size() != expected_len) {
            throw std::invalid_argument("All weight vectors must have the same length.");
        }
    }

    const int inferred_dim = static_cast<int>(expected_len / 2);
    if (dim_original_ != 0 && inferred_dim != dim_original_) {
        throw std::invalid_argument("Weight dimensionality mismatch with model state");
    }

    weights_ = weights;
    weight_l1_.assign(weights_.size(), 0.0);
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weight_l1_[i] = std::accumulate(weights_[i].begin(), weights_[i].end(), 0.0);
    }
    cluster_map_.clear();
    for (std::size_t i = 0; i < cluster_labels.size(); ++i) {
        cluster_map_[static_cast<int>(i)] = cluster_labels[i];
    }
    dim_original_ = inferred_dim;
}

void FuzzyARTMAPCore::fit(
    const double* x,
    std::size_t rows,
    std::size_t cols,
    const int* y,
    std::size_t y_len
) {
    if (y == nullptr || rows != y_len) {
        throw std::invalid_argument("X/y size mismatch");
    }
    validate_x(x, rows, cols);
    if (dim_original_ == 0) {
        dim_original_ = static_cast<int>(cols / 2);
    }
    labels_a_.assign(rows, 0);

    for (std::size_t i = 0; i < rows; ++i) {
        const double* row = x + i * cols;
        labels_a_[i] = step_fit(row, cols, y[i]);
    }
}

std::pair<std::vector<int>, std::vector<int>> FuzzyARTMAPCore::predict(
    const double* x, std::size_t rows, std::size_t cols
) {
    validate_x(x, rows, cols);
    if (weights_.empty()) {
        throw std::runtime_error("Model has no clusters");
    }

    std::vector<int> y_a(rows), y_b(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        const double* row = x + i * cols;
        int best_id = -1;
        double best_t = -1.0;

        for (std::size_t c = 0; c < weights_.size(); ++c) {
            const double t = category_choice(row, weights_[c], weight_l1_[c]);
            if (t > best_t) {
                best_t = t;
                best_id = static_cast<int>(c);
            }
        }
        if (best_id < 0) best_id = 0;
        y_a[i] = best_id;
        y_b[i] = cluster_map_.at(best_id);
    }
    return {y_a, y_b};
}

const std::vector<int>& FuzzyARTMAPCore::labels_a() const { return labels_a_; }
const std::vector<std::vector<double>>& FuzzyARTMAPCore::weights() const { return weights_; }

std::vector<int> FuzzyARTMAPCore::cluster_labels() const {
    std::vector<int> out(weights_.size(), 0);
    for (const auto& kv : cluster_map_) {
        out[kv.first] = kv.second;
    }
    return out;
}

void FuzzyARTMAPCore::reset_rho() { rho_runtime_ = params_.rho; }

double FuzzyARTMAPCore::l1_and(const double* x, const std::vector<double>& w, int len) {
    double s = 0.0;
    for (int j = 0; j < len; ++j) s += std::min(x[j], w[j]);
    return s;
}

double FuzzyARTMAPCore::category_choice(
    const double* sample, const std::vector<double>& w, double w_l1
) const {
    const int len = static_cast<int>(w.size());
    const double num = l1_and(sample, w, len);
    const double denom = params_.alpha + w_l1;
    return num / denom;
}

double FuzzyARTMAPCore::match(const double* sample, const std::vector<double>& w) const {
    const int len = static_cast<int>(w.size());
    const double num = l1_and(sample, w, len);
    return num / static_cast<double>(dim_original_);
}

void FuzzyARTMAPCore::update_weight_inplace(
    const double* sample,
    std::size_t sample_len,
    std::vector<double>& w,
    double& w_l1
) const {
    w_l1 = 0.0;
    for (std::size_t j = 0; j < sample_len; ++j) {
        const double updated =
            params_.beta * std::min(sample[j], w[j]) + (1.0 - params_.beta) * w[j];
        w[j] = updated;
        w_l1 += updated;
    }
}

bool FuzzyARTMAPCore::match_tracking(double m) {
    if (params_.mt == "MT+") rho_runtime_ = m + params_.epsilon;
    else if (params_.mt == "MT-") rho_runtime_ = m - params_.epsilon;
    else if (params_.mt == "MT0") rho_runtime_ = m;
    else if (params_.mt == "MT1") rho_runtime_ = std::numeric_limits<double>::infinity();
    else if (params_.mt == "MT~") { /* no-op */ }
    else throw std::invalid_argument("Invalid MT mode: " + params_.mt);

    return !(params_.mt == "MT1" || rho_runtime_ > 1.0);
}

bool FuzzyARTMAPCore::matches_vigilance(double m) const {
    if (params_.mt == "MT+" || params_.mt == "MT-" || params_.mt == "MT1") {
        return m >= rho_runtime_;
    }
    if (params_.mt == "MT0" || params_.mt == "MT~") {
        return m > rho_runtime_;
    }
    throw std::invalid_argument("Invalid MT mode");
}

int FuzzyARTMAPCore::step_fit(const double* sample, std::size_t sample_len, int c_b) {
    reset_rho();
    if (weights_.empty()) {
        weights_.emplace_back(sample, sample + sample_len);
        weight_l1_.push_back(std::accumulate(sample, sample + sample_len, 0.0));
        cluster_map_[0] = c_b;
        return 0;
    }

    const std::size_t k = weights_.size();
    std::vector<double> t(k), m(k);
    for (std::size_t i = 0; i < k; ++i) {
        t[i] = category_choice(sample, weights_[i], weight_l1_[i]);
        m[i] = match(sample, weights_[i]);
    }

    std::vector<int> order;
    order.reserve(k);
    for (std::size_t i = 0; i < k; ++i) order.push_back(static_cast<int>(i));
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (t[a] != t[b]) return t[a] > t[b];
        return a < b;
    });

    for (const int best : order) {
        if (!matches_vigilance(m[best])) continue;

        if (cluster_map_.count(best) && cluster_map_[best] != c_b) {
            if (!match_tracking(m[best])) break;
            continue;
        }

        update_weight_inplace(sample, sample_len, weights_[best], weight_l1_[best]);
        cluster_map_[best] = c_b;
        return best;
    }

    const int new_id = static_cast<int>(weights_.size());
    weights_.emplace_back(sample, sample + sample_len);
    weight_l1_.push_back(std::accumulate(sample, sample + sample_len, 0.0));
    cluster_map_[new_id] = c_b;
    return new_id;
}

void FuzzyARTMAPCore::validate_x(const double* x, std::size_t /*rows*/, std::size_t cols) const {
    if (x == nullptr) throw std::invalid_argument("X cannot be null");
    if (cols == 0 || (cols % 2) != 0) {
        throw std::invalid_argument("Number of features must be even (2*dim_original).");
    }
    if (dim_original_ != 0 && static_cast<int>(cols) != 2 * dim_original_) {
        throw std::invalid_argument("Number of features do not match existing weights.");
    }
}

}  // namespace artlib_cpp
