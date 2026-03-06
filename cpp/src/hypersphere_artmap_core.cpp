#include "artlib_cpp/hypersphere_artmap_core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace artlib_cpp {

HypersphereARTMAPCore::HypersphereARTMAPCore(HypersphereARTMAPParams params)
    : params_(std::move(params)), dim_(0), rho_runtime_(params_.rho) {
    if (params_.r_hat <= 0.0) throw std::invalid_argument("'r_hat' must be > 0");
}

void HypersphereARTMAPCore::set_state(
    const std::vector<std::vector<double>>& weights,
    const std::vector<int>& cluster_labels
) {
    if (weights.empty() && cluster_labels.empty()) {
        weights_.clear();
        cluster_map_.clear();
        dim_ = 0;
        rho_runtime_ = params_.rho;
        return;
    }
    if (weights.size() != cluster_labels.size()) {
        throw std::invalid_argument("weights / cluster_labels size mismatch");
    }
    const std::size_t expected_len = weights.front().size();
    if (expected_len < 2) {
        throw std::invalid_argument("weight must be at least 2 elements (d >= 1)");
    }
    for (const auto& w : weights) {
        if (w.size() != expected_len) throw std::invalid_argument("inconsistent weight dimensions");
    }
    const int inferred_dim = static_cast<int>(expected_len - 1);
    if (dim_ != 0 && inferred_dim != dim_) {
        throw std::invalid_argument("weight dimensionality mismatch with model state");
    }

    weights_ = weights;
    cluster_map_.clear();
    for (std::size_t i = 0; i < cluster_labels.size(); ++i) {
        cluster_map_[static_cast<int>(i)] = cluster_labels[i];
    }
    dim_ = inferred_dim;
}

void HypersphereARTMAPCore::fit(
    const double* x, std::size_t rows, std::size_t cols, const int* y, std::size_t y_len
) {
    if (y == nullptr || y_len != rows) throw std::invalid_argument("X/y size mismatch");
    validate_x(x, rows, cols);
    if (dim_ == 0) dim_ = static_cast<int>(cols);
    labels_a_.assign(rows, 0);

    for (std::size_t i = 0; i < rows; ++i) {
        std::vector<double> sample(x + i * cols, x + (i + 1) * cols);
        labels_a_[i] = step_fit(sample, y[i]);
    }
}

std::pair<std::vector<int>, std::vector<int>> HypersphereARTMAPCore::predict(
    const double* x, std::size_t rows, std::size_t cols
) {
    if (weights_.empty()) throw std::runtime_error("Model has no clusters");
    validate_x(x, rows, cols);

    std::vector<int> y_a(rows), y_b(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        const double* row = x + i * cols;
        int best_id = -1;
        double best_t = -std::numeric_limits<double>::infinity();
        for (std::size_t c = 0; c < weights_.size(); ++c) {
            const double t = category_choice(row, weights_[c]);
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

const std::vector<int>& HypersphereARTMAPCore::labels_a() const { return labels_a_; }

const std::vector<std::vector<double>>& HypersphereARTMAPCore::weights() const {
    return weights_;
}

std::vector<int> HypersphereARTMAPCore::cluster_labels() const {
    std::vector<int> out(weights_.size(), 0);
    for (const auto& kv : cluster_map_) out[kv.first] = kv.second;
    return out;
}

void HypersphereARTMAPCore::reset_rho() { rho_runtime_ = params_.rho; }

double HypersphereARTMAPCore::euclidean(const double* x, const std::vector<double>& w) const {
    double s = 0.0;
    for (int j = 0; j < dim_; ++j) {
        const double d = x[j] - w[j];
        s += d * d;
    }
    return std::sqrt(s);
}

double HypersphereARTMAPCore::category_choice(
    const double* sample, const std::vector<double>& w
) const {
    const double radius = w.back();
    const double i_rad = euclidean(sample, w);
    const double max_r = std::max(radius, i_rad);
    return (params_.r_hat - max_r) / (params_.r_hat - radius + params_.alpha);
}

double HypersphereARTMAPCore::match(const double* sample, const std::vector<double>& w) const {
    const double radius = w.back();
    const double i_rad = euclidean(sample, w);
    const double max_r = std::max(radius, i_rad);
    return 1.0 - (max_r / params_.r_hat);
}

std::vector<double> HypersphereARTMAPCore::update_weight(
    const std::vector<double>& i,
    const std::vector<double>& w,
    const std::vector<double>& cache
) const {
    const double i_radius = cache[0];
    const double max_r = cache[1];
    const double radius = w.back();

    const double radius_new = radius + (params_.beta / 2.0) * (max_r - radius);

    std::vector<double> out(static_cast<std::size_t>(dim_) + 1, 0.0);
    for (int j = 0; j < dim_; ++j) {
        const double centroid = w[j];
        out[j] = centroid + (params_.beta / 2.0) * (i[j] - centroid)
                 * (1.0 - (std::min(radius, i_radius) / (i_radius + params_.alpha)));
    }
    out.back() = radius_new;
    return out;
}

std::vector<double> HypersphereARTMAPCore::new_weight(const std::vector<double>& i) const {
    std::vector<double> w(i);
    w.push_back(0.0);
    return w;
}

bool HypersphereARTMAPCore::match_tracking(double m) {
    if (params_.mt == "MT+") rho_runtime_ = m + params_.epsilon;
    else if (params_.mt == "MT-") rho_runtime_ = m - params_.epsilon;
    else if (params_.mt == "MT0") rho_runtime_ = m;
    else if (params_.mt == "MT1") rho_runtime_ = std::numeric_limits<double>::infinity();
    else if (params_.mt == "MT~") {/* no-op */}
    else throw std::invalid_argument("Invalid MT mode: " + params_.mt);
    return !(params_.mt == "MT1" || rho_runtime_ > 1.0);
}

bool HypersphereARTMAPCore::matches_vigilance(double m) const {
    if (params_.mt == "MT+" || params_.mt == "MT-" || params_.mt == "MT1")
        return m >= rho_runtime_;
    if (params_.mt == "MT0" || params_.mt == "MT~")
        return m > rho_runtime_;
    throw std::invalid_argument("Invalid MT mode");
}

int HypersphereARTMAPCore::step_fit(const std::vector<double>& sample, int c_b) {
    reset_rho();
    if (weights_.empty()) {
        weights_.push_back(new_weight(sample));
        cluster_map_[0] = c_b;
        return 0;
    }

    const std::size_t k = weights_.size();
    std::vector<double> t(k), m(k);
    std::vector<std::array<double, 2>> caches(k);

    for (std::size_t i = 0; i < k; ++i) {
        const double radius_w = weights_[i].back();
        const double i_rad = euclidean(sample.data(), weights_[i]);
        const double max_r = std::max(radius_w, i_rad);
        caches[i] = {i_rad, max_r};

        t[i] = (params_.r_hat - max_r) / (params_.r_hat - radius_w + params_.alpha);
        m[i] = 1.0 - (max_r / params_.r_hat);
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

        weights_[best] = update_weight(sample, weights_[best], {caches[best][0], caches[best][1]});
        cluster_map_[best] = c_b;
        return best;
    }

    const int new_id = static_cast<int>(weights_.size());
    weights_.push_back(new_weight(sample));
    cluster_map_[new_id] = c_b;
    return new_id;
}

void HypersphereARTMAPCore::validate_x(const double* x, std::size_t /*rows*/, std::size_t cols) const {
    if (x == nullptr) throw std::invalid_argument("X cannot be null");
    if (cols == 0) throw std::invalid_argument("X must have at least one feature");
    if (dim_ != 0 && static_cast<int>(cols) != dim_) {
        throw std::invalid_argument("feature dimension mismatch");
    }
}

}  // namespace artlib_cpp
