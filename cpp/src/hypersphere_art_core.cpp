#include "artlib_cpp/hypersphere_art_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace artlib_cpp {

HypersphereARTCore::HypersphereARTCore(HypersphereARTParams params)
    : params_(std::move(params)), dim_(0) {
    if (params_.rho < 0.0 || params_.rho > 1.0) {
        throw std::invalid_argument("rho must be in [0,1]");
    }
    if (params_.alpha < 0.0) {
        throw std::invalid_argument("alpha must be >= 0");
    }
    if (params_.beta < 0.0 || params_.beta > 1.0) {
        throw std::invalid_argument("beta must be in [0,1]");
    }
    if (params_.r_hat <= 0.0) {
        throw std::invalid_argument("r_hat must be > 0");
    }
}

void HypersphereARTCore::set_weights(const std::vector<std::vector<double>>& weights) {
    if (weights.empty()) {
        weights_.clear();
        dim_ = 0;
        return;
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
    dim_ = inferred_dim;
}

void HypersphereARTCore::fit(const double* x, std::size_t rows, std::size_t cols) {
    validate_x(x, rows, cols);
    if (dim_ == 0) dim_ = static_cast<int>(cols);
    labels_.assign(rows, 0);

    for (std::size_t i = 0; i < rows; ++i) {
        std::vector<double> sample(x + i * cols, x + (i + 1) * cols);
        labels_[i] = step_fit(sample);
    }
}

std::vector<int> HypersphereARTCore::predict(const double* x, std::size_t rows, std::size_t cols) const {
    if (weights_.empty()) throw std::runtime_error("Model has no clusters");
    validate_x(x, rows, cols);

    std::vector<int> y(rows, 0);
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
        y[i] = best_id;
    }
    return y;
}

const std::vector<int>& HypersphereARTCore::labels() const { return labels_; }
const std::vector<std::vector<double>>& HypersphereARTCore::weights() const { return weights_; }

double HypersphereARTCore::euclidean(const double* x, const std::vector<double>& w) const {
    double s = 0.0;
    for (int j = 0; j < dim_; ++j) {
        const double d = x[j] - w[j];
        s += d * d;
    }
    return std::sqrt(s);
}

double HypersphereARTCore::category_choice(const double* sample, const std::vector<double>& w) const {
    const double radius = w.back();
    const double i_radius = euclidean(sample, w);
    const double max_radius = std::max(radius, i_radius);
    return (params_.r_hat - max_radius) / (params_.r_hat - radius + params_.alpha);
}

double HypersphereARTCore::match(const double* sample, const std::vector<double>& w) const {
    const double radius = w.back();
    const double i_radius = euclidean(sample, w);
    const double max_radius = std::max(radius, i_radius);
    return 1.0 - (max_radius / params_.r_hat);
}

std::vector<double> HypersphereARTCore::update_weight(
    const std::vector<double>& i,
    const std::vector<double>& w,
    double i_radius,
    double max_radius
) const {
    const double radius = w.back();
    const double radius_new = radius + (params_.beta / 2.0) * (max_radius - radius);

    std::vector<double> out(static_cast<std::size_t>(dim_) + 1, 0.0);
    for (int j = 0; j < dim_; ++j) {
        const double centroid = w[j];
        out[j] = centroid + (params_.beta / 2.0) * (i[j] - centroid)
                 * (1.0 - (std::min(radius, i_radius) / (i_radius + params_.alpha)));
    }
    out.back() = radius_new;
    return out;
}

std::vector<double> HypersphereARTCore::new_weight(const std::vector<double>& i) const {
    std::vector<double> w(i);
    w.push_back(0.0);
    return w;
}

int HypersphereARTCore::step_fit(const std::vector<double>& sample) {
    if (weights_.empty()) {
        weights_.push_back(new_weight(sample));
        return 0;
    }

    const std::size_t k = weights_.size();
    std::vector<double> t(k), m(k), i_radius(k), max_radius(k);
    for (std::size_t i = 0; i < k; ++i) {
        const double radius = weights_[i].back();
        i_radius[i] = euclidean(sample.data(), weights_[i]);
        max_radius[i] = std::max(radius, i_radius[i]);
        t[i] = (params_.r_hat - max_radius[i]) / (params_.r_hat - radius + params_.alpha);
        m[i] = 1.0 - (max_radius[i] / params_.r_hat);
    }

    std::vector<int> order;
    order.reserve(k);
    for (std::size_t i = 0; i < k; ++i) order.push_back(static_cast<int>(i));
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (t[a] != t[b]) return t[a] > t[b];
        return a < b;
    });

    for (const int best : order) {
        if (m[best] >= params_.rho) {
            weights_[best] = update_weight(sample, weights_[best], i_radius[best], max_radius[best]);
            return best;
        }
    }

    const int new_id = static_cast<int>(weights_.size());
    weights_.push_back(new_weight(sample));
    return new_id;
}

void HypersphereARTCore::validate_x(const double* x, std::size_t /*rows*/, std::size_t cols) const {
    if (x == nullptr) throw std::invalid_argument("X cannot be null");
    if (cols == 0) throw std::invalid_argument("X must have at least one feature");
    if (dim_ != 0 && static_cast<int>(cols) != dim_) {
        throw std::invalid_argument("feature dimension mismatch");
    }
}

}  // namespace artlib_cpp
