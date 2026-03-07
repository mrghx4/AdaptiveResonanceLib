#include "artlib_cpp/ellipsoid_art_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace artlib_cpp {

EllipsoidARTCore::EllipsoidARTCore(EllipsoidARTParams params)
    : params_(std::move(params)), dim_(0) {
    if (params_.rho < 0.0 || params_.rho > 1.0) throw std::invalid_argument("rho must be in [0,1]");
    if (params_.alpha < 0.0 || params_.alpha > 1.0) throw std::invalid_argument("alpha must be in [0,1]");
    if (params_.beta < 0.0 || params_.beta > 1.0) throw std::invalid_argument("beta must be in [0,1]");
    if (params_.mu <= 0.0 || params_.mu > 1.0) throw std::invalid_argument("mu must be in (0,1]");
    if (params_.r_hat <= 0.0) throw std::invalid_argument("r_hat must be > 0");
}

void EllipsoidARTCore::set_weights(const std::vector<std::vector<double>>& weights) {
    if (weights.empty()) {
        weights_.clear();
        dim_ = 0;
        return;
    }

    const std::size_t expected_len = weights.front().size();
    if (expected_len < 3 || ((expected_len - 1) % 2) != 0) {
        throw std::invalid_argument("invalid weight length");
    }
    for (const auto& w : weights) {
        if (w.size() != expected_len) throw std::invalid_argument("inconsistent weight dimensions");
    }

    const int inferred_dim = static_cast<int>((expected_len - 1) / 2);
    if (dim_ != 0 && inferred_dim != dim_) {
        throw std::invalid_argument("weight dimensionality mismatch with model state");
    }

    weights_ = weights;
    dim_ = inferred_dim;
}

void EllipsoidARTCore::fit(const double* x, std::size_t rows, std::size_t cols) {
    validate_x(x, rows, cols);
    if (dim_ == 0) dim_ = static_cast<int>(cols);
    labels_.assign(rows, 0);

    for (std::size_t i = 0; i < rows; ++i) {
        std::vector<double> sample(x + i * cols, x + (i + 1) * cols);
        labels_[i] = step_fit(sample);
    }
}

std::vector<int> EllipsoidARTCore::predict(const double* x, std::size_t rows, std::size_t cols) const {
    if (weights_.empty()) throw std::runtime_error("Model has no clusters");
    validate_x(x, rows, cols);

    std::vector<int> y(rows, 0);
    for (std::size_t i = 0; i < rows; ++i) {
        const double* row = x + i * cols;
        int best_id = -1;
        double best_t = -std::numeric_limits<double>::infinity();
        for (std::size_t c = 0; c < weights_.size(); ++c) {
            double dist = 0.0;
            const double t = category_choice(row, weights_[c], &dist);
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

const std::vector<int>& EllipsoidARTCore::labels() const { return labels_; }
const std::vector<std::vector<double>>& EllipsoidARTCore::weights() const { return weights_; }

double EllipsoidARTCore::l2norm2(const double* a, const double* b) const {
    double s = 0.0;
    for (int j = 0; j < dim_; ++j) {
        const double d = a[j] - b[j];
        s += d * d;
    }
    return s;
}

double EllipsoidARTCore::category_distance(const double* x, const std::vector<double>& w) const {
    const double* centroid = w.data();
    const double* major_axis = w.data() + dim_;

    const double l2 = l2norm2(x, centroid);
    double major_norm2 = 0.0;
    double dot = 0.0;
    for (int j = 0; j < dim_; ++j) {
        major_norm2 += major_axis[j] * major_axis[j];
        dot += major_axis[j] * (x[j] - centroid[j]);
    }

    if (major_norm2 > 0.0) {
        const double inner = l2 - (1.0 - params_.mu * params_.mu) * (dot * dot);
        return (1.0 / params_.mu) * std::sqrt(std::max(inner, 0.0));
    }
    return std::sqrt(std::max(l2, 0.0));
}

double EllipsoidARTCore::category_choice(
    const double* sample, const std::vector<double>& w, double* out_dist
) const {
    const double radius = w.back();
    const double dist = category_distance(sample, w);
    if (out_dist != nullptr) *out_dist = dist;

    return (params_.r_hat - radius - std::max(radius, dist))
           / (params_.r_hat - 2.0 * radius + params_.alpha);
}

double EllipsoidARTCore::match(const std::vector<double>& w, double dist) const {
    const double radius = w.back();
    return 1.0 - (radius + std::max(radius, dist)) / params_.r_hat;
}

std::vector<double> EllipsoidARTCore::update_weight(
    const std::vector<double>& i, const std::vector<double>& w, double dist
) const {
    const double* centroid = w.data();
    const double* major_axis = w.data() + dim_;
    const double radius = w.back();

    const double radius_new = radius + (params_.beta / 2.0) * (std::max(radius, dist) - radius);

    std::vector<double> centroid_new(dim_, 0.0);
    double ratio;
    if (dist <= 1e-12) {
        ratio = (radius > 0.0) ? 1.0 : 0.0;
    } else {
        ratio = std::min(radius, dist) / dist;
    }
    const double factor = 1.0 - ratio;
    for (int j = 0; j < dim_; ++j) {
        centroid_new[j] = centroid[j] + (params_.beta / 2.0) * (i[j] - centroid[j]) * factor;
    }

    std::vector<double> major_axis_new(dim_, 0.0);
    if (radius != 0.0) {
        double norm2 = 0.0;
        for (int j = 0; j < dim_; ++j) {
            const double d = i[j] - centroid_new[j];
            norm2 += d * d;
        }
        const double denom = std::sqrt(std::max(norm2, 1e-24));
        for (int j = 0; j < dim_; ++j) {
            major_axis_new[j] = (i[j] - centroid_new[j]) / denom;
        }
    } else {
        for (int j = 0; j < dim_; ++j) major_axis_new[j] = major_axis[j];
    }

    std::vector<double> out;
    out.reserve(2 * dim_ + 1);
    out.insert(out.end(), centroid_new.begin(), centroid_new.end());
    out.insert(out.end(), major_axis_new.begin(), major_axis_new.end());
    out.push_back(radius_new);
    return out;
}

std::vector<double> EllipsoidARTCore::new_weight(const std::vector<double>& i) const {
    std::vector<double> out;
    out.reserve(2 * dim_ + 1);
    out.insert(out.end(), i.begin(), i.end());
    out.insert(out.end(), static_cast<std::size_t>(dim_), 0.0);
    out.push_back(0.0);
    return out;
}

int EllipsoidARTCore::step_fit(const std::vector<double>& sample) {
    if (weights_.empty()) {
        weights_.push_back(new_weight(sample));
        return 0;
    }

    const std::size_t k = weights_.size();
    std::vector<double> t(k), m(k), d(k);
    for (std::size_t i = 0; i < k; ++i) {
        t[i] = category_choice(sample.data(), weights_[i], &d[i]);
        m[i] = match(weights_[i], d[i]);
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
            weights_[best] = update_weight(sample, weights_[best], d[best]);
            return best;
        }
    }

    const int new_id = static_cast<int>(weights_.size());
    weights_.push_back(new_weight(sample));
    return new_id;
}

void EllipsoidARTCore::validate_x(const double* x, std::size_t /*rows*/, std::size_t cols) const {
    if (x == nullptr) throw std::invalid_argument("X cannot be null");
    if (cols == 0) throw std::invalid_argument("X must have at least one feature");
    if (dim_ != 0 && static_cast<int>(cols) != dim_) {
        throw std::invalid_argument("feature dimension mismatch");
    }
}

}  // namespace artlib_cpp
