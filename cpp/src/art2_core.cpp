#include "artlib_cpp/art2_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace artlib_cpp {

ART2Core::ART2Core(ART2Params params) : params_(std::move(params)), dim_(0) {
    if (params_.rho < 0.0 || params_.rho > 1.0) throw std::invalid_argument("rho must be in [0,1]");
    if (params_.alpha < 0.0 || params_.alpha > 1.0) throw std::invalid_argument("alpha must be in [0,1]");
    if (params_.beta < 0.0 || params_.beta > 1.0) throw std::invalid_argument("beta must be in [0,1]");
}

void ART2Core::set_weights(const std::vector<std::vector<double>>& weights) {
    if (weights.empty()) {
        weights_.clear();
        dim_ = 0;
        return;
    }

    const std::size_t expected_len = weights.front().size();
    if (expected_len == 0) {
        throw std::invalid_argument("invalid weight length");
    }
    for (const auto& w : weights) {
        if (w.size() != expected_len) throw std::invalid_argument("inconsistent weight dimensions");
    }

    const int inferred_dim = static_cast<int>(expected_len);
    if (dim_ != 0 && inferred_dim != dim_) {
        throw std::invalid_argument("weight dimensionality mismatch with model state");
    }
    if (params_.alpha > 1.0 / std::sqrt(static_cast<double>(inferred_dim))) {
        throw std::invalid_argument("alpha must be <= 1/sqrt(dim)");
    }

    weights_ = weights;
    dim_ = inferred_dim;
}

void ART2Core::fit(const double* x, std::size_t rows, std::size_t cols) {
    validate_x(x, rows, cols);
    if (dim_ == 0) dim_ = static_cast<int>(cols);
    if (params_.alpha > 1.0 / std::sqrt(static_cast<double>(dim_))) {
        throw std::invalid_argument("alpha must be <= 1/sqrt(dim)");
    }
    labels_.assign(rows, 0);

    for (std::size_t i = 0; i < rows; ++i) {
        std::vector<double> sample(x + i * cols, x + (i + 1) * cols);
        labels_[i] = step_fit(sample);
    }
}

std::vector<int> ART2Core::predict(const double* x, std::size_t rows, std::size_t cols) const {
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

const std::vector<int>& ART2Core::labels() const { return labels_; }
const std::vector<std::vector<double>>& ART2Core::weights() const { return weights_; }

double ART2Core::category_choice(const double* sample, const std::vector<double>& w) const {
    double act = 0.0;
    for (int j = 0; j < dim_; ++j) {
        act += sample[j] * w[j];
    }
    return act;
}

double ART2Core::match(const double* sample, double activation) const {
    double m_u = 0.0;
    for (int j = 0; j < dim_; ++j) m_u += sample[j];
    m_u *= params_.alpha;
    if (activation < m_u) return -1.0;
    return activation;
}

std::vector<double> ART2Core::update_weight(
    const std::vector<double>& i, const std::vector<double>& w
) const {
    std::vector<double> out(w.size(), 0.0);
    for (int j = 0; j < dim_; ++j) {
        out[j] = params_.beta * i[j] + (1.0 - params_.beta) * w[j];
    }
    return out;
}

std::vector<double> ART2Core::new_weight(const std::vector<double>& i) const {
    return i;
}

int ART2Core::step_fit(const std::vector<double>& sample) {
    if (weights_.empty()) {
        weights_.push_back(new_weight(sample));
        return 0;
    }

    const std::size_t k = weights_.size();
    std::vector<double> t(k), m(k);
    for (std::size_t i = 0; i < k; ++i) {
        t[i] = category_choice(sample.data(), weights_[i]);
        m[i] = match(sample.data(), t[i]);
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
            weights_[best] = update_weight(sample, weights_[best]);
            return best;
        }
    }

    const int new_id = static_cast<int>(weights_.size());
    weights_.push_back(new_weight(sample));
    return new_id;
}

void ART2Core::validate_x(const double* x, std::size_t /*rows*/, std::size_t cols) const {
    if (x == nullptr) throw std::invalid_argument("X cannot be null");
    if (cols == 0) throw std::invalid_argument("X must have at least one feature");
    if (dim_ != 0 && static_cast<int>(cols) != dim_) {
        throw std::invalid_argument("feature dimension mismatch");
    }
}

}  // namespace artlib_cpp
