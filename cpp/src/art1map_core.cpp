#include "artlib_cpp/art1map_core.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace artlib_cpp {

ART1MAPCore::ART1MAPCore(ART1MAPParams params)
    : params_(std::move(params)), dim_(0), rho_runtime_(params_.rho) {
    if (params_.rho < 0.0 || params_.rho > 1.0) {
        throw std::invalid_argument("rho must be in [0,1]");
    }
    if (params_.L < 1.0) {
        throw std::invalid_argument("L must be >= 1");
    }
}

void ART1MAPCore::set_state(
    const std::vector<std::vector<double>>& weights,
    const std::vector<int>& cluster_labels
) {
    if (weights.empty() && cluster_labels.empty()) {
        weights_.clear();
        cluster_map_.clear();
        dim_ = 0;
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
            throw std::invalid_argument("inconsistent weight dimensions");
        }
    }

    if (dim_ != 0 && expected_len != 2 * dim_) {
        throw std::invalid_argument("weight dimensionality mismatch with model state");
    }

    weights_ = weights;
    dim_ = expected_len / 2;
    cluster_map_.clear();
    for (std::size_t i = 0; i < cluster_labels.size(); ++i) {
        cluster_map_[static_cast<int>(i)] = cluster_labels[i];
    }
}

void ART1MAPCore::fit(
    const std::int16_t* x,
    std::size_t rows,
    std::size_t cols,
    const int* y,
    std::size_t y_len
) {
    validate_xy(x, rows, cols, y, y_len);
    if (dim_ == 0) {
        dim_ = cols;
    }
    labels_a_.assign(rows, 0);

    for (std::size_t i = 0; i < rows; ++i) {
        const auto* row = x + i * cols;
        labels_a_[i] = step_fit(row, y[i]);
    }
}

std::pair<std::vector<int>, std::vector<int>> ART1MAPCore::predict(
    const std::int16_t* x, std::size_t rows, std::size_t cols
) {
    validate_x(x, rows, cols);
    if (weights_.empty()) {
        throw std::runtime_error("Model has no clusters");
    }

    std::vector<int> y_a(rows, 0), y_b(rows, 0);
    for (std::size_t i = 0; i < rows; ++i) {
        const auto* row = x + i * cols;
        int best_id = -1;
        double best_t = -1e300;
        for (std::size_t c = 0; c < weights_.size(); ++c) {
            const double t = category_choice(row, weights_[c]);
            if (t > best_t) {
                best_t = t;
                best_id = static_cast<int>(c);
            }
        }
        if (best_id < 0) {
            best_id = 0;
        }
        y_a[i] = best_id;
        y_b[i] = cluster_map_.at(best_id);
    }
    return {y_a, y_b};
}

const std::vector<std::vector<double>>& ART1MAPCore::weights() const {
    return weights_;
}

const std::vector<int>& ART1MAPCore::labels_a() const {
    return labels_a_;
}

std::vector<int> ART1MAPCore::cluster_labels() const {
    std::vector<int> out(weights_.size());
    for (const auto& kv : cluster_map_) {
        out[kv.first] = kv.second;
    }
    return out;
}

bool ART1MAPCore::match_tracking(double m) {
    if (params_.match_tracking == "MT+") {
        rho_runtime_ = m + params_.epsilon;
    } else if (params_.match_tracking == "MT-") {
        rho_runtime_ = m - params_.epsilon;
    } else if (params_.match_tracking == "MT0") {
        rho_runtime_ = m;
    } else if (params_.match_tracking == "MT1") {
        rho_runtime_ = std::numeric_limits<double>::infinity();
    } else if (params_.match_tracking == "MT~") {
        // no-op
    } else {
        throw std::invalid_argument("Invalid MT mode: " + params_.match_tracking);
    }
    return !(params_.match_tracking == "MT1" || rho_runtime_ > 1.0);
}

bool ART1MAPCore::matches_vigilance(double m) const {
    if (params_.match_tracking == "MT+" || params_.match_tracking == "MT-" ||
        params_.match_tracking == "MT1") {
        return m >= rho_runtime_;
    }
    if (params_.match_tracking == "MT0" || params_.match_tracking == "MT~") {
        return m > rho_runtime_;
    }
    throw std::invalid_argument("Invalid MT mode");
}

void ART1MAPCore::reset_rho() {
    rho_runtime_ = params_.rho;
}

void ART1MAPCore::validate_xy(
    const std::int16_t* x,
    std::size_t rows,
    std::size_t cols,
    const int* y,
    std::size_t y_len
) const {
    if (y == nullptr) {
        throw std::invalid_argument("y cannot be null");
    }
    if (rows != y_len) {
        throw std::invalid_argument("X/y size mismatch");
    }
    validate_x(x, rows, cols);
}

void ART1MAPCore::validate_x(
    const std::int16_t* x, std::size_t rows, std::size_t cols
) const {
    if (x == nullptr) {
        throw std::invalid_argument("X cannot be null");
    }
    if (cols == 0) {
        throw std::invalid_argument("X must have at least one feature");
    }
    if (dim_ != 0 && cols != dim_) {
        throw std::invalid_argument("X feature dimension mismatch with model");
    }
    if (!weights_.empty() && cols != weights_.front().size() / 2) {
        throw std::invalid_argument("X feature dimension mismatch with existing weights");
    }
    for (std::size_t i = 0; i < rows; ++i) {
        const auto* row = x + i * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            const auto v = row[j];
            if (!(v == 0 || v == 1)) {
                throw std::invalid_argument("ART1MAP requires binary inputs in {0,1}");
            }
        }
    }
}

int ART1MAPCore::bit(std::int16_t v) {
    return v != 0 ? 1 : 0;
}

double ART1MAPCore::category_choice(const std::int16_t* sample, const std::vector<double>& w) const {
    double s = 0.0;
    for (std::size_t j = 0; j < dim_; ++j) {
        s += static_cast<double>(bit(sample[j])) * w[j];
    }
    return s;
}

double ART1MAPCore::match(const std::int16_t* sample, const std::vector<double>& w) const {
    int count = 0;
    for (std::size_t j = 0; j < dim_; ++j) {
        const int i_bit = bit(sample[j]);
        const int td_bit = (w[dim_ + j] != 0.0) ? 1 : 0;
        if (i_bit & td_bit) {
            ++count;
        }
    }
    return static_cast<double>(count) / static_cast<double>(dim_);
}

std::vector<double> ART1MAPCore::update_weight(
    const std::int16_t* sample, const std::vector<double>& w
) const {
    std::vector<double> out(2 * dim_, 0.0);
    int count = 0;
    for (std::size_t j = 0; j < dim_; ++j) {
        const int td_new = bit(sample[j]) & ((w[dim_ + j] != 0.0) ? 1 : 0);
        out[dim_ + j] = static_cast<double>(td_new);
        if (td_new) {
            ++count;
        }
    }
    const double denom = params_.L - 1.0 + static_cast<double>(count);
    const double sf = denom > 0.0 ? params_.L / denom : 0.0;
    for (std::size_t j = 0; j < dim_; ++j) {
        out[j] = sf * out[dim_ + j];
    }
    return out;
}

std::vector<double> ART1MAPCore::new_weight(const std::int16_t* sample) const {
    std::vector<double> w(2 * dim_, 0.0);
    const double sf = params_.L / (params_.L - 1.0 + static_cast<double>(dim_));
    for (std::size_t j = 0; j < dim_; ++j) {
        const double b = static_cast<double>(bit(sample[j]));
        w[dim_ + j] = b;
        w[j] = sf * b;
    }
    return w;
}

int ART1MAPCore::step_fit(const std::int16_t* sample, int c_b) {
    reset_rho();
    if (weights_.empty()) {
        weights_.push_back(new_weight(sample));
        cluster_map_[0] = c_b;
        return 0;
    }

    const std::size_t k = weights_.size();
    std::vector<double> t(k), m(k);
    for (std::size_t i = 0; i < k; ++i) {
        t[i] = category_choice(sample, weights_[i]);
        m[i] = match(sample, weights_[i]);
    }

    std::vector<int> order;
    order.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        order.push_back(static_cast<int>(i));
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (t[a] != t[b]) {
            return t[a] > t[b];
        }
        return a < b;
    });

    for (const int best : order) {
        if (!matches_vigilance(m[best])) {
            continue;
        }
        if (cluster_map_.count(best) && cluster_map_[best] != c_b) {
            if (!match_tracking(m[best])) {
                break;
            }
            continue;
        }
        weights_[best] = update_weight(sample, weights_[best]);
        cluster_map_[best] = c_b;
        return best;
    }

    const int new_id = static_cast<int>(weights_.size());
    weights_.push_back(new_weight(sample));
    cluster_map_[new_id] = c_b;
    return new_id;
}

}  // namespace artlib_cpp
