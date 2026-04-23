#include "artlib_cpp/quadratic_neuron_art_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace artlib_cpp {

namespace {

std::size_t int_size(int value) {
    return static_cast<std::size_t>(value);
}

std::size_t square_size(int value) {
    const std::size_t n = int_size(value);
    return n * n;
}

std::size_t quadratic_weight_size(int dim) {
    const std::size_t n = int_size(dim);
    return n * n + n + 1;
}

std::size_t matrix_index(int row, int col, int cols) {
    return int_size(row) * int_size(cols) + int_size(col);
}

}  // namespace

QuadraticNeuronARTCore::QuadraticNeuronARTCore(QuadraticNeuronARTParams params)
    : params_(std::move(params)), dim_(0) {
    if (params_.rho < 0.0 || params_.rho > 1.0) {
        throw std::invalid_argument("rho must be in [0,1]");
    }
    if (params_.lr_b <= 0.0 || params_.lr_b > 1.0) {
        throw std::invalid_argument("lr_b must be in (0,1]");
    }
    if (params_.lr_w < 0.0 || params_.lr_w > 1.0) {
        throw std::invalid_argument("lr_w must be in [0,1]");
    }
    if (params_.lr_s < 0.0 || params_.lr_s > 1.0) {
        throw std::invalid_argument("lr_s must be in [0,1]");
    }
}

void QuadraticNeuronARTCore::set_weights(const std::vector<std::vector<double>>& weights) {
    if (weights.empty()) {
        weights_.clear();
        dim_ = 0;
        return;
    }

    const std::size_t expected_len = weights.front().size();
    if (expected_len < 3) {
        throw std::invalid_argument("invalid weight length");
    }

    const double disc = std::sqrt(static_cast<double>(4 * expected_len - 3));
    const int inferred_dim = static_cast<int>(std::llround((disc - 1.0) / 2.0));
    if (inferred_dim <= 0 || quadratic_weight_size(inferred_dim) != expected_len) {
        throw std::invalid_argument("invalid weight length");
    }

    for (const auto& w : weights) {
        if (w.size() != expected_len) throw std::invalid_argument("inconsistent weight dimensions");
    }

    if (dim_ != 0 && inferred_dim != dim_) {
        throw std::invalid_argument("weight dimensionality mismatch with model state");
    }

    weights_ = weights;
    dim_ = inferred_dim;
}

void QuadraticNeuronARTCore::fit(const double* x, std::size_t rows, std::size_t cols) {
    validate_x(x, rows, cols);
    if (dim_ == 0) dim_ = static_cast<int>(cols);
    labels_.assign(rows, 0);

    for (std::size_t i = 0; i < rows; ++i) {
        std::vector<double> sample(x + i * cols, x + (i + 1) * cols);
        labels_[i] = step_fit(sample);
    }
}

std::vector<int> QuadraticNeuronARTCore::predict(const double* x, std::size_t rows, std::size_t cols) const {
    if (weights_.empty()) throw std::runtime_error("Model has no clusters");
    validate_x(x, rows, cols);

    std::vector<int> y(rows, 0);
    for (std::size_t i = 0; i < rows; ++i) {
        const double* row = x + i * cols;
        int best_id = -1;
        double best_t = -std::numeric_limits<double>::infinity();
        for (std::size_t c = 0; c < weights_.size(); ++c) {
            const double t = category_choice(row, weights_[c], nullptr, nullptr);
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

const std::vector<int>& QuadraticNeuronARTCore::labels() const { return labels_; }
const std::vector<std::vector<double>>& QuadraticNeuronARTCore::weights() const { return weights_; }

double QuadraticNeuronARTCore::category_choice(
    const double* sample,
    const std::vector<double>& w,
    double* out_l2,
    std::vector<double>* out_z
) const {
    const std::size_t dim2 = square_size(dim_);
    const double* w_mat = w.data();
    const double* b = w.data() + dim2;
    const double s = w.back();

    std::vector<double> z(static_cast<std::size_t>(dim_), 0.0);
    for (int r = 0; r < dim_; ++r) {
        double acc = 0.0;
        for (int c = 0; c < dim_; ++c) {
            acc += w_mat[matrix_index(r, c, dim_)] * sample[c];
        }
        z[r] = acc;
    }

    double l2 = 0.0;
    for (int j = 0; j < dim_; ++j) {
        const double d = z[j] - b[j];
        l2 += d * d;
    }

    if (out_l2 != nullptr) *out_l2 = l2;
    if (out_z != nullptr) *out_z = z;

    return std::exp(-(s * s) * l2);
}

double QuadraticNeuronARTCore::match(double activation) const { return activation; }

std::vector<double> QuadraticNeuronARTCore::update_weight(
    const std::vector<double>& i,
    const std::vector<double>& w,
    double activation,
    double l2_z_b,
    const std::vector<double>& z
) const {
    const std::size_t dim2 = square_size(dim_);
    const double* w_mat = w.data();
    const double* b = w.data() + dim2;
    const double s = w.back();

    const double sst2 = 2.0 * s * s * activation;

    std::vector<double> out = w;

    for (int j = 0; j < dim_; ++j) {
        const double dz = z[j] - b[j];
        out[dim2 + j] = b[j] + params_.lr_b * (sst2 * dz);
    }

    for (int r = 0; r < dim_; ++r) {
        const double dz = z[r] - b[r];
        for (int c = 0; c < dim_; ++c) {
            const std::size_t idx = matrix_index(r, c, dim_);
            out[idx] = w_mat[idx] + params_.lr_w * (-sst2 * (dz * i[c]));
        }
    }

    out.back() = s + params_.lr_s * (-2.0 * s * activation * l2_z_b);
    return out;
}

std::vector<double> QuadraticNeuronARTCore::new_weight(const std::vector<double>& i) const {
    std::vector<double> out;
    out.reserve(quadratic_weight_size(dim_));

    for (int r = 0; r < dim_; ++r) {
        for (int c = 0; c < dim_; ++c) {
            out.push_back((r == c) ? 1.0 : 0.0);
        }
    }

    out.insert(out.end(), i.begin(), i.end());
    out.push_back(params_.s_init);
    return out;
}

int QuadraticNeuronARTCore::step_fit(const std::vector<double>& sample) {
    if (weights_.empty()) {
        weights_.push_back(new_weight(sample));
        return 0;
    }

    const std::size_t k = weights_.size();
    std::vector<double> t(k), m(k), l2(k);
    std::vector<std::vector<double>> z(k);

    for (std::size_t i = 0; i < k; ++i) {
        t[i] = category_choice(sample.data(), weights_[i], &l2[i], &z[i]);
        m[i] = match(t[i]);
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
            weights_[best] = update_weight(sample, weights_[best], t[best], l2[best], z[best]);
            return best;
        }
    }

    const int new_id = static_cast<int>(weights_.size());
    weights_.push_back(new_weight(sample));
    return new_id;
}

void QuadraticNeuronARTCore::validate_x(const double* x, std::size_t /*rows*/, std::size_t cols) const {
    if (x == nullptr) throw std::invalid_argument("X cannot be null");
    if (cols == 0) throw std::invalid_argument("X must have at least one feature");
    if (dim_ != 0 && static_cast<int>(cols) != dim_) {
        throw std::invalid_argument("feature dimension mismatch");
    }
}

}  // namespace artlib_cpp
