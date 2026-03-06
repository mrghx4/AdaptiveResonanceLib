#include "artlib_cpp/art1_core.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace artlib_cpp {

ART1Core::ART1Core(ART1Params params) : params_(params), dim_(0) {
    if (params_.rho < 0.0 || params_.rho > 1.0) {
        throw std::invalid_argument("rho must be in [0,1]");
    }
    if (params_.L < 1.0) {
        throw std::invalid_argument("L must be >= 1");
    }
}

std::string ART1Core::name() const {
    return "ART1Core";
}

void ART1Core::set_weights(const std::vector<std::vector<double>>& weights) {
    if (weights.empty()) {
        weights_.clear();
        dim_ = 0;
        return;
    }

    const std::size_t expected_len = weights.front().size();
    if (expected_len == 0 || (expected_len % 2) != 0) {
        throw std::invalid_argument("weight length must be even (2*dim)");
    }

    for (const auto& w : weights) {
        if (w.size() != expected_len) {
            throw std::invalid_argument("inconsistent weight dimension across clusters");
        }
    }

    if (dim_ != 0 && expected_len != 2 * dim_) {
        throw std::invalid_argument("weight dimensionality mismatch with model state");
    }

    weights_ = weights;
    dim_ = expected_len / 2;
}

void ART1Core::fit(const MatrixView& x) {
    validate_matrix(x);
    if (dim_ == 0) {
        dim_ = x.cols;
    }

    labels_.assign(x.rows, 0);
    for (std::size_t i = 0; i < x.rows; ++i) {
        const double* row = x.data + i * x.cols;
        std::vector<double> sample(row, row + x.cols);
        labels_[i] = step_fit(sample);
    }
}

std::vector<int> ART1Core::predict(const MatrixView& x) const {
    validate_matrix(x);
    if (weights_.empty()) {
        throw std::runtime_error("Model has no clusters");
    }

    std::vector<int> out(x.rows, 0);
    for (std::size_t i = 0; i < x.rows; ++i) {
        const double* row = x.data + i * x.cols;
        int best_id = -1;
        double best_t = -1e300;

        for (std::size_t c = 0; c < weights_.size(); ++c) {
            const double t = category_choice(row, weights_[c]);
            if (t > best_t) {
                best_t = t;
                best_id = static_cast<int>(c);
            }
        }
        out[i] = best_id < 0 ? 0 : best_id;
    }
    return out;
}

const std::vector<std::vector<double>>& ART1Core::weights() const {
    return weights_;
}

const std::vector<int>& ART1Core::labels() const {
    return labels_;
}

double ART1Core::category_choice(const double* sample, const std::vector<double>& w) const {
    double s = 0.0;
    for (std::size_t j = 0; j < dim_; ++j) {
        s += sample[j] * w[j];
    }
    return s;
}

double ART1Core::match(const double* sample, const std::vector<double>& w) const {
    int count = 0;
    for (std::size_t j = 0; j < dim_; ++j) {
        const int i_bit = (sample[j] != 0.0);
        const int td_bit = (w[dim_ + j] != 0.0);
        if (i_bit & td_bit) {
            ++count;
        }
    }
    return static_cast<double>(count) / static_cast<double>(dim_);
}

std::vector<double> ART1Core::update_weight(
    const double* sample, const std::vector<double>& w
) const {
    std::vector<double> out(2 * dim_, 0.0);

    int count = 0;
    for (std::size_t j = 0; j < dim_; ++j) {
        const int td_new = ((sample[j] != 0.0) & (w[dim_ + j] != 0.0));
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

std::vector<double> ART1Core::new_weight(const double* sample) const {
    std::vector<double> w(2 * dim_, 0.0);
    const double sf = params_.L / (params_.L - 1.0 + static_cast<double>(dim_));
    for (std::size_t j = 0; j < dim_; ++j) {
        const double bit = (sample[j] != 0.0) ? 1.0 : 0.0;
        w[dim_ + j] = bit;
        w[j] = sf * bit;
    }
    return w;
}

int ART1Core::step_fit(const std::vector<double>& sample) {
    if (weights_.empty()) {
        weights_.push_back(new_weight(sample.data()));
        return 0;
    }

    const std::size_t k = weights_.size();
    std::vector<double> t(k), m(k);
    for (std::size_t i = 0; i < k; ++i) {
        t[i] = category_choice(sample.data(), weights_[i]);
        m[i] = match(sample.data(), weights_[i]);
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
        if (m[best] < params_.rho) {
            continue;
        }
        weights_[best] = update_weight(sample.data(), weights_[best]);
        return best;
    }

    const int new_id = static_cast<int>(weights_.size());
    weights_.push_back(new_weight(sample.data()));
    return new_id;
}

void ART1Core::validate_matrix(const MatrixView& x) const {
    if (x.data == nullptr) {
        throw std::invalid_argument("Input matrix pointer cannot be null");
    }
    if (x.cols == 0) {
        throw std::invalid_argument("Input must have at least one feature");
    }
    if (dim_ != 0 && x.cols != dim_) {
        throw std::invalid_argument("X feature dimension mismatch with model");
    }
    if (!weights_.empty() && x.cols != weights_.front().size() / 2) {
        throw std::invalid_argument("X feature dimension mismatch with existing weights");
    }
    for (std::size_t i = 0; i < x.rows; ++i) {
        const double* row = x.data + i * x.cols;
        for (std::size_t j = 0; j < x.cols; ++j) {
            const double v = row[j];
            if (!(v == 0.0 || v == 1.0)) {
                throw std::invalid_argument("ART1 requires binary inputs in {0,1}");
            }
        }
    }
}

}  // namespace artlib_cpp
