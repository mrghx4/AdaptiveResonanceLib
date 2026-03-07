#include "artlib_cpp/bayesian_art_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace artlib_cpp {

namespace {
constexpr double kPi2 = 2.0 * 3.141592653589793238462643383279502884;
}

BayesianARTCore::BayesianARTCore(BayesianARTParams params)
    : params_(std::move(params)), dim_(0) {
    if (params_.rho <= 0.0) throw std::invalid_argument("rho must be > 0");
}

void BayesianARTCore::set_weights(const std::vector<std::vector<double>>& weights) {
    if (weights.empty()) {
        weights_.clear();
        dim_ = 0;
        return;
    }

    const std::size_t expected_len = weights.front().size();
    if (expected_len < 3) throw std::invalid_argument("invalid weight length");

    const double f = static_cast<double>(expected_len - 1);
    const int inferred_dim = static_cast<int>(std::round((std::sqrt(1.0 + 4.0 * f) - 1.0) / 2.0));
    if (inferred_dim <= 0 || static_cast<std::size_t>(inferred_dim + inferred_dim * inferred_dim + 1) != expected_len) {
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

void BayesianARTCore::fit(const double* x, std::size_t rows, std::size_t cols) {
    validate_x(x, rows, cols);
    if (dim_ == 0) dim_ = static_cast<int>(cols);

    if (params_.cov_init.size() != static_cast<std::size_t>(dim_ * dim_)) {
        throw std::invalid_argument("cov_init shape mismatch");
    }

    labels_.assign(rows, 0);
    for (std::size_t i = 0; i < rows; ++i) {
        labels_[i] = step_fit(x + i * cols);
    }
}

std::vector<int> BayesianARTCore::predict(const double* x, std::size_t rows, std::size_t cols) const {
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

const std::vector<int>& BayesianARTCore::labels() const { return labels_; }
const std::vector<std::vector<double>>& BayesianARTCore::weights() const { return weights_; }

bool BayesianARTCore::invert_and_det(const std::vector<double>& a, std::vector<double>& inv, double& det) const {
    const int n = dim_;
    // Fast SPD path: covariance matrices are expected to be symmetric positive definite.
    std::vector<double> l(static_cast<std::size_t>(n * n), 0.0);
    bool chol_ok = true;
    for (int i = 0; i < n && chol_ok; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = a[static_cast<std::size_t>(i * n + j)];
            for (int k = 0; k < j; ++k) {
                sum -= l[static_cast<std::size_t>(i * n + k)] * l[static_cast<std::size_t>(j * n + k)];
            }
            if (i == j) {
                if (sum <= 1e-15) {
                    chol_ok = false;
                    break;
                }
                l[static_cast<std::size_t>(i * n + j)] = std::sqrt(sum);
            } else {
                const double ljj = l[static_cast<std::size_t>(j * n + j)];
                if (std::abs(ljj) <= 1e-15) {
                    chol_ok = false;
                    break;
                }
                l[static_cast<std::size_t>(i * n + j)] = sum / ljj;
            }
        }
    }

    if (chol_ok) {
        det = 1.0;
        for (int i = 0; i < n; ++i) {
            const double d = l[static_cast<std::size_t>(i * n + i)];
            det *= d * d;
        }

        inv.assign(static_cast<std::size_t>(n * n), 0.0);
        std::vector<double> y(static_cast<std::size_t>(n), 0.0);
        std::vector<double> x(static_cast<std::size_t>(n), 0.0);

        for (int col = 0; col < n; ++col) {
            for (int i = 0; i < n; ++i) {
                double sum = (i == col) ? 1.0 : 0.0;
                for (int k = 0; k < i; ++k) {
                    sum -= l[static_cast<std::size_t>(i * n + k)] * y[static_cast<std::size_t>(k)];
                }
                y[static_cast<std::size_t>(i)] = sum / l[static_cast<std::size_t>(i * n + i)];
            }
            for (int i = n - 1; i >= 0; --i) {
                double sum = y[static_cast<std::size_t>(i)];
                for (int k = i + 1; k < n; ++k) {
                    sum -= l[static_cast<std::size_t>(k * n + i)] * x[static_cast<std::size_t>(k)];
                }
                x[static_cast<std::size_t>(i)] = sum / l[static_cast<std::size_t>(i * n + i)];
            }
            for (int i = 0; i < n; ++i) {
                inv[static_cast<std::size_t>(i * n + col)] = x[static_cast<std::size_t>(i)];
            }
        }
        return true;
    }

    // Fallback for non-SPD / numerically problematic inputs.
    inv.assign(static_cast<std::size_t>(n * n), 0.0);
    std::vector<double> m = a;
    for (int i = 0; i < n; ++i) inv[static_cast<std::size_t>(i * n + i)] = 1.0;

    det = 1.0;
    int sign = 1;

    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double max_abs = std::abs(m[static_cast<std::size_t>(col * n + col)]);
        for (int r = col + 1; r < n; ++r) {
            const double v = std::abs(m[static_cast<std::size_t>(r * n + col)]);
            if (v > max_abs) {
                max_abs = v;
                pivot = r;
            }
        }

        if (max_abs <= 1e-15) {
            det = 0.0;
            return false;
        }

        if (pivot != col) {
            for (int c = 0; c < n; ++c) {
                std::swap(m[static_cast<std::size_t>(col * n + c)], m[static_cast<std::size_t>(pivot * n + c)]);
                std::swap(inv[static_cast<std::size_t>(col * n + c)], inv[static_cast<std::size_t>(pivot * n + c)]);
            }
            sign = -sign;
        }

        const double piv = m[static_cast<std::size_t>(col * n + col)];
        det *= piv;

        for (int c = 0; c < n; ++c) {
            m[static_cast<std::size_t>(col * n + c)] /= piv;
            inv[static_cast<std::size_t>(col * n + c)] /= piv;
        }

        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            const double factor = m[static_cast<std::size_t>(r * n + col)];
            if (factor == 0.0) continue;
            for (int c = 0; c < n; ++c) {
                m[static_cast<std::size_t>(r * n + c)] -= factor * m[static_cast<std::size_t>(col * n + c)];
                inv[static_cast<std::size_t>(r * n + c)] -= factor * inv[static_cast<std::size_t>(col * n + c)];
            }
        }
    }

    det *= static_cast<double>(sign);
    return true;
}

double BayesianARTCore::quadratic_form(const std::vector<double>& inv_cov, const double* dist) const {
    const int n = dim_;
    std::vector<double> tmp(static_cast<std::size_t>(n), 0.0);
    for (int r = 0; r < n; ++r) {
        double s = 0.0;
        for (int c = 0; c < n; ++c) {
            s += inv_cov[static_cast<std::size_t>(r * n + c)] * dist[c];
        }
        tmp[static_cast<std::size_t>(r)] = s;
    }

    double q = 0.0;
    for (int i = 0; i < n; ++i) q += dist[i] * tmp[static_cast<std::size_t>(i)];
    return q;
}

double BayesianARTCore::category_choice(const double* sample, const std::vector<double>& w) const {
    const int n = dim_;
    const double* mean = w.data();
    const double* cov = w.data() + n;
    const double count = w.back();

    std::vector<double> cov_vec(cov, cov + n * n);
    std::vector<double> inv_cov;
    double det_cov = 0.0;
    if (!invert_and_det(cov_vec, inv_cov, det_cov) || det_cov <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }

    std::vector<double> dist(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) dist[static_cast<std::size_t>(i)] = mean[i] - sample[i];

    const double q = quadratic_form(inv_cov, dist.data());
    const double exp_term = std::exp(-0.5 * q);
    const double denom = std::sqrt(std::pow(kPi2, n) * det_cov);
    const double p_i_cj = exp_term / denom;

    double total_n = 0.0;
    for (const auto& wi : weights_) total_n += wi.back();
    const double p_cj = count / std::max(total_n, 1e-12);

    return p_i_cj * p_cj;
}

double BayesianARTCore::match_criterion(const double* sample, const std::vector<double>& w) const {
    const std::vector<double> new_w = update_weight(sample, w);
    const int n = dim_;
    const double* cov = new_w.data() + n;
    std::vector<double> cov_vec(cov, cov + n * n);
    std::vector<double> inv_cov;
    double det_cov = 0.0;
    if (!invert_and_det(cov_vec, inv_cov, det_cov)) return 0.0;
    return det_cov;
}

std::vector<double> BayesianARTCore::update_weight(const double* sample, const std::vector<double>& w) const {
    const int n = dim_;
    const double* mean = w.data();
    const double* cov = w.data() + n;
    const double count = w.back();

    const double n_new = count + 1.0;

    std::vector<double> mean_new(static_cast<std::size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        mean_new[static_cast<std::size_t>(i)] = (1.0 - (1.0 / n_new)) * mean[i] + (1.0 / n_new) * sample[i];
    }

    std::vector<double> cov_new(static_cast<std::size_t>(n * n), 0.0);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            const double outer = (sample[r] - mean_new[static_cast<std::size_t>(r)])
                               * (sample[c] - mean_new[static_cast<std::size_t>(c)]);
            cov_new[static_cast<std::size_t>(r * n + c)] = (count / n_new) * cov[static_cast<std::size_t>(r * n + c)]
                                                          + (1.0 / n_new) * outer;
        }
    }

    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(n + n * n + 1));
    out.insert(out.end(), mean_new.begin(), mean_new.end());
    out.insert(out.end(), cov_new.begin(), cov_new.end());
    out.push_back(n_new);
    return out;
}

std::vector<double> BayesianARTCore::new_weight(const double* sample) const {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(dim_ + dim_ * dim_ + 1));
    out.insert(out.end(), sample, sample + dim_);
    out.insert(out.end(), params_.cov_init.begin(), params_.cov_init.end());
    out.push_back(1.0);
    return out;
}

int BayesianARTCore::step_fit(const double* sample) {
    if (weights_.empty()) {
        weights_.push_back(new_weight(sample));
        return 0;
    }

    const std::size_t k = weights_.size();
    std::vector<double> t(k), m(k);
    for (std::size_t i = 0; i < k; ++i) {
        t[i] = category_choice(sample, weights_[i]);
        m[i] = match_criterion(sample, weights_[i]);
    }

    std::vector<int> order;
    order.reserve(k);
    for (std::size_t i = 0; i < k; ++i) order.push_back(static_cast<int>(i));
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (t[a] != t[b]) return t[a] > t[b];
        return a < b;
    });

    for (const int best : order) {
        if (params_.rho >= m[best]) {
            weights_[best] = update_weight(sample, weights_[best]);
            return best;
        }
    }

    const int new_id = static_cast<int>(weights_.size());
    weights_.push_back(new_weight(sample));
    return new_id;
}

void BayesianARTCore::validate_x(const double* x, std::size_t /*rows*/, std::size_t cols) const {
    if (x == nullptr) throw std::invalid_argument("X cannot be null");
    if (cols == 0) throw std::invalid_argument("X must have at least one feature");
    if (dim_ != 0 && static_cast<int>(cols) != dim_) {
        throw std::invalid_argument("feature dimension mismatch");
    }
}

}  // namespace artlib_cpp
