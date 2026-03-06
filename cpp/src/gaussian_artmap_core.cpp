#include "artlib_cpp/gaussian_artmap_core.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace artlib_cpp {

GaussianARTMAPCore::GaussianARTMAPCore(GaussianARTMAPParams params)
    : params_(std::move(params)), dim_(0), rho_runtime_(params_.rho) {
    if (params_.sigma_init.empty()) {
        throw std::invalid_argument("sigma_init cannot be empty");
    }
}

void GaussianARTMAPCore::set_state(
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
    if (expected_len < 5 || ((expected_len - 2) % 3) != 0) {
        throw std::invalid_argument("invalid weight length");
    }
    for (const auto& w : weights) {
        if (w.size() != expected_len) throw std::invalid_argument("inconsistent weight dimensions");
    }
    const int inferred_dim = static_cast<int>((expected_len - 2) / 3);
    if (dim_ != 0 && inferred_dim != dim_) {
        throw std::invalid_argument("weight dimensionality mismatch with model state");
    }

    weights_ = weights;
    cluster_map_.clear();
    for (std::size_t i = 0; i < cluster_labels.size(); ++i) cluster_map_[static_cast<int>(i)] = cluster_labels[i];
    dim_ = inferred_dim;
}

void GaussianARTMAPCore::fit(
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

std::pair<std::vector<int>, std::vector<int>> GaussianARTMAPCore::predict(
    const double* x, std::size_t rows, std::size_t cols
) {
    if (weights_.empty()) throw std::runtime_error("Model has no clusters");
    validate_x(x, rows, cols);

    double total_n = 0.0;
    for (const auto& c : weights_) total_n += c.back();

    std::vector<int> y_a(rows), y_b(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        const double* row = x + i * cols;
        int best_id = -1;
        double best_t = -std::numeric_limits<double>::infinity();
        for (std::size_t c = 0; c < weights_.size(); ++c) {
            const double t = category_choice(row, weights_[c], total_n);
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

const std::vector<int>& GaussianARTMAPCore::labels_a() const { return labels_a_; }
const std::vector<std::vector<double>>& GaussianARTMAPCore::weights() const { return weights_; }

std::vector<int> GaussianARTMAPCore::cluster_labels() const {
    std::vector<int> out(weights_.size(), 0);
    for (const auto& kv : cluster_map_) out[kv.first] = kv.second;
    return out;
}

void GaussianARTMAPCore::reset_rho() { rho_runtime_ = params_.rho; }

double GaussianARTMAPCore::gaussian_exp(const double* x, const std::vector<double>& w) const {
    const double* mean = w.data();
    const double* inv_sig = w.data() + 2 * dim_;
    double q = 0.0;
    for (int j = 0; j < dim_; ++j) {
        const double d = x[j] - mean[j];
        q += d * d * inv_sig[j];
    }
    return std::exp(-0.5 * q);
}

double GaussianARTMAPCore::category_choice(
    const double* sample, const std::vector<double>& w, double total_n
) const {
    const double exp_term = gaussian_exp(sample, w);
    const double sqrt_det = w[3 * dim_];
    const double n_c = w[3 * dim_ + 1];
    const double p_i_cj = exp_term / (params_.alpha + sqrt_det);
    const double p_cj = n_c / std::max(total_n, 1e-12);
    return p_i_cj * p_cj;
}

double GaussianARTMAPCore::match(const double* sample, const std::vector<double>& w) const {
    return gaussian_exp(sample, w);
}

std::vector<double> GaussianARTMAPCore::update_weight(
    const std::vector<double>& i, const std::vector<double>& w
) const {
    const double* mean = w.data();
    const double* sigma = w.data() + dim_;
    double n = w[3 * dim_ + 1];

    const double n_new = n + 1.0;
    std::vector<double> mean_new(dim_), sigma_new(dim_);
    for (int j = 0; j < dim_; ++j) {
        mean_new[j] = (1.0 - 1.0 / n_new) * mean[j] + (1.0 / n_new) * i[j];
        const double sigma2_old = sigma[j] * sigma[j];
        const double sigma2_new =
            (1.0 - 1.0 / n_new) * sigma2_old + (1.0 / n_new) * std::pow(mean_new[j] - i[j], 2);
        sigma_new[j] = std::sqrt(sigma2_new);
    }

    std::vector<double> inv_sig_new(dim_);
    double det = 1.0;
    for (int j = 0; j < dim_; ++j) {
        const double s2 = sigma_new[j] * sigma_new[j];
        inv_sig_new[j] = 1.0 / s2;
        det *= s2;
    }
    const double sqrt_det_new = std::sqrt(det);

    std::vector<double> out;
    out.reserve(3 * dim_ + 2);
    out.insert(out.end(), mean_new.begin(), mean_new.end());
    out.insert(out.end(), sigma_new.begin(), sigma_new.end());
    out.insert(out.end(), inv_sig_new.begin(), inv_sig_new.end());
    out.push_back(sqrt_det_new);
    out.push_back(n_new);
    return out;
}

std::vector<double> GaussianARTMAPCore::new_weight(const std::vector<double>& i) const {
    if (params_.sigma_init.size() != static_cast<std::size_t>(dim_)) {
        throw std::runtime_error("sigma_init dimension mismatch");
    }
    std::vector<double> inv_sig(dim_);
    double det = 1.0;
    for (int j = 0; j < dim_; ++j) {
        const double s2 = params_.sigma_init[j] * params_.sigma_init[j];
        inv_sig[j] = 1.0 / s2;
        det *= s2;
    }
    const double sqrt_det = std::sqrt(det);

    std::vector<double> w;
    w.reserve(3 * dim_ + 2);
    w.insert(w.end(), i.begin(), i.end());
    w.insert(w.end(), params_.sigma_init.begin(), params_.sigma_init.end());
    w.insert(w.end(), inv_sig.begin(), inv_sig.end());
    w.push_back(sqrt_det);
    w.push_back(1.0);
    return w;
}

bool GaussianARTMAPCore::match_tracking(double m) {
    if (params_.mt == "MT+") rho_runtime_ = m + params_.epsilon;
    else if (params_.mt == "MT-") rho_runtime_ = m - params_.epsilon;
    else if (params_.mt == "MT0") rho_runtime_ = m;
    else if (params_.mt == "MT1") rho_runtime_ = std::numeric_limits<double>::infinity();
    else if (params_.mt == "MT~") {/* no-op */}
    else throw std::invalid_argument("Invalid MT mode: " + params_.mt);
    return !(params_.mt == "MT1" || rho_runtime_ > 1.0);
}

bool GaussianARTMAPCore::matches_vigilance(double m) const {
    if (params_.mt == "MT+" || params_.mt == "MT-" || params_.mt == "MT1")
        return m >= rho_runtime_;
    if (params_.mt == "MT0" || params_.mt == "MT~")
        return m > rho_runtime_;
    throw std::invalid_argument("Invalid MT mode");
}

int GaussianARTMAPCore::step_fit(const std::vector<double>& sample, int c_b) {
    reset_rho();
    if (weights_.empty()) {
        weights_.push_back(new_weight(sample));
        cluster_map_[0] = c_b;
        return 0;
    }

    double total_n = 0.0;
    for (const auto& c : weights_) total_n += c.back();

    const std::size_t k = weights_.size();
    std::vector<double> t(k), m(k);
    for (std::size_t i = 0; i < k; ++i) {
        t[i] = category_choice(sample.data(), weights_[i], total_n);
        m[i] = match(sample.data(), weights_[i]);
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
        weights_[best] = update_weight(sample, weights_[best]);
        cluster_map_[best] = c_b;
        return best;
    }

    const int new_id = static_cast<int>(weights_.size());
    weights_.push_back(new_weight(sample));
    cluster_map_[new_id] = c_b;
    return new_id;
}

void GaussianARTMAPCore::validate_x(const double* x, std::size_t /*rows*/, std::size_t cols) const {
    if (x == nullptr) throw std::invalid_argument("X cannot be null");
    if (cols == 0) throw std::invalid_argument("X must have at least one feature");
    if (dim_ != 0 && static_cast<int>(cols) != dim_)
        throw std::invalid_argument("feature dimension mismatch");
}

}  // namespace artlib_cpp
