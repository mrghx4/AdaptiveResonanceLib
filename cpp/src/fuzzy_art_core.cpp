#include "artlib_cpp/fuzzy_art_core.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace artlib_cpp {

FuzzyARTCore::FuzzyARTCore(FuzzyARTParams params)
    : params_(params), dim_original_(0) {}

std::string FuzzyARTCore::name() const {
    return "FuzzyARTCore";
}

void FuzzyARTCore::fit(const MatrixView& x) {
    validate_matrix(x);
    if (dim_original_ == 0) {
        dim_original_ = x.cols / 2;
    }

    labels_.assign(x.rows, 0);
    for (std::size_t i = 0; i < x.rows; ++i) {
        const double* row = x.data + i * x.cols;
        std::vector<double> sample(row, row + x.cols);
        labels_[i] = step_fit(sample);
    }
}

std::vector<int> FuzzyARTCore::predict(const MatrixView& x) const {
    validate_matrix(x);
    if (weights_.empty()) {
        throw std::runtime_error("Model has no clusters");
    }

    std::vector<int> out(x.rows, 0);
    for (std::size_t i = 0; i < x.rows; ++i) {
        const double* row = x.data + i * x.cols;

        int best_id = -1;
        double best_t = -1.0;
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

void FuzzyARTCore::set_weights(const std::vector<std::vector<double>>& weights) {
    if (weights.empty()) {
        weights_.clear();
        dim_original_ = 0;
        return;
    }

    const std::size_t expected_len = weights.front().size();
    if (expected_len == 0 || (expected_len % 2) != 0) {
        throw std::invalid_argument("Weights must be non-empty complement-coded vectors");
    }

    for (const auto& w : weights) {
        if (w.size() != expected_len) {
            throw std::invalid_argument("All weight vectors must have the same dimensionality");
        }
    }

    if (dim_original_ != 0 && expected_len != 2 * dim_original_) {
        throw std::invalid_argument("Weight dimensionality mismatch with model state");
    }

    weights_ = weights;
    dim_original_ = expected_len / 2;
}

const std::vector<std::vector<double>>& FuzzyARTCore::weights() const {
    return weights_;
}

const std::vector<int>& FuzzyARTCore::labels() const {
    return labels_;
}

double FuzzyARTCore::l1_and(const double* x, const std::vector<double>& w, int len) {
    double s = 0.0;
    for (int j = 0; j < len; ++j) {
        s += std::min(x[j], w[j]);
    }
    return s;
}

double FuzzyARTCore::category_choice(const double* sample, const std::vector<double>& w) const {
    const int len = static_cast<int>(w.size());
    const double num = l1_and(sample, w, len);
    const double denom = params_.alpha + std::accumulate(w.begin(), w.end(), 0.0);
    return num / denom;
}

double FuzzyARTCore::match(const double* sample, const std::vector<double>& w) const {
    const int len = static_cast<int>(w.size());
    const double num = l1_and(sample, w, len);
    return num / static_cast<double>(dim_original_);
}

std::vector<double> FuzzyARTCore::update_weight(
    const std::vector<double>& sample,
    const std::vector<double>& w
) const {
    std::vector<double> out(w.size());
    for (std::size_t j = 0; j < w.size(); ++j) {
        out[j] = params_.beta * std::min(sample[j], w[j]) + (1.0 - params_.beta) * w[j];
    }
    return out;
}

int FuzzyARTCore::step_fit(const std::vector<double>& sample) {
    if (weights_.empty()) {
        weights_.push_back(sample);
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
        weights_[best] = update_weight(sample, weights_[best]);
        return best;
    }

    const int new_id = static_cast<int>(weights_.size());
    weights_.push_back(sample);
    return new_id;
}

void FuzzyARTCore::validate_matrix(const MatrixView& x) const {
    if (x.data == nullptr) {
        throw std::invalid_argument("Input matrix pointer cannot be null");
    }
    if (x.cols == 0 || (x.cols % 2) != 0) {
        throw std::invalid_argument("Input must be complement-coded with even feature size");
    }
    if (dim_original_ != 0 && x.cols != 2 * dim_original_) {
        throw std::invalid_argument(
            "Input dimensionality mismatch with existing model weights/state"
        );
    }
    if (!weights_.empty() && x.cols != weights_.front().size()) {
        throw std::invalid_argument(
            "Input dimensionality mismatch with existing model weights"
        );
    }
}

}  // namespace artlib_cpp
