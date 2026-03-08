#include "artlib_cpp/bartmap_metrics_core.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace artlib_cpp {

namespace {

double pearson_corr(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("vector size mismatch");
    const std::size_t n = a.size();
    if (n < 2) return std::numeric_limits<double>::quiet_NaN();

    double mean_a = 0.0;
    double mean_b = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= static_cast<double>(n);
    mean_b /= static_cast<double>(n);

    double num = 0.0;
    double den_a = 0.0;
    double den_b = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double da = a[i] - mean_a;
        const double db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    const double den = std::sqrt(den_a * den_b);
    if (den <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return num / den;
}

std::vector<std::size_t> indices_for_cluster(
    const int* labels,
    std::size_t n,
    int c_b
) {
    std::vector<std::size_t> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (labels[i] == c_b) out.push_back(i);
    }
    return out;
}

std::vector<double> select_components(
    const double* row,
    const std::vector<std::size_t>& comp_idx
) {
    std::vector<double> out;
    out.reserve(comp_idx.size());
    for (std::size_t j : comp_idx) out.push_back(row[j]);
    return out;
}

}  // namespace

double average_pearson_corr(
    const double* x,
    std::size_t rows,
    std::size_t cols,
    const int* column_labels,
    std::size_t labels_len,
    std::size_t k,
    int c_b
) {
    if (x == nullptr || column_labels == nullptr) {
        throw std::invalid_argument("X and column_labels must be non-null");
    }
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("X must be non-empty");
    }
    if (k >= rows) {
        throw std::invalid_argument("k out of bounds");
    }

    // Preserve BARTMAP Python semantics where same labels vector is used for
    // both row selection and component selection.
    if (labels_len != rows || labels_len != cols) {
        throw std::invalid_argument("column_labels length must equal both rows and cols");
    }

    const auto row_idx = indices_for_cluster(column_labels, rows, c_b);
    if (row_idx.empty()) {
        throw std::invalid_argument("no rows for cluster");
    }

    const auto comp_idx = indices_for_cluster(column_labels, cols, c_b);
    const std::vector<double> x_k_cb = select_components(x + k * cols, comp_idx);

    if (x_k_cb.size() < 2) return std::numeric_limits<double>::quiet_NaN();

    double sum = 0.0;
    for (std::size_t r : row_idx) {
        const std::vector<double> x_r_cb = select_components(x + r * cols, comp_idx);
        sum += pearson_corr(x_k_cb, x_r_cb);
    }

    return sum / static_cast<double>(row_idx.size());
}

bool any_cluster_match(
    const double* x,
    std::size_t rows,
    std::size_t cols,
    const int* column_labels,
    std::size_t labels_len,
    std::size_t k,
    std::size_t n_clusters_b,
    double eta
) {
    for (std::size_t c_b = 0; c_b < n_clusters_b; ++c_b) {
        const double m = average_pearson_corr(
            x,
            rows,
            cols,
            column_labels,
            labels_len,
            k,
            static_cast<int>(c_b)
        );
        if (m >= eta) return true;
    }
    return false;
}

}  // namespace artlib_cpp
