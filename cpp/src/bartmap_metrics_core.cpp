#include "artlib_cpp/bartmap_metrics_core.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace artlib_cpp {

namespace {

double pearson_corr_indexed(
    const double* a,
    const double* b,
    const std::vector<std::size_t>& idx
) {
    const std::size_t n = idx.size();
    if (n < 2) return std::numeric_limits<double>::quiet_NaN();

    double mean_a = 0.0;
    double mean_b = 0.0;
    for (std::size_t j : idx) {
        mean_a += a[j];
        mean_b += b[j];
    }
    mean_a /= static_cast<double>(n);
    mean_b /= static_cast<double>(n);

    double num = 0.0;
    double den_a = 0.0;
    double den_b = 0.0;
    for (std::size_t j : idx) {
        const double da = a[j] - mean_a;
        const double db = b[j] - mean_b;
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

std::vector<std::vector<std::size_t>> indices_by_cluster(
    const int* labels,
    std::size_t n,
    std::size_t n_clusters
) {
    std::vector<std::vector<std::size_t>> out(n_clusters);
    for (std::size_t i = 0; i < n; ++i) {
        const int c = labels[i];
        if (c >= 0 && static_cast<std::size_t>(c) < n_clusters) {
            out[static_cast<std::size_t>(c)].push_back(i);
        }
    }
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

    // labels_len == rows == cols above, so row/component masks are identical.
    const auto& comp_idx = row_idx;
    if (comp_idx.size() < 2) return std::numeric_limits<double>::quiet_NaN();

    double sum = 0.0;
    const double* x_k = x + k * cols;
    for (std::size_t r : row_idx) {
        const double* x_r = x + r * cols;
        sum += pearson_corr_indexed(x_k, x_r, comp_idx);
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
    if (x == nullptr || column_labels == nullptr) {
        throw std::invalid_argument("X and column_labels must be non-null");
    }
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("X must be non-empty");
    }
    if (k >= rows) {
        throw std::invalid_argument("k out of bounds");
    }
    if (labels_len != rows || labels_len != cols) {
        throw std::invalid_argument("column_labels length must equal both rows and cols");
    }
    if (n_clusters_b == 0) {
        return false;
    }

    const auto cluster_indices = indices_by_cluster(column_labels, labels_len, n_clusters_b);
    const double* x_k = x + k * cols;

    for (std::size_t c_b = 0; c_b < n_clusters_b; ++c_b) {
        const auto& idx = cluster_indices[c_b];
        if (idx.empty()) {
            throw std::invalid_argument("no rows for cluster");
        }
        if (idx.size() < 2) continue;

        double sum = 0.0;
        for (std::size_t r : idx) {
            const double* x_r = x + r * cols;
            sum += pearson_corr_indexed(x_k, x_r, idx);
        }
        const double m = sum / static_cast<double>(idx.size());
        if (m >= eta) return true;
    }
    return false;
}

}  // namespace artlib_cpp
