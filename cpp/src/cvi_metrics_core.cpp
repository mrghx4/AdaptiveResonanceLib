#include "artlib_cpp/cvi_metrics_core.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace artlib_cpp {

namespace {

struct LabelMap {
    std::vector<int> unique_labels;
    std::vector<int> dense;  // row -> [0, k)
};

LabelMap make_dense_labels(const int* labels, std::size_t rows) {
    std::unordered_map<int, int> idx;
    std::vector<int> unique;
    unique.reserve(rows);
    std::vector<int> dense(rows, 0);

    for (std::size_t i = 0; i < rows; ++i) {
        const int l = labels[i];
        auto it = idx.find(l);
        if (it == idx.end()) {
            const int id = static_cast<int>(unique.size());
            idx.emplace(l, id);
            unique.push_back(l);
            dense[i] = id;
        } else {
            dense[i] = it->second;
        }
    }

    return {std::move(unique), std::move(dense)};
}

inline double sq_l2(const double* a, const double* b, std::size_t cols) {
    double s = 0.0;
    for (std::size_t j = 0; j < cols; ++j) {
        const double d = a[j] - b[j];
        s += d * d;
    }
    return s;
}

inline double l2(const double* a, const double* b, std::size_t cols) {
    return std::sqrt(sq_l2(a, b, cols));
}

double calinski_harabasz(
    const double* x,
    std::size_t rows,
    std::size_t cols,
    const std::vector<int>& dense,
    std::size_t k
) {
    if (k < 2 || rows <= k) {
        throw std::invalid_argument("Calinski-Harabasz requires n_clusters>=2 and n_samples>n_clusters");
    }

    std::vector<std::size_t> counts(k, 0);
    std::vector<double> means(k * cols, 0.0);
    std::vector<double> global(cols, 0.0);

    for (std::size_t i = 0; i < rows; ++i) {
        const int c = dense[i];
        counts[c] += 1;
        const double* xi = x + i * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            means[static_cast<std::size_t>(c) * cols + j] += xi[j];
            global[j] += xi[j];
        }
    }

    for (std::size_t j = 0; j < cols; ++j) global[j] /= static_cast<double>(rows);
    for (std::size_t c = 0; c < k; ++c) {
        if (counts[c] == 0) continue;
        const double inv = 1.0 / static_cast<double>(counts[c]);
        for (std::size_t j = 0; j < cols; ++j) {
            means[c * cols + j] *= inv;
        }
    }

    double tr_w = 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
        const int c = dense[i];
        const double* xi = x + i * cols;
        const double* mc = means.data() + static_cast<std::size_t>(c) * cols;
        tr_w += sq_l2(xi, mc, cols);
    }

    double tr_b = 0.0;
    for (std::size_t c = 0; c < k; ++c) {
        if (counts[c] == 0) continue;
        const double* mc = means.data() + c * cols;
        tr_b += static_cast<double>(counts[c]) * sq_l2(mc, global.data(), cols);
    }

    const double num = tr_b / static_cast<double>(k - 1);
    const double den = tr_w / static_cast<double>(rows - k);
    if (den <= 0.0) throw std::invalid_argument("Calinski-Harabasz undefined for zero within-cluster dispersion");
    return num / den;
}

double davies_bouldin(
    const double* x,
    std::size_t rows,
    std::size_t cols,
    const std::vector<int>& dense,
    std::size_t k
) {
    if (k < 2) {
        throw std::invalid_argument("Davies-Bouldin requires n_clusters>=2");
    }

    std::vector<std::size_t> counts(k, 0);
    std::vector<double> means(k * cols, 0.0);

    for (std::size_t i = 0; i < rows; ++i) {
        const int c = dense[i];
        counts[c] += 1;
        const double* xi = x + i * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            means[static_cast<std::size_t>(c) * cols + j] += xi[j];
        }
    }

    for (std::size_t c = 0; c < k; ++c) {
        if (counts[c] == 0) continue;
        const double inv = 1.0 / static_cast<double>(counts[c]);
        for (std::size_t j = 0; j < cols; ++j) {
            means[c * cols + j] *= inv;
        }
    }

    std::vector<double> scat(k, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        const int c = dense[i];
        const double* xi = x + i * cols;
        const double* mc = means.data() + static_cast<std::size_t>(c) * cols;
        scat[static_cast<std::size_t>(c)] += l2(xi, mc, cols);
    }

    for (std::size_t c = 0; c < k; ++c) {
        if (counts[c] == 0) continue;
        scat[c] /= static_cast<double>(counts[c]);
    }

    double db = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        double worst = -std::numeric_limits<double>::infinity();
        const double* mi = means.data() + i * cols;
        for (std::size_t j = 0; j < k; ++j) {
            if (i == j) continue;
            const double* mj = means.data() + j * cols;
            const double m = l2(mi, mj, cols);
            if (m <= 0.0) continue;
            const double r = (scat[i] + scat[j]) / m;
            worst = std::max(worst, r);
        }
        if (!std::isfinite(worst)) {
            throw std::invalid_argument("Davies-Bouldin undefined for degenerate centroids");
        }
        db += worst;
    }

    return db / static_cast<double>(k);
}

double silhouette(
    const double* x,
    std::size_t rows,
    std::size_t cols,
    const std::vector<int>& dense,
    std::size_t k
) {
    if (k < 2 || rows < 2) {
        throw std::invalid_argument("Silhouette requires n_clusters>=2 and n_samples>=2");
    }

    std::vector<std::vector<std::size_t>> members(k);
    for (std::size_t i = 0; i < rows; ++i) {
        members[static_cast<std::size_t>(dense[i])].push_back(i);
    }

    double total = 0.0;
    for (std::size_t i = 0; i < rows; ++i) {
        const int ci = dense[i];
        const auto& own = members[static_cast<std::size_t>(ci)];
        const double* xi = x + i * cols;

        double a = 0.0;
        if (own.size() > 1) {
            for (std::size_t idx : own) {
                if (idx == i) continue;
                a += l2(xi, x + idx * cols, cols);
            }
            a /= static_cast<double>(own.size() - 1);
        }

        double b = std::numeric_limits<double>::infinity();
        for (std::size_t c = 0; c < k; ++c) {
            if (static_cast<int>(c) == ci) continue;
            const auto& grp = members[c];
            if (grp.empty()) continue;
            double avg = 0.0;
            for (std::size_t idx : grp) {
                avg += l2(xi, x + idx * cols, cols);
            }
            avg /= static_cast<double>(grp.size());
            b = std::min(b, avg);
        }

        if (!std::isfinite(b)) {
            throw std::invalid_argument("Silhouette undefined: no valid neighboring cluster");
        }

        const double denom = std::max(a, b);
        const double s = (denom > 0.0) ? ((b - a) / denom) : 0.0;
        total += s;
    }

    return total / static_cast<double>(rows);
}

}  // namespace

double evaluate_cvi(
    const double* x,
    std::size_t rows,
    std::size_t cols,
    const int* labels,
    CVIType cvi_type
) {
    if (x == nullptr || labels == nullptr) {
        throw std::invalid_argument("X and labels must be non-null");
    }
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("X must be non-empty");
    }

    const auto lm = make_dense_labels(labels, rows);
    const std::size_t k = lm.unique_labels.size();

    switch (cvi_type) {
        case CVIType::CalinskiHarabasz:
            return calinski_harabasz(x, rows, cols, lm.dense, k);
        case CVIType::DaviesBouldin:
            return davies_bouldin(x, rows, cols, lm.dense, k);
        case CVIType::Silhouette:
            return silhouette(x, rows, cols, lm.dense, k);
        default:
            throw std::invalid_argument("Unsupported CVI type");
    }
}

}  // namespace artlib_cpp
