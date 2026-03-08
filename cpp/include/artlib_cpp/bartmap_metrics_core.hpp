#pragma once

#include <cstddef>

namespace artlib_cpp {

double average_pearson_corr(
    const double* x,
    std::size_t rows,
    std::size_t cols,
    const int* column_labels,
    std::size_t labels_len,
    std::size_t k,
    int c_b
);

bool any_cluster_match(
    const double* x,
    std::size_t rows,
    std::size_t cols,
    const int* column_labels,
    std::size_t labels_len,
    std::size_t k,
    std::size_t n_clusters_b,
    double eta
);

}  // namespace artlib_cpp
