#pragma once

#include <cstddef>
#include <vector>

namespace artlib_cpp {

enum class CVIType : int {
    CalinskiHarabasz = 1,
    DaviesBouldin = 2,
    Silhouette = 3,
};

double evaluate_cvi(
    const double* x,
    std::size_t rows,
    std::size_t cols,
    const int* labels,
    CVIType cvi_type
);

}  // namespace artlib_cpp
