#pragma once

#include <cstddef>
#include <vector>

namespace artlib_cpp {

std::vector<int> MapSimpleARTMAPLabels(
    const int* labels_a,
    std::size_t n_labels,
    const int* cluster_labels,
    std::size_t n_clusters
);

}  // namespace artlib_cpp
