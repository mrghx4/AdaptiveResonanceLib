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

std::vector<double> GatherClusterCenters(
    const int* labels,
    std::size_t n_labels,
    const double* centers,
    std::size_t n_centers,
    std::size_t center_dim
);

}  // namespace artlib_cpp
