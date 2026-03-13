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

std::vector<std::vector<double>> GatherClusterCentersBatch(
    const int* labels,
    std::size_t n_labels,
    const std::vector<const double*>& centers_list,
    const std::vector<std::size_t>& n_centers_list,
    const std::vector<std::size_t>& center_dims
);

std::vector<int> MapSimpleARTMAPLabelsChain(
    const int* labels,
    std::size_t n_labels,
    const std::vector<std::vector<int>>& map_chain
);

std::vector<std::vector<int>> MapSimpleARTMAPLabelsChainLevels(
    const int* labels,
    std::size_t n_labels,
    const std::vector<std::vector<int>>& map_chain
);

}  // namespace artlib_cpp
