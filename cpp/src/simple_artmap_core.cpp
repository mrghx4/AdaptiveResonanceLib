#include "artlib_cpp/simple_artmap_core.hpp"

#include <stdexcept>
#include <vector>

namespace artlib_cpp {

std::vector<int> MapSimpleARTMAPLabels(
    const int* labels_a,
    std::size_t n_labels,
    const int* cluster_labels,
    std::size_t n_clusters
) {
    if (labels_a == nullptr) {
        throw std::invalid_argument("labels_a cannot be null");
    }
    if (cluster_labels == nullptr) {
        throw std::invalid_argument("cluster_labels cannot be null");
    }

    std::vector<int> labels_b(n_labels, 0);
    for (std::size_t i = 0; i < n_labels; ++i) {
        const int c_a = labels_a[i];
        if (c_a < 0 || static_cast<std::size_t>(c_a) >= n_clusters) {
            throw std::out_of_range("A-side label out of range for cluster_labels");
        }
        labels_b[i] = cluster_labels[c_a];
    }
    return labels_b;
}

}  // namespace artlib_cpp
