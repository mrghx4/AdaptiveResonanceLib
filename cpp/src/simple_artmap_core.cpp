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

std::vector<double> GatherClusterCenters(
    const int* labels,
    std::size_t n_labels,
    const double* centers,
    std::size_t n_centers,
    std::size_t center_dim
) {
    if (labels == nullptr) {
        throw std::invalid_argument("labels cannot be null");
    }
    if (centers == nullptr) {
        throw std::invalid_argument("centers cannot be null");
    }
    if (center_dim == 0) {
        throw std::invalid_argument("center_dim must be > 0");
    }

    std::vector<double> out(n_labels * center_dim, 0.0);
    for (std::size_t i = 0; i < n_labels; ++i) {
        const int c = labels[i];
        if (c < 0 || static_cast<std::size_t>(c) >= n_centers) {
            throw std::out_of_range("label out of range for centers");
        }
        const std::size_t src_off = static_cast<std::size_t>(c) * center_dim;
        const std::size_t dst_off = i * center_dim;
        for (std::size_t j = 0; j < center_dim; ++j) {
            out[dst_off + j] = centers[src_off + j];
        }
    }
    return out;
}

std::vector<std::vector<double>> GatherClusterCentersBatch(
    const int* labels,
    std::size_t n_labels,
    const std::vector<const double*>& centers_list,
    const std::vector<std::size_t>& n_centers_list,
    const std::vector<std::size_t>& center_dims
) {
    if (labels == nullptr) {
        throw std::invalid_argument("labels cannot be null");
    }
    if (centers_list.size() != n_centers_list.size() ||
        centers_list.size() != center_dims.size()) {
        throw std::invalid_argument("batch center metadata lengths must match");
    }

    std::vector<std::vector<double>> out;
    out.reserve(centers_list.size());
    for (std::size_t i = 0; i < centers_list.size(); ++i) {
        out.push_back(
            GatherClusterCenters(
                labels,
                n_labels,
                centers_list[i],
                n_centers_list[i],
                center_dims[i]
            )
        );
    }
    return out;
}

std::vector<int> MapSimpleARTMAPLabelsChain(
    const int* labels,
    std::size_t n_labels,
    const std::vector<std::vector<int>>& map_chain
) {
    if (labels == nullptr) {
        throw std::invalid_argument("labels cannot be null");
    }

    std::vector<int> out(labels, labels + n_labels);
    for (const auto& map_labels : map_chain) {
        if (map_labels.empty()) {
            throw std::invalid_argument("map chain entries must be non-empty");
        }
        for (std::size_t i = 0; i < n_labels; ++i) {
            const int c = out[i];
            if (c < 0 || static_cast<std::size_t>(c) >= map_labels.size()) {
                throw std::out_of_range("label out of range for map chain");
            }
            out[i] = map_labels[static_cast<std::size_t>(c)];
        }
    }
    return out;
}

std::vector<std::vector<int>> MapSimpleARTMAPLabelsChainLevels(
    const int* labels,
    std::size_t n_labels,
    const std::vector<std::vector<int>>& map_chain
) {
    if (labels == nullptr) {
        throw std::invalid_argument("labels cannot be null");
    }

    std::vector<std::vector<int>> out;
    out.reserve(map_chain.size());
    std::vector<int> current(labels, labels + n_labels);
    for (const auto& map_labels : map_chain) {
        if (map_labels.empty()) {
            throw std::invalid_argument("map chain entries must be non-empty");
        }
        for (std::size_t i = 0; i < n_labels; ++i) {
            const int c = current[i];
            if (c < 0 || static_cast<std::size_t>(c) >= map_labels.size()) {
                throw std::out_of_range("label out of range for map chain");
            }
            current[i] = map_labels[static_cast<std::size_t>(c)];
        }
        out.push_back(current);
    }
    return out;
}

}  // namespace artlib_cpp
