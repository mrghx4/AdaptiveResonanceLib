#include "artlib_cpp/simple_artmap_core.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

namespace {

template <typename Fn>
void expect_out_of_range(Fn fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw);
}

template <typename Fn>
void expect_invalid_argument(Fn fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

}  // namespace

int main() {
    const std::vector<int> labels = {0, 2, 1, 2, 0};
    const std::vector<int> cluster_labels = {10, 20, 30};

    const auto mapped = artlib_cpp::MapSimpleARTMAPLabels(
        labels.data(), labels.size(), cluster_labels.data(), cluster_labels.size());
    assert((mapped == std::vector<int>{10, 30, 20, 30, 10}));

    const std::vector<double> centers = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    };
    const auto gathered = artlib_cpp::GatherClusterCenters(
        labels.data(), labels.size(), centers.data(), 3, 2);
    assert((gathered == std::vector<double>{
        1.0, 2.0,
        5.0, 6.0,
        3.0, 4.0,
        5.0, 6.0,
        1.0, 2.0,
    }));

    const std::vector<double> centers_b = {7.0, 8.0, 9.0};
    const std::vector<const double*> centers_list = {centers.data(), centers_b.data()};
    const std::vector<std::size_t> n_centers = {3, 3};
    const std::vector<std::size_t> dims = {2, 1};
    const auto batch = artlib_cpp::GatherClusterCentersBatch(
        labels.data(), labels.size(), centers_list, n_centers, dims);
    assert(batch.size() == 2);
    assert(batch[0] == gathered);
    assert((batch[1] == std::vector<double>{7.0, 9.0, 8.0, 9.0, 7.0}));

    const std::vector<std::vector<int>> chain = {
        {1, 0, 2},
        {9, 8, 7},
    };
    const auto chain_final = artlib_cpp::MapSimpleARTMAPLabelsChain(
        labels.data(), labels.size(), chain);
    assert((chain_final == std::vector<int>{8, 7, 9, 7, 8}));

    const auto chain_levels = artlib_cpp::MapSimpleARTMAPLabelsChainLevels(
        labels.data(), labels.size(), chain);
    assert(chain_levels.size() == 2);
    assert((chain_levels[0] == std::vector<int>{1, 2, 0, 2, 1}));
    assert(chain_levels[1] == chain_final);

    expect_out_of_range([&]() {
        const std::vector<int> bad = {3};
        (void)artlib_cpp::MapSimpleARTMAPLabels(
            bad.data(), bad.size(), cluster_labels.data(), cluster_labels.size());
    });
    expect_out_of_range([&]() {
        const std::vector<int> bad = {-1};
        (void)artlib_cpp::GatherClusterCenters(
            bad.data(), bad.size(), centers.data(), 3, 2);
    });
    expect_invalid_argument([&]() {
        const std::vector<const double*> bad_centers = {centers.data()};
        (void)artlib_cpp::GatherClusterCentersBatch(
            labels.data(), labels.size(), bad_centers, {}, {});
    });
    expect_invalid_argument([&]() {
        const std::vector<std::vector<int>> bad_chain = {{0, 1, 2}, {}};
        (void)artlib_cpp::MapSimpleARTMAPLabelsChain(
            labels.data(), labels.size(), bad_chain);
    });

    return 0;
}
