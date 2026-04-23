#include "artlib_cpp/fraction_sort_core.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    std::vector<fracsort::Item<std::uint32_t>> items = {
        {1, 2, 0, 1, 0},
        {2, 4, 0, 1, 1},
        {0, 1, 0, 1, 2},
        {2, 3, 0, 1, 3},
        {2, 4, 0, 1, 4},
    };

    fracsort::argsort_items_inplace(items.data(), items.size());
    const std::vector<std::size_t> expected_order = {3, 1, 4, 0, 2};
    for (std::size_t i = 0; i < items.size(); ++i) {
        assert(items[i].idx == expected_order[i]);
    }

    std::vector<fracsort::Item<std::uint32_t>> argmax_items = {
        {1, 2, 0, 1, 0},
        {2, 4, 0, 1, 1},
        {2, 4, 0, 1, 2},
    };
    assert(fracsort::fracargmax_items(argmax_items.data(), argmax_items.size()) == 1);
    assert(fracsort::fracargmax_items<std::uint32_t>(nullptr, 0) == 0);

#if FRACSORT_HAS_U64
    std::vector<fracsort::Item<std::uint64_t>> wide_items = {
        {4294967296ULL, 8589934592ULL, 0, 1, 0},
        {1ULL, 3ULL, 0, 1, 1},
        {2ULL, 3ULL, 0, 1, 2},
    };
    fracsort::argsort_items_inplace(wide_items.data(), wide_items.size());
    assert(wide_items[0].idx == 2);
    assert(wide_items[1].idx == 0);
    assert(wide_items[2].idx == 1);
#endif

    return 0;
}
