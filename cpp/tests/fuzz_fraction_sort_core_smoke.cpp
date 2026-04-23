#include "artlib_cpp/fraction_sort_core.hpp"

#include <cassert>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

namespace {

bool fraction_ge(const fracsort::Item<std::uint32_t>& a, const fracsort::Item<std::uint32_t>& b) {
    const std::uint64_t left = static_cast<std::uint64_t>(a.num) * b.den;
    const std::uint64_t right = static_cast<std::uint64_t>(b.num) * a.den;
    return left >= right;
}

}  // namespace

int main() {
    std::mt19937 rng(4321);
    std::uniform_int_distribution<std::uint32_t> den_dist(1, 100000);

    for (int iter = 0; iter < 200; ++iter) {
        const std::size_t n = 1 + static_cast<std::size_t>(iter % 64);
        std::vector<fracsort::Item<std::uint32_t>> items;
        items.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t den = den_dist(rng);
            std::uniform_int_distribution<std::uint32_t> num_dist(0, den);
            const std::uint32_t num = num_dist(rng);
            items.push_back({num, den, 0, 1, i});
        }

        fracsort::argsort_items_inplace(items.data(), items.size());
        std::set<std::size_t> seen;
        for (std::size_t i = 0; i < items.size(); ++i) {
            assert(seen.insert(items[i].idx).second);
            if (i > 0) {
                assert(fraction_ge(items[i - 1], items[i]));
            }
        }
    }

    return 0;
}
