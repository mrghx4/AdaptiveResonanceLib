#include "artlib_cpp/binary_fuzzy_artmap_core.hpp"

#include <cassert>
#include <vector>

int main() {
    std::vector<int> x = {
        1, 0, 1, 0,
        1, 1, 0, 0,
        0, 1, 0, 1
    };
    std::vector<int> y = {0, 0, 1};

    artlib_cpp::BinaryFuzzyARTMAPParams p{0.8, "MT+", 1};
    artlib_cpp::BinaryFuzzyARTMAPCore model(p);
    model.fit(x.data(), 3, 4, y.data(), y.size());
    auto [ya, yb] = model.predict(x.data(), 3, 4);

    assert(ya.size() == 3);
    assert(yb.size() == 3);
    assert(!model.weights().empty());
    return 0;
}
