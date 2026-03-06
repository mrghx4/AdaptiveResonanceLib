#include "artlib_cpp/fuzzy_artmap_core.hpp"

#include <cassert>
#include <vector>

int main() {
    std::vector<double> x = {
        0.9, 0.1, 0.1, 0.9,
        0.85, 0.15, 0.15, 0.85,
        0.1, 0.9, 0.9, 0.1
    };
    std::vector<int> y = {0, 0, 1};

    artlib_cpp::FuzzyARTMAPParams p{0.8, 1e-10, 1.0, "MT+", 1e-10};
    artlib_cpp::FuzzyARTMAPCore model(p);
    model.fit(x.data(), 3, 4, y.data(), y.size());
    auto [ya, yb] = model.predict(x.data(), 3, 4);

    assert(ya.size() == 3);
    assert(yb.size() == 3);
    assert(!model.weights().empty());
    return 0;
}
