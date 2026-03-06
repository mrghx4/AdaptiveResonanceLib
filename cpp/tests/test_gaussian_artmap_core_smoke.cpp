#include "artlib_cpp/gaussian_artmap_core.hpp"

#include <cassert>
#include <vector>

int main() {
    std::vector<double> x = {
        0.0, 0.0,
        0.05, 0.02,
        5.0, 5.0,
        5.1, 4.95
    };
    std::vector<int> y = {0, 0, 1, 1};

    artlib_cpp::GaussianARTMAPParams p{0.05, 1e-10, {0.5, 0.5}, "MT+", 1e-10};
    artlib_cpp::GaussianARTMAPCore model(p);
    model.fit(x.data(), 4, 2, y.data(), y.size());

    auto [ya, yb] = model.predict(x.data(), 4, 2);
    assert(ya.size() == 4);
    assert(yb.size() == 4);
    assert(!model.weights().empty());
    return 0;
}
