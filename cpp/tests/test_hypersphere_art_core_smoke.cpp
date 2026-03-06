#include "artlib_cpp/hypersphere_art_core.hpp"

#include <cassert>
#include <vector>

int main() {
    std::vector<double> x = {
        0.0, 0.0,
        0.1, 0.1,
        5.0, 5.0,
        5.1, 5.2
    };

    artlib_cpp::HypersphereARTParams p{0.7, 1e-10, 1.0, 8.0};
    artlib_cpp::HypersphereARTCore model(p);
    model.fit(x.data(), 4, 2);

    auto y = model.predict(x.data(), 4, 2);
    assert(y.size() == 4);
    assert(!model.weights().empty());
    return 0;
}
