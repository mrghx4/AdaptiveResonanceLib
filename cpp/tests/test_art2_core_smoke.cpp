#include "artlib_cpp/art2_core.hpp"

#include <cassert>
#include <vector>

int main() {
    std::vector<double> x = {
        0.0, 0.0,
        0.05, 0.02,
        0.95, 0.85,
        0.90, 0.92
    };

    artlib_cpp::ART2Params p{0.7, 0.1, 0.5};
    artlib_cpp::ART2Core model(p);
    model.fit(x.data(), 4, 2);

    auto y = model.predict(x.data(), 4, 2);
    assert(y.size() == 4);
    assert(!model.weights().empty());
    return 0;
}
