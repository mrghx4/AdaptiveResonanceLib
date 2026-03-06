#include "artlib_cpp/art1map_core.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    const std::vector<std::int16_t> x = {
        1, 0, 1,
        1, 1, 0,
        0, 1, 1
    };
    const std::vector<int> y = {0, 0, 1};

    artlib_cpp::ART1MAPParams p{0.9, 1.0, "MT+", 1e-10};
    artlib_cpp::ART1MAPCore model(p);
    model.fit(x.data(), 3, 3, y.data(), y.size());

    auto [ya, yb] = model.predict(x.data(), 3, 3);
    assert(ya.size() == 3);
    assert(yb.size() == 3);
    assert(!model.weights().empty());
    return 0;
}
