#include "artlib_cpp/binary_fuzzy_art_core.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    const std::vector<std::uint8_t> x = {
        1, 0, 1, 0,
        1, 1, 0, 0,
        0, 1, 0, 1
    };

    artlib_cpp::BinaryFuzzyARTCore model(0.8);
    model.fit(x.data(), 3, 4);
    const auto y = model.predict(x.data(), 3, 4);

    assert(y.size() == 3);
    assert(model.labels().size() == 3);
    assert(!model.weights().empty());
    return 0;
}
