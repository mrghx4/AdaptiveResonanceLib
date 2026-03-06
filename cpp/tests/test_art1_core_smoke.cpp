#include "artlib_cpp/art1_core.hpp"

#include <cassert>
#include <vector>

int main() {
    const std::vector<double> raw = {
        1.0, 0.0, 1.0,
        1.0, 1.0, 0.0,
        0.0, 1.0, 1.0
    };

    artlib_cpp::MatrixView x{raw.data(), 3, 3};
    artlib_cpp::ART1Params p{0.8, 1.0};
    artlib_cpp::ART1Core model(p);

    model.fit(x);
    const auto y = model.predict(x);

    assert(y.size() == 3);
    assert(model.labels().size() == 3);
    assert(!model.weights().empty());
    return 0;
}
