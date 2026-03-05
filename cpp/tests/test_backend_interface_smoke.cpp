#include "artlib_cpp/backend_interface.hpp"

#include <cassert>
#include <string>
#include <vector>

namespace {

class StubModel final : public artlib_cpp::BackendModel {
public:
    std::string name() const override {
        return "stub";
    }

    void fit(const artlib_cpp::MatrixView& x) override {
        fitted_rows_ = x.rows;
    }

    std::vector<int> predict(const artlib_cpp::MatrixView& x) const override {
        return std::vector<int>(x.rows, 0);
    }

    std::size_t fitted_rows() const {
        return fitted_rows_;
    }

private:
    std::size_t fitted_rows_ = 0;
};

}  // namespace

int main() {
    std::vector<double> raw(6, 0.0);
    artlib_cpp::MatrixView x{raw.data(), 3, 2};

    StubModel model;
    model.fit(x);
    const auto y = model.predict(x);

    assert(model.name() == "stub");
    assert(model.fitted_rows() == 3);
    assert(y.size() == 3);
    return 0;
}
