#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace artlib_cpp {

struct MatrixView {
    const double* data;
    std::size_t rows;
    std::size_t cols;
};

class BackendModel {
public:
    virtual ~BackendModel() = default;

    virtual std::string name() const = 0;
    virtual void fit(const MatrixView& x) = 0;
    virtual std::vector<int> predict(const MatrixView& x) const = 0;
};

}  // namespace artlib_cpp
