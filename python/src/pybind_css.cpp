#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "css.hpp"

void css_reg(
    Eigen::MatrixXd C,
    size_t k,
    std::vector<size_t>& subset,
    std::vector<double>& rsq_subset
)
{
    const auto dummy = [](){};
    css::css_reg(C, k, subset, rsq_subset, dummy); 
}


PYBIND11_MODULE(css, m) {
    m.doc() = "pybind11 plugin for css"; // optional module docstring

    m.def("css_reg", &css_reg, "Plain vanilla CSS.");
}