#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <vector>
#include "css.hpp"

std::vector<size_t> css_reg(
    Eigen::MatrixXd C,
    size_t k
)
{
    std::vector<size_t> subset;
    std::vector<double> rsq_subset;
    subset.reserve(k);
    rsq_subset.reserve(k);
    const auto dummy = [](){};
    css::css_reg(C, k, subset, rsq_subset, dummy); 
    return subset;
}


PYBIND11_MODULE(pycss_core, m) {
    m.doc() = "pybind11 plugin for css"; // optional module docstring

    m.def("css_reg", &css_reg, "Plain vanilla CSS.");
}