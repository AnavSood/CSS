#pragma once
#include <Eigen/Dense>
#include <vector>

namespace css {

template <class CType, class PrintFType>
void _css_step(
    CType& C,
    double rsq0,
    Eigen::Index& i_max,
    double& rsq,
    PrintFType print_f
)
{
    const size_t p = C.rows();
    const auto dC = C.diagonal();
    Eigen::VectorXd obj = C.array().square().matrix().colwise().sum();
    obj.array() /= dC.array();
    obj.maxCoeff(&i_max);
    const auto vmin = dC.sum() - obj[i_max];
    rsq = 1.0 - vmin / rsq0;
}


template <class CType, class PrintFType>
void css_reg(
    CType C,
    size_t k,
    std::vector<size_t>& subset,
    std::vector<double>& rsq_subset,
    PrintFType& print_f
)
{
    if (k == 0) return;
    
    const size_t p = C.rows();
    k = std::min(k, p);

    subset.clear();
    rsq_subset.clear();
    
    std::vector<size_t> order(p);
    std::iota(order.begin(), order.end(), 0);

    const double rsq0 = C.diagonal().sum();
    
    size_t next_size = p;

    for (size_t i = 0; i < k; ++i) {
        // apply forward-step with current residual matrix
        auto C_sub = C.block(0, 0, next_size, next_size);
        Eigen::Index i_max;
        double rsq;
        _css_step(C_sub, rsq0, i_max, rsq, print_f);
        
        // save results
        subset.push_back(order[i_max]);
        rsq_subset.push_back(rsq);

        // switch near-0 residual row/col with last index
        int last_index = C_sub.cols();
        for (int j = 0; j < last_index; ++j) {
            if ((j != last_index-1) && (C_sub(j,j) <= 1e-10)) {
                --last_index;
                C_sub.col(j).swap(C_sub.col(last_index));
                C_sub.row(j).swap(C_sub.row(last_index));
                std::swap(order[j], order[last_index]);
            }
        }
        next_size = last_index;

        // exit early if R^2 is too high
        //if (rsq >= 1.0-1e-3) break;
    }
}

} // namespace css