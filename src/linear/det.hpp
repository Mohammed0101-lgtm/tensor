#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::det() const {
    if (shape_.size() < 2)
    {
        throw shape_error("det: tensor must be at least 2D");
    }

    index_type h, w;
    if (shape_.size() == 2)
    {
        h = shape_[0];
        w = shape_[1];
    }
    else
    {
        index_type last        = shape_.size() - 1;
        index_type second_last = shape_.size() - 2;

        if (shape_[last] == 1)
        {
            h = shape_[second_last - 1];
            w = shape_[second_last];
        }
        else if (shape_[second_last] == 1)
        {
            h = shape_[last - 1];
            w = shape_[last];
        }
        else
        {
            h = shape_[second_last];
            w = shape_[last];
        }
    }

    if (h != w)
    {
        throw shape_error("det: tensor must be a square matrix (n x n)");
    }

    index_type n = h;

    if (n == 2)
    {
        return tensor<_Tp>({1}, {(*this)({0, 0}) * (*this)({1, 1}) - (*this)({0, 1}) * (*this)({1, 0})});
    }

    value_type determinant = 0;

    for (index_type col = 0; col < n; ++col)
    {
        tensor<_Tp> minor = get_minor(0, col);
        value_type  sign  = (col % 2 == 0) ? 1 : -1;
        determinant += sign * (*this)({0, col}) * minor.det()[0];
    }

    return tensor<_Tp>({1}, {determinant});
}