#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sum(const index_type axis) const {
    if (axis < 0 || axis >= static_cast<index_type>(shape_.size()))
    {
        throw index_error("Invalid axis for sum");
    }

    shape_type ret_sh   = shape_;
    ret_sh[axis]        = 1;
    index_type ret_size = std::accumulate(ret_sh.begin(), ret_sh.end(), 1, std::multiplies<index_type>());
    data_t     ret_data(ret_size, value_type(0.0f));

    index_type i = 0;
    for (; i < static_cast<index_type>(data_.size()); ++i)
    {
        std::vector<index_type> orig(shape_.size());
        index_type              index = i;
        index_type              j     = static_cast<index_type>(shape_.size()) - 1;

        for (; j >= 0; j--)
        {
            orig[j] = index % shape_[j];
            index /= shape_[j];
        }

        orig[axis]           = 0;
        index_type ret_index = 0;
        index_type st        = 1;

        for (j = static_cast<index_type>(shape_.size()) - 1; j >= 0; j--)
        {
            ret_index += orig[j] * st;
            st *= ret_sh[j];
        }
        ret_data[ret_index] += data_[i];
    }

    return self(ret_data, ret_sh);
}