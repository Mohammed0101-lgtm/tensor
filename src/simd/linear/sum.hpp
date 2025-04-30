#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_sum(const index_type axis) const {
    if (axis < 0 || axis >= static_cast<index_type>(shape_.size()))
    {
        throw std::invalid_argument("Invalid axis for sum");
    }

    shape_type ret_sh   = shape_;
    ret_sh[axis]        = 1;
    index_type ret_size = std::accumulate(ret_sh.begin(), ret_sh.end(), 1, std::multiplies<index_type>());
    data_t     ret_data(ret_size, value_type(0.0f));

    const index_type axis_size  = shape_[axis];
    const index_type outer_size = compute_outer_size(axis);
    const index_type inner_size = size(0) / (outer_size * axis_size);
    const index_type simd_end   = data_.size() - (data_.size() % simd_width);

    for (index_type outer = 0; outer < outer_size; ++outer)
    {
        for (index_type inner = 0; inner < inner_size; ++inner)
        {
            neon_type<value_type> sum_vec = neon_dup<value_type>(value_type(0.0f));
            index_type            i       = outer * axis_size * inner_size + inner;
            index_type            j       = 0;

            for (; j < axis_size; j += simd_width)
            {
                neon_type<value_type> data_vec = neon_load<value_type>(&data_[i]);
                sum_vec                        = neon_add<value_type>(sum_vec, data_vec);
                i += inner_size * simd_width;
            }

            value_type sum = neon_addv<value_type>(sum_vec);

            for (; j < axis_size; ++j)
            {
                sum = sum + data_[i];
                i += inner_size;
            }

            ret_data[outer * inner_size + inner] = sum;
        }
    }

    return self(ret_data, ret_sh);
}