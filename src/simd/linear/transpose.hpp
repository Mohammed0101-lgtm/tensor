#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_transpose() const {
    if (!equal_shape(shape_, shape_type({shape_[0], shape_[1]})))
        throw shape_error("Matrix transposition can only be done on 2D tensors");

    tensor           ret({shape_[1], shape_[0]});
    const index_type rows = shape_[0];
    const index_type cols = shape_[1];

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < rows; i += simd_width)
    {
        for (index_type j = 0; j < cols; j += simd_width)
        {
            if (i + simd_width <= rows and j + simd_width <= cols)
            {
                wide_neon_type<value_type> input;

                for (index_type k = 0; k < simd_width; ++k)
                {
                    input[k] = neon_load<value_type>(&data_[(i + k) * cols + j]);
                }

                wide_neon_type<value_type> output = wide_neon_load<value_type>(&input);

                for (index_type k = 0; k < simd_width; ++k)
                {
                    neon_store<value_type>(&ret[(j + k) * rows + i], output[k]);
                }
            }
            else
            {
                for (index_type ii = i; ii < std::min(static_cast<index_type>(i + simd_width), rows); ++ii)
                {
                    for (index_type jj = j; jj < std::min(static_cast<index_type>(j + simd_width), cols); ++jj)
                    {
                        ret.at({jj, ii}) = at({ii, jj});
                    }
                }
            }
        }
    }

    for (; i < rows; ++i)
    {
        index_type j = 0;
        for (; j < cols; ++j)
        {
            ret.at({j, i}) = at({i, j});
        }
    }

    return ret;
}
