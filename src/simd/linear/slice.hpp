#pragma once

#include "tensorbase.hpp"

/*
template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_slice(index_type                dimension,
                                    std::optional<index_type> start,
                                    std::optional<index_type> end,
                                    index_type                step) const {
    if (dimension < 0 or dimension >= static_cast<index_type>(shape_.size()))
    {
        throw index_error("Dimension out of range.");
    }

    if (step == 0)
    {
        throw std::invalid_argument("Step cannot be zero.");
    }

    index_type s       = shape_[dimension];
    index_type start_i = start.value_or(0);
    index_type end_i   = end.value_or(s);

    if (start_i < 0)
    {
        start_i += s;
    }

    if (end_i < 0)
    {
        end_i += s;
    }

    start_i               = std::max(index_type(0), std::min(start_i, s));
    end_i                 = std::max(index_type(0), std::min(end_i, s));
    index_type slice_size = (end_i - start_i + step - 1) / step;
    shape_type ret_dims   = shape_;
    ret_dims[dimension]         = slice_size;
    tensor ret(ret_dims);

    index_type vector_end = start_i + ((end_i - start_i) / _ARM64_REG_WIDTH) * _ARM64_REG_WIDTH;

    if constexpr (std::is_floating_point_v<value_type> and step == 1)
    {
        for (index_type i = start_i, j = 0; i < vector_end; i += _ARM64_REG_WIDTH, j += _ARM64_REG_WIDTH)
        {
            neon_f32 vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            vst1q_f32(&(ret.data_[j]), vec);
        }
    }
    else if (std::is_signed_v<value_type> and step == 1)
    {
        for (index_type i = start_i, j = 0; i < vector_end; i += _ARM64_REG_WIDTH, j += _ARM64_REG_WIDTH)
        {
            neon_s32 vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            vst1q_s32(&(ret.data_[j]), vec);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type> and step == 1)
    {
        for (index_type i = start_i, j = 0; i < vector_end; i += _ARM64_REG_WIDTH, j += _ARM64_REG_WIDTH)
        {
            neon_u32 vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            vst1q_u32(&(ret.data_[j]), vec);
        }
    }

    // Handle remaining elements
    index_type remaining = (end_i - start_i) % _ARM64_REG_WIDTH;
    if (remaining > 0)
    {
        for (index_type i = vector_end, j = vector_end - start_i; i < end_i; ++i, ++j)
        {
            ret.data_[j] = data_[i];
        }
    }

    for (index_type i = start_i; i < end_i; i += step)
    {
        index_type j = (i - start_i) / step;
        ret[j]       = data_[i];
    }

    return ret;
}
*/