#pragma once

#include "tensor.hpp"


namespace internal::simd::neon {

/*
template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_slice(_u64                dimension,
                                    std::optional<_u64> start,
                                    std::optional<_u64> end,
                                    _u64                step) const {
    if (dimension < 0 or dimension >= static_cast<_u64>(this->n_dims()))
    {
        throw error::index_error("Dimension out of range.");
    }

    if (step == 0)
    {
        throw std::invalid_argument("Step cannot be zero.");
    }

    _u64 s       = shape_[dimension];
    _u64 start_i = start.value_or(0);
    _u64 end_i   = end.value_or(s);

    if (start_i < 0)
    {
        start_i += s;
    }

    if (end_i < 0)
    {
        end_i += s;
    }

    start_i               = std::max(_u64(0), std::min(start_i, s));
    end_i                 = std::max(_u64(0), std::min(end_i, s));
    _u64 slice_size = (end_i - start_i + step - 1) / step;
    shape_type ret_dims   = shape_;
    ret_dims[dimension]         = slice_size;
    tensor ret(ret_dims);

    _u64 vector_end = start_i + ((end_i - start_i) / _ARM64_REG_WIDTH) * _ARM64_REG_WIDTH;

    if constexpr (std::is_floating_point_v<_Tp> and step == 1)
    {
        for (_u64 i = start_i, j = 0; i < vector_end; i += _ARM64_REG_WIDTH, j += _ARM64_REG_WIDTH)
        {
            neon_f32 vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            vst1q_f32(&(ret.data_[j]), vec);
        }
    }
    else if (std::is_signed_v<_Tp> and step == 1)
    {
        for (_u64 i = start_i, j = 0; i < vector_end; i += _ARM64_REG_WIDTH, j += _ARM64_REG_WIDTH)
        {
            neon_s32 vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            vst1q_s32(&(ret.data_[j]), vec);
        }
    }
    else if constexpr (std::is_unsigned_v<_Tp> and step == 1)
    {
        for (_u64 i = start_i, j = 0; i < vector_end; i += _ARM64_REG_WIDTH, j += _ARM64_REG_WIDTH)
        {
            neon_u32 vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            vst1q_u32(&(ret.data_[j]), vec);
        }
    }

    // Handle remaining elements
    _u64 remaining = (end_i - start_i) % _ARM64_REG_WIDTH;
    if (remaining > 0)
    {
        for (_u64 i = vector_end, j = vector_end - start_i; i < end_i; ++i, ++j)
        {
            ret.data_[j] = data_[i];
        }
    }

    for (_u64 i = start_i; i < end_i; i += step)
    {
        _u64 j = (i - start_i) / step;
        ret[j]       = data_[i];
    }

    return ret;
}
*/

}