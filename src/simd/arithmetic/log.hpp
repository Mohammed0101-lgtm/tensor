#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_log_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    index_type simd_end = data_.size() - (data_.size() % simd_width);
    index_type i        = 0;

    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec = neon_load<value_type>(&data_[i]);
        value_type            vals[simd_width];
        neon_store<value_type>(vals, vec);

        for (int j = 0; j < simd_width; j++)
        {
            vals[j] = static_cast<value_type>(std::log(vals[j]));
        }

        neon_type<value_type> log_vec = neon_load<value_type>(vals);
        neon_store<value_type>(data_[i], log_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::log(data_[i]));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_log10_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    index_type simd_end = data_.size() - (data_.size() % simd_width);
    index_type i        = 0;

    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec = neon_load<value_type>(&data_[i]);
        value_type            vals[simd_width];
        neon_store<value_type>(vals, vec);

        for (int j = 0; j < simd_width; i++)
        {
            vals[0] = static_cast<value_type>(std::log10(vals[j]));
        }

        neon_type<value_type> log_vec = neon_load<value_type>(&vals);
        neon_store<value_type>(&data_[i], log_vec);
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<value_type>(std::log10(data_[i]));

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_log2_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    index_type i = 0;

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            _f32     vals[_ARM64_REG_WIDTH];
            vst1q_f32(vals, data_vec);

            vals[0] = static_cast<_f32>(std::log2(vals[0]));
            vals[1] = static_cast<_f32>(std::log2(vals[1]));
            vals[2] = static_cast<_f32>(std::log2(vals[2]));
            vals[3] = static_cast<_f32>(std::log2(vals[3]));

            neon_f32 log2_vec = vld1q_f32(vals);
            vst1q_f32(&data_[i], log2_vec);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            _u32     vals[_ARM64_REG_WIDTH];
            vst1q_u32(vals, data_vec);

            vals[0] = static_cast<_u32>(std::log2(vals[0]));
            vals[1] = static_cast<_u32>(std::log2(vals[1]));
            vals[2] = static_cast<_u32>(std::log2(vals[2]));
            vals[3] = static_cast<_u32>(std::log2(vals[3]));

            neon_u32 log2_vec = vld1q_u32(vals);
            vst1q_u32(&data_[i], log2_vec);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            _s32     vals[_ARM64_REG_WIDTH];
            vst1q_s32(vals, data_vec);

            vals[0] = static_cast<_s32>(std::log2(vals[0]));
            vals[1] = static_cast<_s32>(std::log2(vals[1]));
            vals[2] = static_cast<_s32>(std::log2(vals[2]));
            vals[3] = static_cast<_s32>(std::log2(vals[3]));

            neon_s32 log2_vec = vld1q_s32(vals);
            vst1q_s32(&data_[i], log2_vec);
        }
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<value_type>(std::log2(data_[i]));

    return *this;
}
