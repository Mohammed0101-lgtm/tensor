#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sqrt_() {
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

            vals[0] = static_cast<_f32>(std::sqrt(vals[0]));
            vals[1] = static_cast<_f32>(std::sqrt(vals[1]));
            vals[2] = static_cast<_f32>(std::sqrt(vals[2]));
            vals[3] = static_cast<_f32>(std::sqrt(vals[3]));

            neon_f32 sqrt_vec = vld1q_f32(vals);
            vst1q_f32(&data_[i], sqrt_vec);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            _s32     vals[_ARM64_REG_WIDTH];
            vst1q_s32(vals, data_vec);

            vals[0] = static_cast<_s32>(std::sqrt(vals[0]));
            vals[1] = static_cast<_s32>(std::sqrt(vals[1]));
            vals[2] = static_cast<_s32>(std::sqrt(vals[2]));
            vals[3] = static_cast<_s32>(std::sqrt(vals[3]));

            neon_s32 sqrt_vec = vld1q_s32(vals);
            vst1q_s32(&data_[i], sqrt_vec);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            _u32     vals[_ARM64_REG_WIDTH];
            vst1q_u32(vals, data_vec);

            vals[0] = static_cast<_u32>(std::sqrt(vals[0]));
            vals[1] = static_cast<_u32>(std::sqrt(vals[1]));
            vals[2] = static_cast<_u32>(std::sqrt(vals[2]));
            vals[3] = static_cast<_u32>(std::sqrt(vals[3]));

            neon_u32 sqrt_vec = vld1q_u32(vals);
            vst1q_u32(&data_[i], sqrt_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::sqrt(data_[i]));
    }

    return *this;
}