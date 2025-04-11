#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_clamp_(const_reference min_val, const_reference max_val) {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type       i        = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        neon_f32 min_vec = vdupq_n_f32(min_val);
        neon_f32 max_vec = vdupq_n_f32(max_val);

        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 clamped  = vminq_f32(vmaxq_f32(data_vec, min_vec), max_vec);

            vst1q_f32(&data_[i], clamped);
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        neon_s32 min_vec = vdupq_n_s32(min_val);
        neon_s32 max_vec = vdupq_n_s32(max_val);

        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 clamped  = vminq_s32(vmaxq_s32(data_vec, min_vec), max_vec);

            vst1q_s32(&data_[i], clamped);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        neon_u32 min_vec = vdupq_n_u32(min_val);
        neon_u32 max_vec = vdupq_n_u32(max_val);

        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 clamped  = vminq_u32(vmaxq_u32(data_vec, min_vec), max_vec);

            vst1q_u32(&data_[i], clamped);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = std::max(min_val, data_[i]);
        data_[i] = std::min(max_val, data_[i]);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_ceil_() {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    index_type i = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        for (; i + _ARM64_REG_WIDTH <= data_.size(); i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 ceil_vec = vrndpq_f32(data_vec);

            vst1q_f32(&data_[i], ceil_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::ceil(static_cast<_f32>(data_[i])));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_floor_() {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    index_type i = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        for (; i < data_.size(); i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec  = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 floor_vec = vrndmq_f32(data_vec);

            vst1q_f32(&data_[i], floor_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::floor(static_cast<_f32>(data_[i])));
    }

    return *this;
}