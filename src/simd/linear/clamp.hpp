#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_clamp_(const_reference min_val, const_reference max_val) {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    neon_type<value_type> min_vec = neon_dup<value_type>(min_val);
    neon_type<value_type> max_vec = neon_dup<value_type>(max_val);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> data_vec = neon_load<value_type>(&data_[i]);
        neon_type<value_type> clamped  = neon_min<value_type>(neon_max<value_type>(data_vec, min_vec), max_vec);

        neon_store<value_type>(&data_[i], clamped);
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