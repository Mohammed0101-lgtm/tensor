#pragma once

#include "../alias.hpp"
#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& internal::neon::clamp_(tensor<_Tp>& t, const _Tp& min_val, const _Tp& max_val) {
    if (!std::is_arithmetic_v<_Tp>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    neon_type<_Tp>    min_vec  = neon_dup<_Tp>(min_val);
    neon_type<_Tp>    max_vec  = neon_dup<_Tp>(max_val);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> data_vec = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> max_vec_ = neon_max<_Tp>(data_vec, min_vec);
        neon_type<_Tp> clamped  = neon_min<_Tp>(max_vec_, max_vec);
        neon_store<_Tp>(&data_[i], clamped);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = std::max(min_val, data_[i]);
        data_[i] = std::min(max_val, data_[i]);
    }

    return t;
}


template<class _Tp>
tensor<_Tp>& internal::neon::ceil_(tensor<_Tp>& t) {
    if (!std::is_floating_point_v<_Tp>)
    {
        throw error::type_error("Type must be floating point");
    }

    std::vector<_Tp>& data_ = t.storage_();
    _u64              i     = 0;

    if constexpr (std::is_floating_point_v<_Tp>)
    {
        for (; i + t.simd_width <= data_.size(); i += t.simd_width)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 ceil_vec = vrndpq_f32(data_vec);
            vst1q_f32(&data_[i], ceil_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<_Tp>(std::ceil(static_cast<_f32>(data_[i])));
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::floor_(tensor<_Tp>& t) {
    if (!std::is_floating_point_v<_Tp>)
    {
        throw error::type_error("Type must be floating point");
    }

    std::vector<_Tp>& data_ = t.storage_();
    _u64              i     = 0;

    if constexpr (std::is_floating_point_v<_Tp>)
    {
        for (; i < data_.size(); i += t.simd_width)
        {
            neon_f32 data_vec  = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 floor_vec = vrndmq_f32(data_vec);
            vst1q_f32(&data_[i], floor_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<_Tp>(std::floor(static_cast<_f32>(data_[i])));
    }

    return t;
}