#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmod_(const value_type value) {
    if constexpr (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    index_type i = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        const index_type simd_end = data_.size() - (data_.size() - _ARM64_REG_WIDTH);
        neon_f32         b        = vdupq_n_f32(reinterpret_cast<_f32>(value));
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 a         = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 div       = vdivq_f32(a, b);
            neon_f32 floor_div = vrndq_f32(div);
            neon_f32 mult      = vmulq_f32(floor_div, b);
            neon_f32 mod       = vsubq_f32(a, mult);

            vst1q_f32(&data_[i], mod);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::fmod(static_cast<_f32>(data_[i]), static_cast<_f32>(value)));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmod_(const tensor& other) {
    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Cannot divide two tensors of different shapes : fmax");
    }

    index_type i = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 a         = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 b         = vld1q_f32(reinterpret_cast<const _f32*>(&other[i]));
            neon_f32 div       = vdivq_f32(a, b);
            neon_f32 floor_div = vrndq_f32(div);
            neon_f32 mult      = vmulq_f32(floor_div, b);
            neon_f32 mod       = vsubq_f32(a, mult);

            vst1q_f32(&data_[i], mod);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::fmod(static_cast<_f32>(data_[i]), static_cast<_f32>(other[i])));
    }

    return *this;
}