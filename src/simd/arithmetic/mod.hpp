#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& internal::neon::fmod_(tensor<_Tp>& t, const _Tp value) {
    if constexpr (!std::is_floating_point_v<_Tp>)
        throw error::type_error("Type must be floating point");

    std::vector<_Tp>& data_ = t.storage_();
    _u64              i     = 0;

    if constexpr (std::is_floating_point_v<_Tp>)
    {
        const _u64 simd_end = data_.size() - (data_.size() - t.simd_width);
        neon_f32   b        = vdupq_n_f32(reinterpret_cast<_f32>(value));

        for (; i < simd_end; i += t.simd_width)
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
        data_[i] = static_cast<_Tp>(std::fmod(static_cast<_f32>(data_[i]), static_cast<_f32>(value)));

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::fmod_(tensor<_Tp>& t, const tensor<_Tp>& other) {
    if (!t.shape().equal(other.shape()))
        throw error::shape_error("Cannot divide two tensors of different shapes : fmax");

    std::vector<_Tp>& data_ = t.storage_();
    _u64              i     = 0;

    if constexpr (std::is_floating_point_v<_Tp>)
    {
        const _u64 simd_end = data_.size() - (data_.size() % t.simd_width);

        for (; i < simd_end; i += t.simd_width)
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
        data_[i] = static_cast<_Tp>(std::fmod(static_cast<_f32>(data_[i]), static_cast<_f32>(other[i])));

    return t;
}