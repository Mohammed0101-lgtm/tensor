#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmax_(const value_type v) {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    if constexpr (std::is_floating_point_v<value_type>)
    {
        neon_f32 scalar_val = vdupq_n_f32(v);
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 a       = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 max_val = vmaxq_f32(a, scalar_val);

            vst1q_f32(&data_[i], max_val);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = std::fmax(data_[i], v);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmax_(const tensor& other) {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    if constexpr (std::is_floating_point_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 a       = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 b       = vld1q_f32(reinterpret_cast<const _f32*>(&(other[i])));
            neon_f32 max_val = vmaxq_f32(a, b);

            vst1q_f32(&data_[i], max_val);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = std::fmax(data_[i], other[i]);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_maximum_(const tensor& other) {
    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> a   = neon_load<value_type>(&data_[i]);
        neon_type<value_type> b   = neon_load<value_type>(&other[i]);
        neon_type<value_type> max = neon_max<value_type>(a, b);
        neon_store<value_type>(&data_[i], max);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = std::max(data_[i], other.data_[i]);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_maximum_(const value_type value) {
    if constexpr (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Value type must be arithmetic");
    }

    const index_type      simd_end = data_.size() - (data_.size() % simd_width);
    neon_type<value_type> val_vec  = neon_dup<value_type>(value);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> a   = neon_load<value_type>(&data_[i]);
        neon_type<value_type> max = neon_max<value_type>(a, val_vec);
        neon_store<value_type>(&data_[i], max);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = std::max(data_[i], value);
    }

    return *this;
}