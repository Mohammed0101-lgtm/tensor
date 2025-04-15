#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmax_(const value_type v) {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    index_type i = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        const index_type simd_end   = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
        neon_f32         scalar_val = vdupq_n_f32(v);

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

    index_type i = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);

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

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type       i        = 0;

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 a   = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 b   = vld1q_f32(reinterpret_cast<const _f32*>(&other[i]));
            neon_f32 max = vmaxq_f32(a, b);
            vst1q_f32(reinterpret_cast<_f32*>(&data_[i]), max);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 a   = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 b   = vld1q_s32(reinterpret_cast<const _s32*>(&other[i]));
            neon_s32 max = vmaxq_s32(a, b);
            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), max);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 a   = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 b   = vld1q_u32(reinterpret_cast<const _u32*>(&other[i]));
            neon_u32 max = vmaxq_u32(a, b);
            vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), max);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = std::max(data_[i], other.data_[i]);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_maximum_(const value_type val) {
    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type       i        = 0;

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        neon_f32 val_vec = vdupq_n_f32(val);
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 a   = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 max = vmaxq_f32(a, val_vec);
            vst1q_f32(reinterpret_cast<_f32*>(&data_[i]), max);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        neon_s32 val_vec = vdupq_n_s32(val);
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 a   = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 max = vmaxq_s32(a, val_vec);
            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), max);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        neon_u32 val_vec = vdupq_n_u32(val);
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 a   = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 max = vmaxq_u32(a, val_vec);
            vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), max);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = std::max(data_[i], val);
    }

    return *this;
}