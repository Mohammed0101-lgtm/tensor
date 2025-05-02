#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_logical_or_(const value_type value) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot perform logical OR on non-integral values");

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type       i        = 0;

    if constexpr (std::is_signed_v<value_type> || std::is_same_v<value_type, bool>)
    {
        neon_s32 val_vec = vdupq_n_s32(static_cast<_s32>(value));

        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 or_vec   = vorrq_s32(data_vec, val_vec);

            vst1q_s32(&data_[i], or_vec);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        neon_u32 val_vec = vdupq_n_u32(static_cast<_u32>(value));

        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 or_vec   = vorrq_u32(data_vec, val_vec);

            vst1q_u32(&data_[i], or_vec);
        }
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<value_type>(data_[i] || value);

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_logical_xor_(const value_type value) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot get the element wise xor of non-integral value");

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type       i        = 0;

    if constexpr (std::is_signed_v<value_type> || std::is_same_v<value_type, bool>)
    {
        neon_s32 val_vec = vdupq_n_s32(static_cast<_s32>(value));

        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 xor_vec  = veorq_s32(data_vec, val_vec);

            vst1q_s32(&data_[i], xor_vec);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        neon_u32 val_vec = vdupq_n_u32(static_cast<_u32>(value));

        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 xor_vec  = veorq_u32(data_vec, val_vec);

            vst1q_u32(&data_[i], xor_vec);
        }
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<value_type>(data_[i] xor value);

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_logical_and_(const value_type value) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot get the element wise and of non-integral value");

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type       i        = 0;

    if constexpr (std::is_signed_v<value_type>)
    {
        neon_s32 vals = vdupq_n_s32(reinterpret_cast<_s32>(&value));

        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 vec     = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 and_vec = vandq_s32(vec, vals);

            vst1q_s32(&data_[i], and_vec);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        neon_u32 vals = vdupq_n_u32(reinterpret_cast<_u32>(&value));

        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 vec     = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 and_vec = vandq_u32(vec, vals);

            vst1q_u32(&data_[i], and_vec);
        }
    }

    for (; i < data_.size(); ++i)
        data_[i] = static_cast<value_type>(data_[i] && value);

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_logical_or_(const tensor& other) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot get the element wise not of non-integral values");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type       i        = 0;

    if constexpr (std::is_unsigned_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other[i]));
            neon_u32 or_vec    = vornq_u32(data_vec, other_vec);

            vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), or_vec);
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other[i]));
            neon_s32 or_vec    = vornq_s32(data_vec, other_vec);

            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), or_vec);
        }
    }

    for (; i < data_.size(); ++i)
        data_[i] = (data_[i] || other[i]);

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_logical_xor_(const tensor& other) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot get the element wise xor of non-integral value");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type       i        = 0;

    if constexpr (std::is_unsigned_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other[i]));
            neon_u32 xor_vec   = veorq_u32(data_vec, other_vec);

            vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), xor_vec);
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other[i]));
            neon_s32 xor_vec   = veorq_s32(data_vec, other_vec);

            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), xor_vec);
        }
    }

    for (; i < data_.size(); ++i)
        data_[i] = (data_[i] xor other[i]);

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_logical_and_(const tensor& other) {
    if (!std::is_integral_v<value_type>)
        throw type_error("Cannot get the element-wise and of non-integral value");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type       i        = 0;

    if constexpr (std::is_unsigned_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other[i]));
            neon_u32 and_vec   = vandq_u32(data_vec, other_vec);

            vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), and_vec);
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other[i]));
            neon_s32 and_vec   = vandq_s32(data_vec, other_vec);

            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), and_vec);
        }
    }

    for (; i < data_.size(); ++i)
        data_[i] = (data_[i] && other[i]);

    return *this;
}