#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_right_shift_(const int amount) {
    if (!std::is_integral_v<value_type>)
    {
        throw type_error("Type must be integral");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");
    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    int                         right_shift_amount = -amount;
    const neon_type<value_type> shift_amount_vec   = neon_dup<value_type>(&(right_shift_amount));

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> data_vec    = neon_load<value_type>(&data_[i]);
        neon_type<value_type> shifted_vec = neon_shl<value_type>(data_vec, shift_amount_vec);
        neon_store<value_type>(&data_[i], shifted_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] >>= amount;
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_left_shift_(const int amount) {
    return neon_bitwise_right_shift_(-amount);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_or_(const value_type val) {
    if (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot perform a bitwise OR on non-integral values");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");
    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    neon_type<value_type> val_vec = neon_dup<value_type>(&val);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> data_vec = neon_load<value_type>(&data_[i]);
        neon_type<value_type> res_vec  = neon_or<value_type>(data_vec, val_vec);
        neon_store<value_type>(&data_[i], result_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] |= val;
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_xor_(const value_type val) {
    if (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot perform a bitwise XOR on non-integral values");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");
    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    neon_type<value_type> val_vec = neon_dup<value_type>(&val);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> data_vec = neon_load<value_type>(&data_[i]);
        neon_type<value_type> res_vec  = neon_xor<value_type>(data_vec, val_vec);
        neon_store<value_type>(&data_[i], result_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] ^= val;
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_not_() {
    if (!std::is_integral_v<value_type> and !std::is_same_v<value_type, bool>)
    {
        throw type_error("Cannot perform a bitwise not on non-integral value");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");
    const index_type simd_end = data_.size() - (data_.size() % simd_width);


    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type       i        = 0;

    if constexpr (std::is_signed_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 result_vec = vmvnq_s32(data_vec);

            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), result_vec);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 result_vec = vmvnq_u32(data_vec);

            vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), result_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = ~data_[i];
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_and_(const value_type val) {
    if (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot perform a bitwise AND on non-integral values");
    }

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type       i        = 0;

    if constexpr (std::is_signed_v<value_type>)
    {
        neon_s32 val_vec = vdupq_n_s32(reinterpret_cast<_s32>(&val));
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 result_vec = vandq_s32(data_vec, val_vec);

            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), result_vec);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        neon_u32 val_vec = vdupq_n_u32(reinterpret_cast<_u32>(&val));
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 result_vec = vandq_u32(data_vec, val_vec);

            vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), result_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] &= val;
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_and_(const tensor& other) {
    if (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot perform a bitwise AND on non-integral values");
    }

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    const std::size_t simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type        i        = 0;

    if constexpr (std::is_unsigned_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_u32 other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other[i]));
            neon_u32 xor_vec   = vandq_u32(data_vec, other_vec);

            vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), xor_vec);
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other[i]));
            neon_s32 xor_vec   = vandq_s32(data_vec, other_vec);

            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), xor_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] &= other[i];
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_or_(const tensor& other) {
    if (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot perform a bitwise OR on non-integral values");
    }

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
            neon_u32 xor_vec   = vornq_u32(data_vec, other_vec);

            vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), xor_vec);
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other[i]));
            neon_s32 xor_vec   = vornq_s32(data_vec, other_vec);

            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), xor_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] |= other[i];
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_xor_(const tensor& other) {
    if (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot perform a bitwise XOR on non-integral values");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

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
    {
        data_[i] ^= other[i];
    }

    return *this;
}
