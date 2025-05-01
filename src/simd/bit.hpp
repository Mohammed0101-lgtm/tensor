#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_right_shift_(const int amount) {
    if (!std::is_integral_v<value_type>)
    {
        throw type_error("Type must be integral");
    }

    const index_type            simd_end         = data_.size() - (data_.size() % simd_width);
    const neon_type<value_type> shift_amount_vec = neon_dup<value_type>(-amount);

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
tensor<_Tp>& tensor<_Tp>::neon_bitwise_or_(const value_type value) {
    if (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot perform a bitwise OR on non-integral values");
    }

    const index_type      simd_end = data_.size() - (data_.size() % simd_width);
    neon_type<value_type> val_vec  = neon_dup<value_type>(value);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> data_vec = neon_load<value_type>(&data_[i]);
        neon_type<value_type> res_vec  = neon_or<value_type>(data_vec, val_vec);
        neon_store<value_type>(&data_[i], res_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] |= value;
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_xor_(const value_type value) {
    if (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot perform a bitwise XOR on non-integral values");
    }

    const index_type      simd_end = data_.size() - (data_.size() % simd_width);
    neon_type<value_type> val_vec  = neon_dup<value_type>(value);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> data_vec = neon_load<value_type>(&data_[i]);
        neon_type<value_type> res_vec  = neon_xor<value_type>(data_vec, val_vec);
        neon_store<value_type>(&data_[i], res_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] ^= value;
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_not_() {
    if (!std::is_integral_v<value_type> && !std::is_same_v<value_type, bool>)
    {
        throw type_error("Cannot perform a bitwise not on non-integral value");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;

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
tensor<_Tp>& tensor<_Tp>::neon_bitwise_and_(const value_type value) {
    if (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot perform a bitwise AND on non-integral values");
    }

    const index_type      simd_end = data_.size() - (data_.size() % simd_width);
    neon_type<value_type> val_vec  = neon_dup<value_type>(value);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> data_vec = neon_load<value_type>(&data_[i]);
        neon_type<value_type> res_vec  = neon_and<value_type>(data_vec, val_vec);
        neon_store<value_type>(&data_[i], res_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] &= value;
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
    {
        throw shape_error("Tensors shapes must be equal");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> data_vec  = neon_load<value_type>(&data_[i]);
        neon_type<value_type> other_vec = neon_load<value_type>(&other[i]);
        neon_type<value_type> xor_vec   = neon_and<value_type>(data_vec, other_vec);
        neon_store<value_type>(&data_[i], xor_vec);
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
    {
        throw shape_error("Tensors shapes must be equal");
    }

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> data_vec  = neon_load<value_type>(&data_vec[i]);
        neon_type<value_type> other_vec = neon_load<value_type>(&other[i]);
        neon_type<value_type> xor_vec   = neon_or<value_type>(data_vec, other_vec);
        neon_store<value_type>(&data_vec[i], xor_vec);
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

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> data_vec  = neon_load<value_type>(&data_[i]);
        neon_type<value_type> other_vec = neon_load<value_type>(&other[i]);
        neon_type<value_type> xor_vec   = neon_xor<value_type>(data_vec, other_vec);
        neon_store<value_type>(&data_[i], xor_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] ^= other[i];
    }

    return *this;
}
