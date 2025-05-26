#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& internal::neon::bitwise_right_shift_(tensor<_Tp>& t, const int amount) {
    if (!std::is_integral_v<_Tp>)
    {
        throw error::type_error("Type must be integral");
    }

    std::vector<_Tp>&    data_            = t.storage_();
    const _u64           simd_end         = data_.size() - (data_.size() % t.simd_width);
    const neon_type<_Tp> shift_amount_vec = neon_dup<_Tp>(-amount);
    _u64                 i                = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> data_vec    = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> shifted_vec = neon_shl<_Tp>(data_vec, shift_amount_vec);
        neon_store<_Tp>(&data_[i], shifted_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] >>= amount;
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::bitwise_left_shift_(tensor<_Tp>& t, const int amount) {
    return neon_bitwise_right_shift_(t, -amount);
}

template<class _Tp>
tensor<_Tp>& internal::neon::bitwise_or_(tensor<_Tp>& t, const _Tp value) {
    if (!std::is_integral_v<_Tp>)
    {
        throw error::type_error("Cannot perform a bitwise OR on non-integral values");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    neon_type<_Tp>    val_vec  = neon_dup<_Tp>(value);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> data_vec = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> res_vec  = neon_or<_Tp>(data_vec, val_vec);
        neon_store<_Tp>(&data_[i], res_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] |= value;
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::bitwise_xor_(tensor<_Tp>& t, const _Tp value) {
    if (!std::is_integral_v<_Tp>)
    {
        throw error::type_error("Cannot perform a bitwise XOR on non-integral values");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    neon_type<_Tp>    val_vec  = neon_dup<_Tp>(value);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> data_vec = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> res_vec  = neon_xor<_Tp>(data_vec, val_vec);
        neon_store<_Tp>(&data_[i], res_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] ^= value;
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::bitwise_not_(tensor<_Tp>& t) {
    if (!std::is_integral_v<_Tp> && !std::is_same_v<_Tp, bool>)
    {
        throw error::type_error("Cannot perform a bitwise not on non-integral value");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    if constexpr (std::is_signed_v<_Tp>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_s32 result_vec = vmvnq_s32(data_vec);
            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), result_vec);
        }
    }
    else if constexpr (std::is_unsigned_v<_Tp>)
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

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::bitwise_and_(tensor<_Tp>& t, const _Tp value) {
    if (!std::is_integral_v<_Tp>)
    {
        throw error::type_error("Cannot perform a bitwise AND on non-integral values");
    }

    std::vector<_Tp>& data_ = t.storage_();
    const _u64     simd_end = data_.size() - (data_.size() % t.simd_width);
    neon_type<_Tp> val_vec  = neon_dup<_Tp>(value);
    _u64           i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> data_vec = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> res_vec  = neon_and<_Tp>(data_vec, val_vec);
        neon_store<_Tp>(&data_[i], res_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] &= value;
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::bitwise_and_(tensor<_Tp>& t, const tensor<_Tp>& other) {
    if (!std::is_integral_v<_Tp>)
    {
        throw error::type_error("Cannot perform a bitwise AND on non-integral values");
    }

    if (!t.shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> data_vec  = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> other_vec = neon_load<_Tp>(&other[i]);
        neon_type<_Tp> xor_vec   = neon_and<_Tp>(data_vec, other_vec);
        neon_store<_Tp>(&data_[i], xor_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] &= other[i];
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::bitwise_or_(tensor<_Tp>& t, const tensor<_Tp>& other) {
    if (!std::is_integral_v<_Tp>)
    {
        throw error::type_error("Cannot perform a bitwise OR on non-integral values");
    }

    if (!t.shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> data_vec  = neon_load<_Tp>(&data_vec[i]);
        neon_type<_Tp> other_vec = neon_load<_Tp>(&other[i]);
        neon_type<_Tp> xor_vec   = neon_or<_Tp>(data_vec, other_vec);
        neon_store<_Tp>(&data_vec[i], xor_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] |= other[i];
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::bitwise_xor_(tensor<_Tp>& t, const tensor<_Tp>& other) {
    if (!std::is_integral_v<_Tp>)
    {
        throw error::type_error("Cannot perform a bitwise XOR on non-integral values");
    }

    if (!t.shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> data_vec  = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> other_vec = neon_load<_Tp>(&other[i]);
        neon_type<_Tp> xor_vec   = neon_xor<_Tp>(data_vec, other_vec);
        neon_store<_Tp>(&data_[i], xor_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] ^= other[i];
    }

    return t;
}
