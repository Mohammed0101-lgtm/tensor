#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<bool> tensor<_Tp>::neon_equal(const tensor& other) const {
    if constexpr (!has_equal_operator_v<value_type>)
    {
        throw operator_error("Value type must have equal to operator");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    const index_type  simd_end = data_.size() - (data_.size() % simd_width);
    std::vector<bool> ret(data_.size());

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        auto                 data_vec  = neon_load<value_type>(&data_[i]);
        auto                 other_vec = neon_load<value_type>(&other.data_[i]);
        auto                 res_vec   = neon_ceq<value_type>(data_vec, other_vec);
        alignas(16) uint32_t buffer[simd_width];
        vst1q_u32(buffer, res_vec);
        for (int j = 0; j < simd_width; ++j)
        {
            ret[i + j] = buffer[j] == 0xFFFFFFFF;
        }
    }

    for (; i < data_.size(); ++i)
    {
        ret[i] = (data_[i] == other[i]);
    }

    return tensor<bool>(shape_, ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::neon_equal(const value_type val) const {
    if constexpr (!has_equal_operator_v<value_type>)
    {
        throw operator_error("Value type must have equal to operator");
    }

    std::vector<bool> ret(data_.size());
    const index_type  simd_end = data_.size() - (data_.size() % 4);

    index_type            i       = 0;
    neon_type<value_type> val_vec = neon_dup<value_type>(val);
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> data_vec   = neon_load<value_type>(&data_[i]);
        neon_u32              cmp_result = neon_ceq<value_type>(data_vec, val_vec);
        alignas(16) _u32      results[simd_width];
        neon_store<value_type>(results, cmp_result);
        for (int j = 0; j < simd_width; ++j)
        {
            ret[i + j] = result[j] != 0;
        }
    }

    for (; i < data_.size(); ++i)
    {
        ret[i] = (data_[i] == val);
    }

    return tensor<bool>(shape_, ret);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::neon_less_equal(const tensor& other) const {
    if constexpr (!has_less_operator_v<value_type>)
    {
        throw operator_error("Value type must have less than operator");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    std::vector<_u32> ret(data_.size());
    std::size_t       vs = data_.size() / _ARM64_REG_WIDTH * _ARM64_REG_WIDTH;
    index_type        i  = 0;

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        for (; i < vs; i += _ARM64_REG_WIDTH)
        {
            neon_f32 va       = vld1q_f32(data_.data() + i);
            neon_f32 vb       = vld1q_f32(other.data_.data() + i);
            neon_u32 leq_mask = vcleq_f32(va, vb);
            vst1q_u32(&ret[i], leq_mask);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        for (; i < vs; i += _ARM64_REG_WIDTH)
        {
            neon_s32 va       = vld1q_s32(data_.data() + i);
            neon_s32 vb       = vld1q_s32(other.data_.data() + i);
            neon_u32 leq_mask = vcleq_s32(va, vb);
            vst1q_u32(&ret[i], leq_mask);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        for (; i < vs; i += _ARM64_REG_WIDTH)
        {
            neon_u32 va       = vld1q_u32(data_.data() + i);
            neon_u32 vb       = vld1q_u32(other.data_.data() + i);
            neon_u32 leq_mask = vcleq_u32(va, vb);
            vst1q_u32(&ret[i], leq_mask);
        }
    }

    // Convert `ret` (integer masks) to boolean
    std::vector<bool> d(data_.size());

    for (std::size_t j = 0; j < i; ++j)
    {
        d[j] = ret[j] != 0;
    }

    for (; i < d.size(); ++i)
    {
        d[i] = (data_[i] <= other[i]);
    }

    return tensor<bool>(shape_, d);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::neon_less_equal(const value_type val) const {
    if constexpr (!has_less_equal_operator_v<value_type>)
    {
        throw operator_error("Value type must have less than or equal to operator");
    }

    std::vector<_u32> ret(data_.size());
    index_type        i = 0;

    std::size_t vector_size = data_.size() / _ARM64_REG_WIDTH * _ARM64_REG_WIDTH;

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        for (; i < vector_size; i += _ARM64_REG_WIDTH)
        {
            neon_f32 vec_a    = vld1q_f32(data_.data() + i);
            neon_f32 vec_b    = vdupq_n_f32(val);
            neon_u32 leq_mask = vcleq_f32(vec_a, vec_b);

            vst1q_u32(&ret[i], leq_mask);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        for (; i < vector_size; i += _ARM64_REG_WIDTH)
        {
            neon_s32 vec_a    = vld1q_s32(data_.data() + i);
            neon_s32 vec_b    = vdupq_n_s32(val);
            neon_u32 leq_mask = vcleq_s32(vec_a, vec_b);

            vst1q_u32(&ret[i], leq_mask);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        for (; i < vector_size; i += _ARM64_REG_WIDTH)
        {
            neon_u32 vec_a    = vld1q_u32(data_.data() + i);
            neon_u32 vec_b    = vdupq_n_u32(val);
            neon_u32 leq_mask = vcleq_u32(vec_a, vec_b);

            vst1q_u32(&ret[i], leq_mask);
        }
    }

    for (; i < data_.size(); ++i)
    {
        ret[i] = (data_[i] <= val) ? 1 : 0;
    }

    std::vector<bool> to_bool(ret.size());
    i = 0;

    for (int i = i; i >= 0; i--)
    {
        to_bool[i] = ret[i] == 1 ? true : false;
    }

    return tensor<bool>(to_bool, shape_);
}

template<class _Tp>
inline tensor<bool> tensor<_Tp>::neon_less(const value_type val) const {
    if constexpr (!has_less_operator_v<value_type>)
    {
        throw operator_error("Value type must have less than operator");
    }
}