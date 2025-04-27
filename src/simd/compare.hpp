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
            ret[i + j] = results[j] != 0;
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
    const index_type  simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec_a    = neon_load<value_type>(&data_[i]);
        neon_type<value_type> vec_b    = neon_load<value_type>(&other[i]);
        neon_type<value_type> leq_mask = neon_cleq(vec_a, vec_b);
        neon_store<value_type>(&ret[i], leq_mask);
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
    const index_type  simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec_a    = neon_load<value_type>(&data_[i]);
        neon_type<value_type> vec_b    = neon_dup<value_type>(val);
        neon_type<value_type> leq_mask = neon_cleq<value_type>(vec_a, vec_b);
        neon_store<value_type>(&ret[i], leq_mask);
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