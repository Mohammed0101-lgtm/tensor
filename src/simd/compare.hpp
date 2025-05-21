#pragma once

#include "tensorbase.hpp"


template<class _Tp>
tensor<bool> internal::neon::equal(tensor<_Tp>& t, const tensor<_Tp>& other) {
    if constexpr (!internal::types::has_equal_operator_v<_Tp>)
        throw error::operator_error("Value type must have equal to operator");

    if (!equal_shape(shape(), other.shape()))
        throw error::shape_error("Tensors shapes must be equal");

    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    std::vector<bool> ret(data_.size());
    _u64              i = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        auto                 data_vec  = neon_load<_Tp>(&data_[i]);
        auto                 other_vec = neon_load<_Tp>(&other.data_[i]);
        auto                 res_vec   = neon_ceq<_Tp>(data_vec, other_vec);
        alignas(16) uint32_t buffer[t.simd_width];
        vst1q_u32(buffer, res_vec);

        for (int j = 0; j < t.simd_width; ++j)
            ret[i + j] = buffer[j] == 0xFFFFFFFF;
    }

    for (; i < data_.size(); ++i)
        ret[i] = (data_[i] == other[i]);

    return tensor<bool>(shape_, ret);
}

template<class _Tp>
tensor<bool> internal::neon::equal(tensor<_Tp>& t, const _Tp value) {
    if constexpr (!internal::types::has_equal_operator_v<_Tp>)
        throw error::operator_error("Value type must have equal to operator");

    std::vector<bool> ret(data_.size());
    const _u64        simd_end = data_.size() - (data_.size() % 4);
    _u64              i        = 0;
    neon_type<_Tp>    val_vec  = neon_dup<_Tp>(value);

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp>   data_vec   = neon_load<_Tp>(&data_[i]);
        neon_u32         cmp_result = neon_ceq<_Tp>(data_vec, val_vec);
        alignas(16) _u32 results[t.simd_width];
        neon_store<_Tp>(results, cmp_result);

        for (int j = 0; j < t.simd_width; ++j)
            ret[i + j] = results[j] != 0;
    }

    for (; i < data_.size(); ++i)
        ret[i] = (data_[i] == value);

    return tensor<bool>(shape_, ret);
}

template<class _Tp>
tensor<bool> internal::neon::less_equal(tensor<_Tp>& t, const tensor<_Tp>& other) {
    if constexpr (!internal::types::has_less_operator_v<_Tp>)
        throw error::operator_error("Value type must have less than operator");

    if (!equal_shape(shape(), other.shape()))
        throw error::shape_error("Tensors shapes must be equal");

    std::vector<_u32> ret(data_.size());
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> vec_a    = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> vec_b    = neon_load<_Tp>(&other[i]);
        neon_type<_Tp> leq_mask = neon_cleq(vec_a, vec_b);
        neon_store<_Tp>(&ret[i], leq_mask);
    }

    // Convert `ret` (integer masks) to boolean
    std::vector<bool> d(data_.size());

    for (std::size_t j = 0; j < i; ++j)
        d[j] = ret[j] != 0;

    for (; i < d.size(); ++i)
        d[i] = (data_[i] <= other[i]);

    return tensor<bool>(shape_, d);
}

template<class _Tp>
tensor<bool> internal::neon::less_equal(tensor<_Tp>& t, const _Tp value) {
    if constexpr (!internal::types::has_less_equal_operator_v<_Tp>)
        throw error::operator_error("Value type must have less than or equal to operator");

    std::vector<_u32> ret(data_.size());
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> vec_a    = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> vec_b    = neon_dup<_Tp>(value);
        neon_type<_Tp> leq_mask = neon_cleq<_Tp>(vec_a, vec_b);
        neon_store<_Tp>(&ret[i], leq_mask);
    }

    for (; i < data_.size(); ++i)
        ret[i] = (data_[i] <= value) ? 1 : 0;

    std::vector<bool> to_bool(ret.size());
    i = 0;

    for (int i = i; i >= 0; i--)
        to_bool[i] = ret[i] == 1 ? true : false;

    return tensor<bool>(to_bool, shape_);
}

template<class _Tp>
inline tensor<bool> internal::neon::less(tensor<_Tp>& t, const _Tp value) {
    if constexpr (!internal::types::has_less_operator_v<_Tp>)
        throw error::operator_error("Value type must have less than operator");
}