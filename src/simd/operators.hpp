#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> internal::neon::operator_plus(const tensor<_Tp>& t, const tensor<_Tp>& other) {
    if constexpr (!internal::types::has_plus_operator_v<_Tp>)
    {
        throw error::operator_error("Value type must have a plus operator");
    }

    if (!t.shape().equal(other.shape()))
    {
        throw error::shape_error("Cannot add two tensors with different shapes");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    std::vector<_Tp>  d(data_.size());
    _u64              i = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> vec1   = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> vec2   = neon_load<_Tp>(&other[i]);
        neon_type<_Tp> result = neon_add<_Tp>(vec1, vec2);
        neon_store<_Tp>(&d[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        d[i] = data_[i] + other[i];
    }

    return tensor<_Tp>(t.shape(), d);
}

template<class _Tp>
tensor<_Tp> internal::neon::operator_plus(const tensor<_Tp>& t, const _Tp value) {
    if constexpr (!internal::types::has_plus_operator_v<_Tp>)
    {
        throw error::operator_error("Value type must have a plus operator");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    std::vector<_Tp>  d(data_.size());
    neon_type<_Tp>    val_vec = neon_dup<_Tp>(value);
    _u64              i       = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> vec1   = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> result = neon_add<_Tp>(vec1, val_vec);
        neon_store<_Tp>(&d[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        d[i] = data_[i] + value;
    }

    return tensor<_Tp>(t.shape_, d);
}

template<class _Tp>
tensor<_Tp>& internal::neon::operator_plus_eq(tensor<_Tp>& t, const _Tp& value) {
    if constexpr (!internal::types::has_equal_operator_v<_Tp>)
    {
        throw error::operator_error("Value type must have a plus operator");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    neon_type<_Tp>    val_vec  = neon_dup<_Tp>(value);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> vec1   = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> result = neon_add<_Tp>(vec1, val_vec);
        neon_store<_Tp>(&data_[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = data_[i] + value;
    }

    return t;
}

template<class _Tp>
tensor<_Tp> internal::neon::operator_minus(const tensor<_Tp>& t, const tensor<_Tp>& other) {
    if constexpr (!internal::types::has_minus_operator_v<_Tp>)
    {
        throw error::operator_error("Value type must have a minus operator");
    }

    if (!t.shape().equal(other.shape()))
    {
        throw error::shape_error("Cannot add two tensors with different shapes");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    std::vector<_Tp>  d(data_.size());
    _u64              i = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> vec1   = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> vec2   = neon_load<_Tp>(&other[i]);
        neon_type<_Tp> result = neon_sub<_Tp>(vec1, vec2);
        neon_store<_Tp>(&d[i], result);
    }

    for (; i < data_[i]; ++i)
    {
        d[i] = data_[i] - other[i];
    }

    return tensor<_Tp>(t.shape(), d);
}

template<class _Tp>
tensor<_Tp> internal::neon::operator_minus(const tensor<_Tp>& t, const _Tp value) {
    if constexpr (!internal::types::has_minus_operator_v<_Tp>)
    {
        throw error::operator_error("Value type must have a minus operator");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    std::vector<_Tp>  d(data_.size());
    neon_type<_Tp>    val_vec = neon_dup<_Tp>(value);
    _u64              i       = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> vec1   = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> result = neon_sub<_Tp>(vec1, val_vec);
        neon_store<_Tp>(&d[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        d[i] = data_[i] - value;
    }

    return tensor<_Tp>(t.shape(), d);
}

template<class _Tp>
tensor<_Tp>& internal::neon::operator_minus_eq(tensor<_Tp>& t, const tensor<_Tp>& other) {
    if constexpr (!internal::types::has_minus_operator_v<_Tp>)
    {
        throw error::operator_error("Value type must have a minus operator");
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
        neon_type<_Tp> vec1   = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> vec2   = neon_load<_Tp>(&other[i]);
        neon_type<_Tp> result = neon_sub<_Tp>(vec1, vec2);
        neon_store<_Tp>(&data_[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] -= other[i];
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::operator_times_eq(tensor<_Tp>& t, const tensor<_Tp>& other) {
    if constexpr (!internal::types::has_minus_operator_v<_Tp>)
    {
        throw error::operator_error("Value type must have a minus operator");
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
        neon_type<_Tp> vec1   = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> vec2   = neon_load<_Tp>(&other[i]);
        neon_type<_Tp> result = neon_mul<_Tp>(vec1, vec2);
        neon_store<_Tp>(&data_[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] *= other[i];
    }

    return t;
}

template<class _Tp>
tensor<_Tp>& internal::neon::operator_minus_eq(tensor<_Tp>& t, const _Tp& value) {
    if constexpr (!internal::types::has_minus_operator_v<_Tp>)
    {
        throw error::operator_error("Value type must have a minus operator");
    }

    std::vector<_Tp>& data_    = t.storage_();
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    neon_type<_Tp>    val_vec  = neon_dup<_Tp>(value);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        neon_type<_Tp> vec1   = neon_load<_Tp>(&data_[i]);
        neon_type<_Tp> result = neon_mul<_Tp>(vec1, val_vec);
        neon_store<_Tp>(&data_[i], result);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] -= value;
    }

    return t;
}
