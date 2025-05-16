#pragma once

#include "tensorbase.hpp"
#include "types.hpp"


template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::dist(const tensor& other) const {
    self ret = clone();
    ret.dist_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::dist(const value_type value) const {
    self ret = clone();
    ret.dist_(value);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::dist_(const tensor& other) {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    if constexpr (!has_minus_operator_v<value_type>)
        throw operator_error("Value type must have a minus operator");

    index_type i = 0;
    for (auto& elem : data_)
        elem = std::abs(elem - other[i++]);

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::dist_(const tensor& other) const {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    if (!equal_shape(shape(), other.shape()))
        throw shape_error("Tensors shapes must be equal");

    if constexpr (!has_minus_operator_v<value_type>)
        throw operator_error("Value type must have a minus operator");

    index_type i = 0;
    for (auto& elem : data_)
        elem = std::abs(data_[i] - other[i++]);

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::dist_(const value_type value) const {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    if constexpr (!has_minus_operator_v<value_type>)
        throw operator_error("Value type must have a minus operator");

    for (auto& elem : data_)
        elem = std::abs(elem - value);

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::dist_(const value_type value) {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    if constexpr (!has_minus_operator_v<value_type>)
        throw operator_error("Value type must have a minus operator");

    for (auto& elem : data_)
        elem = std::abs(elem - value);

    return *this;
}