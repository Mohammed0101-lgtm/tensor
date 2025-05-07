#pragma once

#include "tensorbase.hpp"

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::remainder(const value_type value) const {
    self ret = clone();
    ret.remainder_(value);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::remainder(const tensor& other) const {
    self ret = clone();
    ret.remainder_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::remainder_(const value_type value) {
    if constexpr (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithemtic");
    }

    if (!value)
    {
        throw std::invalid_argument("Remainder by zero is undefined");
    }

    for (auto& elem : data_)
    {
        elem %= value;
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::remainder_(const value_type value) const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arihtmetic");
    }

    if (!value)
    {
        throw std::invalid_argument("Remainder by zero is undefined");
    }

    for (auto& elem : data_)
    {
        elem %= value;
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::remainder_(const tensor& other) {
    if constexpr (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    assert(other.count_nonzero() == other.size(0) && "Remainder by zero is undefined");

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;
    for (auto& elem : data_)
    {
        elem %= other[i++];
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::remainder_(const tensor& other) const {
    if constexpr (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    assert(other.count_nonzero() == other.size(0) && "Remainder by zero is undefined");

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;
    for (auto& elem : data_)
    {
        elem %= other[i++];
    }

    return *this;
}
