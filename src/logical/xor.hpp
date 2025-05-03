#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_xor_(const value_type value) {
    if constexpr (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot get the element wise xor of non-integral and non-boolean value");
    }

    for (auto& elem : data_)
    {
        elem = (elem xor value);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_xor_(const value_type value) const {
    if constexpr (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot get the element wise xor of non-integral and non-boolean value");
    }

    for (auto& elem : data_)
    {
        elem = (elem xor value);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const tensor& other) const {
    self ret = clone();
    ret.logical_xor_(other);
    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const value_type value) const {
    self ret = clone();
    ret.logical_xor(value);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_xor_(const tensor& other) {
    if constexpr (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot get the element wise xor of non-integral and non-boolean value");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;
    for (auto& elem : data_)
    {
        elem = (elem xor other[i++]);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_xor_(const tensor& other) const {
    if constexpr (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot get the element wise xor of non-integral and non-boolean value");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;
    for (auto& elem : data_)
    {
        elem = (elem xor other[i++]);
    }

    return *this;
}
