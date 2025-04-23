#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_or_(const value_type val) {
    if constexpr (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot perform logical OR on non-integral and non-boolean values");
    }

    for (auto& elem : data_)
    {
        elem = (elem or val);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_or_(const value_type val) const {
    if constexpr (!std::is_integral_v<value_type> and !std::is_same_v<value_type, bool>)
    {
        throw type_error("Cannot perform logical OR on non-integral and non-boolean values");
    }

    for (auto& elem : data_)
    {
        elem = (elem or val);
    }

    return *this;
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const value_type val) const {
    tensor<bool> ret = clone().bool_();
    ret.logical_or_(val);
    return ret;
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const tensor& other) const {
    tensor<bool> ret = clone().bool_();
    ret.logical_or_(other);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_or_(const tensor& other) {
    if constexpr (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot get the element wise not of non-integral and non-boolean value");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;
    for (auto& elem : data_)
    {
        elem = (elem or other[i++]);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_or_(const tensor& other) const {
    if constexpr (!std::is_integral_v<value_type>)
    {
        throw type_error("Cannot get the element wise not of non-integral and non-boolean value");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;
    for (auto& elem : data_)
    {
        elem = (elem or other[i++]);
    }

    return *this;
}