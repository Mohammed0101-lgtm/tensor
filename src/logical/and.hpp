#pragma once

#include "tensorbase.hpp"


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_and_(const value_type value) {
    if (internal::types::using_neon())
    {
        return internal::neon::logical_and_(value);
    }

    if (!std::is_integral_v<value_type>)
    {
        throw error::type_error("Cannot get the element wise and of non-integral and non-boolean value");
    }

    for (auto& elem : data_)
    {
        elem = (elem && value);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const tensor& other) const {
    self ret = clone();
    ret.logical_and_(other);
    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const value_type value) const {
    self ret = clone();
    ret.logical_and_(value);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_and_(const tensor& other) {
    if (internal::types::using_neon())
    {
        return internal::neon::logical_and_(other);
    }

    if (!std::is_integral_v<value_type>)
    {
        throw error::type_error("Cannot get the element-wise and of non-integral and non-boolean value");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;

    for (auto& elem : data_)

    {
        elem = (elem && other[i++]);
    }

    return *this;
}