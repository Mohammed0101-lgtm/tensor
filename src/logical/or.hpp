#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_or_(const value_type value) {
    if constexpr (!std::is_integral_v<value_type>)
    {
        throw error::type_error("Cannot perform logical OR on non-integral and non-boolean values");
    }

    for (auto& elem : data_)
    {
        elem = (elem || value);
    }

    return *this;
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const value_type value) const {
    tensor<bool> ret = clone().bool_();
    ret.logical_or_(value);
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
        throw error::type_error("Cannot get the element wise not of non-integral and non-boolean value");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;
    for (auto& elem : data_)
    {
        elem = (elem || other[i++]);
    }

    return *this;
}
