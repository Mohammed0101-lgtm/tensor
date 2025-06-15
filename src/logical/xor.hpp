#pragma once

#include "tensorbase.hpp"


template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::logical_xor_(const value_type value) {
    if (internal::types::using_neon())
    {
        return internal::neon::logical_xor_(value);
    }

    if constexpr (!std::is_integral_v<value_type>)
    {
        throw error::type_error("Cannot get the element wise xor of non-integral and non-boolean value");
    }

    for (auto& elem : data_)
    {
        elem = (elem xor value);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const tensor& other) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.logical_xor_(other);
    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const value_type value) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.logical_xor(value);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_xor_(const tensor& other) {
    if (internal::types::using_neon())
    {
        return internal::neon::logical_xor_(other);
    }

    if constexpr (!std::is_integral_v<value_type>)
    {
        throw error::type_error("Cannot get the element wise xor of non-integral and non-boolean value");
    }

    if (!shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;

    for (auto& elem : data_)
    {
        elem = (elem xor other[i++]);
    }

    return *this;
}
