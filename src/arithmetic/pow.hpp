#pragma once

#include "tensorbase.hpp"

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::pow_(const value_type value) {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::pow_(*this, value);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::pow(elem, value);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::pow(const tensor& other) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.pow_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::pow(const value_type value) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.pow_(value);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::pow_(const tensor& other) {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::pow_(*this, other);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    if (!shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;

    for (auto& elem : data_)
    {
        elem = std::pow(elem, other[i++]);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::square_() {
    return pow_(static_cast<value_type>(2.0f));
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::square() const {
    return pow(static_cast<value_type>(2.0f));
}
