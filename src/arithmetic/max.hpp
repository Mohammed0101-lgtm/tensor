#pragma once

#include "tensorbase.hpp"

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmax(const tensor& other) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.fmax_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmax(const value_type value) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.fmax_(value);
    return ret;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::fmax_(const value_type value) {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::fmax_(*this, value);
    }

    if (!std::is_floating_point_v<value_type>)
    {
        throw error::type_error("Type must be floating point");
    }

    for (auto& elem : data_)
    {
        elem = std::fmax(elem, value);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const tensor& other) {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::fmax_(*this, other);
    }

    if (!std::is_floating_point_v<value_type>)
    {
        throw error::type_error("Type must be floating point");
    }

    if (!shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;

    for (auto& elem : data_)
    {
        elem = std::fmax(elem, other[i++]);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::maximum(const tensor& other) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.maximum_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::maximum(const_reference value) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.maximum_(value);
    return ret;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::maximum_(const tensor& other) {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::maximum_(*this, other);
    }

    if (!shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;

    for (auto& elem : data_)
    {
        elem = std::max(elem, other[i++]);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::maximum_(const value_type value) {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::maximum_(*this, value);
    }

    for (auto& elem : data_)
    {
        elem = std::max(elem, value);
    }

    return *this;
}
