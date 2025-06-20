#pragma once

#include "tensorbase.hpp"
#include "types.hpp"


template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::dist(const tensor& other) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.dist_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::dist(const value_type value) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.dist_(value);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::dist_(const tensor& other) {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::dist_(*this, other);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    if (!shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    if constexpr (!internal::types::has_minus_operator_v<value_type>)
    {
        throw error::operator_error("Value type must have a minus operator");
    }

    index_type i = 0;

    for (auto& elem : data_)
    {
        elem = std::abs(elem - other[i++]);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::dist_(const value_type value) {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::dist_(*this, value);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    if constexpr (!internal::types::has_minus_operator_v<value_type>)
    {
        throw error::operator_error("Value type must have a minus operator");
    }

    for (auto& elem : data_)
    {
        elem = std::abs(elem - value);
    }

    return *this;
}