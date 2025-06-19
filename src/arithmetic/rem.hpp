#pragma once

#include "tensorbase.hpp"


template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::remainder(const value_type value) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.remainder_(value);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::remainder(const tensor& other) const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.remainder_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::remainder_(const value_type value) {
    if (empty())
    {
        return *this;
    }
    /*
    if (internal::types::using_neon())
    {
        return internal::neon::remainder_(*this, value);
    }
    */
    if constexpr (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithemtic");
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
    if (empty())
    {
        return *this;
    }
    /*
    if (internal::types::using_neon())
    {
        return internal::neon::remainder_(*this, other);
    }
*/
    if constexpr (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    assert(other.count_nonzero() == other.size(0) && "Remainder by zero is undefined");

    if (!shape().equal(other.shape()))
    {
        throw error::shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;

    for (auto& elem : data_)
    {
        elem %= other[i++];
    }

    return *this;
}