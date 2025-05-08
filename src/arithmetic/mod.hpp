#pragma once

#include "tensorbase.hpp"

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmod(const tensor& other) const {
    self ret = clone();
    ret.fmod_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmod(const value_type value) const {
    self ret = clone();
    ret.fmod_(value);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmod_(const value_type value) {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Cannot perform fmod on non-floating point values");
    }

    if (!value)
    {
        throw std::logic_error("Cannot perform fmod with zero");
    }

    for (auto& elem : data_)
    {
        elem = std::fmod(elem, value);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmod_(const value_type value) const {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Cannot perform fmod on non-floating point values");
    }

    if (!value)
    {
        throw std::logic_error("Cannot perform fmod with zero");
    }

    for (auto& elem : data_)
    {
        elem = std::fmod(elem, value);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmod_(const tensor& other) {
    if (!equal_shape(shape(), other.shape()) || data_.size() != other.size(0))
    {
        throw shape_error("Cannot divide two tensors of different shapes : fmod");
    }

    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Cannot perform fmod on non-floating point values");
    }

    if (other.count_nonzero(0) != other.size(0))
    {
        throw std::logic_error("Cannot divide by zero : undefined operation");
    }

    index_type i = 0;
    for (auto& elem : data_)
    {
        elem = std::fmod(elem, other[i++]);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmod_(const tensor& other) const {
    if (equal_shape(shape(), other.shape()) || data_.size() != other.size(0))
    {
        throw shape_error("Cannot divide two tensors of different shapes : fmod");
    }

    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Cannot perform fmod on non-floating point values");
    }

    if (other.count_nonzero(0) != other.size(0))
    {
        throw std::logic_error("Cannot divide by zero : undefined operation");
    }

    index_type i = 0;
    for (auto& elem : data_)
    {
        elem = std::fmod(elem, other[i++]);
    }

    return *this;
}
