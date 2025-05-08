#pragma once

#include "tensorbase.hpp"

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmax(const tensor& other) const {
    self ret = clone();
    ret.fmax_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmax(const value_type value) const {
    self ret = clone();
    ret.fmax_(value);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const value_type value) {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    for (auto& elem : data_)
    {
        elem = std::fmax(elem, value);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmax_(const value_type value) const {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    for (auto& elem : data_)
    {
        elem = std::fmax(elem, value);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const tensor& other) {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;
    for (auto& elem : data_)
    {
        elem = std::fmax(elem, other[i++]);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmax_(const tensor& other) const {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
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
    self ret = clone();
    ret.maximum_(other);
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::maximum(const_reference value) const {
    self ret = clone();
    ret.maximum_(value);
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::maximum_(const tensor& other) {
    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    index_type i = 0;
    for (auto& elem : data_)
    {
        elem = std::max(elem, other[i++]);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::maximum_(const tensor& other) const {
    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
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
    for (auto& elem : data_)
    {
        elem = std::max(elem, value);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::maximum_(const value_type value) const {
    for (auto& elem : data_)
    {
        elem = std::max(elem, value);
    }

    return *this;
}
