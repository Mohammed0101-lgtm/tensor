#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::cos_() {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    for (auto& elem : data_)
        elem = std::cos(elem);

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::cos_() const {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    for (auto& elem : data_)
        elem = std::cos(elem);

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::acos_() {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    for (auto& elem : data_)
    {
        if (elem > 1.0 || elem < -1.0)
            throw std::domain_error("Input data is out of domain for acos()");

        elem = std::acos(elem);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::acos_() const {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    for (auto& elem : data_)
    {
        if (elem > 1.0 || elem < -1.0)
            throw std::domain_error("Input data is out of domain for acos()");

        elem = std::acos(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::acos() const {
    self ret = clone();
    ret.acos_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::cos() const {
    self ret = clone();
    ret.cos_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::cosh() const {
    self ret = clone();
    ret.cosh_();
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::cosh_() {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    for (auto& elem : data_)
        elem = std::cosh(elem);

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::cosh_() const {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    for (auto& elem : data_)
        elem = std::cosh(elem);

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::acosh_() {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    for (auto& elem : data_)
    {
        if (elem < 1.0)
            throw std::domain_error("Input data is out of domain of acosh()");

        elem = std::acosh(elem);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::acosh_() const {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    for (auto& elem : data_)
    {
        if (elem < 1.0)
            throw std::domain_error("Input data is out of domain of acosh()");

        elem = std::acosh(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::acosh() const {
    self ret = clone();
    ret.acosh_();
    return ret;
}