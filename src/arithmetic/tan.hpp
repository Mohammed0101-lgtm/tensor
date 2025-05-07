#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::tan_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::tan(elem);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::tan_() const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::tan(elem);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::tanh_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::tanh(elem);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::tanh_() const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::tanh(elem);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::atan_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::atan(elem);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::atan_() const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::atan(elem);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::atanh_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        if (elem < -1 || elem > 1)
        {
            throw std::domain_error("Input data is out of domain for atanh()");
        }

        elem = std::atanh(elem);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::atanh_() const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        if (elem < -1 || elem > 1)
        {
            throw std::domain_error("Input data is out of domain for atanh()");
        }

        elem = std::atanh(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::atanh() const {
    self ret = clone();
    ret.atanh_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::tanh() const {
    self ret = clone();
    ret.tanh_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::tan() const {
    self ret = clone();
    ret.tan_();
    return ret;
}