#pragma once

#include "tensorbase.hpp"

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::tan_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::tan_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::tan(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::tanh_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::tanh_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::tanh(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::atan_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::atan_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::atan(elem);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::atanh_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::atanh_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
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
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.atanh_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::tanh() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.tanh_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::tan() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.tan_();
    return ret;
}