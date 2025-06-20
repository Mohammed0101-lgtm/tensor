#pragma once

#include "tensorbase.hpp"

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::cos_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::cos_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::cos(elem);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::acos_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::acos_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        if (elem > _Tp(1.0) || elem < _Tp(-1.0))
        {
            throw std::domain_error("Input data is out of domain for acos()");
        }

        elem = std::acos(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::acos() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.acos_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::cos() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.cos_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::cosh() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.cosh_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::cosh_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::cosh_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::cosh(elem);
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::acosh_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::acosh_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        if (elem < 1.0)
        {
            throw std::domain_error("Input data is out of domain of acosh()");
        }

        elem = std::acosh(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::acosh() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.acosh_();
    return ret;
}