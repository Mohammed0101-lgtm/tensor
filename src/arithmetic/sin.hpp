#pragma once

#include "tensorbase.hpp"

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::sin_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::sin_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::sin(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::sin() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.sin_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::asin() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.asin_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::atan() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.atan_();
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sinc_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::sinc_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    constexpr value_type pi  = static_cast<value_type>(3.14159265358979323846);
    const value_type     eps = std::numeric_limits<value_type>::epsilon();

    for (auto& elem : data_)
    {
        elem = (std::abs(elem) < eps) ? value_type(1.0) : std::sin(pi * elem) / (pi * elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::sinc() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.sinc_();
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sinh_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::sinh_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::sinh(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::sinh() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.sinh_();
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::asinh_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::asinh_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::asinh(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::asinh() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.asinh_();
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::asin_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::asin_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        if (elem < _Tp(-1.0) || elem > _Tp(1.0))
        {
            throw std::domain_error("Input data is out of domain for asin()");
        }

        elem = std::asin(elem);
    }

    return *this;
}
