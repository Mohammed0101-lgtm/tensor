#pragma once

#include "tensorbase.hpp"

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::log_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::log_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::log(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::log() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.log_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::log10_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::log10_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::log10(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::log10() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.log10_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::log2_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::log2_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::log2(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::log2() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.log2_();
    return ret;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::log_softmax(const index_type dimension) const {
    self ret = clone();
    ret.log_softmax_(dimension);
    return ret;
}