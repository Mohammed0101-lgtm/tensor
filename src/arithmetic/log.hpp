#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::log_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::log(elem);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log_() const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::log(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::log() const {
    self ret = clone();
    ret.log_();
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::log10_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::log10(elem);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log10_() const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::log10(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::log10() const {
    self ret = clone();
    ret.log10_();
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::log2_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::log2(elem);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log2_() const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::log2(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::log2() const {
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