#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::exp_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::exp(elem);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::exp_() const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::exp(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::exp() const {
    self ret = clone();
    ret.exp_();
    return ret;
}
