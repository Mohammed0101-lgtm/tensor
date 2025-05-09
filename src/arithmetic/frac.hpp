#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::frac_() {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    for (auto& elem : data_)
    {
        elem = frac(elem);
    }

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::frac_() const {
    if (!std::is_floating_point_v<value_type>)
    {
        throw type_error("Type must be floating point");
    }

    for (auto& elem : data_)
    {
        elem = frac(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::frac() const {
    self ret = clone();
    ret.frac_();
    return ret;
}