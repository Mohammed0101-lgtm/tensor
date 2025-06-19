#pragma once

#include "tensorbase.hpp"


template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::frac_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::frac_(*this);
    }

    if (!std::is_floating_point_v<value_type>)
    {
        throw error::type_error("Type must be floating point");
    }

    for (auto& elem : data_)
    {
        elem = frac(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::frac() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.frac_();
    return ret;
}