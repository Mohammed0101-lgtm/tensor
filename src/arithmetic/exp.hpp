#pragma once

#include "tensorbase.hpp"


template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::exp_() {
    if (empty())
    {
        return *this;
    }

    if (internal::types::using_neon())
    {
        return internal::neon::exp_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("Type must be arithmetic");
    }

    for (auto& elem : data_)
    {
        elem = std::exp(elem);
    }

    return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::exp() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.exp_();
    return ret;
}
