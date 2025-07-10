#pragma once

#include "../tensorbase.hpp"
#include "../types.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::abs() const {
    if (empty())
    {
        return self({0});
    }

    self ret = clone();
    ret.abs_();
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::abs_() {
    if (internal::types::using_neon())
    {
        return internal::neon::abs_(*this);
    }

    if (!std::is_arithmetic_v<value_type>)
    {
        throw error::type_error("abs_() is only available for arithmetic types.");
    }

    if (std::is_unsigned_v<value_type>)
    {
        return *this;
    }

    for (auto& elem : data_)
    {
        elem = std::abs(elem);
    }

    return *this;
}
