#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::abs() const {
    self ret = clone();
    ret.abs_();
    return ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::abs_() {
#if defined(__ARM_NEON)
    return neon_abs_();
#endif
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("abs_() is only available for arithmetic types.");

    if (std::is_unsigned_v<value_type>)
        return *this;

    for (auto& elem : data_)
        elem = std::abs(elem);

    return *this;
}

template<class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::abs_() const {
#if defined(__ARM_NEON)
    return neon_abs_();
#endif
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("abs_() is only available for arithmetic types.");

    if (std::is_unsigned_v<value_type>)
        return *this;

    for (auto& elem : data_)
        elem = std::abs(elem);

    return *this;
}