#pragma once

#include "tensorbase.hpp"

// used as a helper function
inline int64_t __lcm(const int64_t a, const int64_t b) { return (a * b) / std::gcd(a, b); }

template<class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::lcm() const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    index_type ret = static_cast<index_type>(data_[0]);

    for (const auto& elem : data_)
    {
        ret = __lcm(elem, ret);
    }

    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::lcm(const tensor& other) const {
    if (!equal_shape(shape(), other.shape()))
    {
        throw shape_error("Tensors shapes must be equal");
    }

    tensor ret = clone();

    index_type i = 0;
    for (const auto& elem : data_)
    {
        ret[i++] = lcm(elem, other[i]);
    }

    return ret;
}
