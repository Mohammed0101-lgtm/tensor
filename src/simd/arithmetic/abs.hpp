#pragma once

#include "../alias.hpp"
#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_abs_() {
    index_type i = 0;
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    if (std::is_unsigned_v<value_type>)
    {
        return *this;
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    index_type simd_end = data_.size() - (data_.size() % simd_width);

    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec     = neon_load<value_type>(&data_[i]);
        neon_type<value_type> abs_vec = neon_abs<value_type>(vec);
        neon_store<value_type>(&data_[i], abs_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::abs(data_[i]));
    }

    return *this;
}