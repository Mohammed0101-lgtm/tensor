#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_exp_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    index_type simd_end = data_.size() - (data_.size() % simd_width);
    index_type i        = 0;

    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec = neon_load<value_type>(&data_[i]);
        value_type            vals[simd_width];
        neon_store<value_type>(vals, vec);

        for (int j = 0; j < simd_width; ++j)
        {
            vals[j] = static_cast<value_type>(std::exp(vals[0]));
        }

        neon_type<value_type> exp_vec = neon_load<value_type>(vals);
        neon_store<value_type>(&data_[i], exp_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::exp(data_[i]));
    }

    return *this;
}