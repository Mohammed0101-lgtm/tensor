#pragma once


#include "../alias.hpp"
#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_cos_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");
    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec = neon_load<value_type>(&data_[i]);
        value_type            vals[simd_width];
        neon_store<value_type>(&vals, vec);

        for (std::size_t j = 0; j < simd_width; j++)
        {
            vals[j] = static_cast<value_type>(std::cos(vals[j]));
        }

        neon_type<value_type> cos_vec = neon_load(&vals);
        neon_store<value_type>(&data_[i], cos_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::cos(data_[i]));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_acos_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");
    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec = neon_load<value_type>(&data_[i]);
        value_type            vals[simd_width];
        neon_store<value_type>(&vals, vec);

        for (std::size_t j = 0; j < simd_width; j++)
        {
            vals[j] = static_cast<value_type>(std::acos(vals[j]));
        }

        neon_type<value_type> acos_vec = neon_load(&vals);
        neon_store<value_type>(&data_[i], acos_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::acos(data_[i]));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_cosh_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");
    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;
    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec = neon_load<value_type>(&data_[i]);
        value_type            vals[simd_width];
        neon_store<value_type>(&vals, vec);

        for (std::size_t j = 0; j < simd_width; j++)
        {
            vals[j] = static_cast<value_type>(std::cosh(vals[j]));
        }

        neon_type<value_type> cosh_vec = neon_load(&vals);
        neon_store<value_type>(&data_[i], cosh_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::cosh(data_[i]));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_acosh_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    index_type i = 0;

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    for (; i < simd_end; i += simd_width)
    {
        neon_type<value_type> vec = neon_load<value_type>(&data_[i]);
        value_type            vals[simd_width];
        neon_store<value_type>(&vals, vec);

        for (std::size_t j = 0; j < simd_width; j++)
        {
            vals[j] = static_cast<value_type>(std::acosh(vals[j]));
        }

        neon_type<value_type> acosh_vec = neon_load(&vals);
        neon_store<value_type>(&data_[i], acosh_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::acosh(data_[i]));
    }

    return *this;
}