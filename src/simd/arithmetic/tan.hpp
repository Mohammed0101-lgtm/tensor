#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_tan_() {
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
        neon_type<value_type> data_vec = neon_load<value_type>(&data_[i]);
        value_type            vals[simd_width];
        neon_load<value_type>(vals, data_vec);

        for (int j = 0; j < simd_width; ++j)
        {
            vals[j] = static_cast<value_type>(std::tan(vals[j]));
        }

        neon_type<value_type> tan_vec = neon_load<value_type>(vals);
        neon_store<value_type>(&data_[i], tan_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::tan(data_[i]));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_tanh_() {
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
        neon_type<value_type> data_vec = neon_load<value_type>(&data_[i]);
        value_type            vals[simd_width];
        neon_load<value_type>(vals, data_vec);

        for (int j = 0; j < simd_width; ++j)
        {
            vals[j] = static_cast<value_type>(std::tanh(vals[j]));
        }

        neon_type<value_type> tanh_vec = neon_load<value_type>(vals);
        neon_store(&data_[i], tanh_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::tanh(static_cast<_f32>(data_[i])));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_atan_() {
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
        neon_type<value_type> data_vec = neon_load<value_type>(&data_[i]);
        value_type            vals[simd_width];
        neon_load<value_type>(vals, data_vec);

        for (int j = 0; j < simd_width; ++j)
        {
            if (vals[j] < static_cast<value_type>(-1) || vals[j] > static_cast<value_type>(1))
            {
                throw std::domain_error("Input value is out of domain for atan()");
            }

            vals[j] = static_cast<value_type>(std::atan(vals[j]));
        }

        neon_type<value_type> atan_vec = neon_load<value_type>(vals);
        neon_store(&data_[i], atan_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::atan(data_[i]));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_atanh_() {
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
        neon_type<value_type> data_vec = neon_load<value_type>(&data_[i]);
        value_type            vals[simd_width];
        neon_load<value_type>(vals, data_vec);

        for (int j = 0; j < simd_width; ++j)
        {
            if (vals[j] < static_cast<value_type>(-1) || vals[j] > static_cast<value_type>(1))
            {
                throw std::domain_error("Input value is out of domain for atanh()");
            }

            vals[j] = static_cast<value_type>(std::atanh(vals[j]));
        }

        neon_type<value_type> atanh_vec = neon_load<value_type>(vals);
        neon_store(&data_[i], atanh_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::atan(data_[i]));
    }

    return *this;
}