#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sin_() {
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
            vals[j] = static_cast<value_type>(std::sin(vals[j]));
        }

        neon_type<value_type> sin_vec = neon_load<value_type>(vals);
        neon_store(&data_[i], sin_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::sin(data_[i]));
    }

    return *this;
}

inline neon_f32 vsinq_f32(neon_f32 x) {
    return {sinf(vgetq_lane_f32(x, 0)), sinf(vgetq_lane_f32(x, 1)), sinf(vgetq_lane_f32(x, 2)),
            sinf(vgetq_lane_f32(x, 3))};
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sinc_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    index_type i = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);

        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 v      = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_f32 pi_v   = vmulq_f32(v, vdupq_n_f32(M_PI));                        // pi * x
            neon_f32 sinc_v = vbslq_f32(vcgeq_f32(vabsq_f32(v), vdupq_n_f32(1e-6f)),  // Check |x| > epsilon
                                        vdivq_f32(vsinq_f32(pi_v), pi_v),  // sinc(x) = sin(pi * x) / (pi * x)
                                        vdupq_n_f32(1.0f));                // sinc(0) = 1

            vst1q_f32(reinterpret_cast<_f32*>(&data_[i]), sinc_v);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = (std::abs(data_[i]) < 1e-6) ? static_cast<value_type>(1.0)
                                               : static_cast<value_type>(std::sin(M_PI * data_[i]) / (M_PI * data_[i]));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sinh_() {
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
            vals[j] = static_cast<value_type>(std::sinh(vals[j]));
        }

        neon_type<value_type> sinh_vec = neon_load<value_type>(vals);
        neon_store(&data_[i], sinh_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::sinh(data_[i]));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_asinh_() {
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
            if (vals[j] < -1 or vals[j] > 1)
            {
                throw std::domain_error("Input value is out of domain dor asinh()");
            }

            vals[j] = static_cast<value_type>(std::asinh(vals[j]));
        }

        neon_type<value_type> asinh_vec = neon_load<value_type>(vals);
        neon_store(&data_[i], asinh_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::asinh(data_[i]));
    }

    return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_asin_() {
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
            if (vals[j] < static_cast<value_type>(-1) or vals[j] > static_cast<value_type>(1))
            {
                throw std::domain_error("Input value is out of domain for asin()");
            }

            vals[j] = static_cast<value_type>(std::asin(vals[j]));
        }

        neon_type<value_type> asin_vec = neon_load<value_type>(vals);
        neon_store(&data_[i], asin_vec);
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::asin(data_[i]));
    }

    return *this;
}