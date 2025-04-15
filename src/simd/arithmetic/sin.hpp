#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sin_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    index_type i = 0;

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            _f32     vals[_ARM64_REG_WIDTH];
            vst1q_f32(vals, data_vec);

            vals[0] = static_cast<_f32>(std::sin(vals[0]));
            vals[1] = static_cast<_f32>(std::sin(vals[1]));
            vals[2] = static_cast<_f32>(std::sin(vals[2]));
            vals[3] = static_cast<_f32>(std::sin(vals[3]));

            neon_f32 sin_vec = vld1q_f32(vals);
            vst1q_f32(&data_[i], sin_vec);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            _s32     vals[_ARM64_REG_WIDTH];
            vst1q_s32(vals, data_vec);

            vals[0] = static_cast<_s32>(std::sin(vals[0]));
            vals[1] = static_cast<_s32>(std::sin(vals[1]));
            vals[2] = static_cast<_s32>(std::sin(vals[2]));
            vals[3] = static_cast<_s32>(std::sin(vals[3]));

            neon_s32 sin_vec = vld1q_s32(vals);
            vst1q_s32(&data_[i], sin_vec);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            _u32     vals[_ARM64_REG_WIDTH];
            vst1q_u32(vals, data_vec);

            vals[0] = static_cast<_u32>(std::sin(vals[0]));
            vals[1] = static_cast<_u32>(std::sin(vals[1]));
            vals[2] = static_cast<_u32>(std::sin(vals[2]));
            vals[3] = static_cast<_u32>(std::sin(vals[3]));

            neon_u32 sin_vec = vld1q_u32(vals);
            vst1q_u32(&data_[i], sin_vec);
        }
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

    index_type i = 0;

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            _f32     vals[_ARM64_REG_WIDTH];
            vst1q_f32(vals, data_vec);

            vals[0] = static_cast<_f32>(std::sinh(vals[0]));
            vals[1] = static_cast<_f32>(std::sinh(vals[1]));
            vals[2] = static_cast<_f32>(std::sinh(vals[2]));
            vals[3] = static_cast<_f32>(std::sinh(vals[3]));

            neon_f32 sinh_vec = vld1q_f32(vals);
            vst1q_f32(&data_[i], sinh_vec);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            _s32     vals[_ARM64_REG_WIDTH];
            vst1q_s32(vals, data_vec);

            vals[0] = static_cast<_s32>(std::sinh(vals[0]));
            vals[1] = static_cast<_s32>(std::sinh(vals[1]));
            vals[2] = static_cast<_s32>(std::sinh(vals[2]));
            vals[3] = static_cast<_s32>(std::sinh(vals[3]));

            neon_s32 sinh_vec = vld1q_s32(vals);
            vst1q_s32(&data_[i], sinh_vec);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            _u32     vals[_ARM64_REG_WIDTH];
            vst1q_u32(vals, data_vec);

            vals[0] = static_cast<_u32>(std::sinh(vals[0]));
            vals[1] = static_cast<_u32>(std::sinh(vals[1]));
            vals[2] = static_cast<_u32>(std::sinh(vals[2]));
            vals[3] = static_cast<_u32>(std::sinh(vals[3]));

            neon_u32 sinh_vec = vld1q_u32(vals);
            vst1q_u32(&data_[i], sinh_vec);
        }
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

    index_type i        = 0;
    index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            _f32     vals[_ARM64_REG_WIDTH];
            vst1q_f32(vals, data_vec);

            vals[0] = static_cast<_f32>(std::asinh(vals[0]));
            vals[1] = static_cast<_f32>(std::asinh(vals[1]));
            vals[2] = static_cast<_f32>(std::asinh(vals[2]));
            vals[3] = static_cast<_f32>(std::asinh(vals[3]));

            neon_f32 asinh_vec = vld1q_f32(vals);
            vst1q_f32(reinterpret_cast<_f32*>(&data_[i]), asinh_vec);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            _s32     vals[_ARM64_REG_WIDTH];
            vst1q_s32(vals, data_vec);

            vals[0] = static_cast<_s32>(std::asinh(vals[0]));
            vals[1] = static_cast<_s32>(std::asinh(vals[1]));
            vals[2] = static_cast<_s32>(std::asinh(vals[2]));
            vals[3] = static_cast<_s32>(std::asinh(vals[3]));

            neon_s32 asinh_vec = vld1q_s32(vals);
            vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), asinh_vec);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            _u32     vals[_ARM64_REG_WIDTH];
            vst1q_u32(vals, data_vec);

            vals[0] = static_cast<_u32>(std::asinh(vals[0]));
            vals[1] = static_cast<_u32>(std::asinh(vals[1]));
            vals[2] = static_cast<_u32>(std::asinh(vals[2]));
            vals[3] = static_cast<_u32>(std::asinh(vals[3]));

            neon_u32 asinh_vec = vld1q_u32(vals);
            vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), asinh_vec);
        }
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

    index_type       i        = 0;
    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            _f32     vals[_ARM64_REG_WIDTH];
            vst1q_f32(vals, data_vec);

            vals[0] = static_cast<_f32>(std::asin(vals[0]));
            vals[1] = static_cast<_f32>(std::asin(vals[1]));
            vals[2] = static_cast<_f32>(std::asin(vals[2]));
            vals[3] = static_cast<_f32>(std::asin(vals[3]));

            neon_f32 asin_vec = vld1q_f32(vals);
            vst1q_f32(&data_[i], asin_vec);
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            _s32     vals[_ARM64_REG_WIDTH];
            vst1q_s32(vals, data_vec);

            vals[0] = static_cast<_s32>(std::asin(vals[0]));
            vals[1] = static_cast<_s32>(std::asin(vals[1]));
            vals[2] = static_cast<_s32>(std::asin(vals[2]));
            vals[3] = static_cast<_s32>(std::asin(vals[3]));

            neon_s32 asin_vec = vld1q_s32(vals);
            vst1q_s32(&data_[i], asin_vec);
        }
    }
    else if constexpr (std::is_same_v<value_type, _u32>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            _u32     vals[_ARM64_REG_WIDTH];
            vst1q_u32(vals, data_vec);

            vals[0] = static_cast<_u32>(std::asin(vals[0]));
            vals[1] = static_cast<_u32>(std::asin(vals[1]));
            vals[2] = static_cast<_u32>(std::asin(vals[2]));
            vals[3] = static_cast<_u32>(std::asin(vals[3]));

            neon_u32 asin_vec = vld1q_u32(vals);
            vst1q_u32(&data_[i], asin_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(std::asin(data_[i]));
    }

    return *this;
}