#pragma once

#include "tensorbase.hpp"


template<class _Tp>
tensor<_s16> internal::neon::int16_(const tensor<_Tp>& t) {
    if (!std::is_convertible_v<_Tp, _s16>)
    {
        throw error::type_error("Type must be convertible to 16 bit signed int");
    }

    if (t.empty())
    {
        return tensor<_s16>(t.shape());
    }

    std::vector<_Tp>& data_ = t.storage_();
    std::vector<_s16> d(data_.size());
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    if constexpr (std::is_floating_point_v<_Tp>)
    {
        for (; i < simd_end; i += t.simd_width)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_s16 int_vec  = vcvtq_s16_f32(data_vec);
            vst1q_s16(reinterpret_cast<_s16*>(&d[i]), int_vec);
        }
    }
    else if constexpr (std::is_unsigned_v<_Tp>)
    {
        for (; i < simd_end; i += t.simd_width)
        {
            neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_s16 int_vec  = vreinterpretq_s16_u32(data_vec);
            vst1q_s16(reinterpret_cast<_s16*>(&d[i]), int_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        d[i] = static_cast<_s16>(data_[i]);
    }

    return tensor<_s16>(t.shape(), d);
}


template<class _Tp>
tensor<_s32> internal::neon::int32_(const tensor<_Tp>& t) {
    if (!std::is_convertible_v<_Tp, _s32>)
    {
        throw error::type_error("Type must be convertible to 32 bit signed int");
    }

    if (t.empty())
    {
        return tensor<_s32>(t.shape());
    }

    std::vector<_Tp>& data_ = t.storage_();
    std::vector<_s32> d(data_.size());
    const _u64        simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    _u64              i        = 0;

    if constexpr (std::is_floating_point_v<_Tp>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_s32 int_vec  = vcvtq_s32_f32(data_vec);
            vst1q_s32(reinterpret_cast<_s32*>(&d[i]), int_vec);
        }
    }
    else if constexpr (std::is_unsigned_v<_Tp>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_s32 int_vec  = vreinterpretq_s32_u32(data_vec);
            vst1q_s32(reinterpret_cast<_s32*>(&d[i]), int_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        d[i] = static_cast<_s32>(data_[i]);
    }

    return tensor<_s32>(t.shape(), d);
}

template<class _Tp>
tensor<_u32> internal::neon::uint32_(const tensor<_Tp>& t) {
    if (!std::is_convertible_v<_Tp, _u32>)
    {
        throw error::type_error("Type must be convertible to unsigned 32 bit int");
    }

    if (t.empty())
    {
        return tensor<_u32>(t.shape());
    }

    std::vector<_Tp>& data_ = t.storage_();
    std::vector<_u32> d(data_.size());
    const _u64        simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    _u64              i        = 0;

    if constexpr (std::is_floating_point_v<_Tp>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_u32 uint_vec = vcvtq_u32_f32(data_vec);
            vst1q_u32(reinterpret_cast<_u32*>(&d[i]), uint_vec);
        }
    }
    else if constexpr (std::is_signed_v<_Tp>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_u32 uint_vec = vreinterpretq_u32_s32(data_vec);
            vst1q_u32(reinterpret_cast<_u32*>(&d[i]), uint_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        d[i] = static_cast<_u32>(data_[i]);
    }

    return tensor<_u32>(t.shape(), d);
}

template<class _Tp>
tensor<_f32> internal::neon::float32_(const tensor<_Tp>& t) {
    if (!std::is_convertible_v<_Tp, _f32>)
    {
        throw error::type_error("Type must be convertible to 32 bit float");
    }

    if (t.empty())
    {
        return tensor<_f32>(t.shape());
    }

    std::vector<_Tp>& data_ = t.storage_();
    std::vector<_f32> d(data_.size());
    _u64              i = 0;

    if constexpr (std::is_same_v<_Tp, _f64>)
    {
        const _u64 simd_end = data_.size() - (data_.size() % (_ARM64_REG_WIDTH / 2));

        for (; i < simd_end; i += (_ARM64_REG_WIDTH / 2))
        {
            neon_f64    data_vec1          = vld1q_f64(reinterpret_cast<const _f64*>(&data_[i]));
            neon_f64    data_vec2          = vld1q_f64(reinterpret_cast<const _f64*>(&data_[i + 2]));
            float32x2_t float_vec1         = vcvt_f32_f64(data_vec1);
            float32x2_t float_vec2         = vcvt_f32_f64(data_vec2);
            neon_f32    float_vec_combined = vcombine_f32(float_vec1, float_vec2);

            vst1q_f32(reinterpret_cast<_f32*>(&d[i]), float_vec_combined);
        }
    }
    else if constexpr (std::is_same_v<_Tp, _s32>)
    {
        const _u64 simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);

        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_f32 float_vec = vcvtq_f32_s32(data_vec);

            vst1q_f32(reinterpret_cast<_f32*>(&d[i]), float_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        d[i] = static_cast<_f32>(data_[i]);
    }

    return tensor<_f32>(t.shape(), d);
}

template<class _Tp>
tensor<_f64> internal::neon::float64_(const tensor<_Tp>& t) {
    if (!std::is_convertible_v<_Tp, _f64>)
    {
        throw error::type_error("Type must be convertible to 64 bit float");
    }

    if (t.empty())
    {
        return tensor<_f64>(t.shape());
    }

    std::vector<_Tp>& data_ = t.storage_();
    std::vector<_f64> d(data_.size());
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    for (; i < simd_end; i += t.simd_width)
    {
        auto data_vec = vld1q_f64(reinterpret_cast<const double*>(&data_[i]));
        vst1q_f64(reinterpret_cast<_f64*>(&d[i]), data_vec);
    }

    for (; i < data_.size(); ++i)
    {
        d[i] = static_cast<_f64>(data_[i]);
    }

    return tensor<_f64>(t.shape(), d);
}

template<class _Tp>
tensor<_u64> internal::neon::uint64_(const tensor<_Tp>& t) {
    if (!std::is_convertible_v<_Tp, _u64>)
    {
        throw error::type_error("Type must be convertible to unsigned 64 bit int");
    }

    if (t.empty())
    {
        return tensor<_u64>(t.shape());
    }

    std::vector<_Tp>& data_ = t.storage_();
    std::vector<_u64> d(data_.size());
    const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
    _u64              i        = 0;

    if constexpr (std::is_unsigned_v<_Tp>)
    {
        for (; i < simd_end; i += t.simd_width)
        {
            neon_u32   data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            uint64x2_t int_vec1 = vmovl_u32(vget_low_u32(data_vec));
            uint64x2_t int_vec2 = vmovl_u32(vget_high_u32(data_vec));
            vst1q_u64(reinterpret_cast<_u64*>(&d[i]), int_vec1);
            vst1q_u64(reinterpret_cast<_u64*>(&d[i + 2]), int_vec2);
        }
    }
    else
    {
        for (; i < simd_end; i += t.simd_width)
        {
            neon_f64 data_vec = vld1q_f64(reinterpret_cast<const _f64*>(&data_[i]));
            neon_f64 uint_vec = vcvtq_u64_f64(data_vec);
            vst1q_u64(reinterpret_cast<_u64*>(&d[i]), uint_vec);
        }
    }

    for (; i < data_.size(); ++i)
    {
        d[i] = static_cast<_u64>(data_[i]);
    }

    return tensor<_u64>(t.shape(), d);
}