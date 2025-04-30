#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_s32> tensor<_Tp>::neon_int32_() const {
    if (!std::is_convertible_v<value_type, _s32>)
        throw type_error("Type must be convertible to 32 bit signed int");

    if (empty())
        return tensor<_s32>(shape_);

    std::vector<_s32> d(data_.size());
    const index_type  simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type        i        = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_s32 int_vec  = vcvtq_s32_f32(data_vec);

            vst1q_s32(reinterpret_cast<_s32*>(&d[i]), int_vec);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            neon_s32 int_vec  = vreinterpretq_s32_u32(data_vec);

            vst1q_s32(reinterpret_cast<_s32*>(&d[i]), int_vec);
        }
    }

    for (; i < data_.size(); ++i)
        d[i] = static_cast<_s32>(data_[i]);

    return tensor<_s32>(shape_, d);
}

template<class _Tp>
tensor<_u32> tensor<_Tp>::neon_uint32_() const {
    if (!std::is_convertible_v<value_type, _u32>)
        throw type_error("Type must be convertible to unsigned 32 bit int");

    if (empty())
        return tensor<_u32>(shape_);

    std::vector<_u32> d(data_.size());
    index_type        i = 0;

    const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);

    if constexpr (std::is_floating_point_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_u32 uint_vec = vcvtq_u32_f32(data_vec);

            vst1q_u32(reinterpret_cast<_u32*>(&d[i]), uint_vec);
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_u32 uint_vec = vreinterpretq_u32_s32(data_vec);

            vst1q_u32(reinterpret_cast<_u32*>(&d[i]), uint_vec);
        }
    }

    for (; i < data_.size(); ++i)
        d[i] = static_cast<_u32>(data_[i]);

    return tensor<_u32>(shape_, d);
}

template<class _Tp>
tensor<_f32> tensor<_Tp>::neon_float32_() const {
    if (!std::is_convertible_v<value_type, _f32>)
        throw type_error("Type must be convertible to 32 bit float");

    if (empty())
        return tensor<_f32>(shape_);

    std::vector<_f32> d(data_.size());
    index_type        i = 0;

    if constexpr (std::is_same_v<value_type, _f64>)
    {
        const index_type simd_end = data_.size() - (data_.size() % (_ARM64_REG_WIDTH / 2));

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
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        const index_type simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);

        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            neon_f32 float_vec = vcvtq_f32_s32(data_vec);

            vst1q_f32(reinterpret_cast<_f32*>(&d[i]), float_vec);
        }
    }

    for (; i < data_.size(); ++i)
        d[i] = static_cast<_f32>(data_[i]);

    return tensor<_f32>(shape_, d);
}

template<class _Tp>
tensor<_f64> tensor<_Tp>::neon_double_() const {
    if (!std::is_convertible_v<value_type, _f64>)
        throw type_error("Type must be convertible to 64 bit float");

    if (empty())
        return tensor<_f64>(shape_);

    std::vector<_f64> d(data_.size());
    const index_type  simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type        i        = 0;

    for (; i < simd_end; i += _ARM64_REG_WIDTH)
    {
        auto data_vec = vld1q_f64(reinterpret_cast<const double*>(&data_[i]));
        vst1q_f64(reinterpret_cast<_f64*>(&d[i]), data_vec);
    }

    for (; i < data_.size(); ++i)
        d[i] = static_cast<_f64>(data_[i]);

    return tensor<_f64>(shape_, d);
}

template<class _Tp>
tensor<uint64_t> tensor<_Tp>::neon_unsigned_long_() const {
    if (!std::is_convertible_v<value_type, uint64_t>)
        throw type_error("Type must be convertible to unsigned 64 bit int");

    if (empty())
        return tensor<uint64_t>(shape_);

    std::vector<uint64_t> d(data_.size());
    const index_type      simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type            i        = 0;

    if constexpr (std::is_unsigned_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32   data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            uint64x2_t int_vec1 = vmovl_u32(vget_low_u32(data_vec));
            uint64x2_t int_vec2 = vmovl_u32(vget_high_u32(data_vec));

            vst1q_u64(reinterpret_cast<uint64_t*>(&d[i]), int_vec1);
            vst1q_u64(reinterpret_cast<uint64_t*>(&d[i + 2]), int_vec2);
        }
    }
    else
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f64 data_vec = vld1q_f64(reinterpret_cast<const _f64*>(&data_[i]));
            neon_f64 uint_vec = vcvtq_u64_f64(data_vec);

            vst1q_u64(reinterpret_cast<uint64_t*>(&d[i]), uint_vec);
        }
    }

    for (; i < data_.size(); ++i)
        d[i] = static_cast<uint64_t>(data_[i]);

    return tensor<uint64_t>(shape_, d);
}

template<class _Tp>
tensor<int64_t> tensor<_Tp>::neon_long_() const {
    if (!std::is_convertible_v<value_type, int64_t>)
        throw type_error("Type must be convertible to 64 bit int");

    if (empty())
        return tensor<int64_t>(shape_);

    std::vector<int64_t> d(data_.size());
    const index_type     simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);
    index_type           i        = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_f64  data_vec1 = vld1q_f64(reinterpret_cast<const _f64*>(&data_[i]));
            neon_f64  data_vec2 = vld1q_f64(reinterpret_cast<const _f64*>(&data_[i + 2]));
            int64x2_t int_vec1  = vcvtq_s64_f64(data_vec1);
            int64x2_t int_vec2  = vcvtq_s64_f64(data_vec2);

            vst1q_s64(reinterpret_cast<int64_t*>(&d[i]), int_vec1);
            vst1q_s64(reinterpret_cast<int64_t*>(&d[i + 2]), int_vec2);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_u32   data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            uint64x2_t int_vec1 = vmovl_u32(vget_low_u32(data_vec));
            uint64x2_t int_vec2 = vmovl_u32(vget_high_u32(data_vec));

            vst1q_u64(reinterpret_cast<uint64_t*>(&d[i]), int_vec1);
            vst1q_u64(reinterpret_cast<uint64_t*>(&d[i + 2]), int_vec2);
        }
    }
    else
    {
        for (; i < simd_end; i += _ARM64_REG_WIDTH)
        {
            neon_s32  data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            int64x2_t int_vec1 = vmovl_s32(vget_low_s32(data_vec));
            int64x2_t int_vec2 = vmovl_s32(vget_high_s32(data_vec));

            vst1q_s64(reinterpret_cast<int64_t*>(&d[i]), int_vec1);
            vst1q_s64(reinterpret_cast<int64_t*>(&d[i + 2]), int_vec2);
        }
    }

    for (; i < data_.size(); ++i)
        d[i] = static_cast<int64_t>(data_[i]);

    return tensor<int64_t>(shape_, d);
}
