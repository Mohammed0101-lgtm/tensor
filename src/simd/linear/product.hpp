#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_cross_product(const tensor& other) const {
    if (!std::is_arithmetic_v<value_type>)
        throw type_error("Type must be arithmetic");

    if (empty() or other.empty())
        throw std::invalid_argument("Cannot cross product an empty vector");

    if (!equal_shape(shape(), shape_type({3})) or !equal_shape(other.shape(), shape_type({3})))
        throw shape_error("Cross product can only be performed on 3-element vectors");

    tensor ret({3});

    if constexpr (std::is_floating_point_v<value_type>)
    {
        neon_f32 a      = vld1q_f32(reinterpret_cast<const _f32*>(data_.data()));
        neon_f32 b      = vld1q_f32(reinterpret_cast<const _f32*>(other.storage().data()));
        neon_f32 a_yzx  = vextq_f32(a, a, 1);
        neon_f32 b_yzx  = vextq_f32(b, b, 1);
        neon_f32 result = vsubq_f32(vmulq_f32(a_yzx, b), vmulq_f32(a, b_yzx));
        result          = vextq_f32(result, result, 3);

        vst1q_f32(reinterpret_cast<_f32*>(ret.storage().data()), result);
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        neon_s32 a      = vld1q_s32(reinterpret_cast<const _s32*>(data_.data()));
        neon_s32 b      = vld1q_s32(reinterpret_cast<const _s32*>(other.storage().data()));
        neon_s32 a_yzx  = vextq_s32(a, a, 1);
        neon_s32 b_yzx  = vextq_s32(b, b, 1);
        neon_s32 result = vsubq_s32(vmulq_s32(a_yzx, b), vmulq_s32(a, b_yzx));
        result          = vextq_s32(result, result, 3);

        vst1q_s32(reinterpret_cast<_s32*>(ret.storage().data()), result);
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        neon_u32 a      = vld1q_u32(reinterpret_cast<const _u32*>(data_.data()));
        neon_u32 b      = vld1q_u32(reinterpret_cast<const _u32*>(other.storage().data()));
        neon_u32 a_yzx  = vextq_u32(a, a, 1);
        neon_u32 b_yzx  = vextq_u32(b, b, 1);
        neon_u32 result = vsubq_u32(vmulq_u32(a_yzx, b), vmulq_u32(a, b_yzx));
        result          = vextq_u32(result, result, 3);

        vst1q_u32(reinterpret_cast<_u32*>(ret.storage().data()), result);
    }

    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_dot(const tensor& other) const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    if (empty() or other.empty())
    {
        throw std::invalid_argument("Cannot dot product an empty vector");
    }

    if (equal_shape(shape_, shape_type({1})) and equal_shape(other.shape(), shape_type({1})))
    {
        if (shape_[0] != other.shape()[0])
        {
            throw shape_error("Vectors must have the same size for dot product");
        }
    }

    const_pointer     this_data  = data_.data();
    const_pointer     other_data = other.storage().data();
    const std::size_t size       = data_.size();
    value_type        ret        = 0;

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");
    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    if constexpr (std::is_floating_point_v<value_type>)
    {
        std::size_t i       = 0;
        neon_f32    sum_vec = vdupq_n_f32(0.0f);

        for (; i + simd_width <= size; i += simd_width)
        {
            neon_f32 a_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this_data[i]));
            neon_f32 b_vec = vld1q_f32(reinterpret_cast<const _f32*>(&other_data[i]));
            sum_vec        = vmlaq_f32(sum_vec, a_vec, b_vec);  // Perform multiply-accumulate
        }

        float32x2_t sum_half = vadd_f32(vget_high_f32(sum_vec), vget_low_f32(sum_vec));
        ret                  = vget_lane_f32(vpadd_f32(sum_half, sum_half), 0);

        for (; i < size; ++i)
        {
            ret += static_cast<value_type>(this_data[i]) * static_cast<value_type>(other_data[i]);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        std::size_t i       = 0;
        neon_u32    sum_vec = vdupq_n_u32(0.0f);

        for (; i + simd_width <= size; i += simd_width)
        {
            neon_u32 a_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this_data[i]));
            neon_u32 b_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other_data[i]));
            sum_vec        = vmlaq_u32(sum_vec, a_vec, b_vec);
        }

        uint32x2_t sum_half = vadd_u32(vget_high_u32(sum_vec), vget_low_u32(sum_vec));
        ret                 = vget_lane_u32(vpadd_u32(sum_half, sum_half), 0);

        for (; i < size; ++i)
        {
            ret += static_cast<value_type>(this_data[i]) * static_cast<value_type>(other_data[i]);
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        std::size_t i       = 0;
        neon_s32    sum_vec = vdupq_n_f32(0.0f);

        for (; i + simd_width <= size; i += simd_width)
        {
            neon_s32 a_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this_data[i]));
            neon_s32 b_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other_data[i]));
            sum_vec        = vmlaq_s32(sum_vec, a_vec, b_vec);  // Perform multiply-accumulate
        }

        int32x2_t sum_half = vadd_s32(vget_high_s32(sum_vec), vget_low_s32(sum_vec));
        ret                = vget_lane_s32(vpadd_s32(sum_half, sum_half), 0);

        for (; i < size; ++i)
        {
            ret += static_cast<value_type>(this_data[i]) * static_cast<value_type>(other_data[i]);
        }
    }

    return self({ret}, {1});
}