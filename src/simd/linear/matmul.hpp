#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_matmul(const tensor& other) const {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    if constexpr (!has_plus_operator_v<value_type>)
    {
        throw operator_error("Value type must have a plus operator");
    }

    if constexpr (!has_times_operator_v<value_type>)
    {
        throw operator_error("Value type must have times operator");
    }

    if (shape_.size() < 2 or other.shape().size() < 2)
    {
        throw shape_error("matmul is only supported for 2D tensors");
    }

    if (!equal_shape(shape_, shape_type({shape_[0], shape_[1]}))
        or !equal_shape(other.shape(), shape_type({other.shape()[0], other.shape()[1]})))
    {
        throw shape_error("matmul is only supported for 2D tensors");
    }

    if (shape_[1] != other.shape()[0])
    {
        if (shape_[0] == other.shape()[1])
        {
            return other.matmul(*this);
        }

        throw shape_error("Shape mismatch for matrix multiplication: "
                          "this shape: ["
                          + std::to_string(shape_[0]) + ", " + std::to_string(shape_[1])
                          + "] "
                            "other shape: ["
                          + std::to_string(other.shape()[0]) + ", " + std::to_string(other.shape()[1]) + "]");
    }

    shape_type ret_sh = {shape_[0], other.shape()[1]};
    data_t     ret_d(ret_sh[0] * ret_sh[1], value_type(0));

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");
    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        for (index_type i = 0; i < ret_sh[0]; i += simd_width)
        {
            for (index_type j = 0; j < ret_sh[1]; j += simd_width)
            {
                for (index_type k = 0; k < shape_[1]; k += simd_width)
                {
                    for (index_type ii = i; ii < std::min(static_cast<index_type>(i + simd_width), ret_sh[0]); ++ii)
                    {
                        for (index_type jj = j; jj < std::min(static_cast<index_type>(j + simd_width), ret_sh[1]); ++jj)
                        {
                            neon_f32 sum_vec = vdupq_n_f32(0);

                            for (index_type kk = k; kk < std::min(static_cast<index_type>(k + simd_width), shape_[1]);
                                 kk += simd_width)
                            {
                                neon_f32 a_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[ii * shape_[1] + kk]));
                                neon_f32 b_vec =
                                  vld1q_f32(reinterpret_cast<const _f32*>(&other.data_[kk * other.shape()[1] + jj]));
                                sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
                            }

                            float32x2_t sum_low  = vget_low_f32(sum_vec);
                            float32x2_t sum_high = vget_high_f32(sum_vec);
                            sum_low              = vadd_f32(sum_low, sum_high);
                            float32x2_t sum_dup  = vpadd_f32(sum_low, sum_low);
                            ret_d[ii * ret_sh[1] + jj] += vget_lane_f32(sum_dup, 0);
                        }
                    }
                }
            }
        }
    }
    else if constexpr (std::is_same_v<value_type, _s32>)
    {
        for (index_type i = 0; i < ret_sh[0]; i += simd_width)
        {
            for (index_type j = 0; j < ret_sh[1]; j += simd_width)
            {
                for (index_type k = 0; k < shape_[1]; k += simd_width)
                {
                    for (index_type ii = i; ii < std::min(static_cast<index_type>(i + simd_width), ret_sh[0]); ++ii)
                    {
                        for (index_type jj = j; jj < std::min(static_cast<index_type>(j + simd_width), ret_sh[1]); ++jj)
                        {
                            neon_s32 sum_vec = vdupq_n_s32(0);

                            for (index_type kk = k; kk < std::min(static_cast<index_type>(k + simd_width), shape_[1]);
                                 kk += simd_width)
                            {
                                neon_s32 a_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[ii * shape_[1] + kk]));
                                neon_s32 b_vec =
                                  vld1q_s32(reinterpret_cast<const _s32*>(&other.data_[kk * other.shape()[1] + jj]));
                                sum_vec = vmlaq_s32(sum_vec, a_vec, b_vec);
                            }

                            int32x2_t sum_low  = vget_low_s32(sum_vec);
                            int32x2_t sum_high = vget_high_s32(sum_vec);
                            sum_low            = vadd_s32(sum_low, sum_high);
                            int32x2_t sum_dup  = vpadd_s32(sum_low, sum_low);
                            ret_d[ii * ret_sh[1] + jj] += vget_lane_s32(sum_dup, 0);
                        }
                    }
                }
            }
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        for (index_type i = 0; i < ret_sh[0]; i += simd_width)
        {
            for (index_type j = 0; j < ret_sh[1]; j += simd_width)
            {
                for (index_type k = 0; k < shape_[1]; k += simd_width)
                {
                    for (index_type ii = i; ii < std::min(static_cast<index_type>(i + simd_width), ret_sh[0]); ++ii)
                    {
                        for (index_type jj = j; jj < std::min(static_cast<index_type>(j + simd_width), ret_sh[1]); ++jj)
                        {
                            neon_u32 sum_vec = vdupq_n_u32(0);

                            for (int64_t kk = k; kk < std::min(static_cast<index_type>(k + simd_width), shape_[1]);
                                 kk += simd_width)
                            {
                                neon_u32 a_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[ii * shape_[1] + kk]));
                                neon_u32 b_vec =
                                  vld1q_u32(reinterpret_cast<const _u32*>(&other.data_[kk * other.shape()[1] + jj]));
                                sum_vec = vmlaq_u32(sum_vec, a_vec, b_vec);
                            }

                            uint32x2_t sum_low  = vget_low_u32(sum_vec);
                            uint32x2_t sum_high = vget_high_u32(sum_vec);
                            sum_low             = vadd_u32(sum_low, sum_high);
                            uint32x2_t sum_dup  = vpadd_u32(sum_low, sum_low);
                            ret_d[ii * ret_sh[1] + jj] += vget_lane_u32(sum_dup, 0);
                        }
                    }
                }
            }
        }
    }

    for (index_type i = 0; i < ret_sh[0]; ++i)
    {
        for (index_type j = 0; j < ret_sh[1]; ++j)
        {
            value_type sum = value_type(0);
            for (index_type k = 0; k < shape_[1]; ++k)
            {
                sum = sum + (data_[i * shape_[1] + k] * other[k * other.shape()[1] + j]);
            }

            ret_d[i * ret_sh[1] + j] = sum;
        }
    }

    return self(ret_sh, ret_d);
}