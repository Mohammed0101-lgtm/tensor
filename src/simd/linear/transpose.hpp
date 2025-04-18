#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_transpose() const {
    if (!equal_shape(shape_, shape_type({shape_[0], shape_[1]})))
        throw shape_error("Matrix transposition can only be done on 2D tensors");

    tensor           ret({shape_[1], shape_[0]});
    const index_type rows = shape_[0];
    const index_type cols = shape_[1];

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");
    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    if constexpr (std::is_same_v<_Tp, _f32>)
    {
        for (index_type i = 0; i < rows; i += simd_width)
        {
            for (index_type j = 0; j < cols; j += simd_width)
            {
                if (i + simd_width <= rows and j + simd_width <= cols)
                {
                    float32x4x4_t input;

                    for (index_type k = 0; k < simd_width; ++k)
                    {
                        input.val[k] = vld1q_f32(reinterpret_cast<const _f32*>(&data_[(i + k) * cols + j]));
                    }

                    float32x4x4_t output = vld4q_f32(reinterpret_cast<const _f32*>(&input));

                    for (index_type k = 0; k < simd_width; ++k)
                    {
                        vst1q_f32(&ret.data_[(j + k) * rows + i], output.val[k]);
                    }
                }
                else
                {
                    for (index_type ii = i; ii < std::min(static_cast<index_type>(i + simd_width), rows); ++ii)
                    {
                        for (index_type jj = j; jj < std::min(static_cast<index_type>(j + simd_width), cols); ++jj)
                        {
                            ret.at({jj, ii}) = at({ii, jj});
                        }
                    }
                }
            }
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        for (index_type i = 0; i < rows; i += simd_width)
        {
            for (index_type j = 0; j < cols; j += simd_width)
            {
                if (i + simd_width <= rows and j + simd_width <= cols)
                {
                    int32x4x4_t input;

                    for (index_type k = 0; k < simd_width; ++k)
                    {
                        input.val[k] = vld1q_s32(reinterpret_cast<const _s32*>(&data_[(i + k) * cols + j]));
                    }

                    int32x4x4_t output = vld4q_s32(reinterpret_cast<const _s32*>(&input));

                    for (index_type k = 0; k < simd_width; k++)
                    {
                        vst1q_s32(&ret.data_[(j + k) * rows + i], output.val[k]);
                    }
                }
                else
                {
                    for (index_type ii = i; ii < std::min(static_cast<index_type>(i + simd_width), rows); ++ii)
                    {
                        for (index_type jj = j; jj < std::min(static_cast<index_type>(j + simd_width), cols); ++jj)
                        {
                            ret.at({jj, ii}) = at({ii, jj});
                        }
                    }
                }
            }
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        for (index_type i = 0; i < rows; i += simd_width)
        {
            for (index_type j = 0; j < cols; j += simd_width)
            {
                if (i + simd_width <= rows and j + simd_width <= cols)
                {
                    uint32x4x4_t input;

                    for (index_type k = 0; k < simd_width; ++k)
                    {
                        input.val[k] = vld1q_u32(reinterpret_cast<const _u32*>(&data_[(i + k) * cols + j]));
                    }

                    uint32x4x4_t output = vld4q_u32(reinterpret_cast<const _u32*>(&input));

                    for (index_type k = 0; k < simd_width; ++k)
                    {
                        vst1q_u32(&ret.data_[(j + k) * rows + i], output.val[k]);
                    }
                }
                else
                {
                    for (index_type ii = i; ii < std::min(static_cast<index_type>(i + simd_width), rows); ++ii)
                    {
                        for (index_type jj = j; jj < std::min(static_cast<index_type>(j + simd_width), cols); ++jj)
                        {
                            ret.at({jj, ii}) = at({ii, jj});
                        }
                    }
                }
            }
        }
    }
    else
    {
        index_type i = 0;

        for (; i < rows; ++i)
        {
            index_type j = 0;
            for (; j < cols; ++j)
            {
                ret.at({j, i}) = at({i, j});
            }
        }
    }

    return ret;
}
