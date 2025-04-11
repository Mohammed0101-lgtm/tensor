#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_sum(const index_type axis) const {
    if (axis < 0 or axis >= static_cast<index_type>(shape_.size()))
    {
        throw std::invalid_argument("Invalid axis for sum");
    }

    shape_type ret_sh   = shape_;
    ret_sh[axis]        = 1;
    index_type ret_size = std::accumulate(ret_sh.begin(), ret_sh.end(), 1, std::multiplies<index_type>());
    data_t     ret_data(ret_size, value_type(0.0f));

    const index_type axis_size  = shape_[axis];
    const index_type outer_size = compute_outer_size(axis);
    const index_type inner_size = size(0) / (outer_size * axis_size);

    if constexpr (std::is_floating_point_v<value_type>)
    {
        for (index_type outer = 0; outer < outer_size; ++outer)
        {
            for (index_type inner = 0; inner < inner_size; ++inner)
            {
                neon_f32   sum_vec = vdupq_n_f32(0.0f);
                index_type i       = outer * axis_size * inner_size + inner;
                index_type j       = 0;

                for (; j + _ARM64_REG_WIDTH <= axis_size; j += _ARM64_REG_WIDTH)
                {
                    neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
                    sum_vec           = vaddq_f32(sum_vec, data_vec);
                    i += inner_size * _ARM64_REG_WIDTH;
                }

                _f32 sum = vaddvq_f32(sum_vec);

                for (; j < axis_size; ++j)
                {
                    sum += data_[i];
                    i += inner_size;
                }

                ret_data[outer * inner_size + inner] = sum;
            }
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        for (index_type outer = 0; outer < outer_size; ++outer)
        {
            for (index_type inner = 0; inner < inner_size; ++inner)
            {
                neon_s32   sum_vec = vdupq_n_s32(0);
                index_type i       = outer * axis_size * inner_size + inner;
                index_type j       = 0;

                for (; j + _ARM64_REG_WIDTH <= axis_size; j += _ARM64_REG_WIDTH)
                {
                    neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
                    sum_vec           = vaddq_s32(sum_vec, data_vec);
                    i += inner_size * _ARM64_REG_WIDTH;
                }

                _s32 sum = vaddvq_s32(sum_vec);

                for (; j < axis_size; ++j)
                {
                    sum += data_[i];
                    i += inner_size;
                }

                ret_data[outer * inner_size + inner] = sum;
            }
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        for (index_type outer = 0; outer < outer_size; ++outer)
        {
            for (index_type inner = 0; inner < inner_size; ++inner)
            {
                neon_u32   sum_vec = vdupq_n_u32(0);
                index_type i       = outer * axis_size * inner_size + inner;
                index_type j       = 0;

                for (; j + _ARM64_REG_WIDTH <= axis_size; j += _ARM64_REG_WIDTH)
                {
                    neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
                    sum_vec           = vaddq_u32(sum_vec, data_vec);
                    i += inner_size * _ARM64_REG_WIDTH;
                }

                _u32 sum = vaddvq_u32(sum_vec);

                for (; j < axis_size; ++j)
                {
                    sum += data_[i];
                    i += inner_size;
                }

                ret_data[outer * inner_size + inner] = sum;
            }
        }
    }
    else
    {
        index_type i = 0;
        for (; i < static_cast<index_type>(data_.size()); ++i)
        {
            std::vector<index_type> orig(shape_.size());
            index_type              index = i;
            index_type              j     = static_cast<index_type>(shape_.size()) - 1;

            for (; j >= 0; j--)
            {
                orig[j] = index % shape_[j];
                index /= shape_[j];
            }

            orig[axis]           = 0;
            index_type ret_index = 0;
            index_type st        = 1;

            for (j = static_cast<index_type>(shape_.size()) - 1; j >= 0; j--)
            {
                ret_index += orig[j] * st;
                st *= ret_sh[j];
            }
            ret_data[ret_index] += data_[i];
        }
    }

    return self(ret_data, ret_sh);
}