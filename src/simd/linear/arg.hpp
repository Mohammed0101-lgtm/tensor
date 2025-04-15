#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::neon_argmax_(index_type dim) const {
    if (dim < 0 or dim >= shape_.size())
    {
        throw index_error("Dimension out of range in argmax");
    }

    tensor<index_type> ret;
    shape_type         ret_sh = shape_;
    ret_sh.erase(ret_sh.begin() + dim);
    ret.shape_ = ret_sh;
    ret.data_.resize(computeSize(ret_sh), 0);

    index_type outer_size = 1;
    index_type inner_size = 1;
    index_type i          = 0;

    for (; i < dim; ++i)
    {
        outer_size *= shape_[i];
    }
    for (i = dim + 1; i < shape_.size(); ++i)
    {
        inner_size *= shape_[i];
    }

    if constexpr (std::is_floating_point_v<value_type>)
    {
        for (i = 0; i < outer_size; ++i)
        {
            index_type j = 0;
            for (; j < inner_size; ++j)
            {
                neon_f32   max_vec       = vdupq_n_f32(-std::numeric_limits<_f32>::infinity());
                neon_u32   index_vec     = vdupq_n_u32(0);
                neon_u32   increment     = vdupq_n_u32(1);
                neon_u32   current_index = {0, 1, 2, 3};
                index_type k             = 0;

                for (; k + _ARM64_REG_WIDTH <= shape_[dim]; k += _ARM64_REG_WIDTH)
                {
                    neon_f32 data_vec =
                      vld1q_f32(reinterpret_cast<const _f32*>(&data_[(i * shape_[dim] + k) * inner_size + j]));
                    neon_u32 mask = vcgtq_f32(data_vec, max_vec);
                    max_vec       = vbslq_f32(mask, data_vec, max_vec);
                    index_vec     = vbslq_u32(mask, current_index, index_vec);
                    current_index = vaddq_u32(current_index, increment);
                }

                _f32 max_values[_ARM64_REG_WIDTH];
                _u32 indices[_ARM64_REG_WIDTH];

                vst1q_f32(max_values, max_vec);
                vst1q_u32(indices, index_vec);

                _f32       max_value = max_values[0];
                index_type max_index = indices[0];

                for (int k = 1; k < _ARM64_REG_WIDTH; ++k)
                {
                    if (max_values[k] > max_value)
                    {
                        max_value = max_values[k];
                        max_index = indices[k];
                    }
                }

                for (; k < shape_[dim]; ++k)
                {
                    _f32 v = data_[(i * shape_[dim] + k) * inner_size + j];
                    if (v > max_value)
                    {
                        max_value = v;
                        max_index = k;
                    }
                }
                ret.data_[i * inner_size + j] = max_index;
            }
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        for (i = 0; i < outer_size; ++i)
        {
            index_type j = 0;
            for (; j < inner_size; ++j)
            {
                neon_s32   max_vec       = vdupq_n_s32(-std::numeric_limits<_s32>::infinity());
                neon_u32   index_vec     = vdupq_n_u32(0);
                neon_u32   increment     = vdupq_n_u32(1);
                neon_u32   current_index = {0, 1, 2, 3};
                index_type k             = 0;

                for (; k + _ARM64_REG_WIDTH <= shape_[dim]; k += _ARM64_REG_WIDTH)
                {
                    neon_s32 data_vec =
                      vld1q_s32(reinterpret_cast<const _s32*>(&data_[(i * shape_[dim] + k) * inner_size + j]));
                    neon_u32 mask = vcgtq_s32(data_vec, max_vec);
                    max_vec       = vbslq_s32(mask, data_vec, max_vec);
                    index_vec     = vbslq_u32(mask, current_index, index_vec);
                    current_index = vaddq_u32(current_index, increment);
                }

                _s32 max_values[_ARM64_REG_WIDTH];
                _u32 indices[_ARM64_REG_WIDTH];

                vst1q_s32(max_values, max_vec);
                vst1q_u32(indices, index_vec);

                _s32       max_value = max_values[0];
                index_type max_index = indices[0];

                for (int k = 1; k < _ARM64_REG_WIDTH; ++k)
                {
                    if (max_values[k] > max_value)
                    {
                        max_value = max_values[k];
                        max_index = indices[k];
                    }
                }

                for (; k < shape_[dim]; ++k)
                {
                    _s32 v = data_[(i * shape_[dim] + k) * inner_size + j];

                    if (v > max_value)
                    {
                        max_value = v;
                        max_index = k;
                    }
                }

                ret.data_[i * inner_size + j] = max_index;
            }
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        for (i = 0; i < outer_size; ++i)
        {
            index_type j = 0;
            for (; j < inner_size; ++j)
            {
                neon_u32   max_vec       = vdupq_n_u32(-std::numeric_limits<_u32>::infinity());
                neon_u32   index_vec     = vdupq_n_u32(0);
                neon_u32   increment     = vdupq_n_u32(1);
                neon_u32   current_index = {0, 1, 2, 3};
                index_type k             = 0;

                for (; k + _ARM64_REG_WIDTH <= shape_[dim]; k += _ARM64_REG_WIDTH)
                {
                    neon_u32 data_vec =
                      vld1q_u32(reinterpret_cast<const _u32*>(&data_[(i * shape_[dim] + k) * inner_size + j]));
                    neon_u32 mask = vcgtq_u32(data_vec, max_vec);
                    max_vec       = vbslq_u32(mask, data_vec, max_vec);
                    index_vec     = vbslq_u32(mask, current_index, index_vec);
                    current_index = vaddq_u32(current_index, increment);
                }

                _u32 max_values[_ARM64_REG_WIDTH];
                _u32 indices[_ARM64_REG_WIDTH];

                vst1q_u32(max_values, max_vec);
                vst1q_u32(indices, index_vec);

                _u32       max_value = max_values[0];
                index_type max_index = indices[0];

                for (int k = 1; k < _ARM64_REG_WIDTH; ++k)
                {
                    if (max_values[k] > max_value)
                    {
                        max_value = max_values[k];
                        max_index = indices[k];
                    }
                }

                for (; k < shape_[dim]; ++k)
                {
                    _u32 v = data_[(i * shape_[dim] + k) * inner_size + j];

                    if (v > max_value)
                    {
                        max_value = v;
                        max_index = k;
                    }
                }
                ret.data_[i * inner_size + j] = max_index;
            }
        }
    }

    for (i = 0; i < outer_size; ++i)
    {
        index_type j = 0;
        for (; j < inner_size; ++j)
        {
            index_type max_index = 0;
            value_type max_value = data_[i * shape_[dim] * inner_size + j];
            index_type k         = 1;
            for (; k < shape_[dim]; ++k)
            {
                value_type v = data_[(i * shape_[dim] + k) * inner_size + j];

                if (v > max_value)
                {
                    max_value = v;
                    max_index = k;
                }
            }
            ret.data_[i * inner_size + j] = max_index;
        }
    }

    return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::neon_argmax(index_type dim) const {
    if (dim < 0 or dim >= shape_.size())
    {
        throw index_error("Dimension out of range in argmax");
    }

    tensor     ret;
    shape_type ret_sh = shape_;

    ret_sh.erase(ret_sh.begin() + dim);
    ret.shape_ = ret_sh;
    ret.data_.resize(computeSize(ret_sh), value_type(0));

    index_type outer_size = 1;
    index_type inner_size = 1;
    index_type i          = 0;

    for (; i < dim; ++i)
    {
        outer_size *= shape_[i];
    }
    for (i = dim + 1; i < static_cast<index_type>(shape_.size()); ++i)
    {
        inner_size *= shape_[i];
    }

    if constexpr (std::is_floating_point_v<value_type>)
    {
        for (i = 0; i < outer_size; ++i)
        {
            for (index_type j = 0; j < inner_size; ++j)
            {
                neon_f32   max_vec = vdupq_n_f32(-std::numeric_limits<_f32>::infinity());
                index_type k       = 0;

                for (; k + _ARM64_REG_WIDTH <= shape_[dim]; k += _ARM64_REG_WIDTH)
                {
                    neon_f32 data_vec =
                      vld1q_f32(reinterpret_cast<const _f32*>(&data_[(i * shape_[dim] + k) * inner_size + j]));
                    max_vec = vmaxq_f32(max_vec, data_vec);
                }

                _f32 max_value = vmaxvq_f32(max_vec);
                for (; k < shape_[dim]; ++k)
                {
                    _f32 v    = data_[(i * shape_[dim] + k) * inner_size + j];
                    max_value = std::max(max_value, v);
                }

                ret.data_[i * inner_size + j] = max_value;
            }
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        for (i = 0; i < outer_size; ++i)
        {
            for (index_type j = 0; j < inner_size; ++j)
            {
                neon_s32   max_vec = vdupq_n_s32(-std::numeric_limits<_s32>::infinity());
                index_type k       = 0;

                for (; k + _ARM64_REG_WIDTH <= shape_[dim]; k += _ARM64_REG_WIDTH)
                {
                    neon_s32 data_vec =
                      vld1q_s32(reinterpret_cast<const _s32*>(&data_[(i * shape_[dim] + k) * inner_size + j]));
                    max_vec = vmaxq_s32(max_vec, data_vec);
                }

                _s32 max_value = vmaxvq_s32(max_vec);
                for (; k < shape_[dim]; ++k)
                {
                    _s32 v    = data_[(i * shape_[dim] + k) * inner_size + j];
                    max_value = std::max(max_value, v);
                }

                ret.data_[i * inner_size + j] = max_value;
            }
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        for (i = 0; i < outer_size; ++i)
        {
            for (index_type j = 0; j < inner_size; ++j)
            {
                neon_u32   max_vec = vdupq_n_u32(-std::numeric_limits<_u32>::infinity());
                index_type k       = 0;

                for (; k + _ARM64_REG_WIDTH <= shape_[dim]; k += _ARM64_REG_WIDTH)
                {
                    neon_u32 data_vec =
                      vld1q_u32(reinterpret_cast<const _u32*>(&data_[(i * shape_[dim] + k) * inner_size + j]));
                    max_vec = vmaxq_u32(max_vec, data_vec);
                }

                _u32 max_value = vmaxvq_u32(max_vec);
                for (; k < shape_[dim]; ++k)
                {
                    _u32 v    = data_[(i * shape_[dim] + k) * inner_size + j];
                    max_value = std::max(max_value, v);
                }

                ret.data_[i * inner_size + j] = max_value;
            }
        }
    }
    else
    {
        for (i = 0; i < outer_size; ++i)
        {
            index_type j = 0;
            for (; j < inner_size; ++j)
            {
                value_type max_value = data_[i * shape_[dim] * inner_size + j];
                index_type k         = 1;
                for (; k < shape_[dim]; ++k)
                {
                    value_type v = data_[(i * shape_[dim] + k) * inner_size + j];

                    if (v > max_value)
                    {
                        max_value = v;
                    }
                }
                ret.data_[i * inner_size + j] = max_value;
            }
        }
    }

    return ret;
}

template<class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::neon_argsort(index_type d, bool ascending) const {
    index_type adjusted = (d < 0) ? d + data_.size() : d;

    if (adjusted != 0)
    {
        throw index_error("Invalid dimension for argsort: only 1D tensors are supported");
    }

    index_type size = static_cast<index_type>(data_.size());
    shape_type indices(size);
    std::iota(indices.begin(), indices.end(), 0);

    index_type i = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        for (; i + _ARM64_REG_WIDTH <= size; i += _ARM64_REG_WIDTH)
        {
            neon_f32    data_vec   = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            float32x2_t min1       = vpmin_f32(vget_low_f32(data_vec), vget_high_f32(data_vec));
            float32x2_t min2       = vpmin_f32(min1, min1);
            neon_f32    cmp_vec    = vdupq_lane_f32(min2, 0);
            neon_u32    cmp_result = ascending ? vcltq_f32(data_vec, cmp_vec) : vcgtq_f32(data_vec, cmp_vec);

            for (int j = 0; j < _ARM64_REG_WIDTH; ++j)
                indices[i + j] = (cmp_result[j] ? i + j : i + j + 1);
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        for (; i + _ARM64_REG_WIDTH <= size; i += _ARM64_REG_WIDTH)
        {
            neon_s32  data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            int32x2_t min1       = vpmin_s32(vget_low_s32(data_vec), vget_high_s32(data_vec));
            int32x2_t min2       = vpmin_s32(min1, min1);
            neon_s32  cmp_vec    = vdupq_lane_s32(min2, 0);
            neon_u32  cmp_result = ascending ? vcltq_s32(data_vec, cmp_vec) : vcgtq_s32(data_vec, cmp_vec);

            for (int j = 0; j < _ARM64_REG_WIDTH; ++j)
                indices[i + j] = (cmp_result[j] ? i + j : i + j + 1);
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        for (; i + _ARM64_REG_WIDTH <= size; i += _ARM64_REG_WIDTH)
        {
            neon_u32   data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            uint32x2_t min1       = vpmin_u32(vget_low_u32(data_vec), vget_high_u32(data_vec));
            uint32x2_t min2       = vpmin_u32(min1, min1);
            neon_u32   cmp_vec    = vdupq_lane_u32(min2, 0);
            neon_u32   cmp_result = ascending ? vcltq_u32(data_vec, cmp_vec) : vcgtq_u32(data_vec, cmp_vec);

            for (int j = 0; j < _ARM64_REG_WIDTH; ++j)
                indices[i + j] = (cmp_result[j] ? i + j : i + j + 1);
        }
    }

    for (; i < size; ++i)
    {
        indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(),
              [&](index_type a, index_type b) { return ascending ? data_[a] < data_[b] : data_[a] > data_[b]; });

    return tensor<index_type>(indices);
}