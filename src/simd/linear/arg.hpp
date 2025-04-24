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

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    for (i = 0; i < outer_size; ++i)
    {
        index_type j = 0;
        for (; j < inner_size; ++j)
        {
            value_type zero(0);
            value_type one(1);
            value_type inf = -std::numeric_limits<value_type>::infinity();

            neon_type<value_type> max_vec       = neon_dup<value_type>(&inf);
            neon_type<value_type> index_vec     = neon_dup<value_type>(&zero);
            neon_type<value_type> increment     = neon_dup<value_type>(&one);
            neon_type<value_type> current_index = {value_type(0), value_type(1), value_type(2), value_type(3)};

            index_type k = 0;
            for (; k + simd_width <= shape_[dim]; k += simd_width)
            {
                neon_type<value_type> data_vec = neon_load<value_type>(&data_[(i * shape_[dim] + k) * inner_size + j]);
                neon_type<value_type> mask     = neon_vcgtq<value_type>(data_vec, max_vec);
                max_vec                        = neon_vbslq<value_type>(mask, data_vec, max_vec);
                index_vec                      = neon_vbslq<value_type>(mask, current_index, index_vec);
                current_index                  = neon_add<value_type>(current_index, increment);
            }

            value_type max_vals[simd_width];
            _u32       indices[simd_width];

            neon_store<value_type>(max_vals, max_vec);
            neon_store<value_type>(indices, index_vec);

            value_type max_val   = max_vals[0];
            _u32       max_index = indices[0];

            for (int k = 1; k < simd_width; ++k)
            {
                if (max_vals[k] > max_val)
                {
                    max_val   = max_vals[k];
                    max_index = indices[k];
                }

                for (; k < shape_[dim]; ++k)
                {
                    value_type v = data_[(i * shape_[dim] + k) * inner_size + j];
                    if (v > max_val)
                    {
                        max_val   = v;
                        max_index = k;
                    }
                }
                ret[i * inner_size + j] = max_index;
            }
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

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    for (i = 0; i < outer_size; ++i)
    {
        for (index_type j = 0; j < inner_size; ++j)
        {
            value_type            inf     = -std::numeric_limits<value_type>::infinity();
            neon_type<value_type> max_vec = neon_dup<value_type>(&inf);
            index_type            k       = 0;

            for (; k + simd_width <= shape_[dim]; k += simd_width)
            {
                neon_type<value_type> data_vec = neon_load<value_type>(&data_[(i * shape_[dim] + k) * inner_size + j]);
                max_vec                        = neon_max<value_type>(max_vec, data_vec);
            }

            value_type max_value = neon_maxv<value_type>(max_vec);
            for (; k < shape_[dim]; ++k)
            {
                value_type v = data_[(i * shape_[dim] + k) * inner_size + j];
                max_value    = std::max(max_value, v);
            }

            ret[i * inner_size + j] = max_value;
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

    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    index_type i = 0;

    if constexpr (std::is_floating_point_v<value_type>)
    {
        for (; i + simd_width <= size; i += simd_width)
        {
            neon_f32    data_vec   = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            float32x2_t min1       = vpmin_f32(vget_low_f32(data_vec), vget_high_f32(data_vec));
            float32x2_t min2       = vpmin_f32(min1, min1);
            neon_f32    cmp_vec    = vdupq_lane_f32(min2, 0);
            neon_u32    cmp_result = ascending ? vcltq_f32(data_vec, cmp_vec) : vcgtq_f32(data_vec, cmp_vec);

            for (int j = 0; j < simd_width; ++j)
            {
                indices[i + j] = (cmp_result[j] ? i + j : i + j + 1);
            }
        }
    }
    else if constexpr (std::is_signed_v<value_type>)
    {
        for (; i + simd_width <= size; i += simd_width)
        {
            neon_s32  data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
            int32x2_t min1       = vpmin_s32(vget_low_s32(data_vec), vget_high_s32(data_vec));
            int32x2_t min2       = vpmin_s32(min1, min1);
            neon_s32  cmp_vec    = vdupq_lane_s32(min2, 0);
            neon_u32  cmp_result = ascending ? vcltq_s32(data_vec, cmp_vec) : vcgtq_s32(data_vec, cmp_vec);

            for (int j = 0; j < simd_width; ++j)
            {
                indices[i + j] = (cmp_result[j] ? i + j : i + j + 1);
            }
        }
    }
    else if constexpr (std::is_unsigned_v<value_type>)
    {
        for (; i + simd_width <= size; i += simd_width)
        {
            neon_u32   data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
            uint32x2_t min1       = vpmin_u32(vget_low_u32(data_vec), vget_high_u32(data_vec));
            uint32x2_t min2       = vpmin_u32(min1, min1);
            neon_u32   cmp_vec    = vdupq_lane_u32(min2, 0);
            neon_u32   cmp_result = ascending ? vcltq_u32(data_vec, cmp_vec) : vcgtq_u32(data_vec, cmp_vec);

            for (int j = 0; j < simd_width; ++j)
            {
                indices[i + j] = (cmp_result[j] ? i + j : i + j + 1);
            }
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