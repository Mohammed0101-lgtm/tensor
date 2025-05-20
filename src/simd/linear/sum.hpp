#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp> internal::neon::sum(tensor<_Tp>& t, const _u64 axis) {
    if (axis < 0 || axis >= static_cast<_u64>(shape_.size()))
        throw std::invalid_argument("Invalid axis for sum");

    std::vector<_Tp>& data_  = t.storage_();
    shape_type        ret_sh = shape_;
    ret_sh[axis]             = 1;
    _u64       ret_size      = std::accumulate(ret_sh.begin(), ret_sh.end(), 1, std::multiplies<_u64>());
    data_t     ret_data(ret_size, _Tp(0.0f));
    const _u64 axis_size  = shape_[axis];
    const _u64 outer_size = compute_outer_size(axis);
    const _u64 inner_size = size(0) / (outer_size * axis_size);
    const _u64 simd_end   = data_.size() - (data_.size() % t.simd_width);

    for (_u64 outer = 0; outer < outer_size; ++outer)
    {
        for (_u64 inner = 0; inner < inner_size; ++inner)
        {
            neon_type<_Tp> sum_vec = neon_dup<_Tp>(_Tp(0.0f));
            _u64           i       = outer * axis_size * inner_size + inner;
            _u64           j       = 0;

            for (; j < axis_size; j += t.simd_width)
            {
                neon_type<_Tp> data_vec = neon_load<_Tp>(&data_[i]);
                sum_vec                 = neon_add<_Tp>(sum_vec, data_vec);
                i += inner_size * t.simd_width;
            }

            _Tp sum = neon_addv<_Tp>(sum_vec);

            for (; j < axis_size; ++j)
            {
                sum = sum + data_[i];
                i += inner_size;
            }

            ret_data[outer * inner_size + inner] = sum;
        }
    }

    return self(ret_data, ret_sh);
}