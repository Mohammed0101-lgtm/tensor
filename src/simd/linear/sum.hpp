#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_sum(const index_type __axis) const {
  if (__axis < 0 || __axis >= static_cast<index_type>(this->__shape_.size()))
    throw std::invalid_argument("Invalid axis for sum");

  shape_type __ret_sh = this->__shape_;
  __ret_sh[__axis]    = 1;
  index_type __ret_size =
      std::accumulate(__ret_sh.begin(), __ret_sh.end(), 1, std::multiplies<index_type>());
  data_t __ret_data(__ret_size, value_type(0.0f));

  const index_type __axis_size  = this->__shape_[__axis];
  const index_type __outer_size = this->__compute_outer_size(__axis);
  const index_type __inner_size = this->size(0) / (__outer_size * __axis_size);

  if constexpr (std::is_floating_point_v<value_type>) {
    for (index_type __outer = 0; __outer < __outer_size; ++__outer) {
      for (index_type __inner = 0; __inner < __inner_size; ++__inner) {
        neon_f32   __sum_vec = vdupq_n_f32(0.0f);
        index_type __i       = __outer * __axis_size * __inner_size + __inner;
        index_type __j       = 0;

        for (; __j + _ARM64_REG_WIDTH <= __axis_size; __j += _ARM64_REG_WIDTH) {
          neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
          __sum_vec           = vaddq_f32(__sum_vec, __data_vec);
          __i += __inner_size * _ARM64_REG_WIDTH;
        }

        _f32 __sum = vaddvq_f32(__sum_vec);

        for (; __j < __axis_size; ++__j) {
          __sum += this->__data_[__i];
          __i += __inner_size;
        }

        __ret_data[__outer * __inner_size + __inner] = __sum;
      }
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (index_type __outer = 0; __outer < __outer_size; ++__outer) {
      for (index_type __inner = 0; __inner < __inner_size; ++__inner) {
        neon_s32   __sum_vec = vdupq_n_s32(0);
        index_type __i       = __outer * __axis_size * __inner_size + __inner;
        index_type __j       = 0;

        for (; __j + _ARM64_REG_WIDTH <= __axis_size; __j += _ARM64_REG_WIDTH) {
          neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
          __sum_vec           = vaddq_s32(__sum_vec, __data_vec);
          __i += __inner_size * _ARM64_REG_WIDTH;
        }

        _s32 __sum = vaddvq_s32(__sum_vec);

        for (; __j < __axis_size; ++__j) {
          __sum += this->__data_[__i];
          __i += __inner_size;
        }

        __ret_data[__outer * __inner_size + __inner] = __sum;
      }
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    for (index_type __outer = 0; __outer < __outer_size; ++__outer) {
      for (index_type __inner = 0; __inner < __inner_size; ++__inner) {
        neon_u32   __sum_vec = vdupq_n_u32(0);
        index_type __i       = __outer * __axis_size * __inner_size + __inner;
        index_type __j       = 0;

        for (; __j + _ARM64_REG_WIDTH <= __axis_size; __j += _ARM64_REG_WIDTH) {
          neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
          __sum_vec           = vaddq_u32(__sum_vec, __data_vec);
          __i += __inner_size * _ARM64_REG_WIDTH;
        }

        _u32 __sum = vaddvq_u32(__sum_vec);

        for (; __j < __axis_size; ++__j) {
          __sum += this->__data_[__i];
          __i += __inner_size;
        }

        __ret_data[__outer * __inner_size + __inner] = __sum;
      }
    }
  } else {
    index_type __i = 0;
    for (; __i < static_cast<index_type>(this->__data_.size()); ++__i) {
      std::vector<index_type> __orig(this->__shape_.size());
      index_type              __index = __i;
      index_type              __j     = static_cast<index_type>(this->__shape_.size()) - 1;

      for (; __j >= 0; __j--) {
        __orig[__j] = __index % this->__shape_[__j];
        __index /= this->__shape_[__j];
      }

      __orig[__axis]         = 0;
      index_type __ret_index = 0;
      index_type __st        = 1;

      for (__j = static_cast<index_type>(this->__shape_.size()) - 1; __j >= 0; __j--) {
        __ret_index += __orig[__j] * __st;
        __st *= __ret_sh[__j];
      }
      __ret_data[__ret_index] += this->__data_[__i];
    }
  }

  return __self(__ret_data, __ret_sh);
}