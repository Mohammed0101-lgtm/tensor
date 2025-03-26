#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_matmul(const tensor& __other) const {
  static_assert(has_times_operator_v<value_type>, "Value type must have a times operator");
  static_assert(has_plus_operator_v<value_type>, "Value type must have a plus operator");

  if (this->__shape_.size() < 2 || __other.shape().size() < 2)
    throw __shape_error__("matmul is only supported for 2D tensors");

  if (!__equal_shape(this->__shape_, shape_type({this->__shape_[0], this->__shape_[1]})) ||
      !__equal_shape(__other.shape(), shape_type({__other.shape()[0], __other.shape()[1]})))
    throw __shape_error__("matmul is only supported for 2D tensors");

  if (this->__shape_[1] != __other.shape()[0]) {
    if (this->__shape_[0] == __other.shape()[1]) return __other.matmul(*this);

    throw __shape_error__(
        "Shape mismatch for matrix multiplication: "
        "this shape: [" +
        std::to_string(this->__shape_[0]) + ", " + std::to_string(this->__shape_[1]) +
        "] "
        "other shape: [" +
        std::to_string(__other.shape()[0]) + ", " + std::to_string(__other.shape()[1]) + "]");
  }

  shape_type __ret_sh = {this->__shape_[0], __other.shape()[1]};
  data_t     __ret_d(__ret_sh[0] * __ret_sh[1], value_type(0));

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (int64_t __i = 0; __i < __ret_sh[0]; __i += _ARM64_REG_WIDTH) {
      for (int64_t __j = 0; __j < __ret_sh[1]; __j += _ARM64_REG_WIDTH) {
        for (int64_t __k = 0; __k < this->__shape_[1]; __k += _ARM64_REG_WIDTH) {
          for (int64_t __ii = __i;
               __ii < std::min(static_cast<index_type>(__i + _ARM64_REG_WIDTH), __ret_sh[0]);
               ++__ii) {
            for (int64_t __jj = __j;
                 __jj < std::min(static_cast<index_type>(__j + _ARM64_REG_WIDTH), __ret_sh[1]);
                 ++__jj) {
              neon_f32 __sum_vec = vdupq_n_f32(0);

              for (int64_t __kk = __k;
                   __kk <
                   std::min(static_cast<index_type>(__k + _ARM64_REG_WIDTH), this->__shape_[1]);
                   __kk += _ARM64_REG_WIDTH) {
                neon_f32 __a_vec = vld1q_f32(
                    reinterpret_cast<const _f32*>(&this->__data_[__ii * this->__shape_[1] + __kk]));
                neon_f32 __b_vec = vld1q_f32(reinterpret_cast<const _f32*>(
                    &__other.__data_[__kk * __other.shape()[1] + __jj]));
                __sum_vec        = vmlaq_f32(__sum_vec, __a_vec, __b_vec);
              }

              float32x2_t __sum_low  = vget_low_f32(__sum_vec);
              float32x2_t __sum_high = vget_high_f32(__sum_vec);
              __sum_low              = vadd_f32(__sum_low, __sum_high);
              float32x2_t __sum_dup  = vpadd_f32(__sum_low, __sum_low);
              __ret_d[__ii * __ret_sh[1] + __jj] += vget_lane_f32(__sum_dup, 0);
            }
          }
        }
      }
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (int64_t __i = 0; __i < __ret_sh[0]; __i += _ARM64_REG_WIDTH) {
      for (int64_t __j = 0; __j < __ret_sh[1]; __j += _ARM64_REG_WIDTH) {
        for (int64_t __k = 0; __k < this->__shape_[1]; __k += _ARM64_REG_WIDTH) {
          for (int64_t __ii = __i;
               __ii < std::min(static_cast<index_type>(__i + _ARM64_REG_WIDTH), __ret_sh[0]);
               ++__ii) {
            for (int64_t __jj = __j;
                 __jj < std::min(static_cast<index_type>(__j + _ARM64_REG_WIDTH), __ret_sh[1]);
                 ++__jj) {
              neon_s32 __sum_vec = vdupq_n_s32(0);

              for (int64_t __kk = __k;
                   __kk <
                   std::min(static_cast<index_type>(__k + _ARM64_REG_WIDTH), this->__shape_[1]);
                   __kk += _ARM64_REG_WIDTH) {
                neon_s32 __a_vec = vld1q_s32(
                    reinterpret_cast<const _s32*>(&this->__data_[__ii * this->__shape_[1] + __kk]));
                neon_s32 __b_vec = vld1q_s32(reinterpret_cast<const _s32*>(
                    &__other.__data_[__kk * __other.shape()[1] + __jj]));
                __sum_vec        = vmlaq_s32(__sum_vec, __a_vec, __b_vec);
              }

              int32x2_t __sum_low  = vget_low_s32(__sum_vec);
              int32x2_t __sum_high = vget_high_s32(__sum_vec);
              __sum_low            = vadd_s32(__sum_low, __sum_high);
              int32x2_t __sum_dup  = vpadd_s32(__sum_low, __sum_low);
              __ret_d[__ii * __ret_sh[1] + __jj] += vget_lane_s32(__sum_dup, 0);
            }
          }
        }
      }
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    for (int64_t __i = 0; __i < __ret_sh[0]; __i += _ARM64_REG_WIDTH) {
      for (int64_t __j = 0; __j < __ret_sh[1]; __j += _ARM64_REG_WIDTH) {
        for (int64_t __k = 0; __k < this->__shape_[1]; __k += _ARM64_REG_WIDTH) {
          for (int64_t __ii = __i;
               __ii < std::min(static_cast<index_type>(__i + _ARM64_REG_WIDTH), __ret_sh[0]);
               ++__ii) {
            for (int64_t __jj = __j;
                 __jj < std::min(static_cast<index_type>(__j + _ARM64_REG_WIDTH), __ret_sh[1]);
                 ++__jj) {
              neon_u32 __sum_vec = vdupq_n_u32(0);

              for (int64_t __kk = __k;
                   __kk <
                   std::min(static_cast<index_type>(__k + _ARM64_REG_WIDTH), this->__shape_[1]);
                   __kk += _ARM64_REG_WIDTH) {
                neon_u32 __a_vec = vld1q_u32(
                    reinterpret_cast<const _u32*>(&this->__data_[__ii * this->__shape_[1] + __kk]));
                neon_u32 __b_vec = vld1q_u32(reinterpret_cast<const _u32*>(
                    &__other.__data_[__kk * __other.shape()[1] + __jj]));
                __sum_vec        = vmlaq_u32(__sum_vec, __a_vec, __b_vec);
              }

              uint32x2_t __sum_low  = vget_low_u32(__sum_vec);
              uint32x2_t __sum_high = vget_high_u32(__sum_vec);
              __sum_low             = vadd_u32(__sum_low, __sum_high);
              uint32x2_t __sum_dup  = vpadd_u32(__sum_low, __sum_low);
              __ret_d[__ii * __ret_sh[1] + __jj] += vget_lane_u32(__sum_dup, 0);
            }
          }
        }
      }
    }
  }

#pragma omp parallel for collapse(2)
  for (int64_t __i = 0; __i < __ret_sh[0]; ++__i) {
    for (int64_t __j = 0; __j < __ret_sh[1]; ++__j) {
      value_type __sum = value_type(0);
      for (int64_t __k = 0; __k < this->__shape_[1]; ++__k)
        __sum = __sum + (this->__data_[__i * this->__shape_[1] + __k] *
                         __other[__k * __other.shape()[1] + __j]);

      __ret_d[__i * __ret_sh[1] + __j] = __sum;
    }
  }

  return __self(__ret_sh, __ret_d);
}