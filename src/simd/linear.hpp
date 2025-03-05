#pragma once

#include "../tensorbase.hpp"
#include "../types.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_matmul(const tensor& __other) const {
  static_assert(has_times_operator_v<value_type>);
  static_assert(has_plus_operator_v<value_type>);

  if (this->__shape_.size() != 2 || __other.shape().size() != 2)
    throw std::invalid_argument("matmul is only supported for 2D tensors");

  if (this->__shape_[1] != __other.shape()[0]) {
    if (this->__shape_[0] == __other.shape()[1]) return __other.matmul(*this);

    throw std::invalid_argument(
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

  return __self(__ret_d, __ret_sh);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_cross_product(const tensor& __other) const {
  if (this->empty() || __other.empty())
    throw std::invalid_argument("Cannot cross product an empty vector");

  if (this->shape() != std::vector<int>{3} || __other.shape() != std::vector<int>{3})
    throw std::invalid_argument("Cross product can only be performed on 3-element vectors");

  tensor __ret({3});

  if constexpr (std::is_floating_point_v<value_type>) {
    neon_f32 __a      = vld1q_f32(reinterpret_cast<const _f32*>(this->__data_.data()));
    neon_f32 __b      = vld1q_f32(reinterpret_cast<const _f32*>(__other.storage().data()));
    neon_f32 __a_yzx  = vextq_f32(__a, __a, 1);
    neon_f32 __b_yzx  = vextq_f32(__b, __b, 1);
    neon_f32 __result = vsubq_f32(vmulq_f32(__a_yzx, __b), vmulq_f32(__a, __b_yzx));
    __result          = vextq_f32(__result, __result, 3);

    vst1q_f32(reinterpret_cast<_f32*>(__ret.storage().data()), __result);
  } else if constexpr (std::is_signed_v<value_type>) {
    neon_s32 __a      = vld1q_s32(reinterpret_cast<const _s32*>(this->__data_.data()));
    neon_s32 __b      = vld1q_s32(reinterpret_cast<const _s32*>(__other.storage().data()));
    neon_s32 __a_yzx  = vextq_s32(__a, __a, 1);
    neon_s32 __b_yzx  = vextq_s32(__b, __b, 1);
    neon_s32 __result = vsubq_s32(vmulq_s32(__a_yzx, __b), vmulq_s32(__a, __b_yzx));
    __result          = vextq_s32(__result, __result, 3);

    vst1q_s32(reinterpret_cast<_s32*>(__ret.storage().data()), __result);
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __a      = vld1q_u32(reinterpret_cast<const _u32*>(this->__data_.data()));
    neon_u32 __b      = vld1q_u32(reinterpret_cast<const _u32*>(__other.storage().data()));
    neon_u32 __a_yzx  = vextq_u32(__a, __a, 1);
    neon_u32 __b_yzx  = vextq_u32(__b, __b, 1);
    neon_u32 __result = vsubq_u32(vmulq_u32(__a_yzx, __b), vmulq_u32(__a, __b_yzx));
    __result          = vextq_u32(__result, __result, 3);

    vst1q_u32(reinterpret_cast<_u32*>(__ret.storage().data()), __result);
  }

  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_absolute(const tensor& __tensor) const {
  index_type __s = __tensor.storage().size();
  data_t     __a;
  __a.reserve(__s);
  index_type __i = 0;

  // TODO : implement neon acceleration for absolute
#pragma omp parallel
  for (; __i < __s; ++__i)
    __a.push_back(static_cast<value_type>(std::fabs(_f32(__tensor.storage()[__i]))));

  return __self(__a, __tensor.__shape_);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_dot(const tensor& __other) const {
  if (this->empty() || __other.empty())
    throw std::invalid_argument("Cannot dot product an empty vector");

  if (this->__shape_.size() == 1 && __other.shape().size() == 1) {
    if (this->__shape_[0] != __other.shape()[0])
      throw std::invalid_argument("Vectors must have the same size for dot product");

    const_pointer __this_data  = this->__data_.data();
    const_pointer __other_data = __other.storage().data();
    const size_t  __size       = this->__data_.size();
    value_type    __ret        = 0;

    if constexpr (std::is_floating_point_v<value_type>) {
      size_t   __i     = 0;
      neon_f32 sum_vec = vdupq_n_f32(0.0f);

      for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH) {
        neon_f32 a_vec = vld1q_f32(reinterpret_cast<const _f32*>(&__this_data[__i]));
        neon_f32 b_vec = vld1q_f32(reinterpret_cast<const _f32*>(&__other_data[__i]));
        sum_vec        = vmlaq_f32(sum_vec, a_vec, b_vec);  // Perform multiply-accumulate
      }

      float32x2_t sum_half = vadd_f32(vget_high_f32(sum_vec), vget_low_f32(sum_vec));
      __ret                = vget_lane_f32(vpadd_f32(sum_half, sum_half), 0);

      for (; __i < __size; ++__i)
        __ret +=
            static_cast<value_type>(__this_data[__i]) * static_cast<value_type>(__other_data[__i]);
    } else if constexpr (std::is_unsigned_v<value_type>) {
      size_t   __i     = 0;
      neon_u32 sum_vec = vdupq_n_u32(0.0f);

      for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH) {
        neon_u32 a_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__this_data[__i]));
        neon_u32 b_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other_data[__i]));
        sum_vec        = vmlaq_u32(sum_vec, a_vec, b_vec);
      }

      uint32x2_t sum_half = vadd_u32(vget_high_u32(sum_vec), vget_low_u32(sum_vec));
      __ret               = vget_lane_u32(vpadd_u32(sum_half, sum_half), 0);

      for (; __i < __size; ++__i)
        __ret +=
            static_cast<value_type>(__this_data[__i]) * static_cast<value_type>(__other_data[__i]);
    } else if constexpr (std::is_signed_v<value_type>) {
      size_t   __i     = 0;
      neon_s32 sum_vec = vdupq_n_f32(0.0f);

      for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH) {
        neon_s32 a_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__this_data[__i]));
        neon_s32 b_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other_data[__i]));
        sum_vec        = vmlaq_s32(sum_vec, a_vec, b_vec);  // Perform multiply-accumulate
      }

      int32x2_t sum_half = vadd_s32(vget_high_s32(sum_vec), vget_low_s32(sum_vec));
      __ret              = vget_lane_s32(vpadd_s32(sum_half, sum_half), 0);

      for (; __i < __size; ++__i)
        __ret +=
            static_cast<value_type>(__this_data[__i]) * static_cast<value_type>(__other_data[__i]);
    }

    return __self({__ret}, {1});
  }

  if (this->__shape_.size() == 2 && __other.shape().size() == 2) return this->matmul(__other);

  if (this->__shape_.size() == 3 && __other.shape().size() == 3)
    return this->cross_product(__other);

  return __self();
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_relu_() {
  if constexpr (std::is_unsigned_v<value_type>) return *this;

  index_type __s = this->__data_.size();
  index_type __i = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    const neon_f32 __vZero = vdupq_n_f32(0.0f);

    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH) {
      neon_f32 __v = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      __v          = vmaxq_f32(__v, __vZero);

      vst1q_f32(&this->__data_[__i], __v);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    const neon_s32 __vZero = vdupq_n_s32(0);

    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH) {
      neon_s32 __v = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      __v          = vmaxq_s32(__v, __vZero);

      vst1q_s32(&this->__data_[__i], __v);
    }
  }

  for (__i = 0; __i < __s; ++__i) this->__data_[__i] = std::max(this->__data_[__i], value_type(0));

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_transpose() const {
  if (this->__shape_.size() != 2)
    throw std::invalid_argument("Matrix transposition can only be done on 2D tensors");

  tensor           __ret({this->__shape_[1], this->__shape_[0]});
  const index_type __rows = this->__shape_[0];
  const index_type __cols = this->__shape_[1];

  if constexpr (std::is_same_v<_Tp, _f32>) {
    for (index_type __i = 0; __i < __rows; __i += _ARM64_REG_WIDTH) {
      for (index_type __j = 0; __j < __cols; __j += _ARM64_REG_WIDTH) {
        if (__i + _ARM64_REG_WIDTH <= __rows && __j + _ARM64_REG_WIDTH <= __cols) {
          float32x4x4_t __input;

          for (index_type __k = 0; __k < _ARM64_REG_WIDTH; ++__k)
            __input.val[__k] = vld1q_f32(
                reinterpret_cast<const _f32*>(&this->__data_[(__i + __k) * __cols + __j]));

          float32x4x4_t __output = vld4q_f32(reinterpret_cast<const _f32*>(&__input));

          for (index_type __k = 0; __k < _ARM64_REG_WIDTH; ++__k)
            vst1q_f32(&__ret.__data_[(__j + __k) * __rows + __i], __output.val[__k]);
        } else {
          for (index_type __ii = __i;
               __ii < std::min(static_cast<index_type>(__i + _ARM64_REG_WIDTH), __rows); ++__ii) {
            for (index_type __jj = __j;
                 __jj < std::min(static_cast<index_type>(__j + _ARM64_REG_WIDTH), __cols); ++__jj) {
              __ret.at({__jj, __ii}) = this->at({__ii, __jj});
            }
          }
        }
      }
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (index_type __i = 0; __i < __rows; __i += _ARM64_REG_WIDTH) {
      for (index_type __j = 0; __j < __cols; __j += _ARM64_REG_WIDTH) {
        if (__i + _ARM64_REG_WIDTH <= __rows && __j + _ARM64_REG_WIDTH <= __cols) {
          int32x4x4_t __input;

          for (index_type __k = 0; __k < _ARM64_REG_WIDTH; ++__k)
            __input.val[__k] = vld1q_s32(
                reinterpret_cast<const _s32*>(&this->__data_[(__i + __k) * __cols + __j]));

          int32x4x4_t __output = vld4q_s32(reinterpret_cast<const _s32*>(&__input));

          for (index_type __k = 0; __k < _ARM64_REG_WIDTH; __k++)
            vst1q_s32(&__ret.__data_[(__j + __k) * __rows + __i], __output.val[__k]);
        } else {
          for (index_type __ii = __i;
               __ii < std::min(static_cast<index_type>(__i + _ARM64_REG_WIDTH), __rows); ++__ii) {
            for (index_type __jj = __j;
                 __jj < std::min(static_cast<index_type>(__j + _ARM64_REG_WIDTH), __cols); ++__jj) {
              __ret.at({__jj, __ii}) = this->at({__ii, __jj});
            }
          }
        }
      }
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    for (index_type __i = 0; __i < __rows; __i += _ARM64_REG_WIDTH) {
      for (index_type __j = 0; __j < __cols; __j += _ARM64_REG_WIDTH) {
        if (__i + _ARM64_REG_WIDTH <= __rows && __j + _ARM64_REG_WIDTH <= __cols) {
          uint32x4x4_t __input;

          for (index_type __k = 0; __k < _ARM64_REG_WIDTH; ++__k)
            __input.val[__k] = vld1q_u32(
                reinterpret_cast<const _u32*>(&this->__data_[(__i + __k) * __cols + __j]));

          uint32x4x4_t __output = vld4q_u32(reinterpret_cast<const _u32*>(&__input));

          for (index_type __k = 0; __k < _ARM64_REG_WIDTH; ++__k)
            vst1q_u32(&__ret.__data_[(__j + __k) * __rows + __i], __output.val[__k]);
        } else {
          for (index_type __ii = __i;
               __ii < std::min(static_cast<index_type>(__i + _ARM64_REG_WIDTH), __rows); ++__ii) {
            for (index_type __jj = __j;
                 __jj < std::min(static_cast<index_type>(__j + _ARM64_REG_WIDTH), __cols); ++__jj) {
              __ret.at({__jj, __ii}) = this->at({__ii, __jj});
            }
          }
        }
      }
    }
  } else {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < __rows; ++__i) {
      index_type __j = 0;
      for (; __j < __cols; ++__j) __ret.at({__j, __i}) = this->at({__i, __j});
    }
  }

  return __ret;
}

template <class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::neon_argsort(index_type __d,
                                                                   bool       __ascending) const {
  index_type __adjusted = (__d < 0) ? __d + this->__data_.size() : __d;

  if (__adjusted != 0)
    throw std::out_of_range("Invalid dimension for argsort: only 1D tensors are supported");

  index_type __size = static_cast<index_type>(this->__data_.size());
  shape_type __indices(__size);
  std::iota(__indices.begin(), __indices.end(), 0);

  index_type __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH) {
      neon_f32    __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      float32x2_t __min1     = vpmin_f32(vget_low_f32(__data_vec), vget_high_f32(__data_vec));
      float32x2_t __min2     = vpmin_f32(__min1, __min1);
      neon_f32    __cmp_vec  = vdupq_lane_f32(__min2, 0);
      neon_u32    __cmp_result =
          __ascending ? vcltq_f32(__data_vec, __cmp_vec) : vcgtq_f32(__data_vec, __cmp_vec);

      for (int __j = 0; __j < _ARM64_REG_WIDTH; ++__j)
        __indices[__i + __j] = (__cmp_result[__j] ? __i + __j : __i + __j + 1);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH) {
      neon_s32  __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      int32x2_t __min1     = vpmin_s32(vget_low_s32(__data_vec), vget_high_s32(__data_vec));
      int32x2_t __min2     = vpmin_s32(__min1, __min1);
      neon_s32  __cmp_vec  = vdupq_lane_s32(__min2, 0);
      neon_u32  __cmp_result =
          __ascending ? vcltq_s32(__data_vec, __cmp_vec) : vcgtq_s32(__data_vec, __cmp_vec);

      for (int __j = 0; __j < _ARM64_REG_WIDTH; ++__j)
        __indices[__i + __j] = (__cmp_result[__j] ? __i + __j : __i + __j + 1);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH) {
      neon_u32   __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      uint32x2_t __min1     = vpmin_u32(vget_low_u32(__data_vec), vget_high_u32(__data_vec));
      uint32x2_t __min2     = vpmin_u32(__min1, __min1);
      neon_u32   __cmp_vec  = vdupq_lane_u32(__min2, 0);
      neon_u32   __cmp_result =
          __ascending ? vcltq_u32(__data_vec, __cmp_vec) : vcgtq_u32(__data_vec, __cmp_vec);

      for (int __j = 0; __j < _ARM64_REG_WIDTH; ++__j)
        __indices[__i + __j] = (__cmp_result[__j] ? __i + __j : __i + __j + 1);
    }
  }
#pragma omp parallel
  for (; __i < __size; ++__i) __indices[__i] = __i;

  std::sort(__indices.begin(), __indices.end(), [&](index_type __a, index_type __b) {
    return __ascending ? this->__data_[__a] < this->__data_[__b]
                       : this->__data_[__a] > this->__data_[__b];
  });

  return tensor<index_type>(__indices);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sigmoid_() {
  index_type __i = 0;

  using neon_type =
      typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, void>::type;

  if constexpr (std::is_same_v<value_type, _f32>) {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_type __v         = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_type __exp_neg_v = vexpq_f32(vnegq_f32(__v));  // e^(-x)
      neon_type __sigmoid =
          vrecpeq_f32(vaddq_f32(vdupq_n_f32(1.0f), __exp_neg_v));  // 1 / (1 + e^(-x))

      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __sigmoid);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] =
        static_cast<value_type>(1.0 / (1.0 + std::exp(-static_cast<double>(this->__data_[__i]))));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_clipped_relu_(const value_type __clip_limit) {
  if constexpr (std::is_unsigned_v<value_type>) return *this;

  index_type __s = this->__data_.size();
  index_type __i = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    const neon_f32 __vZero = vdupq_n_f32(0.0f);
    const neon_f32 __vClip = vdupq_n_f32(__clip_limit);

    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH) {
      neon_f32 __v = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      __v          = vminq_f32(vmaxq_f32(__v, __vZero), __vClip);

      vst1q_f32(&this->__data_[__i], __v);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    const neon_s32 __vZero = vdupq_n_s32(0);
    const neon_s32 __vClip = vdupq_n_s32(__clip_limit);

    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH) {
      neon_s32 __v = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      __v          = vminq_s32(vmaxq_s32(__v, __vZero), __vClip);

      vst1q_s32(&this->__data_[__i], __v);
    }
  }
#pragma omp parallel
  for (; __i < __s; ++__i)
    this->__data_[__i] = std::min(std::max(this->__data_[__i], value_type(0)), __clip_limit);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_clamp_(const_reference __min_val, const_reference __max_val) {
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    neon_f32 __min_vec = vdupq_n_f32(__min_val);
    neon_f32 __max_vec = vdupq_n_f32(__max_val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __clamped  = vminq_f32(vmaxq_f32(__data_vec, __min_vec), __max_vec);

      vst1q_f32(&this->__data_[__i], __clamped);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    neon_s32 __min_vec = vdupq_n_s32(__min_val);
    neon_s32 __max_vec = vdupq_n_s32(__max_val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __clamped  = vminq_s32(vmaxq_s32(__data_vec, __min_vec), __max_vec);

      vst1q_s32(&this->__data_[__i], __clamped);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __min_vec = vdupq_n_u32(__min_val);
    neon_u32 __max_vec = vdupq_n_u32(__max_val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __clamped  = vminq_u32(vmaxq_u32(__data_vec, __min_vec), __max_vec);

      vst1q_u32(&this->__data_[__i], __clamped);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) {
    this->__data_[__i] = std::max(__min_val, this->__data_[__i]);
    this->__data_[__i] = std::min(__max_val, this->__data_[__i]);
  }

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_floor_() {
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    for (; __i < this->__data_.size(); __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec  = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __floor_vec = vrndmq_f32(__data_vec);

      vst1q_f32(&this->__data_[__i], __floor_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::floor(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_ceil_() {
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    for (; __i + _ARM64_REG_WIDTH <= this->__data_.size(); __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __ceil_vec = vrndpq_f32(__data_vec);

      vst1q_f32(&this->__data_[__i], __ceil_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::ceil(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::neon_argmax_(index_type __dim) const {
  if (__dim < 0 || __dim >= this->__shape_.size())
    throw std::out_of_range("Dimension out of range in argmax");

  tensor<index_type> __ret;
  shape_type         __ret_sh = this->__shape_;
  __ret_sh.erase(__ret_sh.begin() + __dim);
  __ret.__shape_ = __ret_sh;
  __ret.__data_.resize(this->__computeSize(__ret_sh), 0);

  index_type __outer_size = 1;
  index_type __inner_size = 1;
  index_type __i          = 0;

#pragma omp parallel
  for (; __i < __dim; ++__i) __outer_size *= this->__shape_[__i];

#pragma omp parallel
  for (__i = __dim + 1; __i < this->__shape_.size(); ++__i) __inner_size *= this->__shape_[__i];

  if constexpr (std::is_floating_point_v<value_type>) {
    for (__i = 0; __i < __outer_size; ++__i) {
      index_type __j = 0;
      for (; __j < __inner_size; ++__j) {
        neon_f32   __max_vec       = vdupq_n_f32(-std::numeric_limits<_f32>::infinity());
        neon_u32   __index_vec     = vdupq_n_u32(0);
        neon_u32   __increment     = vdupq_n_u32(1);
        neon_u32   __current_index = {0, 1, 2, 3};
        index_type __k             = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH) {
          neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(
              &this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          neon_u32 __mask     = vcgtq_f32(__data_vec, __max_vec);
          __max_vec           = vbslq_f32(__mask, __data_vec, __max_vec);
          __index_vec         = vbslq_u32(__mask, __current_index, __index_vec);
          __current_index     = vaddq_u32(__current_index, __increment);
        }

        _f32 __max_values[_ARM64_REG_WIDTH];
        _u32 __indices[_ARM64_REG_WIDTH];

        vst1q_f32(__max_values, __max_vec);
        vst1q_u32(__indices, __index_vec);

        _f32       __max_value = __max_values[0];
        index_type __max_index = __indices[0];

        for (int __k = 1; __k < _ARM64_REG_WIDTH; ++__k) {
          if (__max_values[__k] > __max_value) {
            __max_value = __max_values[__k];
            __max_index = __indices[__k];
          }
        }

        for (; __k < this->__shape_[__dim]; ++__k) {
          _f32 __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          if (__v > __max_value) {
            __max_value = __v;
            __max_index = __k;
          }
        }
        __ret.__data_[__i * __inner_size + __j] = __max_index;
      }
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (__i = 0; __i < __outer_size; ++__i) {
      index_type __j = 0;
      for (; __j < __inner_size; ++__j) {
        neon_s32   __max_vec       = vdupq_n_s32(-std::numeric_limits<_s32>::infinity());
        neon_u32   __index_vec     = vdupq_n_u32(0);
        neon_u32   __increment     = vdupq_n_u32(1);
        neon_u32   __current_index = {0, 1, 2, 3};
        index_type __k             = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH) {
          neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(
              &this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          neon_u32 __mask     = vcgtq_s32(__data_vec, __max_vec);
          __max_vec           = vbslq_s32(__mask, __data_vec, __max_vec);
          __index_vec         = vbslq_u32(__mask, __current_index, __index_vec);
          __current_index     = vaddq_u32(__current_index, __increment);
        }

        _s32 __max_values[_ARM64_REG_WIDTH];
        _u32 __indices[_ARM64_REG_WIDTH];

        vst1q_s32(__max_values, __max_vec);
        vst1q_u32(__indices, __index_vec);

        _s32       __max_value = __max_values[0];
        index_type __max_index = __indices[0];

        for (int __k = 1; __k < _ARM64_REG_WIDTH; ++__k) {
          if (__max_values[__k] > __max_value) {
            __max_value = __max_values[__k];
            __max_index = __indices[__k];
          }
        }

        for (; __k < this->__shape_[__dim]; ++__k) {
          _s32 __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];

          if (__v > __max_value) {
            __max_value = __v;
            __max_index = __k;
          }
        }

        __ret.__data_[__i * __inner_size + __j] = __max_index;
      }
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    for (__i = 0; __i < __outer_size; ++__i) {
      index_type __j = 0;
      for (; __j < __inner_size; ++__j) {
        neon_u32   __max_vec       = vdupq_n_u32(-std::numeric_limits<_u32>::infinity());
        neon_u32   __index_vec     = vdupq_n_u32(0);
        neon_u32   __increment     = vdupq_n_u32(1);
        neon_u32   __current_index = {0, 1, 2, 3};
        index_type __k             = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH) {
          neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(
              &this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          neon_u32 __mask     = vcgtq_u32(__data_vec, __max_vec);
          __max_vec           = vbslq_u32(__mask, __data_vec, __max_vec);
          __index_vec         = vbslq_u32(__mask, __current_index, __index_vec);
          __current_index     = vaddq_u32(__current_index, __increment);
        }

        _u32 __max_values[_ARM64_REG_WIDTH];
        _u32 __indices[_ARM64_REG_WIDTH];

        vst1q_u32(__max_values, __max_vec);
        vst1q_u32(__indices, __index_vec);

        _u32       __max_value = __max_values[0];
        index_type __max_index = __indices[0];

        for (int __k = 1; __k < _ARM64_REG_WIDTH; ++__k) {
          if (__max_values[__k] > __max_value) {
            __max_value = __max_values[__k];
            __max_index = __indices[__k];
          }
        }

        for (; __k < this->__shape_[__dim]; ++__k) {
          _u32 __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];

          if (__v > __max_value) {
            __max_value = __v;
            __max_index = __k;
          }
        }
        __ret.__data_[__i * __inner_size + __j] = __max_index;
      }
    }
  }

  for (__i = 0; __i < __outer_size; ++__i) {
    index_type __j = 0;
    for (; __j < __inner_size; ++__j) {
      index_type __max_index = 0;
      value_type __max_value = this->__data_[__i * this->__shape_[__dim] * __inner_size + __j];
      index_type __k         = 1;
      for (; __k < this->__shape_[__dim]; ++__k) {
        value_type __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];

        if (__v > __max_value) {
          __max_value = __v;
          __max_index = __k;
        }
      }
      __ret.__data_[__i * __inner_size + __j] = __max_index;
    }
  }

  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_argmax(index_type __dim) const {
  if (__dim < 0 || __dim >= this->__shape_.size())
    throw std::out_of_range("Dimension out of range in argmax");

  tensor     __ret;
  shape_type __ret_sh = this->__shape_;

  __ret_sh.erase(__ret_sh.begin() + __dim);
  __ret.__shape_ = __ret_sh;
  __ret.__data_.resize(this->__computeSize(__ret_sh), value_type(0));

  index_type __outer_size = 1;
  index_type __inner_size = 1;
  index_type __i          = 0;
#pragma omp parallel
  for (; __i < __dim; ++__i) __outer_size *= this->__shape_[__i];

#pragma omp parallel
  for (__i = __dim + 1; __i < static_cast<index_type>(this->__shape_.size()); ++__i)
    __inner_size *= this->__shape_[__i];

  if constexpr (std::is_floating_point_v<value_type>) {
    for (__i = 0; __i < __outer_size; ++__i) {
      for (index_type __j = 0; __j < __inner_size; ++__j) {
        neon_f32   __max_vec = vdupq_n_f32(-std::numeric_limits<_f32>::infinity());
        index_type __k       = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH) {
          neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(
              &this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          __max_vec           = vmaxq_f32(__max_vec, __data_vec);
        }

        _f32 __max_value = vmaxvq_f32(__max_vec);
        for (; __k < this->__shape_[__dim]; ++__k) {
          _f32 __v    = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          __max_value = std::max(__max_value, __v);
        }

        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (__i = 0; __i < __outer_size; ++__i) {
      for (index_type __j = 0; __j < __inner_size; ++__j) {
        neon_s32   __max_vec = vdupq_n_s32(-std::numeric_limits<_s32>::infinity());
        index_type __k       = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH) {
          neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(
              &this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          __max_vec           = vmaxq_s32(__max_vec, __data_vec);
        }

        _s32 __max_value = vmaxvq_s32(__max_vec);
        for (; __k < this->__shape_[__dim]; ++__k) {
          _s32 __v    = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          __max_value = std::max(__max_value, __v);
        }

        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    for (__i = 0; __i < __outer_size; ++__i) {
      for (index_type __j = 0; __j < __inner_size; ++__j) {
        neon_u32   __max_vec = vdupq_n_u32(-std::numeric_limits<_u32>::infinity());
        index_type __k       = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH) {
          neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(
              &this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          __max_vec           = vmaxq_u32(__max_vec, __data_vec);
        }

        _u32 __max_value = vmaxvq_u32(__max_vec);
        for (; __k < this->__shape_[__dim]; ++__k) {
          _u32 __v    = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          __max_value = std::max(__max_value, __v);
        }

        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  } else {
    for (__i = 0; __i < __outer_size; ++__i) {
      index_type __j = 0;
      for (; __j < __inner_size; ++__j) {
        value_type __max_value = this->__data_[__i * this->__shape_[__dim] * __inner_size + __j];
        index_type __k         = 1;
        for (; __k < this->__shape_[__dim]; ++__k) {
          value_type __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];

          if (__v > __max_value) __max_value = __v;
        }
        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  }

  return __ret;
}

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

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_slice(index_type __dim, std::optional<index_type> __start,
                                    std::optional<index_type> __end, index_type __step) const {
  if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
    throw std::out_of_range("Dimension out of range.");

  index_type __s       = this->__shape_[__dim];
  index_type __start_i = __start.value_or(0);
  index_type __end_i   = __end.value_or(__s);

  if (__start_i < 0) __start_i += __s;
  if (__end_i < 0) __end_i += __s;

  __start_i               = std::max(index_type(0), std::min(__start_i, __s));
  __end_i                 = std::max(index_type(0), std::min(__end_i, __s));
  index_type __slice_size = (__end_i - __start_i + __step - 1) / __step;
  shape_type __ret_dims   = this->__shape_;
  __ret_dims[-__dim]      = __slice_size;
  tensor __ret(__ret_dims);

  index_type __vector_end =
      __start_i + ((__end_i - __start_i) / _ARM64_REG_WIDTH) * _ARM64_REG_WIDTH;

  if constexpr (std::is_floating_point_v<value_type> && __step == 1) {
    for (index_type __i = __start_i, __j = 0; __i < __vector_end;
         __i += _ARM64_REG_WIDTH, __j += _ARM64_REG_WIDTH) {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      vst1q_f32(&(__ret.__data_[__j]), __vec);
    }

    for (index_type __i = __vector_end, __j = __vector_end - __start_i; __i < __end_i; ++__i, ++__j)
      __ret.__data_[__j] = this->__data_[__i];
  } else if (std::is_signed_v<value_type> && __step == 1) {
    for (index_type __i = __start_i, __j = 0; __i < __vector_end;
         __i += _ARM64_REG_WIDTH, __j += _ARM64_REG_WIDTH) {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      vst1q_s32(&(__ret.__data_[__j]), __vec);
    }
  } else if constexpr (std::is_unsigned_v<value_type> && __step == 1) {
    for (index_type __i = __start_i, __j = 0; __i < __vector_end;
         __i += _ARM64_REG_WIDTH, __j += _ARM64_REG_WIDTH) {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      vst1q_u32(&(__ret.__data_[__j]), __vec);
    }
  }

  index_type __i = __vector_end;
  index_type __j = __vector_end - __start_i;
#pragma omp parallel
  for (; __i < __end_i; ++__i, ++__j) __ret.__data_[__j] = this->__data_[__i];

  __i = __start_i;
  __j = 0;
#pragma omp parallel
  for (; __i < __end_i; __i += __step, ++__j) __ret[__j] = this->__data_[__i];

  return __ret;
}
