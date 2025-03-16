#pragma once

#include "tensorbase.hpp"

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
