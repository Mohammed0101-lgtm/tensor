#pragma once

#include "tensorbase.hpp"

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

  for (; __i < __dim; ++__i) __outer_size *= this->__shape_[__i];
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

  for (; __i < __dim; ++__i) __outer_size *= this->__shape_[__i];
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