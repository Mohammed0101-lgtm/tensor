#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_slice(index_type __dim, std::optional<index_type> __start,
                                    std::optional<index_type> __end, index_type __step) const {
  if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
    throw __index_error__("Dimension out of range.");

  if (__step == 0) throw std::invalid_argument("Step cannot be zero.");

  index_type __s       = this->__shape_[__dim];
  index_type __start_i = __start.value_or(0);
  index_type __end_i   = __end.value_or(__s);

  if (__start_i < 0) __start_i += __s;
  if (__end_i < 0) __end_i += __s;

  __start_i               = std::max(index_type(0), std::min(__start_i, __s));
  __end_i                 = std::max(index_type(0), std::min(__end_i, __s));
  index_type __slice_size = (__end_i - __start_i + __step - 1) / __step;
  shape_type __ret_dims   = this->__shape_;
  __ret_dims[__dim]       = __slice_size;
  tensor __ret(__ret_dims);

  index_type __vector_end =
      __start_i + ((__end_i - __start_i) / _ARM64_REG_WIDTH) * _ARM64_REG_WIDTH;

  if constexpr (std::is_floating_point_v<value_type> && __step == 1) {
    for (index_type __i = __start_i, __j = 0; __i < __vector_end;
         __i += _ARM64_REG_WIDTH, __j += _ARM64_REG_WIDTH) {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      vst1q_f32(&(__ret.__data_[__j]), __vec);
    }
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

  // Handle remaining elements
  index_type remaining = (__end_i - __start_i) % _ARM64_REG_WIDTH;
  if (remaining > 0) {
    for (index_type __i = __vector_end, __j = __vector_end - __start_i; __i < __end_i;
         ++__i, ++__j) {
      __ret.__data_[__j] = this->__data_[__i];
    }
  }

#pragma omp parallel for
  for (index_type __i = __start_i; __i < __end_i; __i += __step) {
    index_type __j = (__i - __start_i) / __step;
    __ret[__j]     = this->__data_[__i];
  }

  return __ret;
}