#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::sum(const index_type __axis) const {
#if defined(__ARM_NEON)
  return this->neon_sum(__axis);
#endif
  if (__axis < 0 || __axis >= static_cast<index_type>(this->__shape_.size()))
    throw __index_error__("Invalid axis for sum");

  shape_type __ret_sh = this->__shape_;
  __ret_sh[__axis]    = 1;
  index_type __ret_size =
      std::accumulate(__ret_sh.begin(), __ret_sh.end(), 1, std::multiplies<index_type>());
  data_t __ret_data(__ret_size, value_type(0.0f));

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

  return __self(__ret_data, __ret_sh);
}