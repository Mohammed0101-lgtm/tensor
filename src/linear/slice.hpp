#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::slice(index_type __dim, std::optional<index_type> __start,
                               std::optional<index_type> __end, int64_t __step) const {
  if (this->empty()) return __self();

  if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
    throw __index_error__("Invalid dimension provided");

  if (__step == 0) throw std::invalid_argument("Step cannot be zero");

  index_type __s         = this->__shape_[__dim];
  index_type __start_val = __start.value_or(__step > 0 ? 0 : __s - 1);
  index_type __end_val   = __end.value_or(__step > 0 ? __s : -1);

  if (__start_val < 0) __start_val += __s;
  if (__end_val < 0) __end_val += __s;

  __start_val = std::clamp(__start_val, index_type(0), __s);
  __end_val   = std::clamp(__end_val, index_type(0), __s);

  if ((__step > 0 && __start_val >= __end_val) || (__step < 0 && __start_val <= __end_val))
    return __self();

  shape_type __ret_sh     = this->__shape_;
  index_type __slice_size = 0;
  data_t     __ret_data;

  for (index_type __idx = __start_val; (__step > 0 ? __idx < __end_val : __idx > __end_val);
       __idx += __step) {
    __ret_data.push_back(this->__data_[__idx]);
    __slice_size++;
  }

  __ret_sh[__dim] = __slice_size;

  return __self(__ret_sh, __ret_data);
}

/*
template <class _Tp>
tensor<_Tp> tensor<_Tp>::slice(index_type __dim, std::optional<index_type> __start,
                               std::optional<index_type> __end, int64_t __step) {
  if (this->empty()) return __self();

  if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
    throw std::invalid_argument("Invalid dimension provided");

  if (__step == 0) throw std::invalid_argument("Step cannot be zero");
}
*/
