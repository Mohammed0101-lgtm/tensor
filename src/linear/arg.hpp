#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argmax_(index_type __dim) const {
#if defined(__ARM_NEON)
  return this->neon_argmax_(__dim);
#endif
  if (__dim < 0 || __dim >= this->__shape_.size())
    throw __index_error__("Dimension out of range in argmax");

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

  {
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
  }

  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::argmax(index_type __dim) const {
#if defined(__ARM_NEON)
  return this->neon_argmax(__dim);
#endif
  if (__dim < 0 || __dim >= this->__shape_.size())
    throw __index_error__("Dimension out of range in argmax");

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

  {
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
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argsort(index_type __d,
                                                              bool       __ascending) const {
#if defined(__ARM_NEON)
  return this->neon_argsort(__d, __ascending);
#endif
  index_type __adjusted = (__d < 0) ? __d + this->__data_.size() : __d;

  if (__adjusted != 0)
    throw __index_error__("Invalid dimension for argsort: only 1D tensors are supported");

  index_type __size = static_cast<index_type>(this->__data_.size());
  shape_type __indices(__size);
  std::iota(__indices.begin(), __indices.end(), 0);
  std::sort(__indices.begin(), __indices.end(), [&](index_type __a, index_type __b) {
    return __ascending ? this->__data_[__a] < this->__data_[__b]
                       : this->__data_[__a] > this->__data_[__b];
  });

  return tensor<index_type>(__indices);
}