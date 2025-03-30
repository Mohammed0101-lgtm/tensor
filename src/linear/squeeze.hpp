#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::unsqueeze(index_type __dim) const {
  if (__dim < 0 || __dim > static_cast<index_type>(this->__shape_.size()))
    throw __index_error__("Dimension out of range in unsqueeze");

  shape_type __s = this->__shape_;
  __s.insert(__s.begin() + __dim, 1);

  tensor __ret;
  __ret.__shape_ = __s;
  __ret.__data_  = this->__data_;

  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::unsqueeze_(index_type __dim) {
  if (__dim < 0 || __dim > static_cast<index_type>(this->__shape_.size()))
    throw __index_error__("Dimension out of range in unsqueeze");

  this->__shape_.insert(this->__shape_.begin() + __dim, 1);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::unsqueeze_(index_type __dim) const {
  if (__dim < 0 || __dim > static_cast<index_type>(this->__shape_.size()))
    throw __index_error__("Dimension out of range in unsqueeze");

  this->__shape_.insert(this->__shape_.begin() + __dim, 1);

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::squeeze(index_type __dim) const {
  __self __ret = this->clone();
  __ret.squeeze_(__dim);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::squeeze_(index_type __dim) {
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::squeeze_(index_type __dim) const {
  return this->squeeze_(__dim);
}
