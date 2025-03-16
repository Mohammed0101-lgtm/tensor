#pragma once

#include "tensorbase.hpp"

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::remainder(const value_type __val) const {
  __self __ret = this->clone();
  __ret.remainder_(__val);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::remainder(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.remainder_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::remainder_(const value_type __val) {
  assert(__val != 0 && "Remainder by zero is undefined");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] %= __val;
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::remainder_(const value_type __val) const {
  assert(__val != 0 && "Remainder by zero is undefined");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] %= __val;
  return *this;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::remainder_(const tensor& __other) {
  assert(__other.count_nonzero() == __other.size(0) && "Remainder by zero is undefined");
  assert(__equal_shape(this->shape(), __other.shape()));

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] %= __other[__i];
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::remainder_(const tensor& __other) const {
  assert(__other.count_nonzero() == __other.size(0) && "Remainder by zero is undefined");
  assert(__equal_shape(this->shape(), __other.shape()));

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i) this->__data_[__i] %= __other[__i];
  return *this;
}
