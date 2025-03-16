#pragma once

#include "tensorbase.hpp"

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::logical_not_() {
  this->bitwise_not_();
  this->bool_();
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::logical_not_() const {
  this->bitwise_not_();
  this->bool_();
  return *this;
}

template <class _Tp>
tensor<bool> tensor<_Tp>::logical_not() const {
  tensor<bool> __ret = this->bool_();
  __ret.logical_not_();
  return __ret;
}
