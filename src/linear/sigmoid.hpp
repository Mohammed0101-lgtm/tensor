#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::sigmoid() const {
  __self __ret = this->clone();
  __ret.sigmoid_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sigmoid_() {
#if defined(__ARM_NEON)
  return this->neon_sigmoid_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] =
        static_cast<value_type>(1.0 / (1.0 + std::exp(-static_cast<double>(this->__data_[__i]))));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sigmoid_() const {
#if defined(__ARM_NEON)
  return this->neon_sigmoid_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] =
        static_cast<value_type>(1.0 / (1.0 + std::exp(-static_cast<double>(this->__data_[__i]))));

  return *this;
}
