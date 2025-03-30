#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::cos_() {
#if defined(__ARM_NEON)
  return this->neon_cos_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::cos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::cos_() const {
#if defined(__ARM_NEON)
  return this->neon_cos_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::cos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::acos_() {
#if defined(__ARM_NEON)
  return this->neon_acos_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::acos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::acos_() const {
#if defined(__ARM_NEON)
  return this->neon_acos_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::acos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::acos() const {
  __self __ret = this->clone();
  __ret.acos_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::cos() const {
  __self __ret = this->clone();
  __ret.cos_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::cosh() const {
  __self __ret = this->clone();
  __ret.cosh_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::cosh_() {
#if defined(__ARM_NEON)
  return this->neon_cosh_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::cosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::cosh_() const {
#if defined(__ARM_NEON)
  return this->neon_cosh_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::cosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::acosh_() {
#if defined(__ARM_NEON)
  return this->neon_acosh_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::acosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::acosh_() const {
#if defined(__ARM_NEON)
  return this->neon_acosh_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::acosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::acosh() const {
  __self __ret = this->clone();
  __ret.acosh_();
  return __ret;
}