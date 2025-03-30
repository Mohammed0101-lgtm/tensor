#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sin_() {
#if defined(__ARM_NEON)
  return this->neon_sin_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sin(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sin_() const {
#if defined(__ARM_NEON)
  return this->neon_sin_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sin(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sin() const {
  __self __ret = this->clone();
  __ret.sin_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::asin() const {
  __self __ret = this->clone();
  __ret.asin_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::atan() const {
  __self __ret = this->clone();
  __ret.atan_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sinc_() {
#if defined(__ARM_NEON)
  return this->neon_sinc_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (std::abs(this->__data_[__i]) < 1e-6)
                             ? static_cast<value_type>(1.0)
                             : static_cast<value_type>(std::sin(M_PI * this->__data_[__i]) /
                                                       (M_PI * this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sinc_() const {
#if defined(__ARM_NEON)
  return this->neon_sinc_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (std::abs(this->__data_[__i]) < 1e-6)
                             ? static_cast<value_type>(1.0)
                             : static_cast<value_type>(std::sin(M_PI * this->__data_[__i]) /
                                                       (M_PI * this->__data_[__i]));
  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sinc() const {
  __self __ret = this->clone();
  __ret.sinc_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sinh_() {
#if defined(__ARM_NEON)
  return this->neon_sinh_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sinh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sinh_() const {
#if defined(__ARM_NEON)
  return this->neon_sinh_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sinh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sinh() const {
  __self __ret = this->clone();
  __ret.sinh_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::asinh_() {
#if defined(__ARM_NEON)
  return this->neon_sinh_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::asinh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::asinh_() const {
#if defined(__ARM_NEON)
  return this->neon_sinh_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::asinh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::asinh() const {
  __self __ret = this->clone();
  __ret.asinh_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::asin_() {
#if defined(__ARM_NEON)
  return this->neon_asin_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::asin(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::asin_() const {
#if defined(__ARM_NEON)
  return this->neon_asin_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::asin(this->__data_[__i]));

  return *this;
}