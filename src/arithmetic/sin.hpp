#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sin_() {
#if defined(__ARM_NEON)
  return this->neon_sin_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw type_error("Type must be arithmetic");

#pragma omp parallel
  for (index_type i = 0; i < this->data_.size(); ++i)
    this->data_[i] = static_cast<value_type>(std::sin(this->data_[i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sin_() const {
#if defined(__ARM_NEON)
  return this->neon_sin_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw type_error("Type must be arithmetic");

#pragma omp parallel
  for (index_type i = 0; i < this->data_.size(); ++i)
    this->data_[i] = static_cast<value_type>(std::sin(this->data_[i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sin() const {
  self ret = this->clone();
  ret.sin_();
  return ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::asin() const {
  self ret = this->clone();
  ret.asin_();
  return ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::atan() const {
  self ret = this->clone();
  ret.atan_();
  return ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sinc_() {
#if defined(__ARM_NEON)
  return this->neon_sinc_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw type_error("Type must be arithmetic");

#pragma omp parallel
  for (index_type i = 0; i < this->data_.size(); ++i)
    this->data_[i] =
        (std::abs(this->data_[i]) < 1e-6)
            ? static_cast<value_type>(1.0)
            : static_cast<value_type>(std::sin(M_PI * this->data_[i]) / (M_PI * this->data_[i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sinc_() const {
#if defined(__ARM_NEON)
  return this->neon_sinc_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw type_error("Type must be arithmetic");

#pragma omp parallel
  for (index_type i = 0; i < this->data_.size(); ++i)
    this->data_[i] =
        (std::abs(this->data_[i]) < 1e-6)
            ? static_cast<value_type>(1.0)
            : static_cast<value_type>(std::sin(M_PI * this->data_[i]) / (M_PI * this->data_[i]));
  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sinc() const {
  self ret = this->clone();
  ret.sinc_();
  return ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sinh_() {
#if defined(__ARM_NEON)
  return this->neon_sinh_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw type_error("Type must be arithmetic");

#pragma omp parallel
  for (index_type i = 0; i < this->data_.size(); ++i)
    this->data_[i] = static_cast<value_type>(std::sinh(this->data_[i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sinh_() const {
#if defined(__ARM_NEON)
  return this->neon_sinh_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw type_error("Type must be arithmetic");

#pragma omp parallel
  for (index_type i = 0; i < this->data_.size(); ++i)
    this->data_[i] = static_cast<value_type>(std::sinh(this->data_[i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sinh() const {
  self ret = this->clone();
  ret.sinh_();
  return ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::asinh_() {
#if defined(__ARM_NEON)
  return this->neon_sinh_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw type_error("Type must be arithmetic");

#pragma omp parallel
  for (index_type i = 0; i < this->data_.size(); ++i)
    this->data_[i] = static_cast<value_type>(std::asinh(this->data_[i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::asinh_() const {
#if defined(__ARM_NEON)
  return this->neon_sinh_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw type_error("Type must be arithmetic");

#pragma omp parallel
  for (index_type i = 0; i < this->data_.size(); ++i)
    this->data_[i] = static_cast<value_type>(std::asinh(this->data_[i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::asinh() const {
  self ret = this->clone();
  ret.asinh_();
  return ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::asin_() {
#if defined(__ARM_NEON)
  return this->neon_asin_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw type_error("Type must be arithmetic");

#pragma omp parallel
  for (index_type i = 0; i < this->data_.size(); ++i)
    this->data_[i] = static_cast<value_type>(std::asin(this->data_[i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::asin_() const {
#if defined(__ARM_NEON)
  return this->neon_asin_();
#endif
  if (!std::is_arithmetic_v<value_type>) throw type_error("Type must be arithmetic");

#pragma omp parallel
  for (index_type i = 0; i < this->data_.size(); ++i)
    this->data_[i] = static_cast<value_type>(std::asin(this->data_[i]));

  return *this;
}