#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::tan_() {
#if defined(__ARM_NEON)
  return this->neon_tan_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::tan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::tan_() const {
#if defined(__ARM_NEON)
  return this->neon_tan_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::tan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::tanh_() {
#if defined(__ARM_NEON)
  return this->neon_tanh_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::tanh(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::tanh_() const {
#if defined(__ARM_NEON)
  return this->neon_tanh_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::tanh(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::atan_() {
#if defined(__ARM_NEON)
  return this->neon_atan_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::atan_() const {
#if defined(__ARM_NEON)
  return this->neon_atan_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::atanh_() {
#if defined(__ARM_NEON)
  return this->neon_atanh_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::atanh_() const {
#if defined(__ARM_NEON)
  return this->neon_atanh_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::atanh() const {
  __self __ret = this->clone();
  __ret.atanh_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::tanh() const {
  __self __ret = this->clone();
  __ret.tanh_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::tan() const {
  __self __ret = this->clone();
  __ret.tan_();
  return __ret;
}