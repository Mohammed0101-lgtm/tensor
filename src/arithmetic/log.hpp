#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::log_() {
#if defined(__ARM_NEON)
  return this->neon_log_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log_() const {
#if defined(__ARM_NEON)
  return this->neon_log_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::log() const {
  __self __ret = this->clone();
  __ret.log_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::log10_() {
#if defined(__ARM_NEON)
  return this->neon_log10_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log10(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log10_() const {
#if defined(__ARM_NEON)
  return this->neon_log10_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log10(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::log10() const {
  __self __ret = this->clone();
  __ret.log10_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::log2_() {
#if defined(__ARM_NEON)
  return this->neon_log2_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log2(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log2_() const {
#if defined(__ARM_NEON)
  return this->neon_log2_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log2(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::log2() const {
  __self __ret = this->clone();
  __ret.log2_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::log_softmax(const index_type __dim) const {
  __self __ret = this->clone();
  __ret.log_softmax_(__dim);
  return __ret;
}