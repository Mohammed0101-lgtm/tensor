#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sqrt_() {
#if defined(__ARM_NEON)
  return this->neon_sqrt_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sqrt(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sqrt_() const {
#if defined(__ARM_NEON)
  return this->neon_sqrt_();
#endif

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sqrt(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sqrt() const {
  __self __ret = this->clone();
  __ret.sqrt_();
  return __ret;
}