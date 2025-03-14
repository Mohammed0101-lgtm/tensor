#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::abs() const {
  __self __ret = this->clone();
  __ret.abs_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::abs_() {
#if defined(__ARM_NEON)
  return this->neon_abs_();
#endif
  if (std::is_unsigned_v<value_type>) return *this;

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::abs(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::abs_() const {
#if defined(__ARM_NEON)
  return this->neon_abs_();
#endif
  if (std::is_unsigned_v<value_type>) return *this;

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::abs(this->__data_[__i]));

  return *this;
}
