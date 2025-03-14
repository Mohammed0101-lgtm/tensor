#pragma once

#include "tensorbase.hpp"

template <class _Tp>
double tensor<_Tp>::mode(const index_type __dim) const {
  if (__dim >= this->n_dims() || __dim < -1)
    throw std::invalid_argument("given dimension is out of range of the tensor dimensions");

  index_type __stride = (__dim == -1) ? 0 : this->__strides_[__dim];
  index_type __end    = (__dim == -1) ? this->__data_.size() : this->__strides_[__dim];

  if (this->__data_.empty()) return std::numeric_limits<double>::quiet_NaN();

  std::unordered_map<value_type, size_t> __counts;
  for (index_type __i = __stride; __i < __end; ++__i) ++__counts[this->__data_[__i]];

  value_type __ret  = 0;
  size_t     __most = 0;
  for (const std::pair<value_type, size_t>& __pair : __counts) {
    if (__pair.second > __most) {
      __ret  = __pair.first;
      __most = __pair.second;
    }
  }

  return static_cast<double>(__ret);
}

template <class _Tp>
double tensor<_Tp>::median(const index_type __dim) const {
  if (__dim >= this->n_dims() || __dim < -1)
    throw std::invalid_argument("given dimension is out of range of the tensor dimensions");

  index_type __stride = (__dim == -1) ? 0 : this->__strides_[__dim];
  index_type __end    = (__dim == -1) ? this->__data_.size() : this->__strides_[__dim];

  data_t __d(this->__data_.begin() + __stride, this->__data_.begin() + __end);

  if (__d.empty()) return std::numeric_limits<double>::quiet_NaN();

  std::nth_element(__d.begin(), __d.begin() + __d.size() / 2, __d.end());

  if (__d.size() % 2 == 0) {
    std::nth_element(__d.begin(), __d.begin() + __d.size() / 2 - 1, __d.end());
    return (static_cast<double>(__d[__d.size() / 2]) + __d[__d.size() / 2 - 1]) / 2.0;
  }

  return __d[__d.size() / 2];
}

template <class _Tp>
double tensor<_Tp>::mean() const {
#if defined(__ARM_NEON)
  return this->neon_mean();
#endif
  double __m = 0.0;

  if (this->empty()) return __m;

  for (index_type __i = 0; __i < this->__data_.size(); ++__i) __m += this->__data_[__i];

  return static_cast<double>(__m) / static_cast<double>(this->__data_.size());
}
