#pragma once

#include "tensorbase.hpp"

// used as a helper function
inline int64_t __lcm(const int64_t __a, const int64_t __b) {
  return (__a * __b) / std::gcd(__a, __b);
}

template <class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::lcm() const {
  index_type __ret = static_cast<index_type>(this->__data_[0]);

  for (index_type __i = 1; __i < this->__data_.size(); ++__i)
    __ret = __lcm(static_cast<index_type>(this->__data_[__i]), __ret);

  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::lcm(const tensor& __other) const {
  assert(__equal_shape(this->shape(), __other.shape()));
  tensor __ret = this->clone();

#pragma omp parallel
  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
    __ret[__i] = static_cast<value_type>(this->__lcm(static_cast<index_type>(this->__data_[__i]),
                                                     static_cast<index_type>(__other[__i])));

  return __ret;
}
