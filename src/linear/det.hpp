#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::det() const {
  if (this->__shape_.size() < 2) throw std::logic_error("det: tensor must be at least 2D");

  index_type __h, __w;
  if (this->__shape_.size() == 2) {
    __h = this->__shape_[0];
    __w = this->__shape_[1];
  } else {
    index_type __last        = this->__shape_.size() - 1;
    index_type __second_last = this->__shape_.size() - 2;

    if (this->__shape_[__last] == 1) {
      __h = this->__shape_[__second_last - 1];
      __w = this->__shape_[__second_last];
    } else if (this->__shape_[__second_last] == 1) {
      __h = this->__shape_[__last - 1];
      __w = this->__shape_[__last];
    } else {
      __h = this->__shape_[__second_last];
      __w = this->__shape_[__last];
    }
  }

  if (__h != __w) throw std::invalid_argument("det: tensor must be a square matrix (n x n)");

  index_type __n = __h;

  if (__n == 2)
    return tensor<_Tp>(this->operator()(0, 0) * this->operator()(1, 1) -
                       this->operator()(0, 1) * this->operator()(1, 0));

  value_type __determinant = 0;

  for (index_type __col = 0; __col < __n; ++__col) {
    tensor<_Tp> __minor = this->get_minor(0, __col);
    value_type  __sign  = (__col % 2 == 0) ? 1 : -1;
    __determinant += __sign * this->operator()(0, __col) * __minor.det();
  }

  return tensor<_Tp>(__determinant);
}