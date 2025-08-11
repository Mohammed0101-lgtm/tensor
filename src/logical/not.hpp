#pragma once

#include "internal/simd/neon/logical/not.hpp"
#include "tensor.hpp"


template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::logical_not_() {
  if (this->empty())
  {
    return *this;
  }

  bitwise_not_();
  bool_();
  return *this;
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_not() const {
  if (this->empty())
  {
    return tensor<bool>({0});
  }

  tensor<bool> ret = bool_();
  ret.logical_not_();
  return ret;
}
