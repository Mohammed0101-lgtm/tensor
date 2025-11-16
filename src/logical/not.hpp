#pragma once

#include "internal/simd/neon/logical/not.hpp"
#include "tensor.hpp"


template<class _Tp>
inline arch::tensor<_Tp>& arch::tensor<_Tp>::logical_not_()
{
  if (this->empty())
  {
    return *this;
  }

  bitwise_not_();
  bool_();
  return *this;
}

template<class _Tp>
arch::tensor<bool> arch::tensor<_Tp>::logical_not() const
{
  if (this->empty())
  {
    return arch::tensor<bool>({0});
  }

  arch::tensor<bool> ret = bool_();
  ret.logical_not_();
  return ret;
}
