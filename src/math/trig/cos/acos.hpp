#pragma once

#include "internal/simd/neon/math/trig/cos/acos.hpp"
#include "tensor.hpp"


template<class _Tp>
arch::tensor<_Tp>& arch::tensor<_Tp>::acos_()
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::acos_(*this);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    if (elem > _Tp(1.0) || elem < _Tp(-1.0))
    {
      throw std::domain_error("Input data is out of domain for acos()");
    }

    elem = std::acos(elem);
  }

  return *this;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::acos() const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.acos_();
  return ret;
}