#pragma once

#include "internal/simd/neon/math/trig/tan/atanh.hpp"
#include "tensor.hpp"


template<class _Tp>
arch::tensor<_Tp>& arch::tensor<_Tp>::atanh_()
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::atanh_(*this);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    if (elem < -1 || elem > 1)
    {
      throw std::domain_error("Input data is out of domain for atanh()");
    }

    elem = std::atanh(elem);
  }

  return *this;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::atanh() const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.atanh_();
  return ret;
}
