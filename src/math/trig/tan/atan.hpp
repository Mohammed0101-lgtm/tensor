#pragma once

#include "internal/simd/neon/math/trig/tan/atan.hpp"
#include "tensor.hpp"


template<class _Tp>
inline arch::tensor<_Tp>& arch::tensor<_Tp>::atan_()
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::atan_(*this);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::atan(elem);
  }

  return *this;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::atan() const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.atan_();
  return ret;
}