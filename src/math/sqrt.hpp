#pragma once

#include "tensor.hpp"

template<class _Tp>
inline arch::tensor<_Tp>& arch::tensor<_Tp>::sqrt_()
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::sqrt_(*this);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  const container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::sqrt(elem);
  }

  return *this;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::sqrt() const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.sqrt_();
  return ret;
}