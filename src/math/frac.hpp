#pragma once

#include "tensor.hpp"


template<class _Tp>
inline arch::tensor<_Tp>& arch::tensor<_Tp>::frac_()
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::frac_(*this);
  }

  if (!std::is_floating_point_v<value_type>)
  {
    throw error::type_error("Type must be floating point");
  }

  const container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = frac(elem);
  }

  return *this;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::frac() const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.frac_();
  return ret;
}