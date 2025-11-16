#pragma once

#include "internal/simd/neon/math/trig/sin/sinc.hpp"
#include "tensor.hpp"


template<class _Tp>
arch::tensor<_Tp>& arch::tensor<_Tp>::sinc_()
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::sinc_(*this);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  constexpr value_type pi  = static_cast<value_type>(3.14159265358979323846);
  const value_type     eps = std::numeric_limits<value_type>::epsilon();
  container_type&      a   = this->storage_();

  for (auto& elem : a)
  {
    elem = (std::abs(elem) < eps) ? value_type(1.0) : std::sin(pi * elem) / (pi * elem);
  }

  return *this;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::sinc() const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.sinc_();
  return ret;
}