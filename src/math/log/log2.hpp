#pragma once

#include "internal/simd/neon/math/log/log2.hpp"
#include "tensor.hpp"


template<class _Tp>
arch::tensor<_Tp>& arch::tensor<_Tp>::log2_()
{
  if (this->empty())
  {
    return *this;
  }

  if (using_neon())
  {
    return internal::simd::neon::log2_(*this);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::log2(elem);
  }

  return *this;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::log2() const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.log2_();
  return ret;
}
