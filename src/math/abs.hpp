#pragma once

#include "../tensorbase.hpp"
#include "../types.hpp"

template<class _Tp>
arch::tensor<_Tp> arch::tensor<_Tp>::abs() const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.abs_();
  return ret;
}

template<class _Tp>
arch::tensor<_Tp>& arch::tensor<_Tp>::abs_()
{
  if (using_neon())
  {
    return internal::simd::neon::abs_(*this);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("abs_() is only available for arithmetic types.");
  }

  if (std::is_unsigned_v<value_type>)
  {
    return *this;
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::abs(elem);
  }

  return *this;
}
