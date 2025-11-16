#pragma once

#include "concepts.hpp"
#include "internal/simd/neon/bit/rsh.hpp"
#include "tensor.hpp"


template<class _Tp>
inline arch::tensor<_Tp>& arch::tensor<_Tp>::bitwise_right_shift_(const int amount)
{
  if (using_neon())
  {
    return internal::simd::neon::bitwise_right_shift_(*this, amount);
  }

  if constexpr (!std::is_integral_v<value_type>)
  {
    throw error::type_error("Type must be integral");
  }

  if constexpr (!internal::concepts::has_right_shift_operator_v<value_type>)
  {
    return error::operator_error("Type must have right shift operator");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem >> amount;
  }

  return *this;
}

template<class _Tp>
inline arch::tensor<_Tp> arch::tensor<_Tp>::bitwise_right_shift(const int amount) const
{
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.bitwise_right_shift_(amount);
  return ret;
}