#pragma once

#include "internal/simd/neon/bit/lsh.hpp"
#include "tensor.hpp"


template<class _Tp>
inline tensor<_Tp>& tensor<_Tp>::bitwise_left_shift_(const int amount) {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::bitwise_left_shift_(*this, amount);
  }

  if constexpr (!std::is_integral_v<value_type>)
  {
    throw error::type_error("Type must be integral");
  }

  if constexpr (!internal::types::has_left_shift_operator_v<value_type>)
  {
    return error::operator_error("Type must have left shift operator");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = elem << amount;
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_left_shift(const int amount) const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.bitwise_left_shift_(amount);
  return ret;
}