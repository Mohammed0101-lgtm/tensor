#pragma once

#include "internal/simd/neon/bit/not.hpp"
#include "tensor.hpp"


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_not_() {
  if (internal::types::using_neon())
  {
    return internal::simd::neon::bitwise_not_(*this);
  }

  if constexpr (!std::is_integral_v<value_type>)
  {
    throw error::type_error("Cannot perform a bitwise NOT on non integral values");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = ~elem;
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::bitwise_not() const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.bitwise_not_();
  return ret;
}