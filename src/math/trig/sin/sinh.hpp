#pragma once

#include "internal/simd/neon/math/trig/sin/sinh.hpp"
#include "tensor.hpp"


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sinh_() {
  if (this->empty())
  {
    return *this;
  }

  if (internal::types::using_neon())
  {
    return internal::simd::neon::sinh_(*this);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    elem = std::sinh(elem);
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::sinh() const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.sinh_();
  return ret;
}