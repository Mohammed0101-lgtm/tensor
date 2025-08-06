#pragma once

#include "internal/simd/neon/math/trig/cos/acosh.hpp"
#include "tensor.hpp"


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::acosh_() {
  if (this->empty())
  {
    return *this;
  }

  if (internal::types::using_neon())
  {
    return internal::simd::neon::acosh_(*this);
  }

  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  container_type& a = this->storage_();

  for (auto& elem : a)
  {
    if (elem < 1.0)
    {
      throw std::domain_error("Input data is out of domain of acosh()");
    }

    elem = std::acosh(elem);
  }

  return *this;
}

template<class _Tp>
inline tensor<_Tp> tensor<_Tp>::acosh() const {
  if (this->empty())
  {
    return self({0});
  }

  self ret = clone();
  ret.acosh_();
  return ret;
}