#pragma once

#include "tensor.hpp"


// used as a helper function
inline int64_t __lcm(const int64_t a, const int64_t b) { return (a * b) / std::gcd(a, b); }

template<class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::lcm() const {
  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  index_type            ret = static_cast<index_type>((*this)[0]);
  const container_type& a   = this->storage_();

  for (const auto& elem : a)
  {
    ret = __lcm(elem, ret);
  }

  return ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::lcm(const tensor& other) const {
  if (this->empty())
  {
    return self({0});
  }

  if (!this->shape().equal(other.shape()))
  {
    throw error::shape_error("Tensors shapes must be equal");
  }

  tensor                ret = clone();
  const container_type& a   = this->storage_();
  index_type            i   = 0;

  for (const auto& elem : a)
  {
    ret[i++] = __lcm(elem, other[i]);
  }

  return ret;
}
