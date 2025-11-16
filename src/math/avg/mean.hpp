#pragma once

#include "tensor.hpp"


template<class _Tp>
double arch::tensor<_Tp>::mean() const
{
  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  double m = 0.0;

  if (this->empty())
  {
    return m;
  }

  const Container& a = this->storage_();

  for (const auto& elem : a)
  {
    m += elem;
  }

  return static_cast<double>(m) / static_cast<double>(data_.size());
}
