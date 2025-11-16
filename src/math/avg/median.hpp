#pragma once

#include "tensor.hpp"


template<class _Tp>
double arch::tensor<_Tp>::median(const index_type dimension) const
{
  if (!std::is_arithmetic_v<value_type>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  if (dimension >= n_dims() || dimension < -1)
  {
    throw error::index_error("given dimension is out of range of the tensor dimensions");
  }

  index_type stride = (dimension == -1) ? 0 : strides_[dimension];
  index_type end    = (dimension == -1) ? data_.size() : strides_[dimension];

  Container d(data_.begin() + stride, data_.begin() + end);

  if (d.empty())
  {
    return std::numeric_limits<double>::quiet_NaN();
  }

  std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());

  if (d.size() % 2 == 0)
  {
    std::nth_element(d.begin(), d.begin() + d.size() / 2 - 1, d.end());
    return (static_cast<double>(d[d.size() / 2]) + d[d.size() / 2 - 1]) / 2.0;
  }

  return d[d.size() / 2];
}