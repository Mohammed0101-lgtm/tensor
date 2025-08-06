#pragma once

#include "tensor.hpp"


template<class _Tp>
double tensor<_Tp>::mode(const index_type dimension) const {
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

  if (data_.empty())
  {
    return std::numeric_limits<double>::quiet_NaN();
  }

  std::unordered_map<value_type, std::size_t> counts;

  for (index_type i = stride; i < end; ++i)
  {
    ++counts[data_[i]];
  }

  value_type  ret  = 0;
  std::size_t most = 0;

  for (const auto& pair : counts)
  {
    if (pair.second > most)
    {
      ret  = pair.first;
      most = pair.second;
    }
  }

  return static_cast<double>(ret);
}
