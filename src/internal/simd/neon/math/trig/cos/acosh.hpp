#pragma once

#include "internal/simd/neon/alias.hpp"
#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
tensor<_Tp>& acosh_(tensor<_Tp>& t) {
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  std::vector<_Tp>& data_    = t.storage_();
  const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64              i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp>  vec = neon_load<_Tp>(&data_[i]);
    alignas(16) _Tp vals[t.simd_width];
    neon_store<_Tp>(vals, vec);

    for (std::size_t j = 0; j < t.simd_width; j++)
    {
      if (vals[j] < static_cast<_Tp>(1.0))
      {
        throw std::domain_error("Input data is out of domain for acos()");
      }

      vals[j] = static_cast<_Tp>(std::acosh(vals[j]));
    }

    neon_type<_Tp> acosh_vec = neon_load<_Tp>(vals);
    neon_store<_Tp>(&data_[i], acosh_vec);
  }

  for (; i < data_.size(); ++i)
  {
    if (data_[i] < static_cast<_Tp>(1.0))
    {
      throw std::domain_error("Input data is out of domain for acosh()");
    }

    data_[i] = static_cast<_Tp>(std::acosh(data_[i]));
  }

  return t;
}

}