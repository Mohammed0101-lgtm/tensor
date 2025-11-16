#pragma once

#include "internal/simd/neon/alias.hpp"
#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
arch::tensor<_Tp>& atanh_(arch::tensor<_Tp>& t)
{
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  typename arch::tensor<_Tp>::container_type& data_    = t.storage_();
  const _u64                                  simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64                                        i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp>                      data_vec = neon_load<_Tp>(&data_[i]);
    alignas(sizeof(neon_type<_Tp>)) _Tp vals[t.simd_width];
    neon_store<_Tp>(vals, data_vec);

    for (int j = 0; j < t.simd_width; ++j)
    {
      if (vals[j] <= static_cast<_Tp>(-1.0) || vals[j] >= static_cast<_Tp>(1.0))
      {
        throw std::domain_error("Input value is out of domain for atanh()");
      }

      vals[j] = static_cast<_Tp>(std::atanh(vals[j]));
    }

    neon_type<_Tp> atanh_vec = neon_load<_Tp>(vals);
    neon_store<_Tp>(&data_[i], atanh_vec);
  }

  for (; i < data_.size(); ++i)
  {
    if (data_[i] <= static_cast<_Tp>(-1.0) || data_[i] >= static_cast<_Tp>(1.0))
    {
      throw std::domain_error("Input value is out of domain for atanh()");
    }

    data_[i] = static_cast<_Tp>(std::atanh(data_[i]));
  }

  return t;
}

}