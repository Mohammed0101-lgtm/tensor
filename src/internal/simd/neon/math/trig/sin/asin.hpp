#pragma once

#include "internal/simd/neon/alias.hpp"
#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
arch::tensor<_Tp>& asin_(arch::tensor<_Tp>& t)
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
    neon_type<_Tp>  data_vec = neon_load<_Tp>(&data_[i]);
    alignas(16) _Tp vals[t.simd_width];
    neon_store<_Tp>(vals, data_vec);

    for (int j = 0; j < t.simd_width; ++j)
    {
      if (vals[j] < static_cast<_Tp>(-1.0) || vals[j] > static_cast<_Tp>(1.0))
      {
        throw std::domain_error("Input value is out of domain for asin()");
      }

      vals[j] = static_cast<_Tp>(std::asin(vals[j]));
    }

    neon_type<_Tp> asin_vec = neon_load<_Tp>(vals);
    neon_store(&data_[i], asin_vec);
  }

  for (; i < data_.size(); ++i)
  {
    if (data_[i] < static_cast<_Tp>(-1.0) || data_[i] > static_cast<_Tp>(1.0))
    {
      throw std::domain_error("Input value is out of domain for asin()");
    }

    data_[i] = static_cast<_Tp>(std::asin(data_[i]));
  }

  return t;
}

}