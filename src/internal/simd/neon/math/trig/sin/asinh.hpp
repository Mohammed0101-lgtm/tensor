#pragma once

#include "internal/simd/neon/alias.hpp"
#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
arch::tensor<_Tp>& asinh_(arch::tensor<_Tp>& t)
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
    neon_type<_Tp>           data_vec = neon_load<_Tp>(&data_[i]);
    alignas(sizeof(_Tp)) _Tp vals[t.simd_width];
    neon_store<_Tp>(vals, data_vec);

    for (int j = 0; j < t.simd_width; ++j)
    {
      vals[j] = static_cast<_Tp>(std::asinh(vals[j]));
    }

    neon_type<_Tp> asinh_vec = neon_load<_Tp>(vals);
    neon_store(&data_[i], asinh_vec);
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = static_cast<_Tp>(std::asinh(data_[i]));
  }

  return t;
}

}