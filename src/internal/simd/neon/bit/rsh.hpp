#pragma once

#include "internal/simd/neon/alias.hpp"
#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
tensor<_Tp>& bitwise_right_shift_(tensor<_Tp>& t, const int amount) {
  if (!std::is_integral_v<_Tp>)
  {
    throw error::type_error("Type must be integral");
  }

  std::vector<_Tp>& data_    = t.storage_();
  const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64              i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> data_vec    = neon_load<_Tp>(&data_[i]);
    neon_type<_Tp> shifted_vec = neon_shr<_Tp>(data_vec, amount);  // Use correct NEON right shift
    neon_store<_Tp>(&data_[i], shifted_vec);
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = data_[i] >> amount;
  }

  return t;
}

}