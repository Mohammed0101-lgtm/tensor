#pragma once

#include "internal/simd/neon/alias.hpp"
#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
tensor<_Tp>& log2_(tensor<_Tp>& t) {
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  std::vector<_Tp>& a        = t.storage_();
  std::size_t       size     = a.size();
  const _u64        simd_end = size - (size % t.simd_width);
  _Tp* __restrict a_ptr      = a.data();

  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp>  vec = neon_load<_Tp>(a_ptr + i);
    alignas(16) _Tp vals[t.simd_width];
    neon_store<_Tp>(vals, vec);

    for (int j = 0; j < t.simd_width; i++)
    {
      vals[0] = static_cast<_Tp>(std::log2(vals[j]));
    }

    neon_type<_Tp> log_vec = neon_load<_Tp>(vals);
    neon_store<_Tp>(a_ptr + i, log_vec);
  }

  for (std::size_t i = simd_end; i < size; ++i)
  {
    a_ptr[i] = static_cast<_Tp>(std::log2(a_ptr[i]));
  }

  return t;
}

}