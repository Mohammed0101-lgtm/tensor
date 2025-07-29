#pragma once

#include "internal/simd/neon/alias.hpp"
#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
tensor<_Tp>& bitwise_not_(tensor<_Tp>& t) {
  if (!std::is_integral_v<_Tp> && !std::is_same_v<_Tp, bool>)
  {
    throw error::type_error("Cannot perform a bitwise not on non-integral value");
  }

  std::vector<_Tp>& data_    = t.storage_();
  const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64              i        = 0;

  if constexpr (std::is_signed_v<_Tp>)
  {
    for (; i < simd_end; i += _ARM64_REG_WIDTH)
    {
      neon_s32 data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
      neon_s32 result_vec = vmvnq_s32(data_vec);
      vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), result_vec);
    }
  }
  else if constexpr (std::is_unsigned_v<_Tp>)
  {
    for (; i < simd_end; i += _ARM64_REG_WIDTH)
    {
      neon_u32 data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
      neon_u32 result_vec = vmvnq_u32(data_vec);
      vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), result_vec);
    }
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = ~data_[i];
  }

  return t;
}

}