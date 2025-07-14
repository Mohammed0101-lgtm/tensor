#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& internal::neon::sigmoid_(tensor<_Tp>& t) {
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  std::vector<_Tp>& data_ = t.storage_();
  using neon_type         = typename std::conditional<std::is_same_v<_Tp, _f32>, neon_f32, void>::type;
  const _u64 simd_end     = data_.size() - (data_.size() % t.simd_width);
  _u64       i            = 0;

  if constexpr (std::is_same_v<_Tp, _f32>)
  {
    for (; i < simd_end; i += t.simd_width)
    {
      neon_type v         = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
      neon_type exp_neg_v = vexpq_f32(vnegq_f32(v));                               // e^(-x)
      neon_type sigmoid   = vrecpeq_f32(vaddq_f32(vdupq_n_f32(1.0f), exp_neg_v));  // 1 / (1 + e^(-x))
      vst1q_f32(reinterpret_cast<_f32*>(&data_[i]), sigmoid);
    }
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = static_cast<_Tp>(1.0 / (1.0 + std::exp(-static_cast<double>(data_[i]))));
  }

  return t;
}
