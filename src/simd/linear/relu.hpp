#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_relu_() {
    return neon_clamp_min_(value_type(0));
}
/*
template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_clipped_relu_(const value_type clip_limit) {
  if constexpr (std::is_unsigned_v<value_type>) return *this;

  index_type s = data_.size();
  index_type i = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    const neon_f32 vZero = vdupq_n_f32(0.0f);
    const neon_f32 vClip = vdupq_n_f32(clip_limit);

    for (; i + _ARM64_REG_WIDTH <= s; i += _ARM64_REG_WIDTH) {
      neon_f32 v = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
      v          = vminq_f32(vmaxq_f32(v, vZero), vClip);

      vst1q_f32(&data_[i], v);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    const neon_s32 vZero = vdupq_n_s32(0);
    const neon_s32 vClip = vdupq_n_s32(clip_limit);

    for (; i + _ARM64_REG_WIDTH <= s; i += _ARM64_REG_WIDTH) {
      neon_s32 v = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
      v          = vminq_s32(vmaxq_s32(v, vZero), vClip);

      vst1q_s32(&data_[i], v);
    }
  }

  for (; i < s; ++i)
    data_[i] = std::min(std::max(data_[i], value_type(0)), clip_limit);

  return *this;
}
*/
template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_clipped_relu_(const value_type clip_limit) {
    if constexpr (std::is_unsigned_v<value_type>)
    {
        return *this;
    }

    neon_clamp_(value_type(0), std::numeric_limits<value_type>::max());
    neon_clamp_(std::numeric_limits<value_type>::lowest(), clip_limit);

    return *this;
}