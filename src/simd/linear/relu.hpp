#pragma once

#include "../alias.hpp"
#include "clamp.hpp"
#include "tensor.hpp"


template<class _Tp>
tensor<_Tp>& internal::neon::relu_(tensor<_Tp>& t) {
  return internal::neon::clamp_min_(_Tp(0));
}

/*
template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_clipped_relu_(const _Tp clip_limit) {
  if constexpr (std::is_unsigned_v<_Tp>) return *this;

  _u64 s = data_.size();
  _u64 i = 0;

  if constexpr (std::is_same_v<_Tp, _f32>) {
    const neon_f32 vZero = vdupq_n_f32(0.0f);
    const neon_f32 vClip = vdupq_n_f32(clip_limit);

    for (; i + _ARM64_REG_WIDTH <= s; i += _ARM64_REG_WIDTH) {
      neon_f32 v = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
      v          = vminq_f32(vmaxq_f32(v, vZero), vClip);

      vst1q_f32(&data_[i], v);
    }
  } else if constexpr (std::is_same_v<_Tp, _s32>) {
    const neon_s32 vZero = vdupq_n_s32(0);
    const neon_s32 vClip = vdupq_n_s32(clip_limit);

    for (; i + _ARM64_REG_WIDTH <= s; i += _ARM64_REG_WIDTH) {
      neon_s32 v = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
      v          = vminq_s32(vmaxq_s32(v, vZero), vClip);

      vst1q_s32(&data_[i], v);
    }
  }

  for (; i < s; ++i)
    data_[i] = std::min(std::max(data_[i], _Tp(0)), clip_limit);

  return *this;
}
*/

template<class _Tp>
tensor<_Tp>& internal::neon::clipped_relu_(tensor<_Tp>& t, const _Tp clip_limit) {
  if constexpr (std::is_unsigned_v<_Tp>)
  {
    return t;
  }

  t = internal::neon::clamp_(t, _Tp(0), clip_limit);
  return t;
}