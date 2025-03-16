#pragma once

#include "tensorbase.hpp"

template <class _Tp>
double tensor<_Tp>::neon_mean() const {
  double __m = 0.0;

  if (this->empty()) return __m;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  neon_s32         __sum_vec  = vdupq_n_s32(0);
  index_type       __i        = 0;
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
    neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
    __sum_vec           = vaddq_s32(__sum_vec, __data_vec);
  }

  _s32 __partial_sum[4];
  vst1q_s32(__partial_sum, __sum_vec);
  __m += __partial_sum[0] + __partial_sum[1] + __partial_sum[2] + __partial_sum[3];

  for (; __i < this->__data_.size(); ++__i) __m += this->__data_[__i];

  return static_cast<double>(__m) / static_cast<double>(this->__data_.size());
}
