#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sigmoid_() {
  index_type __i = 0;

  using neon_type =
      typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, void>::type;

  if constexpr (std::is_same_v<value_type, _f32>) {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_type __v         = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_type __exp_neg_v = vexpq_f32(vnegq_f32(__v));  // e^(-x)
      neon_type __sigmoid =
          vrecpeq_f32(vaddq_f32(vdupq_n_f32(1.0f), __exp_neg_v));  // 1 / (1 + e^(-x))

      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __sigmoid);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] =
        static_cast<value_type>(1.0 / (1.0 + std::exp(-static_cast<double>(this->__data_[__i]))));

  return *this;
}
