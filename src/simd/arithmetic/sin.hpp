#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sin_() {
  index_type __i = 0;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::sin(__vals[0]));
      __vals[1] = static_cast<_f32>(std::sin(__vals[1]));
      __vals[2] = static_cast<_f32>(std::sin(__vals[2]));
      __vals[3] = static_cast<_f32>(std::sin(__vals[3]));

      neon_f32 __sin_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __sin_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::sin(__vals[0]));
      __vals[1] = static_cast<_s32>(std::sin(__vals[1]));
      __vals[2] = static_cast<_s32>(std::sin(__vals[2]));
      __vals[3] = static_cast<_s32>(std::sin(__vals[3]));

      neon_s32 __sin_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __sin_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::sin(__vals[0]));
      __vals[1] = static_cast<_u32>(std::sin(__vals[1]));
      __vals[2] = static_cast<_u32>(std::sin(__vals[2]));
      __vals[3] = static_cast<_u32>(std::sin(__vals[3]));

      neon_u32 __sin_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __sin_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sin(this->__data_[__i]));

  return *this;
}

inline neon_f32 vsinq_f32(neon_f32 __x) {
  return {sinf(vgetq_lane_f32(__x, 0)), sinf(vgetq_lane_f32(__x, 1)), sinf(vgetq_lane_f32(__x, 2)),
          sinf(vgetq_lane_f32(__x, 3))};
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sinc_() {
  index_type __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __v    = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __pi_v = vmulq_f32(__v, vdupq_n_f32(M_PI));  // pi * x
      neon_f32 __sinc_v =
          vbslq_f32(vcgeq_f32(vabsq_f32(__v), vdupq_n_f32(1e-6f)),  // Check |x| > epsilon
                    vdivq_f32(vsinq_f32(__pi_v), __pi_v),  // sinc(x) = sin(pi * x) / (pi * x)
                    vdupq_n_f32(1.0f));                    // sinc(0) = 1

      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __sinc_v);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (std::abs(this->__data_[__i]) < 1e-6)
                             ? static_cast<value_type>(1.0)
                             : static_cast<value_type>(std::sin(M_PI * this->__data_[__i]) /
                                                       (M_PI * this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sinh_() {
  index_type __i = 0;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::sinh(__vals[0]));
      __vals[1] = static_cast<_f32>(std::sinh(__vals[1]));
      __vals[2] = static_cast<_f32>(std::sinh(__vals[2]));
      __vals[3] = static_cast<_f32>(std::sinh(__vals[3]));

      neon_f32 __sinh_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __sinh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::sinh(__vals[0]));
      __vals[1] = static_cast<_s32>(std::sinh(__vals[1]));
      __vals[2] = static_cast<_s32>(std::sinh(__vals[2]));
      __vals[3] = static_cast<_s32>(std::sinh(__vals[3]));

      neon_s32 __sinh_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __sinh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::sinh(__vals[0]));
      __vals[1] = static_cast<_u32>(std::sinh(__vals[1]));
      __vals[2] = static_cast<_u32>(std::sinh(__vals[2]));
      __vals[3] = static_cast<_u32>(std::sinh(__vals[3]));

      neon_u32 __sinh_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __sinh_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sinh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_asinh_() {
  index_type __i        = 0;
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::asinh(__vals[0]));
      __vals[1] = static_cast<_f32>(std::asinh(__vals[1]));
      __vals[2] = static_cast<_f32>(std::asinh(__vals[2]));
      __vals[3] = static_cast<_f32>(std::asinh(__vals[3]));

      neon_f32 __asinh_vec = vld1q_f32(__vals);
      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __asinh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::asinh(__vals[0]));
      __vals[1] = static_cast<_s32>(std::asinh(__vals[1]));
      __vals[2] = static_cast<_s32>(std::asinh(__vals[2]));
      __vals[3] = static_cast<_s32>(std::asinh(__vals[3]));

      neon_s32 __asinh_vec = vld1q_s32(__vals);
      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __asinh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::asinh(__vals[0]));
      __vals[1] = static_cast<_u32>(std::asinh(__vals[1]));
      __vals[2] = static_cast<_u32>(std::asinh(__vals[2]));
      __vals[3] = static_cast<_u32>(std::asinh(__vals[3]));

      neon_u32 __asinh_vec = vld1q_u32(__vals);
      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __asinh_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::asinh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_asin_() {
  index_type       __i        = 0;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::asin(__vals[0]));
      __vals[1] = static_cast<_f32>(std::asin(__vals[1]));
      __vals[2] = static_cast<_f32>(std::asin(__vals[2]));
      __vals[3] = static_cast<_f32>(std::asin(__vals[3]));

      neon_f32 __asin_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __asin_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::asin(__vals[0]));
      __vals[1] = static_cast<_s32>(std::asin(__vals[1]));
      __vals[2] = static_cast<_s32>(std::asin(__vals[2]));
      __vals[3] = static_cast<_s32>(std::asin(__vals[3]));

      neon_s32 __asin_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __asin_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::asin(__vals[0]));
      __vals[1] = static_cast<_u32>(std::asin(__vals[1]));
      __vals[2] = static_cast<_u32>(std::asin(__vals[2]));
      __vals[3] = static_cast<_u32>(std::asin(__vals[3]));

      neon_u32 __asin_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __asin_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::asin(this->__data_[__i]));

  return *this;
}