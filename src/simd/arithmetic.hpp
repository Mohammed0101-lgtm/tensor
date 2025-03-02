#pragma once

#include "../tensorbase.hpp"

#define __builtin_neon_vgetq_lane_f32
#define __builtin_neon_vsetq_lane_f32

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmax_(const value_type __v) {
  index_type __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    neon_f32         __scalar_val = vdupq_n_f32(__v);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a       = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __max_val = vmaxq_f32(__a, __scalar_val);

      vst1q_f32(&this->__data_[__i], __max_val);
    }
  }

  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = std::fmax(this->__data_[__i], __v);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmax_(const tensor& __other) {
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a       = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b       = vld1q_f32(reinterpret_cast<const _f32*>(&(__other[__i])));
      neon_f32 __max_val = vmaxq_f32(__a, __b);

      vst1q_f32(&this->__data_[__i], __max_val);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::fmax(this->__data_[__i], __other[__i]);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmod_(const value_type __val) {
  assert(std::is_floating_point_v<value_type> &&
         "fmod : template class must be a floating point type");
  index_type __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() - _ARM64_REG_WIDTH);
    neon_f32         __b        = vdupq_n_f32(reinterpret_cast<_f32>(__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a         = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __div       = vdivq_f32(__a, __b);
      neon_f32 __floor_div = vrndq_f32(__div);
      neon_f32 __mult      = vmulq_f32(__floor_div, __b);
      neon_f32 __mod       = vsubq_f32(__a, __mult);

      vst1q_f32(&this->__data_[__i], __mod);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__val)));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_fmod_(const tensor& __other) {
  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");

  index_type __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a         = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b         = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __div       = vdivq_f32(__a, __b);
      neon_f32 __floor_div = vrndq_f32(__div);
      neon_f32 __mult      = vmulq_f32(__floor_div, __b);
      neon_f32 __mod       = vsubq_f32(__a, __mult);

      vst1q_f32(&this->__data_[__i], __mod);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_frac_() {
  index_type __i = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = this->__frac(__vals[0]);
      __vals[1] = this->__frac(__vals[1]);
      __vals[2] = this->__frac(__vals[2]);
      __vals[3] = this->__frac(__vals[3]);

      neon_f32 __atan_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __atan_vec);
    }
  }
  if constexpr (std::is_same_v<value_type, _f64>) {
    index_type __simd_end = this->__data_.size() - (this->__data_.size() % (_ARM64_REG_WIDTH / 2));

    for (; __i < __simd_end; __i += (_ARM64_REG_WIDTH / 2)) {
      neon_f64 __data_vec = vld1q_f64(reinterpret_cast<const _f64*>(&this->__data_[__i]));
      _f64     __vals[_ARM64_REG_WIDTH];
      vst1q_f64(__vals, __data_vec);

      __vals[0] = static_cast<_f64>(this->__frac(__vals[0]));
      __vals[1] = static_cast<_f64>(this->__frac(__vals[1]));
      __vals[2] = static_cast<_f64>(this->__frac(__vals[2]));
      __vals[3] = static_cast<_f64>(this->__frac(__vals[3]));

      neon_f64 __atan_vec = vld1q_f64(__vals);
      vst1q_f64(&this->__data_[__i], __atan_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_log_() {
  index_type __i = 0;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::log(static_cast<_f32>(__vals[0])));
      __vals[1] = static_cast<_f32>(std::log(static_cast<_f32>(__vals[1])));
      __vals[2] = static_cast<_f32>(std::log(static_cast<_f32>(__vals[2])));
      __vals[3] = static_cast<_f32>(std::log(static_cast<_f32>(__vals[3])));

      neon_f32 __log_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __log_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::log(static_cast<_u32>(__vals[0])));
      __vals[1] = static_cast<_u32>(std::log(static_cast<_u32>(__vals[1])));
      __vals[2] = static_cast<_u32>(std::log(static_cast<_u32>(__vals[2])));
      __vals[3] = static_cast<_u32>(std::log(static_cast<_u32>(__vals[3])));

      neon_u32 __log_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __log_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::log(static_cast<_s32>(__vals[0])));
      __vals[1] = static_cast<_s32>(std::log(static_cast<_s32>(__vals[1])));
      __vals[2] = static_cast<_s32>(std::log(static_cast<_s32>(__vals[2])));
      __vals[3] = static_cast<_s32>(std::log(static_cast<_s32>(__vals[3])));

      neon_s32 __log_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __log_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_log10_() {
  index_type __i = 0;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::log10(__vals[0]));
      __vals[1] = static_cast<_f32>(std::log10(__vals[1]));
      __vals[2] = static_cast<_f32>(std::log10(__vals[2]));
      __vals[3] = static_cast<_f32>(std::log10(__vals[3]));

      neon_f32 __log_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __log_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::log10(__vals[0]));
      __vals[1] = static_cast<_u32>(std::log10(__vals[1]));
      __vals[2] = static_cast<_u32>(std::log10(__vals[2]));
      __vals[3] = static_cast<_u32>(std::log10(__vals[3]));

      neon_u32 __log_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __log_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::log10(__vals[0]));
      __vals[1] = static_cast<_s32>(std::log10(__vals[1]));
      __vals[2] = static_cast<_s32>(std::log10(__vals[2]));
      __vals[3] = static_cast<_s32>(std::log10(__vals[3]));

      neon_s32 __log_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __log_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log10(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_log2_() {
  index_type __i = 0;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::log2(__vals[0]));
      __vals[1] = static_cast<_f32>(std::log2(__vals[1]));
      __vals[2] = static_cast<_f32>(std::log2(__vals[2]));
      __vals[3] = static_cast<_f32>(std::log2(__vals[3]));

      neon_f32 __log2_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __log2_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::log2(__vals[0]));
      __vals[1] = static_cast<_u32>(std::log2(__vals[1]));
      __vals[2] = static_cast<_u32>(std::log2(__vals[2]));
      __vals[3] = static_cast<_u32>(std::log2(__vals[3]));

      neon_u32 __log2_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __log2_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::log2(__vals[0]));
      __vals[1] = static_cast<_s32>(std::log2(__vals[1]));
      __vals[2] = static_cast<_s32>(std::log2(__vals[2]));
      __vals[3] = static_cast<_s32>(std::log2(__vals[3]));

      neon_s32 __log2_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __log2_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::log2(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_exp_() {
  index_type __i = 0;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::exp(__vals[0]));
      __vals[1] = static_cast<_f32>(std::exp(__vals[1]));
      __vals[2] = static_cast<_f32>(std::exp(__vals[2]));
      __vals[3] = static_cast<_f32>(std::exp(__vals[3]));

      neon_f32 __exp_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __exp_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::exp(__vals[0]));
      __vals[1] = static_cast<_u32>(std::exp(__vals[1]));
      __vals[2] = static_cast<_u32>(std::exp(__vals[2]));
      __vals[3] = static_cast<_u32>(std::exp(__vals[3]));

      neon_u32 __exp_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __exp_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::exp(__vals[0]));
      __vals[1] = static_cast<_s32>(std::exp(__vals[1]));
      __vals[2] = static_cast<_s32>(std::exp(__vals[2]));
      __vals[3] = static_cast<_s32>(std::exp(__vals[3]));

      neon_s32 __exp_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __exp_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::exp(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sqrt_() {
  index_type __i = 0;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::sqrt(__vals[0]));
      __vals[1] = static_cast<_f32>(std::sqrt(__vals[1]));
      __vals[2] = static_cast<_f32>(std::sqrt(__vals[2]));
      __vals[3] = static_cast<_f32>(std::sqrt(__vals[3]));

      neon_f32 __sqrt_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __sqrt_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::sqrt(__vals[0]));
      __vals[1] = static_cast<_s32>(std::sqrt(__vals[1]));
      __vals[2] = static_cast<_s32>(std::sqrt(__vals[2]));
      __vals[3] = static_cast<_s32>(std::sqrt(__vals[3]));

      neon_s32 __sqrt_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __sqrt_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::sqrt(__vals[0]));
      __vals[1] = static_cast<_u32>(std::sqrt(__vals[1]));
      __vals[2] = static_cast<_u32>(std::sqrt(__vals[2]));
      __vals[3] = static_cast<_u32>(std::sqrt(__vals[3]));

      neon_u32 __sqrt_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __sqrt_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sqrt(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_cos_() {
  index_type __i = 0;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::cos(__vals[0]));
      __vals[1] = static_cast<_f32>(std::cos(__vals[1]));
      __vals[2] = static_cast<_f32>(std::cos(__vals[2]));
      __vals[3] = static_cast<_f32>(std::cos(__vals[3]));

      neon_f32 __cos_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __cos_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::cos(__vals[0]));
      __vals[1] = static_cast<_s32>(std::cos(__vals[1]));
      __vals[2] = static_cast<_s32>(std::cos(__vals[2]));
      __vals[3] = static_cast<_s32>(std::cos(__vals[3]));

      neon_s32 __cos_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __cos_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::cos(__vals[0]));
      __vals[1] = static_cast<_u32>(std::cos(__vals[1]));
      __vals[2] = static_cast<_u32>(std::cos(__vals[2]));
      __vals[3] = static_cast<_u32>(std::cos(__vals[3]));

      neon_u32 __cos_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __cos_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::cos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_acos_() {
  index_type __i = 0;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::acos(__vals[0]));
      __vals[1] = static_cast<_f32>(std::acos(__vals[1]));
      __vals[2] = static_cast<_f32>(std::acos(__vals[2]));
      __vals[3] = static_cast<_f32>(std::acos(__vals[3]));

      neon_f32 __cos_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __cos_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::acos(__vals[0]));
      __vals[1] = static_cast<_s32>(std::acos(__vals[1]));
      __vals[2] = static_cast<_s32>(std::acos(__vals[2]));
      __vals[3] = static_cast<_s32>(std::acos(__vals[3]));

      neon_s32 __cos_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __cos_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::acos(this->__data_[__i]));

  return *this;
}

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

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::sin(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_tan_() {
  index_type __i        = 0;
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::tan(__vals[0]));
      __vals[1] = static_cast<_f32>(std::tan(__vals[1]));
      __vals[2] = static_cast<_f32>(std::tan(__vals[2]));
      __vals[3] = static_cast<_f32>(std::tan(__vals[3]));

      neon_f32 __tan_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __tan_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::tan(__vals[0]));
      __vals[1] = static_cast<_s32>(std::tan(__vals[1]));
      __vals[2] = static_cast<_s32>(std::tan(__vals[2]));
      __vals[3] = static_cast<_s32>(std::tan(__vals[3]));

      neon_s32 __tan_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __tan_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::tan(__vals[0]));
      __vals[1] = static_cast<_u32>(std::tan(__vals[1]));
      __vals[2] = static_cast<_u32>(std::tan(__vals[2]));
      __vals[3] = static_cast<_u32>(std::tan(__vals[3]));

      neon_u32 __tan_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __tan_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::tan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_tanh_() {
  index_type __i = 0;

  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::tanh(__vals[0]));
      __vals[1] = static_cast<_f32>(std::tanh(__vals[1]));
      __vals[2] = static_cast<_f32>(std::tanh(__vals[2]));
      __vals[3] = static_cast<_f32>(std::tanh(__vals[3]));

      neon_f32 __tanh_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __tanh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::tanh(__vals[0]));
      __vals[1] = static_cast<_s32>(std::tanh(__vals[1]));
      __vals[2] = static_cast<_s32>(std::tanh(__vals[2]));
      __vals[3] = static_cast<_s32>(std::tanh(__vals[3]));

      neon_s32 __tanh_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __tanh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::tanh(__vals[0]));
      __vals[1] = static_cast<_u32>(std::tanh(__vals[1]));
      __vals[2] = static_cast<_u32>(std::tanh(__vals[2]));
      __vals[3] = static_cast<_u32>(std::tanh(__vals[3]));

      neon_u32 __tanh_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __tanh_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::tanh(static_cast<_f32>(this->__data_[__i])));

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

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = (std::abs(this->__data_[__i]) < 1e-6)
                             ? static_cast<value_type>(1.0)
                             : static_cast<value_type>(std::sin(M_PI * this->__data_[__i]) /
                                                       (M_PI * this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_atan_() {
  index_type __i        = 0;
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::atan(static_cast<_f32>(__vals[0])));
      __vals[1] = static_cast<_f32>(std::atan(static_cast<_f32>(__vals[1])));
      __vals[2] = static_cast<_f32>(std::atan(static_cast<_f32>(__vals[2])));
      __vals[3] = static_cast<_f32>(std::atan(static_cast<_f32>(__vals[3])));

      neon_f32 __atan_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __atan_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::atan(static_cast<_u32>(__vals[0])));
      __vals[1] = static_cast<_u32>(std::atan(static_cast<_u32>(__vals[1])));
      __vals[2] = static_cast<_u32>(std::atan(static_cast<_u32>(__vals[2])));
      __vals[3] = static_cast<_u32>(std::atan(static_cast<_u32>(__vals[3])));

      neon_u32 __atan_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __atan_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::atan(static_cast<_s32>(__vals[0])));
      __vals[1] = static_cast<_s32>(std::atan(static_cast<_s32>(__vals[1])));
      __vals[2] = static_cast<_s32>(std::atan(static_cast<_s32>(__vals[2])));
      __vals[3] = static_cast<_s32>(std::atan(static_cast<_s32>(__vals[3])));

      neon_s32 __atan_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __atan_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_atanh_() {
  index_type __i        = 0;
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::atanh(__vals[0]));
      __vals[1] = static_cast<_f32>(std::atanh(__vals[1]));
      __vals[2] = static_cast<_f32>(std::atanh(__vals[2]));
      __vals[3] = static_cast<_f32>(std::atanh(__vals[3]));

      neon_f32 __atan_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __atan_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::atanh(__vals[0]));
      __vals[1] = static_cast<_u32>(std::atanh(__vals[1]));
      __vals[2] = static_cast<_u32>(std::atanh(__vals[2]));
      __vals[3] = static_cast<_u32>(std::atanh(__vals[3]));

      neon_u32 __atan_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __atan_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::atanh(__vals[0]));
      __vals[1] = static_cast<_s32>(std::atanh(__vals[1]));
      __vals[2] = static_cast<_s32>(std::atanh(__vals[2]));
      __vals[3] = static_cast<_s32>(std::atanh(__vals[3]));

      neon_s32 __atan_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __atan_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

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

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::asin(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_cosh_() {
  index_type       __i        = 0;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::cosh(__vals[0]));
      __vals[1] = static_cast<_f32>(std::cosh(__vals[1]));
      __vals[2] = static_cast<_f32>(std::cosh(__vals[2]));
      __vals[3] = static_cast<_f32>(std::cosh(__vals[3]));

      neon_f32 __cosh_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __cosh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::cosh(__vals[0]));
      __vals[1] = static_cast<_s32>(std::cosh(__vals[1]));
      __vals[2] = static_cast<_s32>(std::cosh(__vals[2]));
      __vals[3] = static_cast<_s32>(std::cosh(__vals[3]));

      neon_s32 __cosh_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __cosh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::cosh(__vals[0]));
      __vals[1] = static_cast<_u32>(std::cosh(__vals[1]));
      __vals[2] = static_cast<_u32>(std::cosh(__vals[2]));
      __vals[3] = static_cast<_u32>(std::cosh(__vals[3]));

      neon_u32 __cosh_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __cosh_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::cosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_acosh_() {
  index_type       __i        = 0;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::acosh(__vals[0]));
      __vals[1] = static_cast<_f32>(std::acosh(__vals[1]));
      __vals[2] = static_cast<_f32>(std::acosh(__vals[2]));
      __vals[3] = static_cast<_f32>(std::acosh(__vals[3]));

      neon_f32 __acosh_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __acosh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::acosh(__vals[0]));
      __vals[1] = static_cast<_s32>(std::acosh(__vals[1]));
      __vals[2] = static_cast<_s32>(std::acosh(__vals[2]));
      __vals[3] = static_cast<_s32>(std::acosh(__vals[3]));

      neon_s32 __acosh_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __acosh_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_u32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::acosh(__vals[0]));
      __vals[1] = static_cast<_u32>(std::acosh(__vals[1]));
      __vals[2] = static_cast<_u32>(std::acosh(__vals[2]));
      __vals[3] = static_cast<_u32>(std::acosh(__vals[3]));

      neon_u32 __acosh_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __acosh_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::acosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_pow_(const value_type __val) {
  index_type       __i        = 0;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] =
          static_cast<_f32>(std::pow(static_cast<_f32>(__vals[0]), static_cast<_f32>(__val)));
      __vals[1] =
          static_cast<_f32>(std::pow(static_cast<_f32>(__vals[1]), static_cast<_f32>(__val)));
      __vals[2] =
          static_cast<_f32>(std::pow(static_cast<_f32>(__vals[2]), static_cast<_f32>(__val)));
      __vals[3] =
          static_cast<_f32>(std::pow(static_cast<_f32>(__vals[3]), static_cast<_f32>(__val)));

      neon_f32 __pow_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __pow_vec);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::pow(this->__data_[__i], __val));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_pow_(const tensor& __other) {
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    neon_f32 __base_vec   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
    neon_f32 __exp_vec    = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
    neon_f32 __result_vec = {
        static_cast<_f32>(std::pow(static_cast<_f32>(vgetq_lane_f32(__base_vec, 0)),
                                   static_cast<_f32>(vgetq_lane_f32(__exp_vec, 0)))),
        static_cast<_f32>(std::pow(static_cast<_f32>(vgetq_lane_f32(__base_vec, 1)),
                                   static_cast<_f32>(vgetq_lane_f32(__exp_vec, 1)))),
        static_cast<_f32>(std::pow(static_cast<_f32>(vgetq_lane_f32(__base_vec, 2)),
                                   static_cast<_f32>(vgetq_lane_f32(__exp_vec, 2)))),
        static_cast<_f32>(std::pow(static_cast<_f32>(vgetq_lane_f32(__base_vec, 3)),
                                   static_cast<_f32>(vgetq_lane_f32(__exp_vec, 3))))};
    vst1q_f32(&this->__data_[__i], __result_vec);
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    neon_s32 __base_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
    neon_s32 __exp_vec    = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
    neon_s32 __result_vec = {
        static_cast<_s32>(std::pow(static_cast<_s32>(vgetq_lane_s32(__base_vec, 0)),
                                   static_cast<_s32>(vgetq_lane_s32(__exp_vec, 0)))),
        static_cast<_s32>(std::pow(static_cast<_s32>(vgetq_lane_s32(__base_vec, 1)),
                                   static_cast<_s32>(vgetq_lane_s32(__exp_vec, 1)))),
        static_cast<_s32>(std::pow(static_cast<_s32>(vgetq_lane_s32(__base_vec, 2)),
                                   static_cast<_s32>(vgetq_lane_s32(__exp_vec, 2)))),
        static_cast<_s32>(std::pow(static_cast<_s32>(vgetq_lane_s32(__base_vec, 3)),
                                   static_cast<_s32>(vgetq_lane_s32(__exp_vec, 3))))};

    vst1q_s32(&this->__data_[__i], __result_vec);
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    neon_u32 __base_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
    neon_u32 __exp_vec    = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
    neon_u32 __result_vec = {
        static_cast<_u32>(std::pow(static_cast<_u32>(vgetq_lane_u32(__base_vec, 0)),
                                   static_cast<_u32>(vgetq_lane_u32(__exp_vec, 0)))),
        static_cast<_u32>(std::pow(static_cast<_u32>(vgetq_lane_u32(__base_vec, 1)),
                                   static_cast<_u32>(vgetq_lane_u32(__exp_vec, 1)))),
        static_cast<_u32>(std::pow(static_cast<_u32>(vgetq_lane_u32(__base_vec, 2)),
                                   static_cast<_u32>(vgetq_lane_u32(__exp_vec, 2)))),
        static_cast<_u32>(std::pow(static_cast<_u32>(vgetq_lane_u32(__base_vec, 3)),
                                   static_cast<_u32>(vgetq_lane_u32(__exp_vec, 3))))};

    vst1q_u32(&this->__data_[__i], __result_vec);
  }
#endif

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::pow(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_abs_() {
  index_type __i = 0;

  if (std::is_unsigned_v<value_type>) return *this;

  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::abs(__vals[0]));
      __vals[1] = static_cast<_f32>(std::abs(__vals[1]));
      __vals[2] = static_cast<_f32>(std::abs(__vals[2]));
      __vals[3] = static_cast<_f32>(std::abs(__vals[3]));

      neon_f32 __abs_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __abs_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::abs(__vals[0]));
      __vals[1] = static_cast<_s32>(std::abs(__vals[1]));
      __vals[2] = static_cast<_s32>(std::abs(__vals[2]));
      __vals[3] = static_cast<_s32>(std::abs(__vals[3]));

      neon_s32 __abs_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __abs_vec);
    }
  }
  // don't need to check for unsigned values because they are already positive

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::abs(this->__data_[__i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_dist_(const tensor& __other) {
  assert(this->__shape_ == __other.shape());
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a    = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b    = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __diff = vabdq_f32(__a, __b);
      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __diff);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __a    = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __b    = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __diff = vabdq_s32(__a, __b);
      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __diff);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __a    = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __b    = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __diff = vabdq_u32(__a, __b);
      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __diff);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::abs(static_cast<_f64>(this->__data_[__i] - __other.__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_dist_(const value_type __val) {
  index_type       __i        = 0;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a    = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b    = vdupq_n_f32(&__val);
      neon_f32 __diff = vabdq_f32(__a, __b);
      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __diff);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __a    = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __b    = vdupq_n_s32(&__val);
      neon_s32 __diff = vabdq_s32(__a, __b);
      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __diff);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __a    = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __b    = vdupq_n_u32(&__val);
      neon_u32 __diff = vabdq_u32(__a, __b);
      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __diff);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] =
        static_cast<value_type>(std::abs(static_cast<_f64>(this->__data_[__i] - __val)));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_maximum_(const tensor& __other) {
  assert(this->__shape_ == __other.shape());
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b   = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __max = vmaxq_f32(__a, __b);
      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __max);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __a   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __b   = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __max = vmaxq_s32(__a, __b);
      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __max);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __a   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __b   = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __max = vmaxq_u32(__a, __b);
      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __max);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::max(this->__data_[__i], __other.__data_[__i]);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_maximum_(const value_type __val) {
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    neon_f32 __val_vec = vdupq_n_f32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __max = vmaxq_f32(__a, __val_vec);
      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __max);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    neon_s32 __val_vec = vdupq_n_s32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __a   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __max = vmaxq_s32(__a, __val_vec);
      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __max);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    neon_u32 __val_vec = vdupq_n_u32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __a   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __max = vmaxq_u32(__a, __val_vec);
      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __max);
    }
  }

  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = std::max(this->__data_[__i], __val);

  return *this;
}

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
