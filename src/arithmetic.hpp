#pragma once

#include "tensorbase.hpp"

#define __builtin_neon_vgetq_lane_f32
#define __builtin_neon_vsetq_lane_f32

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmax(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fmax_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmax(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fmax_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const value_type __val) {
  size_t __i = 0;
#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value) {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    neon_f32         __scalar_val = vdupq_n_f32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a       = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __max_val = vmaxq_f32(__a, __scalar_val);

      vst1q_f32(&this->__data_[__i], __max_val);
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = std::fmax(this->__data_[__i], __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmax_(const value_type __val) const {
  return this->fmax_(__val);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const tensor& __other) {
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value) {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __a       = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b       = vld1q_f32(reinterpret_cast<const _f32*>(&(__other[__i])));
      neon_f32 __max_val = vmaxq_f32(__a, __b);

      vst1q_f32(&this->__data_[__i], __max_val);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = std::fmax(this->__data_[__i], __other[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmax_(const tensor& __other) const {
  return this->fmax_(__other);
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmod(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fmod_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::fmod(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fmod_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::fmod_(const value_type __val) {
  assert(std::is_floating_point<value_type>::value &&
         "fmod : template class must be a floating point type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value) {
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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__val)));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmod_(const value_type __val) const {
  return this->fmod_(__val);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::fmod_(const tensor& __other) {
  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");

  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value) {
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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(
        std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::fmod_(const tensor& __other) const {
  return this->fmod_(__other);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::frac_() {
  index_type __i = 0;

#if defined(__ARM_NEON)
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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::frac_() const {
  return this->frac_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::frac() const {
  __self __ret = this->clone();
  __ret.frac_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::log_() {
  index_type __i = 0;

#if defined(__ARM_NEON)

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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::log(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log_() const {
  return this->log_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::log() const {
  __self __ret = this->clone();
  __ret.log_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::log10_() {
  index_type __i = 0;

#if defined(__ARM_NEON)

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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::log10(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log10_() const {
  return this->log10_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::log10() const {
  __self __ret = this->clone();
  __ret.log10_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::log2_() {
  index_type __i = 0;

#if defined(__ARM_NEON)

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

#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::log2(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log2_() const {
  return this->log2_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::log2() const {
  __self __ret = this->clone();
  __ret.log2_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::exp_() {
  index_type __i = 0;

#if defined(__ARM_NEON)

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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::exp(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::exp_() const {
  return this->exp_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::exp() const {
  __self __ret = this->clone();
  __ret.exp_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sqrt_() {
  index_type __i = 0;

#if defined(__ARM_NEON)
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

#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::sqrt(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sqrt_() const {
  return this->sqrt_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sqrt() const {
  __self __ret = this->clone();
  __ret.sqrt_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::square() const {
  __self __ret = this->clone();
  __ret.square_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::cos_() {
  index_type __i = 0;

#if defined(__ARM_NEON)
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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::cos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::cos_() const {
  return this->cos_();
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::acos_() {
  index_type __i = 0;

#if defined(__ARM_NEON)

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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::acos(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::acos_() const {
  return this->acos_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::acos() const {
  __self __ret = this->clone();
  __ret.acos_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::cos() const {
  __self __ret = this->clone();
  __ret.cos_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sin_() {
  index_type __i = 0;

#if defined(__ARM_NEON)

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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::sin(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sin_() const {
  return this->sin_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sin() const {
  __self __ret = this->clone();
  __ret.sin_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::tan_() {
  index_type __i = 0;

#if defined(__ARM_NEON)

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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::tan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::tan_() const {
  return this->tan_();
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::tanh_() {
  index_type __i = 0;

#if defined(__ARM_NEON)

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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::tanh(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::tanh_() const {
  return this->tanh_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::asin() const {
  __self __ret = this->clone();
  __ret.asin_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::cosh() const {
  __self __ret = this->clone();
  __ret.cosh_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::atan() const {
  __self __ret = this->clone();
  __ret.atan_();
  return __ret;
}

inline neon_f32 vsinq_f32(neon_f32 __x) {
  return {sinf(vgetq_lane_f32(__x, 0)), sinf(vgetq_lane_f32(__x, 1)), sinf(vgetq_lane_f32(__x, 2)),
          sinf(vgetq_lane_f32(__x, 3))};
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sinc_() {
  index_type __i = 0;

#if defined(__ARM_NEON)

  if constexpr (std::is_same<value_type, _f32>::value) {
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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = (std::abs(this->__data_[__i]) < 1e-6)
                             ? static_cast<value_type>(1.0)
                             : static_cast<value_type>(std::sin(M_PI * this->__data_[__i]) /
                                                       (M_PI * this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sinc_() const {
  return this->sinc_();
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::atan_() {
  index_type __i = 0;

#if defined(__ARM_NEON)
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

#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::atan_() const {
  return this->atan_();
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::atanh_() {
  index_type __i = 0;

#if defined(__ARM_NEON)
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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::atanh_() const {
  return this->atanh_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::atanh() const {
  __self __ret = this->clone();
  __ret.atanh_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sinc() const {
  __self __ret = this->clone();
  __ret.sinc_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sinh_() {
  index_type __i = 0;

#if defined(__ARM_NEON)
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

#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::sinh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sinh_() const {
  return this->sinh_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::sinh() const {
  __self __ret = this->clone();
  __ret.sinh_();
  return __ret;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::asinh_() const {
  return this->asinh_();
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::asinh_() {
  index_type __i = 0;

#if defined(__ARM_NEON)
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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::asinh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::asinh() const {
  __self __ret = this->clone();
  __ret.asinh_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::asin_() {
  index_type __i = 0;

#if defined(__ARM_NEON)
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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::asin(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::asin_() const {
  return this->asin_();
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::cosh_() {
  index_type __i = 0;

#if defined(__ARM_NEON)

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

#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::cosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::cosh_() const {
  return this->cosh_();
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::acosh_() {
  index_type __i = 0;

#if defined(__ARM_NEON)

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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::acosh(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::acosh_() const {
  return this->acosh_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::acosh() const {
  __self __ret = this->clone();
  __ret.acosh_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::pow_(const value_type __val) {
  index_type __i = 0;

#if defined(__ARM_NEON)

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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::pow(this->__data_[__i], __val));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::pow_(const value_type __val) const {
  return this->pow_(__val);
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::pow(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.pow_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::pow(const value_type __val) const {
  __self __ret = this->clone();
  __ret.pow_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::pow_(const tensor& __other) {
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

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(
        std::pow(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::pow_(const tensor& __other) const {
  return this->pow_(__other);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::abs() const {
  __self __ret = this->clone();
  __ret.abs_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::abs_() {
  index_type __i = 0;

  if (std::is_unsigned<value_type>::value) return *this;

#if defined(__ARM_NEON)
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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::abs(this->__data_[__i]));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::abs_() const {
  return this->abs_();
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::log_softmax(const index_type __dim) const {
  __self __ret = this->clone();
  __ret.log_softmax_(__dim);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::dist(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.dist_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::dist(const value_type __val) const {
  __self __ret = this->clone();
  __ret.dist_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::dist_(const tensor& __other) {
  assert(this->__shape_ == __other.shape());

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(
        std::abs(static_cast<_f64>(this->__data_[__i] - __other.__data_[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::dist_(const tensor& __other) const {
  return this->dist_(__other);
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::dist_(const value_type __val) {
  // TODO: implement dist_
  return *this;
}

template <class _Tp>
const tensor<_Tp>& tensor<_Tp>::dist_(const value_type __val) const {
  index_type __i = 0;

#if defined(__ARM_NEON)
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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] =
        static_cast<value_type>(std::abs(static_cast<_f64>(this->__data_[__i] - __val)));

  return *this;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::remainder(const value_type __val) const {
  __self __ret = this->clone();
  __ret.remainder_(__val);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::remainder(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.remainder_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::remainder_(const value_type __val) {
  assert(__val != 0 && "Remainder by zero is undefined");

  for (index_type __i = 0; __i < this->__data_.size(); __i++) this->__data_[__i] %= __val;

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::remainder_(const value_type __val) const {
  return this->remainder_(__val);
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::remainder_(const tensor& __other) {
  assert(__other.count_nonzero() == __other.size(0) && "Remainder by zero is undefined");
  assert(this->__shape_ == __other.shape());

  index_type __i = 0;
  for (; __i < this->__data_.size(); __i++) this->__data_[__i] %= __other.__data_[__i];

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::remainder_(const tensor& __other) const {
  return this->remainder_(__other);
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::maximum(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.maximum_(__other);
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::maximum(const_reference __val) const {
  __self __ret = this->clone();
  __ret.maximum_(__val);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::maximum_(const tensor& __other) {
  assert(this->__shape_ == __other.shape());

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

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

#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = std::max(this->__data_[__i], __other.__data_[__i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::maximum_(const tensor& __other) const {
  return this->maximum_(__other);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::maximum_(const value_type __val) {
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

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
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = std::max(this->__data_[__i], __val);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::maximum_(const value_type __val) const {
  return this->maximum_(__val);
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::square_() {
  return this->pow_(static_cast<value_type>(2.0f));
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::square_() const {
  return this->square_();
}

template <class _Tp>
double tensor<_Tp>::mean() const {
  double __m = 0.0;

  if (this->empty()) return __m;

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  neon_s32         __sum_vec  = vdupq_n_s32(0);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
    neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
    __sum_vec           = vaddq_s32(__sum_vec, __data_vec);
  }

  _s32 __partial_sum[4];
  vst1q_s32(__partial_sum, __sum_vec);
  __m += __partial_sum[0] + __partial_sum[1] + __partial_sum[2] + __partial_sum[3];
#endif

  for (; __i < this->__data_.size(); __i++) __m += this->__data_[__i];

  return static_cast<double>(__m) / static_cast<double>(this->__data_.size());
}

// used as a helper function
inline int64_t __lcm(const int64_t __a, const int64_t __b) {
  return (__a * __b) / std::gcd(__a, __b);
}

template <class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::lcm() const {
  index_type __ret = static_cast<index_type>(this->__data_[0]);
  index_type __i   = 1;

  for (; __i < this->__data_.size(); __i++)
    __ret = __lcm(static_cast<index_type>(this->__data_[__i]), __ret);

  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::lcm(const tensor& __other) const {
  assert(this->__shape_ == __other.shape());

  tensor     __ret = this->clone();
  index_type __i   = 0;

  for (; __i < this->__data_.size(); __i++)
    __ret[__i] = static_cast<value_type>(this->__lcm(static_cast<index_type>(this->__data_[__i]),
                                                     static_cast<index_type>(__other[__i])));

  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::tanh() const {
  __self __ret = this->clone();
  __ret.tanh_();
  return __ret;
}

template <class _Tp>
inline tensor<_Tp> tensor<_Tp>::tan() const {
  __self __ret = this->clone();
  __ret.tan_();
  return __ret;
}

template <class _Tp>
double tensor<_Tp>::mode(const index_type __dim) const {
  if (__dim >= this->n_dims() || __dim < -1)
    throw std::invalid_argument("given dimension is out of range of the tensor dimensions");

  index_type __stride = (__dim == -1) ? 0 : this->__strides_[__dim];
  index_type __end    = (__dim == -1) ? this->__data_.size() : this->__strides_[__dim];

  if (this->__data_.empty()) return std::numeric_limits<double>::quiet_NaN();

  std::unordered_map<value_type, size_t> __counts;
  for (index_type __i = __stride; __i < __end; __i++) __counts[this->__data_[__i]]++;

  value_type __ret  = 0;
  size_t     __most = 0;

  for (const std::pair<value_type, size_t>& __pair : __counts) {
    if (__pair.second > __most) {
      __ret  = __pair.first;
      __most = __pair.second;
    }
  }

  return static_cast<double>(__ret);
}

template <class _Tp>
double tensor<_Tp>::median(const index_type __dim) const {
  if (__dim >= this->n_dims() || __dim < -1)
    throw std::invalid_argument("given dimension is out of range of the tensor dimensions");

  index_type __stride = (__dim == -1) ? 0 : this->__strides_[__dim];
  index_type __end    = (__dim == -1) ? this->__data_.size() : this->__strides_[__dim];

  data_t __d(this->__data_.begin() + __stride, this->__data_.begin() + __end);

  if (__d.empty()) return std::numeric_limits<double>::quiet_NaN();

  std::nth_element(__d.begin(), __d.begin() + __d.size() / 2, __d.end());

  if (__d.size() % 2 == 0) {
    std::nth_element(__d.begin(), __d.begin() + __d.size() / 2 - 1, __d.end());
    return (static_cast<double>(__d[__d.size() / 2]) + __d[__d.size() / 2 - 1]) / 2.0;
  }

  return __d[__d.size() / 2];
}