#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_operator_plus(const tensor& __other) const {
  static_assert(has_plus_operator_v<value_type>, "Value type must have a plus operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Cannot add two tensors with different shapes");

  data_t           __d(this->__data_.size());
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __vec1   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __vec2   = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __result = vaddq_f32(__vec1, __vec2);

      vst1q_f32(reinterpret_cast<_f32*>(&__d[__i]), __result);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __vec1   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __vec2   = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __result = vaddq_s32(__vec1, __vec2);

      vst1q_s32(reinterpret_cast<_s32*>(&__d[__i]), __result);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __vec1   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __vec2   = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __result = vaddq_u32(__vec1, __vec2);

      vst1q_u32(reinterpret_cast<_u32*>(&__d[__i]), __result);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) __d[__i] = this->__data_[__i] + __other[__i];

  return __self(this->__shape_, __d);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_operator_plus(const value_type __val) const {
  static_assert(has_plus_operator_v<value_type>, "Value type must have a plus operator");

  data_t           __d(this->__data_.size());
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    neon_f32 __val_vec = vdupq_n_f32(reinterpret_cast<_f32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __vec = vld1q_f32(reinterpret_castr<const _f32*>(&this->__data_[__i]));
      neon_f32 __res = vaddq_f32(__vec, __val_vec);

      vst1q_f32(&__d[__i], __res);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    neon_s32 __val_vec = vdupq_n_s32(reinterpret_cast<_s32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __vec = vld1q_s32(reinterpret_castr<const _s32*>(&this->__data_[__i]));
      neon_s32 __res = vaddq_s32(__vec, __val_vec);

      vst1q_s32(&__d[__i], __res);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __val_vec = vdupq_n_u32(reinterpret_cast<_u32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __vec = vld1q_u32(reinterpret_castr<const _u32*>(&this->__data_[__i]));
      neon_u32 __res = vaddq_u32(__vec, __val_vec);

      vst1q_u32(&__d[__i], __res);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) __d[__i] = this->__data_[__i] + __val;

  return __self(__d, this->__shape_);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_operator_plus_eq(const_reference __val) const {
  static_assert(has_plus_operator_v<value_type>, "Value type must have a plus operator");

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    neon_f32 __val_vec = vdupq_n_f32(reinterpret_cast<_f32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __add_vec  = vaddq_f32(__data_vec, __val_vec);

      vst1q_f32(&this->__data_[__i], __add_vec);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    neon_s32 __val_vec = vdupq_n_s32(reinterpret_cast<_s32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __add_vec  = vaddq_s32(__data_vec, __val_vec);

      vst1q_s32(&this->__data_[__i], __add_vec);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __val_vec = vdupq_n_u32(reinterpret_cast<_u32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __add_vec  = vaddq_u32(__data_vec, __val_vec);

      vst1q_u32(&this->__data_[__i], __add_vec);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = this->__data_[__i] + __val;

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_operator_minus(const tensor& __other) const {
  static_assert(has_minus_operator_v<value_type>, "Value type must have a minus operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Cannot add two tensors with different shapes");

  data_t           __d(this->__data_.size());
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __oth = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __sub = vsubq_f32(__vec, __oth);

      vst1q_f32(&__d[__i], __sub);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __oth = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __sub = vsubq_s32(__vec, __oth);

      vst1q_s32(&__d[__i], __sub);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __oth = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __sub = vsubq_u32(__vec, __oth);

      vst1q_u32(&__d[__i], __sub);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_[__i]; ++__i) __d[__i] = this->__data_[__i] - __other[__i];

  return __self(this->__shape_, __d);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_operator_minus(const value_type __val) const {
  static_assert(has_minus_operator_v<value_type>, "Value type must have a minus operator");
  data_t           __d(this->__data_.size());
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    neon_f32 __vals = vdupq_n_f32(reinterpret_cast<_f32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __sub = vsubq_f32(__vec, __vals);

      vst1q_f32(&__d[__i], __sub);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    neon_s32 __vals = vdupq_n_s32(reinterpret_cast<_s32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __sub = vsubq_s32(__vec, __vals);

      vst1q_s32(&__d[__i], __sub);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __vals = vdupq_n_u32(reinterpret_cast<_u32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __sub = vsubq_u32(__vec, __vals);

      vst1q_u32(&__d[__i], __sub);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) __d[__i] = this->__data_[__i] - __val;

  return __self(*this);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_operator_minus_eq(const tensor& __other) const {
  static_assert(has_minus_operator_v<value_type>, "Value type must have a minus operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __oth = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __sub = vsubq_f32(__vec, __oth);

      vst1q_f32(&this->__data_[__i], __sub);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __oth = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __sub = vsubq_s32(__vec, __oth);

      vst1q_s32(&this->__data_[__i], __sub);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __oth = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __sub = vsubq_u32(__vec, __oth);

      vst1q_u32(&this->__data_[__i], __sub);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] -= __other[__i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_operator_times_eq(const tensor& __other) const {
  static_assert(has_times_operator_v<value_type>, "Value type must have a times operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

  index_type __i = 0;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __oth = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __mul = vmulq_f16(__vec, __oth);

      vst1q_f32(&this->__data_[__i], __mul);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __oth = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __mul = vmulq_s16(__vec, __oth);

      vst1q_s32(&this->__data_[__i], __mul);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __oth = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __mul = vmulq_u16(__vec, __oth);

      vst1q_u32(&this->__data_[__i], __mul);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] *= __other[__i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_operator_minus_eq(const_reference __val) const {
  static_assert(has_minus_operator_v<value_type>, "Value type must have a minus operator");

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    neon_f32 __val = vld1q_f32(reinterpret_cast<const _f32*>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __sub = vsubq_f32(__vec, __val);

      vst1q_f32(&this->__data_[__i], __sub);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __val = vld1q_u32(reinterpret_cast<const _u32*>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __sub = vsubq_u32(__vec, __val);

      vst1q_u32(&this->__data_[__i], __sub);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    neon_s32 __val = vld1q_s32(reinterpret_cast<const _s32*>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __sub = vsubq_s32(__vec, __val);

      vst1q_s32(&this->__data_[__i], __sub);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] -= __val;

  return *this;
}
