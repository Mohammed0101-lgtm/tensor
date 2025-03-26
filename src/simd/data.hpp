#pragma once

#include "tensorbase.hpp"

template <class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::neon_count_nonzero(index_type __dim) const {
  index_type __c           = 0;
  index_type __local_count = 0;
  index_type __i           = 0;
  if (__dim == 0) {
    if constexpr (std::is_floating_point_v<value_type>) {
      for (; __i < this->__data_.size(); __i += _ARM64_REG_WIDTH) {
        neon_f32 __vec          = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
        neon_u32 __nonzero_mask = vcgtq_f32(__vec, vdupq_n_f32(0.0f));
        __local_count += vaddvq_u32(vandq_u32(__nonzero_mask, vdupq_n_u32(1)));
      }
    } else if constexpr (std::is_unsigned_v<value_type>) {
      for (; __i < this->__data_.size(); __i += _ARM64_REG_WIDTH) {
        neon_u32 __vec          = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
        neon_u32 __nonzero_mask = vcgtq_u32(__vec, vdupq_n_u32(0));
        __local_count += vaddvq_u32(vandq_u32(__nonzero_mask, vdupq_n_u32(1)));
      }
    } else if constexpr (std::is_signed_v<value_type>) {
      for (; __i < this->__data_.size(); __i += _ARM64_REG_WIDTH) {
        neon_s32 __vec          = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
        neon_u32 __nonzero_mask = vcgtq_s32(__vec, vdupq_n_s32(0));
        __local_count += vaddvq_u32(vandq_u32(__nonzero_mask, vdupq_n_u32(1)));
      }
    }
#pragma omp parallel for reduction(+ : __local_count)
    for (index_type __j = __i; __j < this->__data_.size(); ++__j)
      if (this->__data_[__j] != 0) ++__local_count;

    __c += __local_count;
  } else {
    if (__dim < 0 || __dim >= static_cast<index_type>(__shape_.size()))
      throw __index_error__("Invalid dimension provided.");

    throw std::runtime_error("Dimension-specific non-zero counting is not implemented yet.");
  }

  return __c;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_zeros_(shape_type __sh) {
  if (__sh.empty())
    __sh = this->__shape_;
  else
    this->__shape_ = __sh;

  size_t __s = this->__computeSize(this->__shape_);
  this->__data_.resize(__s);
  this->__compute_strides();
  const index_type __simd_end = __s - (__s % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    neon_f32 __zero_vec = vdupq_n_f32(0.0f);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) vst1q_f32(&this->__data_[__i], __zero_vec);
  } else if constexpr (std::is_signed_v<value_type>) {
    neon_s32 __zero_vec = vdupq_n_s32(0);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) vst1q_s32(&this->__data_[__i], __zero_vec);
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __zero_vec = vdupq_n_u32(0);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) vst1q_u32(&this->__data_[__i], __zero_vec);
  }
#pragma omp parallel
  for (; __i < __s; ++__i) this->__data_[__i] = value_type(0.0);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_ones_(shape_type __sh) {
  if (__sh.empty())
    __sh = this->__shape_;
  else
    this->__shape_ = __sh;

  size_t __s = this->__computeSize(this->__shape_);
  this->__data_.resize(__s);
  this->__compute_strides();
  const index_type __simd_end = __s - (__s % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    neon_f32 __one_vec = vdupq_n_f32(1.0f);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __one_vec);
  } else if constexpr (std::is_signed_v<value_type>) {
    neon_s32 __one_vec = vdupq_n_s32(1);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __one_vec);
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 __one_vec = vdupq_n_u32(1);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __one_vec);
  }
#pragma omp parallel
  for (; __i < __s; ++__i) this->__data_[__i] = value_type(1.0);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_randomize_(const shape_type& __sh, bool __bounded) {
  if (__bounded)
    assert(std::is_floating_point_v<value_type> && "Cannot bound non floating point data type");

  if (__sh.empty() && this->__shape_.empty())
    throw __shape_error__("randomize_ : Shape must be initialized");

  if (this->__shape_.empty() || this->__shape_ != __sh) this->__shape_ = __sh;

  index_type __s = this->__computeSize(this->__shape_);
  this->__data_.resize(__s);
  this->__compute_strides();

  std::random_device                   __rd;
  std::mt19937                         __gen(__rd());
  std::uniform_real_distribution<_f32> __unbounded_dist(1.0f, static_cast<_f32>(RAND_MAX));
  std::uniform_real_distribution<_f32> __bounded_dist(0.0f, 1.0f);
  index_type                           __i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    const neon_f32 __scale = vdupq_n_f32(__bounded ? static_cast<_f32>(RAND_MAX) : 1.0f);
    for (; __i + _ARM64_REG_WIDTH <= static_cast<index_type>(__s); __i += _ARM64_REG_WIDTH) {
      neon_f32 __random_values;

      if (__bounded)
        __random_values = {__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen),
                           __bounded_dist(__gen)};
      else
        __random_values = {__unbounded_dist(__gen), __unbounded_dist(__gen),
                           __unbounded_dist(__gen), __unbounded_dist(__gen)};

      if (!__bounded) __random_values = vmulq_f32(__random_values, vrecpeq_f32(__scale));

      vst1q_f32(&this->__data_[__i], __random_values);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    const neon_f32 __scale = vdupq_n_f32(static_cast<_f32>(RAND_MAX));
    for (; __i + _ARM64_REG_WIDTH <= static_cast<index_type>(__s); __i += _ARM64_REG_WIDTH) {
      neon_f32 __rand_vals = {
          static_cast<_f32>(__unbounded_dist(__gen)), static_cast<_f32>(__unbounded_dist(__gen)),
          static_cast<_f32>(__unbounded_dist(__gen)), static_cast<_f32>(__unbounded_dist(__gen))};
      __rand_vals         = vmulq_f32(__rand_vals, vrecpeq_f32(__scale));
      neon_u32 __int_vals = vcvtq_u32_f32(__rand_vals);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __int_vals);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    const neon_f32 __scale = vdupq_n_f32(static_cast<_f32>(RAND_MAX));
    for (; __i + _ARM64_REG_WIDTH <= static_cast<index_type>(__s); __i += _ARM64_REG_WIDTH) {
      neon_f32 __rand_vals = {
          static_cast<_f32>(__unbounded_dist(__gen)), static_cast<_f32>(__unbounded_dist(__gen)),
          static_cast<_f32>(__unbounded_dist(__gen)), static_cast<_f32>(__unbounded_dist(__gen))};
      __rand_vals         = vmulq_f32(__rand_vals, vrecpeq_f32(__scale));
      neon_s32 __int_vals = vcvtq_s32_f32(__rand_vals);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __int_vals);
    }
  }
#pragma omp parallel
  for (; __i < static_cast<index_type>(__s); ++__i)
    this->__data_[__i] = value_type(__bounded ? __bounded_dist(__gen) : __unbounded_dist(__gen));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_negative_() {
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    neon_f32 __neg_multiplier = vdupq_n_f32(-1);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __v   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __neg = vmulq_f32(__v, __neg_multiplier);
      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __neg);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    neon_s32 __neg_multiplier = vdupq_n_s32(-1);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_s32 __v   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __neg = vmulq_s32(__v, __neg_multiplier);
      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __neg);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    neon_s32 __neg_multiplier = vdupq_n_s32(-1);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __v   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __neg = vmulq_u32(__v, __neg_multiplier);
      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __neg);
    }
  }
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = -this->__data_[__i];

  return *this;
}