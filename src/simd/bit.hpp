#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_right_shift_(const int amount) {
  if (!std::is_integral_v<value_type>) throw type_error("Type must be integral");

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);
  index_type       i        = 0;

  if constexpr (std::is_signed_v<value_type>) {
    const neon_s32 shift_amount_vec = vdupq_n_s32(-amount);

    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec    = vld1q_s32(reinterpret_cast<_s32*>(&this->data_[i]));
      neon_s32 shifted_vec = vshlq_s32(data_vec, shift_amount_vec);

      vst1q_s32(&this->data_[i], shifted_vec);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    const neon_s32 shift_amount_vec = vdupq_n_s32(-amount);

    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 data_vec    = vld1q_u32(reinterpret_cast<const _u32*>(&this->data_[i]));
      neon_u32 shifted_vec = vshlq_u32(data_vec, shift_amount_vec);

      vst1q_u32(&this->data_[i], shifted_vec);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] >>= amount;

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_left_shift_(const int amount) {
  if (!std::is_integral_v<value_type>) throw type_error("Type must be integral");

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);
  index_type       i        = 0;

  if constexpr (std::is_signed_v<value_type>) {
    const neon_s32 shift_amount_vec = vdupq_n_s32(amount);

    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec    = vld1q_s32(reinterpret_cast<_s32*>(&this->data_[i]));
      neon_s32 shifted_vec = vshlq_s32(data_vec, shift_amount_vec);

      vst1q_s32(&this->data_[i], shifted_vec);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    const neon_s32 shift_amount_vec = vdupq_n_s32(amount);

    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 data_vec    = vld1q_u32(reinterpret_cast<_u32*>(&this->data_[i]));
      neon_u32 shifted_vec = vshlq_u32(data_vec, shift_amount_vec);

      vst1q_u32(&this->data_[i], shifted_vec);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] <<= amount;

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_or_(const value_type val) {
  if (!std::is_integral_v<value_type>)
    throw type_error("Cannot perform a bitwise OR on non-integral values");

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);
  index_type       i        = 0;

  if constexpr (std::is_signed_v<value_type>) {
    neon_s32 val_vec = vdupq_n_s32(val);
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec   = vld1q_s32(reinterpret_cast<_s32*>(&this->data_[i]));
      neon_s32 result_vec = vorrq_s32(data_vec, val_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->data_[i]), result_vec);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 val_vec = vdupq_n_u32(val);
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 data_vec   = vld1q_u32(reinterpret_cast<_u32*>(&this->data_[i]));
      neon_u32 result_vec = vorrq_u32(data_vec, val_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->data_[i]), result_vec);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] |= val;

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_xor_(const value_type val) {
  if (!std::is_integral_v<value_type>)
    throw type_error("Cannot perform a bitwise XOR on non-integral values");

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);
  index_type       i        = 0;

  if constexpr (std::is_signed_v<value_type>) {
    neon_s32 val_vec = vdupq_n_s32(val);
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec   = vld1q_s32(reinterpret_cast<_s32*>(&this->data_[i]));
      neon_s32 result_vec = veorq_s32(data_vec, val_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->data_[i]), result_vec);
    }
  } else {
    neon_u32 val_vec = vdupq_n_u32(val);
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 data_vec   = vld1q_u32(reinterpret_cast<_u32*>(&this->data_[i]));
      neon_u32 result_vec = veorq_u32(data_vec, val_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->data_[i]), result_vec);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] ^= val;

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_not_() {
  if (!std::is_integral_v<value_type> and !std::is_same_v<value_type, bool>)
    throw type_error("Cannot perform a bitwise not on non-integral value");

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);
  index_type       i        = 0;

  if constexpr (std::is_signed_v<value_type>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->data_[i]));
      neon_s32 result_vec = vmvnq_s32(data_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->data_[i]), result_vec);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->data_[i]));
      neon_u32 result_vec = vmvnq_u32(data_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->data_[i]), result_vec);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] = ~this->data_[i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_and_(const value_type val) {
  if (!std::is_integral_v<value_type>)
    throw type_error("Cannot perform a bitwise AND on non-integral values");

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);
  index_type       i        = 0;

  if constexpr (std::is_signed_v<value_type>) {
    neon_s32 val_vec = vdupq_n_s32(reinterpret_cast<_s32>(&val));
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->data_[i]));
      neon_s32 result_vec = vandq_s32(data_vec, val_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->data_[i]), result_vec);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 val_vec = vdupq_n_u32(reinterpret_cast<_u32>(&val));
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->data_[i]));
      neon_u32 result_vec = vandq_u32(data_vec, val_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->data_[i]), result_vec);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] &= val;

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_and_(const tensor& other) {
  if (!std::is_integral_v<value_type>)
    throw type_error("Cannot perform a bitwise AND on non-integral values");

  if (!equal_shape(this->shape(), other.shape())) throw shape_error("Tensors shapes must be equal");

  const size_t simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);
  index_type   i        = 0;

  if constexpr (std::is_unsigned_v<value_type>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->data_[i]));
      neon_u32 other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other[i]));
      neon_u32 xor_vec   = vandq_u32(data_vec, other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->data_[i]), xor_vec);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->data_[i]));
      neon_s32 other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other[i]));
      neon_s32 xor_vec   = vandq_s32(data_vec, other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->data_[i]), xor_vec);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] &= other[i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_or_(const tensor& other) {
  if (!std::is_integral_v<value_type>)
    throw type_error("Cannot perform a bitwise OR on non-integral values");

  if (!equal_shape(this->shape(), other.shape())) throw shape_error("Tensors shapes must be equal");

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);
  index_type       i        = 0;

  if constexpr (std::is_unsigned_v<value_type>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->data_[i]));
      neon_u32 other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other[i]));
      neon_u32 xor_vec   = vornq_u32(data_vec, other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->data_[i]), xor_vec);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->data_[i]));
      neon_s32 other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other[i]));
      neon_s32 xor_vec   = vornq_s32(data_vec, other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->data_[i]), xor_vec);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] |= other[i];

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_bitwise_xor_(const tensor& other) {
  if (!std::is_integral_v<value_type>)
    throw type_error("Cannot perform a bitwise XOR on non-integral values");

  if (!equal_shape(this->shape(), other.shape())) throw shape_error("Tensors shapes must be equal");

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);
  index_type       i        = 0;

  if constexpr (std::is_unsigned_v<value_type>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->data_[i]));
      neon_u32 other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&other[i]));
      neon_u32 xor_vec   = veorq_u32(data_vec, other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->data_[i]), xor_vec);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->data_[i]));
      neon_s32 other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&other[i]));
      neon_s32 xor_vec   = veorq_s32(data_vec, other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->data_[i]), xor_vec);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) this->data_[i] ^= other[i];

  return *this;
}
