#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<bool> tensor<_Tp>::neon_equal(const tensor& __other) const {
  static_assert(has_equal_operator_v<value_type>, "Value type must have equal to operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

  std::vector<bool> __ret(this->__data_.size());
  const index_type  __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type        __i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_f32 __data_vec1  = vld1q_f32(reinterpret_cast<const float*>(&this->__data_[__i]));
      neon_f32 __data_vec2  = vld1q_f32(reinterpret_cast<const float*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vceqq_f32(__data_vec1, __data_vec2);
      neon_u8  __mask       = vreinterpretq_u8_u32(__cmp_result);

      __ret[__i]     = vgetq_lane_u8(__mask, 0);
      __ret[__i + 1] = vgetq_lane_u8(__mask, 4);
      __ret[__i + 2] = vgetq_lane_u8(__mask, 8);
      __ret[__i + 3] = vgetq_lane_u8(__mask, 12);
    }
  } else {  // Handles both signed and unsigned integers
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH) {
      neon_u32 __data_vec1  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
      neon_u32 __data_vec2  = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vceqq_u32(__data_vec1, __data_vec2);
      neon_u8  __mask       = vreinterpretq_u8_u32(__cmp_result);

      __ret[__i]     = vgetq_lane_u8(__mask, 0);
      __ret[__i + 1] = vgetq_lane_u8(__mask, 4);
      __ret[__i + 2] = vgetq_lane_u8(__mask, 8);
      __ret[__i + 3] = vgetq_lane_u8(__mask, 12);
    }
  }

#pragma omp parallel
  // Handle remaining elements
  for (; __i < this->__data_.size(); ++__i)
    __ret[__i] = (this->__data_[__i] == __other.__data_[__i]);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::neon_equal(const value_type __val) const {
  static_assert(has_equal_operator_v<value_type>, "Value type must have equal to operator");

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i        = 0;
  const index_type  __simd_end = this->__data_.size() - (this->__data_.size() % 4);

  if constexpr (std::is_floating_point_v<value_type>) {
    float32x4_t __val_vec = vdupq_n_f32(__val);

    for (; __i < __simd_end; __i += 4) {
      float32x4_t __data_vec   = vld1q_f32(reinterpret_cast<const float*>(&this->__data_[__i]));
      uint32x4_t  __cmp_result = vceqq_f32(__data_vec, __val_vec);

      uint32_t results[4];
      vst1q_u32(results, __cmp_result);  // Store results into an array
      for (int j = 0; j < 4; ++j) __ret[__i + j] = results[j] != 0;
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    int32x4_t __val_vec = vdupq_n_s32(__val);

    for (; __i < __simd_end; __i += 4) {
      int32x4_t  __data_vec   = vld1q_s32(reinterpret_cast<const int32_t*>(&this->__data_[__i]));
      uint32x4_t __cmp_result = vceqq_s32(__data_vec, __val_vec);

      uint32_t results[4];
      vst1q_u32(results, __cmp_result);
      for (int j = 0; j < 4; ++j) __ret[__i + j] = results[j] != 0;
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    uint32x4_t __val_vec = vdupq_n_u32(__val);

    for (; __i < __simd_end; __i += 4) {
      uint32x4_t __data_vec   = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
      uint32x4_t __cmp_result = vceqq_u32(__data_vec, __val_vec);

      uint32_t results[4];
      vst1q_u32(results, __cmp_result);
      for (int j = 0; j < 4; ++j) __ret[__i + j] = results[j] != 0;
    }
  }

#pragma omp parallel
  // Handle the remaining elements that don't fit in a SIMD register
  for (; __i < this->__data_.size(); ++__i) __ret[__i] = (this->__data_[__i] == __val);

  return tensor<bool>(this->__shape_, __ret);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::neon_less_equal(const tensor& __other) const {
  static_assert(has_less_operator_v<value_type>, "Value type must have a less than operator");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

  std::vector<_u32> __ret(this->__data_.size());
  size_t            __vs = this->__data_.size() / _ARM64_REG_WIDTH * _ARM64_REG_WIDTH;
  index_type        __i  = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < __vs; __i += _ARM64_REG_WIDTH) {
      neon_f32 __va       = vld1q_f32(this->__data_.data() + __i);
      neon_f32 __vb       = vld1q_f32(__other.__data_.data() + __i);
      neon_u32 __leq_mask = vcleq_f32(__va, __vb);
      vst1q_u32(&__ret[__i], __leq_mask);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < __vs; __i += _ARM64_REG_WIDTH) {
      neon_s32 __va       = vld1q_s32(this->__data_.data() + __i);
      neon_s32 __vb       = vld1q_s32(__other.__data_.data() + __i);
      neon_u32 __leq_mask = vcleq_s32(__va, __vb);
      vst1q_u32(&__ret[__i], __leq_mask);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < __vs; __i += _ARM64_REG_WIDTH) {
      neon_u32 __va       = vld1q_u32(this->__data_.data() + __i);
      neon_u32 __vb       = vld1q_u32(__other.__data_.data() + __i);
      neon_u32 __leq_mask = vcleq_u32(__va, __vb);
      vst1q_u32(&__ret[__i], __leq_mask);
    }
  }

  // Convert `__ret` (integer masks) to boolean
  std::vector<bool> __d(this->__data_.size());

#pragma omp parallel
  for (size_t __j = 0; __j < __i; ++__j) __d[__j] = __ret[__j] != 0;

#pragma omp parallel
  for (; __i < __d.size(); ++__i) __d[__i] = (this->__data_[__i] <= __other[__i]);

  return tensor<bool>(this->__shape_, __d);
}

template <class _Tp>
tensor<bool> tensor<_Tp>::neon_less_equal(const value_type __val) const {
  static_assert(has_less_equal_operator_v<value_type>,
                "Value type must have a less than or equal operator");

  std::vector<_u32> __ret(this->__data_.size());
  index_type        __i = 0;

  size_t vector_size = this->__data_.size() / _ARM64_REG_WIDTH * _ARM64_REG_WIDTH;

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i < vector_size; __i += _ARM64_REG_WIDTH) {
      neon_f32 vec_a    = vld1q_f32(this->__data_.data() + __i);
      neon_f32 vec_b    = vdupq_n_f32(__val);
      neon_u32 leq_mask = vcleq_f32(vec_a, vec_b);

      vst1q_u32(&__ret[__i], leq_mask);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; __i < vector_size; __i += _ARM64_REG_WIDTH) {
      neon_s32 vec_a    = vld1q_s32(this->__data_.data() + __i);
      neon_s32 vec_b    = vdupq_n_s32(__val);
      neon_u32 leq_mask = vcleq_s32(vec_a, vec_b);

      vst1q_u32(&__ret[__i], leq_mask);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; __i < vector_size; __i += _ARM64_REG_WIDTH) {
      neon_u32 vec_a    = vld1q_u32(this->__data_.data() + __i);
      neon_u32 vec_b    = vdupq_n_u32(__val);
      neon_u32 leq_mask = vcleq_u32(vec_a, vec_b);

      vst1q_u32(&__ret[__i], leq_mask);
    }
  }

#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) __ret[__i] = (this->__data_[__i] <= __val) ? 1 : 0;

  std::vector<bool> __to_bool(__ret.size());
  __i = 0;

#pragma omp parallel
  for (int i = __i; i >= 0; i--) __to_bool[i] = __ret[i] == 1 ? true : false;

  return tensor<bool>(__to_bool, this->__shape_);
}