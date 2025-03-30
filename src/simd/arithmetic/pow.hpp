#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_pow_(const value_type __val) {
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

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
  if (!std::is_arithmetic_v<value_type>) throw __type_error__("Type must be arithmetic");

  if (!__equal_shape(this->shape(), __other.shape()))
    throw __shape_error__("Tensors shapes must be equal");

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_type       __i        = 0;

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

#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(
        std::pow(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));

  return *this;
}