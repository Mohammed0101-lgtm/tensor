#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_pow_(const value_type val) {
  if (!std::is_arithmetic_v<value_type>) {
    throw type_error("Type must be arithmetic");
  }

  index_type       i        = 0;
  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->data_[i]));
      _f32     vals[_ARM64_REG_WIDTH];
      vst1q_f32(vals, data_vec);

      vals[0] = static_cast<_f32>(std::pow(static_cast<_f32>(vals[0]), static_cast<_f32>(val)));
      vals[1] = static_cast<_f32>(std::pow(static_cast<_f32>(vals[1]), static_cast<_f32>(val)));
      vals[2] = static_cast<_f32>(std::pow(static_cast<_f32>(vals[2]), static_cast<_f32>(val)));
      vals[3] = static_cast<_f32>(std::pow(static_cast<_f32>(vals[3]), static_cast<_f32>(val)));

      neon_f32 pow_vec = vld1q_f32(vals);
      vst1q_f32(&this->data_[i], pow_vec);
    }
  }

  for (; i < this->data_.size(); ++i) {
    this->data_[i] = static_cast<value_type>(std::pow(this->data_[i], val));
  }

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_pow_(const tensor& other) {
  if (!std::is_arithmetic_v<value_type>) {
    throw type_error("Type must be arithmetic");
  }

  if (!equal_shape(this->shape(), other.shape())) {
    throw shape_error("Tensors shapes must be equal");
  }

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);
  index_type       i        = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    neon_f32 base_vec   = vld1q_f32(reinterpret_cast<const _f32*>(&this->data_[i]));
    neon_f32 exp_vec    = vld1q_f32(reinterpret_cast<const _f32*>(&other[i]));
    neon_f32 result_vec = {
        static_cast<_f32>(std::pow(static_cast<_f32>(vgetq_lane_f32(base_vec, 0)),
                                   static_cast<_f32>(vgetq_lane_f32(exp_vec, 0)))),
        static_cast<_f32>(std::pow(static_cast<_f32>(vgetq_lane_f32(base_vec, 1)),
                                   static_cast<_f32>(vgetq_lane_f32(exp_vec, 1)))),
        static_cast<_f32>(std::pow(static_cast<_f32>(vgetq_lane_f32(base_vec, 2)),
                                   static_cast<_f32>(vgetq_lane_f32(exp_vec, 2)))),
        static_cast<_f32>(std::pow(static_cast<_f32>(vgetq_lane_f32(base_vec, 3)),
                                   static_cast<_f32>(vgetq_lane_f32(exp_vec, 3))))};
    vst1q_f32(&this->data_[i], result_vec);
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    neon_s32 base_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->data_[i]));
    neon_s32 exp_vec    = vld1q_s32(reinterpret_cast<const _s32*>(&other[i]));
    neon_s32 result_vec = {
        static_cast<_s32>(std::pow(static_cast<_s32>(vgetq_lane_s32(base_vec, 0)),
                                   static_cast<_s32>(vgetq_lane_s32(exp_vec, 0)))),
        static_cast<_s32>(std::pow(static_cast<_s32>(vgetq_lane_s32(base_vec, 1)),
                                   static_cast<_s32>(vgetq_lane_s32(exp_vec, 1)))),
        static_cast<_s32>(std::pow(static_cast<_s32>(vgetq_lane_s32(base_vec, 2)),
                                   static_cast<_s32>(vgetq_lane_s32(exp_vec, 2)))),
        static_cast<_s32>(std::pow(static_cast<_s32>(vgetq_lane_s32(base_vec, 3)),
                                   static_cast<_s32>(vgetq_lane_s32(exp_vec, 3))))};

    vst1q_s32(&this->data_[i], result_vec);
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    neon_u32 base_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->data_[i]));
    neon_u32 exp_vec    = vld1q_u32(reinterpret_cast<const _u32*>(&other[i]));
    neon_u32 result_vec = {
        static_cast<_u32>(std::pow(static_cast<_u32>(vgetq_lane_u32(base_vec, 0)),
                                   static_cast<_u32>(vgetq_lane_u32(exp_vec, 0)))),
        static_cast<_u32>(std::pow(static_cast<_u32>(vgetq_lane_u32(base_vec, 1)),
                                   static_cast<_u32>(vgetq_lane_u32(exp_vec, 1)))),
        static_cast<_u32>(std::pow(static_cast<_u32>(vgetq_lane_u32(base_vec, 2)),
                                   static_cast<_u32>(vgetq_lane_u32(exp_vec, 2)))),
        static_cast<_u32>(std::pow(static_cast<_u32>(vgetq_lane_u32(base_vec, 3)),
                                   static_cast<_u32>(vgetq_lane_u32(exp_vec, 3))))};

    vst1q_u32(&this->data_[i], result_vec);
  }

#pragma omp parallel
  for (; i < this->data_.size(); ++i) {
    this->data_[i] = static_cast<value_type>(
        std::pow(static_cast<_f32>(this->data_[i]), static_cast<_f32>(other[i])));
  }

  return *this;
}