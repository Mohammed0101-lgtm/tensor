#pragma once

#include "tensorbase.hpp"

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_log_() {
  if (!std::is_arithmetic_v<value_type>) {
    throw type_error("Type must be arithmetic");
  }

  index_type i = 0;

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->data_[i]));
      _f32     vals[_ARM64_REG_WIDTH];
      vst1q_f32(vals, data_vec);

      vals[0] = static_cast<_f32>(std::log(static_cast<_f32>(vals[0])));
      vals[1] = static_cast<_f32>(std::log(static_cast<_f32>(vals[1])));
      vals[2] = static_cast<_f32>(std::log(static_cast<_f32>(vals[2])));
      vals[3] = static_cast<_f32>(std::log(static_cast<_f32>(vals[3])));

      neon_f32 log_vec = vld1q_f32(vals);
      vst1q_f32(&this->data_[i], log_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->data_[i]));
      _u32     vals[_ARM64_REG_WIDTH];
      vst1q_u32(vals, data_vec);

      vals[0] = static_cast<_u32>(std::log(static_cast<_u32>(vals[0])));
      vals[1] = static_cast<_u32>(std::log(static_cast<_u32>(vals[1])));
      vals[2] = static_cast<_u32>(std::log(static_cast<_u32>(vals[2])));
      vals[3] = static_cast<_u32>(std::log(static_cast<_u32>(vals[3])));

      neon_u32 log_vec = vld1q_u32(vals);
      vst1q_u32(&this->data_[i], log_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->data_[i]));
      _s32     vals[_ARM64_REG_WIDTH];
      vst1q_s32(vals, data_vec);

      vals[0] = static_cast<_s32>(std::log(static_cast<_s32>(vals[0])));
      vals[1] = static_cast<_s32>(std::log(static_cast<_s32>(vals[1])));
      vals[2] = static_cast<_s32>(std::log(static_cast<_s32>(vals[2])));
      vals[3] = static_cast<_s32>(std::log(static_cast<_s32>(vals[3])));

      neon_s32 log_vec = vld1q_s32(vals);
      vst1q_s32(&this->data_[i], log_vec);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) {
    this->data_[i] = static_cast<value_type>(std::log(this->data_[i]));
  }

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_log10_() {
  if (!std::is_arithmetic_v<value_type>) {
    throw type_error("Type must be arithmetic");
  }

  index_type i = 0;

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->data_[i]));
      _f32     vals[_ARM64_REG_WIDTH];
      vst1q_f32(vals, data_vec);

      vals[0] = static_cast<_f32>(std::log10(vals[0]));
      vals[1] = static_cast<_f32>(std::log10(vals[1]));
      vals[2] = static_cast<_f32>(std::log10(vals[2]));
      vals[3] = static_cast<_f32>(std::log10(vals[3]));

      neon_f32 log_vec = vld1q_f32(vals);
      vst1q_f32(&this->data_[i], log_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->data_[i]));
      _u32     vals[_ARM64_REG_WIDTH];
      vst1q_u32(vals, data_vec);

      vals[0] = static_cast<_u32>(std::log10(vals[0]));
      vals[1] = static_cast<_u32>(std::log10(vals[1]));
      vals[2] = static_cast<_u32>(std::log10(vals[2]));
      vals[3] = static_cast<_u32>(std::log10(vals[3]));

      neon_u32 log_vec = vld1q_u32(vals);
      vst1q_u32(&this->data_[i], log_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->data_[i]));
      _s32     vals[_ARM64_REG_WIDTH];
      vst1q_s32(vals, data_vec);

      vals[0] = static_cast<_s32>(std::log10(vals[0]));
      vals[1] = static_cast<_s32>(std::log10(vals[1]));
      vals[2] = static_cast<_s32>(std::log10(vals[2]));
      vals[3] = static_cast<_s32>(std::log10(vals[3]));

      neon_s32 log_vec = vld1q_s32(vals);
      vst1q_s32(&this->data_[i], log_vec);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i)
    this->data_[i] = static_cast<value_type>(std::log10(this->data_[i]));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_log2_() {
  if (!std::is_arithmetic_v<value_type>) {
    throw type_error("Type must be arithmetic");
  }

  index_type i = 0;

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->data_[i]));
      _f32     vals[_ARM64_REG_WIDTH];
      vst1q_f32(vals, data_vec);

      vals[0] = static_cast<_f32>(std::log2(vals[0]));
      vals[1] = static_cast<_f32>(std::log2(vals[1]));
      vals[2] = static_cast<_f32>(std::log2(vals[2]));
      vals[3] = static_cast<_f32>(std::log2(vals[3]));

      neon_f32 log2_vec = vld1q_f32(vals);
      vst1q_f32(&this->data_[i], log2_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->data_[i]));
      _u32     vals[_ARM64_REG_WIDTH];
      vst1q_u32(vals, data_vec);

      vals[0] = static_cast<_u32>(std::log2(vals[0]));
      vals[1] = static_cast<_u32>(std::log2(vals[1]));
      vals[2] = static_cast<_u32>(std::log2(vals[2]));
      vals[3] = static_cast<_u32>(std::log2(vals[3]));

      neon_u32 log2_vec = vld1q_u32(vals);
      vst1q_u32(&this->data_[i], log2_vec);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->data_[i]));
      _s32     vals[_ARM64_REG_WIDTH];
      vst1q_s32(vals, data_vec);

      vals[0] = static_cast<_s32>(std::log2(vals[0]));
      vals[1] = static_cast<_s32>(std::log2(vals[1]));
      vals[2] = static_cast<_s32>(std::log2(vals[2]));
      vals[3] = static_cast<_s32>(std::log2(vals[3]));

      neon_s32 log2_vec = vld1q_s32(vals);
      vst1q_s32(&this->data_[i], log2_vec);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i)
    this->data_[i] = static_cast<value_type>(std::log2(this->data_[i]));

  return *this;
}
