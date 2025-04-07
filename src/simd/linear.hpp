#pragma once

#include "tensorbase.hpp"
#include "types.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::neon_absolute_(const tensor& tensor) const {
  if (!std::is_arithmetic_v<value_type>) {
    throw type_error("Type must be arithmetic");
  }

  if (std::is_unsigned_v<value_type>) {
    return *this;
  }

  index_type s = tensor.storage().size();
  data_t     a(s);
  index_type i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    for (; i < this->data_.size(); i += _ARM64_REG_WIDTH) {
      neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->data_[i]));
      neon_f32 abs_vec  = vabsq_f32(data_vec);
      vst1q_f32(reinterpret_cast<_f32*>(&this->data_[i]), abs_vec);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    for (; i < this->data_.size(); i += _ARM64_REG_WIDTH) {
      neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->data_[i]));
      neon_s32 abs_vec  = vabsq_s32(data_vec);
      vst1q_s32(reinterpret_cast<_s32*>(&this->data_[i]), abs_vec);
    }
  }

#pragma omp parallel
  for (; i < s; ++i) {
    this->data_[i] = static_cast<value_type>(std::abs(this->data_[i]));
  }

  return self(a, tensor.shape_);
}