#pragma once

#include "tensorbase.hpp"

template <class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::neon_count_nonzero(index_type dim) const {
  if (!std::is_arithmetic_v<value_type>) {
    throw type_error("Type must be arithmetic");
  }

  index_type c           = 0;
  index_type local_count = 0;
  index_type i           = 0;
  if (dim == 0) {
    if constexpr (std::is_floating_point_v<value_type>) {
      for (; i < this->data_.size(); i += _ARM64_REG_WIDTH) {
        neon_f32 vec          = vld1q_f32(reinterpret_cast<const _f32*>(&this->data_[i]));
        neon_u32 nonzero_mask = vcgtq_f32(vec, vdupq_n_f32(0.0f));
        local_count += vaddvq_u32(vandq_u32(nonzero_mask, vdupq_n_u32(1)));
      }
    } else if constexpr (std::is_unsigned_v<value_type>) {
      for (; i < this->data_.size(); i += _ARM64_REG_WIDTH) {
        neon_u32 vec          = vld1q_u32(reinterpret_cast<const _u32*>(&this->data_[i]));
        neon_u32 nonzero_mask = vcgtq_u32(vec, vdupq_n_u32(0));
        local_count += vaddvq_u32(vandq_u32(nonzero_mask, vdupq_n_u32(1)));
      }
    } else if constexpr (std::is_signed_v<value_type>) {
      for (; i < this->data_.size(); i += _ARM64_REG_WIDTH) {
        neon_s32 vec          = vld1q_s32(reinterpret_cast<const _s32*>(&this->data_[i]));
        neon_u32 nonzero_mask = vcgtq_s32(vec, vdupq_n_s32(0));
        local_count += vaddvq_u32(vandq_u32(nonzero_mask, vdupq_n_u32(1)));
      }
    }
#pragma omp parallel for reduction(+ : local_count)
    for (index_type j = i; j < this->data_.size(); ++j)
      if (this->data_[j] != 0) {
        ++local_count;
      }

    c += local_count;
  } else {
    if (dim < 0 or dim >= static_cast<index_type>(shape_.size())) {
      throw index_error("Invalid dimension provided.");
    }

    throw std::runtime_error("Dimension-specific non-zero counting is not implemented yet.");
  }

  return c;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_zeros_(shape_type sh) {
  if (!std::is_arithmetic_v<value_type>) {
    throw type_error("Type must be arithmetic");
  }

  if (sh.empty()) {
    sh = this->shape_;
  } else {
    this->shape_ = sh;
  }

  size_t s = this->computeSize(this->shape_);
  this->data_.resize(s);
  this->compute_strides();
  const index_type simd_end = s - (s % _ARM64_REG_WIDTH);
  index_type       i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    neon_f32 zero_vec = vdupq_n_f32(0.0f);
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      vst1q_f32(&this->data_[i], zero_vec);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    neon_s32 zero_vec = vdupq_n_s32(0);
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      vst1q_s32(&this->data_[i], zero_vec);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 zero_vec = vdupq_n_u32(0);
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      vst1q_u32(&this->data_[i], zero_vec);
    }
  }
#pragma omp parallel
  for (; i < s; ++i) {
    this->data_[i] = value_type(0.0);
  }

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_ones_(shape_type sh) {
  if (!std::is_arithmetic_v<value_type>) {
    throw type_error("Type must be arithmetic");
  }

  if (sh.empty()) {
    sh = this->shape_;
  } else {
    this->shape_ = sh;
  }

  size_t s = this->computeSize(this->shape_);
  this->data_.resize(s);
  this->compute_strides();
  const index_type simd_end = s - (s % _ARM64_REG_WIDTH);
  index_type       i        = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    neon_f32 one_vec = vdupq_n_f32(1.0f);

    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      vst1q_f32(reinterpret_cast<_f32*>(&this->data_[i]), one_vec);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    neon_s32 one_vec = vdupq_n_s32(1);

    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      vst1q_s32(reinterpret_cast<_s32*>(&this->data_[i]), one_vec);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    neon_u32 one_vec = vdupq_n_u32(1);

    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      vst1q_u32(reinterpret_cast<_u32*>(&this->data_[i]), one_vec);
    }
  }
#pragma omp parallel
  for (; i < s; ++i) {
    this->data_[i] = value_type(1.0);
  }

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_randomize_(const shape_type& sh, bool bounded) {
  if (!std::is_arithmetic_v<value_type>) {
    throw type_error("Type must be arithmetic");
  }

  if (bounded) {
    assert(std::is_floating_point_v<value_type> and "Cannot bound non floating point data type");
  }

  if (sh.empty() and this->shape_.empty()) {
    throw shape_error("randomize_ : Shape must be initialized");
  }

  if (this->shape_.empty() or this->shape_ != sh) {
    this->shape_ = sh;
  }

  index_type s = this->computeSize(this->shape_);
  this->data_.resize(s);
  this->compute_strides();

  std::random_device                   rd;
  std::mt19937                         gen(rd());
  std::uniform_real_distribution<_f32> unbounded_dist(1.0f, static_cast<_f32>(RAND_MAX));
  std::uniform_real_distribution<_f32> bounded_dist(0.0f, 1.0f);
  index_type                           i = 0;

  if constexpr (std::is_floating_point_v<value_type>) {
    const neon_f32 scale = vdupq_n_f32(bounded ? static_cast<_f32>(RAND_MAX) : 1.0f);
    for (; i + _ARM64_REG_WIDTH <= static_cast<index_type>(s); i += _ARM64_REG_WIDTH) {
      neon_f32 random_values;

      if (bounded) {
        random_values = {bounded_dist(gen), bounded_dist(gen), bounded_dist(gen),
                         bounded_dist(gen)};
      } else {
        random_values = {unbounded_dist(gen), unbounded_dist(gen), unbounded_dist(gen),
                         unbounded_dist(gen)};
      }

      if (!bounded) {
        random_values = vmulq_f32(random_values, vrecpeq_f32(scale));
      }

      vst1q_f32(&this->data_[i], random_values);
    }
  } else if constexpr (std::is_unsigned_v<value_type>) {
    const neon_f32 scale = vdupq_n_f32(static_cast<_f32>(RAND_MAX));
    for (; i + _ARM64_REG_WIDTH <= static_cast<index_type>(s); i += _ARM64_REG_WIDTH) {
      neon_f32 rand_vals = {
          static_cast<_f32>(unbounded_dist(gen)), static_cast<_f32>(unbounded_dist(gen)),
          static_cast<_f32>(unbounded_dist(gen)), static_cast<_f32>(unbounded_dist(gen))};
      rand_vals         = vmulq_f32(rand_vals, vrecpeq_f32(scale));
      neon_u32 int_vals = vcvtq_u32_f32(rand_vals);

      vst1q_u32(reinterpret_cast<_u32*>(&this->data_[i]), int_vals);
    }
  } else if constexpr (std::is_signed_v<value_type>) {
    const neon_f32 scale = vdupq_n_f32(static_cast<_f32>(RAND_MAX));
    for (; i + _ARM64_REG_WIDTH <= static_cast<index_type>(s); i += _ARM64_REG_WIDTH) {
      neon_f32 rand_vals = {
          static_cast<_f32>(unbounded_dist(gen)), static_cast<_f32>(unbounded_dist(gen)),
          static_cast<_f32>(unbounded_dist(gen)), static_cast<_f32>(unbounded_dist(gen))};
      rand_vals         = vmulq_f32(rand_vals, vrecpeq_f32(scale));
      neon_s32 int_vals = vcvtq_s32_f32(rand_vals);

      vst1q_s32(reinterpret_cast<_s32*>(&this->data_[i]), int_vals);
    }
  }
#pragma omp parallel
  for (; i < static_cast<index_type>(s); ++i) {
    this->data_[i] = value_type(bounded ? bounded_dist(gen) : unbounded_dist(gen));
  }

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_negative_() {
  if (!std::is_arithmetic_v<value_type>) {
    throw type_error("Type must be arithmetic");
  }

  const index_type simd_end = this->data_.size() - (this->data_.size() % _ARM64_REG_WIDTH);
  index_type       i        = 0;

  if constexpr (std::is_same_v<value_type, _f32>) {
    neon_f32 neg_multiplier = vdupq_n_f32(-1);
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_f32 v   = vld1q_f32(reinterpret_cast<const _f32*>(&this->data_[i]));
      neon_f32 neg = vmulq_f32(v, neg_multiplier);
      vst1q_f32(reinterpret_cast<_f32*>(&this->data_[i]), neg);
    }
  } else if constexpr (std::is_same_v<value_type, _s32>) {
    neon_s32 neg_multiplier = vdupq_n_s32(-1);
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_s32 v   = vld1q_s32(reinterpret_cast<const _s32*>(&this->data_[i]));
      neon_s32 neg = vmulq_s32(v, neg_multiplier);
      vst1q_s32(reinterpret_cast<_s32*>(&this->data_[i]), neg);
    }
  } else if constexpr (std::is_same_v<value_type, _u32>) {
    neon_s32 neg_multiplier = vdupq_n_s32(-1);
    for (; i < simd_end; i += _ARM64_REG_WIDTH) {
      neon_u32 v   = vld1q_u32(reinterpret_cast<const _u32*>(&this->data_[i]));
      neon_u32 neg = vmulq_u32(v, neg_multiplier);
      vst1q_u32(reinterpret_cast<_u32*>(&this->data_[i]), neg);
    }
  }
#pragma omp parallel
  for (; i < this->data_.size(); ++i) {
    this->data_[i] = -this->data_[i];
  }

  return *this;
}