#pragma once

#include "alias.hpp"
#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
_u64 count_nonzero(const tensor<_Tp>& t, _u64 dimension) {
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  std::vector<_Tp>& data_       = t.storage_();
  _u64              c           = 0;
  _u64              local_count = 0;
  _u64              i           = 0;
  const _u64        simd_end    = data_.size() - (data_.size() % t.simd_width);

  if (dimension == 0)
  {
    for (; i < simd_end; i += t.simd_width)
    {
      neon_type<_Tp> vec           = neon_load<_Tp>(&data_[i]);
      neon_type<_Tp> zero_vec      = neon_dup<_Tp>(_Tp(0.0f));
      neon_type<_Tp> one_vec       = neon_dup<_Tp>(_Tp(0.0f));
      neon_u32       nonzero_mask  = neon_cgt<_Tp>(vec, zero_vec);
      neon_type<_Tp> nonzero_mask_ = reinterpret_cast<neon_type<_Tp>>(nonzero_mask);
      neon_type<_Tp> temp          = neon_and<_Tp>(nonzero_mask_, one_vec);
      local_count += neon_addv<_Tp>(temp);
    }

    for (_u64 j = i; j < data_.size(); ++j)
    {
      if (data_[j] != 0)
      {
        ++local_count;
      }
    }

    c += local_count;
  }
  else
  {
    if (dimension < 0 || dimension >= static_cast<_u64>(t.shape().size()))
    {
      throw error::index_error("Invalid dimension provided.");
    }

    throw std::runtime_error("Dimension-specific non-zero counting is not implemented yet.");
  }

  return c;
}

template<class _Tp>
tensor<_Tp>& zeros_(tensor<_Tp>& t, shape::Shape shape_) {
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  if (shape_.empty())
  {
    shape_ = shape_;
  }
  else
  {
    shape_ = shape_;
  }

  std::vector<_Tp>& data_ = t.storage_();
  std::size_t       s     = shape_.flatten_size();
  data_.resize(s);
  t.shape().compute_strides();
  const _u64     simd_end = s - (s % t.simd_width);
  neon_type<_Tp> zero_vec = neon_dup<_Tp>(_Tp(0.0f));
  _u64           i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_store<_Tp>(&data_[i], zero_vec);
  }

  for (; i < s; ++i)
  {
    data_[i] = _Tp(0.0);
  }

  return t;
}

template<class _Tp>
tensor<_Tp>& ones_(tensor<_Tp>& t, shape::Shape shape_) {
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  if (shape_.empty())
  {
    shape_ = shape_;
  }
  else
  {
    shape_ = shape_;
  }

  std::vector<_Tp>& data_ = t.storage_();
  std::size_t       s     = t.shape_.flatten_size();
  data_.resize(s);
  t.shape_.compute_strides();
  const _u64     simd_end = s - (s % t.simd_width);
  neon_type<_Tp> one_vec  = neon_dup<_Tp>(1.0f);
  _u64           i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_store<_Tp>(&data_[i], one_vec);
  }

  for (; i < s; ++i)
  {
    data_[i] = _Tp(1.0);
  }

  return t;
}

template<class _Tp>
tensor<_Tp>& randomize_(tensor<_Tp>& t, const shape::Shape& sh, bool bounded) {
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  if (bounded && !std::is_floating_point_v<_Tp>)
  {
    throw error::type_error("Cannot bound non floating point data type");
  }

  if (t.shape_.empty() && sh.empty())
  {
    throw error::shape_error("randomize_ : Shape must be initialized");
  }

  if (t.shape_.empty() || !sh.equal(t.shape_))
  {
    t.shape_ = sh;
  }

  std::vector<_Tp>& data_ = t.storage_();
  _u64              s     = t.shape_.flatten_size();
  data_.resize(s);
  t.shape_.compute_strides();
  std::random_device                   rd;
  std::mt19937                         gen(rd());
  std::uniform_real_distribution<_f32> unbounded_dist(1.0f, static_cast<_f32>(RAND_MAX));
  std::uniform_real_distribution<_f32> bounded_dist(0.0f, 1.0f);
  _u64                                 i = 0;

  if constexpr (std::is_floating_point_v<_Tp>)
  {
    const neon_f32 scale = vdupq_n_f32(bounded ? static_cast<_f32>(RAND_MAX) : 1.0f);

    for (; i + _ARM64_REG_WIDTH <= static_cast<_u64>(s); i += _ARM64_REG_WIDTH)
    {
      neon_f32 random_values;

      if (bounded)
      {
        random_values = {bounded_dist(gen), bounded_dist(gen), bounded_dist(gen), bounded_dist(gen)};
      }
      else
      {
        random_values = {unbounded_dist(gen), unbounded_dist(gen), unbounded_dist(gen), unbounded_dist(gen)};
      }

      if (!bounded)
      {
        random_values = vmulq_f32(random_values, vrecpeq_f32(scale));
      }

      vst1q_f32(&data_[i], random_values);
    }
  }
  else if constexpr (std::is_unsigned_v<_Tp>)
  {
    const neon_f32 scale = vdupq_n_f32(static_cast<_f32>(RAND_MAX));

    for (; i + t.simd_width <= static_cast<_u64>(s); i += t.simd_width)
    {
      neon_f32 rand_vals = {static_cast<_f32>(unbounded_dist(gen)), static_cast<_f32>(unbounded_dist(gen)),
                            static_cast<_f32>(unbounded_dist(gen)), static_cast<_f32>(unbounded_dist(gen))};
      rand_vals          = vmulq_f32(rand_vals, vrecpeq_f32(scale));
      neon_u32 int_vals  = vcvtq_u32_f32(rand_vals);
      vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), int_vals);
    }
  }
  else if constexpr (std::is_signed_v<_Tp>)
  {
    const neon_f32 scale = vdupq_n_f32(static_cast<_f32>(RAND_MAX));

    for (; i + t.simd_width <= static_cast<_u64>(s); i += t.simd_width)
    {
      neon_f32 rand_vals = {static_cast<_f32>(unbounded_dist(gen)), static_cast<_f32>(unbounded_dist(gen)),
                            static_cast<_f32>(unbounded_dist(gen)), static_cast<_f32>(unbounded_dist(gen))};
      rand_vals          = vmulq_f32(rand_vals, vrecpeq_f32(scale));
      neon_s32 int_vals  = vcvtq_s32_f32(rand_vals);
      vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), int_vals);
    }
  }

  for (; i < static_cast<_u64>(s); ++i)
  {
    data_[i] = _Tp(bounded ? bounded_dist(gen) : unbounded_dist(gen));
  }

  return t;
}

template<class _Tp>
tensor<_Tp>& negative_(tensor<_Tp>& t) {
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  std::vector<_Tp>& data_    = t.storage_();
  const _u64        simd_end = data_.size() - (data_.size() % t.simd_width);
  _u64              i        = 0;

  if constexpr (std::is_same_v<_Tp, _f32>)
  {
    neon_f32 neg_multiplier = vdupq_n_f32(-1);

    for (; i < simd_end; i += t.simd_width)
    {
      neon_f32 v   = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
      neon_f32 neg = vmulq_f32(v, neg_multiplier);
      vst1q_f32(reinterpret_cast<_f32*>(&data_[i]), neg);
    }
  }
  else if constexpr (std::is_same_v<_Tp, _s32>)
  {
    neon_s32 neg_multiplier = vdupq_n_s32(-1);

    for (; i < simd_end; i += t.simd_width)
    {
      neon_s32 v   = vld1q_s32(reinterpret_cast<const _s32*>(&data_[i]));
      neon_s32 neg = vmulq_s32(v, neg_multiplier);
      vst1q_s32(reinterpret_cast<_s32*>(&data_[i]), neg);
    }
  }
  else if constexpr (std::is_same_v<_Tp, _u32>)
  {
    neon_s32 neg_multiplier = vdupq_n_s32(-1);

    for (; i < simd_end; i += t.simd_width)
    {
      neon_u32 v   = vld1q_u32(reinterpret_cast<const _u32*>(&data_[i]));
      neon_u32 neg = vmulq_u32(v, neg_multiplier);
      vst1q_u32(reinterpret_cast<_u32*>(&data_[i]), neg);
    }
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = -data_[i];
  }

  return t;
}

template<class _Tp>
tensor<_Tp>& fill_(tensor<_Tp>& t, const _Tp& value) {
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  std::vector<_Tp>& data_ = t.storage_();
  const _u64        s     = t.shape().flatten_size();
  data_.resize(s);
  t.shape_().compute_strides();
  const _u64     simd_end = s - (s % t.simd_width);
  neon_type<_Tp> fill_vec = neon_dup<_Tp>(value);
  _u64           i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_store<_Tp>(&data_[i], fill_vec);
  }

  for (; i < s; ++i)
  {
    data_[i] = value;
  }

  return t;
}

template<class _Tp>
tensor<_Tp>& fill_(tensor<_Tp>& t, const tensor<_Tp>& other) {
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  if (!t.shape().equal(other.shape()))
  {
    throw error::shape_error("Shapes of tensors must match for fill operation");
  }

  std::vector<_Tp>& data_ = t.storage_();
  const _u64        s     = t.shape().flatten_size();
  data_.resize(s);
  t.shape_().compute_strides();
  const _u64 simd_end = s - (s % t.simd_width);
  _u64       i        = 0;

  for (; i < simd_end; i += t.simd_width)
  {
    neon_type<_Tp> vec = neon_load<_Tp>(&other[i]);
    neon_store<_Tp>(&data_[i], vec);
  }

  for (; i < s; ++i)
  {
    data_[i] = other[i];
  }

  return t;
}

}