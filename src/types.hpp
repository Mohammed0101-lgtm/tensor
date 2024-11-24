#pragma once

#include "tensorbase.hpp"


template<class _Tp>
tensor<_s32> tensor<_Tp>::int32_() const {
  static_assert(std::is_convertible<value_type, _s32>::value);

  data_t     __d;
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_s32 __int_vec  = vcvtq_s32_f32(__data_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&__d[__i]), __int_vec);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_s32 __int_vec  = vreinterpretq_s32_u32(__data_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&__d[__i]), __int_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __d.push_back(static_cast<_s32>(this->__data_[__i]));
  }

  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<_u32> tensor<_Tp>::uint32_() const {
  static_assert(std::is_convertible<value_type, _u32>::value);

  data_t     __d;
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() - _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_u32 __uint_vec = vcvtq_u32_f32(__data_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&__d[__i]), __uint_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_u32 __uint_vec = vreinterpretq_u32_s32(__data_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&__d[__i]), __uint_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i += _ARM64_REG_WIDTH)
  {
    __d.push_back(static_cast<_u32>(this->__data_[__i]));
  }

  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<_f32> tensor<_Tp>::float32_() const {
  static_assert(std::is_convertible<value_type, _f32>::value, "Tensor value type must be convertible to _f32.");

  if (this->empty())
  {
    return __self({}, this->__shape_);
  }

  data_t     __d(this->__data_.size());
  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_same_v<value_type, _f64>)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % (_ARM64_REG_WIDTH / 2));

    for (; __i < __simd_end; __i += (_ARM64_REG_WIDTH / 2))
    {
      neon_f64    __data_vec1          = vld1q_f64(reinterpret_cast<const _f64*>(&this->__data_[__i]));
      neon_f64    __data_vec2          = vld1q_f64(reinterpret_cast<const _f64*>(&this->__data_[__i + 2]));
      float32x2_t __float_vec1         = vcvt_f32_f64(__data_vec1);
      float32x2_t __float_vec2         = vcvt_f32_f64(__data_vec2);
      neon_f32    __float_vec_combined = vcombine_f32(__float_vec1, __float_vec2);

      vst1q_f32(reinterpret_cast<_f32*>(&__d[__i]), __float_vec_combined);
    }
  }
  else if constexpr (std::is_same_v<value_type, _s32>)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_f32 __float_vec = vcvtq_f32_s32(__data_vec);

      vst1q_f32(reinterpret_cast<_f32*>(&__d[__i]), __float_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __d[__i] = static_cast<_f32>(this->__data_[__i]);
  }

  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<_f64> tensor<_Tp>::double_() const {
  static_assert(std::is_convertible<value_type, _f64>::value, "Tensor value type must be convertible to _f64.");

  if (this->empty())
  {
    return __self({}, this->__shape_);
  }

  data_t     __d(this->__data_.size());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    auto __data_vec = vld1q_f64(reinterpret_cast<const double*>(&this->__data_[__i]));
    vst1q_f64(reinterpret_cast<_f64*>(&__d[__i]), __data_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __d[__i] = static_cast<_f64>(this->__data_[__i]);
  }

  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<uint64_t> tensor<_Tp>::unsigned_long_() const {
  static_assert(std::is_convertible<value_type, uint64_t>::value, "Tensor value type must be convertible to uint64_t.");

  if (this->empty())
  {
    return __self({}, this->__shape_);
  }

  data_t     __d(this->__data_.size());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32   __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      uint64x2_t __int_vec1 = vmovl_u32(vget_low_u32(__data_vec));
      uint64x2_t __int_vec2 = vmovl_u32(vget_high_u32(__data_vec));

      vst1q_u64(reinterpret_cast<uint64_t*>(&__d[__i]), __int_vec1);
      vst1q_u64(reinterpret_cast<uint64_t*>(&__d[__i + 2]), __int_vec2);
    }
  }
  else
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f64 __data_vec = vld1q_f64(reinterpret_cast<const _f64*>(&this->__data_[__i]));
      neon_f64 __uint_vec = vcvtq_u64_f64(__data_vec);

      vst1q_u64(reinterpret_cast<uint64_t*>(&__d[__i]), __uint_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __d[__i] = static_cast<uint64_t>(this->__data_[__i]);
  }

  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<int64_t> tensor<_Tp>::long_() const {
  static_assert(std::is_convertible<value_type, int64_t>::value, "Tensor value type must be convertible to int64_t.");

  if (this->empty())
  {
    return __self({}, this->__shape_);
  }

  data_t     __d(this->__data_.size());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f64  __data_vec1 = vld1q_f64(reinterpret_cast<const _f64*>(&this->__data_[__i]));
      neon_f64  __data_vec2 = vld1q_f64(reinterpret_cast<const _f64*>(&this->__data_[__i + 2]));
      int64x2_t __int_vec1  = vcvtq_s64_f64(__data_vec1);
      int64x2_t __int_vec2  = vcvtq_s64_f64(__data_vec2);

      vst1q_s64(reinterpret_cast<int64_t*>(&__d[__i]), __int_vec1);
      vst1q_s64(reinterpret_cast<int64_t*>(&__d[__i + 2]), __int_vec2);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32   __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      uint64x2_t __int_vec1 = vmovl_u32(vget_low_u32(__data_vec));
      uint64x2_t __int_vec2 = vmovl_u32(vget_high_u32(__data_vec));

      vst1q_u64(reinterpret_cast<uint64_t*>(&__d[__i]), __int_vec1);
      vst1q_u64(reinterpret_cast<uint64_t*>(&__d[__i + 2]), __int_vec2);
    }
  }
  else
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32  __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      int64x2_t __int_vec1 = vmovl_s32(vget_low_s32(__data_vec));
      int64x2_t __int_vec2 = vmovl_s32(vget_high_s32(__data_vec));

      vst1q_s64(reinterpret_cast<int64_t*>(&__d[__i]), __int_vec1);
      vst1q_s64(reinterpret_cast<int64_t*>(&__d[__i + 2]), __int_vec2);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __d[__i] = static_cast<int64_t>(this->__data_[__i]);
  }

  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::bool_() const {
  std::vector<bool> __d;

  static_assert(std::is_convertible<value_t, bool>::value);

  for (index_type __i = 0; __i < this->__data_.size(); __i++)
  {
    __d.push_back(bool(this->__data_[__i]));
  }

  return tensor<bool>(__d, this->__shape_);
}