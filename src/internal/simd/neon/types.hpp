#pragma once

#include "tensor.hpp"


namespace internal::simd::neon {

template<class _Tp>
tensor<_s16> int16_(const tensor<_Tp>& t) {
  if (!std::is_convertible_v<_Tp, _s16>)
  {
    throw error::type_error("Type must be convertible to 16 bit signed int");
  }

  if (t.empty())
  {
    return tensor<_s16>(t.shape());
  }

  std::vector<_Tp>& a    = t.storage_();
  std::size_t       size = a.size();
  std::vector<_s16> ret(size);
  const _u64        simd_end = size - (size % t.simd_width);

  _s16* __restrict r_ptr      = ret.data();
  const _Tp* __restrict a_ptr = a.data();

  if constexpr (std::is_floating_point_v<_Tp>)
  {
    for (std::size_t i = 0; i < simd_end; i += t.simd_width)
    {
      neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(a_ptr + i));
      neon_s16 int_vec  = vcvtq_f16_s16(data_vec);
      vst1q_s16(reinterpret_cast<_s16*>(r_ptr + i), int_vec);
    }
  }
  else if constexpr (std::is_unsigned_v<_Tp>)
  {
    for (std::size_t i = 0; i < simd_end; i += t.simd_width)
    {
      neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(a_ptr + i));
      neon_s16 int_vec  = vreinterpretq_s16_u32(data_vec);
      vst1q_s16(reinterpret_cast<_s16*>(r_ptr + i), int_vec);
    }
  }

  for (std::size_t i = simd_end; i < size; ++i)
  {
    r_ptr[i] = static_cast<_s16>(a_ptr[i]);
  }

  return tensor<_s16>(t.shape(), std::move(ret));
}

template<class _Tp>
tensor<_s32> int32_(const tensor<_Tp>& t) {
  if (!std::is_convertible_v<_Tp, _s32>)
  {
    throw error::type_error("Type must be convertible to 32 bit signed int");
  }

  if (t.empty())
  {
    return tensor<_s32>(t.shape());
  }

  std::vector<_Tp>& a    = t.storage_();
  std::size_t       size = a.size();
  std::vector<_s32> ret(size);
  const _u64        simd_end = size - (size % t.simd_width);

  _Tp* __restrict r_ptr       = ret.data();
  const _Tp* __restrict a_ptr = a.data();

  if constexpr (std::is_floating_point_v<_Tp>)
  {
    for (std::size_t i = 0; i < simd_end; i += t.simd_width)
    {
      neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(a_ptr + i));
      neon_s32 int_vec  = vcvtq_s32_f32(data_vec);
      vst1q_s32(reinterpret_cast<_s32*>(r_ptr + i), int_vec);
    }
  }
  else if constexpr (std::is_unsigned_v<_Tp>)
  {
    for (std::size_t i = 0; i < simd_end; i += t.simd_width)
    {
      neon_u32 data_vec = vld1q_u32(reinterpret_cast<const _u32*>(a_ptr + i));
      neon_s32 int_vec  = vreinterpretq_s32_u32(data_vec);
      vst1q_s32(reinterpret_cast<_s32*>(r_ptr + i), int_vec);
    }
  }

  for (std::size_t i = simd_end; i < size; ++i)
  {
    r_ptr[i] = static_cast<_s32>(a_ptr[i]);
  }

  return tensor<_s32>(t.shape(), std::move(ret));
}

template<class _Tp>
tensor<_u32> uint32_(const tensor<_Tp>& t) {
  if (!std::is_convertible_v<_Tp, _u32>)
  {
    throw error::type_error("Type must be convertible to unsigned 32 bit int");
  }

  if (t.empty())
  {
    return tensor<_u32>(t.shape());
  }

  std::vector<_Tp>& a    = t.storage_();
  std::size_t       size = a.size();
  std::vector<_u32> ret(size);
  const _u64        simd_end = size - (size % t.simd_width);

  _u32* __restrict r_ptr      = ret.data();
  const _Tp* __restrict a_ptr = a.data();

  if constexpr (std::is_floating_point_v<_Tp>)
  {
    for (std::size_t i = 0; i < simd_end; i += t.simd_width)
    {
      neon_f32 data_vec = vld1q_f32(reinterpret_cast<const _f32*>(a_ptr + i));
      neon_u32 uint_vec = vcvtq_u32_f32(data_vec);
      vst1q_u32(reinterpret_cast<_u32*>(r_ptr + i), uint_vec);
    }
  }
  else if constexpr (std::is_signed_v<_Tp>)
  {
    for (std::size_t i = 0; i < simd_end; i += t.simd_width)
    {
      neon_s32 data_vec = vld1q_s32(reinterpret_cast<const _s32*>(a_ptr + i));
      neon_u32 uint_vec = vreinterpretq_u32_s32(data_vec);
      vst1q_u32(reinterpret_cast<_u32*>(r_ptr + i), uint_vec);
    }
  }

  for (std::size_t i = simd_end; i < size; ++i)
  {
    r_ptr[i] = static_cast<_u32>(a_ptr[i]);
  }

  return tensor<_u32>(t.shape(), std::move(ret));
}

template<class _Tp>
tensor<_f32> float32_(const tensor<_Tp>& t) {
  if (!std::is_convertible_v<_Tp, _f32>)
  {
    throw error::type_error("Type must be convertible to 32 bit float");
  }

  if (t.empty())
  {
    return tensor<_f32>(t.shape());
  }

  std::vector<_Tp>& a    = t.storage_();
  std::size_t       size = a.size();
  std::vector<_f32> ret(size);
  const _u64        simd_end = size - (size % (t.simd_width / 2));

  _f32* __restrict r_ptr      = ret.data();
  const _Tp* __restrict a_ptr = a.data();

  if constexpr (std::is_same_v<_Tp, _f64>)
  {
    for (std::size_t i = 0; i < simd_end; i += (t.simd_width / 2))
    {
      neon_f64    data_vec1          = vld1q_f64(reinterpret_cast<const _f64*>(a_ptr + i));
      neon_f64    data_vec2          = vld1q_f64(reinterpret_cast<const _f64*>(a_ptr + i));
      float32x2_t float_vec1         = vcvt_f32_f64(data_vec1);
      float32x2_t float_vec2         = vcvt_f32_f64(data_vec2);
      neon_f32    float_vec_combined = vcombine_f32(float_vec1, float_vec2);

      vst1q_f32(reinterpret_cast<_f32*>(r_ptr + i), float_vec_combined);
    }
  }
  else if constexpr (std::is_same_v<_Tp, _s32>)
  {
    for (std::size_t i = 0; i < simd_end; i += t.simd_width)
    {
      neon_s32 data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(a_ptr + i));
      neon_f32 float_vec = vcvtq_f32_s32(data_vec);

      vst1q_f32(reinterpret_cast<_f32*>(r_ptr + i), float_vec);
    }
  }

  for (std::size_t i = simd_end; i < size; ++i)
  {
    r_ptr[i] = static_cast<_f32>(a_ptr[i]);
  }

  return tensor<_f32>(t.shape(), std::move(ret));
}

template<class _Tp>
tensor<_f64> float64_(const tensor<_Tp>& t) {
  if (!std::is_convertible_v<_Tp, _f64>)
  {
    throw error::type_error("Type must be convertible to 64 bit float");
  }

  if (t.empty())
  {
    return tensor<_f64>(t.shape());
  }

  std::vector<_Tp>& a    = t.storage_();
  std::size_t       size = a.size();
  std::vector<_f64> ret(size);
  const _u64        simd_end = size - (size % t.simd_width);

  _f64* __restrict r_ptr      = ret.data();
  const _Tp* __restrict a_ptr = a.data();

  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    auto data_vec = vld1q_f64(reinterpret_cast<const double*>(a_ptr + i));
    vst1q_f64(reinterpret_cast<_f64*>(r_ptr + i), data_vec);
  }

  for (std::size_t i = simd_end; i < a.size(); ++i)
  {
    r_ptr[i] = static_cast<_f64>(a_ptr[i]);
  }

  return tensor<_f64>(t.shape(), std::move(ret));
}

template<class _Tp>
tensor<_u64> uint64_(const tensor<_Tp>& t) {
  if (!std::is_convertible_v<_Tp, _u64>)
  {
    throw error::type_error("Type must be convertible to unsigned 64 bit int");
  }

  if (t.empty())
  {
    return tensor<_u64>(t.shape());
  }

  std::vector<_Tp>& a    = t.storage_();
  std::size_t       size = a.size();
  std::vector<_u64> ret(size);
  const _u64        simd_end = size - (size % t.simd_width);

  _u64* __restrict r_ptr      = ret.data();
  const _Tp* __restrict a_ptr = a.data();

  if constexpr (std::is_unsigned_v<_Tp>)
  {
    for (std::size_t i = 0; i < simd_end; i += t.simd_width)
    {
      neon_u32   data_vec = vld1q_u32(reinterpret_cast<const _u32*>(a_ptr + i));
      uint64x2_t int_vec1 = vmovl_u32(vget_low_u32(data_vec));
      uint64x2_t int_vec2 = vmovl_u32(vget_high_u32(data_vec));
      vst1q_u64(reinterpret_cast<_u64*>(r_ptr + i), int_vec1);
      vst1q_u64(reinterpret_cast<_u64*>(r_ptr + i + 2), int_vec2);
    }
  }
  else
  {
    for (std::size_t i = 0; i < simd_end; i += t.simd_width)
    {
      neon_f64 data_vec = vld1q_f64(reinterpret_cast<const _f64*>(&a[i]));
      neon_f64 uint_vec = vcvtq_u64_f64(data_vec);
      vst1q_u64(reinterpret_cast<_u64*>(r_ptr + i), uint_vec);
    }
  }

  for (std::size_t i = simd_end; i < size; ++i)
  {
    r_ptr[i] = static_cast<_u64>(a_ptr[i]);
  }

  return tensor<_u64>(t.shape(), std::move(ret));
}

template<class _Tp>
tensor<_s64> int64_(const tensor<_Tp>& t) {
  if (!std::is_convertible_v<_Tp, _s64>)
  {
    throw error::type_error("Type must be convertible to 64 bit signed int");
  }

  if (t.empty())
  {
    return tensor<_s64>(t.shape());
  }

  std::vector<_Tp>& a    = t.storage_();
  std::size_t       size = a.size();
  std::vector<_s64> ret(size);
  const _u64        simd_end = size - (size % t.simd_width);

  _s64* __restrict r_ptr      = ret.data();
  const _Tp* __restrict a_ptr = a.data();

  for (std::size_t i = 0; i < simd_end; i += t.simd_width)
  {
    auto data_vec = vld1q_s64(reinterpret_cast<const _s64*>(a_ptr + i));
    vst1q_s64(reinterpret_cast<_s64*>(r_ptr + i), data_vec);
  }

  for (std::size_t i = simd_end; i < a.size(); ++i)
  {
    r_ptr[i] = static_cast<_s64>(a_ptr[i]);
  }

  return tensor<_s64>(t.shape(), std::move(ret));
}

}