#pragma once

#include "internal/simd/neon/alias.hpp"
#include "tensor.hpp"


inline neon_f32 vsinq_f32(neon_f32 x) {
  return {sinf(vgetq_lane_f32(x, 0)), sinf(vgetq_lane_f32(x, 1)), sinf(vgetq_lane_f32(x, 2)),
          sinf(vgetq_lane_f32(x, 3))};
}

namespace internal::simd::neon {

template<class _Tp>
tensor<_Tp>& sinc_(tensor<_Tp>& t) {
  if (!std::is_arithmetic_v<_Tp>)
  {
    throw error::type_error("Type must be arithmetic");
  }

  std::vector<_Tp>& data_ = t.storage_();
  _u64              i     = 0;

  if constexpr (std::is_floating_point_v<_Tp>)
  {
    const _u64 simd_end = data_.size() - (data_.size() % _ARM64_REG_WIDTH);

    for (; i < simd_end; i += _ARM64_REG_WIDTH)
    {
      neon_f32 v      = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
      neon_f32 pi_v   = vmulq_f32(v, vdupq_n_f32(M_PI));                        // pi * x
      neon_f32 sinc_v = vbslq_f32(vcgeq_f32(vabsq_f32(v), vdupq_n_f32(1e-6f)),  // Check |x| > epsilon
                                  vdivq_f32(vsinq_f32(pi_v), pi_v),             // sinc(x) = sin(pi * x) / (pi * x)
                                  vdupq_n_f32(1.0f));                           // sinc(0) = 1

      vst1q_f32(reinterpret_cast<_f32*>(&data_[i]), sinc_v);
    }
  }

  for (; i < data_.size(); ++i)
  {
    data_[i] = (std::abs(data_[i]) < 1e-6) ? static_cast<_Tp>(1.0)

                                           : static_cast<_Tp>(std::sin(M_PI * data_[i]) / (M_PI * data_[i]));
  }
  return t;
}

}