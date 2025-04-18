#pragma once

#include "tensorbase.hpp"

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::neon_sigmoid_() {
    if (!std::is_arithmetic_v<value_type>)
    {
        throw type_error("Type must be arithmetic");
    }

    index_type i = 0;

    using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, void>::type;

    constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");
    const index_type simd_end = data_.size() - (data_.size() % simd_width);

    if constexpr (std::is_same_v<value_type, _f32>)
    {
        for (; i < simd_end; i += simd_width)
        {
            neon_type v         = vld1q_f32(reinterpret_cast<const _f32*>(&data_[i]));
            neon_type exp_neg_v = vexpq_f32(vnegq_f32(v));                               // e^(-x)
            neon_type sigmoid   = vrecpeq_f32(vaddq_f32(vdupq_n_f32(1.0f), exp_neg_v));  // 1 / (1 + e^(-x))

            vst1q_f32(reinterpret_cast<_f32*>(&data_[i]), sigmoid);
        }
    }

    for (; i < data_.size(); ++i)
    {
        data_[i] = static_cast<value_type>(1.0 / (1.0 + std::exp(-static_cast<double>(data_[i]))));
    }

    return *this;
}
