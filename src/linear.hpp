#pragma once

#include "tensorbase.hpp"


template<class _Tp>
tensor<_Tp> tensor<_Tp>::matmul(const tensor& __other) const {
  if (this->__shape_.size() != 2 || __other.shape().size() != 2)
  {
    throw std::invalid_argument("matmul is only supported for 2D tensors");
  }

  if (this->__shape_[1] != __other.shape()[0])
  {
    if (this->__shape_[0] == __other.shape()[1])
    {
      return __other.matmul(*this);
    }

    throw std::invalid_argument("Shape mismatch for matrix multiplication: "
                                "this shape: ["
                                + std::to_string(this->__shape_[0]) + ", " + std::to_string(this->__shape_[1])
                                + "] "
                                  "other shape: ["
                                + std::to_string(__other.shape()[0]) + ", " + std::to_string(__other.shape()[1]) + "]");
  }

  shape_type __ret_sh = {this->__shape_[0], __other.shape()[1]};
  data_t     __ret_d(__ret_sh[0] * __ret_sh[1], 0);

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (int __i = 0; __i < __ret_sh[0]; __i += _ARM64_REG_WIDTH)
    {
      for (int __j = 0; __j < __ret_sh[1]; __j += _ARM64_REG_WIDTH)
      {
        for (int __k = 0; __k < this->__shape_[1]; __k += _ARM64_REG_WIDTH)
        {
          for (int __ii = __i; __ii < std::min(static_cast<index_type>(__i + _ARM64_REG_WIDTH), __ret_sh[0]); __ii++)
          {
            for (int __jj = __j; __jj < std::min(static_cast<index_type>(__j + _ARM64_REG_WIDTH), __ret_sh[1]); __jj++)
            {
              neon_f32 __sum_vec = vdupq_n_f32(0);

              for (int __kk = __k; __kk < std::min(static_cast<index_type>(__k + _ARM64_REG_WIDTH), this->__shape_[1]);
                   __kk += _ARM64_REG_WIDTH)
              {
                neon_f32 __a_vec =
                  vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__ii * this->__shape_[1] + __kk]));
                neon_f32 __b_vec =
                  vld1q_f32(reinterpret_cast<const _f32*>(&__other.__data_[__kk * __other.shape()[1] + __jj]));
                __sum_vec = vmlaq_f32(__sum_vec, __a_vec, __b_vec);
              }

              float32x2_t __sum_low  = vget_low_f32(__sum_vec);
              float32x2_t __sum_high = vget_high_f32(__sum_vec);
              __sum_low              = vadd_f32(__sum_low, __sum_high);
              float32x2_t __sum_dup  = vpadd_f32(__sum_low, __sum_low);
              __ret_d[__ii * __ret_sh[1] + __jj] += vget_lane_f32(__sum_dup, 0);
            }
          }
        }
      }
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (int __i = 0; __i < __ret_sh[0]; __i += _ARM64_REG_WIDTH)
    {
      for (int __j = 0; __j < __ret_sh[1]; __j += _ARM64_REG_WIDTH)
      {
        for (int __k = 0; __k < this->__shape_[1]; __k += _ARM64_REG_WIDTH)
        {
          for (int __ii = __i; __ii < std::min(static_cast<index_type>(__i + _ARM64_REG_WIDTH), __ret_sh[0]); __ii++)
          {
            for (int __jj = __j; __jj < std::min(static_cast<index_type>(__j + _ARM64_REG_WIDTH), __ret_sh[1]); __jj++)
            {
              neon_s32 __sum_vec = vdupq_n_s32(0);

              for (int __kk = __k; __kk < std::min(static_cast<index_type>(__k + _ARM64_REG_WIDTH), this->__shape_[1]);
                   __kk += _ARM64_REG_WIDTH)
              {
                neon_s32 __a_vec =
                  vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__ii * this->__shape_[1] + __kk]));
                neon_s32 __b_vec =
                  vld1q_s32(reinterpret_cast<const _s32*>(&__other.__data_[__kk * __other.shape()[1] + __jj]));
                __sum_vec = vmlaq_s32(__sum_vec, __a_vec, __b_vec);
              }

              int32x2_t __sum_low  = vget_low_s32(__sum_vec);
              int32x2_t __sum_high = vget_high_s32(__sum_vec);
              __sum_low            = vadd_s32(__sum_low, __sum_high);
              int32x2_t __sum_dup  = vpadd_s32(__sum_low, __sum_low);
              __ret_d[__ii * __ret_sh[1] + __jj] += vget_lane_s32(__sum_dup, 0);
            }
          }
        }
      }
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (int __i = 0; __i < __ret_sh[0]; __i += _ARM64_REG_WIDTH)
    {
      for (int __j = 0; __j < __ret_sh[1]; __j += _ARM64_REG_WIDTH)
      {
        for (int __k = 0; __k < this->__shape_[1]; __k += _ARM64_REG_WIDTH)
        {
          for (int __ii = __i; __ii < std::min(static_cast<index_type>(__i + _ARM64_REG_WIDTH), __ret_sh[0]); __ii++)
          {
            for (int __jj = __j; __jj < std::min(static_cast<index_type>(__j + _ARM64_REG_WIDTH), __ret_sh[1]); __jj++)
            {
              neon_u32 __sum_vec = vdupq_n_u32(0);

              for (int __kk = __k; __kk < std::min(static_cast<index_type>(__k + _ARM64_REG_WIDTH), this->__shape_[1]);
                   __kk += _ARM64_REG_WIDTH)
              {
                neon_u32 __a_vec =
                  vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__ii * this->__shape_[1] + __kk]));
                neon_u32 __b_vec =
                  vld1q_u32(reinterpret_cast<const _u32*>(&__other.__data_[__kk * __other.shape()[1] + __jj]));
                __sum_vec = vmlaq_u32(__sum_vec, __a_vec, __b_vec);
              }

              uint32x2_t __sum_low  = vget_low_u32(__sum_vec);
              uint32x2_t __sum_high = vget_high_u32(__sum_vec);
              __sum_low             = vadd_u32(__sum_low, __sum_high);
              uint32x2_t __sum_dup  = vpadd_u32(__sum_low, __sum_low);
              __ret_d[__ii * __ret_sh[1] + __jj] += vget_lane_u32(__sum_dup, 0);
            }
          }
        }
      }
    }
  }

#else
  #ifdef __CUDACC__
  const int __threadsPerBlock = 256;
  const int __blocksPerGrid   = (__ret_sh[0] * __ret_sh[1] + __threadsPerBlock - 1) / __threadsPerBlock;

  pointer __d_a, __d_b, __d_c;
  cudaMalloc(&__d_a, this->__data_.size() * sizeof(value_type));
  cudaMalloc(&__d_b, __other.__data_.size() * sizeof(value_type));
  cudaMalloc(&__d_c, __ret_d.size() * sizeof(value_type));

  cudaMemcpy(__d_a, this->__data_.data(), this->__data_.size() * sizeof(value_type), cudaMemcpyHostToDevice);
  cudaMemcpy(__d_b, __other.__data_.data(), __other.__data_.size() * sizeof(value_type), cudaMemcpyHostToDevice);

  matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(__d_a, __d_b, __d_c, this->__shape_[0], this->__shape_[1],
                                                    __other.shape()[1]);

  cudaMemcpy(__ret_d.data(), __d_c, __ret_d.size() * sizeof(value_type), cudaMemcpyDeviceToHost);

  cudaFree(__d_a);
  cudaFree(__d_b);
  cudaFree(__d_c);
  #endif
  #pragma omp parallel
  const int __blockSize = 64;

  for (int __i = 0; __i < __ret_sh[0]; __i += __blockSize)
  {
    for (int __j = 0; __j < __ret_sh[1]; __j += __blockSize)
    {
      for (int __k = 0; __k < this->__shape_[1]; __k += __blockSize)
      {
        for (int __ii = __i; __ii < std::min(static_cast<index_type>(__i + __blockSize), __ret_sh[0]); __ii++)
        {
          for (int __jj = __j; __jj < std::min(static_cast<index_type>(__j + __blockSize), __ret_sh[1]); __jj++)
          {
            value_type __sum = 0;
            for (int __kk = __k; __kk < std::min(static_cast<index_type>(__k + __blockSize), this->__shape_[1]); __kk++)
            {
              __sum += this->at({__ii, __kk}) * __other.at({__kk, __jj});
            }
            __ret_d[__ii * __ret_sh[1] + __jj] += __sum;
          }
        }
      }
    }
  }
#endif

  return __self(__ret_d, __ret_sh);
}

#ifdef __CUDACC__
template<class _Tp>
__global__ void matmul_kernel(_Tp* __a, _Tp* __b, _Tp* __c, int __m, int __n, int __k) {
  int __row = blockIdx.y * blockDim.y + threadIdx.y;
  int __col = blockIdx.x * blockDim.x + threadIdx.x;

  if (__row < __m && __col < __k)
  {
    _Tp __sum = 0;
    for (int __i = 0; __i < __n; __i++)
    {
      __sum += __a[__row * __n + __i] * __b[__i * __k + __col];
    }

    __c[__row * __k + __col] = __sum;
  }
}
#endif

template<class _Tp>
tensor<_Tp> tensor<_Tp>::reshape(const shape_type __sh) const {
  data_t     __d = this->__data_;
  index_type __s = this->__computeSize(__sh);

  if (__s != this->__data_.size())
  {
    throw std::invalid_argument(
      "input shape must have size of elements equal to the current number of elements in the tensor data");
  }

  return __self(__d, __sh);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cross_product(const tensor& __other) const {
  this->__check_is_arithmetic_type("Cannot perform a cross product on non-scalar data types");
  if (this->empty() || __other.empty())
  {
    throw std::invalid_argument("Cannot cross product an empty vector");
  }

  if (this->shape() != std::vector<int>{3} || __other.shape() != std::vector<int>{3})
  {
    throw std::invalid_argument("Cross product can only be performed on 3-element vectors");
  }

  tensor __ret({3});

#if defined(__ARM_NEON) && defined(__aarch64__)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __a      = vld1q_f32(reinterpret_cast<const _f32*>(this->__data_.data()));
    neon_f32 __b      = vld1q_f32(reinterpret_cast<const _f32*>(__other.storage().data()));
    neon_f32 __a_yzx  = vextq_f32(__a, __a, 1);
    neon_f32 __b_yzx  = vextq_f32(__b, __b, 1);
    neon_f32 __result = vsubq_f32(vmulq_f32(__a_yzx, __b), vmulq_f32(__a, __b_yzx));
    __result          = vextq_f32(__result, __result, 3);

    vst1q_f32(reinterpret_cast<_f32*>(__ret.storage().data()), __result);
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __a      = vld1q_s32(reinterpret_cast<const _s32*>(this->__data_.data()));
    neon_s32 __b      = vld1q_s32(reinterpret_cast<const _s32*>(__other.storage().data()));
    neon_s32 __a_yzx  = vextq_s32(__a, __a, 1);
    neon_s32 __b_yzx  = vextq_s32(__b, __b, 1);
    neon_s32 __result = vsubq_s32(vmulq_s32(__a_yzx, __b), vmulq_s32(__a, __b_yzx));
    __result          = vextq_s32(__result, __result, 3);

    vst1q_s32(reinterpret_cast<_s32*>(__ret.storage().data()), __result);
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __a      = vld1q_u32(reinterpret_cast<const _u32*>(this->__data_.data()));
    neon_u32 __b      = vld1q_u32(reinterpret_cast<const _u32*>(__other.storage().data()));
    neon_u32 __a_yzx  = vextq_u32(__a, __a, 1);
    neon_u32 __b_yzx  = vextq_u32(__b, __b, 1);
    neon_u32 __result = vsubq_u32(vmulq_u32(__a_yzx, __b), vmulq_u32(__a, __b_yzx));
    __result          = vextq_u32(__result, __result, 3);

    vst1q_u32(reinterpret_cast<_u32*>(__ret.storage().data()), __result);
  }

#elif defined(__CUDACC__)
  pointer __d_a;
  pointer __d_b;
  pointer __d_c;

  cudaMalloc(&__d_a, 3 * sizeof(value_type));
  cudaMalloc(&__d_b, 3 * sizeof(value_type));
  cudaMalloc(&__d_c, 3 * sizeof(value_type));

  cudaMemcpy(__d_a, this->__data_.data(), 3 * sizeof(value_type), cudaMemcpyHostToDevice);
  cudaMemcpy(__d_b, __other.storage().data(), 3 * sizeof(value_type), cudaMemcpyHostToDevice);

  dim3 block(1);
  dim3 grid(1);
  cross_product_kernel<<<grid, block>>>(__d_a, __d_b, __d_c);

  cudaMemcpy(__ret.storage().data(), __d_c, 3 * sizeof(value_type), cudaMemcpyDeviceToHost);

  cudaFree(__d_a);
  cudaFree(__d_b);
  cudaFree(__d_c);

#else
  const_reference __a1 = this->__data_[0];
  const_reference __a2 = this->__data_[1];
  const_reference __a3 = this->__data_[2];

  const_reference __b1 = __other[0];
  const_reference __b2 = __other[1];
  const_reference __b3 = __other[2];

  __ret[0] = __a2 * __b3 - __a3 * __b2;
  __ret[1] = __a3 * __b1 - __a1 * __b3;
  __ret[2] = __a1 * __b2 - __a2 * __b1;
#endif

#if defined(__AVX2__)
  __m256 __a      = _mm256_loadu_ps(reinterpret_cast<const _f32*>(this->__data_.data()));
  __m256 __b      = _mm256_loadu_ps(reinterpret_cast<const _f32*>(__other.storage().data()));
  __m256 __a_yzx  = _mm256_permute_ps(__a, _MM_SHUFFLE(3, 0, 2, 1));
  __m256 __b_yzx  = _mm256_permute_ps(__b, _MM_SHUFFLE(3, 0, 2, 1));
  __m256 __result = _mm256_sub_ps(_mm256_mul_ps(__a_yzx, __b), _mm256_mul_ps(__a, __b_yzx));
  __result        = _mm256_permute_ps(__result, _MM_SHUFFLE(3, 0, 2, 1));
  _mm256_storeu_ps(reinterpret_cast<_f32*>(__ret.storage().data()), __result);
#endif

#if defined(__SSE__)
  __m128 __a      = _mm_loadu_ps(reinterpret_cast<const _f32*>(this->__data_.data()));
  __m128 __b      = _mm_loadu_ps(reinterpret_cast<const _f32*>(__other.storage().data()));
  __m128 __a_yzx  = _mm_shuffle_ps(__a, __a, _MM_SHUFFLE(3, 0, 2, 1));
  __m128 __b_yzx  = _mm_shuffle_ps(__b, __b, _MM_SHUFFLE(3, 0, 2, 1));
  __m128 __result = _mm_sub_ps(_mm_mul_ps(__a_yzx, __b), _mm_mul_ps(__a, __b_yzx));
  __result        = _mm_shuffle_ps(__result, __result, _MM_SHUFFLE(3, 0, 2, 1));
  _mm_storeu_ps(reinterpret_cast<_f32*>(__ret.storage().data()), __result);

#endif

  return __ret;
}

#ifdef __CUDACC__
template<class _Tp>
__global__ void cross_product_kernel(_Tp* __a, _Tp* __b, _Tp* __c) {
  __c[0] = __a[1] * __b[2] - __a[2] * __b[1];
  __c[1] = __a[2] * __b[0] - __a[0] * __b[2];
  __c[2] = __a[0] * __b[1] - __a[1] * __b[0];
}
#endif

template<class _Tp>
tensor<_Tp> tensor<_Tp>::absolute(const tensor& __tensor) const {
  this->__check_is_scalar_type("Cannot call absolute on non-scalar value");

  index_type __s = __tensor.storage().size();
  data_t     __a;
  __a.reserve(__s);
  index_type __i = 0;

#ifdef __AVX__
  if constexpr (std::is_same_v<value_type, _f32>)
  {
    for (; __i + _AVX_REG_WIDTH <= __s; __i += _AVX_REG_WIDTH)
    {
      __m256 __input     = _mm256_loadu_ps(&__tensor.storage()[__i]);
      __m256 __abs_value = _mm256_abs_ps(__input);
      _mm256_storeu_ps(&__a[__i], __abs_value);
    }
  }
#endif

#if defined(__SSE__)
  if constexpr (std::is_same_v<value_type, _f32>)
  {
    for (; __i + 4 <= __s; __i += 4)
    {
      __m128 __input     = _mm_loadu_ps(&__tensor.storage()[__i]);
      __m128 __abs_value = _mm_abs_ps(__input);
      _mm_storeu_ps(&__a[__i], __abs_value);
    }
  }
#endif

  for (; __i < __s; __i++)
  {
    __a.push_back(static_cast<value_type>(std::fabs(_f32(__tensor.storage()[__i]))));
  }

  return __self(__a, __tensor.__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::dot(const tensor& __other) const {
  this->__check_is_scalar_type("Cannot perform a dot product on non-scalar data types");

  if (this->empty() || __other.empty())
  {
    throw std::invalid_argument("Cannot dot product an empty vector");
  }

  if (this->__shape_.size() == 1 && __other.shape().size() == 1)
  {
    if (this->__shape_[0] != __other.shape()[0])
    {
      throw std::invalid_argument("Vectors must have the same size for dot product");
    }

    const_pointer __this_data  = this->__data_.data();
    const_pointer __other_data = __other.storage().data();
    const size_t  __size       = this->__data_.size();
    value_type    __ret        = 0;

#if defined(__ARM_NEON)
    if constexpr (std::is_floating_point<value_type>::value)
    {
      size_t   __i     = 0;
      neon_f32 sum_vec = vdupq_n_f32(0.0f);

      for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
      {
        neon_f32 a_vec = vld1q_f32(reinterpret_cast<const _f32*>(&__this_data[__i]));
        neon_f32 b_vec = vld1q_f32(reinterpret_cast<const _f32*>(&__other_data[__i]));
        sum_vec        = vmlaq_f32(sum_vec, a_vec, b_vec);  // Perform multiply-accumulate
      }

      float32x2_t sum_half = vadd_f32(vget_high_f32(sum_vec), vget_low_f32(sum_vec));
      __ret                = vget_lane_f32(vpadd_f32(sum_half, sum_half), 0);

      for (; __i < __size; ++__i)
      {
        __ret += static_cast<value_type>(__this_data[__i]) * static_cast<value_type>(__other_data[__i]);
      }
    }
    else if constexpr (std::is_unsigned<value_type>::value)
    {
      size_t   __i     = 0;
      neon_u32 sum_vec = vdupq_n_u32(0.0f);

      for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
      {
        neon_u32 a_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__this_data[__i]));
        neon_u32 b_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other_data[__i]));
        sum_vec        = vmlaq_u32(sum_vec, a_vec, b_vec);
      }

      uint32x2_t sum_half = vadd_u32(vget_high_u32(sum_vec), vget_low_u32(sum_vec));
      __ret               = vget_lane_u32(vpadd_u32(sum_half, sum_half), 0);

      for (; __i < __size; __i++)
      {
        __ret += static_cast<value_type>(__this_data[__i]) * static_cast<value_type>(__other_data[__i]);
      }
    }
    else if constexpr (std::is_signed<value_type>::value)
    {
      size_t   __i     = 0;
      neon_s32 sum_vec = vdupq_n_f32(0.0f);

      for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
      {
        neon_s32 a_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__this_data[__i]));
        neon_s32 b_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other_data[__i]));
        sum_vec        = vmlaq_s32(sum_vec, a_vec, b_vec);  // Perform multiply-accumulate
      }

      int32x2_t sum_half = vadd_s32(vget_high_s32(sum_vec), vget_low_s32(sum_vec));
      __ret              = vget_lane_s32(vpadd_s32(sum_half, sum_half), 0);

      for (; __i < __size; __i++)
      {
        __ret += static_cast<value_type>(__this_data[__i]) * static_cast<value_type>(__other_data[__i]);
      }
    }
#else
    __ret = std::inner_product(__this_data, __this_data + __size, __other_data, value_type(0));
#endif
    return __self({__ret}, {1});
  }

  if (this->__shape_.size() == 2 && __other.shape().size() == 2)
  {
    return this->matmul(__other);
  }

  if (this->__shape_.size() == 3 && __other.shape().size() == 3)
  {
    return this->cross_product(__other);
  }
  return __self();
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::relu() const {
  __self __ret = this->clone();
  __ret.relu_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::relu_() const {
  this->__check_is_scalar_type("Cannot relu non-scalar type");

  if constexpr (std::is_unsigned<value_type>::value)
  {
    return *this;
  }

  index_type __s = this->__data_.size();
  index_type __i = 0;

#pragma omp parallel

#ifdef __CUDACC__
  if (this->__is_cuda_tensor)
  {
    value_type* __d_data = thrust::raw_pointer_cast(this->__data_.data());
    thrust::transform(thrust::device, __d_data, d_data + __s, __d_data,
                      [] __device__(value_type __x) { return max(__x, value_type(0)); });
    return;
  }

#elif defined(__SSE__)
  if constexpr (std::is_same_v<value_type, _f32>)
  {
    __m128 __zero = _mm_setzero_ps();

    for (; __i + 4 <= __s; __i += 4)
    {
      __m128 __x      = _mm_loadu_ps(&this->__data_[__i]);
      __m128 __result = _mm_max_ps(__x, __zero);
      _mm_storeu_ps(&this->__data_[__i], __result);
    }
  }

#elif defined(__AVX__)
  if constexpr (std::is_same_v<value_type, _f32>)
  {
    __m256 __zero = _mm256_setzero_ps();

    for (; __i + _AVX_REG_WIDTH <= __s; __i += _AVX_REG_WIDTH)
    {
      __m256 __x      = _mm256_loadu_ps(&this->__data_[__i]);
      __m256 __result = _mm256_max_ps(__x, __zero);
      _mm256_storeu_ps(&this->__data_[__i], __result);
    }
  }

#elif defined(__ARM_NEON)
  if constexpr (std::is_same_v<value_type, _f32>)
  {
    const neon_f32 __vZero = vdupq_n_f32(0.0f);

    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __v = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      __v          = vmaxq_f32(__v, __vZero);

      vst1q_f32(&this->__data_[__i], __v);
    }
  }
  else if constexpr (std::is_same_v<value_type, _s32>)
  {
    const neon_s32 __vZero = vdupq_n_s32(0);

    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __v = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      __v          = vmaxq_s32(__v, __vZero);

      vst1q_s32(&this->__data_[__i], __v);
    }
  }
#endif

  for (__i = 0; __i < __s; __i++)
  {
    this->__data_[__i] = std::max(this->__data_[__i], value_type(0));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::transpose() const {
  if (this->__shape_.size() != 2)
  {
    throw std::invalid_argument("Matrix transposition can only be done on 2D tensors");
  }

  tensor           __ret({this->__shape_[1], this->__shape_[0]});
  const index_type __rows = this->__shape_[0];
  const index_type __cols = this->__shape_[1];

#ifdef __CUDACC__
  if (this->__is_cuda_tensor)
  {
    dim3 blockDim(16, 16);
    dim3 gridDim((__cols + blockDim.x - 1) / blockDim.x, (__rows + blockDim.y - 1) / blockDim.y);
    transpose_kernel<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(this->__data_.data()),
                                            thrust::raw_pointer_cast(__ret.__data_.data()), __rows, __cols);
    cudaDeviceSynchronize();
    return __ret;
  }
#endif

#if defined(__ARM_NEON)
  if constexpr (std::is_same_v<_Tp, _f32>)
  {
    for (index_type __i = 0; __i < __rows; __i += _ARM64_REG_WIDTH)
    {
      for (index_type __j = 0; __j < __cols; __j += _ARM64_REG_WIDTH)
      {
        if (__i + _ARM64_REG_WIDTH <= __rows && __j + _ARM64_REG_WIDTH <= __cols)
        {
          float32x4x4_t __input;

          for (index_type __k = 0; __k < _ARM64_REG_WIDTH; __k++)
          {
            __input.val[__k] = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[(__i + __k) * __cols + __j]));
          }

          float32x4x4_t __output = vld4q_f32(reinterpret_cast<const _f32*>(&__input));

          for (index_type __k = 0; __k < _ARM64_REG_WIDTH; __k++)
          {
            vst1q_f32(&__ret.__data_[(__j + __k) * __rows + __i], __output.val[__k]);
          }
        }
        else
        {
          for (index_type __ii = __i; __ii < std::min(static_cast<index_type>(__i + _ARM64_REG_WIDTH), __rows); __ii++)
          {
            for (index_type __jj = __j; __jj < std::min(static_cast<index_type>(__j + _ARM64_REG_WIDTH), __cols);
                 __jj++)
            {
              __ret.at({__jj, __ii}) = this->at({__ii, __jj});
            }
          }
        }
      }
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (index_type __i = 0; __i < __rows; __i += _ARM64_REG_WIDTH)
    {
      for (index_type __j = 0; __j < __cols; __j += _ARM64_REG_WIDTH)
      {
        if (__i + _ARM64_REG_WIDTH <= __rows && __j + _ARM64_REG_WIDTH <= __cols)
        {
          int32x4x4_t __input;

          for (index_type __k = 0; __k < _ARM64_REG_WIDTH; __k++)
          {
            __input.val[__k] = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[(__i + __k) * __cols + __j]));
          }

          int32x4x4_t __output = vld4q_s32(reinterpret_cast<const _s32*>(&__input));

          for (index_type __k = 0; __k < _ARM64_REG_WIDTH; __k++)
          {
            vst1q_s32(&__ret.__data_[(__j + __k) * __rows + __i], __output.val[__k]);
          }
        }
        else
        {
          for (index_type __ii = __i; __ii < std::min(static_cast<index_type>(__i + _ARM64_REG_WIDTH), __rows); __ii++)
          {
            for (index_type __jj = __j; __jj < std::min(static_cast<index_type>(__j + _ARM64_REG_WIDTH), __cols);
                 __jj++)
            {
              __ret.at({__jj, __ii}) = this->at({__ii, __jj});
            }
          }
        }
      }
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (index_type __i = 0; __i < __rows; __i += _ARM64_REG_WIDTH)
    {
      for (index_type __j = 0; __j < __cols; __j += _ARM64_REG_WIDTH)
      {
        if (__i + _ARM64_REG_WIDTH <= __rows && __j + _ARM64_REG_WIDTH <= __cols)
        {
          uint32x4x4_t __input;

          for (index_type __k = 0; __k < _ARM64_REG_WIDTH; __k++)
          {
            __input.val[__k] = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[(__i + __k) * __cols + __j]));
          }

          uint32x4x4_t __output = vld4q_u32(reinterpret_cast<const _u32*>(&__input));

          for (index_type __k = 0; __k < _ARM64_REG_WIDTH; __k++)
          {
            vst1q_u32(&__ret.__data_[(__j + __k) * __rows + __i], __output.val[__k]);
          }
        }
        else
        {
          for (index_type __ii = __i; __ii < std::min(static_cast<index_type>(__i + _ARM64_REG_WIDTH), __rows); __ii++)
          {
            for (index_type __jj = __j; __jj < std::min(static_cast<index_type>(__j + _ARM64_REG_WIDTH), __cols);
                 __jj++)
            {
              __ret.at({__jj, __ii}) = this->at({__ii, __jj});
            }
          }
        }
      }
    }
  }
  else
#endif
  {
    index_type __i = 0;

    for (; __i < __rows; __i++)
    {
      index_type __j = 0;

      for (; __j < __cols; __j++)
      {
        __ret.at({__j, __i}) = this->at({__i, __j});
      }
    }
  }

  return __ret;
}

#ifdef __CUDACC__
template<class _Tp>
__global__ void transpose_kernel(_Tp* __input, _Tp* __output, int __rows, int __cols) {
  int __i = blockIdx.y * blockDim.y + threadIdx.y;
  int __j = blockIdx.x * blockDim.x + threadIdx.x;

  if (__i < __rows && __j < __cols)
  {
    output[__j * __rows + __i] = input[__i * __cols + __j];
  }
}
#endif

template<class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argsort(index_type __d, bool __ascending) const {
  index_type __adjusted = (__d < 0) ? __d + this->__data_.size() : __d;

  if (__adjusted != 0)
  {
    throw std::out_of_range("Invalid dimension for argsort: only 1D tensors are supported");
  }

  index_type __size = static_cast<index_type>(this->__data_.size());
  shape_type __indices(__size);
  std::iota(__indices.begin(), __indices.end(), 0);

#if defined(__ARM_NEON)
  index_type __i = 0;

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
    {
      neon_f32    __data_vec   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      float32x2_t __min1       = vpmin_f32(vget_low_f32(__data_vec), vget_high_f32(__data_vec));
      float32x2_t __min2       = vpmin_f32(__min1, __min1);
      neon_f32    __cmp_vec    = vdupq_lane_f32(__min2, 0);
      neon_u32    __cmp_result = __ascending ? vcltq_f32(__data_vec, __cmp_vec) : vcgtq_f32(__data_vec, __cmp_vec);

      for (int __j = 0; __j < _ARM64_REG_WIDTH; __j++)
      {
        __indices[__i + __j] = (__cmp_result[__j] ? __i + __j : __i + __j + 1);
      }
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
    {
      neon_s32  __data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      int32x2_t __min1       = vpmin_s32(vget_low_s32(__data_vec), vget_high_s32(__data_vec));
      int32x2_t __min2       = vpmin_s32(__min1, __min1);
      neon_s32  __cmp_vec    = vdupq_lane_s32(__min2, 0);
      neon_u32  __cmp_result = __ascending ? vcltq_s32(__data_vec, __cmp_vec) : vcgtq_s32(__data_vec, __cmp_vec);

      for (int __j = 0; __j < _ARM64_REG_WIDTH; __j++)
      {
        __indices[__i + __j] = (__cmp_result[__j] ? __i + __j : __i + __j + 1);
      }
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
    {
      neon_u32   __data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      uint32x2_t __min1       = vpmin_u32(vget_low_u32(__data_vec), vget_high_u32(__data_vec));
      uint32x2_t __min2       = vpmin_u32(__min1, __min1);
      neon_u32   __cmp_vec    = vdupq_lane_u32(__min2, 0);
      neon_u32   __cmp_result = __ascending ? vcltq_u32(__data_vec, __cmp_vec) : vcgtq_u32(__data_vec, __cmp_vec);

      for (int __j = 0; __j < _ARM64_REG_WIDTH; __j++)
      {
        __indices[__i + __j] = (__cmp_result[__j] ? __i + __j : __i + __j + 1);
      }
    }
  }

  for (; __i < __size; __i++)
  {
    __indices[__i] = __i;
  }
#endif

  std::sort(__indices.begin(), __indices.end(), [&](index_type __a, index_type __b) {
    return __ascending ? this->__data_[__a] < this->__data_[__b] : this->__data_[__a] > this->__data_[__b];
  });

  return tensor<index_type>(__indices);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sigmoid() const {
  __self __ret = this->clone();
  __ret.sigmoid_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sigmoid_() const {
  this->__check_is_arithmetic_type("sigmoid_: template type must be an arithmetic type");

  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, void>::type;

  if constexpr (std::is_same<value_type, _f32>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_type __v         = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_type __exp_neg_v = vexpq_f32(vnegq_f32(__v));                               // e^(-x)
      neon_type __sigmoid   = vrecpeq_f32(vaddq_f32(vdupq_n_f32(1.0f), __exp_neg_v));  // 1 / (1 + e^(-x))

      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __sigmoid);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(1.0 / (1.0 + std::exp(-static_cast<double>(this->__data_[__i]))));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::clipped_relu_(const value_type __clip_limit) const {
  this->__check_is_scalar_type("Cannot apply clipped ReLU to a non-scalar type");

  if constexpr (std::is_unsigned<value_type>::value)
  {
    return *this;
  }

  index_type __s = this->__data_.size();
  index_type __i = 0;

#pragma omp parallel

#ifdef __CUDACC__
  if (this->__is_cuda_tensor)
  {
    pointer __d_data = thrust::raw_pointer_cast(this->__data_.data());
    thrust::transform(thrust::device, __d_data, __d_data + __s, __d_data,
                      [] __device__(value_type __x) { return min(max(__x, value_type(0)), __clip_limit); });
    return *this;
  }

#elif defined(__SSE__)
  if constexpr (std::is_same_v<value_type, _f32>)
  {
    __m128 __zero = _mm_setzero_ps();
    __m128 __clip = _mm_set1_ps(__clip_limit);

    for (; __i + 4 <= __s; __i += 4)
    {
      __m128 __x      = _mm_loadu_ps(&this->__data_[__i]);
      __m128 __result = _mm_min_ps(_mm_max_ps(__x, __zero), __clip);

      _mm_storeu_ps(&this->__data_[__i], __result);
    }
  }

#elif defined(__AVX__)
  if constexpr (std::is_same_v<value_type, _f32>)
  {
    __m256 __zero = _mm256_setzero_ps();
    __m256 __clip = _mm256_set1_ps(__clip_limit);

    for (; __i + _AVX_REG_WIDTH <= __s; __i += _AVX_REG_WIDTH)
    {
      __m256 __x      = _mm256_loadu_ps(&this->__data_[__i]);
      __m256 __result = _mm256_min_ps(_mm256_max_ps(__x, __zero), __clip);

      _mm256_storeu_ps(&this->__data_[__i], __result);
    }
  }

#elif defined(__ARM_NEON)
  if constexpr (std::is_same_v<value_type, _f32>)
  {
    const neon_f32 __vZero = vdupq_n_f32(0.0f);
    const neon_f32 __vClip = vdupq_n_f32(__clip_limit);

    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __v = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      __v          = vminq_f32(vmaxq_f32(__v, __vZero), __vClip);

      vst1q_f32(&this->__data_[__i], __v);
    }
  }
  else if constexpr (std::is_same_v<value_type, _s32>)
  {
    const neon_s32 __vZero = vdupq_n_s32(0);
    const neon_s32 __vClip = vdupq_n_s32(__clip_limit);

    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __v = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      __v          = vminq_s32(vmaxq_s32(__v, __vZero), __vClip);

      vst1q_s32(&this->__data_[__i], __v);
    }
  }
#endif

  for (; __i < __s; __i++)
  {
    this->__data_[__i] = std::min(std::max(this->__data_[__i], value_type(0)), __clip_limit);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::unsqueeze(index_type __dim) const {
  if (__dim < 0 || __dim > static_cast<index_type>(this->__shape_.size()))
    throw std::out_of_range("Dimension out of range in unsqueeze");

  shape_type __s = this->__shape_;
  __s.insert(__s.begin() + __dim, 1);

  tensor __ret;
  __ret.__shape_ = __s;
  __ret.__data_  = this->__data_;

  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::unsqueeze_(index_type __dim) const {
  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::clamp_(const_pointer __min_val, const_pointer __max_val) const {
  index_type __i = 0;

#if defined(__AVX2__)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _AVX_REG_WIDTH);
  __m256           __min_vec  = _mm256_set1_ps(__min_val ? *__min_val : std::numeric_limits<_Tp>::lowest());
  __m256           __max_vec  = _mm256_set1_ps(__max_val ? *__max_val : std::numeric_limits<_Tp>::max());

  for (; __i < __simd_end; __i += _AVX_REG_WIDTH)
  {
    __m256 __data_vec = _mm256_loadu_ps(&this->__data_[__i]);
    __m256 __clamped  = _mm256_min_ps(_mm256_max_ps(data_vec, __min_vec), __max_vec);

    _mm256_storeu_ps(&this->__data_[__i], __clamped);
  }

#elif defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __min_vec = vdupq_n_f32(__min_val ? *__min_val : std::numeric_limits<value_type>::lowest());
    neon_f32 __max_vec = vdupq_n_f32(__max_val ? *__max_val : std::numeric_limits<value_type>::max());

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __clamped  = vminq_f32(vmaxq_f32(__data_vec, __min_vec), __max_vec);

      vst1q_f32(&this->__data_[__i], __clamped);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __min_vec = vdupq_n_s32(__min_val ? *__min_val : std::numeric_limits<value_type>::lowest());
    neon_s32 __max_vec = vdupq_n_s32(__max_val ? *__max_val : std::numeric_limits<value_type>::max());

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __clamped  = vminq_s32(vmaxq_s32(__data_vec, __min_vec), __max_vec);

      vst1q_s32(&this->__data_[__i], __clamped);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __min_vec = vdupq_n_u32(__min_val ? *__min_val : std::numeric_limits<value_type>::lowest());
    neon_u32 __max_vec = vdupq_n_u32(__max_val ? *__max_val : std::numeric_limits<value_type>::max());

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __clamped  = vminq_u32(vmaxq_u32(__data_vec, __min_vec), __max_vec);

      vst1q_u32(&this->__data_[__i], __clamped);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    if (__min_val)
    {
      this->__data_[__i] = std::max(*__min_val, this->__data_[__i]);
    }

    if (__max_val)
    {
      this->__data_[__i] = std::min(*__max_val, this->__data_[__i]);
    }
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::clamp(const_pointer __min_val, const_pointer __max_val) const {
  __self __ret = this->clone();
  __ret.clamp_(__min_val, __max_val);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::floor() const {
  __self __ret = this->clone();
  __ret.floor_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::floor_() const {
  static_assert(std::is_floating_point<value_type>::value);
  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < this->__data_.size(); __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec  = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __floor_vec = vrndmq_f32(__data_vec);

      vst1q_f32(&this->__data_[__i], __floor_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::floor(static_cast<_f32>(this->__data_[__i])));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::ceil_() const {
  static_assert(std::is_floating_point<value_type>::value);
  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i + _ARM64_REG_WIDTH <= this->__data_.size(); __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __ceil_vec = vrndpq_f32(__data_vec);

      vst1q_f32(&this->__data_[__i], __ceil_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::ceil(static_cast<_f32>(this->__data_[__i])));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::ceil() const {
  __self __ret = this->clone();
  __ret.ceil_();
  return __ret;
}

template<class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argmax_(index_type __dim) const {
  if (__dim < 0 || __dim >= this->__shape_.size())
  {
    throw std::out_of_range("Dimension out of range in argmax");
  }

  tensor<index_type> __ret;
  shape_type         __ret_sh = this->__shape_;
  __ret_sh.erase(__ret_sh.begin() + __dim);
  __ret.__shape_ = __ret_sh;
  __ret.__data_.resize(this->__computeSize(__ret_sh), 0);

  index_type __outer_size = 1;
  index_type __inner_size = 1;
  index_type __i          = 0;

  for (; __i < __dim; __i++)
  {
    __outer_size *= this->__shape_[__i];
  }

  for (__i = __dim + 1; __i < this->__shape_.size(); __i++)
  {
    __inner_size *= this->__shape_[__i];
  }

#if defined(__AVX2__)
  if constexpr (std::is_same_v<_Tp, _f32>)
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      index_type __j = 0;
      for (; __j < __inner_size; __j++)
      {
        __m256     __max_vec       = _mm256_set1_ps(-std::numeric_limits<_f32>::infinity());
        __m256i    __index_vec     = _mm256_setzero_si256();
        __m256i    __increment     = _mm256_set1_epi32(1);
        __m256i    __current_index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        index_type __k             = 0;

        for (; __k + _AVX_REG_WIDTH <= this->__shape_[__dim]; __k += _AVX_REG_WIDTH)
        {
          __m256 __data_vec = _mm256_loadu_ps(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
          __m256 __mask     = _mm256_cmp_ps(__data_vec, __max_vec, _CMP_GT_OQ);
          __max_vec         = _mm256_blendv_ps(__max_vec, __data_vec, __mask);
          __index_vec       = _mm256_blendv_epi8(__index_vec, __current_index, _mm256_castps_si256(__mask));
          __current_index   = _mm256_add_epi32(__current_index, __increment);
        }

        _f32 __max_values[_AVX_REG_WIDTH];
        _s32 __indices[_AVX_REG_WIDTH];

        _mm256_storeu_ps(__max_values, __max_vec);
        _mm256_storeu_si256((__m256i*) __indices, __index_vec);

        _f32       __max_value = __max_values[0];
        index_type __max_index = __indices[0];

        for (int __k = 1; __k < _AVX_REG_WIDTH; __k++)
        {
          if (__max_values[__k] > __max_value)
          {
            __max_value = __max_values[__k];
            __max_index = __indices[__k];
          }
        }

        for (; __k < this->__shape_[__dim]; __k++)
        {
          _f32 __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          if (__v > __max_value)
          {
            __max_value = __v;
            __max_index = __k;
          }
        }
        __ret.__data_[__i * __inner_size + __j] = __max_index;
      }
    }
  }

#elif defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      index_type __j = 0;
      for (; __j < __inner_size; __j++)
      {
        neon_f32   __max_vec       = vdupq_n_f32(-std::numeric_limits<_f32>::infinity());
        neon_u32   __index_vec     = vdupq_n_u32(0);
        neon_u32   __increment     = vdupq_n_u32(1);
        neon_u32   __current_index = {0, 1, 2, 3};
        index_type __k             = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH)
        {
          neon_f32 __data_vec = vld1q_f32(
            reinterpret_cast<const _f32*>(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          neon_u32 __mask = vcgtq_f32(__data_vec, __max_vec);
          __max_vec       = vbslq_f32(__mask, __data_vec, __max_vec);
          __index_vec     = vbslq_u32(__mask, __current_index, __index_vec);
          __current_index = vaddq_u32(__current_index, __increment);
        }

        _f32 __max_values[_ARM64_REG_WIDTH];
        _u32 __indices[_ARM64_REG_WIDTH];

        vst1q_f32(__max_values, __max_vec);
        vst1q_u32(__indices, __index_vec);

        _f32       __max_value = __max_values[0];
        index_type __max_index = __indices[0];

        for (int __k = 1; __k < _ARM64_REG_WIDTH; __k++)
        {
          if (__max_values[__k] > __max_value)
          {
            __max_value = __max_values[__k];
            __max_index = __indices[__k];
          }
        }

        for (; __k < this->__shape_[__dim]; __k++)
        {
          _f32 __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          if (__v > __max_value)
          {
            __max_value = __v;
            __max_index = __k;
          }
        }
        __ret.__data_[__i * __inner_size + __j] = __max_index;
      }
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      index_type __j = 0;
      for (; __j < __inner_size; __j++)
      {
        neon_s32   __max_vec       = vdupq_n_s32(-std::numeric_limits<_s32>::infinity());
        neon_u32   __index_vec     = vdupq_n_u32(0);
        neon_u32   __increment     = vdupq_n_u32(1);
        neon_u32   __current_index = {0, 1, 2, 3};
        index_type __k             = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH)
        {
          neon_s32 __data_vec = vld1q_s32(
            reinterpret_cast<const _s32*>(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          neon_u32 __mask = vcgtq_s32(__data_vec, __max_vec);
          __max_vec       = vbslq_s32(__mask, __data_vec, __max_vec);
          __index_vec     = vbslq_u32(__mask, __current_index, __index_vec);
          __current_index = vaddq_u32(__current_index, __increment);
        }

        _s32 __max_values[_ARM64_REG_WIDTH];
        _u32 __indices[_ARM64_REG_WIDTH];

        vst1q_s32(__max_values, __max_vec);
        vst1q_u32(__indices, __index_vec);

        _s32       __max_value = __max_values[0];
        index_type __max_index = __indices[0];

        for (int __k = 1; __k < _ARM64_REG_WIDTH; __k++)
        {
          if (__max_values[__k] > __max_value)
          {
            __max_value = __max_values[__k];
            __max_index = __indices[__k];
          }
        }

        for (; __k < this->__shape_[__dim]; __k++)
        {
          _s32 __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];

          if (__v > __max_value)
          {
            __max_value = __v;
            __max_index = __k;
          }
        }

        __ret.__data_[__i * __inner_size + __j] = __max_index;
      }
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      index_type __j = 0;
      for (; __j < __inner_size; __j++)
      {
        neon_u32   __max_vec       = vdupq_n_u32(-std::numeric_limits<_u32>::infinity());
        neon_u32   __index_vec     = vdupq_n_u32(0);
        neon_u32   __increment     = vdupq_n_u32(1);
        neon_u32   __current_index = {0, 1, 2, 3};
        index_type __k             = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH)
        {
          neon_u32 __data_vec = vld1q_u32(
            reinterpret_cast<const _u32*>(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          neon_u32 __mask = vcgtq_u32(__data_vec, __max_vec);
          __max_vec       = vbslq_u32(__mask, __data_vec, __max_vec);
          __index_vec     = vbslq_u32(__mask, __current_index, __index_vec);
          __current_index = vaddq_u32(__current_index, __increment);
        }

        _u32 __max_values[_ARM64_REG_WIDTH];
        _u32 __indices[_ARM64_REG_WIDTH];

        vst1q_u32(__max_values, __max_vec);
        vst1q_u32(__indices, __index_vec);

        _u32       __max_value = __max_values[0];
        index_type __max_index = __indices[0];

        for (int __k = 1; __k < _ARM64_REG_WIDTH; __k++)
        {
          if (__max_values[__k] > __max_value)
          {
            __max_value = __max_values[__k];
            __max_index = __indices[__k];
          }
        }

        for (; __k < this->__shape_[__dim]; __k++)
        {
          _u32 __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];

          if (__v > __max_value)
          {
            __max_value = __v;
            __max_index = __k;
          }
        }
        __ret.__data_[__i * __inner_size + __j] = __max_index;
      }
    }
  }

#endif
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      index_type __j = 0;
      for (; __j < __inner_size; __j++)
      {
        index_type __max_index = 0;
        value_type __max_value = this->__data_[__i * this->__shape_[__dim] * __inner_size + __j];
        index_type __k         = 1;
        for (; __k < this->__shape_[__dim]; __k++)
        {
          value_type __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];

          if (__v > __max_value)
          {
            __max_value = __v;
            __max_index = __k;
          }
        }
        __ret.__data_[__i * __inner_size + __j] = __max_index;
      }
    }
  }

  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::argmax(index_type __dim) const {
  if (__dim < 0 || __dim >= this->__shape_.size())
  {
    throw std::out_of_range("Dimension out of range in argmax");
  }

  tensor     __ret;
  shape_type __ret_sh = this->__shape_;

  __ret_sh.erase(__ret_sh.begin() + __dim);
  __ret.__shape_ = __ret_sh;
  __ret.__data_.resize(this->__computeSize(__ret_sh), value_type(0));

  index_type __outer_size = 1;
  index_type __inner_size = 1;
  index_type __i          = 0;

  for (; __i < __dim; __i++)
  {
    __outer_size *= this->__shape_[__i];
  }

  for (__i = __dim + 1; __i < static_cast<index_type>(this->__shape_.size()); __i++)
  {
    __inner_size *= this->__shape_[__i];
  }
#if defined(__AVX2__)
  if constexpr (std::is_same_v<_Tp, _f32>)
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      for (index_type __j = 0; __j < __inner_size; __j++)
      {
        __m256     __max_vec = _mm256_set1_ps(-std::numeric_limits<_f32>::infinity());
        index_type __k       = 0;

        for (; __k + _AVX_REG_WIDTH <= this->__shape_[__dim]; __k += _AVX_REG_WIDTH)
        {
          __m256 __data_vec = _mm256_loadu_ps(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
          __max_vec         = _mm256_max_ps(__max_vec, __data_vec);
        }

        _f32 __max_value = _mm256_reduce_max_ps(__max_vec);
        for (; __k < this->__shape_[__dim]; __k++)
        {
          _f32 __v    = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          __max_value = std::max(__max_value, __v);
        }

        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  }

#elif defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      for (index_type __j = 0; __j < __inner_size; __j++)
      {
        neon_f32   __max_vec = vdupq_n_f32(-std::numeric_limits<_f32>::infinity());
        index_type __k       = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH)
        {
          neon_f32 __data_vec = vld1q_f32(
            reinterpret_cast<const _f32*>(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          __max_vec = vmaxq_f32(__max_vec, __data_vec);
        }

        _f32 __max_value = vmaxvq_f32(__max_vec);
        for (; __k < this->__shape_[__dim]; __k++)
        {
          _f32 __v    = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          __max_value = std::max(__max_value, __v);
        }

        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      for (index_type __j = 0; __j < __inner_size; __j++)
      {
        neon_s32   __max_vec = vdupq_n_s32(-std::numeric_limits<_s32>::infinity());
        index_type __k       = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH)
        {
          neon_s32 __data_vec = vld1q_s32(
            reinterpret_cast<const _s32*>(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          __max_vec = vmaxq_s32(__max_vec, __data_vec);
        }

        _s32 __max_value = vmaxvq_s32(__max_vec);
        for (; __k < this->__shape_[__dim]; __k++)
        {
          _s32 __v    = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          __max_value = std::max(__max_value, __v);
        }

        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      for (index_type __j = 0; __j < __inner_size; __j++)
      {
        neon_u32   __max_vec = vdupq_n_u32(-std::numeric_limits<_u32>::infinity());
        index_type __k       = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH)
        {
          neon_u32 __data_vec = vld1q_u32(
            reinterpret_cast<const _u32*>(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          __max_vec = vmaxq_u32(__max_vec, __data_vec);
        }

        _u32 __max_value = vmaxvq_u32(__max_vec);
        for (; __k < this->__shape_[__dim]; __k++)
        {
          _u32 __v    = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          __max_value = std::max(__max_value, __v);
        }

        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  }
  else

#endif
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      index_type __j = 0;
      for (; __j < __inner_size; __j++)
      {
        value_type __max_value = this->__data_[__i * this->__shape_[__dim] * __inner_size + __j];
        index_type __k         = 1;
        for (; __k < this->__shape_[__dim]; __k++)
        {
          value_type __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];

          if (__v > __max_value)
          {
            __max_value = __v;
          }
        }
        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  }

  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::pop_back() const {
  this->__data_.pop_back();
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sum(const index_type __axis) const {
  this->__check_is_scalar_type("Cannot reduce tensor with non scalar type");

  if (__axis < 0 || __axis >= static_cast<index_type>(this->__shape_.size()))
  {
    throw std::invalid_argument("Invalid axis for sum");
  }

  shape_type __ret_sh   = this->__shape_;
  __ret_sh[__axis]      = 1;
  index_type __ret_size = std::accumulate(__ret_sh.begin(), __ret_sh.end(), 1, std::multiplies<index_type>());
  data_t     __ret_data(__ret_size, value_type(0.0f));

#if defined(__ARM_NEON)
  const index_type __axis_size  = this->__shape_[__axis];
  const index_type __outer_size = this->__compute_outer_size(__axis);
  const index_type __inner_size = this->size(0) / (__outer_size * __axis_size);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (index_type __outer = 0; __outer < __outer_size; __outer++)
    {
      for (index_type __inner = 0; __inner < __inner_size; __inner++)
      {
        neon_f32   __sum_vec = vdupq_n_f32(0.0f);
        index_type __i       = __outer * __axis_size * __inner_size + __inner;
        index_type __j       = 0;

        for (; __j + _ARM64_REG_WIDTH <= __axis_size; __j += _ARM64_REG_WIDTH)
        {
          neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
          __sum_vec           = vaddq_f32(__sum_vec, __data_vec);
          __i += __inner_size * _ARM64_REG_WIDTH;
        }

        _f32 __sum = vaddvq_f32(__sum_vec);

        for (; __j < __axis_size; __j++)
        {
          __sum += this->__data_[__i];
          __i += __inner_size;
        }

        __ret_data[__outer * __inner_size + __inner] = __sum;
      }
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (index_type __outer = 0; __outer < __outer_size; __outer++)
    {
      for (index_type __inner = 0; __inner < __inner_size; __inner++)
      {
        neon_s32   __sum_vec = vdupq_n_s32(0);
        index_type __i       = __outer * __axis_size * __inner_size + __inner;
        index_type __j       = 0;

        for (; __j + _ARM64_REG_WIDTH <= __axis_size; __j += _ARM64_REG_WIDTH)
        {
          neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
          __sum_vec           = vaddq_s32(__sum_vec, __data_vec);
          __i += __inner_size * _ARM64_REG_WIDTH;
        }

        _s32 __sum = vaddvq_s32(__sum_vec);

        for (; __j < __axis_size; __j++)
        {
          __sum += this->__data_[__i];
          __i += __inner_size;
        }

        __ret_data[__outer * __inner_size + __inner] = __sum;
      }
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (index_type __outer = 0; __outer < __outer_size; __outer++)
    {
      for (index_type __inner = 0; __inner < __inner_size; __inner++)
      {
        neon_u32   __sum_vec = vdupq_n_u32(0);
        index_type __i       = __outer * __axis_size * __inner_size + __inner;
        index_type __j       = 0;

        for (; __j + _ARM64_REG_WIDTH <= __axis_size; __j += _ARM64_REG_WIDTH)
        {
          neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
          __sum_vec           = vaddq_u32(__sum_vec, __data_vec);
          __i += __inner_size * _ARM64_REG_WIDTH;
        }

        _u32 __sum = vaddvq_u32(__sum_vec);

        for (; __j < __axis_size; __j++)
        {
          __sum += this->__data_[__i];
          __i += __inner_size;
        }

        __ret_data[__outer * __inner_size + __inner] = __sum;
      }
    }
  }
  else
  {
#endif
    index_type __i = 0;
    for (; __i < static_cast<index_type>(this->__data_.size()); __i++)
    {
      std::vector<index_type> __orig(this->__shape_.size());
      index_type              __index = __i;
      index_type              __j     = static_cast<index_type>(this->__shape_.size()) - 1;

      for (; __j >= 0; __j--)
      {
        __orig[__j] = __index % this->__shape_[__j];
        __index /= this->__shape_[__j];
      }

      __orig[__axis]         = 0;
      index_type __ret_index = 0;
      index_type __st        = 1;

      for (__j = static_cast<index_type>(this->__shape_.size()) - 1; __j >= 0; __j--)
      {
        __ret_index += __orig[__j] * __st;
        __st *= __ret_sh[__j];
      }
      __ret_data[__ret_index] += this->__data_[__i];
    }

#if defined(__ARM_NEON)
  }
#endif

  return __self(__ret_data, __ret_sh);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::slice(index_type                __dim,
                               std::optional<index_type> __start,
                               std::optional<index_type> __end,
                               index_type                __step) const {
  if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
  {
    throw std::out_of_range("Dimension out of range.");
  }

  tensor     __ret;
  index_type __s       = this->__shape_[__dim];
  index_type __start_i = __start.value_or(0);
  index_type __end_i   = __end.value_or(__s);

  if (__start_i < 0)
  {
    __start_i += __s;
  }

  if (__end_i < 0)
  {
    __end_i += __s;
  }

  __start_i               = std::max(index_type(0), std::min(__start_i, __s));
  __end_i                 = std::max(index_type(0), std::min(__end_i, __s));
  index_type __slice_size = (__end_i - __start_i + __step - 1) / __step;
  shape_type __ret_dims   = this->__shape_;
  __ret_dims[-__dim]      = __slice_size;
  __ret                   = __self(__ret_dims);

#if defined(__CUDACC__)
  if (this->__data_.size() >= 1024)
  {
    pointer __d_input;
    pointer __d_output;
    cudaMalloc(&__d_input, this->__data_.size() * sizeof(value_type));
    cudaMalloc(&__d_output, __ret.__data_.size() * sizeof(value_type));

    cudaMemcpy(__d_input, this->__data_.data(), this->__data_.size() * sizeof(value_type), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid(((__slice_size + block.x - 1) / block.x));
    slice_kernel<<<grid, block>>>(__d_input, __d_output, __start_i, __end_i, __step, __slice_size);

    cudaMemcpy(__ret.__data_.data(), __d_output, __ret.__data_.size() * sizeof(value_type), cudaMemcpyDeviceToHost);

    cudaFree(__d_input);
    cudaFree(__d_output);
  }
  else
  {
#endif

#if defined(__ARM_NEON)
    index_type __vector_end = __start_i + ((__end_i - __start_i) / _ARM64_REG_WIDTH) * _ARM64_REG_WIDTH;

    if constexpr (std::is_floating_point<value_type>::value && __step == 1)
    {
      for (index_type __i = __start_i, __j = 0; __i < __vector_end; __i += _ARM64_REG_WIDTH, __j += _ARM64_REG_WIDTH)
      {
        neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
        vst1q_f32(&(__ret.__data_[__j]), __vec);
      }

      for (index_type __i = __vector_end, __j = __vector_end - __start_i; __i < __end_i; __i++, __j++)
      {
        __ret.__data_[__j] = this->__data_[__i];
      }
    }
    else if constexpr (std::is_signed<value_type>::value && __step == 1)
    {
      for (index_type __i = __start_i, __j = 0; __i < __vector_end; __i += _ARM64_REG_WIDTH, __j += _ARM64_REG_WIDTH)
      {
        neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
        vst1q_s32(&(__ret.__data_[__j]), __vec);
      }
    }
    else if constexpr (std::is_unsigned<value_type>::value && __step == 1)
    {
      for (index_type __i = __start_i, __j = 0; __i < __vector_end; __i += _ARM64_REG_WIDTH, __j += _ARM64_REG_WIDTH)
      {
        neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
        vst1q_u32(&(__ret.__data_[__j]), __vec);
      }
    }

    for (index_type __i = __vector_end, __j = __vector_end - __start_i; __i < __end_i; __i++, __j++)
    {
      __ret.__data_[__j] = this->__data_[__i];
    }

#endif
    index_type __i = __start_i, __j = 0;

    for (; __i < __end_i; __i += __step, __j++)
    {
      __ret({__j}) = this->at({__i});
    }
#if defined(__CUDACC__)
  }
#endif

  return __ret;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::cumprod(index_type __dim) const {
  if (__dim == -1)
  {
    data_t __flat = this->__data_;
    data_t __ret(__flat.size());
    __ret[0] = __flat[0];

#if defined(__AVX2__)
    if constexpr (std::is_same_v<_Tp, _f32>)
    {
      index_type __i = 1;
      for (; __i + _AVX_REG_WIDTH <= __flat.size(); __i += _AVX_REG_WIDTH)
      {
        __m256 __prev   = _mm256_loadu_ps(&__ret[__i - 1]);
        __m256 __curr   = _mm256_loadu_ps(&__flat[__i]);
        __m256 __result = _mm256_mul_ps(__prev, __curr);
        _mm256_storeu_ps(&__ret[__i], __result);
      }

      for (; __i < __flat.size(); __i++)
      {
        __ret[__i] = __ret[__i - 1] * __flat[__i];
      }
    }
    else
    {
      index_type __i = 1;
      for (; __i < __flat.size(); __i++)
      {
        __ret[__i] = __ret[__i - 1] * __flat[__i];
      }
    }
#else
    index_type __i = 1;

    for (; __i < __flat.size(); __i++)
    {
      __ret[__i] = __ret[__i - 1] * __flat[__i];
    }
#endif

    return __self(__ret, {__flat.size()});
  }
  else
  {
    if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
    {
      throw std::invalid_argument("Invalid dimension provided.");
    }

    data_t __ret(this->__data_);
    // TODO : compute_outer_size() implementation
    index_type __outer_size = this->__compute_outer_size(__dim);
    index_type __inner_size = this->__shape_[__dim];
    index_type __st         = this->__strides_[__dim];

#if defined(__AVX2__)

    if constexpr (std::is_same_v<_Tp, _f32>)
    {
      for (index_type __i = 0; __i < __outer_size; __i++)
      {
        index_type __base = __i * __st;
        __ret[__base]     = __data_[__base];
        index_type __j    = 1;

        for (; __j + _AVX_REG_WIDTH <= __inner_size; __j += _AVX_REG_WIDTH)
        {
          __m256 __prev   = _mm256_loadu_ps(&__ret[__base + __j - 1]);
          __m256 __curr   = _mm256_loadu_ps(&__data_[__base + __j]);
          __m256 __result = _mm256_mul_ps(__prev, __curr);
          _mm256_storeu_ps(&__ret[__base + __j], __result);
        }

        for (; __j < __inner_size; __j++)
        {
          index_type __curr = __base + __j;
          __ret[__curr]     = __ret[__base + __j - 1] * __data_[__curr];
        }
      }
    }
    else
    {
      for (index_type __i = 0; __i < __outer_size; ++__i)
      {
        index_type __base = __i * __st;
        __ret[__base]     = __data_[__base];

        for (index_type __j = 1; __j < __inner_size; __j++)
        {
          index_type __curr = __base + __j;
          __ret[__curr]     = __ret[__base + __j - 1] * __data_[__curr];
        }
      }
    }

#else
    index_type __i = 0;

    for (; __i < __outer_size; __i++)
    {
      index_type __base = __i * __st;
      __ret[__base]     = __data_[__base];
      index_type __j    = 1;

      for (; __j < __inner_size; __j++)
      {
        index_type __curr = __base + __j;
        __ret[__curr]     = __ret[__base + __j - 1] * __data_[__curr];
      }
    }
#endif

    return __self(__ret, this->__shape_);
  }
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::cat(const std::vector<tensor<_Tp>>& __others, index_type __dim) const {
  for (const tensor& __t : __others)
  {
    index_type __i = 0;
    for (; __i < this->__shape_.size(); __i++)
    {
      if (__i != __dim && this->__shape_[__i] != __t.__shape_[__i])
      {
        throw std::invalid_argument(
          "Cannot concatenate tensors with different shapes along non-concatenation dimensions");
      }
    }
  }

  shape_type __ret_sh = this->__shape_;

  for (const tensor& __t : __others)
  {
    __ret_sh[__dim] += __t.__shape_[__dim];
  }

  data_t __c;
  __c.reserve(this->__data_.size());
  __c.insert(__c.end(), this->__data_.begin(), this->__data_.end());

  for (const tensor& __t : __others)
  {
    __c.insert(__c.end(), __t.__data_.begin(), __t.__data_.end());
  }

  return __self(__c, __ret_sh);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::transpose_() const {
  this->__check_is_scalar_type("Cannot transpose a non-scalar tensor");

  if (this->__shape_.size() != 2)
  {
    throw std::runtime_error("Transpose operation is only valid for 2D tensors");
  }

  const auto rows = this->__shape_[0];
  const auto cols = this->__shape_[1];

  if (rows != cols)
  {
    throw std::runtime_error("In-place transpose is only supported for square tensors");
  }

  for (index_type i = 0; i < rows; i++)
  {
    for (index_type j = i + 1; j < cols; j++)
    {
      std::swap(this->__data_[i * cols + j], this->__data_[j * cols + i]);
    }
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::log_softmax_(const index_type __dim) const {
  this->__check_is_scalar_type("Cannot apply log_softmax on a non-scalar tensor");

  assert(__dim < this->__shape_.size() && "Dimension out of range for log_softmax");

  tensor<value_type> __max_values  = this->max(__dim);
  tensor<value_type> __shifted     = *this - __max_values.expand_as(this->__shape_, __dim);
  tensor<value_type> __exp_values  = __shifted.exp();
  tensor<value_type> __sum_exp     = __exp_values.sum(__dim);
  tensor<value_type> __log_sum_exp = __sum_exp.log();
  *this                            = __shifted - __log_sum_exp.expand_as(this->__shape_, __dim);

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::det() const {
  this->__check_is_arithmetic_type("det: template type must be an arithmetic type");

  if (this->__shape_.size() != 2 || this->__shape_[0] != this->__shape_[1])
  {
    throw std::invalid_argument("det: tensor must be a square matrix (n x n)");
  }

  index_type n = this->__shape_[0];

  if (n == 2)
  {
    return tensor<_Tp>(this->__data_[0] * this->__data_[3] - this->__data_[1] * this->__data_[2]);
  }

  value_type determinant = 0;
  tensor     minor;

  for (index_type col = 0; col < n; ++col)
  {
    minor           = this->get_minor(0, col);
    value_type sign = (col % 2 == 0) ? 1 : -1;
    determinant += sign * this->__data_[col] * minor.det();
  }

  return __self(determinant);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::clipped_relu() const {
  __self __ret = this->clone();
  __ret.clipped_relu_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::squeeze(index_type __dim) const {
  __self __ret = this->clone();
  __ret.squeeze_(__dim);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::squeeze_(index_type __dim) const {
  return *this;
}