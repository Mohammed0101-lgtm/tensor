#pragma once

#include "tensorbase.hpp"
#include "types.hpp"

template <class _Tp>
tensor<_Tp> tensor<_Tp>::matmul(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_matmul(__other);
#endif
  static_assert(has_times_operator_v<value_type>);
  static_assert(has_plus_operator_v<value_type>);

  if (this->__shape_.size() != 2 || __other.shape().size() != 2)
    throw std::invalid_argument("matmul is only supported for 2D tensors");

  if (this->__shape_[1] != __other.shape()[0]) {
    if (this->__shape_[0] == __other.shape()[1]) return __other.matmul(*this);

    throw std::invalid_argument(
        "Shape mismatch for matrix multiplication: "
        "this shape: [" +
        std::to_string(this->__shape_[0]) + ", " + std::to_string(this->__shape_[1]) +
        "] "
        "other shape: [" +
        std::to_string(__other.shape()[0]) + ", " + std::to_string(__other.shape()[1]) + "]");
  }

  shape_type __ret_sh = {this->__shape_[0], __other.shape()[1]};
  data_t     __ret_d(__ret_sh[0] * __ret_sh[1], value_type(0));

#pragma omp parallel for collapse(2)
  for (int64_t __i = 0; __i < __ret_sh[0]; ++__i) {
    for (int64_t __j = 0; __j < __ret_sh[1]; ++__j) {
      value_type __sum = value_type(0);
      for (int64_t __k = 0; __k < this->__shape_[1]; ++__k)
        __sum = __sum + (this->__data_[__i * this->__shape_[1] + __k] *
                         __other[__k * __other.shape()[1] + __j]);

      __ret_d[__i * __ret_sh[1] + __j] = __sum;
    }
  }

  return __self(__ret_d, __ret_sh);
}

#ifdef __CUDACC__
template <class _Tp>
__global__ void matmul_kernel(_Tp* __a, _Tp* __b, _Tp* __c, int __m, int __n, int __k) {
  int __row = blockIdx.y * blockDim.y + threadIdx.y;
  int __col = blockIdx.x * blockDim.x + threadIdx.x;

  if (__row < __m && __col < __k) {
    _Tp __sum = 0;
    for (int __i = 0; __i < __n; ++__i) __sum += __a[__row * __n + __i] * __b[__i * __k + __col];

    __c[__row * __k + __col] = __sum;
  }
}
#endif

template <class _Tp>
tensor<_Tp> tensor<_Tp>::reshape(const shape_type __sh) const {
  data_t     __d = this->__data_;
  index_type __s = this->__computeSize(__sh);

  if (__s != this->__data_.size())
    throw std::invalid_argument(
        "input shape must have size of elements equal to the current number of elements in the "
        "tensor data");

  return __self(__d, __sh);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::cross_product(const tensor& __other) const {
  if (this->empty() || __other.empty())
    throw std::invalid_argument("Cannot cross product an empty vector");

  if (this->shape() != std::vector<int>{3} || __other.shape() != std::vector<int>{3})
    throw std::invalid_argument("Cross product can only be performed on 3-element vectors");

  tensor __ret({3});

#if defined(__ARM_NEON) && defined(__aarch64__)
  return this->neon_cross_product(__other);
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
template <class _Tp>
__global__ void cross_product_kernel(_Tp* __a, _Tp* __b, _Tp* __c) {
  __c[0] = __a[1] * __b[2] - __a[2] * __b[1];
  __c[1] = __a[2] * __b[0] - __a[0] * __b[2];
  __c[2] = __a[0] * __b[1] - __a[1] * __b[0];
}
#endif

template <class _Tp>
tensor<_Tp> tensor<_Tp>::absolute(const tensor& __tensor) const {
  index_type __s = __tensor.storage().size();
  data_t     __a;
  __a.reserve(__s);
  index_type __i = 0;

#ifdef __AVX__
  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i + _AVX_REG_WIDTH <= __s; __i += _AVX_REG_WIDTH) {
      __m256 __input     = _mm256_loadu_ps(&__tensor.storage()[__i]);
      __m256 __abs_value = _mm256_abs_ps(__input);
      _mm256_storeu_ps(&__a[__i], __abs_value);
    }
  }
#endif

#if defined(__SSE__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    for (; __i + 4 <= __s; __i += 4) {
      __m128 __input     = _mm_loadu_ps(&__tensor.storage()[__i]);
      __m128 __abs_value = _mm_abs_ps(__input);
      _mm_storeu_ps(&__a[__i], __abs_value);
    }
  }
#endif

  for (; __i < __s; ++__i)
    __a.push_back(static_cast<value_type>(std::fabs(_f32(__tensor.storage()[__i]))));

  return __self(__a, __tensor.__shape_);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::dot(const tensor& __other) const {
#if defined(__ARM_NEON)
  return this->neon_dot(__other);
#endif
  if (this->empty() || __other.empty())
    throw std::invalid_argument("Cannot dot product an empty vector");

  if (this->__shape_.size() == 1 && __other.shape().size() == 1) {
    if (this->__shape_[0] != __other.shape()[0])
      throw std::invalid_argument("Vectors must have the same size for dot product");

    const_pointer __this_data  = this->__data_.data();
    const_pointer __other_data = __other.storage().data();
    const size_t  __size       = this->__data_.size();
    value_type    __ret        = 0;

    __ret = std::inner_product(__this_data, __this_data + __size, __other_data, value_type(0));
    return __self({__ret}, {1});
  }

  if (this->__shape_.size() == 2 && __other.shape().size() == 2) return this->matmul(__other);

  if (this->__shape_.size() == 3 && __other.shape().size() == 3)
    return this->cross_product(__other);

  return __self();
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::relu() const {
  __self __ret = this->clone();
  __ret.relu_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::relu_() {
#if defined(__ARM_NEON)
  return this->neon_relu_();
#endif
  if constexpr (std::is_unsigned_v<value_type>) return *this;

  index_type __s = this->__data_.size();
  index_type __i = 0;

#pragma omp parallel

#ifdef __CUDACC__
  if (this->__is_cuda_tensor) {
    pointer __d_data = thrust::raw_pointer_cast(this->__data_.data());
    thrust::transform(thrust::device, __d_data, d_data + __s, __d_data,
                      [] __device__(value_type __x) { return max(__x, value_type(0)); });
    return;
  }

#elif defined(__SSE__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    __m128 __zero = _mm_setzero_ps();

    for (; __i + 4 <= __s; __i += 4) {
      __m128 __x      = _mm_loadu_ps(&this->__data_[__i]);
      __m128 __result = _mm_max_ps(__x, __zero);
      _mm_storeu_ps(&this->__data_[__i], __result);
    }
  }

#elif defined(__AVX__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    __m256 __zero = _mm256_setzero_ps();

    for (; __i + _AVX_REG_WIDTH <= __s; __i += _AVX_REG_WIDTH) {
      __m256 __x      = _mm256_loadu_ps(&this->__data_[__i]);
      __m256 __result = _mm256_max_ps(__x, __zero);
      _mm256_storeu_ps(&this->__data_[__i], __result);
    }
  }
#endif

  for (__i = 0; __i < __s; ++__i) this->__data_[__i] = std::max(this->__data_[__i], value_type(0));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::relu_() const {
#if defined(__ARM_NEON)
  return this->neon_relu_();
#endif
  if constexpr (std::is_unsigned_v<value_type>) return *this;

  index_type __s = this->__data_.size();
  index_type __i = 0;

#pragma omp parallel

#ifdef __CUDACC__
  if (this->__is_cuda_tensor) {
    pointer __d_data = thrust::raw_pointer_cast(this->__data_.data());
    thrust::transform(thrust::device, __d_data, d_data + __s, __d_data,
                      [] __device__(value_type __x) { return max(__x, value_type(0)); });
    return;
  }

#elif defined(__SSE__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    __m128 __zero = _mm_setzero_ps();

    for (; __i + 4 <= __s; __i += 4) {
      __m128 __x      = _mm_loadu_ps(&this->__data_[__i]);
      __m128 __result = _mm_max_ps(__x, __zero);
      _mm_storeu_ps(&this->__data_[__i], __result);
    }
  }

#elif defined(__AVX__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    __m256 __zero = _mm256_setzero_ps();

    for (; __i + _AVX_REG_WIDTH <= __s; __i += _AVX_REG_WIDTH) {
      __m256 __x      = _mm256_loadu_ps(&this->__data_[__i]);
      __m256 __result = _mm256_max_ps(__x, __zero);
      _mm256_storeu_ps(&this->__data_[__i], __result);
    }
  }
#endif

  for (__i = 0; __i < __s; ++__i) this->__data_[__i] = std::max(this->__data_[__i], value_type(0));

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::transpose() const {
  if (this->__shape_.size() != 2)
    throw std::invalid_argument("Matrix transposition can only be done on 2D tensors");

  tensor           __ret({this->__shape_[1], this->__shape_[0]});
  const index_type __rows = this->__shape_[0];
  const index_type __cols = this->__shape_[1];

#ifdef __CUDACC__
  if (this->__is_cuda_tensor) {
    dim3 blockDim(16, 16);
    dim3 gridDim((__cols + blockDim.x - 1) / blockDim.x, (__rows + blockDim.y - 1) / blockDim.y);
    transpose_kernel<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(this->__data_.data()),
                                            thrust::raw_pointer_cast(__ret.__data_.data()), __rows,
                                            __cols);
    cudaDeviceSynchronize();
    return __ret;
  }
#endif

#if defined(__ARM_NEON)
  return this->neon_transpose();
#endif

  index_type __i = 0;

  for (; __i < __rows; ++__i) {
    index_type __j = 0;

    for (; __j < __cols; ++__j) __ret.at({__j, __i}) = this->at({__i, __j});
  }

  return __ret;
}

#ifdef __CUDACC__
template <class _Tp>
__global__ void transpose_kernel(_Tp* __input, _Tp* __output, int __rows, int __cols) {
  int __i = blockIdx.y * blockDim.y + threadIdx.y;
  int __j = blockIdx.x * blockDim.x + threadIdx.x;

  if (__i < __rows && __j < __cols) output[__j * __rows + __i] = input[__i * __cols + __j];
}
#endif

template <class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argsort(index_type __d,
                                                              bool       __ascending) const {
#if defined(__ARM_NEON)
  return this->neon_argsort(__d, __ascending);
#endif
  index_type __adjusted = (__d < 0) ? __d + this->__data_.size() : __d;

  if (__adjusted != 0)
    throw std::out_of_range("Invalid dimension for argsort: only 1D tensors are supported");

  index_type __size = static_cast<index_type>(this->__data_.size());
  shape_type __indices(__size);
  std::iota(__indices.begin(), __indices.end(), 0);
  std::sort(__indices.begin(), __indices.end(), [&](index_type __a, index_type __b) {
    return __ascending ? this->__data_[__a] < this->__data_[__b]
                       : this->__data_[__a] > this->__data_[__b];
  });

  return tensor<index_type>(__indices);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::sigmoid() const {
  __self __ret = this->clone();
  __ret.sigmoid_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::sigmoid_() {
#if defined(__ARM_NEON)
  return this->neon_sigmoid_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] =
        static_cast<value_type>(1.0 / (1.0 + std::exp(-static_cast<double>(this->__data_[__i]))));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::sigmoid_() const {
#if defined(__ARM_NEON)
  return this->neon_sigmoid_();
#endif
  index_type __i = 0;
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] =
        static_cast<value_type>(1.0 / (1.0 + std::exp(-static_cast<double>(this->__data_[__i]))));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::clipped_relu_(const value_type __clip_limit) {
#if defined(__ARM_NEON)
  return this->neon_clipped_relu_(__clip_limit);
#endif
  if constexpr (std::is_unsigned_v<value_type>) return *this;

  index_type __s = this->__data_.size();
  index_type __i = 0;

#pragma omp parallel

#ifdef __CUDACC__
  if (this->__is_cuda_tensor) {
    pointer __d_data = thrust::raw_pointer_cast(this->__data_.data());
    thrust::transform(
        thrust::device, __d_data, __d_data + __s, __d_data,
        [] __device__(value_type __x) { return min(max(__x, value_type(0)), __clip_limit); });
    return *this;
  }

#elif defined(__SSE__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    __m128 __zero = _mm_setzero_ps();
    __m128 __clip = _mm_set1_ps(__clip_limit);

    for (; __i + 4 <= __s; __i += 4) {
      __m128 __x      = _mm_loadu_ps(&this->__data_[__i]);
      __m128 __result = _mm_min_ps(_mm_max_ps(__x, __zero), __clip);

      _mm_storeu_ps(&this->__data_[__i], __result);
    }
  }

#elif defined(__AVX__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    __m256 __zero = _mm256_setzero_ps();
    __m256 __clip = _mm256_set1_ps(__clip_limit);

    for (; __i + _AVX_REG_WIDTH <= __s; __i += _AVX_REG_WIDTH) {
      __m256 __x      = _mm256_loadu_ps(&this->__data_[__i]);
      __m256 __result = _mm256_min_ps(_mm256_max_ps(__x, __zero), __clip);

      _mm256_storeu_ps(&this->__data_[__i], __result);
    }
  }
#endif

  for (; __i < __s; ++__i)
    this->__data_[__i] = std::min(std::max(this->__data_[__i], value_type(0)), __clip_limit);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::clipped_relu_(const value_type __clip_limit) const {
#if defined(__ARM_NEON)
  return this->neon_clipped_relu_(__clip_limit);
#endif
  if constexpr (std::is_unsigned_v<value_type>) return *this;

  index_type __s = this->__data_.size();
  index_type __i = 0;

#pragma omp parallel

#ifdef __CUDACC__
  if (this->__is_cuda_tensor) {
    pointer __d_data = thrust::raw_pointer_cast(this->__data_.data());
    thrust::transform(
        thrust::device, __d_data, __d_data + __s, __d_data,
        [] __device__(value_type __x) { return min(max(__x, value_type(0)), __clip_limit); });
    return *this;
  }

#elif defined(__SSE__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    __m128 __zero = _mm_setzero_ps();
    __m128 __clip = _mm_set1_ps(__clip_limit);

    for (; __i + 4 <= __s; __i += 4) {
      __m128 __x      = _mm_loadu_ps(&this->__data_[__i]);
      __m128 __result = _mm_min_ps(_mm_max_ps(__x, __zero), __clip);

      _mm_storeu_ps(&this->__data_[__i], __result);
    }
  }

#elif defined(__AVX__)
  if constexpr (std::is_same_v<value_type, _f32>) {
    __m256 __zero = _mm256_setzero_ps();
    __m256 __clip = _mm256_set1_ps(__clip_limit);

    for (; __i + _AVX_REG_WIDTH <= __s; __i += _AVX_REG_WIDTH) {
      __m256 __x      = _mm256_loadu_ps(&this->__data_[__i]);
      __m256 __result = _mm256_min_ps(_mm256_max_ps(__x, __zero), __clip);

      _mm256_storeu_ps(&this->__data_[__i], __result);
    }
  }
#endif

  for (; __i < __s; ++__i)
    this->__data_[__i] = std::min(std::max(this->__data_[__i], value_type(0)), __clip_limit);

  return *this;
}

template <class _Tp>
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

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::unsqueeze_(index_type __dim) {
  if (__dim < 0 || __dim > static_cast<index_type>(this->__shape_.size()))
    throw std::out_of_range("Dimension out of range in unsqueeze");

  this->__shape_.insert(this->__shape_.begin() + __dim, 1);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::unsqueeze_(index_type __dim) const {
  if (__dim < 0 || __dim > static_cast<index_type>(this->__shape_.size()))
    throw std::out_of_range("Dimension out of range in unsqueeze");

  this->__shape_.insert(this->__shape_.begin() + __dim, 1);

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::clamp_(const_reference __min_val, const_reference __max_val) {
  index_type __i = 0;

#if defined(__AVX2__)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _AVX_REG_WIDTH);
  __m256           __min_vec  = _mm256_set1_ps(__min_val);
  __m256           __max_vec  = _mm256_set1_ps(__max_val);

  for (; __i < __simd_end; __i += _AVX_REG_WIDTH) {
    __m256 __data_vec = _mm256_loadu_ps(&this->__data_[__i]);
    __m256 __clamped  = _mm256_min_ps(_mm256_max_ps(data_vec, __min_vec), __max_vec);

    _mm256_storeu_ps(&this->__data_[__i], __clamped);
  }

#elif defined(__ARM_NEON)
  return this->neon_clamp_(__min_val, __max_val);
#endif
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) {
    this->__data_[__i] = std::max(__min_val, this->__data_[__i]);
    this->__data_[__i] = std::min(__max_val, this->__data_[__i]);
  }

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::clamp_(const_reference __min_val,
                                              const_reference __max_val) const {
  index_type __i = 0;

#if defined(__AVX2__)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _AVX_REG_WIDTH);
  __m256           __min_vec  = _mm256_set1_ps(__min_val);
  __m256           __max_vec  = _mm256_set1_ps(__max_val);

  for (; __i < __simd_end; __i += _AVX_REG_WIDTH) {
    __m256 __data_vec = _mm256_loadu_ps(&this->__data_[__i]);
    __m256 __clamped  = _mm256_min_ps(_mm256_max_ps(data_vec, __min_vec), __max_vec);

    _mm256_storeu_ps(&this->__data_[__i], __clamped);
  }

#elif defined(__ARM_NEON)
  return this->neon_clamp_(__min_val, __max_val);
#endif
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i) {
    this->__data_[__i] = std::max(__min_val, this->__data_[__i]);
    this->__data_[__i] = std::min(__max_val, this->__data_[__i]);
  }

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::clamp(const_reference __min_val, const_reference __max_val) const {
  __self __ret = this->clone();
  __ret.clamp_(__min_val, __max_val);
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::floor() const {
  __self __ret = this->clone();
  __ret.floor_();
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::floor_() {
#if defined(__ARM_NEON)
  return this->neon_floor_();
#endif
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::floor(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::floor_() const {
#if defined(__ARM_NEON)
  return this->neon_floor_();
#endif
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::floor(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::ceil_() {
#if defined(__ARM_NEON)
  return this->neon_ceil_();
#endif
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::ceil(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::ceil_() const {
#if defined(__ARM_NEON)
  return this->neon_ceil_();
#endif
  static_assert(std::is_floating_point_v<value_type>);
  index_type __i = 0;
#pragma omp parallel
  for (; __i < this->__data_.size(); ++__i)
    this->__data_[__i] = static_cast<value_type>(std::ceil(static_cast<_f32>(this->__data_[__i])));

  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::ceil() const {
  __self __ret = this->clone();
  __ret.ceil_();
  return __ret;
}

template <class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argmax_(index_type __dim) const {
  if (__dim < 0 || __dim >= this->__shape_.size())
    throw std::out_of_range("Dimension out of range in argmax");

  tensor<index_type> __ret;
  shape_type         __ret_sh = this->__shape_;
  __ret_sh.erase(__ret_sh.begin() + __dim);
  __ret.__shape_ = __ret_sh;
  __ret.__data_.resize(this->__computeSize(__ret_sh), 0);

  index_type __outer_size = 1;
  index_type __inner_size = 1;
  index_type __i          = 0;

  for (; __i < __dim; ++__i) __outer_size *= this->__shape_[__i];
  for (__i = __dim + 1; __i < this->__shape_.size(); ++__i) __inner_size *= this->__shape_[__i];

#if defined(__AVX2__)
  if constexpr (std::is_same_v<_Tp, _f32>) {
    for (__i = 0; __i < __outer_size; ++__i) {
      index_type __j = 0;
      for (; __j < __inner_size; ++__j) {
        __m256     __max_vec       = _mm256_set1_ps(-std::numeric_limits<_f32>::infinity());
        __m256i    __index_vec     = _mm256_setzero_si256();
        __m256i    __increment     = _mm256_set1_epi32(1);
        __m256i    __current_index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        index_type __k             = 0;

        for (; __k + _AVX_REG_WIDTH <= this->__shape_[__dim]; __k += _AVX_REG_WIDTH) {
          __m256 __data_vec = _mm256_loadu_ps(
              &this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
          __m256 __mask = _mm256_cmp_ps(__data_vec, __max_vec, _CMP_GT_OQ);
          __max_vec     = _mm256_blendv_ps(__max_vec, __data_vec, __mask);
          __index_vec =
              _mm256_blendv_epi8(__index_vec, __current_index, _mm256_castps_si256(__mask));
          __current_index = _mm256_add_epi32(__current_index, __increment);
        }

        _f32 __max_values[_AVX_REG_WIDTH];
        _s32 __indices[_AVX_REG_WIDTH];

        _mm256_storeu_ps(__max_values, __max_vec);
        _mm256_storeu_si256((__m256i*)__indices, __index_vec);

        _f32       __max_value = __max_values[0];
        index_type __max_index = __indices[0];

        for (int __k = 1; __k < _AVX_REG_WIDTH; ++__k) {
          if (__max_values[__k] > __max_value) {
            __max_value = __max_values[__k];
            __max_index = __indices[__k];
          }
        }

        for (; __k < this->__shape_[__dim]; ++__k) {
          _f32 __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          if (__v > __max_value) {
            __max_value = __v;
            __max_index = __k;
          }
        }
        __ret.__data_[__i * __inner_size + __j] = __max_index;
      }
    }
  }

#elif defined(__ARM_NEON)
  return this->neon_argmax_(__dim);
#endif
  {
    for (__i = 0; __i < __outer_size; ++__i) {
      index_type __j = 0;
      for (; __j < __inner_size; ++__j) {
        index_type __max_index = 0;
        value_type __max_value = this->__data_[__i * this->__shape_[__dim] * __inner_size + __j];
        index_type __k         = 1;
        for (; __k < this->__shape_[__dim]; ++__k) {
          value_type __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];

          if (__v > __max_value) {
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

template <class _Tp>
tensor<_Tp> tensor<_Tp>::argmax(index_type __dim) const {
  if (__dim < 0 || __dim >= this->__shape_.size())
    throw std::out_of_range("Dimension out of range in argmax");

  tensor     __ret;
  shape_type __ret_sh = this->__shape_;

  __ret_sh.erase(__ret_sh.begin() + __dim);
  __ret.__shape_ = __ret_sh;
  __ret.__data_.resize(this->__computeSize(__ret_sh), value_type(0));

  index_type __outer_size = 1;
  index_type __inner_size = 1;
  index_type __i          = 0;

  for (; __i < __dim; ++__i) __outer_size *= this->__shape_[__i];
  for (__i = __dim + 1; __i < static_cast<index_type>(this->__shape_.size()); ++__i)
    __inner_size *= this->__shape_[__i];
#if defined(__AVX2__)
  if constexpr (std::is_same_v<_Tp, _f32>) {
    for (__i = 0; __i < __outer_size; ++__i) {
      for (index_type __j = 0; __j < __inner_size; ++__j) {
        __m256     __max_vec = _mm256_set1_ps(-std::numeric_limits<_f32>::infinity());
        index_type __k       = 0;

        for (; __k + _AVX_REG_WIDTH <= this->__shape_[__dim]; __k += _AVX_REG_WIDTH) {
          __m256 __data_vec = _mm256_loadu_ps(
              &this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
          __max_vec = _mm256_max_ps(__max_vec, __data_vec);
        }

        _f32 __max_value = _mm256_reduce_max_ps(__max_vec);
        for (; __k < this->__shape_[__dim]; ++__k) {
          _f32 __v    = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          __max_value = std::max(__max_value, __v);
        }

        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  }

#elif defined(__ARM_NEON)
  return this->neon_argmax(__dim);
#endif
  {
    for (__i = 0; __i < __outer_size; ++__i) {
      index_type __j = 0;
      for (; __j < __inner_size; ++__j) {
        value_type __max_value = this->__data_[__i * this->__shape_[__dim] * __inner_size + __j];
        index_type __k         = 1;
        for (; __k < this->__shape_[__dim]; ++__k) {
          value_type __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];

          if (__v > __max_value) __max_value = __v;
        }
        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  }

  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::pop_back() const {
  if (this->__shape_.size() != 1)
    throw std::range_error("push_back is only supported for one dimensional tensors");

  this->__data_.pop_back();
  --(this->__shape_[0]);
  this->__compute_strides();
  return *this;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::sum(const index_type __axis) const {
#if defined(__ARM_NEON)
  return this->neon_sum(__axis);
#endif
  if (__axis < 0 || __axis >= static_cast<index_type>(this->__shape_.size()))
    throw std::invalid_argument("Invalid axis for sum");

  shape_type __ret_sh = this->__shape_;
  __ret_sh[__axis]    = 1;
  index_type __ret_size =
      std::accumulate(__ret_sh.begin(), __ret_sh.end(), 1, std::multiplies<index_type>());
  data_t __ret_data(__ret_size, value_type(0.0f));

  index_type __i = 0;
  for (; __i < static_cast<index_type>(this->__data_.size()); ++__i) {
    std::vector<index_type> __orig(this->__shape_.size());
    index_type              __index = __i;
    index_type              __j     = static_cast<index_type>(this->__shape_.size()) - 1;

    for (; __j >= 0; __j--) {
      __orig[__j] = __index % this->__shape_[__j];
      __index /= this->__shape_[__j];
    }

    __orig[__axis]         = 0;
    index_type __ret_index = 0;
    index_type __st        = 1;

    for (__j = static_cast<index_type>(this->__shape_.size()) - 1; __j >= 0; __j--) {
      __ret_index += __orig[__j] * __st;
      __st *= __ret_sh[__j];
    }
    __ret_data[__ret_index] += this->__data_[__i];
  }

  return __self(__ret_data, __ret_sh);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::slice(index_type __dim, std::optional<index_type> __start,
                               std::optional<index_type> __end, index_type __step) const {
  if (this->empty()) return __self();

  if (__dim <= 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
    throw std::invalid_argument("Invalid dimension provided");

  if (__start.has_value() && __end.has_value() && __start.value() >= __end.value())
    throw std::invalid_argument("Start index must be less than end index");

  if (__step == 0) throw std::invalid_argument("Step cannot be zero");

  index_type __start_val = __start.has_value() ? __start.value() : 0;
  index_type __end_val   = __end.has_value() ? __end.value() : this->__shape_[__dim - 1];

  if (__start_val < 0 || __start_val >= this->__shape_[__dim - 1])
    throw std::out_of_range("Start index out of range");

  if (__end_val < 0 || __end_val > this->__shape_[__dim - 1])
    throw std::out_of_range("End index out of range");

  if (__step < 0 && __start_val < __end_val)
    throw std::invalid_argument("Step must be positive");
  else if (__start_val > __end_val && __step > 0)
    throw std::invalid_argument("Step must be negative");

  shape_type __ret_sh = this->__shape_;
  __ret_sh[__dim - 1] = (__end_val - __start_val) / __step;

  data_t     __ret_data;
  index_type __i = 0;

  for (; __i < this->__data_.size(); ++__i) {
    shape_type __orig(this->__shape_.size());
    index_type __index = __i;
    index_type __j     = static_cast<index_type>(this->__shape_.size()) - 1;

    for (; __j >= 0; __j--) {
      __orig[__j] = __index % this->__shape_[__j];
      __index /= this->__shape_[__j];
    }

    if (__orig[__dim - 1] < __start_val || __orig[__dim - 1] >= __end_val ||
        (__orig[__dim - 1] - __start_val) % __step != 0)
      continue;

    __orig[__dim - 1]      = (__orig[__dim - 1] - __start_val) / __step;
    index_type __ret_index = 0;
    index_type __st        = 1;

    for (__j = static_cast<index_type>(this->__shape_.size()) - 1; __j >= 0; __j--) {
      __ret_index += __orig[__j] * __st;
      __st *= __ret_sh[__j];
    }

    __ret_data.push_back(this->__data_[__i]);
  }

  return __self(__ret_sh, __ret_data);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::cumprod(index_type __dim) const {
  if (__dim == -1) {
    data_t __flat = this->__data_;
    data_t __ret(__flat.size());
    __ret[0] = __flat[0];

#if defined(__AVX2__)
    if constexpr (std::is_same_v<_Tp, _f32>) {
      index_type __i = 1;
      for (; __i + _AVX_REG_WIDTH <= __flat.size(); __i += _AVX_REG_WIDTH) {
        __m256 __prev   = _mm256_loadu_ps(&__ret[__i - 1]);
        __m256 __curr   = _mm256_loadu_ps(&__flat[__i]);
        __m256 __result = _mm256_mul_ps(__prev, __curr);
        _mm256_storeu_ps(&__ret[__i], __result);
      }

      for (; __i < __flat.size(); ++__i) __ret[__i] = __ret[__i - 1] * __flat[__i];
    } else {
      index_type __i = 1;
      for (; __i < __flat.size(); ++__i) __ret[__i] = __ret[__i - 1] * __flat[__i];
    }
#else
    index_type __i = 1;

    for (; __i < __flat.size(); ++__i) __ret[__i] = __ret[__i - 1] * __flat[__i];
#endif

    return __self(__ret, {__flat.size()});
  } else {
    if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
      throw std::invalid_argument("Invalid dimension provided.");

    data_t __ret(this->__data_);
    // TODO : compute_outer_size() implementation
    index_type __outer_size = this->__compute_outer_size(__dim);
    index_type __inner_size = this->__shape_[__dim];
    index_type __st         = this->__strides_[__dim];

#if defined(__AVX2__)

    if constexpr (std::is_same_v<_Tp, _f32>) {
      for (index_type __i = 0; __i < __outer_size; ++__i) {
        index_type __base = __i * __st;
        __ret[__base]     = __data_[__base];
        index_type __j    = 1;

        for (; __j + _AVX_REG_WIDTH <= __inner_size; __j += _AVX_REG_WIDTH) {
          __m256 __prev   = _mm256_loadu_ps(&__ret[__base + __j - 1]);
          __m256 __curr   = _mm256_loadu_ps(&__data_[__base + __j]);
          __m256 __result = _mm256_mul_ps(__prev, __curr);
          _mm256_storeu_ps(&__ret[__base + __j], __result);
        }

        for (; __j < __inner_size; ++__j) {
          index_type __curr = __base + __j;
          __ret[__curr]     = __ret[__base + __j - 1] * __data_[__curr];
        }
      }
    } else {
      for (index_type __i = 0; __i < __outer_size; ++__i) {
        index_type __base = __i * __st;
        __ret[__base]     = __data_[__base];

        for (index_type __j = 1; __j < __inner_size; ++__j) {
          index_type __curr = __base + __j;
          __ret[__curr]     = __ret[__base + __j - 1] * __data_[__curr];
        }
      }
    }

#else
    index_type __i = 0;

    for (; __i < __outer_size; ++__i) {
      index_type __base = __i * __st;
      __ret[__base]     = __data_[__base];
      index_type __j    = 1;

      for (; __j < __inner_size; ++__j) {
        index_type __curr = __base + __j;
        __ret[__curr]     = __ret[__base + __j - 1] * __data_[__curr];
      }
    }
#endif

    return __self(__ret, this->__shape_);
  }
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::cat(const std::vector<tensor<_Tp>>& __others, index_type __dim) const {
  for (const tensor& __t : __others) {
    index_type __i = 0;
    for (; __i < this->__shape_.size(); ++__i)
      if (__i != __dim && this->__shape_[__i] != __t.__shape_[__i])
        throw std::invalid_argument(
            "Cannot concatenate tensors with different shapes along non-concatenation "
            "dimensions");
  }

  shape_type __ret_sh = this->__shape_;
  for (const tensor& __t : __others) __ret_sh[__dim] += __t.__shape_[__dim];

  data_t __c;
  __c.reserve(this->__data_.size());
  __c.insert(__c.end(), this->__data_.begin(), this->__data_.end());
  for (const tensor& __t : __others) __c.insert(__c.end(), __t.__data_.begin(), __t.__data_.end());

  return __self(__ret_sh, __c);
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::transpose_() {
  if (this->__shape_.size() != 2)
    throw std::runtime_error("Transpose operation is only valid for 2D tensors");

  const index_type __rows = this->__shape_[0];
  const index_type __cols = this->__shape_[1];

  if (__rows != __cols)
    throw std::runtime_error("In-place transpose is only supported for square tensors");

  for (index_type __i = 0; __i < __rows; ++__i)
    for (index_type __j = __i + 1; __j < __cols; ++__j)
      std::swap(this->__data_[__i * __cols + __j], this->__data_[__j * __cols + __i]);

  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::transpose_() const {
  return this->transpose_();
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::log_softmax_(const index_type __dim) {
  assert(__dim < this->__shape_.size() && "Dimension out of range for log_softmax");
  tensor __max_values  = this->argmax_(__dim);
  tensor __shifted     = *this - __max_values.expand_as(this->__shape_, __dim);
  tensor __exp_values  = __shifted.exp();
  tensor __sum_exp     = __exp_values.sum(__dim);
  tensor __log_sum_exp = __sum_exp.log();
  *this                = __shifted - __log_sum_exp.expand_as(this->__shape_, __dim);
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::log_softmax_(const index_type __dim) const {
  return this->log_softmax_(__dim);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::det() const {
  if (this->__shape_.size() != 2 || this->__shape_[0] != this->__shape_[1])
    throw std::invalid_argument("det: tensor must be a square matrix (n x n)");

  index_type __n = this->__shape_[0];

  if (__n == 2)
    return tensor<_Tp>(this->__data_[0] * this->__data_[3] - this->__data_[1] * this->__data_[2]);

  value_type __determinant = 0;
  tensor     __minor;

  for (index_type __col = 0; __col < __n; ++__col) {
    __minor           = this->get_minor(0, __col);
    value_type __sign = (__col % 2 == 0) ? 1 : -1;
    __determinant += __sign * this->__data_[__col] * __minor.det();
  }

  return __self(__determinant);
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::clipped_relu() const {
  __self __ret = this->clone();
  __ret.clipped_relu_();
  return __ret;
}

template <class _Tp>
tensor<_Tp> tensor<_Tp>::squeeze(index_type __dim) const {
  __self __ret = this->clone();
  __ret.squeeze_(__dim);
  return __ret;
}

template <class _Tp>
tensor<_Tp>& tensor<_Tp>::squeeze_(index_type __dim) {
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::squeeze_(index_type __dim) const {
  return this->squeeze_(__dim);
}

template <class _Tp>
inline tensor<_Tp>& tensor<_Tp>::resize_as_(const shape_type __sh) {
  // TODO: implement in place resize as here
  return *this;
}

template <class _Tp>
inline const tensor<_Tp>& tensor<_Tp>::resize_as_(const shape_type __sh) const {
  return this->resize_as_(__sh);
}
