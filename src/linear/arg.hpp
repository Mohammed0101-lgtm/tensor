#pragma once

#include "tensorbase.hpp"

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