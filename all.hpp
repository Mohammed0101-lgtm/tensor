#pragma once

//
// tensor base
//

#pragma once

#include <__algorithm/clamp.h>
#include <__algorithm/comp.h>
#include <__algorithm/sort.h>
#include <__functional/hash.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <compare>
#include <complex>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <omp-tools.h>
#include <omp.h>
#include <optional>
#include <random>
#include <stack>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <valarray>
#include <vector>
#include <version>

#if defined(__ARM_NEON)
  #include <arm_neon.h>
  #include <arm_vector_types.h>
#endif

#if defined(__AVX__) || defined(__SSE__)
  #include <immintrin.h>
#endif

#if defined(USE_CUDA)
  #include <cuda_runtime.h>
#endif

#if defined(__ARM_NEON)

using _s32 = int32_t;
using _u32 = uint32_t;
using _f32 = float32_t;
using _f64 = float64_t;

using neon_s32 = int32x4_t;
using neon_u32 = uint32x4_t;
using neon_f32 = float32x4_t;
using neon_f64 = float64x2_t;

#endif

const int _ARM64_REG_WIDTH = 4;
const int _AVX_REG_WIDTH   = 8;


enum class Device {
  CPU,
  CUDA
};


template<class _Tp>
class tensor
{
 public:
  using __self                 = tensor;
  using value_type             = _Tp;
  using data_t                 = std::vector<value_type>;
  using index_type             = int64_t;
  using shape_type             = std::vector<index_type>;
  using reference              = value_type&;
  using const_reference        = const value_type&;
  using pointer                = value_type*;
  using const_pointer          = const value_type*;
  using iterator               = std::__wrap_iter<pointer>;
  using const_iterator         = std::__wrap_iter<const_pointer>;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

 private:
  data_t                  __data_;
  shape_type              __shape_;
  std::vector<index_type> __strides_;
  Device                  __device_;
  bool                    __is_cuda_tensor_ = false;

 public:
  tensor() = default;

  explicit tensor(const shape_type& __sh, const value_type& __v, Device __d = Device::CPU) :
      __shape_(__sh),
      __data_(this->__computeSize(__sh), __v),
      __device_(__d) {
    this->__compute_strides();
  }

  explicit tensor(const shape_type& __sh, Device __d = Device::CPU) :
      __shape_(__sh),
      __device_(__d) {
    index_type __s = this->__computeSize(__sh);
    this->__data_  = data_t(__s);
    this->__compute_strides();
  }

  explicit tensor(const data_t& __d, const shape_type& __sh, Device __dev = Device::CPU) :
      __data_(__d),
      __shape_(__sh),
      __device_(__dev) {
    this->__compute_strides();
  }

  tensor(const tensor& __t) :
      __data_(__t.storage()),
      __shape_(__t.shape()),
      __strides_(__t.strides()),
      __device_(__t.device()) {}

  tensor(tensor&& __t) noexcept :
      __data_(std::move(__t.storage())),
      __shape_(std::move(__t.shape())),
      __strides_(std::move(__t.strides())),
      __device_(std::move(__t.device())) {}

  tensor(const shape_type& __sh, std::initializer_list<value_type> init_list, Device __d = Device::CPU) :
      __shape_(__sh),
      __device_(__d) {
    index_type __s = this->__computeSize(__sh);
    assert(init_list.size() == static_cast<size_t>(__s) && "Initializer list size must match tensor size");
    this->__data_ = data_t(init_list);
    this->__compute_strides();
  }

  tensor(const shape_type& __sh, const tensor& __other) :
      __data_(__other.storage()),
      __shape_(__sh),
      __device_(__other.device()) {
    this->__compute_strides();
  }

 private:
  class __destroy_tensor
  {
   public:
    explicit __destroy_tensor(tensor& __tens) :
        __tens_(__tens) {}

    void operator()() {}

   private:
    tensor& __tens_;
  };

 public:
  ~tensor() { __destroy_tensor (*this)(); }

  data_t storage() const noexcept;

  iterator begin() noexcept;
  iterator end() noexcept;

  const_iterator begin() const noexcept;
  const_iterator end() const noexcept;

  reverse_iterator rbegin() noexcept;
  reverse_iterator rend() noexcept;

  const_reverse_iterator rbegin() const noexcept;
  const_reverse_iterator rend() const noexcept;

  tensor<int64_t>   long_() const;
  tensor<int32_t>   int32_() const;
  tensor<uint32_t>  uint32_() const;
  tensor<uint64_t>  unsigned_long_() const;
  tensor<float32_t> float32_() const;
  tensor<float64_t> double_() const;

  shape_type shape() const noexcept;
  shape_type strides() const noexcept;

  Device device() const noexcept { return this->__device_; }
  size_t n_dims() const noexcept;

  index_type size(const index_type __dim) const;
  index_type capacity() const noexcept;
  index_type count_nonzero(index_type __dim = -1) const;
  index_type lcm() const;
  index_type hash() const;

  reference at(shape_type __idx);
  reference operator[](const index_type __in);

  const_reference at(const shape_type __idx) const;
  const_reference operator[](const index_type __in) const;

  bool empty() const;

  tensor<bool> bool_() const;
  tensor<bool> logical_not() const;
  tensor<bool> logical_or(const value_type __val) const;
  tensor<bool> logical_or(const tensor& __other) const;
  tensor<bool> less_equal(const tensor& __other) const;
  tensor<bool> less_equal(const value_type __val) const;
  tensor<bool> greater_equal(const tensor& __other) const;
  tensor<bool> greater_equal(const value_type __val) const;
  tensor<bool> equal(const tensor& __other) const;
  tensor<bool> equal(const value_type __val) const;
  tensor<bool> not_equal(const tensor& __other) const;
  tensor<bool> not_equal(const value_type __val) const;
  tensor<bool> less(const tensor& __other) const;
  tensor<bool> less(const value_type __val) const;
  tensor<bool> greater(const tensor& __other) const;
  tensor<bool> greater(const value_type __val) const;

  bool          operator==(const tensor& __other) const;
  bool          operator!=(const tensor& __other) const;
  tensor        operator+(const tensor& __other) const;
  tensor        operator-(const tensor& __other) const;
  tensor        operator+(const value_type __val) const;
  tensor        operator-(const value_type __val) const;
  tensor&       operator-=(const tensor& __other) const;
  tensor&       operator+=(const tensor& __other) const;
  tensor&       operator*=(const tensor& __other) const;
  tensor&       operator/=(const tensor& __other) const;
  tensor&       operator+=(const_reference __val) const;
  tensor&       operator-=(const_reference __val) const;
  tensor&       operator/=(const_reference __val) const;
  tensor&       operator*=(const_reference __val) const;
  tensor&       operator=(const tensor& __other) const;
  tensor&       operator=(tensor&& __other) const noexcept;
  tensor<bool>& operator!() const;


  tensor
  slice(index_type __dim, std::optional<index_type> __start, std::optional<index_type> __end, index_type __step) const;
  tensor fmax(const tensor& __other) const;
  tensor fmax(const value_type __val) const;
  tensor fmod(const tensor& __other) const;
  tensor fmod(const value_type __val) const;
  tensor frac() const;
  tensor log() const;
  tensor log10() const;
  tensor log2() const;
  tensor exp() const;
  tensor sqrt() const;
  tensor sum(const index_type __axis) const;
  tensor row(const index_type __index) const;
  tensor col(const index_type __index) const;
  tensor ceil() const;
  tensor floor() const;
  tensor clone() const;
  tensor clamp(const_pointer __min_val = nullptr, const_pointer __max_val = nullptr) const;
  tensor cos() const;
  tensor cosh() const;
  tensor acos() const;
  tensor acosh() const;
  tensor tan() const;
  tensor tanh() const;
  tensor atan() const;
  tensor atanh() const;
  tensor sin() const;
  tensor sinc() const;
  tensor sinh() const;
  tensor asin() const;
  tensor asinh() const;
  tensor abs() const;
  tensor logical_xor(const tensor& __other) const;
  tensor logical_xor(const value_type __val) const;
  tensor logical_and(const tensor& __other) const;
  tensor logical_and(const value_type __val) const;
  tensor bitwise_not() const;
  tensor bitwise_and(const value_type __val) const;
  tensor bitwise_and(const tensor& __other) const;
  tensor bitwise_or(const value_type __val) const;
  tensor bitwise_or(const tensor& __other) const;
  tensor bitwise_xor(const value_type __val) const;
  tensor bitwise_xor(const tensor& __other) const;
  tensor bitwise_left_shift(const int __amount) const;
  tensor bitwise_right_shift(const int __amount) const;
  tensor matmul(const tensor& __other) const;
  tensor reshape(const shape_type __shape) const;
  tensor reshape_as(const tensor& __other) const;
  tensor cross_product(const tensor& __other) const;
  tensor absolute(const tensor& __tensor) const;
  tensor dot(const tensor& __other) const;
  tensor relu() const;
  tensor transpose() const;
  tensor fill(const value_type __val) const;
  tensor fill(const tensor& __other) const;
  tensor resize_as(const shape_type __sh) const;
  tensor all() const;
  tensor any() const;
  tensor det() const;
  tensor square() const;
  tensor sigmoid() const;
  tensor clipped_relu() const;
  tensor sort(index_type __dim, bool __descending = false) const;
  tensor remainder(const value_type __val) const;
  tensor remainder(const tensor& __other) const;
  tensor maximum(const tensor& __other) const;
  tensor maximum(const value_type& __val) const;
  tensor dist(const tensor& __other) const;
  tensor dist(const value_type __val) const;
  tensor squeeze(index_type __dim) const;
  tensor negative() const;
  tensor repeat(const data_t& __d) const;
  tensor permute(const index_type __dim) const;
  tensor log_softmax(const index_type __dim) const;
  tensor gcd(const tensor& __other) const;
  tensor gcd(const value_type __val) const;
  tensor pow(const tensor& __other) const;
  tensor pow(const value_type __val) const;
  tensor cumprod(index_type __dim = -1) const;
  tensor cat(const std::vector<tensor>& _others, index_type _dim) const;
  tensor argmax(index_type __dim) const;
  tensor unsqueeze(index_type __dim) const;
  tensor zeros(const shape_type& __sh);
  tensor ones(const shape_type& __sh);
  tensor randomize(const shape_type& __sh, bool __bounded = false);
  tensor expand_as(shape_type __sh, index_type __dim) const;

  double mean() const;
  double median(const index_type __dim) const;
  double mode(const index_type __dim) const;

  tensor& push_back(value_type __v) const;
  tensor& pop_back() const;
  tensor& sqrt_() const;
  tensor& exp_() const;
  tensor& log2_() const;
  tensor& log10_() const;
  tensor& log_() const;
  tensor& frac_() const;
  tensor& fmod_(const tensor& __other) const;
  tensor& fmod_(const value_type __val) const;
  tensor& cos_() const;
  tensor& cosh_() const;
  tensor& acos_() const;
  tensor& acosh_() const;
  tensor& tan_() const;
  tensor& tanh_() const;
  tensor& atan_() const;
  tensor& atanh_() const;
  tensor& sin_() const;
  tensor& sinh_() const;
  tensor& asin_() const;
  tensor& asinh_() const;
  tensor& ceil_() const;
  tensor& floor_() const;
  tensor& relu_() const;
  tensor& clamp_(const_pointer __min_val = nullptr, const_pointer __max_val = nullptr) const;
  tensor& logical_not_() const { this->bitwise_not_(); }
  tensor& logical_or_(const tensor& __other) const;
  tensor& logical_or_(const value_type __val) const;
  tensor& logical_xor_(const tensor& __other) const;
  tensor& logical_xor_(const value_type __val) const;
  tensor& logical_and_(const tensor& __other) const;
  tensor& logical_and_(const value_type __val) const;
  tensor& abs_() const;
  tensor& log_softmax_(const index_type __dim) const;
  tensor& permute_(const index_type __dim) const;
  tensor& repeat_(const data_t& __d) const;
  tensor& negative_() const;
  tensor& transpose_() const;
  tensor& unsqueeze_(index_type __dim) const;
  tensor& squeeze_(index_type __dim) const;
  tensor& dist_(const tensor& __other) const;
  tensor& dist_(const value_type __val) const;
  tensor& maximum_(const tensor& __other) const;
  tensor& maximum_(const value_type __val) const;
  tensor& remainder_(const value_type __val) const;
  tensor& remainder_(const tensor& __other) const;
  tensor& fill_(const value_type __val) const;
  tensor& fill_(const tensor& __other) const;
  tensor& less_(const tensor& __other) const;
  tensor& less_(const value_type __val) const;
  tensor& greater_(const tensor& __other) const;
  tensor& greater_(const value_type __val) const;
  tensor& sigmoid_() const;
  tensor& clipped_relu_(const value_type __clip_limit) const;
  tensor& square_() const;
  tensor& pow_(const tensor& __other);
  tensor& pow_(const value_type __val);
  tensor& sinc_() const;
  tensor& bitwise_left_shift_(const int __amount);
  tensor& bitwise_right_shift_(const int __amount);
  tensor& bitwise_and_(const value_type __val);
  tensor& bitwise_and_(const tensor& __other);
  tensor& bitwise_or_(const value_type __val);
  tensor& bitwise_or_(const tensor& __other);
  tensor& bitwise_xor_(const value_type __val);
  tensor& bitwise_xor_(const tensor& __other);
  tensor& bitwise_not_();
  tensor& view(std::initializer_list<index_type> __new_sh);
  tensor& fmax_(const tensor& __other);
  tensor& fmax_(const value_type __val);
  tensor& randomize_(const shape_type& __sh, bool __bounded = false);
  tensor& zeros_(shape_type __sh = {});
  tensor& ones_(shape_type __sh = {});
  tensor& print() const {
    this->printRecursive(0, 0, __shape_);
    std::cout << std::endl;
  }

  tensor<index_type> argmax_(index_type __dim) const;
  tensor<index_type> argsort(index_type __dim = -1, bool __ascending = true) const;


 private:
  static void __check_is_scalar_type(const std::string __msg) {
    assert(!__msg.empty());

    if (!std::is_scalar<value_type>::value)
    {
      throw std::runtime_error(__msg);
    }
  }

  static void __check_is_integral_type(const std::string __msg) {
    assert(!__msg.empty());

    if (!std::is_integral<value_type>::value)
    {
      throw std::runtime_error(__msg);
    }
  }

  template<typename __t>
  static void __check_is_same_type(const std::string __msg) {
    assert(!__msg.empty());

    if (!std::is_same<value_type, __t>::value)
    {
      throw std::runtime_error(__msg);
    }
  }

  static void __check_is_arithmetic_type(const std::string __msg) {
    assert(!__msg.empty());

    if (!std::is_arithmetic<value_type>::value)
    {
      throw std::runtime_error(__msg);
    }
  }

  [[nodiscard]] size_t computeStride(size_t dim, const shape_type& shape) const noexcept {
    size_t stride = 1;

    for (size_t i = dim; i < shape.size(); i++)
    {
      stride *= shape[i];
    }

    return stride;
  }

  void printRecursive(size_t index, size_t depth, const shape_type& shape) const {
    if (depth == shape.size() - 1)
    {
      std::cout << "[";

      for (size_t i = 0; i < shape[depth]; ++i)
      {
        if constexpr (std::is_floating_point<_Tp>::value)
        {
          std::cout << std::fixed << std::setprecision(4) << __data_[index + i];
        }
        else
        {
          std::cout << __data_[index + i];
        }

        if (i < shape[depth] - 1)
        {
          std::cout << ", ";
        }
      }
      std::cout << "]";
    }
    else
    {
      std::cout << "[\n";
      size_t stride = computeStride(depth + 1, shape);

      for (size_t i = 0; i < shape[depth]; ++i)
      {
        if (i > 0)
        {
          std::cout << "\n";
        }

        for (size_t j = 0; j < depth + 1; j++)
        {
          std::cout << " ";
        }

        printRecursive(index + i * stride, depth + 1, shape);

        if (i < shape[depth] - 1)
        {
          std::cout << ",";
        }
      }
      std::cout << "\n";

      for (size_t j = 0; j < depth; j++)
      {
        std::cout << " ";
      }

      std::cout << "]";
    }
  }

  void __compute_strides() {
    if (this->__shape_.empty())
    {
      std::cerr << "Shape must be initialized before computing strides" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    this->__strides_ = shape_type(this->__shape_.size(), 1);
    index_type __st = 1, __i = this->__shape_.size() - 1;

    for (; __i >= 0; __i--)
    {
      this->__strides_[__i] = __st;
      __st *= this->__shape_[__i];
    }
  }

  [[nodiscard]] index_type __compute_index(const std::vector<index_type>& __idx) const {
    if (__idx.size() != this->__shape_.size())
    {
      throw std::out_of_range("input indices does not match the tensor __shape_");
    }

    index_type __index = 0;
    index_type __i     = 0;

    for (; __i < this->__shape_.size(); __i++)
    {
      __index += __idx[__i] * this->__strides_[__i];
    }

    return __index;
  }

  [[nodiscard]] static uint64_t __computeSize(const shape_type& __dims) noexcept {
    uint64_t __ret = 1;

    for (const index_type& __d : __dims)
    {
      __ret *= __d;
    }

    return __ret;
  }

  uint64_t __compute_outer_size(const index_type __dim) const {
    // just a placeholder for now
    return 0;
  }

  [[nodiscard]] static float32_t __frac(const_reference __val) noexcept {
    return std::fmod(static_cast<float32_t>(__val), 1.0f);
  }

  // where the tensor is stored
  bool __is_cuda_device() const { return (this->__device_ == Device::CUDA); }

};  // tensor class


template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmax(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fmax_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmax(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fmax_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const value_type __val) {

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const index_type __simd_end   = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    neon_f32         __scalar_val = vdupq_n_f32(__val);

    for (index_type __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __a       = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __max_val = vmaxq_f32(__a, __scalar_val);

      vst1q_f32(&this->__data_[__i], __max_val);
    }

    for (index_type __i = __simd_end; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = std::fmax(this->__data_[__i], __val);
    }
  }
#else
  std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                 [&__val](const_reference __v) { return std::fmax(__v, __val); });
#endif

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmax_(const tensor<value_type>& __other) {
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __a       = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b       = vld1q_f32(reinterpret_cast<const _f32*>(&(__other[__i])));
      neon_f32 __max_val = vmaxq_f32(__a, __b);

      vst1q_f32(&this->__data_[__i], __max_val);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = std::fmax(this->__data_[__i], __other[__i]);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmod(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fmod_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmod(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fmod_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmod_(const value_type __val) const {
  assert(std::is_floating_point<value_type>::value && "fmod : template class must be a floating point type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() - _ARM64_REG_WIDTH);
    neon_f32         __b        = vdupq_n_f32(reinterpret_cast<_f32>(__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __a         = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __div       = vdivq_f32(__a, __b);
      neon_f32 __floor_div = vrndq_f32(__div);
      neon_f32 __mult      = vmulq_f32(__floor_div, __b);
      neon_f32 __mod       = vsubq_f32(__a, __mult);

      vst1q_f32(&this->__data_[__i], __mod);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] =
      static_cast<value_type>(std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__val)));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::fmod_(const tensor& __other) const {
  this->__check_is_scalar_type("Cannot divide non scalar values");

  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
  {
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");
  }

  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __a         = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __b         = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __div       = vdivq_f32(__a, __b);
      neon_f32 __floor_div = vrndq_f32(__div);
      neon_f32 __mult      = vmulq_f32(__floor_div, __b);
      neon_f32 __mod       = vsubq_f32(__a, __mult);

      vst1q_f32(&this->__data_[__i], __mod);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] =
      static_cast<value_type>(std::fmod(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::frac_() const {
  this->__check_is_scalar_type("Cannot get the fraction of a non-scalar type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_same_v<value_type, _f32>)
  {
    index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = this->__frac(__vals[0]);
      __vals[1] = this->__frac(__vals[1]);
      __vals[2] = this->__frac(__vals[2]);
      __vals[3] = this->__frac(__vals[3]);

      neon_f32 __atan_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __atan_vec);
    }
  }
  if constexpr (std::is_same_v<value_type, _f64>)
  {
    index_type __simd_end = this->__data_.size() - (this->__data_.size() % (_ARM64_REG_WIDTH / 2));

    for (; __i < __simd_end; __i += (_ARM64_REG_WIDTH / 2))
    {
      neon_f64 __data_vec = vld1q_f64(reinterpret_cast<const _f64*>(&this->__data_[__i]));
      _f64     __vals[_ARM64_REG_WIDTH];
      vst1q_f64(__vals, __data_vec);

      __vals[0] = static_cast<_f64>(this->__frac(__vals[0]));
      __vals[1] = static_cast<_f64>(this->__frac(__vals[1]));
      __vals[2] = static_cast<_f64>(this->__frac(__vals[2]));
      __vals[3] = static_cast<_f64>(this->__frac(__vals[3]));

      neon_f64 __atan_vec = vld1q_f64(__vals);
      vst1q_f64(&this->__data_[__i], __atan_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::frac() const {
  __self __ret = this->clone();
  __ret.frac_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::log_() const {
  this->__check_is_integral_type("Given data type must be an integral");
  index_type __i = 0;

#if defined(__ARM_NEON)
  
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same_v<value_type, _f32>)
  {
    for (; __i < _simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      _f32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_f32>(std::log(static_cast<_f32>(__vals[0])));
      __vals[1] = static_cast<_f32>(std::log(static_cast<_f32>(__vals[1])));
      __vals[2] = static_cast<_f32>(std::log(static_cast<_f32>(__vals[2])));
      __vals[3] = static_cast<_f32>(std::log(static_cast<_f32>(__vals[3])));

      neon_f32 __log_vec = vld1q_f32(__vals);
      vst1q_f32(&this->__data_[__i], __log_vec);
    }
  }
  else if constexpr (std::is_same_v<value_type, _u32>)
  {
    for (; __i < _simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      _u32     __vals[_ARM64_REG_WIDTH];
      vst1q_f32(__vals, __data_vec);

      __vals[0] = static_cast<_u32>(std::log(static_cast<_u32>(__vals[0])));
      __vals[1] = static_cast<_u32>(std::log(static_cast<_u32>(__vals[1])));
      __vals[2] = static_cast<_u32>(std::log(static_cast<_u32>(__vals[2])));
      __vals[3] = static_cast<_u32>(std::log(static_cast<_u32>(__vals[3])));

      neon_u32 __log_vec = vld1q_u32(__vals);
      vst1q_u32(&this->__data_[__i], __log_vec);
    }
  }
  else if constexpr (std::is_same_v<value_type, _s32>)
  {
    for (; __i < _simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      _s32     __vals[_ARM64_REG_WIDTH];
      vst1q_s32(__vals, __data_vec);

      __vals[0] = static_cast<_s32>(std::log(static_cast<_s32>(__vals[0])));
      __vals[1] = static_cast<_s32>(std::log(static_cast<_s32>(__vals[1])));
      __vals[2] = static_cast<_s32>(std::log(static_cast<_s32>(__vals[2])));
      __vals[3] = static_cast<_s32>(std::log(static_cast<_s32>(__vals[3])));

      neon_s32 __log_vec = vld1q_s32(__vals);
      vst1q_s32(&this->__data_[__i], __log_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::log(this->__data_[__i]));

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log() const {
  __self __ret = this->clone();
  __ret.log_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::log10_() const {
  this->__check_is_integral_type("Given data type must be an integral");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::log10(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::log10(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::log10(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::log10(static_cast<_f32>(__vals[3])));

    neon_type __log_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __log_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_type>(std::log10(this->__data_[__i]));

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log10() const {
  __self __ret = this->clone();
  __ret.log10_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::log2_() const {
  this->__check_is_integral_type("Given data type must be an integral");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::log2(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::log2(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::log2(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::log2(static_cast<_f32>(__vals[3])));

    neon_type __log2_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __log2_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::log2(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log2() const {
  __self __ret = this->clone();
  __ret.log2_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::exp_() const {
  this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::exp(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::exp(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::exp(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::exp(static_cast<_f32>(__vals[3])));

    neon_type __exp_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __exp_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::exp(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::exp() const {
  __self __ret = this->clone();
  __ret.exp_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sqrt_() const {
  this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::sqrt(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::sqrt(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::sqrt(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::sqrt(static_cast<_f32>(__vals[3])));

    neon_type __sqrt_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __sqrt_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::sqrt(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sqrt() const {
  __self __ret = this->clone();
  __ret.sqrt_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::square() const {
  __self __ret = this->clone();
  __ret.square_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::cos_() const {
  this->__check_is_scalar_type("Cannot perform a cosine on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::cos(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::cos(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::cos(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::cos(static_cast<_f32>(__vals[3])));

    neon_type __cos_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __cos_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::cos(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::acos_() const {
  this->__check_is_scalar_type("Cannot perform a acos on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::acos(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::acos(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::acos(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::acos(static_cast<_f32>(__vals[3])));

    neon_type __cos_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __cos_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::acos(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::acos() const {
  __self __ret = this->clone();
  __ret.acos_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cos() const {
  __self __ret = this->clone();
  __ret.cos_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sin_() const {
  this->__check_is_integral_type("Cannot perform a sin on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::sin(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::sin(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::sin(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::sin(static_cast<_f32>(__vals[3])));

    neon_type __sin_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __sin_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::sin(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sin() const {
  __self __ret = this->clone();
  __ret.sin_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::tan_() const {
  this->__check_is_integral_type("template class must be integral type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::tan(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::tan(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::tan(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::tan(static_cast<_f32>(__vals[3])));

    neon_type __tan_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __tan_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::tan(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::tanh_() const {
  this->__check_is_integral_type("template class must be integral type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::tanh(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::tanh(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::tanh(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::tanh(static_cast<_f32>(__vals[3])));

    neon_type __tanh_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __tanh_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::tanh(static_cast<_f32>(this->__data_[__i])));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::asin() const {
  __self __ret = this->clone();
  __ret.asin_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cosh() const {
  __self __ret = this->clone();
  __ret.cosh_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::atan() const {
  __self __ret = this->clone();
  __ret.atan_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sinc_() const {
  this->__check_is_arithmetic_type("sinc_: template type must be an arithmetic type");

  index_type __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_same<value_type, _f32>::value)
  {
    using neon_type             = neon_f32;
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_type __v      = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_type __pi_v   = vmulq_f32(__v, vdupq_n_f32(M_PI));                        // pi * x
      neon_type __sinc_v = vbslq_f32(vcgeq_f32(vabsq_f32(__v), vdupq_n_f32(1e-6f)),  // Check |x| > epsilon
                                     vdivq_f32(vsinq_f32(__pi_v), __pi_v),           // sinc(x) = sin(pi * x) / (pi * x)
                                     vdupq_n_f32(1.0f));                             // sinc(0) = 1

      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __sinc_v);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    value_type x = this->__data_[__i];
    this->__data_[__i] =
      (std::abs(x) < 1e-6) ? static_cast<value_type>(1.0) : static_cast<value_type>(std::sin(M_PI * x) / (M_PI * x));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::atan_() const {
  this->__check_is_integral_type("template class must be integral type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type       = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::atan(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::atan(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::atan(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::atan(static_cast<_f32>(__vals[3])));

    neon_type __atan_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __atan_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::atanh_() const {
  this->__check_is_integral_type("template class must be integral type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type       = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::atanh(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::atanh(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::atanh(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::atanh(static_cast<_f32>(__vals[3])));

    neon_type __atan_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __atan_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::atanh() const {
  __self __ret = this->clone();
  __ret.atanh_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sinc() const {
  __self __ret = this->clone();
  __ret.sinc_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::sinh_() const {
  this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type             = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::sinh(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::sinh(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::sinh(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::sinh(static_cast<_f32>(__vals[3])));

    neon_type __sinh_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __sinh_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::sinh(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sinh() const {
  __self __ret = this->clone();
  __ret.sinh_();
  return __ret;
}


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::asinh_() const {
  this->__check_is_scalar_type("Cannot perform asinh on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  using neon_type       = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::asinh(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::asinh(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::asinh(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::asinh(static_cast<_f32>(__vals[3])));

    neon_type __asinh_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __asinh_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::asinh(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::asinh() const {
  __self __ret = this->clone();
  __ret.asinh_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::asin_() const {
  this->__check_is_scalar_type("Cannot perform asin on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type             = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::asin(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::asin(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::asin(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::asin(static_cast<_f32>(__vals[3])));

    neon_type __asin_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __asin_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::asin(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::cosh_() const {
  this->__check_is_scalar_type("Cannot perform a cosh on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type             = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::cosh(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::cosh(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::cosh(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::cosh(static_cast<_f32>(__vals[3])));

    neon_type __cosh_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __cosh_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::cosh(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::acosh_() const {
  this->__check_is_scalar_type("Cannot perform a acosh on non-scalar data type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type             = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::acosh(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::acosh(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::acosh(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::acosh(static_cast<_f32>(__vals[3])));

    neon_type __acosh_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __acosh_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::acosh(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::acosh() const {
  __self __ret = this->clone();
  __ret.acosh_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::pow_(const value_type __val) {
  this->__check_is_integral_type("cannot get the power of a non-integral value");
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type             = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::pow(static_cast<_f32>(__vals[0]), static_cast<_f32>(__val)));
    __vals[1] = static_cast<value_type>(std::pow(static_cast<_f32>(__vals[1]), static_cast<_f32>(__val)));
    __vals[2] = static_cast<value_type>(std::pow(static_cast<_f32>(__vals[2]), static_cast<_f32>(__val)));
    __vals[3] = static_cast<value_type>(std::pow(static_cast<_f32>(__vals[3]), static_cast<_f32>(__val)));

    neon_type __pow_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __pow_vec);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::pow(this->__data_[__i], __val));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::pow(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.pow_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::pow(const value_type __val) const {
  __self __ret = this->clone();
  __ret.pow_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::pow_(const tensor& __other) {
  this->__check_is_integral_type("cannot get the power of a non integral value");
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same_v<value_type, _f32>, neon_f32, neon_s32>;

  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  neon_type __base_vec   = vld1q(reinterpret_cast<const value_type*>(&this->__data_[__i]));
  neon_type __exp_vec    = vld1q(reinterpret_cast<const value_type*>(&__other[__i]));
  neon_type __result_vec = {static_cast<value_type>(std::pow(static_cast<_f32>(vget_lane(__base_vec, 0)),
                                                             static_cast<_f32>(vget_lane(__exp_vec, 0)))),
                            static_cast<value_type>(std::pow(static_cast<_f32>(vget_lane(__base_vec, 1)),
                                                             static_cast<_f32>(vget_lane(__exp_vec, 1)))),
                            static_cast<value_type>(std::pow(static_cast<_f32>(vget_lane(__base_vec, 2)),
                                                             static_cast<_f32>(vget_lane(__exp_vec, 2)))),
                            static_cast<value_type>(std::pow(static_cast<_f32>(vget_lane(__base_vec, 3)),
                                                             static_cast<_f32>(vget_lane(__exp_vec, 3))))};

  vst1q(&this->__data_[__i], __result_vec);
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] =
      static_cast<value_type>(std::pow(static_cast<_f32>(this->__data_[__i]), static_cast<_f32>(__other[__i])));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::abs() const {
  __self __ret = this->clone();
  __ret.abs_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::abs_() const {
  this->__check_is_integral_type("template class must be integral type");
  index_type __i = 0;

  if (std::is_unsigned<value_type>::value)
  {
    return *this;
  }

#if defined(__ARM_NEON)
  index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  using neon_type       = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type  __data_vec = vld1q(reinterpret_cast<neon_type*>(&this->__data_[__i]));
    value_type __vals[_ARM64_REG_WIDTH];
    vst1q(__vals, __data_vec);

    __vals[0] = static_cast<value_type>(std::abs(static_cast<_f32>(__vals[0])));
    __vals[1] = static_cast<value_type>(std::abs(static_cast<_f32>(__vals[1])));
    __vals[2] = static_cast<value_type>(std::abs(static_cast<_f32>(__vals[2])));
    __vals[3] = static_cast<value_type>(std::abs(static_cast<_f32>(__vals[3])));

    neon_type __abs_vec = vld1q(__vals);
    vst1q(&this->__data_[__i], __abs_vec);
  }

#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::abs(this->__data_[__i]));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log_softmax(const index_type __dim) const {
  __self __ret = this->clone();
  __ret.log_softmax_(__dim);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::dist(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.dist_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::dist(const value_type __val) const {
  __self __ret = this->clone();
  __ret.dist_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::dist_(const tensor& __other) const {
  this->__check_is_arithmetic_type("dist: template type must be an arithmetic type");

  assert(this->__shape_ == __other.shape());

  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type __a    = vld1q(reinterpret_cast<const neon_type*>(&this->__data_[__i]));
    neon_type __b    = vld1q(reinterpret_cast<const neon_type*>(&__other.__data_[__i]));
    neon_type __diff = vabdq(__a, __b);
    vst1q(reinterpret_cast<neon_type*>(&this->__data_[__i]), __diff);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] =
      static_cast<value_type>(std::abs(static_cast<_f64>(this->__data_[__i] - __other.__data_[__i])));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::dist_(const value_type __val) const {
  this->__check_is_arithmetic_type("dist: template type must be an arithmetic type");

  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type __a    = vld1q(reinterpret_cast<const neon_type*>(&this->__data_[__i]));
    neon_type __b    = vdupq(reinterpret_cast<const neon_type*>(__val));
    neon_type __diff = vabdq(__a, __b);
    vst1q(reinterpret_cast<neon_type*>(&this->__data_[__i]), __diff);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::abs(static_cast<_f64>(this->__data_[__i] - __val)));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::remainder(const value_type __val) const {
  __self __ret = this->clone();
  __ret.remainder_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::remainder(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.remainder_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::remainder_(const value_type __val) const {
  this->__check_is_arithmetic_type("remainder_: template type must be an arithmetic type");
  assert(__val != 0 && "Remainder by zero is undefined");

  for (index_type __i = 0; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] %= __val;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::remainder_(const tensor& __other) const {
  this->__check_is_arithmetic_type("remainder_: template type must be an arithmetic type");

  assert(this->__shape_ == __other.shape());

  for (index_type __i = 0; __i < this->__data_.size(); __i++)
  {
    assert(__other.__data_[__i] != 0 && "Remainder by zero is undefined");
    this->__data_[__i] %= __other.__data_[__i];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::maximum(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.maximum_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::maximum(const value_type& __val) const {
  __self __ret = this->clone();
  __ret.maximum_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::maximum_(const tensor& __other) const {
  this->__check_is_arithmetic_type("maximum_: template type must be an arithmetic type");

  assert(this->__shape_ == __other.shape());

  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type __a   = vld1q(reinterpret_cast<const neon_type*>(&this->__data_[__i]));
    neon_type __b   = vld1q(reinterpret_cast<const neon_type*>(&__other.__data_[__i]));
    neon_type __max = vmaxq(__a, __b);
    vst1q(reinterpret_cast<neon_type*>(&this->__data_[__i]), __max);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = std::max(this->__data_[__i], __other.__data_[__i]);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::maximum_(const value_type __val) const {
  this->__check_is_arithmetic_type("maximum_: template type must be an arithmetic type");

  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  neon_type        __val_vec  = vdupq_n(__val);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type __a   = vld1q(reinterpret_cast<const neon_type*>(&this->__data_[__i]));
    neon_type __max = vmaxq(__a, __val_vec);
    vst1q(reinterpret_cast<neon_type*>(&this->__data_[__i]), __max);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = std::max(this->__data_[__i], __val);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::square_() const {
  return this->pow_(static_cast<value_type>(2.0f));
}


template<class _Tp>
double tensor<_Tp>::mean() const {
  this->__check_is_integral_type("Input must be of integral type to calculate the mean.");

  double __m = 0.0;

  if (this->empty())
  {
    return __m;
  }

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  neon_s32         __sum_vec  = vdupq_n_s32(0);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
    __sum_vec           = vaddq_s32(__sum_vec, __data_vec);
  }

  _s32 __partial_sum[4];
  vst1q_s32(__partial_sum, __sum_vec);
  __m += __partial_sum[0] + __partial_sum[1] + __partial_sum[2] + __partial_sum[3];
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __m += this->__data_[__i];
  }

  return static_cast<double>(__m) / static_cast<double>(this->__data_.size());
}


// used as a helper function
int64_t __lcm(const int64_t __a, const int64_t __b) { return (__a * __b) / std::gcd(__a, __b); }


template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::lcm() const {
  this->__check_is_scalar_type("Given template type must be an int");

  index_type __ret = static_cast<index_type>(this->__data_[0]);
  index_type __i   = 1;

  for (; __i < this->__data_.size(); __i++)
  {
    __ret = __lcm(static_cast<index_type>(this->__data_[__i]), __ret);
  }

  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::tanh() const {
  __self __ret = this->clone();
  __ret.tanh_();
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::tan() const {
  __self __ret = this->clone();
  __ret.tan_();
  return __ret;
}


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_right_shift_(const int __amount) {
  this->__check_is_integral_type("Cannot perform a bitwise right shift on non-integral values");

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    const neon_s32 __shift_amount_vec = vdupq_n_s32(-__amount);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec    = vld1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]));
      neon_s32 __shifted_vec = vshlq_s32(__data_vec, __shift_amount_vec);

      vst1q_s32(&this->__data_[__i], __shifted_vec);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    const neon_s32 __shift_amount_vec = vdupq_n_s32(-__amount);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec    = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __shifted_vec = vshlq_u32(__data_vec, __shift_amount_vec);

      vst1q_u32(&this->__data_[__i], __shifted_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] >>= __amount;
  }

  return *this;
}


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_left_shift_(const int __amount) {
  this->__check_is_integral_type("Cannot perform a bitwise left shift on non-integral values");

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    const neon_s32 __shift_amount_vec = vdupq_n_s32(__amount);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec    = vld1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]));
      neon_s32 __shifted_vec = vshlq_s32(__data_vec, __shift_amount_vec);

      vst1q_s32(&this->__data_[__i], __shifted_vec);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    const neon_s32 __shift_amount_vec = vdupq_n_s32(__amount);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec    = vld1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]));
      neon_u32 __shifted_vec = vshlq_u32(__data_vec, __shift_amount_vec);

      vst1q_u32(&this->__data_[__i], __shifted_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] <<= __amount;
  }

  return *this;
}


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_or_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");
  }

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec   = vld1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]));
      neon_s32 __result_vec = vorrq_s32(__data_vec, __val_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __result_vec);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    neon_u32 __val_vec = vdupq_n_u32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec   = vld1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]));
      neon_u32 __result_vec = vorrq_u32(__data_vec, __val_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __result_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] |= __val;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");
  }

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec   = vld1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]));
      neon_s32 __result_vec = veorq_s32(__data_vec, __val_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __result_vec);
    }
  }
  else
  {
    neon_u32 __val_vec = vdupq_n_u32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec   = vld1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]));
      neon_u32 __result_vec = veorq_u32(__data_vec, __val_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __result_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] ^= __val;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_xor(const value_type __val) const {
  __self __ret = this->clone();
  __ret.bitwise_xor_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_not() const {
  __self __ret = this->clone();
  __ret.bitwise_not_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_not_() {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise not on non integral or boolean value");
  }

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __result_vec = vmvnq_s32(__data_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __result_vec);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __result_vec = vmvnq_u32(__data_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __result_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = ~this->__data_[__i];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_and_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");
  }

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(reinterpret_cast<_s32>(&__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __result_vec = vandq_s32(__data_vec, __val_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __result_vec);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    neon_u32 __val_vec = vdupq_n_u32(reinterpret_cast<_u32>(&__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __result_vec = vandq_u32(__data_vec, __val_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __result_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] &= __val;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_and(const value_type __val) const {
  __self __ret = this->clone();
  __ret.bitwise_and_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_and(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.bitwise_and_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_left_shift(const int __amount) const {
  __self __ret = this->clone();
  __ret.bitwise_left_shift_(__amount);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_xor(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.bitwise_xor_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_right_shift(const int __amount) const {
  __self __ret = this->clone();
  __ret.bitwise_right_shift_(__amount);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_and_(const tensor& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");
  }

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __xor_vec   = vandq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __xor_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __xor_vec   = vandq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __xor_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] &= __other[__i];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_or(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.bitwise_or_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_or(const value_type __val) const {
  __self __ret = this->clone();
  __ret.bitwise_or_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_or_(const tensor& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");
  }

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __xor_vec   = vornq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __xor_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __xor_vec   = vornq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __xor_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] |= __other[__i];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::bitwise_xor_(const tensor& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");
  }

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __xor_vec   = veorq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __xor_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __xor_vec   = veorq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __xor_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] ^= __other[__i];
  }

  return *this;
}


template<class _Tp>
tensor<bool> tensor<_Tp>::not_equal(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  assert(this->__shape_ == __other.shape() && "equal : tensor shapes");
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] != __other[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::not_equal(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] != __val);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] < __other[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] < __val);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] > __other[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] > __val);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  assert(this->__shape_ == __other.shape() && "equal : tensor shapes");
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec1  = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __data_vec2  = vld1q_f32(reinterpret_cast<const _f32*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vceqq_f32(__data_vec1, __data_vec2);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec1  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __data_vec2  = vld1q_s32(reinterpret_cast<const _s32*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vceqq_s32(__data_vec1, __data_vec2);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec1  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __data_vec2  = vld1q_u32(reinterpret_cast<const _u32*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vceqq_u32(__data_vec1, __data_vec2);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] == __other[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    neon_f32         __val_vec  = vdupq_n_f32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_u32 __cmp_result = vceqq_f32(__data_vec, __val_vec);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    neon_s32         __val_vec  = vdupq_n_s32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_u32 __cmp_result = vceqq_s32(__data_vec, __val_vec);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    neon_u32         __val_vec  = vdupq_n_u32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __cmp_result = vceqq_u32(__data_vec, __val_vec);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] == __val);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

#if defined(__ARM_NEON)
  using neon_type    = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;
  size_t vector_size = this->__data_.size() / _ARM64_REG_WIDTH * _ARM64_REG_WIDTH;

  for (; __i < vector_size; __i += _ARM64_REG_WIDTH)
  {
    neon_type vec_a    = vld1q(this->__data_.data() + __i);
    neon_type vec_b    = vld1q(__other.__data_.data() + __i);
    neon_u32  leq_mask = std::is_same_v<value_type, _f32> ? vcleq_f32(vec_a, vec_b) : vcleq_s32(vec_a, vec_b);
    vst1q_u32(reinterpret_cast<_u32*>(&__ret[__i]), leq_mask);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] <= __other[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

#if defined(__ARM_NEON)
  using neon_type    = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;
  size_t vector_size = this->__data_.size() / _ARM64_REG_WIDTH * _ARM64_REG_WIDTH;

  for (; __i < vector_size; __i += _ARM64_REG_WIDTH)
  {
    neon_type vec_a    = vld1q(this->__data_.data() + __i);
    neon_type vec_b    = std::is_same_v<value_type, _f32> ? vdupq_n_f32(__val) : vdupq_n_s32(__val);
    neon_u32  leq_mask = std::is_same_v<value_type, _f32> ? vcleq_f32(vec_a, vec_b) : vcleq_s32(vec_a, vec_b);

    vst1q_u32(reinterpret_cast<_u32*>(&__ret[__i]), leq_mask);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] <= __val);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % (_ARM64_REG_WIDTH / 2));

    for (; __i < __simd_end; __i += (_ARM64_REG_WIDTH / 2))
    {
      neon_f32 __data_vec1  = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __data_vec2  = vld1q_f32(reinterpret_cast<const _f32*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vcgeq_f32(__data_vec1, __data_vec2);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 1) & 1;
      __ret[__i + 2] = (__mask >> 2) & 1;
      __ret[__i + 3] = (__mask >> 3) & 1;
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec1  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __data_vec2  = vld1q_s32(reinterpret_cast<const _s32*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vcgeq_s32(__data_vec1, __data_vec2);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 1) & 1;
      __ret[__i + 2] = (__mask >> 2) & 1;
      __ret[__i + 3] = (__mask >> 3) & 1;
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec1  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __data_vec2  = vld1q_u32(reinterpret_cast<const _u32*>(&__other.__data_[__i]));
      neon_u32 __cmp_result = vcgeq_u32(__data_vec1, __data_vec2);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 1) & 1;
      __ret[__i + 2] = (__mask >> 2) & 1;
      __ret[__i + 3] = (__mask >> 3) & 1;
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] >= __other[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  std::vector<bool> __ret(this->__data_.size());
  index_type        __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __val_vec = vdupq_n_f32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_u32 __cmp_result = vcgeq_f32(__data_vec, __val_vec);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_u32 __cmp_result = vcgeq_s32(__data_vec, __val_vec);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __val_vec = vdupq_n_u32(__val);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __cmp_result = vcgeq_u32(__data_vec, __val_vec);
      _u32     __mask       = vaddvq_u32(__cmp_result);

      __ret[__i]     = __mask & 1;
      __ret[__i + 1] = (__mask >> 8) & 1;
      __ret[__i + 2] = (__mask >> 16) & 1;
      __ret[__i + 3] = (__mask >> 24) & 1;
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] >= __val);
  }

  return tensor<bool>(__ret, this->__shape_);
}


template<class _Tp>
size_t tensor<_Tp>::n_dims() const noexcept {
  return this->__shape_.size();
}

template<class _Tp>
typename tensor<_Tp>::shape_type tensor<_Tp>::shape() const noexcept {
  return this->__shape_;
}

template<class _Tp>
typename tensor<_Tp>::data_t tensor<_Tp>::storage() const noexcept {
  return this->__data_;
}

template<class _Tp>
typename tensor<_Tp>::shape_type tensor<_Tp>::strides() const noexcept {
  return this->__strides_;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::reshape_as(const tensor& __other) const {
  return this->reshape(__other.__shape_);
}

template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::capacity() const noexcept {
  return this->__data_.capacity();
}

template<class _Tp>
bool tensor<_Tp>::operator!=(const tensor& __other) const {
  return !(*this == __other);
}

template<class _Tp>
bool tensor<_Tp>::empty() const {
  return this->__data_.empty();
}

template<class _Tp>
typename tensor<_Tp>::iterator tensor<_Tp>::begin() noexcept {
  return this->__data_.begin();
}

template<class _Tp>
typename tensor<_Tp>::const_iterator tensor<_Tp>::begin() const noexcept {
  return this->__data_.begin();
}

template<class _Tp>
typename tensor<_Tp>::iterator tensor<_Tp>::end() noexcept {
  return this->__data_.end();
}

template<class _Tp>
typename tensor<_Tp>::const_iterator tensor<_Tp>::end() const noexcept {
  return this->__data_.end();
}

template<class _Tp>
typename tensor<_Tp>::reverse_iterator tensor<_Tp>::rbegin() noexcept {
  return this->__data_.rbegin();
}

template<class _Tp>
typename tensor<_Tp>::const_reverse_iterator tensor<_Tp>::rbegin() const noexcept {
  return this->__data_.rbegin();
}

template<class _Tp>
typename tensor<_Tp>::reverse_iterator tensor<_Tp>::rend() noexcept {
  return this->__data_.rend();
}

template<class _Tp>
typename tensor<_Tp>::const_reverse_iterator tensor<_Tp>::rend() const noexcept {
  return this->__data_.rend();
}

template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::size(const index_type __dim) const {
  if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
  {
    throw std::invalid_argument("dimension input is out of range");
  }

  if (this->__data_.empty())
  {
    return 0;
  }

  if (__dim == 0)
  {
    return this->__data_.size();
  }
  return this->__shape_[__dim];
}

template<class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::at(tensor<_Tp>::shape_type __idx) {
  if (__idx.empty())
  {
    throw std::invalid_argument("Passing an empty vector as indices for a tensor");
  }

  index_type __i = this->__compute_index(__idx);
  if (__i < 0 || __i >= this->__data_.size())
  {
    throw std::invalid_argument("input indices are out of bounds");
  }

  return this->__data_[__i];
}

template<class _Tp>
typename tensor<_Tp>::const_reference tensor<_Tp>::at(const tensor<_Tp>::shape_type __idx) const {
  if (__idx.empty())
  {
    throw std::invalid_argument("Passing an empty vector as indices for a tensor");
  }

  index_type __i = this->__compute_index(__idx);

  if (__i < 0 || __i >= this->__data_.size())
  {
    throw std::invalid_argument("input indices are out of bounds");
  }

  return this->__data_[__i];
}

template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::count_nonzero(index_type __dim) const {
  this->__check_is_scalar_type("Cannot compare a non-scalar value to zero");
  index_type __c = 0;

  if (__dim == -1)
  {

#pragma omp parallel
    {
      index_type __local_count = 0;

#ifdef __AVX__
      if constexpr (std::is_same_v<value_type, _f32>)
      {
        index_type __size = this->__data_.size();
        index_type __i    = 0;

        for (; __i + _AVX_REG_WIDTH <= __size; __i += _AVX_REG_WIDTH)
        {
          __m256 __vec          = _mm256_loadu_ps(&this->__data_[__i]);
          __m256 __nonzero_mask = _mm256_cmp_ps(__vec, _mm256_setzero_ps(), _CMP_NEQ_OQ);
          __local_count += _mm256_movemask_ps(__nonzero_mask);
        }
      }

#endif
      index_type __i = 0;

#if defined(__ARM_NEON)
      if constexpr (std::is_floating_point<value_type>::value)
      {
        index_type __size = this->__data_.size();

        for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
        {
          neon_f32 __vec          = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
          neon_u32 __nonzero_mask = vcgtq_f32(__vec, vdupq_n_f32(0.0f));
          __local_count += vaddvq_u32(__nonzero_mask);
        }
      }
      else if constexpr (std::is_unsigned<value_type>::value)
      {
        index_type __size = this->__data_.size();

        for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
        {
          neon_u32 __vec          = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
          neon_u32 __nonzero_mask = vcgtq_u32(__vec, vdupq_n_u32(0));
          __local_count += vaddvq_u32(__nonzero_mask);
        }
      }
      else if constexpr (std::is_signed<value_type>::value)
      {
        index_type __size = this->__data_.size();

        for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
        {
          neon_s32 __vec          = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
          neon_s32 __nonzero_mask = vcgtq_s32(__vec, vdupq_n_s32(0));
          __local_count += vaddvq_s32(__nonzero_mask);
        }
      }

#endif
      for (index_type __j = __i; __j < this->__data_.size(); __j++)
      {
        if (this->__data_[__j] != 0)
        {
          __local_count++;
        }
      }

#pragma omp atomic
      __c += __local_count;
    }
  }
  else
  {
    if (__dim < 0 || __dim >= static_cast<index_type>(__shape_.size()))
    {
      throw std::invalid_argument("Invalid dimension provided.");
    }

    throw std::runtime_error("Dimension-specific non-zero counting is not implemented yet.");
  }

  return __c;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::zeros(const shape_type& __sh) {
  __self __ret = this->clone();
  __ret.zeros_(__sh);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::zeros_(shape_type __sh) {
  if (__sh.empty())
  {
    __sh = this->__shape_;
  }
  else
  {
    this->__shape_ = __sh;
  }

  size_t __s = this->__computeSize(this->__shape_);

  this->__data_.resize(__s);
  this->__compute_strides();

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = __s - (__s % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __zero_vec = vdupq_n_f32(0.0f);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      vst1q_f32(&this->__data_[__i], __zero_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __zero_vec = vdupq_n_s32(0);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      vst1q_s32(&this->__data_[__i], __zero_vec);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __zero_vec = vdupq_n_u32(0);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      vst1q_u32(&this->__data_[__i], __zero_vec);
    }
  }
#endif

  for (; __i < __s; __i++)
  {
    this->__data_[__i] = value_type(0.0);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::ones_(shape_type __sh) {
  if (__sh.empty())
  {
    __sh = this->__shape_;
  }
  else
  {
    this->__shape_ = __sh;
  }

  size_t __s = this->__computeSize(this->__shape_);

  this->__data_.resize(__s);
  this->__compute_strides();
  this->__check_is_scalar_type("template type must be a scalar : tensor.ones()");

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = __s - (__s % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __one_vec = vdupq_n_f32(1.0f);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      vst1q_f32(reinterpret_cast<_f32*>(&this->__data_[__i]), __one_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __one_vec = vdupq_n_s32(1);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __one_vec);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __one_vec = vdupq_n_u32(1);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __one_vec);
    }
  }
#endif

  for (; __i < __s; __i++)
  {
    this->__data_[__i] = value_type(1.0);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::ones(const shape_type& __sh) {
  __self __ret = this->clone();
  __ret.ones_(__sh);
  return __ret;
}

template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::hash() const {
  index_type            __hash_val = 0;
  std::hash<value_type> __hasher;

  index_type __i = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    __hash_val ^= __hasher(this->__data_[__i]) + 0x9e3779b9 + (__hash_val << 6) + (__hash_val >> 2);
  }

  return __hash_val;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::row(const index_type __index) const {
  if (this->__shape_.size() != 2)
  {
    throw std::runtime_error("Cannot get a row from a non two dimensional tensor");
  }

  if (this->__shape_[0] <= __index || __index < 0)
  {
    throw std::invalid_argument("Index input is out of range");
  }

  data_t     __r;
  index_type __start = this->__shape_[1] * __index;
  index_type __end   = this->__shape_[1] * __index + this->__shape_[1];
  index_type __i     = __start;

  for (; __i < __end; __i++)
  {
    __r.push_back(this->__data_[__i]);
  }

  return __self(__r, {this->__shape_[1]});
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::col(const index_type __index) const {
  if (this->__shape_.size() != 2)
  {
    throw std::runtime_error("Cannot get a column from a non two dimensional tensor");
  }

  if (this->__shape_[1] <= __index || __index < 0)
  {
    throw std::invalid_argument("Index input out of range");
  }

  data_t     __c;
  index_type __i = 0;

  for (; __i < this->__shape_[0]; __i++)
  {
    __c.push_back(this->__data_[this->__compute_index({__i, __index})]);
  }

  return __self(__c, {this->__shape_[0]});
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::view(std::initializer_list<index_type> __sh) {
  index_type __s = this->__computeSize(__sh);

  if (__s != this->__data_.size())
  {
    throw std::invalid_argument("Total elements do not match for new shape");
  }

  this->__shape_ = __sh;
  this->__compute_strides();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::randomize(const shape_type& __sh, bool __bounded) {
  __self __ret = this->clone();
  __ret.randomize_(__sh, __bounded);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::randomize_(const shape_type& __sh, bool __bounded) {
  if (__bounded)
  {
    assert(std::is_floating_point<value_type>::value && "Cannot bound non floating point data type");
  }

  if (__sh.empty() && this->__shape_.empty())
  {
    throw std::invalid_argument("randomize_ : Shape must be initialized");
  }

  if (this->__shape_.empty() || this->__shape_ != __sh)
  {
    this->__shape_ = __sh;
  }

  index_type __s = this->__computeSize(this->__shape_);
  this->__data_.resize(__s);
  this->__compute_strides();

  std::random_device                   __rd;
  std::mt19937                         __gen(__rd());
  std::uniform_real_distribution<_f32> __unbounded_dist(1.0f, static_cast<_f32>(RAND_MAX));
  std::uniform_real_distribution<_f32> __bounded_dist(0.0f, 1.0f);
  index_type                           __i = 0;

#if defined(__AVX__)
  const __m256 __scale = _mm256_set1_ps(__bounded ? static_cast<_f32>(RAND_MAX) : 1.0f);
  for (; __i + _AVX_REG_WIDTH <= static_cast<index_type>(__s); __i += _AVX_REG_WIDTH)
  {
    __m256 __random_values =
      _mm256_setr_ps(__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen),
                     __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen));

    if (!__bounded)
    {
      __random_values = _mm256_div_ps(__random_values, __scale);
    }

    _mm256_storeu_ps(&this->__data_[__i], __random_values);
  }

#elif defined(__SSE__)
  const __m128 __scale = _mm_set1_ps(__bounded ? static_cast<_f32>(RAND_MAX) : 1.0f);
  for (; __i + 4 <= static_cast<index_type>(__s); __i += 4)
  {
    __m128 __random_values =
      _mm_setr_ps(__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen));

    if (!__bounded)
    {
      __random_values = _mm_div_ps(__random_values, __scale);
    }

    _mm_storeu_ps(&this->__data_[__i], __random_values);
  }

#elif defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_type>::value)
  {
    const neon_f32 __scale = vdupq_n_f32(__bounded ? static_cast<_f32>(RAND_MAX) : 1.0f);
    for (; __i + _ARM64_REG_WIDTH <= static_cast<index_type>(__s); __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __random_values;

      if (__bounded)
      {
        __random_values = {__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen)};
      }
      else
      {
        __random_values = {__unbounded_dist(__gen), __unbounded_dist(__gen), __unbounded_dist(__gen),
                           __unbounded_dist(__gen)};
      }

      if (!__bounded)
      {
        __random_values = vmulq_f32(__random_values, vrecpeq_f32(__scale));
      }

      vst1q_f32(&this->__data_[__i], __random_values);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    const neon_f32 __scale = vdupq_n_f32(static_cast<_f32>(RAND_MAX));
    for (; __i + _ARM64_REG_WIDTH <= static_cast<index_type>(__s); __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __rand_vals = {static_cast<_f32>(__unbounded_dist(__gen)), static_cast<_f32>(__unbounded_dist(__gen)),
                              static_cast<_f32>(__unbounded_dist(__gen)), static_cast<_f32>(__unbounded_dist(__gen))};
      __rand_vals          = vmulq_f32(__rand_vals, vrecpeq_f32(__scale));
      neon_u32 __int_vals  = vcvtq_u32_f32(__rand_vals);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __int_vals);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    const neon_f32 __scale = vdupq_n_f32(static_cast<_f32>(RAND_MAX));
    for (; __i + _ARM64_REG_WIDTH <= static_cast<index_type>(__s); __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __rand_vals = {static_cast<_f32>(__unbounded_dist(__gen)), static_cast<_f32>(__unbounded_dist(__gen)),
                              static_cast<_f32>(__unbounded_dist(__gen)), static_cast<_f32>(__unbounded_dist(__gen))};
      __rand_vals          = vmulq_f32(__rand_vals, vrecpeq_f32(__scale));
      neon_s32 __int_vals  = vcvtq_s32_f32(__rand_vals);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __int_vals);
    }
  }
#endif

  for (; __i < static_cast<index_type>(__s); __i++)
  {
    this->__data_[__i] = value_type(__bounded ? __bounded_dist(__gen) : __unbounded_dist(__gen));
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::clone() const {
  data_t     __d = this->__data_;
  shape_type __s = this->__shape_;
  return __self(__d, __s);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::negative_() const {
  this->__check_is_arithmetic_type("negative_: template type must be an arithmetic type");

  index_type __i = 0;

#if defined(__ARM_NEON)
  using neon_type = typename std::conditional<std::is_same<value_type, _f32>::value, neon_f32, neon_s32>::type;

  const index_type __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  neon_type        __neg_multiplier = vdupq_n(-1);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    neon_type __v   = vld1q(reinterpret_cast<const neon_type*>(&this->__data_[__i]));
    neon_type __neg = vmulq(__v, __neg_multiplier);
    vst1q(reinterpret_cast<neon_type*>(&this->__data_[__i]), __neg);
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = -this->__data_[__i];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::negative() const {
  __self __ret = this->clone();
  __ret.negative_();
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::permute_(const index_type __dim) const {
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::permute(const index_type __dim) const {}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::repeat_(const data_t& __d) const {
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::repeat(const data_t& __d) const {
  __self __ret = this->clone();
  __ret.repeat_(__d);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sort(index_type __dim, bool __descending) const {}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::fill(const value_type __val) const {
  __self __ret = this->clone();
  __ret.fill_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fill(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.fill_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::resize_as(const shape_type __sh) const {
  /*
    
  */
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::all() const {
  this->__check_is_arithmetic_type("all: template type must be an arithmetic type");

  bool       result = true;
  index_type __i    = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    if (this->__data_[__i] == static_cast<value_type>(0))
    {
      result = false;
      break;
    }
  }

  tensor output;
  output.__data_ = {result ? static_cast<value_type>(1) : static_cast<value_type>(0)};

  return output;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::any() const {
  this->__check_is_arithmetic_type("any: template type must be an arithmetic type");

  bool result = false;

  for (index_type __i = 0; __i < this->__data_.size(); ++__i)
  {
    if (this->__data_[__i] != static_cast<value_type>(0))
    {
      result = true;
      break;
    }
  }

  tensor output;
  output.__data_ = {result ? static_cast<value_type>(1) : static_cast<value_type>(0)};

  return output;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::gcd(const tensor& __other) const {}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::gcd(const value_type __val) const {}


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

  tensor<value_type> __max_values  = this->argmax(__dim);
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
    // minor           = this->get_minor(0, col);
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


template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_or_(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform logical OR on non-integral and non-boolean values");
  }

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value || std::is_same<value_type, bool>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(static_cast<_s32>(__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __or       = vorrq_s32(__data_vec, __val_vec);

      vst1q_s32(&this->__data_[__i], __or);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    neon_u32 __val_vec = vdupq_n_u32(static_cast<_u32>(__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __or       = vorrq_u32(__data_vec, __val_vec);

      vst1q_u32(&this->__data_[__i], __or);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] || __val);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_xor_(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
  }

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value || std::is_same<value_type, bool>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(static_cast<_s32>(__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __xor      = veorq_s32(__data_vec, __val_vec);

      vst1q_s32(&this->__data_[__i], __xor);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    neon_u32 __val_vec = vdupq_n_u32(static_cast<_u32>(__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __xor      = veorq_u32(__data_vec, __val_vec);

      vst1q_u32(&this->__data_[__i], __xor);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] ^ __val);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_and_(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise and of non-integral and non-boolean value");
  }

  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_type, _s32>::value)
  {
    neon_s32 __vals = vdupq_n_s32(reinterpret_cast<_s32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __and = vandq_s32(__vec, __vals);

      vst1q_s32(&this->__data_[__i], __and);
    }
  }
  else if constexpr (std::is_same<value_type, _u32>::value)
  {
    neon_u32 __vals = vdupq_n_u32(reinterpret_cast<_u32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __and = vandq_u32(__vec, __vals);

      vst1q_u32(&this->__data_[__i], __and);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(this->__data_[__i] && __val);
  }

  return *this;
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_not() const {
  tensor<bool> __ret = this->bool_();
  __ret.logical_not_();
  return __ret;
}


template<class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const value_type __val) const {
  tensor<bool> __ret = this->clone().bool_();
  __ret.logical_or_(__val);
  return __ret;
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const tensor& __other) const {
  tensor<bool> __ret = this->clone().bool_();
  __ret.logical_or_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.logical_xor_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const value_type __val) const {
  __self __ret = this->clone();
  __ret.logical_xor(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const tensor& __other) const {
  __self __ret = this->clone();
  __ret.logical_and_(__other);
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const value_type __val) const {
  __self __ret = this->clone();
  __ret.logical_and_(__val);
  return __ret;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_or_(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise not of non-integral and non-boolean value");
  }

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __or_vec    = vornq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __or_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __or_vec    = vornq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __or_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = (this->__data_[__i] || __other[__i]);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_xor_(const tensor<_Tp>& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
  }

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __xor_vec   = veorq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __xor_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __xor_vec   = veorq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __xor_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = (this->__data_[__i] ^ __other[__i]);
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::logical_and_(const tensor<_Tp>& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element-wise and of non-integral and non-boolean value");
  }

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec  = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __other_vec = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __and_vec   = vandq_u32(__data_vec, __other_vec);

      vst1q_u32(reinterpret_cast<_u32*>(&this->__data_[__i]), __and_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec  = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __other_vec = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __and_vec   = vandq_s32(__data_vec, __other_vec);

      vst1q_s32(reinterpret_cast<_s32*>(&this->__data_[__i]), __and_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = (this->__data_[__i] && __other[__i]);
  }

  return *this;
}


template<class _Tp>
bool tensor<_Tp>::operator!=(const tensor& __other) const {
  return !(*this == __other);
}

template<class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::operator[](const index_type __in) {
  if (__in >= this->__data_.size() || __in < 0)
  {
    throw std::out_of_range("Access index is out of range");
  }

  return this->__data_[__in];
}

template<class _Tp>
tensor<_Tp>::const_reference tensor<_Tp>::operator[](const index_type __in) const {
  if (__in >= this->__data_.size() || __in < 0)
  {
    throw std::out_of_range("Access index is out of range");
  }

  return this->__data_[__in];
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  if (__other.shape() != this->__shape_)
  {
    throw std::invalid_argument("Cannot add two tensors with different shapes");
  }

  data_t     __d(this->__data_.size());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec1   = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __vec2   = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __result = vaddq_f32(__vec1, __vec2);

      vst1q_f32(reinterpret_cast<_f32*>(&__d[__i]), __result);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec1   = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __vec2   = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __result = vaddq_s32(__vec1, __vec2);

      vst1q_s32(reinterpret_cast<_s32*>(&__d[__i]), __result);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec1   = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __vec2   = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __result = vaddq_u32(__vec1, __vec2);

      vst1q_u32(reinterpret_cast<_u32*>(&__d[__i]), __result);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __d[__i] = this->__data_[__i] + __other[__i];
  }

  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const value_type __val) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  data_t     __d(this->__data_.size());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __val_vec = vdupq_n_f32(reinterpret_cast<_f32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec = vld1q_f32(reinterpret_castr<const _f32*>(&this->__data_[__i]));
      neon_f32 __res = vaddq_f32(__vec, __val_vec);

      vst1q_f32(&__d[__i], __res);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(reinterpret_cast<_s32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_castr<const _s32*>(&this->__data_[__i]));
      neon_s32 __res = vaddq_s32(__vec, __val_vec);

      vst1q_s32(&__d[__i], __res);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __val_vec = vdupq_n_u32(reinterpret_cast<_u32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_castr<const _u32*>(&this->__data_[__i]));
      neon_u32 __res = vaddq_u32(__vec, __val_vec);

      vst1q_u32(&__d[__i], __res);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __d[__i] = this->__data_[__i] + __val;
  }

  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator+=(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");

  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)

#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] += __other[__i];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator+=(const_reference __val) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __val_vec = vdupq_n_f32(reinterpret_cast<_f32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __data_vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __add_vec  = vaddq_f32(__data_vec, __val_vec);

      vst1q_f32(&this->__data_[__i], __add_vec);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __val_vec = vdupq_n_s32(reinterpret_cast<_s32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __data_vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __add_vec  = vaddq_s32(__data_vec, __val_vec);

      vst1q_s32(&this->__data_[__i], __add_vec);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __val_vec = vdupq_n_u32(reinterpret_cast<_u32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __data_vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __add_vec  = vaddq_u32(__data_vec, __val_vec);

      vst1q_u32(&this->__data_[__i], __add_vec);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = this->__data_[__i] + __val;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");

  if (__other.shape() != this->__shape_)
  {
    throw std::invalid_argument("Cannot add two tensors with different shapes");
  }

  data_t     __d(this->__data_.size());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __oth = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __sub = vsubq_f32(__vec, __oth);

      vst1q_f32(&__d[__i], __sub);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __oth = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __sub = vsubq_s32(__vec, __oth);

      vst1q_s32(&__d[__i], __sub);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __oth = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __sub = vsubq_u32(__vec, __oth);

      vst1q_u32(&__d[__i], __sub);
    }
  }
#endif

  for (; __i < this->__data_[__i]; __i++)
  {
    __d[__i] = this->__data_[__i] - __other[__i];
  }

  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const value_type __val) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  data_t     __d(this->__data_.size());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __vals = vdupq_n_f32(reinterpret_cast<_f32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __sub = vsubq_f32(__vec, __vals);

      vst1q_f32(&__d[__i], __sub);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __vals = vdupq_n_s32(reinterpret_cast<_s32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __sub = vsubq_s32(__vec, __vals);

      vst1q_s32(&__d[__i], __sub);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __vals = vdupq_n_u32(reinterpret_cast<_u32>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __sub = vsubq_u32(__vec, __vals);

      vst1q_u32(&__d[__i], __sub);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    __d[__i] = this->__data_[__i] - __val;
  }

  return __self(*this);
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator-=(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __oth = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __sub = vsubq_f32(__vec, __oth);

      vst1q_f32(&this->__data_[__i], __sub);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __oth = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __sub = vsubq_s32(__vec, __oth);

      vst1q_s32(&this->__data_[__i], __sub);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __oth = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __sub = vsubq_u32(__vec, __oth);

      vst1q_u32(&this->__data_[__i], __sub);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] -= __other[__i];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator*=(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __oth = vld1q_f32(reinterpret_cast<const _f32*>(&__other[__i]));
      neon_f32 __mul = vmulq_f16(__vec, __oth);

      vst1q_f32(&this->__data_[__i], __mul);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __oth = vld1q_s32(reinterpret_cast<const _s32*>(&__other[__i]));
      neon_s32 __mul = vmulq_s16(__vec, __oth);

      vst1q_s32(&this->__data_[__i], __mul);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __oth = vld1q_u32(reinterpret_cast<const _u32*>(&__other[__i]));
      neon_u32 __mul = vmulq_u16(__vec, __oth);

      vst1q_u32(&this->__data_[__i], __mul);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] *= __other[__i];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator*=(const_reference __val) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  index_type __i = 0;

#if defined(__ARM_NEON)

#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] *= __val;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator/=(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  assert(this->__shape_ == __other.shape());
  index_type __i = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] /= __other[__i];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator/=(const_reference __val) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  index_type __i = 0;

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] /= __val;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator-=(const_reference __val) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  index_type __i = 0;

#if defined(__ARM_NEON)
  const index_type __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_floating_point<value_type>::value)
  {
    neon_f32 __val = vld1q_f32(reinterpret_cast<const _f32*>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_f32 __vec = vld1q_f32(reinterpret_cast<const _f32*>(&this->__data_[__i]));
      neon_f32 __sub = vsubq_f32(__vec, __val);

      vst1q_f32(&this->__data_[__i], __sub);
    }
  }
  else if constexpr (std::is_unsigned<value_type>::value)
  {
    neon_u32 __val = vld1q_u32(reinterpret_cast<const _u32*>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_u32 __vec = vld1q_u32(reinterpret_cast<const _u32*>(&this->__data_[__i]));
      neon_u32 __sub = vsubq_u32(__vec, __val);

      vst1q_u32(&this->__data_[__i], __sub);
    }
  }
  else if constexpr (std::is_signed<value_type>::value)
  {
    neon_s32 __val = vld1q_s32(reinterpret_cast<const _s32*>(&__val));

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      neon_s32 __vec = vld1q_s32(reinterpret_cast<const _s32*>(&this->__data_[__i]));
      neon_s32 __sub = vsubq_s32(__vec, __val);

      vst1q_s32(&this->__data_[__i], __sub);
    }
  }
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] -= __val;
  }

  return *this;
}

template<class _Tp>
bool tensor<_Tp>::operator==(const tensor& __other) const {
  if ((this->__shape_ != __other.shape()) && (this->__strides_ != __other.strides()))
  {
    return false;
  }

  return this->__data_ == __other.storage();
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator=(const tensor& __other) const {
  if (this != &__other)
  {
    this->__data_    = __other.storage();
    this->__shape_   = __other.shape();
    this->__strides_ = __other.strides();
  }
  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator=(tensor&& __other) const noexcept {
  if (this != &__other)
  {
    this->__data_    = std::move(__other.storage());
    this->__shape_   = std::move(__other.shape());
    this->__strides_ = std::move(__other.strides());
  }
  return *this;
}

template<class _Tp>
tensor<bool>& tensor<_Tp>::operator!() const {
  static_assert(std::is_same<value_type, bool>::value);

  size_t __i = 0;

#if defined(__ARM_NEON)
#endif

  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = !(this->__data_[__i]);
  }

  return *this;
}


template<class _Tp>
tensor<_s32> tensor<_Tp>::int32_() const {
  static_assert(std::is_convertible<value_type, _s32>::value);

  if (this->empty())
  {
    return tensor<_s32>({}, this->__shape_);
  }

  std::vector<_s32> __d;
  index_type        __i = 0;

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

  return tensor<_s32>(__d, this->__shape_);
}

template<class _Tp>
tensor<_u32> tensor<_Tp>::uint32_() const {
  static_assert(std::is_convertible<value_type, _u32>::value);

  if (this->empty())
  {
    return tensor<_u32>({}, this->__shape_);
  }

  std::vector<_u32> __d;
  index_type        __i = 0;

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

  return tensor<_u32>(__d, this->__shape_);
}

template<class _Tp>
tensor<_f32> tensor<_Tp>::float32_() const {
  static_assert(std::is_convertible<value_type, _f32>::value, "Tensor value type must be convertible to _f32.");

  if (this->empty())
  {
    return tensor<_f32>({}, this->__shape_);
  }

  std::vector<_f32> __d(this->__data_.size());
  index_type        __i = 0;

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

  return tensor<_f32>(__d, this->__shape_);
}

template<class _Tp>
tensor<_f64> tensor<_Tp>::double_() const {
  static_assert(std::is_convertible<value_type, _f64>::value, "Tensor value type must be convertible to _f64.");

  if (this->empty())
  {
    return tensor<_f64>({}, this->__shape_);
  }

  std::vector<_f64> __d(this->__data_.size());
  index_type        __i = 0;

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

  return tensor<_f64>(__d, this->__shape_);
}

template<class _Tp>
tensor<uint64_t> tensor<_Tp>::unsigned_long_() const {
  static_assert(std::is_convertible<value_type, uint64_t>::value, "Tensor value type must be convertible to uint64_t.");

  if (this->empty())
  {
    return tensor<uint64_t>({}, this->__shape_);
  }

  std::vector<uint64_t> __d(this->__data_.size());
  index_type            __i = 0;

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

  return tensor<uint64_t>(__d, this->__shape_);
}

template<class _Tp>
tensor<int64_t> tensor<_Tp>::long_() const {
  static_assert(std::is_convertible<value_type, int64_t>::value, "Tensor value type must be convertible to int64_t.");

  if (this->empty())
  {
    return tensor<int64_t>({}, this->__shape_);
  }

  std::vector<int64_t> __d(this->__data_.size());
  index_type           __i = 0;

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

  return tensor<int64_t>(__d, this->__shape_);
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
