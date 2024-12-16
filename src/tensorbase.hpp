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

  bool    operator==(const tensor& __other) const;
  bool    operator!=(const tensor& __other) const;
  tensor  operator+(const tensor& __other) const;
  tensor  operator-(const tensor& __other) const;
  tensor  operator+(const value_type __val) const;
  tensor  operator-(const value_type __val) const;
  tensor& operator-=(const tensor& __other) const;
  tensor& operator+=(const tensor& __other) const;
  tensor& operator*=(const tensor& __other) const;
  tensor& operator/=(const tensor& __other) const;
  tensor& operator+=(const_reference __val) const;
  tensor& operator-=(const_reference __val) const;
  tensor& operator/=(const_reference __val) const;
  tensor& operator*=(const_reference __val) const;
  tensor& operator=(const tensor& __other) const;
  tensor& operator=(tensor&& __other) const noexcept;


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
