#pragma once

#include <arm_neon.h>
#include <cstdint>
#include <cstdlib>
#include <shape.hpp>
#include <stdint.h>
#include <vector>


using _s8  = int8_t;
using _s16 = int16_t;
using _s32 = int32_t;
using _s64 = int64_t;

using _u8  = uint8_t;
using _u16 = uint16_t;
using _u32 = uint32_t;
using _u64 = uint64_t;

using _f16 = float16_t;
using _f32 = float32_t;
using _f64 = float64_t;

using neon_s8  = int8x16_t;
using neon_s16 = int16x8_t;
using neon_s32 = int32x4_t;
using neon_s64 = int64x2_t;

using neon_u8  = uint8x16_t;
using neon_u16 = uint16x8_t;
using neon_u32 = uint32x4_t;
using neon_u64 = uint64x2_t;

using neon_f16 = float16x8_t;
using neon_f32 = float32x4_t;
using neon_f64 = float64x2_t;


constexpr int _ARM64_REG_WIDTH = 128;  // 128 bit wide register

enum class Device : int {
  CPU,
  CUDA
};

template<class _Tp>
class TensorBase
{
 public:
  using self            = TensorBase;
  using value_type      = _Tp;
  using data_t          = std::vector<value_type>;
  using index_type      = uint64_t;
  using shape_type      = std::vector<index_type>;
  using reference       = value_type&;
  using const_reference = const value_type&;

  static constexpr std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
  static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

  TensorBase() = default;

  TensorBase(const TensorBase& t) :
      __data_(t.storage()),
      __shape_(t.shape()),
      __device_(t.device()) {}

  TensorBase(TensorBase&& t) noexcept :
      __data_(std::move(t.storage())),
      __shape_(std::move(t.shape())),
      __device_(std::move(t.device())) {}

  TensorBase(const shape::Shape& sh, const TensorBase& other) :
      __data_(other.storage()),
      __shape_(__shape_),
      __device_(other.device()) {}

  TensorBase(const shape::Shape& sh, std::initializer_list<value_type> init_list, Device d = Device::CPU) :
      __shape_(sh),
      __device_(d) {
    if (init_list.size() != static_cast<std::size_t>(__shape_.flatten_size()))
    {
      throw std::invalid_argument("Initializer list size must match tensor size");
    }

    __data_ = data_t(init_list);
  }

  explicit TensorBase(const shape::Shape& sh, const_reference v, Device d = Device::CPU) :
      __shape_(sh),
      __data_(sh.flatten_size(), v),
      __device_(d) {
    __shape_.compute_strides();
  }

  explicit TensorBase(const shape::Shape& sh, Device d = Device::CPU) :
      __shape_(sh),
      __device_(d) {
    __data_ = data_t(__shape_.flatten_size());
    __shape_.compute_strides();
  }

  explicit TensorBase(shape::Shape& sh, const data_t& d, Device dev = Device::CPU) :
      __shape_(sh),
      __device_(dev) {

    if (d.size() != static_cast<std::size_t>(__shape_.flatten_size()))
    {
      throw std::invalid_argument("Initial data vector must match the tensor size : " + std::to_string(d.size())
                                  + " != " + std::to_string(__shape_.flatten_size()));
    }

    __data_ = d;
    __shape_.compute_strides();
  }

 protected:
  void compute_strides() const {
    if (__shape_.empty())
    {
      throw error::shape_error("Shape must be initialized before computing strides");
    }

    __shape_.compute_strides();
  }

  mutable data_t       __data_;
  mutable shape::Shape __shape_;
  Device               __device_;
  bool                 __is_cuda_tensor_ = false;

 public:
  data_t storage() const noexcept { return __data_; }

  shape::Shape shape() const noexcept { return __shape_; }

  Device device() const noexcept { return __device_; }

  void set_device(const Device d) noexcept {
    __device_         = d;
    __is_cuda_tensor_ = __device_ == Device::CUDA;
  }

  data_t& storage_() const { return std::ref<data_t>(__data_); }

  shape::Shape& shape_() const { return std::ref<shape::Shape>(__shape_); }

  // Device& device_() const { return std::ref<Device>(__device_); }

  std::size_t n_dims() const noexcept { return __shape_.size(); }

  index_type size(const index_type dimension) const {
    if (dimension < 0 || dimension > static_cast<index_type>(n_dims()))
    {
      throw std::invalid_argument("dimension input is out of range");
    }

    if (dimension == 0)
    {
      return __data_.size();
    }

    return __shape_[dimension - 1];
  }

  index_type capacity() const noexcept { return __data_.capacity(); }

  index_type hash() const {
    index_type            hash_v = 0;
    std::hash<value_type> hasher;
    for (const auto& elem : __data_)
    {
      hash_v ^= hasher(elem) + 0x9e3779b9 + (hash_v << 6) + (hash_v >> 2);
    }
    return hash_v;
  }

  reference at_(shape::Shape idx) {
    if (idx.empty())
    {
      throw error::index_error("Passing an empty vector as indices for a tensor");
    }

    index_type i = this->shape().compute_index(idx);

    if (i < 0 || i >= __data_.size())
    {
      throw error::index_error("input indices are out of bounds");
    }

    return __data_[i];
  }

  const_reference at(const shape::Shape idx) const { return at_(idx); }

  reference operator[](const index_type idx) {
    if (idx < 0 || idx >= __data_.size())
    {
      throw error::index_error("input index is out of bounds");
    }

    return __data_[idx];
  }

  const_reference operator[](const index_type idx) const { return (*this)[idx]; }

  reference operator()(std::initializer_list<index_type> index_list) { return at_(shape::Shape(index_list)); }

  const_reference operator()(std::initializer_list<index_type> index_list) const {
    return at(shape::Shape(index_list));
  }

  bool empty() const { return __data_.empty(); }

  TensorBase<bool> bool_() const {
    if (!std::is_convertible_v<value_type, bool>)
    {
      throw error::type_error("Type must be convertible to bool");
    }

    std::vector<bool> d(__data_.size());
    index_type        i = 0;

    for (const auto& elem : __data_)
    {
      d[i++] = bool(elem);
    }

    return tensor<bool>(this->shape(), d);
  }

  void print() const {
    printRecursive(0, 0, this->shape());
    std::cout << std::endl;
  }
};  // class TensorBase