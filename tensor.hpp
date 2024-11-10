/**
 * @file tensor.hpp
 * @brief Tensor Library for Efficient Mathematical and Linear Algebra Operations
 * 
 * This library provides a comprehensive implementation of a tensor class,
 * designed for high-performance numerical computing, matrix operations, and
 * machine learning applications. Built with support for scalar operations,
 * element-wise transformations, and SIMD acceleration on supported architectures 
 * (e.g., ARM NEON), as well as CUDA GPU acceleration.
 * 
 * Key Features:
 * - Basic arithmetic operations on tensors (addition, subtraction, multiplication, etc.)
 * - Element-wise operations (floor, ceil, arctangent, etc.)
 * - SIMD optimization for ARM NEON for faster computation on supported hardware
 * - Flexible scalar type support (float, double, etc.)
 * 
 * Example Usage:
 * ```
 * tensor<float> A({3, 3});  // Creates a 3x3 tensor of type float
 * A.fill(1.0f);              // Fills tensor with value 1.0
 * A.ceil_();                 // Applies ceil operation element-wise
 * ```
 * 
 * Dependencies:
 * - Standard C++ libraries (`<vector>`, `<algorithm>`, `<cmath>`)
 * - ARM NEON for SIMD support (optional, for ARM processors)
 * 
 * License:
 * - MIT License
 * 
 * Author: [Your Name]
 * Date: [Creation Date]
 * Version: 1.0.0
 */


#pragma once

// mandate standard headers

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

constexpr int _ARM64_REG_WIDTH = 4;
constexpr int _AVX_REG_WIDTH   = 8;


enum class Device {
  CPU,
  CUDA
};


// this should be a replacement for std::vector as a container of the tensor storage
// because you might want to use a crazy tensor with 100 gb of data
// still needs some work to be done
template<class _Tp>
class __container__
{
 private:
  using value_t         = _Tp;
  using size_type       = uint64_t;
  using container       = std::vector<value_t>;
  using reference       = value_t&;
  using const_reference = const value_t&;

 protected:
  // the underlying storage will be a stack of vector
  // providing as much memory capacity as possible
  std::vector<container> __data_;
  size_type              __size_ = 0;

  size_type __max_vector_size() const noexcept { return std::vector<value_t>().max_size(); }

  std::tuple<int, size_t> __compute_index(const size_type __idx) const {
    assert(__idx <= this->__size_ && "Index out of range");
    int    __vec   = static_cast<int>(__idx / this->__max_vector_size());
    size_t __index = static_cast<size_t>(__idx % this->__max_vector_size());
    return std::make_tuple(__vec, __index);
  }

  std::tuple<int, size_t> __compute_size(const size_type __idx) const noexcept {
    int    __v = static_cast<int>(__idx / this->__max_vector_size());
    size_t __i = static_cast<size_t>(__idx % this->__max_vector_size());
    return std::make_tuple(__v, __i);
  }

 public:
  __container__() = default;

  __container__(const size_type __s, value_t __v) {
    if (__s <= 0)
      throw std::invalid_argument("Cannot allocate negative or zero memory");

    if (__s <= this->__max_vector_size())
    {
      this->__data_.push_back(container(__s, __v));
      return;
    }

    auto [__vecs, __rem] = this->__compute_size(__s);
    for (int __i = 0; __i < __vecs; __i++)
    {
      container __c(__s, __v);
      this->__data_.push_back(__c);
    }

    container __c(__rem, __v);
    this->__data_.push_back(__c);
  }

  __container__(std::initializer_list<value_t> __l) {
    size_type __size = static_cast<size_type>(__l.size());

    if (__size <= this->__max_vector_size())
    {
      this->__data_.push_back(container(__l));
      return;
    }

    auto [__vecs, __rem] = this->__compute_size(__size);
    int __i = 0, __iter = this->__max_vector_size(), __start = 0;

    while (__i++ < __size)
    {
      this->__data_.push_back(container(__l.begin() + __start, __l.begin() + __iter));
      __iter += this->__max_vector_size();
      __start += this->__max_vector_size();
    }

    this->__data_.push_back(container(__l.begin() + __start, __l.begin() + __rem));
  }

  __container__(const size_type __s) noexcept {
    assert(__s >= 0 && __s <= INTMAX_MAX && "Invalid size initializer");
    this->__size_ = __s;
    int    __vecs = static_cast<int>(__s / this->__max_vector_size());
    size_t __rem  = static_cast<size_t>(__s % this->__max_vector_size());
    this->__data_.resize(__vecs + (__rem > 0 ? 1 : 0));
    for (int i = 0; i < __vecs; ++i)
      this->__data_[i].resize(this->__max_vector_size());

    if (__rem > 0)
      this->__data_[__vecs].resize(__rem);
  }

  reference operator[](const size_type __idx) {
    auto [__vec, __index] = this->__compute_index(__idx);
    return this->__data_[__vec][__index];
  }

  const_reference operator[](const size_type __idx) const {
    auto [__vec, __index] = this->__compute_index(__idx);
    return this->__data_[__vec][__index];
  }

  size_type size() const noexcept { return this->__size_; }

  void resize(const size_type __s) {
    if (__s < 0 || __s > INTMAX_MAX)
      throw std::invalid_argument("Invalid size type");

    if (__s > this->__size_)
    {
      size_type __vecs = __s / this->__max_vector_size();
      size_type __rem  = __s % this->__max_vector_size();
      this->__data_.resize(__vecs + (__rem > 0 ? 1 : 0));
      for (int i = 0; i < __vecs; ++i)
      {
        if (this->__data_[i].size() != this->__max_vector_size())
          this->__data_[i].resize(this->__max_vector_size());
      }

      if (__rem > 0)
        this->__data_[__vecs].resize(__rem);
    }
    else if (__s < this->__size_)
    {
      auto [__vec, __index] = this->__compute_index(__s);
      this->__data_.resize(__vec + 1);
      this->__data_.back().resize(__index);
    }
    this->__size_ = __s;
  }

  void push_back(const_reference __value) {
    if (this->__size_ == this->__max_vector_size() * this->__data_.size())
      this->__data_.emplace_back();

    auto [__vec, __index] = this->__compute_index(this->__size_);
    this->__data_[__vec].push_back(__value);
    this->__size_++;
  }

  void pop_back() {
    if (this->__size_ == 0)
      throw std::underflow_error("Container is empty.");

    auto [__vec, __index] = this->__compute_index(this->__size_ - 1);
    this->__data_[__vec].pop_back();
    this->__size_--;

    if (this->__data_.back().empty())
      this->__data_.pop_back();
  }
};  // __container__ class


template<class _Tp>
class tensor
{
 public:
  using __self          = tensor;
  using value_t         = _Tp;
  using index_t         = int64_t;
  using shape_t         = std::vector<index_t>;
  using reference       = value_t&;
  using const_reference = const value_t&;
  using pointer         = value_t*;
  using const_pointer   = const value_t*;

 private:
  __container__<value_t> __data_;
  shape_t                __shape_;
  std::vector<index_t>   __strides_;
  Device                 __device_;

 public:
  tensor() = default;

  explicit tensor(const shape_t& __sh, const value_t& __v, Device __d = Device::CPU) :
      __shape_(__sh),
      __data_(this->__computeSize(__sh), __v),
      __device_(__d) {
    this->__compute_strides();
  }

  explicit tensor(const shape_t& __sh, Device __d = Device::CPU) :
      __shape_(__sh),
      __device_(__d) {
    index_t __s   = this->__computeSize(__sh);
    this->__data_ = __container__<value_t>(__s);
    this->__compute_strides();
  }

  explicit tensor(const __container__<value_t>& __d, const shape_t& __sh, Device __dev = Device::CPU) :
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

  tensor(const shape_t& __sh, std::initializer_list<value_t> init_list, Device __d = Device::CPU) :
      __shape_(__sh),
      __device_(__d) {
    index_t __s = this->__computeSize(__sh);
    assert(init_list.size() == static_cast<size_t>(__s) && "Initializer list size must match tensor size");
    this->__data_ = __container__<value_t>(init_list);
    this->__compute_strides();
  }

  tensor(const shape_t& __sh, const tensor& __other) :
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

  tensor& operator=(const tensor& __other);

  tensor& operator=(tensor&& __other) noexcept;

  /**
     * @brief returns the linear data as stored in memory.
     * @return vector of elements stored in this->__data_.
     */
  __container__<value_t> storage() const noexcept { return this->__data_; }

  /**
     * @brief returns the shape of a tensor
     * @return vector of 64 bit integers indicating the size of each dimension 
    */
  shape_t shape() const noexcept { return this->__shape_; }

  /** 
     * @brief returns the strides of the dimensions
     * @return vector of 64 bit integers indicating the 'stride' of each dimension
    */
  shape_t strides() const noexcept { return this->__strides_; }

  /**
     * @brief returns which device the tensor is stored on
     * @return an element of the enum class Device (e.i CPU, GPU)
     */
  Device device() const noexcept { return this->__device_; }

  /**
     * @brief how many dimensions are in a tensor
     * @return unsigned long (aka 64 bit unsigned integer, or 32 bits depending on the system and compiler)
     */
  size_t n_dims() const noexcept { return this->__shape_.size(); }

  /**
     * @brief whats the size of a given dimension
     * @param __dim const long long
     * @return unsigned long (aka 64 bit unsigned integer, or 32 bits depending on the system and compiler)
     */
  index_t size(const index_t __dim) const;

  /**
     * @brief whats the capacity of the tensor (memory reserved)
     * @return unsigned long (aka 64 bit unsigned integer, or 32 bits depending on the system and compiler)
     */
  index_t capacity() const noexcept { return this->__data_.capacity(); }

  /**
     * @brief for direct modifiable access
     * @param __idx vector of integers that specify the multidimensional index 
     * @return reference to the element at the input index
     */
  reference at(shape_t __idx);

  /**
     * @brief for direct unmodifiable access
     * @param __idx vector of integers that specify the multidimensional index 
     * @return constant reference to the element at the input index
     */
  const_reference at(const shape_t __idx) const;

  /**
     * @brief for linear modifiable access
     * @param __in linear index in the tensor storage
     * @return reference to the element at the input index
     */
  reference operator[](const index_t __in);

  /**
     * @brief for linear unmodifiable access
     * @param __in linear index in the tensor storage
     * @return constant reference to the element at the input index
     */
  const_reference operator[](const index_t __in) const;

  /**
     * @brief adds two tensors of the same shape element wise
     * @param __other other operand
     * @return the sum of two tensors element wise
     */
  tensor operator+(const tensor& __other) const;

  /**
     * @brief subtracts the input tensor from self element wise
     * @param __other other operand
     * @return the difference of two tensors element wise
     */
  tensor operator-(const tensor& __other) const;

  /**
     * @brief in place version this - other
     * @param __other other operand
     * @return this - other
     */
  tensor operator-=(const tensor& __other) const;

  /**
     * @brief in place version of this + other
     * @param __other other operand
     * @return this + other
     */
  tensor operator+=(const tensor& __other) const;

  /**
     * @brief in place version of this * other
     * @param __other other operand
     * @return this * other
     */
  tensor operator*=(const tensor& __other) const;

  /** 
     * @brief in place version of this / other
     * @param __other other operand
     * @return this / other
     */
  tensor operator/=(const tensor& __other) const;

  /**
     * @brief adds a scalar to each element of the tensor
     * @param __scalar a value of an arithmetic type
     * @return the sum of self and __scalar element wise
     */
  tensor operator+(const value_t _scalar) const;

  /**
     * @brief subtracts a scalar from each element of the tensor 
     * @param __scalar a value of an arithmetic type
     * @return the difference between self and __scalar element wise
     */
  tensor operator-(const value_t __scalar) const;

  /**
     * @brief in place version of this + __scalar
     * @param __scalar a value of an arithmetic type
     * @return this + __scalar
     */
  tensor operator+=(const_reference __scalar) const;

  /**
     * @brief in place version of this - __scalar
     * @param __scalar a value of an arithmetic type
     * @return this - __scalar
     */
  tensor operator-=(const_reference __scalar) const;

  /**
     * @brief in place version of this / __scalar
     * @param __scalar a value of an arithmetic type
     * @return this / __scalar
     */
  tensor operator/=(const_reference __scalar) const;

  /**
     * @brief in place version of this * __scalar
     * @param __scalar a value of an arithmetic type
     * @return this * __scalar
     */
  tensor operator*=(const_reference __scalar) const;

  /**
     * @brief check whether or not two tensors are equal
     * @param __other other tensor
     * @return boolean value indicating if the tensors are equal or not
     */
  bool operator==(const tensor& __other) const;

  /**
     * @brief check whether or not two tensors are not equal
     * @param __other other tensor
     * @return boolean value indicating if the tensors are equal or not
     */
  bool operator!=(const tensor& __other) const { return !(*this == __other); }

  /**
     * @brief checks whether or not the tensor contains at least one element
     * @return boolean value (true if empty, false if not)
     */
  bool empty() const { return this->__data_.empty(); }

  /**
     * @brief calculates the mean average of all elements in the tensor
     * @return the mean average 
     */
  double mean() const;

  /**
     * @brief logical not over all elements in the tensor
     * @return a new tensor where each element is the logical not of the corresponding element in this
     */
  tensor<bool> logical_not() const;

  /**
     * @brief logical or over all elements in the tensor with an input value
     * @param __val a value that supports the operator 'or'
     * @return a new tensor where each element is the logical or of the corresponding element in this with __val
     */
  tensor<bool> logical_or(const value_t __val) const;

  /**
     * @brief logical or of this and other element wise
     * @param __other a tensor of the same shape as this
     * @return a new tensor where each element is the logical or of the corresponding element in this with __other
     */
  tensor<bool> logical_or(const tensor& __other) const;

  /**
     * @brief Performs a less-than-or-equal-to comparison element-wise with another tensor.
     * 
     * The tensors must have the same shape.
     * 
     * @param __other The other tensor to compare with.
     * @return A new tensor of booleans indicating the result of the comparison.
     */
  tensor<bool> less_equal(const tensor& __other) const;

  /**
     * @brief Performs a less-than-or-equal-to comparison with a scalar value.
     * 
     * @param __val The scalar value to compare with.
     * @return A new tensor of booleans indicating the result of the comparison.
     */
  tensor<bool> less_equal(const value_t __val) const;

  /**
     * @brief Performs a greater-than-or-equal-to comparison element-wise with another tensor.
     * 
     * The tensors must have the same shape.
     * 
     * @param __other The other tensor to compare with.
     * @return A new tensor of booleans indicating the result of the comparison.
     */
  tensor<bool> greater_equal(const tensor& __other) const;

  /**
     * @brief Performs a greater-than-or-equal-to comparison with a scalar value.
     * 
     * @param __val The scalar value to compare with.
     * @return A new tensor of booleans indicating the result of the comparison.
     */
  tensor<bool> greater_equal(const value_t __val) const;

  /**
     * @brief Performs an element-wise equality comparison with another tensor.
     * 
     * The tensors must have the same shape.
     * 
     * @param __other The other tensor to compare with.
     * @return A new tensor of booleans indicating equality element-wise.
     */
  tensor<bool> equal(const tensor& __other) const;

  /**
     * @brief Performs an equality comparison with a scalar value.
     * 
     * @param __val The scalar value to compare with.
     * @return A new tensor of booleans indicating equality element-wise.
     */
  tensor<bool> equal(const value_t __val) const;

  /**
     * @brief Counts the number of non-zero elements in the tensor.
     * 
     * Optionally, can count non-zero elements along a specific dimension.
     * 
     * @param __dim (Optional) The dimension to count along. Defaults to -1, meaning all dimensions.
     * @return The count of non-zero elements.
     */
  index_t count_nonzero(index_t __dim = -1) const;

  /**
     * @brief Slices the tensor along a specified dimension.
     * 
     * Creates a new tensor that is a slice of the original tensor.
     * 
     * @param __dim The dimension to slice along.
     * @param __start (Optional) The starting index for the slice. Defaults to the beginning of the dimension.
     * @param __end (Optional) The ending index for the slice. Defaults to the end of the dimension.
     * @param __step The step size for the slice.
     * @return A new tensor representing the sliced data.
     */
  tensor slice(index_t __dim, std::optional<index_t> __start, std::optional<index_t> __end, index_t __step) const;

  /**
     * @brief Computes the element-wise maximum between two tensors.
     * 
     * The tensors must have the same shape.
     * 
     * @param __other The other tensor to compare with.
     * @return A new tensor where each element is the maximum of the corresponding elements.
     */
  tensor fmax(const tensor& __other) const;

  /**
     * @brief Computes the element-wise maximum between the tensor and a scalar value.
     * 
     * @param __val The scalar value to compare with.
     * @return A new tensor where each element is the maximum of the tensor element and the scalar.
     */
  tensor fmax(const value_t __val) const;

  /**
     * @brief Performs an in-place element-wise maximum between the tensor and another tensor.
     * 
     * The tensors must have the same shape.
     * 
     * @param __other The other tensor to compare with.
     */
  void fmax_(const tensor& __other);

  /**
     * @brief Performs an in-place element-wise maximum between the tensor and a scalar value.
     * 
     * @param __val The scalar value to compare with.
     */
  void fmax_(const value_t __val);

  /**
     * @brief Computes the element-wise remainder (modulus) between two tensors.
     * 
     * The tensors must have the same shape.
     * 
     * @param __other The other tensor to use for the modulus operation.
     * @return A new tensor where each element is the result of the modulus operation.
     */
  tensor fmod(const tensor& __other) const;

  /**
     * @brief Computes the element-wise remainder (modulus) between the tensor and a scalar value.
     * 
     * @param __val The scalar value to use for the modulus operation.
     * @return A new tensor where each element is the result of the modulus operation.
     */
  tensor fmod(const value_t __val) const;

  /**
     * @brief Returns the fractional part of each element in the tensor.
     * 
     * The fractional part is the element minus its integer component.
     * 
     * @return A new tensor containing the fractional part of each element.
     */
  tensor frac() const;

  /**
     * @brief Computes the natural logarithm (base e) of each element in the tensor.
     * 
     * @return A new tensor containing the natural logarithm of each element.
     */
  tensor log() const;

  /**
     * @brief Computes the base-10 logarithm of each element in the tensor.
     * 
     * @return A new tensor containing the base-10 logarithm of each element.
     */
  tensor log10() const;

  /**
     * @brief Computes the base-2 logarithm of each element in the tensor.
     * 
     * @return A new tensor containing the base-2 logarithm of each element.
     */
  tensor log2() const;

  /**
     * @brief Computes the exponential (base e) of each element in the tensor.
     * 
     * @return A new tensor where each element is the exponential of the corresponding element in the original tensor.
     */
  tensor exp() const;

  /**
     * @brief Computes the square root of each element in the tensor.
     * 
     * @return A new tensor containing the square root of each element.
     */
  tensor sqrt() const;

  /**
     * @brief Sums the elements of the tensor along the specified axis.
     * 
     * @param __axis The axis along which to compute the sum.
     * @return A tensor containing the sum of the elements along the specified axis.
     */
  tensor sum(const index_t __axis) const;

  /**
     * @brief Returns the row at the specified index in the tensor.
     * 
     * @param __index The index of the row to retrieve.
     * @return A new tensor representing the row at the specified index.
     */
  tensor row(const index_t __index) const;

  /**
     * @brief Returns the column at the specified index in the tensor.
     * 
     * @param __index The index of the column to retrieve.
     * @return A new tensor representing the column at the specified index.
     */
  tensor col(const index_t __index) const;

  /**
     * @brief Computes the ceiling of each element in the tensor.
     * 
     * Each element is rounded up to the nearest integer.
     * 
     * @return A new tensor where each element is the ceiling of the corresponding element in the original tensor.
     */
  tensor ceil() const;

  /**
     * @brief Computes the floor of each element in the tensor.
     * 
     * Each element is rounded down to the nearest integer.
     * 
     * @return A new tensor where each element is the floor of the corresponding element in the original tensor.
     */
  tensor floor() const;

  /**
     * @brief Creates a clone of the tensor.
     * 
     * The clone is a deep copy, meaning the data and shape are duplicated.
     * 
     * @return A new tensor that is a clone of the original tensor.
     */
  tensor clone() const;

  /**
     * @brief Clamps all elements in the tensor to be within the specified range.
     * 
     * Each element is clamped such that it is not less than `__min_val` and not greater than `__max_val`.
     * 
     * @param __min_val The minimum value for clamping. If `nullptr`, no minimum clamping is applied.
     * @param __max_val The maximum value for clamping. If `nullptr`, no maximum clamping is applied.
     * @return A new tensor where each element is clamped within the specified range.
     */
  tensor clamp(const_pointer __min_val = nullptr, const_pointer __max_val = nullptr) const;

  /**
     * @brief Computes the cosine of each element in the tensor.
     * 
     * @return A new tensor containing the cosine of each element.
     */
  tensor cos() const;

  /**
     * @brief Computes the sine of each element in the tensor.
     * 
     * @return A new tensor containing the sine of each element.
     */
  tensor sin() const;

  /**
     * @brief Computes the  arctanjent (atan) of each element in the tensor
     * 
     * @return A new tensor containing the inverse tangent of each element 
     */
  tensor atan() const;

  /**
     * @brief Computes the hyperbolic sine (sinh) of each element in the tensor.
     * 
     * @return A new tensor containing the hyperbolic sine of each element.
     */
  tensor sinh() const;

  /**
     * @brief Computes the inverse hyperbolic sine (asinh) of each element in the tensor.
     * 
     * @return A new tensor containing the inverse hyperbolic sine of each element.
     */
  tensor asinh() const;

  /**
     * @brief Computes the inverse hyperbolic sine (asin) of each element in the tensor.
     * 
     * @return A new tensor containing the inverse sine of each element.
     */
  tensor asin() const;

  /**
     * @brief Computes the hyperbolic cosine (cosh) of each element in the tensor.
     * 
     * @return A new tensor containing the hyperbolic cosine of each element.
     */
  tensor cosh() const;

  /**
     * @brief Computes the inverse hyperbolic cosine (acosh) of each element in the tensor.
     * 
     * @return A new tensor containing the inverse hyperbolic cosine of each element.
     */
  tensor acosh() const;

  /**
     * @brief Computes the element-wise logical XOR (exclusive or) with another tensor.
     * 
     * @param __other A tensor of the same shape as this.
     * @return A new tensor where each element is the result of logical XOR between corresponding elements in this and __other.
     */
  tensor logical_xor(const tensor& __other) const;

  /**
     * @brief Computes the element-wise logical XOR (exclusive or) with a scalar value.
     * 
     * @param __val A scalar value that supports the operator '!='.
     * @return A new tensor where each element is the result of logical XOR between the tensor element and the scalar value __val.
     */
  tensor logical_xor(const value_t __val) const;

  /**
     * @brief Computes the element-wise logical AND with another tensor.
     * 
     * @param __other A tensor of the same shape as this.
     * @return A new tensor where each element is the result of logical AND between corresponding elements in this and __other.
     */
  tensor logical_and(const tensor& __other) const;

  /**
     * @brief Computes the element-wise logical AND with a scalar value.
     * 
     * @param __val A scalar value that supports the operator 'and'.
     * @return A new tensor where each element is the result of logical AND between the tensor element and the scalar value __val.
     */
  tensor logical_and(const value_t __val) const;

  /**
     * @brief Performs a bitwise NOT operation on each element of the tensor.
     * 
     * @return A new tensor where each element is the bitwise NOT of the corresponding element in the original tensor.
     */
  tensor bitwise_not() const;

  /**
     * @brief Computes the element-wise bitwise AND with a scalar value.
     * 
     * @param __val A scalar value.
     * @return A new tensor where each element is the result of bitwise AND between the tensor element and the scalar value __val.
     */
  tensor bitwise_and(const value_t __val) const;

  /**
     * @brief Computes the element-wise bitwise AND with another tensor.
     * 
     * @param __other A tensor of the same shape as this.
     * @return A new tensor where each element is the result of bitwise AND between corresponding elements in this and __other.
     */
  tensor bitwise_and(const tensor& __other) const;

  /**
     * @brief Computes the element-wise bitwise OR with a scalar value.
     * 
     * @param __val A scalar value.
     * @return A new tensor where each element is the result of bitwise OR between the tensor element and the scalar value __val.
     */
  tensor bitwise_or(const value_t __val) const;

  /**
     * @brief Computes the element-wise bitwise OR with another tensor.
     * 
     * @param __other A tensor of the same shape as this.
     * @return A new tensor where each element is the result of bitwise OR between corresponding elements in this and __other.
     */
  tensor bitwise_or(const tensor& __other) const;

  /**
     * @brief Computes the element-wise bitwise XOR (exclusive or) with a scalar value.
     * 
     * @param __val A scalar value.
     * @return A new tensor where each element is the result of bitwise XOR between the tensor element and the scalar value __val.
     */
  tensor bitwise_xor(const value_t __val) const;

  /**
     * @brief Computes the element-wise bitwise XOR (exclusive or) with another tensor.
     * 
     * @param __other A tensor of the same shape as this.
     * @return A new tensor where each element is the result of bitwise XOR between corresponding elements in this and __other.
     */
  tensor bitwise_xor(const tensor& __other) const;

  /**
     * @brief Performs a bitwise left shift on each element of the tensor by a specified amount.
     * 
     * @param __amount The number of positions to shift each element to the left.
     * @return A new tensor where each element is left-shifted by __amount positions.
     */
  tensor bitwise_left_shift(const int __amount) const;

  /**
     * @brief Performs a bitwise right shift on each element of the tensor by a specified amount.
     * 
     * @param __amount The number of positions to shift each element to the right.
     * @return A new tensor where each element is right-shifted by __amount positions.
     */
  tensor bitwise_right_shift(const int __amount) const;

  /**
     * @brief Performs matrix multiplication between this tensor and another tensor.
     * 
     * @param __other The other tensor to perform matrix multiplication with.
     * @return A new tensor resulting from matrix multiplication of this and __other.
     * 
     * @throws std::invalid_argument If the dimensions are not compatible for matrix multiplication.
     */
  tensor matmul(const tensor& __other) const;

  /**
     * @brief Reshapes the tensor to a new shape.
     * 
     * @param __shape The new shape for the tensor.
     * @return A new tensor with the specified shape.
     * 
     * @throws std::invalid_argument If the new shape is incompatible with the number of elements in the tensor.
     */
  tensor reshape(const shape_t __shape) const;

  /**
     * @brief Reshapes the tensor to have the same shape as another tensor.
     * 
     * @param __other A tensor whose shape will be used for reshaping.
     * @return A new tensor with the same shape as __other.
     */
  tensor reshape_as(const tensor& __other) { return this->reshape(__other.__shape_); }
  /**
     * @brief Computes the cross product of this tensor with another tensor.
     * 
     * @param __other A tensor of the same dimensionality as this tensor, which must be 3D.
     * @return A new tensor representing the cross product of the two tensors.
     * 
     * @throws std::invalid_argument If the dimensions of the tensors are not compatible for cross product.
     */
  tensor cross_product(const tensor& __other) const;

  /**
     * @brief Computes the absolute values of the elements in the given tensor.
     * 
     * @param __tensor The tensor for which to compute the absolute values.
     * @return A new tensor containing the absolute values of the elements in __tensor.
     */
  tensor absolute(const tensor& __tensor) const;

  /**
     * @brief Computes the dot product of this tensor with another tensor.
     * 
     * @param __other A tensor with compatible dimensions for the dot product.
     * @return A new tensor representing the result of the dot product.
     * 
     * @throws std::invalid_argument If the dimensions of the tensors are not compatible for dot product.
     */
  tensor dot(const tensor& __other) const;

  /**
     * @brief Applies the ReLU (Rectified Linear Unit) activation function to each element of the tensor.
     * 
     * @return A new tensor where each element is the result of the ReLU function applied to the corresponding element in this tensor.
     */
  tensor relu() const;

  /**
     * @brief Computes the transpose of the tensor.
     * 
     * @return A new tensor that is the transpose of this tensor.
     * 
     * @throws std::invalid_argument If the tensor cannot be transposed due to incompatible dimensions.
     */
  tensor transpose() const;

  /**
     * @brief Raises each element of this tensor to the power of the corresponding element in another tensor.
     * 
     * @param __other A tensor with the same shape as this tensor.
     * @return A new tensor containing the result of raising each element in this tensor to the power of the corresponding element in __other.
     * 
     * @throws std::invalid_argument If the shapes of the tensors are not the same.
     */
  tensor pow(const tensor& __other) const;

  /**
     * @brief Raises each element of this tensor to the power of a specified scalar value.
     * 
     * @param __val A scalar value to which each element of the tensor will be raised.
     * @return A new tensor containing the result of raising each element in this tensor to the power of __val.
     */
  tensor pow(const value_t __val) const;

  /**
     * @brief Computes the cumulative product of the elements along a specified dimension.
     * 
     * @param __dim The dimension along which to compute the cumulative product. Defaults to -1 (last dimension).
     * @return A new tensor containing the cumulative product along the specified dimension.
     */
  tensor cumprod(index_t __dim = -1) const;

  /**
     * @brief Concatenates a vector of tensors along a specified dimension.
     * 
     * @param _others A vector of tensors to concatenate. All tensors must have the same shape except for the specified dimension.
     * @param _dim The dimension along which to concatenate the tensors.
     * @return A new tensor formed by concatenating the input tensors along the specified dimension.
     * 
     * @throws std::invalid_argument If the tensors in _others have incompatible shapes for concatenation.
     */
  tensor cat(const std::vector<tensor>& _others, index_t _dim) const;

  /**
     * @brief Returns the indices of the maximum values along a specified dimension.
     * 
     * @param __dim The dimension along which to find the indices of the maximum values.
     * @return A new tensor containing the indices of the maximum values along the specified dimension.
     */
  tensor argmax(index_t __dim) const;

  /**
     * @brief Inserts a new dimension of size one at the specified index.
     * 
     * @param __dim The dimension index at which to add the new dimension.
     * @return A new tensor with the added dimension.
     */
  tensor unsqueeze(index_t __dim) const;

  /**
     * @brief computes the least common multiple for all the elements in the tensor
     * @return returns the least common multiple as a 64 bit integer
    */
  index_t lcm() const;

  /**
     * @brief in place version of sqrt()
     */
  void sqrt_();

  /**
     * @brief in place version of exp()
     */
  void exp_();

  /**
     * @brief in place version of log2()
     */
  void log2_();

  /**
     * @brief in place version of log10()
     */
  void log10_();


  /**
     * @brief in place version of log()
     */
  void log_();

  /**
     * @brief in place version of frac()
     */
  void frac_();

  /**
     * @brief in place version of fmod(tensor)
     */
  void fmod_(const tensor& __other);

  /** 
     * @brief in place version of fmod(value)
    */
  void fmod_(const value_t __val);

  /**
     * @brief in place version of cos()
     */
  void cos_();

  /**
     * @brief in place version of cosh()
     */
  void cosh_();

  /**
     * @brief in place version of atan()
     */
  void atan_();

  /**
     * @brief in place version of acosh()
     */
  void acosh_();

  /**
     * @brief in place version of sinh()
     */
  void sinh_();

  /**
     * @brief in place version of asinh()
     */
  void asinh_();

  /**
     * @brief in place version of asin()
     */
  void asin_();

  /**
     * @brief in place version of ceil()
     */
  void ceil_();

  /**
     * @brief in place version of floor()
     */
  void floor_();

  /**
     * @brief in place version of sin()
     */
  void sin_();

  /**
     * @brief in place version of relu()
     */
  void relu_();

  /**
     * @brief in place version of clamp()
     */
  void clamp_(const_pointer __min_val = nullptr, const_pointer __max_val = nullptr);

  /**
     * @brief in place version of logical_not()
     */
  void logical_not_() const { this->bitwise_not_(); }

  /**
     * @brief in place version of logical_or(tensor)
     */
  void logical_or_(const tensor& __other);

  /**
     * @brief in place version of logical_or(value)
     */
  void logical_or_(const value_t __val);

  /**
     * @brief in place version of logical_xor(tensor) 
     */
  void logical_xor_(const tensor& __other);

  /**
     * @brief in place version of 
     */
  void logical_xor_(const value_t __val);

  /**
     * @brief in place version of logical_and(tensor)
     */
  void logical_and_(const tensor& __other);

  /**
     * @brief in place version of logical_and(value)
     */
  void logical_and_(const value_t __val);

  /**
     * @brief in place version of pow(tensor)
     */
  void pow_(const tensor& __other);

  /**
     * @brief in place version of pow(value)
     */
  void pow_(const value_t __val);

  /**
     * @brief in place version of bitwise_left_shift(amount)
     */
  void bitwise_left_shift_(const int __amount);

  /**
     * @brief in place version bitwise_right_shift(amount)
     */
  void bitwise_right_shift_(const int __amount);

  /**
     * @brief in place version of bitwise_and(value)
     */
  void bitwise_and_(const value_t __val);

  /**
     * @brief in place version of bitwise_and(tensor)
     */
  void bitwise_and_(const tensor& __other);

  /**
     * @brief in place version of bitwise_or(value)
     */
  void bitwise_or_(const value_t __val);

  /**
     * @brief in place version of bitwise_or(tensor)
     */
  void bitwise_or_(const tensor& __other);

  /**
     * @brief in place version of bitwise_xor(value)
     */
  void bitwise_xor_(const value_t __val);

  /**
     * @brief in place version of bitwise_xor(tensor)
     */
  void bitwise_xor_(const tensor& __other);

  /**
     * @brief in place version of bitwise_not()
     */
  void bitwise_not_();

  /**
     * @brief changes the tensor aspects (shape, strides) to reshape the tensor internally
     * @param __new_sh a list of dimensions that will be the new shape
     */
  void view(std::initializer_list<index_t> __new_sh);

  /**
     * @brief prints a tensor to stdout
     */
  void print() const noexcept;

  /**
     * @brief sets all the element of the tensor to zero
     * @param __sh the shape of the returned tensor
     * @return a tensor of the input shape where all elements are set to zero
     */
  static tensor zeros(const shape_t& __sh);

  /**
     * @brief sets all the element of the tensor to one
     * @param __sh the shape of the returned tensor 
     * @return a tensor of the input shape where all the elements are set to one
     */
  static tensor ones(const shape_t& __sh);

  /**
     * @brief in place version of zeros(shape)
     */
  void zeros_(shape_t __sh = {});

  /**
     * @brief in place version of ones(shape)
     */
  void ones_(shape_t __sh = {});

  /**
     * @brief sets all the elements of the tensor to random values
     * @param __sh the shape of the returned tensor
     * @param __bounded wether the elements should be between zero or one or not
     * @return a tensor of the input shape where all the elements are set to random values 
     */
  tensor randomize(const shape_t& __sh, bool __bounded = false);

  /**
     * @brief in place version of randomize(shape, bounded)
     */
  void randomize_(const shape_t& __sh, bool __bounded = false);

  /**
     * @brief gets the indices of the maximum values along each dimension
     * @param __dim 64 bit integer indicating the desired dimension to be evaluated
     * @return a tensor of 64 bit integers
     */
  tensor<index_t> argmax_(index_t __dim) const;

  /**
     * @brief sorts the elements along a given dimension
     * @param __dim the dimension along which to sort the elements
     * @param __ascending true to sort in ascending order 
     * @return a tensor of 64 bit integer (indicies of the sorted elements in the sort order)
     */
  tensor<index_t> argsort(index_t __dim = -1, bool __ascending = true) const;

  /**
   * @brief creates a hash of the tensor
   * 
   * @return 64 bit integer 
  */
  index_t hash() const;

 private:
  static void __check_is_scalar_type(const std::string __msg) {
    assert(!__msg.empty());
    if (!std::is_scalar<value_t>::value)
      throw std::runtime_error(__msg);
  }

  static void __check_is_integral_type(const std::string __msg) {
    assert(!__msg.empty());
    if (!std::is_integral<value_t>::value)
      throw std::runtime_error(__msg);
  }

  template<typename __t>
  static void __check_is_same_type(const std::string __msg) {
    assert(!__msg.empty());
    if (!std::is_same<value_t, __t>::value)
      throw std::runtime_error(__msg);
  }

  static void __check_is_arithmetic_type(const std::string __msg) {
    assert(!__msg.empty());
    if (!std::is_arithmetic<value_t>::value)
      throw std::runtime_error(__msg);
  }

  void __compute_strides() {
    if (this->__shape_.empty())
    {
      std::cerr << "Shape must be initialized before computing strides" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    this->__strides_ = shape_t(this->__shape_.size(), 1);
    index_t __st = 1, __i = this->__shape_.size() - 1;
    for (; __i >= 0; __i--)
    {
      this->__strides_[__i] = __st;
      __st *= this->__shape_[__i];
    }
  }

  index_t __compute_index(const std::vector<index_t>& __idx) const {
    if (__idx.size() != this->__shape_.size())
      throw std::out_of_range("input indices does not match the tensor __shape_");

    index_t __index = 0, __i = 0;
    for (; __i < this->__shape_.size(); __i++)
      __index += __idx[__i] * this->__strides_[__i];

    return __index;
  }

  static uint64_t __computeSize(const shape_t& __dims) {
    uint64_t __ret = 1;
    for (const index_t& __d : __dims)
      __ret *= __d;

    return __ret;
  }

  uint64_t __compute_outer_size(const index_t __dim) const {
    // just a placeholder for now
    return 0;
  }

  float __frac(const_reference __scalar) { return std::fmod(static_cast<float>(__scalar), 1.0f); }

  // where the tensor is stored
  bool __is_cuda_device() const { return (this->__device_ == Device::CUDA); }

};  // tensor class

template<class _Tp>
tensor<_Tp> tensor<_Tp>::zeros(const shape_t& __sh) {
  __check_is_scalar_type("template type must be a scalar : tensor.zeros()");
  index_t                __i = 0;
  __container__<value_t> __d;
#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_t>::value)
  {
    constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    float32x4_t       __zero_vec = vdupq_n_f32(0.0f);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
      vst1q_f32(reinterpret_cast<float*>(&__d[__i]), __zero_vec);
  }
  else
  {
#endif
    for (; __i < this->__data_.size(); __i++)
      __d[__i] = value_t(0.0);
#if defined(__ARM_NEON)
  }
#endif
  return __self(__d, __sh);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::ones(const shape_t& __sh) {
  __check_is_scalar_type("template type must be a scalar : tensor.ones()");
  __container__<value_t> __d;
  index_t                __i = 0;
#if defined(__ARM_NEON)
  if constexpr (std::is_floating_point<value_t>::value)
  {
    constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    float32x4_t       __one_vec  = vdupq_n_f32(1.0f);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
      vst1q_f32(reinterpret_cast<float32_t*>(&__d[__i]), __one_vec);
  }
  else
  {
#endif
    for (; __i < this->__data_.size(); __i++)
      __d[__i] = value_t(1.0);
#if defined(__ARM_NEON)
  }
#endif
  return __self(__d, __sh);
}

template<class _Tp>
void tensor<_Tp>::zeros_(shape_t __sh) {
  if (__sh.empty())
    __sh = this->__shape_;

  this->__data_ = __container__<value_t>(std::accumulate(__sh.begin(), __sh.end(), 1, std::multiplies<index_t>()), value_t(0));
}

template<class _Tp>
void tensor<_Tp>::ones_(shape_t __sh) {
  if (__sh.empty())
    __sh = this->__shape_;

  this->__data_ = __container__<value_t>(std::accumulate(__sh.begin(), __sh.end(), 1, std::multiplies<index_t>()), value_t(0));
}

template<class _Tp>
typename tensor<_Tp>::index_t tensor<_Tp>::size(const index_t __dim) const {
  if (__dim < 0 || __dim >= static_cast<index_t>(this->__shape_.size()))
    throw std::invalid_argument("dimension input is out of range");

  if (this->__data_.empty())
    return 0;

  if (__dim == 0)
    return this->__data_.size();

  return this->__shape_[__dim];
}

template<class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::at(tensor<_Tp>::shape_t __idx) {
  if (__idx.empty())
    throw std::invalid_argument("Passing an empty vector as indices for a tensor");

  index_t __i = this->__compute_index(__idx);
  if (__i < 0 || __i >= this->__data_.size())
    throw std::invalid_argument("input indices are out of bounds");

  return this->__data_[__i];
}

template<class _Tp>
typename tensor<_Tp>::const_reference tensor<_Tp>::at(const tensor<_Tp>::shape_t __idx) const {
  if (__idx.empty())
    throw std::invalid_argument("Passing an empty vector as indices for a tensor");

  index_t __i = this->__compute_index(__idx);
  if (__i < 0 || __i >= this->__data_.size())
    throw std::invalid_argument("input indices are out of bounds");

  return this->__data_[__i];
}

template<class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::operator[](const index_t __in) {
  if (__in >= this->__data_.size() || __in < 0)
    throw std::out_of_range("Access index is out of range");

  return this->__data_[__in];
}

template<class _Tp>
tensor<_Tp>::const_reference tensor<_Tp>::operator[](const index_t __in) const {
  if (__in >= this->__data_.size() || __in < 0)
    throw std::out_of_range("Access index is out of range");

  return this->__data_[__in];
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  if (__other.shape() != this->__shape_)
    throw std::invalid_argument("Cannot add two tensors with different shapes");

  __container__<value_t> __d(this->__data_.size());
  index_t                __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __vec1   = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __vec2   = vld1q_f32(reinterpret_cast<const float32_t*>(&__other[__i]));
    float32x4_t __result = vaddq_f32(__vec1, __vec2);
    vst1q_f32(reinterpret_cast<float*>(&__d[__i]), __result);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    __d[__i] = this->__data_[__i] + __other[__i];
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), __d.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_t>(__v + __w); });
    */
  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const value_t __scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  __container__<value_t> __d(this->__data_.size());
  index_t                __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  float32x4_t       __val_vec  = vdupq_n_f32(reinterpret_cast<float32_t>(&__scalar));
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __vec = vld1q_f32(reinterpret_castr<const float32_t*>(&this->__data_[__i]));
    float32x4_t __res = vaddq_f32(__vec, __val_vec);
    vst1q_f32(&__d[__i], __res);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    __d[__i] = this->__data_[__i] + __scalar;
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__scalar](const_reference __v) { return static_cast<value_t>(__v + __scalar); });
    */
  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+=(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  assert(this->__shape_ == __other.shape());
  index_t __i = 0;
#if defined(__ARM_NEON)
#endif
  for (__i; __i < this->__data_.size(); __i++)
    this->__data_[__i] += __other[__i];
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_t>(__v + __w); });
    */
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+=(const_reference __scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  float32x4_t       __val_vec  = vdupq_n_f32(reinterpret_cast<float32_t>(&__scalar));
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __add_vec  = vaddq_f32(__data_vec, __val_vec);
    vst1q_f32(&this->__data_[__i], __add_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = this->__data_[__i] + __scalar;
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__scalar](const_reference __v) { return static_cast<value_t>(__v + __scalar); });
    */
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  if (__other.shape() != this->__shape_)
    throw std::invalid_argument("Cannot add two tensors with different shapes");

  __container__<value_t> __d(this->__data_.size());
  index_t                __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __oth = vld1q_f32(reinterpret_cast<const float32_t*>(&__other[__i]));
    float32x4_t __sub = vsubq_f32(__vec, __oth);
    vst1q_f32(&__d[__i], __sub);
  }
#endif
  for (; __i < this->__data_[__i]; __i++)
    __d[__i] = this->__data_[__i] - __other[__i];
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), __d.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_t>(__v - __w); });
    */
  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const value_t __scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  __container__<value_t> __d(this->__data_.size());
  index_t                __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  float32x4_t       __vals     = vdupq_n_f32(reinterpret_cast<float32_t>(&__scalar));
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __sub = vsubq_f32(__vec, __vals);
    vst1q_f32(&__d[__i], __sub);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    __d[__i] = this->__data_[__i] - __scalar;
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__scalar](const_reference __v) { return static_cast<value_t>(__v - __scalar); });
    */
  return __self(*this);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-=(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  assert(this->__shape_ == __other.shape());
  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __oth = vld1q_f32(reinterpret_cast<const float32_t*>(&__other[__i]));
    float32x4_t __sub = vsubq_f32(__vec, __oth);
    vst1q_f32(&this->__data_[__i], __sub);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] -= __other[__i];
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_t>(__v - __w); });
    */
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator*=(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  assert(this->__shape_ == __other.shape());
  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __oth = vld1q_f32(reinterpret_cast<const float32_t*>(&__other[__i]));
    float32x4_t __mul = vmulq_f16(__vec, __oth);
    vst1q_f32(&this->__data_[__i], __mul);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] *= __other[__i];
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_t>(__v * __w); });
    */
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator*=(const_reference __scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  index_t __i = 0;
#if defined(__ARM_NEON)
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] *= __scalar;
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_t>(__v * __scalar); });
    */
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator/=(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  assert(this->__shape_ == __other.shape());
  index_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] /= __other[__i];
  /*    
    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_t>(__v / __w); });
    */
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator/=(const_reference __scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  index_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] /= __scalar;
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__scalar](const_reference __v) { return static_cast<value_t>(__v / __scalar); });
    */
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-=(const_reference __scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  float32x4_t       __val      = vld1q_f32(reinterpret_cast<const float32_t*>(&__val));
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __sub = vsubq_f32(__vec, __val);
    vst1q_f32(&this->__data_[__i], __sub);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] -= __scalar;
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_t>(__v - __w); });
    */
  return *this;
}

template<class _Tp>
void tensor<_Tp>::log2_() {
  this->__check_is_integral_type("Given data type must be an integral");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("log2 : operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = std::log2(__vals[0]);
    __vals[1]              = std::log2(__vals[1]);
    __vals[2]              = std::log2(__vals[2]);
    __vals[3]              = std::log2(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::log2(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_t>(std::log2(static_cast<float>(this->__data_[__i])));
    });
    */
}

template<class _Tp>
void tensor<_Tp>::asinh_() {
  this->__check_is_scalar_type("Cannot perform asinh on non-scalar data type");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("asinh : operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = std::asinh(__vals[0]);
    __vals[1]              = std::asinh(__vals[1]);
    __vals[2]              = std::asinh(__vals[2]);
    __vals[3]              = std::asinh(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::asinh(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_t>(std::asinh(static_cast<double>(__v))) });
    */
}

template<class _Tp>
void tensor<_Tp>::atan_() {
  this->__check_is_integral_type("template class must be integral type");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("Arctangent operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = std::atan(__vals[0]);
    __vals[1]              = std::atan(__vals[1]);
    __vals[2]              = std::atan(__vals[2]);
    __vals[3]              = std::atan(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::atan(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_t>(std::atan(static_cast<double>(__v))); });
    */
}

template<class _Tp>
void tensor<_Tp>::floor_() {
  this->__check_is_scalar_type("Cannot get the floor of a non scalar value");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("Floor operation only supported for floating-point types.");
  for (; __i < this->__data_.size(); __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec  = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __floor_vec = vrndmq_f32(__data_vec);
    vst1q_f32(&this->__data_[__i], __floor_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::floor(static_cast<double>(this->__data_[__i])));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_t>(std::floor(static_cast<double>(__v))); });
    */
}

template<class _Tp>
void tensor<_Tp>::ceil_() {
  this->__check_is_scalar_type("Cannot get the ceiling of a non scalar value");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("Ceiling operation only supported for floating-point types.");
  for (; __i + _ARM64_REG_WIDTH <= this->__data_.size(); __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __ceil_vec = vrndpq_f32(__data_vec);
    vst1q_f32(&this->__data_[__i], __ceil_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::ceil(static_cast<double>(this->__data_[__i])));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_t>(std::ceil(static_cast<double>(__v))) });
    */
}

template<class _Tp>
void tensor<_Tp>::sin_() {
  this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32_t   __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]             = std::sin(__vals[0]);
    __vals[1]             = std::sin(__vals[1]);
    __vals[2]             = std::sin(__vals[2]);
    __vals[3]             = std::sin(__vals[3]);
    float32x4_t __sin_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __sin_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::sin(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_t>(std::sin(static_cast<double>(__v))) });
    */
}

template<class _Tp>
void tensor<_Tp>::asin_() {
  this->__check_is_scalar_type("Cannot perform asin on non-scalar data type");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("asin : operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = std::asin(__vals[0]);
    __vals[1]              = std::asin(__vals[1]);
    __vals[2]              = std::asin(__vals[2]);
    __vals[3]              = std::asin(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::asin(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_t>(std::asin(static_cast<double>(__v))); });
    */
}

template<class _Tp>
void tensor<_Tp>::log10_() {
  this->__check_is_integral_type("Given data type must be an integral");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("log10 : operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = std::log10(__vals[0]);
    __vals[1]              = std::log10(__vals[1]);
    __vals[2]              = std::log10(__vals[2]);
    __vals[3]              = std::log10(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }

#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::log10(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_t>(std::log10(static_cast<double>(this->__data_[__i])));
    });
    */
}

template<class _Tp>
void tensor<_Tp>::log_() {
  this->__check_is_integral_type("Given data type must be an integral");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("log : operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = std::log(__vals[0]);
    __vals[1]              = std::log(__vals[1]);
    __vals[2]              = std::log(__vals[2]);
    __vals[3]              = std::log(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::log(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_t>(std::log(static_cast<double>(this->__data_[__i])));
    });
    */
}

template<class _Tp>
void tensor<_Tp>::exp_() {
  this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("exp operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = std::exp(__vals[0]);
    __vals[1]              = std::exp(__vals[1]);
    __vals[2]              = std::exp(__vals[2]);
    __vals[3]              = std::exp(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::exp(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_t>(std::exp(static_cast<double>(this->__data_[__i])));
    });
    */
}

template<class _Tp>
void tensor<_Tp>::sqrt_() {
  this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("sqrt : operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = std::sqrt(__vals[0]);
    __vals[1]              = std::sqrt(__vals[1]);
    __vals[2]              = std::sqrt(__vals[2]);
    __vals[3]              = std::sqrt(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::sqrt(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_t>(std::sqrt(static_cast<double>(this->__data_[__i])));
    });
    */
}

template<class _Tp>
void tensor<_Tp>::sinh_() {
  this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("sinh : operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = std::sinh(__vals[0]);
    __vals[1]              = std::sinh(__vals[1]);
    __vals[2]              = std::sinh(__vals[2]);
    __vals[3]              = std::sinh(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::sinh(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_t>(std::sinh(static_cast<double>(__v))) });
    */
}

template<class _Tp>
void tensor<_Tp>::acosh_() {
  this->__check_is_scalar_type("Cannot perform a acosh on non-scalar data type");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("acosh : operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = std::acosh(__vals[0]);
    __vals[1]              = std::acosh(__vals[1]);
    __vals[2]              = std::acosh(__vals[2]);
    __vals[3]              = std::acosh(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::acosh(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_t>(std::acosh(static_cast<double>(this->__data_[__i])));
    });
    */
}

template<class _Tp>
void tensor<_Tp>::cosh_() {
  this->__check_is_scalar_type("Cannot perform a cosh on non-scalar data type");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("cosh : operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = std::cosh(__vals[0]);
    __vals[1]              = std::cosh(__vals[1]);
    __vals[2]              = std::cosh(__vals[2]);
    __vals[3]              = std::cosh(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::cosh(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_t>(std::cosh(static_cast<double>(this->__data_[__i])));
    });
    */
}

template<class _Tp>
void tensor<_Tp>::cos_() {
  this->__check_is_scalar_type("Cannot perform a cosine on non-scalar data type");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("cos : operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = std::cos(__vals[0]);
    __vals[1]              = std::cos(__vals[1]);
    __vals[2]              = std::cos(__vals[2]);
    __vals[3]              = std::cos(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::cos(this->__data_[__i]));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_t>(std::cos(static_cast<double>(this->__data_[__i])));
    });
    */
}

template<class _Tp>
void tensor<_Tp>::frac_() {
  this->__check_is_scalar_type("Cannot get the fraction of a non-scalar type");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("frac : operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0]              = this->__frac(__vals[0]);
    __vals[1]              = this->__frac(__vals[1]);
    __vals[2]              = this->__frac(__vals[2]);
    __vals[3]              = this->__frac(__vals[3]);
    float32x4_t __atan_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __atan_vec);
  }

#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(this->__frac(this->__data_[__i]));
}

template<class _Tp>
void tensor<_Tp>::pow_(const value_t __val) {
  this->__check_is_integral_type("cannot get the power of a non-integral value");
  index_t __i = 0;
#if defined(__ARM_NEON)
  this->__check_is_same_type<float32_t>("pow: operation only supported for floating-point types.");
  index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float       __vals[_ARM64_REG_WIDTH];
    vst1q_f32(__vals, __data_vec);
    __vals[0] = std::pow(__vals[0], __val);
    __vals[1] = std::pow(__vals[1], __val);
    __vals[2] = std::pow(__vals[2], __val);
    __vals[3] = std::pow(__vals[3], __val);

    float32x4_t __pow_vec = vld1q_f32(__vals);
    vst1q_f32(&this->__data_[__i], __pow_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::pow(this->__data_[__i], __val));
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [&__val](const_reference __v) {
        return static_cast<value_t>(std::pow(static_cast<double>(__v), static_cast<double>(__val)));
    });
    */
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::clone() const {
  __container__<value_t> __d = this->__data_;
  shape_t                __s = this->__shape_;
  return __self(__d, __s);
}

template<class _Tp>
void tensor<_Tp>::bitwise_right_shift_(const int __amount) {
  this->__check_is_integral_type("Cannot perform a bitwise right shift on non-integral values");

  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_t, int32_t>::value)
  {
    const int32x4_t __shift_amount_vec = vdupq_n_s32(-__amount);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      int32x4_t __data_vec    = vld1q_s32(reinterpret_cast<int32_t*>(&this->__data_[__i]));
      int32x4_t __shifted_vec = vshlq_s32(__data_vec, __shift_amount_vec);
      vst1q_s32(&this->__data_[__i], __shifted_vec);
    }
  }
  else if constexpr (std::is_same<value_t, uint32_t>::value)
  {
    const int32x4_t __shift_amount_vec = vdupq_n_s32(-__amount);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      uint32x4_t __data_vec    = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
      uint32x4_t __shifted_vec = vshlq_u32(__data_vec, __shift_amount_vec);
      vst1q_u32(&this->__data_[__i], __shifted_vec);
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] >>= __amount;
}

template<class _Tp>
void tensor<_Tp>::bitwise_left_shift_(const int __amount) {
  this->__check_is_integral_type("Cannot perform a bitwise left shift on non-integral values");

  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_t, int32_t>::value)
  {
    const int32x4_t __shift_amount_vec = vdupq_n_s32(__amount);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      int32x4_t __data_vec    = vld1q_s32(reinterpret_cast<const int32_t*>(&this->__data_[__i]));
      int32x4_t __shifted_vec = vshlq_s32(__data_vec, __shift_amount_vec);
      vst1q_s32(&this->__data_[__i], __shifted_vec);
    }
  }
  else if constexpr (std::is_same<value_t, uint32_t>::value)
  {
    const int32x4_t __shift_amount_vec = vdupq_n_s32(__amount);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      uint32x4_t __data_vec    = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
      uint32x4_t __shifted_vec = vshlq_u32(__data_vec, __shift_amount_vec);
      vst1q_u32(&this->__data_[__i], __shifted_vec);
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] <<= __amount;
}

template<class _Tp>
bool tensor<_Tp>::operator==(const tensor& __other) const {
  if ((this->__shape_ != __other.shape()) && (this->__strides_ != __other.strides()))
    return false;

  return this->__data_ == __other.storage();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sinh() const {
  return this->clone().sinh_();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::asin() const {
  return this->clone().asin_();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sin() const {
  return this->clone().sin_();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::frac() const {
  return this->clone().frac_();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cos() const {
  return this->clone().cos_();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log() const {
  return this->clone().log_();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::asinh() const {
  return this->clone().asinh_();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cosh() const {
  return this->clone().cosh_();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::atan() const {
  return this->clone().atan_();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sqrt() const {
  return this->sqrt_();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::acosh() const {
  return this->clone().acosh();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log10() const {
  return this->clone().log10_();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log2() const {
  return this->clone().log2_();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::exp() const {
  return this->clone().exp_();
}

// used as a helper function
int64_t __lcm(const int64_t __a, const int64_t __b) { return (__a * __b) / std::gcd(__a, __b); }

template<class _Tp>
typename tensor<_Tp>::index_t tensor<_Tp>::lcm() const {
  this->__check_is_scalar_type("Given template type must be an int");
  index_t __ret = static_cast<index_t>(this->__data_[0]);
  index_t __i   = 1;
  for (; __i < this->__data_.size(); __i++)
    __ret = __lcm(static_cast<index_t>(this->__data_[__i]), __ret);

  return __ret;
}

template<class _Tp>
void tensor<_Tp>::logical_or_(const value_t __val) {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot perform logical OR on non-integral and non-boolean values");

  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_t, int32_t>::value || std::is_same<value_t, bool>::value)
  {
    int32x4_t __val_vec = vdupq_n_s32(static_cast<int32_t>(__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      int32x4_t __data_vec = vld1q_s32(reinterpret_cast<const int32_t*>(&this->__data_[__i]));
      int32x4_t __or       = vorrq_s32(__data_vec, __val_vec);
      vst1q_s32(&this->__data_[__i], __or);
    }
  }
  else if constexpr (std::is_same<value_t, uint32_t>::value)
  {
    uint32x4_t __val_vec = vdupq_n_u32(static_cast<uint32_t>(__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      uint32x4_t __data_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
      uint32x4_t __or       = vorrq_u32(__data_vec, __val_vec);
      vst1q_u32(&this->__data_[__i], __or);
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(this->__data_[__i] || __val);
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__val](const_reference __v) { return static_cast<value_t>(__v || __val); });
    */
}

template<class _Tp>
void tensor<_Tp>::logical_xor_(const value_t __val) {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_t, int32_t>::value || std::is_same<value_t, bool>::value)
  {
    int32x4_t __val_vec = vdupq_n_s32(static_cast<int32_t>(__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      int32x4_t __data_vec = vld1q_s32(reinterpret_cast<const int32_t*>(&this->__data_[__i]));
      int32x4_t __xor      = veorq_s32(__data_vec, __val_vec);
      vst1q_s32(&this->__data_[__i], __xor);
    }
  }
  else if constexpr (std::is_same<value_t, uint32_t>::value)
  {
    uint32x4_t __val_vec = vdupq_n_u32(static_cast<uint32_t>(__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      uint32x4_t __data_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
      uint32x4_t __xor      = veorq_u32(__data_vec, __val_vec);
      vst1q_u32(&this->__data_[__i], __xor);
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(this->__data_[__i] ^ __val);
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__val](const_reference __v) { return static_cast<value_t>(__v ^ __val); });
    */
}

template<class _Tp>
void tensor<_Tp>::logical_and_(const value_t __val) {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot get the element wise and of non-integral and non-boolean value");
  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_t, int32_t>::value)
  {
    int32x4_t __vals = vdupq_n_s32(reinterpret_cast<int32_t>(&__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      int32x4_t __vec = vld1q_s32(reinterpret_cast<const int32_t*>(&this->__data_[__i]));
      int32x4_t __and = vandq_s32(__vec, __vals);
      vst1q_s32(&this->__data_[__i], __and);
    }
  }
  else if constexpr (std::is_same<value_t, uint32_t>::value)
  {
    uint32x4_t __vals = vdupq_n_u32(reinterpret_cast<uint32_t>(&__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      uint32x4_t __vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
      uint32x4_t __and = vandq_u32(__vec, __vals);
      vst1q_u32(&this->__data_[__i], __and);
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(this->__data_[__i] && __val);
  /*
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__val](const_reference __v) { return static_cast<value_t>(__v && __val); });
    */
}

template<class _Tp>
void tensor<_Tp>::bitwise_or_(const value_t __val) {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");

  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_t, int32_t>::value)
  {
    int32x4_t __val_vec = vdupq_n_s32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      int32x4_t __data_vec   = vld1q_s32(reinterpret_cast<int32_t*>(&this->__data_[__i]));
      int32x4_t __result_vec = vorrq_s32(__data_vec, __val_vec);
      vst1q_s32(reinterpret_cast<int32_t*>(&this->__data_[__i]), __result_vec);
    }
  }
  else if constexpr (std::is_same<value_t, uint32_t>::value)
  {
    uint32x4_t __val_vec = vdupq_n_u32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      uint32x4_t __data_vec   = vld1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]));
      uint32x4_t __result_vec = vorrq_u32(__data_vec, __val_vec);
      vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __result_vec);
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] |= __val;
}

template<class _Tp>
void tensor<_Tp>::bitwise_xor_(const value_t __val) {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");

  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_t, int32_t>::value)
  {
    int32x4_t __val_vec = vdupq_n_s32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      int32x4_t __data_vec   = vld1q_s32(reinterpret_cast<int32_t*>(&this->__data_[__i]));
      int32x4_t __result_vec = veorq_s32(__data_vec, __val_vec);
      vst1q_s32(reinterpret_cast<int32_t*>(&this->__data_[__i]), __result_vec);
    }
  }
  else
  {
    uint32x4_t __val_vec = vdupq_n_u32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      uint32x4_t __data_vec   = vld1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]));
      uint32x4_t __result_vec = veorq_u32(__data_vec, __val_vec);
      vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __result_vec);
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] ^= __val;
}

template<class _Tp>
void tensor<_Tp>::bitwise_not_() {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise not on non integral or boolean value");

  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_t, int32_t>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      int32x4_t __data_vec   = vld1q_s32(reinterpret_cast<const int32_t*>(&this->__data_[__i]));
      int32x4_t __result_vec = vmvnq_s32(__data_vec);
      vst1q_s32(reinterpret_cast<int32_t*>(&this->__data_[__i]), __result_vec);
    }
  }
  else if constexpr (std::is_same<value_t, uint32_t>::value)
  {
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      uint32x4_t __data_vec   = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
      uint32x4_t __result_vec = vmvnq_u32(__data_vec);
      vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __result_vec);
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = ~this->__data_[__i];
}

template<class _Tp>
void tensor<_Tp>::bitwise_and_(const value_t __val) {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");

  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  if constexpr (std::is_same<value_t, int32_t>::value)
  {
    int32x4_t __val_vec = vdupq_n_s32(reinterpret_cast<int32_t>(&__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      int32x4_t __data_vec   = vld1q_s32(reinterpret_cast<const int32_t*>(&this->__data_[__i]));
      int32x4_t __result_vec = vandq_s32(__data_vec, __val_vec);
      vst1q_s32(reinterpret_cast<int32_t*>(&this->__data_[__i]), __result_vec);
    }
  }
  else if constexpr (std::is_same<value_t, uint32_t>::value)
  {
    uint32x4_t __val_vec = vdupq_n_u32(reinterpret_cast<uint32_t>(&__val));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      uint32x4_t __data_vec   = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
      uint32x4_t __result_vec = vandq_u32(__data_vec, __val_vec);
      vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __result_vec);
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] &= __val;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator=(const tensor& __other) {
  if (this != &__other)
  {
    this->__data_    = __other.storage();
    this->__shape_   = __other.shape();
    this->__strides_ = __other.strides();
  }
  return *this;
}

template<class _Tp>
tensor<_Tp>& tensor<_Tp>::operator=(tensor&& __other) noexcept {
  if (this != &__other)
  {
    this->__data_    = std::move(__other.storage());
    this->__shape_   = std::move(__other.shape());
    this->__strides_ = std::move(__other.strides());
  }
  return *this;
}

template<class _Tp>
void tensor<_Tp>::print() const noexcept {
  std::cout << "Shape: [";
  for (index_t i = 0; i < this->__shape_.size(); ++i)
  {
    std::cout << this->__shape_[i];
    if (i < this->__shape_.size() - 1)
      std::cout << ", ";
  }
  std::cout << "]\nData: ";
  if (!this->__data_.empty())
    for (const auto& __v : this->__data_)
      std::cout << __v << " ";
  else
    std::cout << "Empty";

  std::cout << std::endl;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::randomize(const shape_t& __sh, bool __bounded) {
  return this->clone().randomize_(__sh, __bounded);
}

template<class _Tp>
void tensor<_Tp>::randomize_(const shape_t& __sh, bool __bounded) {
  this->__check_is_scalar_type("template class must be a scalar type");
  index_t __s;

  if (this->__shape_ != __sh)
    __s = this->__computeSize(__sh);
  else
    __s = this->__data_.size();

  if (__s != this->__data_.size())
    this->__data_.resize(__s);

  std::random_device                    __rd;
  std::mt19937                          __gen(__rd());
  std::uniform_real_distribution<float> __bounded_dist(0.0f, static_cast<float>(RAND_MAX));
  std::uniform_real_distribution<float> __unbounded_dist(0.0f, 1.0f);
  index_t                               __i = 0;
#if defined(__AVX__)
  const __m256 __scale = _mm256_set1_ps(__bounded ? static_cast<float>(RAND_MAX) : 1.0f);
  for (; __i + _AVX_REG_WIDTH <= static_cast<index_t>(__s); __i += _AVX_REG_WIDTH)
  {
    __m256 __random_values = _mm256_setr_ps(__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen),
                                            __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen));
    if (!__bounded)
      __random_values = _mm256_div_ps(__random_values, __scale);

    _mm256_storeu_ps(&this->__data_[__i], __random_values);
  }
#elif defined(__SSE__)
  const __m128 __scale = _mm_set1_ps(__bounded ? static_cast<float>(RAND_MAX) : 1.0f);
  for (; __i + 4 <= static_cast<index_t>(__s); __i += 4)
  {
    __m128 __random_values = _mm_setr_ps(__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen));
    if (!__bounded)
      __random_values = _mm_div_ps(__random_values, __scale);

    _mm_storeu_ps(&this->__data_[__i], __random_values);
  }
#elif defined(__ARM_NEON)
  const float32x4_t __scale = vdupq_n_f32(__bounded ? static_cast<float>(RAND_MAX) : 1.0f);
  for (; __i + _ARM64_REG_WIDTH <= static_cast<index_t>(__s); __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __random_values = {__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen)};
    if (!__bounded)
      __random_values = vmulq_f32(__random_values, vrecpeq_f32(__scale));

    vst1q_f32(&this->__data_[__i], __random_values);
  }
#endif
  for (; __i < static_cast<index_t>(__s); __i++)
    this->__data_[__i] = value_t(__bounded ? __bounded_dist(__gen) : __unbounded_dist(__gen));
}

template<class _Tp>
tensor<typename tensor<_Tp>::index_t> tensor<_Tp>::argmax_(index_t __dim) const {
  if (__dim < 0 || __dim >= this->__shape_.size())
    throw std::out_of_range("Dimension out of range in argmax");

  tensor<index_t> __ret;
  shape_t         __ret_sh = this->__shape_;
  __ret_sh.erase(__ret_sh.begin() + __dim);
  __ret.__shape_ = __ret_sh;
  __ret.__data_.resize(this->__computeSize(__ret_sh), 0);
  index_t __outer_size = 1;
  index_t __inner_size = 1;
  index_t __i          = 0;

  for (; __i < __dim; __i++)
    __outer_size *= this->__shape_[__i];

  for (__i = __dim + 1; __i < this->__shape_.size(); __i++)
    __inner_size *= this->__shape_[__i];
#if defined(__AVX2__)
  if constexpr (std::is_same_v<_Tp, float>)
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      index_t __j = 0;
      for (; __j < __inner_size; __j++)
      {
        __m256  __max_vec       = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        __m256i __index_vec     = _mm256_setzero_si256();
        __m256i __increment     = _mm256_set1_epi32(1);
        __m256i __current_index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        index_t __k             = 0;

        for (; __k + _AVX_REG_WIDTH <= this->__shape_[__dim]; __k += _AVX_REG_WIDTH)
        {
          __m256 __data_vec = _mm256_loadu_ps(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
          __m256 __mask     = _mm256_cmp_ps(__data_vec, __max_vec, _CMP_GT_OQ);
          __max_vec         = _mm256_blendv_ps(__max_vec, __data_vec, __mask);
          __index_vec       = _mm256_blendv_epi8(__index_vec, __current_index, _mm256_castps_si256(__mask));
          __current_index   = _mm256_add_epi32(__current_index, __increment);
        }

        float32_t __max_values[_AVX_REG_WIDTH];
        int32_t   __indices[_AVX_REG_WIDTH];
        _mm256_storeu_ps(__max_values, __max_vec);
        _mm256_storeu_si256((__m256i*) __indices, __index_vec);
        float32_t __max_value = __max_values[0];
        index_t   __max_index = __indices[0];

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
          float __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
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
  else
#elif defined(__ARM_NEON)
  if constexpr (std::is_same_v<_Tp, float>)
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      index_t __j = 0;
      for (; __j < __inner_size; __j++)
      {
        float32x4_t __max_vec       = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        uint32x4_t  __index_vec     = vdupq_n_u32(0);
        uint32x4_t  __increment     = vdupq_n_u32(1);
        uint32x4_t  __current_index = {0, 1, 2, 3};
        index_t     __k             = 0;

        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH)
        {
          float32x4_t __data_vec =
            vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          uint32x4_t __mask = vcgtq_f32(__data_vec, __max_vec);
          __max_vec         = vbslq_f32(__mask, __data_vec, __max_vec);
          __index_vec       = vbslq_u32(__mask, __current_index, __index_vec);
          __current_index   = vaddq_u32(__current_index, __increment);
        }

        float32_t __max_values[_ARM64_REG_WIDTH];
        uint32_t  __indices[_ARM64_REG_WIDTH];
        vst1q_f32(__max_values, __max_vec);
        vst1q_u32(__indices, __index_vec);
        float32_t __max_value = __max_values[0];
        index_t   __max_index = __indices[0];

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
          float32_t __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
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
  else
#endif
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      index_t __j = 0;
      for (; __j < __inner_size; __j++)
      {
        index_t __max_index = 0;
        value_t __max_value = this->__data_[__i * this->__shape_[__dim] * __inner_size + __j];
        index_t __k         = 1;
        for (; __k < this->__shape_[__dim]; __k++)
        {
          value_t __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
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
tensor<_Tp> tensor<_Tp>::argmax(index_t __dim) const {
  if (__dim < 0 || __dim >= this->__shape_.size())
    throw std::out_of_range("Dimension out of range in argmax");

  tensor  __ret;
  shape_t __ret_sh = this->__shape_;
  __ret_sh.erase(__ret_sh.begin() + __dim);
  __ret.__shape_ = __ret_sh;
  __ret.__data_.resize(this->__computeSize(__ret_sh), value_t(0));
  index_t __outer_size = 1;
  index_t __inner_size = 1;
  index_t __i          = 0;

  for (; __i < __dim; __i++)
    __outer_size *= this->__shape_[__i];

  for (__i = __dim + 1; __i < static_cast<index_t>(this->__shape_.size()); __i++)
    __inner_size *= this->__shape_[__i];
#if defined(__AVX2__)
  if constexpr (std::is_same_v<_Tp, float>)
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      for (index_t __j = 0; __j < __inner_size; __j++)
      {
        __m256  __max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        index_t __k       = 0;
        for (; __k + _AVX_REG_WIDTH <= this->__shape_[__dim]; __k += _AVX_REG_WIDTH)
        {
          __m256 __data_vec = _mm256_loadu_ps(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
          __max_vec         = _mm256_max_ps(__max_vec, __data_vec);
        }

        float32_t __max_value = _mm256_reduce_max_ps(__max_vec);
        for (; __k < this->__shape_[__dim]; __k++)
        {
          float32_t __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          __max_value   = std::max(__max_value, __v);
        }
        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  }
  else
#elif defined(__ARM_NEON)
  if constexpr (std::is_same_v<_Tp, float>)
  {
    for (__i = 0; __i < __outer_size; __i++)
    {
      for (index_t __j = 0; __j < __inner_size; __j++)
      {
        float32x4_t __max_vec = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        index_t     __k       = 0;
        for (; __k + _ARM64_REG_WIDTH <= this->__shape_[__dim]; __k += _ARM64_REG_WIDTH)
        {
          float32x4_t __data_vec =
            vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]));
          __max_vec = vmaxq_f32(__max_vec, __data_vec);
        }
        float32_t __max_value = vmaxvq_f32(__max_vec);
        for (; __k < this->__shape_[__dim]; __k++)
        {
          float32_t __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          __max_value   = std::max(__max_value, __v);
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
      index_t __j = 0;
      for (; __j < __inner_size; __j++)
      {
        value_t __max_value = this->__data_[__i * this->__shape_[__dim] * __inner_size + __j];
        index_t __k         = 1;
        for (; __k < this->__shape_[__dim]; __k++)
        {
          value_t __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
          if (__v > __max_value)
            __max_value = __v;
        }
        __ret.__data_[__i * __inner_size + __j] = __max_value;
      }
    }
  }
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::unsqueeze(index_t __dim) const {
  if (__dim < 0 || __dim > static_cast<index_t>(this->__shape_.size()))
    throw std::out_of_range("Dimension out of range in unsqueeze");

  shape_t __s = this->__shape_;
  __s.insert(__s.begin() + __dim, 1);
  tensor __ret;
  __ret.__shape_ = __s;
  __ret.__data_  = this->__data_;
  return __ret;
}

template<class _Tp>
void tensor<_Tp>::view(std::initializer_list<index_t> __sh) {
  index_t __s = this->__computeSize(__sh);
  if (__s != this->__data_.size())
    throw std::invalid_argument("Total elements do not match for new shape");

  this->__shape_ = __sh;
  this->__compute_strides();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cat(const std::vector<tensor<_Tp>>& __others, index_t __dim) const {
  for (const tensor& __t : __others)
  {
    index_t __i = 0;
    for (; __i < this->__shape_.size(); __i++)
    {
      if (__i != __dim && this->__shape_[__i] != __t.__shape_[__i])
        throw std::invalid_argument("Cannot concatenate tensors with different shapes along non-concatenation dimensions");
    }
  }

  shape_t __ret_sh = this->__shape_;

  for (const tensor& __t : __others)
    __ret_sh[__dim] += __t.__shape_[__dim];

  __container__<value_t> __c;
  __c.reserve(this->__data_.size());
  __c.insert(__c.end(), this->__data_.begin(), this->__data_.end());

  for (const tensor& __t : __others)
    __c.insert(__c.end(), __t.__data_.begin(), __t.__data_.end());

  return __self(__c, __ret_sh);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cumprod(index_t __dim) const {
  if (__dim == -1)
  {
    __container__<value_t> __flat = this->__data_;
    __container__<value_t> __ret(__flat.size());
    __ret[0] = __flat[0];
#if defined(__AVX2__)
    if constexpr (std::is_same_v<_Tp, float>)
    {
      index_t __i = 1;
      for (; __i + _AVX_REG_WIDTH <= __flat.size(); __i += _AVX_REG_WIDTH)
      {
        __m256 __prev   = _mm256_loadu_ps(&__ret[__i - 1]);
        __m256 __curr   = _mm256_loadu_ps(&__flat[__i]);
        __m256 __result = _mm256_mul_ps(__prev, __curr);
        _mm256_storeu_ps(&__ret[__i], __result);
      }
      for (; __i < __flat.size(); __i++)
        __ret[__i] = __ret[__i - 1] * __flat[__i];
    }
    else
    {
      index_t __i = 1;
      for (; __i < __flat.size(); __i++)
        __ret[__i] = __ret[__i - 1] * __flat[__i];
    }
#else
    index_t __i = 1;
    for (; __i < __flat.size(); __i++)
      __ret[__i] = __ret[__i - 1] * __flat[__i];
#endif
    return __self(__ret, {__flat.size()});
  }
  else
  {
    if (__dim < 0 || __dim >= static_cast<index_t>(this->__shape_.size()))
      throw std::invalid_argument("Invalid dimension provided.");

    __container__<value_t> __ret(this->__data_);
    // TODO : compute_outer_size() implementation
    index_t __outer_size = this->__compute_outer_size(__dim);
    index_t __inner_size = this->__shape_[__dim];
    index_t __st         = this->__strides_[__dim];
#if defined(__AVX2__)
    if constexpr (std::is_same_v<_Tp, float>)
    {
      for (index_t __i = 0; __i < __outer_size; __i++)
      {
        index_t __base = __i * __st;
        __ret[__base]  = __data_[__base];

        index_t __j = 1;
        for (; __j + _AVX_REG_WIDTH <= __inner_size; __j += _AVX_REG_WIDTH)
        {
          __m256 __prev   = _mm256_loadu_ps(&__ret[__base + __j - 1]);
          __m256 __curr   = _mm256_loadu_ps(&__data_[__base + __j]);
          __m256 __result = _mm256_mul_ps(__prev, __curr);
          _mm256_storeu_ps(&__ret[__base + __j], __result);
        }
        for (; __j < __inner_size; __j++)
        {
          index_t __curr = __base + __j;
          __ret[__curr]  = __ret[__base + __j - 1] * __data_[__curr];
        }
      }
    }
    else
    {
      for (index_t __i = 0; __i < __outer_size; ++__i)
      {
        index_t __base = __i * __st;
        __ret[__base]  = __data_[__base];

        for (index_t __j = 1; __j < __inner_size; __j++)
        {
          index_t __curr = __base + __j;
          __ret[__curr]  = __ret[__base + __j - 1] * __data_[__curr];
        }
      }
    }
#else
    index_t __i = 0;
    for (; __i < __outer_size; __i++)
    {
      index_t __base = __i * __st;
      __ret[__base]  = __data_[__base];
      index_t __j    = 1;
      for (; __j < __inner_size; __j++)
      {
        index_t __curr = __base + __j;
        __ret[__curr]  = __ret[__base + __j - 1] * __data_[__curr];
      }
    }
#endif
    return __self(__ret, this->__shape_);
  }
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::slice(index_t __dim, std::optional<index_t> __start, std::optional<index_t> __end, index_t __step) const {
  if (__dim < 0 || __dim >= static_cast<index_t>(this->__shape_.size()))
    throw std::out_of_range("Dimension out of range.");

  tensor  __ret;
  index_t __s       = this->__shape_[__dim];
  index_t __start_i = __start.value_or(0);
  index_t __end_i   = __end.value_or(__s);

  if (__start_i < 0)
    __start_i += __s;

  if (__end_i < 0)
    __end_i += __s;

  __start_i            = std::max(index_t(0), std::min(__start_i, __s));
  __end_i              = std::max(index_t(0), std::min(__end_i, __s));
  index_t __slice_size = (__end_i - __start_i + __step - 1) / __step;
  shape_t __ret_dims   = this->__shape_;
  __ret_dims[-__dim]   = __slice_size;
  __ret                = __self(__ret_dims);
#if defined(__CUDACC__)
  if (this->__data_.size() >= 1024)
  {
    pointer __d_input;
    pointer __d_output;
    cudaMalloc(&__d_input, this->__data_.size() * sizeof(value_t));
    cudaMalloc(&__d_output, __ret.__data_.size() * sizeof(value_t));

    cudaMemcpy(__d_input, this->__data_.data(), this->__data_.size() * sizeof(value_t), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid(((__slice_size + block.x - 1) / block.x));
    slice_kernel<<<grid, block>>>(__d_input, __d_output, __start_i, __end_i, __step, __slice_size);

    cudaMemcpy(__ret.__data_.data(), __d_output, __ret.__data_.size() * sizeof(value_t), cudaMemcpyDeviceToHost);

    cudaFree(__d_input);
    cudaFree(__d_output);
  }
  else
  {
#endif
#if defined(__ARM_NEON)
    if constexpr (std::is_same_v<_Tp, float> && __step == 1)
    {
      index_t __vector_end = __start_i + ((__end_i - __start_i) / _ARM64_REG_WIDTH) * _ARM64_REG_WIDTH;

      for (index_t __i = __start_i, __j = 0; __i < __vector_end; __i += _ARM64_REG_WIDTH, __j += _ARM64_REG_WIDTH)
      {
        float32x4_t __vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
        vst1q_f32(&(__ret.__data_[__j]), __vec);
      }

      for (index_t __i = __vector_end, __j = __vector_end - __start_i; __i < __end_i; __i++, __j++)
        __ret.__data_[__j] = this->__data_[__i];
    }
    else
    {
#endif
      index_t __i = __start_i, __j = 0;
      for (; __i < __end_i; __i += __step, __j++)
        __ret({__j}) = this->at({__i});

#if defined(__ARM_NEON)
    }
#endif

#if defined(__CUDACC__)
  }
#endif
  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmod(const tensor<_Tp>& __other) const {
  return this->clone().fmod_(__other);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmod(const value_t __val) const {
  return this->clone().fmod_(__val);
}

template<class _Tp>
void tensor<_Tp>::fmod_(const value_t __val) {
  this->__check_is_scalar_type("Cannot divide a non scalar type");
  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() - _ARM64_REG_WIDTH);
  float32x4_t       __b        = vdupq_n_f32(reinterpret_cast<float32_t>(__val));
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __a         = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __div       = vdivq_f32(__a, __b);
    float32x4_t __floor_div = vrndq_f32(__div);
    float32x4_t __mult      = vmulq_f32(__floor_div, __b);
    float32x4_t __mod       = vsubq_f32(__a, __mult);
    vst1q_f32(&this->__data_[__i], __mod);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::fmod(static_cast<double>(this->__data_[__i]), static_cast<double>(__val)));
}

template<class _Tp>
void tensor<_Tp>::fmod_(const tensor& __other) {
  this->__check_is_scalar_type("Cannot divide non scalar values");
  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");

  index_t __i = 0;
#if defined(__ARM_NEON)
  constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __a         = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __b         = vld1q_f32(reinterpret_cast<const float32_t*>(&__other[__i]));
    float32x4_t __div       = vdivq_f32(__a, __b);
    float32x4_t __floor_div = vrndq_f32(__div);
    float32x4_t __mult      = vmulq_f32(__floor_div, __b);
    float32x4_t __mod       = vsubq_f32(__a, __mult);
    vst1q_f32(&this->__data_[__i], __mod);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::fmod(static_cast<double>(this->__data_[__i]), static_cast<double>(__other[__i])));
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmax(const tensor& __other) const {
  return this->clone().fmax_(__other);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmax(const value_t __val) const {
  return this->clone().fmax_(__val);
}

template<class _Tp>
void tensor<_Tp>::fmax_(const value_t __val) {
#if defined(__ARM_NEON)
  if (std::is_floating_point<value_t>::value)
  {
    const index_t __simd_end   = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    float32x4_t   __scalar_val = vdupq_n_f32(__val);

    for (index_t __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      float32x4_t __a       = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
      float32x4_t __max_val = vmaxq_f32(__a, __scalar_val);
      vst1q_f32(&this->__data_[__i], __max_val);
    }

    for (index_t __i = __simd_end; __i < this->__data_.size(); __i++)
      this->__data_[__i] = std::fmax(this->__data_[__i], __val);
  }
  else
  {
#endif
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__val](const_reference __v) { return std::fmax(__v, __val); });
  }
}

template<class _Tp>
void tensor<_Tp>::fmax_(const tensor<value_t>& __other) {
  assert(this->__shape_ == __other.shape());
#if defined(__ARM_NEON)
  const index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  index_t       __i        = 0;
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __a       = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __b       = vld1q_f32(reinterpret_cast<const float32_t*>(&(__other[__i])));
    float32x4_t __max_val = vmaxq_f32(__a, __b);
    vst1q_f32(&this->__data_[__i], __max_val);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = std::fmax(this->__data_[__i], __other[__i]);
  /*
    std::transform(this->__data.begin(), this->__data_.end(), __other.begin(), this->__data.begin(),
                   [](const_reference __v, const_reference __w) { return std::fmax(__v, __w); });
    */
}

template<class _Tp>
typename tensor<_Tp>::index_t tensor<_Tp>::count_nonzero(index_t __dim) const {
  this->__check_is_scalar_type("Cannot compare a non-scalar value to zero");
  index_t __c = 0;

  if (__dim == -1)
  {
#pragma omp parallel
    {
      index_t __local_count = 0;
#ifdef __AVX__
      if constexpr (std::is_same_v<value_t, float>)
      {
        index_t __size = this->__data_.size();
        index_t __i    = 0;

        for (; __i + _AVX_REG_WIDTH <= __size; __i += _AVX_REG_WIDTH)
        {
          __m256 __vec          = _mm256_loadu_ps(&this->__data_[__i]);
          __m256 __nonzero_mask = _mm256_cmp_ps(__vec, _mm256_setzero_ps(), _CMP_NEQ_OQ);
          __local_count += _mm256_movemask_ps(__nonzero_mask);
        }
      }
#endif
      index_t __i = 0;
#if defined(__ARM_NEON)
      if constexpr (std::is_same_v<value_t, float>)
      {
        index_t __size = this->__data_.size();

        for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
        {
          float32x4_t __vec          = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
          uint32x4_t  __nonzero_mask = vcgtq_f32(__vec, vdupq_n_f32(0.0f));
          __local_count += vaddvq_u32(__nonzero_mask);
        }
      }
#endif
      for (index_t __j = __i; __j < this->__data_.size(); __j++)
      {
        if (this->__data_[__j] != 0)
          __local_count++;
      }

#pragma omp atomic
      __c += __local_count;
    }
  }
  else
  {
    if (__dim < 0 || __dim >= static_cast<index_t>(__shape_.size()))
      throw std::invalid_argument("Invalid dimension provided.");

    throw std::runtime_error("Dimension-specific non-zero counting is not implemented yet.");
  }
  return __c;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::matmul(const tensor& __other) const {
  if (this->__shape_.size() != 2 || __other.shape().size() != 2)
    throw std::invalid_argument("matmul is only supported for 2D tensors");

  if (this->__shape_[1] != __other.shape()[0])
    throw std::invalid_argument("Shape mismatch for matrix multiplication");

  shape_t                __ret_sh = {this->__shape_[0], __other.shape()[1]};
  __container__<value_t> __ret_d(__ret_sh[0] * __ret_sh[1], 0);
#pragma omp parallel
  const int __blockSize = 64;

  for (int __i = 0; __i < __ret_sh[0]; __i += __blockSize)
  {
    for (int __j = 0; __j < __ret_sh[1]; __j += __blockSize)
    {
      for (int __k = 0; __k < this->__shape_[1]; __k += __blockSize)
      {
        for (int __ii = __i; __ii < std::min(static_cast<index_t>(__i + __blockSize), __ret_sh[0]); __ii++)
        {
          for (int __jj = __j; __jj < std::min(static_cast<index_t>(__j + __blockSize), __ret_sh[1]); __jj++)
          {
            value_t __sum = 0;
            for (int __kk = __k; __kk < std::min(static_cast<index_t>(__k + __blockSize), this->__shape_[1]); __kk++)
            {
              __sum += this->at({__ii, __kk}) * __other.at({__kk, __jj});
            }
            __ret_d[__ii * __ret_sh[1] + __jj] += __sum;
          }
        }
      }
    }
  }
#if defined(__ARM_NEON)
  for (int __i = 0; __i < __ret_sh[0]; __i += _ARM64_REG_WIDTH)
  {
    for (int __j = 0; __j < __ret_sh[1]; __j += _ARM64_REG_WIDTH)
    {
      for (int __k = 0; __k < this->__shape_[1]; __k += _ARM64_REG_WIDTH)
      {
        for (int __ii = __i; __ii < std::min(static_cast<index_t>(__i + _ARM64_REG_WIDTH), __ret_sh[0]); __ii++)
        {
          for (int __jj = __j; __jj < std::min(static_cast<index_t>(__j + _ARM64_REG_WIDTH), __ret_sh[1]); __jj++)
          {
            float32x4_t __sum_vec = vdupq_n_f32(0);

            for (int __kk = __k; __kk < std::min(static_cast<index_t>(__k + _ARM64_REG_WIDTH), this->__shape_[1]); __kk += _ARM64_REG_WIDTH)
            {
              float32x4_t __a_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__ii * this->__shape_[1] + __kk]));
              float32x4_t __b_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&__other.__data_[__kk * __other.shape()[1] + __jj]));
              __sum_vec           = vmlaq_f32(__sum_vec, __a_vec, __b_vec);
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
#endif
#ifdef __CUDACC__
  const int __threadsPerBlock = 256;
  const int __blocksPerGrid   = (__ret_sh[0] * __ret_sh[1] + __threadsPerBlock - 1) / __threadsPerBlock;

  pointer __d_a, __d_b, __d_c;
  cudaMalloc(&__d_a, this->__data_.size() * sizeof(value_t));
  cudaMalloc(&__d_b, __other.__data_.size() * sizeof(value_t));
  cudaMalloc(&__d_c, __ret_d.size() * sizeof(value_t));

  cudaMemcpy(__d_a, this->__data_.data(), this->__data_.size() * sizeof(value_t), cudaMemcpyHostToDevice);
  cudaMemcpy(__d_b, __other.__data_.data(), __other.__data_.size() * sizeof(value_t), cudaMemcpyHostToDevice);

  matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(__d_a, __d_b, __d_c, this->__shape_[0], this->__shape_[1], __other.shape()[1]);

  cudaMemcpy(__ret_d.data(), __d_c, __ret_d.size() * sizeof(value_t), cudaMemcpyDeviceToHost);

  cudaFree(__d_a);
  cudaFree(__d_b);
  cudaFree(__d_c);
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
      __sum += __a[__row * __n + __i] * __b[__i * __k + __col];

    __c[__row * __k + __col] = __sum;
  }
}
#endif

template<class _Tp>
tensor<_Tp> tensor<_Tp>::reshape(const shape_t __sh) const {
  __container__<value_t> __d = this->__data_;
  index_t                __s = this->__computeSize(__sh);

  if (__s != this->__data_.size())
    throw std::invalid_argument("input shape must have size of elements equal to the current number of elements in the tensor data");

  return __self(__d, __sh);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cross_product(const tensor& __other) const {
  this->__check_is_arithmetic_type("Cannot perform a cross product on non-scalar data types");
  if (this->empty() || __other.empty())
    throw std::invalid_argument("Cannot cross product an empty vector");

  if (this->shape() != std::vector<int>{3} || __other.shape() != std::vector<int>{3})
    throw std::invalid_argument("Cross product can only be performed on 3-element vectors");

  tensor __ret({3});
#if defined(__ARM_NEON) && defined(__aarch64__)
  float32x4_t __a      = vld1q_f32(reinterpret_cast<const float32_t*>(this->__data_.data()));
  float32x4_t __b      = vld1q_f32(reinterpret_cast<const float32_t*>(__other.storage().data()));
  float32x4_t __a_yzx  = vextq_f32(__a, __a, 1);
  float32x4_t __b_yzx  = vextq_f32(__b, __b, 1);
  float32x4_t __result = vsubq_f32(vmulq_f32(__a_yzx, __b), vmulq_f32(__a, __b_yzx));
  __result             = vextq_f32(__result, __result, 3);
  vst1q_f32(reinterpret_cast<float*>(__ret.storage().data()), __result);
#elif defined(__CUDACC__)
  pointer __d_a;
  pointer __d_b;
  pointer __d_c;

  cudaMalloc(&__d_a, 3 * sizeof(value_t));
  cudaMalloc(&__d_b, 3 * sizeof(value_t));
  cudaMalloc(&__d_c, 3 * sizeof(value_t));

  cudaMemcpy(__d_a, this->__data_.data(), 3 * sizeof(value_t), cudaMemcpyHostToDevice);
  cudaMemcpy(__d_b, __other.storage().data(), 3 * sizeof(value_t), cudaMemcpyHostToDevice);

  dim3 block(1);
  dim3 grid(1);
  cross_product_kernel<<<grid, block>>>(__d_a, __d_b, __d_c);

  cudaMemcpy(__ret.storage().data(), __d_c, 3 * sizeof(value_t), cudaMemcpyDeviceToHost);

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
  __m256 __a      = _mm256_loadu_ps(reinterpret_cast<const float*>(this->__data_.data()));
  __m256 __b      = _mm256_loadu_ps(reinterpret_cast<const float*>(__other.storage().data()));
  __m256 __a_yzx  = _mm256_permute_ps(__a, _MM_SHUFFLE(3, 0, 2, 1));
  __m256 __b_yzx  = _mm256_permute_ps(__b, _MM_SHUFFLE(3, 0, 2, 1));
  __m256 __result = _mm256_sub_ps(_mm256_mul_ps(__a_yzx, __b), _mm256_mul_ps(__a, __b_yzx));
  __result        = _mm256_permute_ps(__result, _MM_SHUFFLE(3, 0, 2, 1));
  _mm256_storeu_ps(reinterpret_cast<float*>(__ret.storage().data()), __result);
#endif
#if defined(__SSE__)
  __m128 __a      = _mm_loadu_ps(reinterpret_cast<const float*>(this->__data_.data()));
  __m128 __b      = _mm_loadu_ps(reinterpret_cast<const float*>(__other.storage().data()));
  __m128 __a_yzx  = _mm_shuffle_ps(__a, __a, _MM_SHUFFLE(3, 0, 2, 1));
  __m128 __b_yzx  = _mm_shuffle_ps(__b, __b, _MM_SHUFFLE(3, 0, 2, 1));
  __m128 __result = _mm_sub_ps(_mm_mul_ps(__a_yzx, __b), _mm_mul_ps(__a, __b_yzx));
  __result        = _mm_shuffle_ps(__result, __result, _MM_SHUFFLE(3, 0, 2, 1));
  _mm_storeu_ps(reinterpret_cast<float*>(__ret.storage().data()), __result);

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
  index_t                __s = __tensor.storage().size();
  __container__<value_t> __a;
  __a.reserve(__s);
  index_t __i = 0;
#ifdef __AVX__
  if constexpr (std::is_same_v<value_t, float>)
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
  if constexpr (std::is_same_v<value_t, float>)
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
    __a.push_back(static_cast<value_t>(std::fabs(float(__tensor.storage()[__i]))));

  return __self(__a, __tensor.__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::dot(const tensor& __other) const {
  this->__check_is_scalar_type("Cannot perform a dot product on non scalar data types");
  if (this->empty() || __other.empty())
    throw std::invalid_argument("Cannot dot product an empty vector");

  if (this->__shape_.size() == 1 && __other.shape().size() == 1)
  {
    if (this->__shape_[0] != __other.shape()[0])
      throw std::invalid_argument("Vectors must have the same size for dot product");

    const auto* __this_data  = this->__data_.data();
    const auto* __other_data = __other.storage().data();
    const auto  __size       = this->__data_.size();
    value_t     __ret        = 0;
#ifdef __CUDACC__
    if (this->__is_cuda_tensor && __other.__is_cuda_tensor)
    {
      thrust::device_vector<value_t> __d_result(1);
      thrust::transform(thrust::device, __this_data, __this_data + __size, __other_data, __d_result.begin(), thrust::multiplies<value_t>());
      __ret = thrust::reduce(__d_result.begin(), __d_result.end());
    }
    else
#endif
    {
#if defined(__AVX__)
      if constexpr (std::is_same_v<value_t, float> && __size >= _AVX_REG_WIDTH)
      {
        __m256 __sum = _mm256_setzero_ps();
        for (index_t __i = 0; __i < __size - 7; __i += _AVX_REG_WIDTH)
        {
          __m256 __a = _mm256_loadu_ps(__this_data + __i);
          __m256 __b = _mm256_loadu_ps(__other_data + __i);
          __sum      = _mm256_add_ps(__sum, _mm256_mul_ps(__a, __b));
        }
        __ret = _mm256_reduce_add_ps(__sum);
        for (index_t __i = __size - (__size % _AVX_REG_WIDTH); __i < __size; __i++)
          __ret += __this_data[__i] * __other_data[__i];
      }
      else
#elif defined(__SSE__)
      if constexpr (std::is_same_v<value_t, float> && __size >= 4)
      {
        __m128 __sum = _mm_setzero_ps();
        for (index_t __i = 0; __i < __size - 3; __i += 4)
        {
          __m128 __a = _mm_loadu_ps(__this_data + __i);
          __m128 __b = _mm_loadu_ps(__other_data + __i);
          __sum      = _mm_add_ps(__sum, _mm_mul_ps(__a, __b));
        }
        __ret = _mm_reduce_add_ps(__sum);
        for (index_t __i = __size - (__size % 4); __i < __size; __i++)
          __ret += __this_data[__i] * __other_data[__i];
      }
      else
#endif
      {
        __ret = std::inner_product(__this_data, __this_data + __size, __other_data, value_t(0));
      }
    }

    return __self({__ret}, {1});
  }

  if (this->__shape_.size() == 2 && __other.shape().size() == 2)
    return this->matmul(__other);

  if (this->__shape_.size() == 3 && __other.shape().size() == 3)
    return this->cross_product(__other);

  return __self();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::relu() const {
  return this->clone().relu_();
}

template<class _Tp>
void tensor<_Tp>::relu_() {
  this->__check_is_scalar_type("Cannot relu non-scalar type");
  index_t __s = this->__data_.size();
  index_t __i = 0;
#pragma omp parallel
#ifdef __CUDACC__
  if (this->__is_cuda_tensor)
  {
    value_t* __d_data = thrust::raw_pointer_cast(this->__data_.data());
    thrust::transform(thrust::device, __d_data, d_data + __s, __d_data, [] __device__(value_t __x) { return max(__x, value_t(0)); });
    return;
  }
#elif defined(__SSE__)
  if constexpr (std::is_same_v<value_t, float>)
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
  if constexpr (std::is_same_v<value_t, float>)
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
  if constexpr (std::is_same_v<value_t, float>)
  {
    const float32x4_t __vZero = vdupq_n_f32(0.0f);
    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH)
    {
      float32x4_t __v = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
      __v             = vmaxq_f32(__v, __vZero);
      vst1q_f32(&this->__data_[__i], __v);
    }
  }
  else if constexpr (std::is_same_v<value_t, int32_t>)
  {
    const int32x4_t __vZero = vdupq_n_s32(0);
    for (; __i + _ARM64_REG_WIDTH <= __s; __i += _ARM64_REG_WIDTH)
    {
      int32x4_t __v = vld1q_s32(reinterpret_cast<const int32_t*>(&this->__data_[__i]));
      __v           = vmaxq_s32(__v, __vZero);
      vst1q_s32(&this->__data_[__i], __v);
    }
  }
#endif
  for (__i = 0; __i < __s; __i++)
    this->__data_[__i] = std::max(this->__data_[__i], value_t(0));
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::transpose() const {
  if (this->__shape_.size() != 2)
    throw std::invalid_argument("Matrix transposition can only be done on 2D tensors");

  tensor        __ret({this->__shape_[1], this->__shape_[0]});
  const index_t __rows = this->__shape_[0];
  const index_t __cols = this->__shape_[1];
#ifdef __CUDACC__
  if (this->__is_cuda_tensor)
  {
    dim3 blockDim(16, 16);
    dim3 gridDim((__cols + blockDim.x - 1) / blockDim.x, (__rows + blockDim.y - 1) / blockDim.y);
    transpose_kernel<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(this->__data_.data()), thrust::raw_pointer_cast(__ret.__data_.data()),
                                            __rows, __cols);
    cudaDeviceSynchronize();
    return __ret;
  }
#endif
#if defined(__ARM_NEON)
  if constexpr (std::is_same_v<_Tp, float>)
  {
    for (index_t __i = 0; __i < __rows; __i += _ARM64_REG_WIDTH)
    {
      for (index_t __j = 0; __j < __cols; __j += _ARM64_REG_WIDTH)
      {
        if (__i + _ARM64_REG_WIDTH <= __rows && __j + _ARM64_REG_WIDTH <= __cols)
        {
          float32x4x4_t __input;
          for (index_t __k = 0; __k < _ARM64_REG_WIDTH; __k++)
            __input.val[__k] = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[(__i + __k) * __cols + __j]));

          float32x4x4_t __output = vld4q_f32(reinterpret_cast<const float32_t*>(&__input));

          for (index_t __k = 0; __k < _ARM64_REG_WIDTH; __k++)
            vst1q_f32(&__ret.__data_[(__j + __k) * __rows + __i], __output.val[__k]);
        }
        else
        {
          for (index_t __ii = __i; __ii < std::min(static_cast<index_t>(__i + _ARM64_REG_WIDTH), __rows); __ii++)
          {
            for (index_t __jj = __j; __jj < std::min(static_cast<index_t>(__j + _ARM64_REG_WIDTH), __cols); __jj++)
              __ret.at({__jj, __ii}) = this->at({__ii, __jj});
          }
        }
      }
    }
  }
  else
#endif
  {
    index_t __i = 0;
    for (; __i < __rows; __i++)
    {
      index_t __j = 0;
      for (; __j < __cols; __j++)
        __ret.at({__j, __i}) = this->at({__i, __j});
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
    output[__j * __rows + __i] = input[__i * __cols + __j];
}
#endif

template<class _Tp>
tensor<typename tensor<_Tp>::index_t> tensor<_Tp>::argsort(index_t __d, bool __ascending) const {
  index_t __adjusted = (__d < 0) ? __d + this->__data_.size() : __d;

  if (__adjusted != 0)
    throw std::out_of_range("Invalid dimension for argsort: only 1D tensors are supported");

  index_t                __size = static_cast<index_t>(this->__data_.size());
  __container__<index_t> __indices(__size);
  std::iota(__indices.begin(), __indices.end(), 0);
#if defined(__ARM_NEON)
  index_t __i = 0;

  if constexpr (std::is_same<value_t, float>::value)
  {
    for (; __i + _ARM64_REG_WIDTH <= __size; __i += _ARM64_REG_WIDTH)
    {
      float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
      float32x2_t __min1     = vpmin_f32(vget_low_f32(__data_vec), vget_high_f32(__data_vec));
      float32x2_t __min2     = vpmin_f32(__min1, __min1);
      float32x4_t __cmp_vec  = vdupq_lane_f32(__min2, 0);
      uint32x4_t  __cmp_result;

      if (__ascending)
        __cmp_result = vcltq_f32(__data_vec, __cmp_vec);
      else
        __cmp_result = vcgtq_f32(__data_vec, __cmp_vec);

      for (int __j = 0; __j < _ARM64_REG_WIDTH; __j++)
        __indices[__i + __j] = (__cmp_result[__j] ? __i + __j : __i + __j + 1);
    }
  }

  for (; __i < __size; __i++)
    __indices[__i] = __i;
#endif
  std::sort(__indices.begin(), __indices.end(), [&](index_t __a, index_t __b) {
    return __ascending ? this->__data_[__a] < this->__data_[__b] : this->__data_[__a] > this->__data_[__b];
  });

  return __self(__indices);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_left_shift(const int __amount) const {
  return this->clone().bitwise_left_shift_(__amount);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_xor(const tensor& __other) const {
  return this->clone().bitwise_xor_(__other);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_right_shift(const int __amount) const {
  return this->clone().bitwise_right_shift_(__amount);
}

template<class _Tp>
void tensor<_Tp>::bitwise_and_(const tensor& __other) {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape());
  index_t __i = 0;
#if defined(__ARM_NEON)
  const size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    uint32x4_t __data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
    uint32x4_t __other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));
    uint32x4_t __xor_vec   = vandq_u32(__data_vec, __other_vec);
    vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __xor_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] &= __other[__i];
}

template<class _Tp>
void tensor<_Tp>::bitwise_or_(const tensor& __other) {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape());
  index_t __i = 0;
#if defined(__ARM_NEON)
  const index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    uint32x4_t __data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
    uint32x4_t __other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));
    uint32x4_t __xor_vec   = vornq_u32(__data_vec, __other_vec);
    vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __xor_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] |= __other[__i];
}

template<class _Tp>
void tensor<_Tp>::bitwise_xor_(const tensor& __other) {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape());
  index_t __i = 0;
#if defined(__ARM_NEON)
  const index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    uint32x4_t __data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
    uint32x4_t __other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));
    uint32x4_t __xor_vec   = veorq_u32(__data_vec, __other_vec);
    vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __xor_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] ^= __other[__i];
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_not() const {
  return this->clone().logical_not_();
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const value_t __val) const {
  return this->clone().logical_or_(__val);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const tensor<_Tp>& __other) const {
  return this->clone().logical_or_(__other);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const tensor& __other) const {
  return this->clone().logical_xor_(__other);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const value_t __val) const {
  return this->clone().logical_xor_(__val);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const tensor& __other) const {
  return this->clone().logical_and_(__other);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const value_t __val) const {
  return this->clone().logical_and_(__val);
}

template<class _Tp>
void tensor<_Tp>::logical_or_(const tensor& __other) {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot get the element wise not of non-integral and non-boolean value");

  assert(this->__shape_ == __other.shape());
  index_t __i = 0;
#if defined(__ARM_NEON)
  const index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    uint32x4_t __data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
    uint32x4_t __other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));
    uint32x4_t __or_vec    = vornq_u32(__data_vec, __other_vec);
    vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __or_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = (this->__data_[__i] || __other[__i]);
}

template<class _Tp>
void tensor<_Tp>::logical_xor_(const tensor<_Tp>& __other) {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");

  assert(this->__shape_ == __other.shape());
  index_t __i = 0;
#if defined(__ARM_NEON)
  const index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    uint32x4_t __data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
    uint32x4_t __other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));
    uint32x4_t __xor_vec   = veorq_u32(__data_vec, __other_vec);
    vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __xor_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = (this->__data_[__i] ^ __other[__i]);
}

template<class _Tp>
void tensor<_Tp>::logical_and_(const tensor<_Tp>& __other) {
  if (!std::is_integral<value_t>::value && !std::is_same<value_t, bool>::value)
    throw std::runtime_error("Cannot get the element-wise and of non-integral and non-boolean value");

  assert(this->__shape_ == __other.shape());
  index_t __i = 0;
#if defined(__ARM_NEON)
  const index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    uint32x4_t __data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
    uint32x4_t __other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));
    uint32x4_t __and_vec   = vandq_u32(__data_vec, __other_vec);
    vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __and_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = (this->__data_[__i] && __other[__i]);
}

template<class _Tp>
double tensor<_Tp>::mean() const {
  this->__check_is_integral_type("input must be of integral type to get the mean average");
  double __m = 0.0f;

  if (this->empty())
    return __m;

  index_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
    __m += this->__data_[__i];

  return static_cast<double>(__m / static_cast<double>(this->__data_.size()));
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::pow(const tensor& __other) const {
  return this->clone().pow_(__other);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::pow(const value_t __val) const {
  return this->clone().pow_(__val);
}

template<class _Tp>
void tensor<_Tp>::pow_(const tensor& __other) {
  this->__check_is_integral_type("cannot get the power of a non integral value");
  assert(this->__shape_ == __other.shape());
  index_t __i = 0;
#if defined(__ARM_NEON)
  const index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __base_vec   = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __exp_vec    = vld1q_f32(reinterpret_cast<const float32_t*>(&__other[__i]));
    float32x4_t __result_vec = {std::pow(vgetq_lane_f32(__base_vec, 0), vgetq_lane_f32(__exp_vec, 0)),
                                std::pow(vgetq_lane_f32(__base_vec, 1), vgetq_lane_f32(__exp_vec, 1)),
                                std::pow(vgetq_lane_f32(__base_vec, 2), vgetq_lane_f32(__exp_vec, 2)),
                                std::pow(vgetq_lane_f32(__base_vec, 3), vgetq_lane_f32(__exp_vec, 3))};
    vst1q_f32(&this->__data_[__i], __result_vec);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    this->__data_[__i] = static_cast<value_t>(std::pow(double(this->__data_[__i]), double(__other[__i])));
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const tensor& __other) const {
  if (!std::is_integral<value_t>::value && !std::is_scalar<value_t>::value)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  assert(this->__shape_ == __other.shape());
  __container__<bool> __ret(this->__data_.size());
  index_t             __i = 0;
#if defined(__ARM_NEON)
  if constexpr (std::is_same_v<_Tp, float>)
  {
    constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      float32x4_t __data_vec1  = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
      float32x4_t __data_vec2  = vld1q_f32(reinterpret_cast<const float32_t*>(&__other.__data_[__i]));
      uint32x4_t  __cmp_result = vcleq_f32(__data_vec1, __data_vec2);
      uint32_t    __mask       = vaddvq_u32(__cmp_result);
      __ret[__i]               = __mask & 1;
      __ret[__i + 1]           = (__mask >> _AVX_REG_WIDTH) & 1;
      __ret[__i + 2]           = (__mask >> 16) & 1;
      __ret[__i + 3]           = (__mask >> 24) & 1;
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] <= __other[__i]);

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const value_t __val) const {
  if (!std::is_integral<value_t>::value && !std::is_scalar<value_t>::value)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  __container__<bool> __ret(this->__data_.size());
  index_t             __i = 0;
#if defined(__ARM_NEON)
  if constexpr (std::is_same_v<_Tp, float>)
  {
    constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      float32x4_t __data_vec   = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
      float32x4_t __val_vec    = vdupq_n_f32(__val);
      uint32x4_t  __cmp_result = vcleq_f32(__data_vec, __val_vec);
      uint32_t    __mask       = vaddvq_u32(__cmp_result);
      __ret[__i]               = __mask & 1;
      __ret[__i + 1]           = (__mask >> _AVX_REG_WIDTH) & 1;
      __ret[__i + 2]           = (__mask >> _AVX_REG_WIDTH * 2) & 1;
      __ret[__i + 3]           = (__mask >> _AVX_REG_WIDTH * 4) & 1;
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] <= __val);

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const tensor& __other) const {
  if (!std::is_integral<value_t>::value && !std::is_scalar<value_t>::value)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  assert(this->__shape_ == __other.shape());
  __container__<bool> __ret(this->__data_.size());
  index_t             __i = 0;
#if defined(__ARM_NEON)
  if constexpr (std::is_same_v<_Tp, float>)
  {
    constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      float32x4_t __data_vec1  = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
      float32x4_t __data_vec2  = vld1q_f32(reinterpret_cast<const float32_t*>(&__other.__data_[__i]));
      uint32x4_t  __cmp_result = vcgeq_f32(__data_vec1, __data_vec2);
      uint32_t    __mask       = vaddvq_u32(__cmp_result);
      __ret[__i]               = __mask & 1;
      __ret[__i + 1]           = (__mask >> _AVX_REG_WIDTH) & 1;
      __ret[__i + 2]           = (__mask >> _AVX_REG_WIDTH * 2) & 1;
      __ret[__i + 3]           = (__mask >> _AVX_REG_WIDTH * 4) & 1;
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] >= __other[__i]);

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const value_t __val) const {
  if (!std::is_integral<value_t>::value && !std::is_scalar<value_t>::value)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  __container__<bool> __ret(this->__data_.size());
  index_t             __i = 0;
#if defined(__ARM_NEON)
  if constexpr (std::is_same_v<_Tp, float>)
  {
    constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    float32x4_t       __val_vec  = vdupq_n_f32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      float32x4_t __data_vec   = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
      uint32x4_t  __cmp_result = vcgeq_f32(__data_vec, __val_vec);
      uint32_t    __mask       = vaddvq_u32(__cmp_result);
      __ret[__i]               = __mask & 1;
      __ret[__i + 1]           = (__mask >> 8) & 1;
      __ret[__i + 2]           = (__mask >> 16) & 1;
      __ret[__i + 3]           = (__mask >> 24) & 1;
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] >= __val);

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const tensor& __other) const {
  if (!std::is_integral<value_t>::value && !std::is_scalar<value_t>::value)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  assert(this->__shape_ == __other.shape() && "equal : tensor shapes");
  __container__<bool> __ret(this->__data_.size());
  index_t             __i = 0;
#if defined(__ARM_NEON)
  if constexpr (std::is_same_v<_Tp, float>)
  {
    constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      float32x4_t __data_vec1  = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
      float32x4_t __data_vec2  = vld1q_f32(reinterpret_cast<const float32_t*>(&__other.__data_[__i]));
      uint32x4_t  __cmp_result = vceqq_f32(__data_vec1, __data_vec2);
      uint32_t    __mask       = vaddvq_u32(__cmp_result);
      __ret[__i]               = __mask & 1;
      __ret[__i + 1]           = (__mask >> 8) & 1;
      __ret[__i + 2]           = (__mask >> 16) & 1;
      __ret[__i + 3]           = (__mask >> 24) & 1;
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] == __other[__i]);

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const value_t __val) const {
  if (!std::is_integral<value_t>::value && !std::is_scalar<value_t>::value)
    throw std::runtime_error("Cannot compare non-integral or scalar value");

  __container__<bool> __ret(this->__data_.size());
  index_t             __i = 0;
#if defined(__ARM_NEON)
  if constexpr (std::is_same_v<_Tp, float>)
  {
    constexpr index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    float32x4_t       __val_vec  = vdupq_n_f32(__val);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
      float32x4_t __data_vec   = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
      uint32x4_t  __cmp_result = vceqq_f32(__data_vec, __val_vec);
      uint32_t    __mask       = vaddvq_u32(__cmp_result);
      __ret[__i]               = __mask & 1;
      __ret[__i + 1]           = (__mask >> 8) & 1;
      __ret[__i + 2]           = (__mask >> 16) & 1;
      __ret[__i + 3]           = (__mask >> 24) & 1;
    }
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] == __val);

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sum(const index_t __axis) const {
  this->__check_is_scalar_type("Cannot reduce tensor with non scalar type");
  if (__axis < 0 || __axis >= static_cast<index_t>(this->__shape_.size()))
    throw std::invalid_argument("Invalid axis for sum");

  shape_t __ret_sh                  = this->__shape_;
  __ret_sh[__axis]                  = 1;
  index_t                __ret_size = std::accumulate(__ret_sh.begin(), __ret_sh.end(), 1, std::multiplies<index_t>());
  __container__<value_t> __ret_data(__ret_size, value_t(0.0f));
#if defined(__ARM_NEON)
  if constexpr (std::is_same_v<_Tp, float>)
  {
    const index_t __axis_size  = this->__shape_[__axis];
    const index_t __outer_size = this->__compute_outer_size(__axis);
    const index_t __inner_size = this->size(0) / (__outer_size * __axis_size);

    for (index_t __outer = 0; __outer < __outer_size; __outer++)
    {
      for (index_t __inner = 0; __inner < __inner_size; __inner++)
      {
        float32x4_t __sum_vec = vdupq_n_f32(0.0f);
        index_t     __i       = __outer * __axis_size * __inner_size + __inner;
        index_t     __j       = 0;

        for (; __j + _ARM64_REG_WIDTH <= __axis_size; __j += _ARM64_REG_WIDTH)
        {
          float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
          __sum_vec              = vaddq_f32(__sum_vec, __data_vec);
          __i += __inner_size * _ARM64_REG_WIDTH;
        }

        float __sum = vaddvq_f32(__sum_vec);
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
    index_t __i = 0;
    for (; __i < static_cast<index_t>(this->__data_.size()); __i++)
    {
      std::vector<index_t> __orig(this->__shape_.size());
      index_t              __index = __i;
      index_t              __j     = static_cast<index_t>(this->__shape_.size()) - 1;
      for (; __j >= 0; __j--)
      {
        __orig[__j] = __index % this->__shape_[__j];
        __index /= this->__shape_[__j];
      }

      __orig[__axis]      = 0;
      index_t __ret_index = 0;
      index_t __st        = 1;

      for (__j = static_cast<index_t>(this->__shape_.size()) - 1; __j >= 0; __j--)
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
tensor<_Tp> tensor<_Tp>::row(const index_t __index) const {
  if (this->__shape_.size() != 2)
    throw std::runtime_error("Cannot get a row from a non two dimensional tensor");

  if (this->__shape_[0] <= __index || __index < 0)
    throw std::invalid_argument("Index input is out of range");

  __container__<value_t> __r;
  index_t                __start = this->__shape_[1] * __index;
  index_t                __end   = this->__shape_[1] * __index + this->__shape_[1];
  index_t                __i     = __start;

  for (; __i < __end; __i++)
    __r.push_back(this->__data_[__i]);

  return __self(__r, {this->__shape_[1]});
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::col(const index_t __index) const {
  if (this->__shape_.size() != 2)
    throw std::runtime_error("Cannot get a column from a non two dimensional tensor");

  if (this->__shape_[1] <= __index || __index < 0)
    throw std::invalid_argument("Index input out of range");

  __container__<value_t> __c;
  index_t                __i = 0;
  for (; __i < this->__shape_[0]; __i++)
    __c.push_back(this->__data_[this->__compute_index({__i, __index})]);

  return __self(__c, {this->__shape_[0]});
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::ceil() const {
  this->__check_is_scalar_type("Cannot get the ceiling of a non scalar value");
  __container__<value_t> __ret(this->__data_.size());
  index_t                __i = 0;
#if defined(__ARM_NEON)
  const index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data   = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __ceiled = vrndpq_f32(__data);
    vst1q_f32(&__ret[__i], __ceiled);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    __ret[__i] = std::ceil(this->__data_[__i]);

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::floor() const {
  this->__check_is_scalar_type("Cannot get the floor of a non scalar value");
  __container__<value_t> __ret(this->__data_.size());
  index_t                __i = 0;
#if defined(__ARM_NEON)
  const index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  float32x4_t   zero       = vdupq_n_f32(0.0f);

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data    = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __floored = vrndmq_f32(__data);
    vst1q_f32(&__ret[__i], __floored);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
    __ret[__i] = std::floor(this->__data_[__i]);

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::clamp(const_pointer __min_val, const_pointer __max_val) const {
  return this->clone().clamp_(__min_val, __max_val);
}

template<class _Tp>
void tensor<_Tp>::clamp_(const_pointer __min_val, const_pointer __max_val) {
  index_t __i = 0;
#if defined(__AVX2__)
  const index_t __simd_end = this->__data_.size() - (this->__data_.size() % _AVX_REG_WIDTH);
  __m256        __min_vec  = _mm256_set1_ps(__min_val ? *__min_val : std::numeric_limits<_Tp>::lowest());
  __m256        __max_vec  = _mm256_set1_ps(__max_val ? *__max_val : std::numeric_limits<_Tp>::max());

  for (; __i < __simd_end; __i += _AVX_REG_WIDTH)
  {
    __m256 __data_vec = _mm256_loadu_ps(&this->__data_[__i]);
    __m256 __clamped  = _mm256_min_ps(_mm256_max_ps(data_vec, __min_vec), __max_vec);
    _mm256_storeu_ps(&this->__data_[__i], __clamped);
  }
#elif defined(__ARM_NEON)
  const index_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
  float32x4_t   __min_vec  = vdupq_n_f32(__min_val ? *__min_val : std::numeric_limits<_Tp>::lowest());
  float32x4_t   __max_vec  = vdupq_n_f32(__max_val ? *__max_val : std::numeric_limits<_Tp>::max());

  for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
  {
    float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
    float32x4_t __clamped  = vminq_f32(vmaxq_f32(__data_vec, __min_vec), __max_vec);
    vst1q_f32(&this->__data_[__i], __clamped);
  }
#endif
  for (; __i < this->__data_.size(); __i++)
  {
    if (__min_val)
      this->__data_[__i] = std::max(*__min_val, this->__data_[__i]);

    if (__max_val)
      this->__data_[__i] = std::min(*__max_val, this->__data_[__i]);
  }
}

template<class _Tp>
typename tensor<_Tp>::index_t tensor<_Tp>::hash() const {
  index_t            __hash_val = 0;
  std::hash<value_t> __hasher;

  index_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
    __hash_val ^= __hasher(this->__data_[__i]) + 0x9e3779b9 + (__hash_val << 6) + (__hash_val >> 2);

  return __hash_val;
}
