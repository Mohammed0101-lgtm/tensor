//
// tensor base
//

#pragma once

#include <__algorithm/clamp.h>
#include <__algorithm/comp.h>
#include <__algorithm/sort.h>
#include <__functional/hash.h>
#include <omp-tools.h>
#include <omp.h>

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
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
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
using neon_u8  = uint8x16_t;

#endif

const int _ARM64_REG_WIDTH = 4;
const int _AVX_REG_WIDTH   = 8;

template <class _Tp>
class tensor {
 public:
  using __self                 = tensor;
  using value_type             = _Tp;
  using data_t                 = std::vector<value_type>;
  using index_type             = uint64_t;
  using shape_type             = std::vector<index_type>;
  using reference              = value_type&;
  using const_reference        = const value_type&;
  using pointer                = value_type*;
  using const_pointer          = const value_type*;
  using iterator               = typename data_t::iterator;
  using const_iterator         = typename data_t::const_iterator;
  using reverse_iterator       = typename data_t::reverse_iterator;
  using const_reverse_iterator = typename data_t::const_reverse_iterator;

  enum class Device { CPU, CUDA };

 private:
  mutable data_t     __data_;
  mutable shape_type __shape_;
  mutable shape_type __strides_;
  Device             __device_;
  bool               __is_cuda_tensor_ = false;

 public:
  tensor() = default;
  explicit tensor(const shape_type& __sh, const_reference __v, Device __d = Device::CPU);
  explicit tensor(const shape_type& __sh, Device __d = Device::CPU);
  explicit tensor(const shape_type& __sh, const data_t& __d, Device __dev = Device::CPU);
  tensor(const tensor& __t);
  tensor(tensor&& __t) noexcept;
  tensor(const shape_type& __sh, const tensor& __other);
  tensor(const shape_type& __sh, std::initializer_list<value_type> init_list,
         Device __d = Device::CPU);

 private:
  class __destroy_tensor {
   private:
    tensor& __tens_;

   public:
    explicit __destroy_tensor(tensor& __tens) : __tens_(__tens) {}

    void operator()() {}
  };

 public:
  ~tensor() { __destroy_tensor (*this)(); }

  data_t storage() const noexcept;

  tensor<int64_t> long_() const;

  tensor<int32_t> int32_() const;

  tensor<uint32_t> uint32_() const;

  tensor<uint64_t> unsigned_long_() const;

  tensor<float32_t> float32_() const;

  tensor<float64_t> double_() const;

  shape_type shape() const noexcept;

  shape_type strides() const noexcept;

  Device device() const noexcept;

  size_t n_dims() const noexcept;

  index_type size(const index_type __dim) const;

  index_type capacity() const noexcept;

  index_type count_nonzero(index_type __dim = 0) const;

  index_type lcm() const;

  index_type hash() const;

  reference at(shape_type __idx);

  reference operator[](const index_type __idx);

  const_reference at(const shape_type __idx) const;

  const_reference operator[](const index_type __idx) const;

  reference operator()(std::initializer_list<index_type> __index_list);

  const_reference operator()(std::initializer_list<index_type> __index_list) const;

  bool empty() const;

  /// @brief Converts the tensor elements to boolean values.
  /// @return A tensor of boolean values.
  tensor<bool> bool_() const;

  /// @brief Computes the logical NOT operation for each element in the tensor.
  /// @return A tensor of boolean values representing the logical NOT of each
  /// element.
  tensor<bool> logical_not() const;

  /// @brief Computes the logical OR operation between each element and a scalar
  /// value.
  /// @param __val The scalar value to compute the logical OR with.
  /// @return A tensor of boolean values.
  tensor<bool> logical_or(const value_type __val) const;

  /// @brief Computes the logical OR operation between corresponding elements of
  /// two tensors.
  /// @param __other The tensor to compute the logical OR with.
  /// @return A tensor of boolean values.
  tensor<bool> logical_or(const tensor& __other) const;

  /// @brief Checks if each element is less than or equal to the corresponding
  /// element in another tensor.
  /// @param __other The tensor to compare with.
  /// @return A tensor of boolean values.
  tensor<bool> less_equal(const tensor& __other) const;

  /// @brief Checks if each element is less than or equal to a scalar value.
  /// @param __val The scalar value to compare with.
  /// @return A tensor of boolean values.
  tensor<bool> less_equal(const value_type __val) const;

  /// @brief Checks if each element is greater than or equal to the
  /// corresponding element in another tensor.
  /// @param __other The tensor to compare with.
  /// @return A tensor of boolean values.
  tensor<bool> greater_equal(const tensor& __other) const;

  /// @brief Checks if each element is greater than or equal to a scalar value.
  /// @param __val The scalar value to compare with.
  /// @return A tensor of boolean values.
  tensor<bool> greater_equal(const value_type __val) const;

  /// @brief Checks if each element is equal to the corresponding element in
  /// another tensor.
  /// @param __other The tensor to compare with.
  /// @return A tensor of boolean values.
  tensor<bool> equal(const tensor& __other) const;

  /// @brief Checks if each element is equal to a scalar value.
  /// @param __val The scalar value to compare with.
  /// @return A tensor of boolean values.
  tensor<bool> equal(const value_type __val) const;

  /// @brief Checks if each element is not equal to the corresponding element in
  /// another tensor.
  /// @param __other The tensor to compare with.
  /// @return A tensor of boolean values.
  tensor<bool> not_equal(const tensor& __other) const;

  /// @brief Checks if each element is not equal to a scalar value.
  /// @param __val The scalar value to compare with.
  /// @return A tensor of boolean values.
  tensor<bool> not_equal(const value_type __val) const;

  /// @brief Checks if each element is less than the corresponding element in
  /// another tensor.
  /// @param __other The tensor to compare with.
  /// @return A tensor of boolean values.
  tensor<bool> less(const tensor& __other) const;

  /// @brief Checks if each element is less than a scalar value.
  /// @param __val The scalar value to compare with.
  /// @return A tensor of boolean values.
  tensor<bool> less(const value_type __val) const;

  /// @brief Checks if each element is greater than the corresponding element in
  /// another tensor.
  /// @param __other The tensor to compare with.
  /// @return A tensor of boolean values.
  tensor<bool> greater(const tensor& __other) const;

  /// @brief Checks if each element is greater than a scalar value.
  /// @param __val The scalar value to compare with.
  /// @return A tensor of boolean values.
  tensor<bool> greater(const value_type __val) const;

  /// @brief Compares this tensor with another tensor for equality.
  /// @param __other The tensor to compare with.
  /// @return True if the tensors are equal, false otherwise.
  bool operator==(const tensor& __other) const;

  /// @brief Compares this tensor with another tensor for inequality.
  /// @param __other The tensor to compare with.
  /// @return True if the tensors are not equal, false otherwise.
  bool operator!=(const tensor& __other) const;

  /// @brief Adds the elements of another tensor to this tensor.
  /// @param __other The tensor to add.
  /// @return A new tensor containing the result of the addition.
  tensor operator+(const tensor& __other) const;

  /// @brief Subtracts the elements of another tensor from this tensor.
  /// @param __other The tensor to subtract.
  /// @return A new tensor containing the result of the subtraction.
  tensor operator-(const tensor& __other) const;

  /// @brief Adds a scalar value to each element of the tensor.
  /// @param __val The scalar value to add.
  /// @return A new tensor containing the result of the addition.
  tensor operator+(const value_type __val) const;

  /// @brief Subtracts a scalar value from each element of the tensor.
  /// @param __val The scalar value to subtract.
  /// @return A new tensor containing the result of the subtraction.
  tensor operator-(const value_type __val) const;

  /// @brief Multiply a scalar value with each element of the tensor.
  /// @param __val The scalar value to multiply with.
  /// @return A new tensor containing the result of the multiplication.
  tensor operator*(const value_type __val) const;

  /// @brief Multiply each element of the input tensor with each element of the
  /// tensor.
  /// @param __other The tensor to multiply with.
  /// @return A new tensor containing the result of the multiplication.
  tensor operator*(const tensor& __other) const;

  /// @brief Subtracts the elements of another tensor from this tensor and
  /// updates the current tensor.
  /// @param __other The tensor to subtract.
  /// @return A reference to the current tensor after the subtraction.
  tensor& operator-=(const tensor& __other) const;

  /// @brief Adds the elements of another tensor to this tensor and updates the
  /// current tensor.
  /// @param __other The tensor to add.
  /// @return A reference to the current tensor after the addition.
  tensor& operator+=(const tensor& __other) const;

  /// @brief Multiplies the elements of this tensor by the elements of another
  /// tensor and updates the current tensor.
  /// @param __other The tensor to multiply with.
  /// @return A reference to the current tensor after the multiplication.
  tensor& operator*=(const tensor& __other) const;

  /// @brief Divides the elements of this tensor by the elements of another
  /// tensor and updates the current tensor.
  /// @param __other The tensor to divide by.
  /// @return A reference to the current tensor after the division.
  tensor& operator/=(const tensor& __other) const;

  /// @brief Adds a scalar value to each element of the tensor and updates the
  /// current tensor.
  /// @param __val The scalar value to add.
  /// @return A reference to the current tensor after the addition.
  tensor& operator+=(const_reference __val) const;

  /// @brief Subtracts a scalar value from each element of the tensor and
  /// updates the current tensor.
  /// @param __val The scalar value to subtract.
  /// @return A reference to the current tensor after the subtraction.
  tensor& operator-=(const_reference __val) const;

  /// @brief Divides each element of the tensor by a scalar value and updates
  /// the current tensor.
  /// @param __val The scalar value to divide by.
  /// @return A reference to the current tensor after the division.
  tensor& operator/=(const_reference __val) const;

  /// @brief Divides each element of the tensor by the corresponding element of
  /// the other tensor
  /// @param __other The other tensor to divide by (should not contain a zero)
  /// @return A tensor the contains the result of the division
  tensor operator/(const tensor& __other) const;

  /// @brief Divides each element of the tensor by a scalar value
  /// @param __val The scalar value to divide by (non zero)
  /// @return A tensor the contains the result of the division
  tensor operator/(const_reference __val) const;

  /// @brief Multiplies each element of the tensor by a scalar value and updates
  /// the current tensor.
  /// @param __val The scalar value to multiply with.
  /// @return A reference to the current tensor after the multiplication.
  tensor& operator*=(const_reference __val) const;

  /// @brief Assigns the values of another tensor to this tensor.
  /// @param __other The tensor to assign from.
  /// @return A reference to the current tensor after the assignment.
  tensor& operator=(const tensor& __other) const;

  /// @brief Moves the values of another tensor to this tensor.
  /// @param __other The tensor to move from.
  /// @return A reference to the current tensor after the move.
  tensor& operator=(tensor&& __other) const noexcept;

  /// @brief Assigns the values of another tensor to this tensor (default
  /// implementation).
  /// @param __other The tensor to assign from.
  /// @return A reference to the current tensor after the assignment.
  tensor& operator=(const tensor&) = default;

  tensor<bool>& operator!() const;

  /// @brief Extracts a slice of the tensor along the specified dimension.
  /// @param __dim The dimension along which to slice.
  /// @param __start The starting index of the slice (optional).
  /// @param __end The ending index of the slice (optional).
  /// @param __step The step size for the slice.
  /// @return A new tensor containing the sliced data.
  tensor slice(index_type __dim, std::optional<index_type> __start, std::optional<index_type> __end,
               int64_t __step) const;

  /// @brief Computes the element-wise maximum between this tensor and another
  /// tensor.
  /// @param __other The tensor to compare with.
  /// @return A new tensor containing the element-wise maximum values.
  tensor fmax(const tensor& __other) const;

  /// @brief Computes the element-wise maximum between this tensor and a scalar
  /// value.
  /// @param __val The scalar value to compare with.
  /// @return A new tensor containing the element-wise maximum values.
  tensor fmax(const value_type __val) const;

  /// @brief Computes the element-wise floating-point remainder of division with
  /// another tensor.
  /// @param __other The tensor to divide by.
  /// @return A new tensor containing the element-wise remainders.
  tensor fmod(const tensor& __other) const;

  /// @brief Computes the element-wise floating-point remainder of division with
  /// a scalar value.
  /// @param __val The scalar value to divide by.
  /// @return A new tensor containing the element-wise remainders.
  tensor fmod(const value_type __val) const;

  /// @brief Computes the fractional part of each element in the tensor.
  /// @return A new tensor containing the fractional parts.
  tensor frac() const;

  /// @brief Computes the natural logarithm (log base e) of each element in the
  /// tensor.
  /// @return A new tensor containing the logarithm of each element.
  tensor log() const;

  /// @brief Computes the base-10 logarithm of each element in the tensor.
  /// @return A new tensor containing the base-10 logarithm of each element.
  tensor log10() const;

  /// @brief Computes the base-2 logarithm of each element in the tensor.
  /// @return A new tensor containing the base-2 logarithm of each element.
  tensor log2() const;

  /// @brief Computes the exponential (e^x) of each element in the tensor.
  /// @return A new tensor containing the exponential of each element.
  tensor exp() const;

  /// @brief Computes the square root of each element in the tensor.
  /// @return A new tensor containing the square root of each element.
  tensor sqrt() const;

  /// @brief Computes the sum of the elements along a specified axis.
  /// @param __axis The axis along which to compute the sum.
  /// @return A new tensor containing the sums along the specified axis.
  tensor sum(const index_type __axis) const;

  /// @brief Extracts a specific row from the tensor.
  /// @param __index The index of the row to extract.
  /// @return A new tensor containing the specified row.
  tensor row(const index_type __index) const;

  /// @brief Extracts a specific column from the tensor.
  /// @param __index The index of the column to extract.
  /// @return A new tensor containing the specified column.
  tensor col(const index_type __index) const;

  /// @brief Computes the ceiling of each element in the tensor (rounds up to
  /// the nearest integer).
  /// @return A new tensor containing the ceiling of each element.
  tensor ceil() const;

  /// @brief Computes the floor of each element in the tensor (rounds down to
  /// the nearest integer).
  /// @return A new tensor containing the floor of each element.
  tensor floor() const;

  /// @brief Creates a copy of the tensor.
  /// @return A new tensor that is a clone of the current tensor.
  tensor clone() const;

  /// @brief Clamps the values of the tensor to the specified minimum and
  /// maximum range.
  /// @param __min_val A pointer to the minimum value (optional).
  /// @param __max_val A pointer to the maximum value (optional).
  /// @return A new tensor with values clamped to the specified range.
  tensor clamp(const_reference __min_val = std::numeric_limits<value_type>::lowest(),
               const_reference __max_val = std::numeric_limits<value_type>::max()) const;

  /// @brief Computes the cosine of each element in the tensor.
  /// @return A new tensor containing the cosine of each element.
  tensor cos() const;

  /// @brief Computes the hyperbolic cosine (cosh) of each element in the
  /// tensor.
  /// @return A new tensor containing the hyperbolic cosine of each element.
  tensor cosh() const;

  /// @brief Computes the arc cosine (inverse cosine) of each element in the
  /// tensor.
  /// @return A new tensor containing the arc cosine of each element.
  tensor acos() const;

  /// @brief Computes the inverse hyperbolic cosine (acosh) of each element in
  /// the tensor.
  /// @return A new tensor containing the inverse hyperbolic cosine of each
  /// element.
  tensor acosh() const;

  /// @brief Computes the tangent of each element in the tensor.
  /// @return A new tensor containing the tangent of each element.
  tensor tan() const;

  /// @brief Computes the hyperbolic tangent (tanh) of each element in the
  /// tensor.
  /// @return A new tensor containing the hyperbolic tangent of each element.
  tensor tanh() const;

  /// @brief Computes the arc tangent (inverse tangent) of each element in the
  /// tensor.
  /// @return A new tensor containing the arc tangent of each element.
  tensor atan() const;

  /// @brief Computes the inverse hyperbolic tangent (atanh) of each element in
  /// the tensor.
  /// @return A new tensor containing the inverse hyperbolic tangent of each
  /// element.
  tensor atanh() const;

  /// @brief Computes the sine of each element in the tensor.
  /// @return A new tensor containing the sine of each element.
  tensor sin() const;

  /// @brief Computes the normalized sinc function (sine(x)/x) of each element
  /// in the tensor.
  /// @return A new tensor containing the sinc function values of each element.
  tensor sinc() const;

  /// @brief Computes the hyperbolic sine (sinh) of each element in the tensor.
  /// @return A new tensor containing the hyperbolic sine of each element.
  tensor sinh() const;

  /// @brief Computes the arc sine (inverse sine) of each element in the tensor.
  /// @return A new tensor containing the arc sine of each element.
  tensor asin() const;

  /// @brief Computes the inverse hyperbolic sine (asinh) of each element in the
  /// tensor.
  /// @return A new tensor containing the inverse hyperbolic sine of each
  /// element.
  tensor asinh() const;

  /// @brief Computes the absolute value of each element in the tensor.
  /// @return A new tensor containing the absolute value of each element.
  tensor abs() const;

  /// @brief Performs a logical XOR operation between the tensor and another
  /// tensor.
  /// @param __other The tensor to apply the logical XOR operation with.
  /// @return A new tensor containing the result of the logical XOR operation.
  tensor logical_xor(const tensor& __other) const;

  /// @brief Performs a logical XOR operation between the tensor and a scalar
  /// value.
  /// @param __val The scalar value to apply the logical XOR operation with.
  /// @return A new tensor containing the result of the logical XOR operation.
  tensor logical_xor(const value_type __val) const;

  /// @brief Performs a logical AND operation between the tensor and another
  /// tensor.
  /// @param __other The tensor to apply the logical AND operation with.
  /// @return A new tensor containing the result of the logical AND operation.
  tensor logical_and(const tensor& __other) const;

  /// @brief Performs a logical AND operation between the tensor and a scalar
  /// value.
  /// @param __val The scalar value to apply the logical AND operation with.
  /// @return A new tensor containing the result of the logical AND operation.
  tensor logical_and(const value_type __val) const;

  /// @brief Performs a bitwise NOT operation on each element of the tensor.
  /// @return A new tensor containing the result of the bitwise NOT operation.
  tensor bitwise_not() const;

  /// @brief Performs a bitwise AND operation between the tensor and a scalar
  /// value.
  /// @param __val The scalar value to apply the bitwise AND operation with.
  /// @return A new tensor containing the result of the bitwise AND operation.
  tensor bitwise_and(const value_type __val) const;

  /// @brief Performs a bitwise AND operation between the tensor and another
  /// tensor.
  /// @param __other The tensor to apply the bitwise AND operation with.
  /// @return A new tensor containing the result of the bitwise AND operation.
  tensor bitwise_and(const tensor& __other) const;

  /// @brief Performs a bitwise OR operation between the tensor and a scalar
  /// value.
  /// @param __val The scalar value to apply the bitwise OR operation with.
  /// @return A new tensor containing the result of the bitwise OR operation.
  tensor bitwise_or(const value_type __val) const;

  /// @brief Performs a bitwise OR operation between the tensor and another
  /// tensor.
  /// @param __other The tensor to apply the bitwise OR operation with.
  /// @return A new tensor containing the result of the bitwise OR operation.
  tensor bitwise_or(const tensor& __other) const;

  /// @brief Performs a bitwise XOR operation between the tensor and a scalar
  /// value.
  /// @param __val The scalar value to apply the bitwise XOR operation with.
  /// @return A new tensor containing the result of the bitwise XOR operation.
  tensor bitwise_xor(const value_type __val) const;

  /// @brief Performs a bitwise XOR operation between the tensor and another
  /// tensor.
  /// @param __other The tensor to apply the bitwise XOR operation with.
  /// @return A new tensor containing the result of the bitwise XOR operation.
  tensor bitwise_xor(const tensor& __other) const;

  /// @brief Performs a bitwise left shift operation on each element of the
  /// tensor by a specified amount.
  /// @param __amount The number of bits to shift left.
  /// @return A new tensor containing the result of the bitwise left shift
  /// operation.
  tensor bitwise_left_shift(const int __amount) const;

  /// @brief Performs a bitwise right shift operation on each element of the
  /// tensor by a specified amount.
  /// @param __amount The number of bits to shift right.
  /// @return A new tensor containing the result of the bitwise right shift
  /// operation.
  tensor bitwise_right_shift(const int __amount) const;

  /// @brief Performs matrix multiplication between the tensor and another
  /// tensor.
  /// @param __other The tensor to multiply with.
  /// @return A new tensor containing the result of the matrix multiplication.
  tensor matmul(const tensor& __other) const;

  /// @brief Reshapes the tensor to a specified shape.
  /// @param __shape The desired shape for the tensor.
  /// @return A new tensor with the specified shape.
  tensor reshape(const shape_type __shape) const;

  /// @brief Reshapes the tensor to match the shape of another tensor.
  /// @param __other The tensor whose shape is to be matched.
  /// @return A new tensor reshaped to match the shape of the given tensor.
  tensor reshape_as(const tensor& __other) const;

  /// @brief Computes the cross product between the tensor and another tensor.
  /// @param __other The tensor to compute the cross product with.
  /// @return A new tensor containing the cross product.
  tensor cross_product(const tensor& __other) const;

  /// @brief Computes the absolute value of each element in a given tensor.
  /// @param __tensor The tensor to compute the absolute values of.
  /// @return A new tensor containing the absolute values of the input tensor.
  tensor absolute(const tensor& __tensor) const;

  /// @brief Computes the dot product between the tensor and another tensor.
  /// @param __other The tensor to compute the dot product with.
  /// @return A scalar tensor containing the result of the dot product.
  tensor dot(const tensor& __other) const;

  /// @brief Applies the ReLU activation function to the tensor.
  /// @return A new tensor where negative values are replaced with zero.
  tensor relu() const;

  /// @brief Transposes the tensor, swapping rows and columns.
  /// @return A new tensor that is the transpose of the original tensor.
  tensor transpose() const;

  /// @brief Fills the tensor with a specified scalar value.
  /// @param __val The scalar value to fill the tensor with.
  /// @return A new tensor filled with the specified value.
  tensor fill(const value_type __val) const;

  /// @brief Fills the tensor with the values from another tensor.
  /// @param __other The tensor whose values are to be used for filling.
  /// @return A new tensor filled with the values from the specified tensor.
  tensor fill(const tensor& __other) const;

  /// @brief Resizes the tensor to a specified shape, keeping its data
  /// consistent.
  /// @param __sh The desired shape for the tensor.
  /// @return A new tensor resized to the specified shape.
  tensor resize_as(const shape_type __sh) const;

  /// @brief Checks if all elements in the tensor are non-zero.
  /// @return A scalar tensor containing true if all elements are non-zero,
  /// otherwise false.
  tensor all() const;

  /// @brief Checks if any element in the tensor is non-zero.
  /// @return A scalar tensor containing true if any element is non-zero,
  /// otherwise false.
  tensor any() const;

  /// @brief Computes the determinant of the tensor.
  /// @return A scalar tensor containing the determinant of the tensor.
  tensor det() const;

  /// @brief Computes the square of each element in the tensor.
  /// @return A new tensor containing the squares of the elements.
  tensor square() const;

  /// @brief Applies the sigmoid activation function to each element in the
  /// tensor.
  /// @return A new tensor containing the sigmoid values of the elements.
  tensor sigmoid() const;

  /// @brief Applies a clipped ReLU activation function to the tensor.
  /// @return A new tensor where negative values are replaced with zero, and
  /// positive values are clipped to a maximum value.
  tensor clipped_relu() const;

  /// @brief Sorts the elements of the tensor along a specified dimension.
  /// @param __dim The dimension along which to sort the tensor.
  /// @param __descending Whether to sort in descending order (default is
  /// false).
  /// @return A new tensor with the sorted elements.
  tensor sort(index_type __dim, bool __descending = false) const;

  /// @brief Computes the remainder of division between each element and a
  /// scalar value.
  /// @param __val The scalar value to divide by.
  /// @return A new tensor containing the remainders.
  tensor remainder(const value_type __val) const;

  /// @brief Computes the remainder of division between each element of the
  /// tensor and another tensor.
  /// @param __other The tensor to divide by.
  /// @return A new tensor containing the remainders.
  tensor remainder(const tensor& __other) const;

  /// @brief Computes the element-wise maximum between the tensor and another
  /// tensor.
  /// @param __other The tensor to compare with.
  /// @return A new tensor containing the element-wise maximum values.
  tensor maximum(const tensor& __other) const;

  /// @brief Computes the element-wise maximum between the tensor and a scalar
  /// value.
  /// @param __val The scalar value to compare with.
  /// @return A new tensor containing the element-wise maximum values.
  tensor maximum(const_reference __val) const;

  /// @brief Computes the distance between the tensor and another tensor.
  /// @param __other The tensor to compute the distance from.
  /// @return A scalar tensor containing the computed distance.
  tensor dist(const tensor& __other) const;

  /// @brief Computes the distance between the tensor and a scalar value.
  /// @param __val The scalar value to compute the distance from.
  /// @return A scalar tensor containing the computed distance.
  tensor dist(const value_type __val) const;

  /// @brief Removes dimensions of size 1 from the specified axis of the tensor.
  /// @param __dim The dimension to squeeze.
  /// @return A new tensor with the specified dimension removed if its size
  /// is 1.
  tensor squeeze(index_type __dim) const;

  /// @brief Computes the element-wise negation of the tensor.
  /// @return A new tensor with all elements negated.
  tensor negative() const;

  /// @brief Repeats the tensor along specified dimensions.
  /// @param __d A data structure specifying the number of repetitions along
  /// each dimension.
  /// @return A new tensor with repeated elements along the specified
  /// dimensions.
  tensor repeat(const data_t& __d, int __dim = 0) const;

  /// @brief Permutes the dimensions of the tensor.
  /// @param __dim The order of dimensions for permutation.
  /// @return A new tensor with permuted dimensions.
  tensor permute(const index_type __dim) const;

  /// @brief Computes the log softmax along a specified dimension of the tensor.
  /// @param __dim The dimension along which to compute the log softmax.
  /// @return A new tensor containing the log softmax values.
  tensor log_softmax(const index_type __dim) const;

  /// @brief Computes the element-wise greatest common divisor (GCD) between the
  /// tensor and another tensor.
  /// @param __other The tensor to compute the GCD with.
  /// @return A new tensor containing the element-wise GCD values.
  tensor gcd(const tensor& __other) const;

  /// @brief Computes the element-wise greatest common divisor (GCD) between the
  /// tensor and a scalar value.
  /// @param __val The scalar value to compute the GCD with.
  /// @return A new tensor containing the element-wise GCD values.
  tensor gcd(const value_type __val) const;

  /// @brief Computes the element-wise power of the tensor, raising each element
  /// to the power of the corresponding element in another tensor.
  /// @param __other The tensor representing the exponents.
  /// @return A new tensor containing the element-wise power values.
  tensor pow(const tensor& __other) const;

  /// @brief Computes the element-wise power of the tensor, raising each element
  /// to the power of a scalar value.
  /// @param __val The scalar exponent value.
  /// @return A new tensor containing the element-wise power values.
  tensor pow(const value_type __val) const;

  /// @brief Computes the cumulative product of elements along a specified
  /// dimension.
  /// @param __dim The dimension along which to compute the cumulative product.
  /// Default is -1 (last dimension).
  /// @return A new tensor containing the cumulative product along the specified
  /// dimension.
  tensor cumprod(index_type __dim = -1) const;

  /// @brief Concatenates the tensor with a list of other tensors along a
  /// specified dimension.
  /// @param _others A vector of tensors to concatenate.
  /// @param _dim The dimension along which to concatenate the tensors.
  /// @return A new tensor resulting from the concatenation.
  tensor cat(const std::vector<tensor>& _others, index_type _dim) const;

  /// @brief Finds the indices of the maximum values along a specified
  /// dimension.
  /// @param __dim The dimension along which to compute the indices of the
  /// maximum values.
  /// @return A new tensor containing the indices of the maximum values along
  /// the specified dimension.
  tensor argmax(index_type __dim) const;

  /// @brief Adds a dimension of size 1 at the specified axis.
  /// @param __dim The dimension where the new axis will be added.
  /// @return A new tensor with the added dimension.
  tensor unsqueeze(index_type __dim) const;

  /// @brief Creates a tensor filled with zeros, with the specified shape.
  /// @param __sh The shape of the tensor to be created.
  /// @return A new tensor filled with zeros.
  tensor zeros(const shape_type& __sh);

  /// @brief Creates a tensor filled with ones, with the specified shape.
  /// @param __sh The shape of the tensor to be created.
  /// @return A new tensor filled with ones.
  tensor ones(const shape_type& __sh);

  /// @brief Creates a tensor filled with random values, with the specified
  /// shape.
  /// @param __sh The shape of the tensor to be created.
  /// @param __bounded Whether the random values should be bounded (default is
  /// false).
  /// @return A new tensor filled with random values.
  tensor randomize(const shape_type& __sh, bool __bounded = false);

  /// @brief Extracts the minor matrix by removing the specified row and column.
  /// @param __a The row index to remove.
  /// @param __b The column index to remove.
  /// @return A new tensor representing the minor matrix.
  tensor get_minor(index_type __a, index_type __b) const;

  /// @brief Expands the tensor to match the shape of another tensor along a
  /// specified dimension.
  /// @param __sh The target shape for expansion.
  /// @param __dim The dimension along which the tensor will be expanded.
  /// @return A new tensor expanded to the specified shape.
  tensor expand_as(shape_type __sh, index_type __dim) const;

  /// @brief Computes the element-wise least common multiple (LCM) with another
  /// tensor.
  /// @param __other The tensor to compute the LCM with.
  /// @return A new tensor containing the element-wise LCM values.
  tensor lcm(const tensor& __other) const;

  /// @brief Computes the mean (average) of all elements in the tensor.
  /// @return The mean value of all elements in the tensor.
  double mean() const;

  /// @brief Computes the median of the elements along a specified dimension.
  /// @param __dim The dimension along which to compute the median.
  /// @return The median value of the elements along the specified dimension.
  double median(const index_type __dim) const;

  /// @brief Computes the mode (most frequent value) of the elements along a
  /// specified dimension.
  /// @param __dim The dimension along which to compute the mode.
  /// @return The mode value of the elements along the specified dimension.
  double mode(const index_type __dim) const;

  /// @brief Appends a value to the end of the tensor.
  /// @param __v The value to be appended to the tensor.
  /// @return A reference to the tensor after the value has been appended.
  tensor& push_back(value_type __v) const;

  /// @brief Removes the last element from the tensor.
  /// @return A reference to the tensor after the last element has been removed.
  tensor& pop_back() const;

  /// @brief Computes the square root of each element in the tensor, modifying
  /// the tensor in place.
  /// @return A reference to the tensor after the square root has been computed.
  tensor&       sqrt_();
  const tensor& sqrt_() const;

  /// @brief Computes the exponential of each element in the tensor, modifying
  /// the tensor in place.
  /// @return A reference to the tensor after the exponential has been computed.
  tensor&       exp_();
  const tensor& exp_() const;

  /// @brief Computes the base-2 logarithm of each element in the tensor,
  /// modifying the tensor in place.
  /// @return A reference to the tensor after the base-2 logarithm has been
  /// computed.
  tensor&       log2_();
  const tensor& log2_() const;

  /// @brief Computes the base-10 logarithm of each element in the tensor,
  /// modifying the tensor in place.
  /// @return A reference to the tensor after the base-10 logarithm has been
  /// computed.
  tensor&       log10_();
  const tensor& log10_() const;

  /// @brief Computes the natural logarithm (base-e) of each element in the
  /// tensor, modifying the tensor in place.
  /// @return A reference to the tensor after the natural logarithm has been
  /// computed.
  tensor&       log_();
  const tensor& log_() const;

  /// @brief Computes the fractional part of each element in the tensor,
  /// modifying the tensor in place.
  /// @return A reference to the tensor after the fractional part has been
  /// computed.
  tensor&       frac_();
  const tensor& frac_() const;

  /// @brief Computes the element-wise modulus (remainder) of each element in
  /// the tensor with another tensor, modifying the tensor in place.
  /// @param __other The tensor to compute the modulus with.
  /// @return A reference to the tensor after the modulus operation has been
  /// applied.
  tensor&       fmod_(const tensor& __other);
  const tensor& fmod_(const tensor& __other) const;

  /// @brief Computes the element-wise modulus (remainder) of each element in
  /// the tensor with a scalar value, modifying the tensor in place.
  /// @param __val The scalar value to compute the modulus with.
  /// @return A reference to the tensor after the modulus operation has been
  /// applied.
  tensor&       fmod_(const value_type __val);
  const tensor& fmod_(const value_type __val) const;

  /// @brief Computes the cosine of each element in the tensor, modifying the
  /// tensor in place.
  /// @return A reference to the tensor after the cosine has been computed.
  tensor&       cos_();
  const tensor& cos_() const;

  /// @brief Computes the hyperbolic cosine of each element in the tensor,
  /// modifying the tensor in place.
  /// @return A reference to the tensor after the hyperbolic cosine has been
  /// computed.
  tensor&       cosh_();
  const tensor& cosh_() const;

  /// @brief Computes the inverse cosine (arc cosine) of each element in the
  /// tensor, modifying the tensor in place.
  /// @return A reference to the tensor after the inverse cosine has been
  /// computed.
  tensor&       acos_();
  const tensor& acos_() const;

  /// @brief Computes the inverse hyperbolic cosine of each element in the
  /// tensor, modifying the tensor in place.
  /// @return A reference to the tensor after the inverse hyperbolic cosine has
  /// been computed.
  tensor&       acosh_();
  const tensor& acosh_() const;

  /// @brief Computes the tangent of each element in the tensor, modifying the
  /// tensor in place.
  /// @return A reference to the tensor after the tangent has been computed.
  tensor&       tan_();
  const tensor& tan_() const;

  /// @brief Computes the hyperbolic tangent of each element in the tensor,
  /// modifying the tensor in place.
  /// @return A reference to the tensor after the hyperbolic tangent has been
  /// computed.
  tensor&       tanh_();
  const tensor& tanh_() const;

  /// @brief Computes the inverse tangent (arc tangent) of each element in the
  /// tensor, modifying the tensor in place.
  /// @return A reference to the tensor after the inverse tangent has been
  /// computed.
  tensor&       atan_();
  const tensor& atan_() const;

  /// @brief Computes the inverse hyperbolic tangent of each element in the
  /// tensor, modifying the tensor in place.
  /// @return A reference to the tensor after the inverse hyperbolic tangent has
  /// been computed.
  tensor&       atanh_();
  const tensor& atanh_() const;

  /// @brief Computes the sine of each element in the tensor, modifying the
  /// tensor in place.
  /// @return A reference to the tensor after the sine has been computed.
  tensor&       sin_();
  const tensor& sin_() const;

  /// @brief Computes the hyperbolic sine of each element in the tensor,
  /// modifying the tensor in place.
  /// @return A reference to the tensor after the hyperbolic sine has been
  /// computed.
  tensor&       sinh_();
  const tensor& sinh_() const;

  /// @brief Computes the inverse sine (arc sine) of each element in the tensor,
  /// modifying the tensor in place.
  /// @return A reference to the tensor after the inverse sine has been
  /// computed.
  tensor&       asin_();
  const tensor& asin_() const;

  /// @brief Computes the inverse hyperbolic sine of each element in the tensor,
  /// modifying the tensor in place.
  /// @return A reference to the tensor after the inverse hyperbolic sine has
  /// been computed.
  tensor&       asinh_();
  const tensor& asinh_() const;

  /// @brief Applies the ceiling function to each element in the tensor,
  /// modifying the tensor in place.
  /// @return A reference to the tensor after the ceiling function has been
  /// applied.
  tensor&       ceil_();
  const tensor& ceil_() const;

  /// @brief Applies the floor function to each element in the tensor, modifying
  /// the tensor in place.
  /// @return A reference to the tensor after the floor function has been
  /// applied.
  tensor&       floor_();
  const tensor& floor_() const;

  /// @brief Applies the ReLU activation function to each element in the tensor,
  /// modifying the tensor in place.
  /// @return A reference to the tensor after the ReLU function has been
  /// applied.
  tensor&       relu_();
  const tensor& relu_() const;

  /// @brief Clamps each element of the tensor to the given range [min_val,
  /// max_val], modifying the tensor in place.
  /// @param __min_val The minimum value for clamping, or nullptr to skip the
  /// minimum bound.
  /// @param __max_val The maximum value for clamping, or nullptr to skip the
  /// maximum bound.
  /// @return A reference to the tensor after the clamping has been applied.
  tensor&       clamp_(const_reference __min_val = std::numeric_limits<value_type>::lowest(),
                       const_reference __max_val = std::numeric_limits<value_type>::max());
  const tensor& clamp_(const_reference __min_val = std::numeric_limits<value_type>::lowest(),
                       const_reference __max_val = std::numeric_limits<value_type>::max()) const;

  /// @brief Applies bitwise NOT operation to each element in the tensor,
  /// modifying the tensor in place.
  /// @return A reference to the tensor after the bitwise NOT operation has been
  /// applied.
  tensor&       logical_not_();
  const tensor& logical_not_() const;

  /// @brief Performs logical OR between the tensor and another tensor,
  /// modifying the tensor in place.
  /// @param __other The tensor to perform logical OR with.
  /// @return A reference to the tensor after the logical OR operation has been
  /// applied.
  tensor&       logical_or_(const tensor& __other);
  const tensor& logical_or_(const tensor& __other) const;

  /// @brief Performs logical OR between the tensor and a scalar value,
  /// modifying the tensor in place.
  /// @param __val The scalar value to perform logical OR with.
  /// @return A reference to the tensor after the logical OR operation has been
  /// applied.
  tensor&       logical_or_(const value_type __val);
  const tensor& logical_or_(const value_type __val) const;

  /// @brief Performs logical XOR between the tensor and another tensor,
  /// modifying the tensor in place.
  /// @param __other The tensor to perform logical XOR with.
  /// @return A reference to the tensor after the logical XOR operation has been
  /// applied.
  tensor&       logical_xor_(const tensor& __other);
  const tensor& logical_xor_(const tensor& __other) const;

  /// @brief Performs logical XOR between the tensor and a scalar value,
  /// modifying the tensor in place.
  /// @param __val The scalar value to perform logical XOR with.
  /// @return A reference to the tensor after the logical XOR operation has been
  /// applied.
  tensor&       logical_xor_(const value_type __val);
  const tensor& logical_xor_(const value_type __val) const;

  /// @brief Performs logical AND between the tensor and another tensor,
  /// modifying the tensor in place.
  /// @param __other The tensor to perform logical AND with.
  /// @return A reference to the tensor after the logical AND operation has been
  /// applied.
  tensor&       logical_and_(const tensor& __other);
  const tensor& logical_and_(const tensor& __other) const;

  /// @brief Performs logical AND between the tensor and a scalar value,
  /// modifying the tensor in place.
  /// @param __val The scalar value to perform logical AND with.
  /// @return A reference to the tensor after the logical AND operation has been
  /// applied.
  tensor&       logical_and_(const value_type __val);
  const tensor& logical_and_(const value_type __val) const;

  /// @brief Computes the absolute value of each element in the tensor,
  /// modifying the tensor in place.
  /// @return A reference to the tensor after the absolute value has been
  /// computed.
  tensor&       abs_();
  const tensor& abs_() const;

  /// @brief Applies the log softmax function along the specified dimension,
  /// modifying the tensor in place.
  /// @param __dim The dimension along which the log softmax is applied.
  /// @return A reference to the tensor after the log softmax operation has been
  /// applied.
  tensor&       log_softmax_(const index_type __dim);
  const tensor& log_softmax_(const index_type __dim) const;

  /// @brief Permutes the dimensions of the tensor according to the specified
  /// dimension, modifying the tensor in place.
  /// @param __dim The dimension along which the permutation is applied.
  /// @return A reference to the tensor after the permutation has been applied.
  tensor&       permute_(const index_type __dim);
  const tensor& permute_(const index_type __dim) const;

  /// @brief Repeats the tensor according to the specified dimensions, modifying
  /// the tensor in place.
  /// @param __d The dimensions to repeat the tensor along.
  /// @return A reference to the tensor after the repeat operation has been
  /// applied.
  tensor&       repeat_(const data_t& __d, int __dim = 0);
  const tensor& repeat_(const data_t& __d, int __dim = 0) const;

  /// @brief Negates each element in the tensor, modifying the tensor in place.
  /// @return A reference to the tensor after the negation has been applied.
  tensor&       negative_();
  const tensor& negative_() const;

  /// @brief Transposes the tensor, modifying the tensor in place.
  /// @return A reference to the tensor after the transposition has been
  /// applied.
  tensor&       transpose_();
  const tensor& transpose_() const;

  /// @brief Adds an extra dimension at the specified index, modifying the
  /// tensor in place.
  /// @param __dim The index at which to insert the new dimension.
  /// @return A reference to the tensor after the unsqueeze operation has been
  /// applied.
  tensor&       unsqueeze_(index_type __dim);
  const tensor& unsqueeze_(index_type __dim) const;

  /// @brief Removes the dimension at the specified index, modifying the tensor
  /// in place.
  /// @param __dim The index of the dimension to squeeze.
  /// @return A reference to the tensor after the squeeze operation has been
  /// applied.
  tensor&       squeeze_(index_type __dim);
  const tensor& squeeze_(index_type __dim) const;

  /// @brief Resizes the tensor to the specified shape, modifying the tensor in
  /// place.
  /// @param __sh The new shape to resize the tensor to.
  /// @return A reference to the tensor after the resize operation has been
  /// applied.
  tensor&       resize_as_(const shape_type __sh);
  const tensor& resize_as_(const shape_type __sh) const;

  /// @brief Computes the distance between the tensor and another tensor,
  /// modifying the tensor in place.
  /// @param __other The tensor to compute the distance with.
  /// @return A reference to the tensor after the distance operation has been
  /// applied.
  tensor&       dist_(const tensor& __other);
  const tensor& dist_(const tensor& __other) const;

  /// @brief Computes the distance between the tensor and a scalar value,
  /// modifying the tensor in place.
  /// @param __val The scalar value to compute the distance with.
  /// @return A reference to the tensor after the distance operation has been
  /// applied.
  tensor&       dist_(const value_type __val);
  const tensor& dist_(const value_type __val) const;

  /// @brief Computes the element-wise maximum of the tensor and another tensor,
  /// modifying the tensor in place.
  /// @param __other The tensor to compare with.
  /// @return A reference to the tensor after the element-wise maximum operation
  /// has been applied.
  tensor&       maximum_(const tensor& __other);
  const tensor& maximum_(const tensor& __other) const;

  /// @brief Computes the element-wise maximum of the tensor and a scalar value,
  /// modifying the tensor in place.
  /// @param __val The scalar value to compare with.
  /// @return A reference to the tensor after the element-wise maximum operation
  /// has been applied.
  tensor&       maximum_(const value_type __val);
  const tensor& maximum_(const value_type __val) const;

  /// @brief Computes the element-wise remainder of the tensor and a scalar
  /// value, modifying the tensor in place.
  /// @param __val The scalar value to compute the remainder with.
  /// @return A reference to the tensor after the element-wise remainder
  /// operation has been applied.
  tensor&       remainder_(const value_type __val);
  const tensor& remainder_(const value_type __val) const;

  /// @brief Computes the element-wise remainder of the tensor and another
  /// tensor, modifying the tensor in place.
  /// @param __other The tensor to compute the remainder with.
  /// @return A reference to the tensor after the element-wise remainder
  /// operation has been applied.
  tensor&       remainder_(const tensor& __other);
  const tensor& remainder_(const tensor& __other) const;

  /// @brief Fills the tensor with the specified scalar value, modifying the
  /// tensor in place.
  /// @param __val The scalar value to fill the tensor with.
  /// @return A reference to the tensor after the fill operation has been
  /// applied.
  tensor&       fill_(const value_type __val);
  const tensor& fill_(const value_type __val) const;

  /// @brief Fills the tensor with the values from another tensor, modifying the
  /// tensor in place.
  /// @param __other The tensor whose values are used to fill this tensor.
  /// @return A reference to the tensor after the fill operation has been
  /// applied.
  tensor&       fill_(const tensor& __other);
  const tensor& fill_(const tensor& __other) const;

  /// @brief Applies the element-wise sigmoid function to the tensor, modifying
  /// the tensor in place.
  /// @return A reference to the tensor after the sigmoid operation has been
  /// applied.
  tensor&       sigmoid_();
  const tensor& sigmoid_() const;

  /// @brief Applies the element-wise clipped ReLU function to the tensor,
  /// modifying the tensor in place.
  ///        Values exceeding the specified clip limit are clipped.
  /// @param __clip_limit The limit to which values in the tensor are clipped.
  /// @return A reference to the tensor after the clipped ReLU operation has
  /// been applied.
  tensor&       clipped_relu_(const value_type __clip_limit);
  const tensor& clipped_relu_(const value_type __clip_limit) const;

  /// @brief Applies the element-wise square function to the tensor, modifying
  /// the tensor in place.
  /// @return A reference to the tensor after the square operation has been
  /// applied.
  tensor&       square_();
  const tensor& square_() const;

  /// @brief Raises the tensor elements to the power of the elements of another
  /// tensor, modifying the tensor in place.
  /// @param __other The tensor whose elements are used as exponents.
  /// @return A reference to the tensor after the element-wise power operation
  /// has been applied.
  tensor&       pow_(const tensor& __other);
  const tensor& pow_(const tensor& __other) const;

  /// @brief Raises the tensor elements to the power of a scalar value,
  /// modifying the tensor in place.
  /// @param __val The scalar value to which tensor elements are raised.
  /// @return A reference to the tensor after the element-wise power operation
  /// has been applied.
  tensor&       pow_(const value_type __val);
  const tensor& pow_(const value_type __val) const;

  /// @brief Applies the element-wise sinc function to the tensor, modifying the
  /// tensor in place.
  /// @return A reference to the tensor after the sinc operation has been
  /// applied.
  tensor&       sinc_();
  const tensor& sinc_() const;

  /// @brief Performs a bitwise left shift operation on the tensor elements,
  /// modifying the tensor in place.
  /// @param __amount The number of positions to shift the bits.
  /// @return A reference to the tensor after the bitwise left shift operation
  /// has been applied.
  tensor&       bitwise_left_shift_(const int __amount);
  const tensor& bitwise_left_shift_(const int __amount) const;

  /// @brief Performs a bitwise right shift operation on the tensor elements,
  /// modifying the tensor in place.
  /// @param __amount The number of positions to shift the bits.
  /// @return A reference to the tensor after the bitwise right shift operation
  /// has been applied.
  tensor&       bitwise_right_shift_(const int __amount);
  const tensor& bitwise_right_shift_(const int __amount) const;

  /// @brief Performs a bitwise AND operation between the tensor and a scalar
  /// value, modifying the tensor in place.
  /// @param __val The scalar value to apply the bitwise AND operation with.
  /// @return A reference to the tensor after the bitwise AND operation has been
  /// applied.
  tensor&       bitwise_and_(const value_type __val);
  const tensor& bitwise_and_(const value_type __val) const;

  /// @brief Performs a bitwise AND operation between the tensor and another
  /// tensor, modifying the tensor in place.
  /// @param __other The tensor to apply the bitwise AND operation with.
  /// @return A reference to the tensor after the bitwise AND operation has been
  /// applied.
  tensor&       bitwise_and_(const tensor& __other);
  const tensor& bitwise_and_(const tensor& __other) const;

  /// @brief Performs a bitwise OR operation between the tensor and a scalar
  /// value, modifying the tensor in place.
  /// @param __val The scalar value to apply the bitwise OR operation with.
  /// @return A reference to the tensor after the bitwise OR operation has been
  /// applied.
  tensor&       bitwise_or_(const value_type __val);
  const tensor& bitwise_or_(const value_type __val) const;

  /// @brief Performs a bitwise OR operation between the tensor and another
  /// tensor, modifying the tensor in place.
  /// @param __other The tensor to apply the bitwise OR operation with.
  /// @return A reference to the tensor after the bitwise OR operation has been
  /// applied.
  tensor&       bitwise_or_(const tensor& __other);
  const tensor& bitwise_or_(const tensor& __other) const;

  /// @brief Performs a bitwise XOR operation between the tensor and a scalar
  /// value, modifying the tensor in place.
  /// @param __val The scalar value to apply the bitwise XOR operation with.
  /// @return A reference to the tensor after the bitwise XOR operation has been
  /// applied.
  tensor&       bitwise_xor_(const value_type __val);
  const tensor& bitwise_xor_(const value_type __val) const;

  /// @brief Performs a bitwise XOR operation between the tensor and another
  /// tensor, modifying the tensor in place.
  /// @param __other The tensor to apply the bitwise XOR operation with.
  /// @return A reference to the tensor after the bitwise XOR operation has been
  /// applied.
  tensor&       bitwise_xor_(const tensor& __other);
  const tensor& bitwise_xor_(const tensor& __other) const;

  /// @brief Performs a bitwise NOT operation on the tensor elements, modifying
  /// the tensor in place.
  /// @return A reference to the tensor after the bitwise NOT operation has been
  /// applied.
  tensor&       bitwise_not_();
  const tensor& bitwise_not_() const;

  /// @brief Reshapes the tensor to a new shape specified by an initializer list
  /// of dimensions, modifying the tensor in place.
  /// @param __new_sh The new shape of the tensor.
  /// @return A reference to the tensor after the reshape operation has been
  /// applied.
  tensor&       view(std::initializer_list<index_type> __new_sh);
  const tensor& view(std::initializer_list<index_type> __new_sh) const;

  /// @brief Applies the element-wise maximum function between the tensor and
  /// another tensor, modifying the tensor in place.
  /// @param __other The tensor to apply the element-wise maximum operation
  /// with.
  /// @return A reference to the tensor after the element-wise maximum operation
  /// has been applied.
  tensor&       fmax_(const tensor& __other);
  const tensor& fmax_(const tensor& __other) const;

  /// @brief Applies the element-wise maximum function between the tensor and a
  /// scalar value, modifying the tensor in place.
  /// @param __val The scalar value to apply the element-wise maximum operation
  /// with.
  /// @return A reference to the tensor after the element-wise maximum operation
  /// has been applied.
  tensor&       fmax_(const value_type __val);
  const tensor& fmax_(const value_type __val) const;

  /// @brief Randomizes the values in the tensor with the specified shape,
  /// modifying the tensor in place.
  /// @param __sh The shape to which the tensor will be randomized.
  /// @param __bounded If true, the random values will be bounded; otherwise,
  /// they will not.
  /// @return A reference to the tensor after the randomization operation has
  /// been applied.
  tensor&       randomize_(const shape_type& __sh, bool __bounded = false);
  const tensor& randomize_(const shape_type& __sh, bool __bounded = false) const;

  /// @brief Fills the tensor with zeros, modifying the tensor in place.
  /// @param __sh The shape to which the tensor will be resized before being
  /// filled with zeros.
  /// @return A reference to the tensor after the zero-fill operation has been
  /// applied.
  tensor&       zeros_(shape_type __sh = {});
  const tensor& zeros_(shape_type __sh = {}) const;

  /// @brief Fills the tensor with ones, modifying the tensor in place.
  /// @param __sh The shape to which the tensor will be resized before being
  /// filled with ones.
  /// @return A reference to the tensor after the one-fill operation has been
  /// applied.
  tensor&       ones_(shape_type __sh = {});
  const tensor& ones_(shape_type __sh = {}) const;

  void print() const {
    this->printRecursive(0, 0, __shape_);
    std::cout << std::endl;
  }

  tensor<index_type> argmax_(index_type __dim) const;
  tensor<index_type> argsort(index_type __dim = -1, bool __ascending = true) const;

#if defined(__ARM_NEON)
 private:
  tensor&      neon_fmax_(const value_type __v);
  tensor&      neon_fmax_(const tensor& __other);
  tensor&      neon_fmod_(const value_type __val);
  tensor&      neon_fmod_(const tensor& __other);
  tensor&      neon_frac_();
  tensor&      neon_log_();
  tensor&      neon_log10_();
  tensor&      neon_log2_();
  tensor&      neon_exp_();
  tensor&      neon_sqrt_();
  tensor&      neon_cos_();
  tensor&      neon_acos_();
  tensor&      neon_sin_();
  tensor&      neon_tan_();
  tensor&      neon_tanh_();
  tensor&      neon_sinc_();
  tensor&      neon_atan_();
  tensor&      neon_atanh_();
  tensor&      neon_sinh_();
  tensor&      neon_asinh_();
  tensor&      neon_asin_();
  tensor&      neon_cosh_();
  tensor&      neon_acosh_();
  tensor&      neon_pow_(const value_type __val);
  tensor&      neon_pow_(const tensor& __other);
  tensor&      neon_abs_();
  tensor&      neon_dist_(const tensor& __other);
  tensor&      neon_dist_(const value_type __val);
  tensor&      neon_maximum_(const tensor& __other);
  tensor&      neon_maximum_(const value_type __val);
  tensor&      neon_bitwise_right_shift_(const int __amount);
  tensor&      neon_bitwise_left_shift_(const int __amount);
  tensor&      neon_bitwise_or_(const value_type __val);
  tensor&      neon_bitwise_xor_(const value_type __val);
  tensor&      neon_bitwise_not_();
  tensor&      neon_bitwise_and_(const value_type __val);
  tensor&      neon_bitwise_and_(const tensor& __other);
  tensor&      neon_bitwise_or_(const tensor& __other);
  tensor&      neon_bitwise_xor_(const tensor& __other);
  tensor&      neon_zeros_(shape_type __sh = {});
  tensor&      neon_ones_(shape_type __sh);
  tensor&      neon_randomize_(const shape_type& __sh, bool __bounded);
  tensor&      neon_negative_();
  tensor&      neon_relu_();
  tensor&      neon_sigmoid_();
  tensor&      neon_clipped_relu_(const value_type __clip_limit);
  tensor&      neon_clamp_(const_reference __min_val = std::numeric_limits<value_type>::lowest(),
                           const_reference __max_val = std::numeric_limits<value_type>::max());
  tensor&      neon_floor_();
  tensor&      neon_ceil_();
  tensor&      neon_logical_or_(const value_type __val);
  tensor&      neon_logical_xor_(const value_type __val);
  tensor&      neon_logical_and_(const value_type __val);
  tensor&      neon_logical_or_(const tensor& __other);
  tensor&      neon_logical_xor_(const tensor& __other);
  tensor&      neon_logical_and_(const tensor& __other);
  tensor&      neon_operator_plus_eq(const_reference __val) const;
  tensor&      neon_operator_minus_eq(const tensor& __other) const;
  tensor&      neon_operator_times_eq(const tensor& __other) const;
  tensor&      neon_operator_minus_eq(const_reference __val) const;
  tensor<_s32> neon_int32_() const;
  tensor<_u32> neon_uint32_() const;
  tensor<_f32> neon_float32_() const;
  tensor<_f64> neon_double_() const;
  tensor<uint64_t>   neon_unsigned_long_() const;
  tensor<int64_t>    neon_long_() const;
  tensor             neon_operator_plus(const tensor& __other) const;
  tensor             neon_operator_plus(const value_type __val) const;
  tensor             neon_operator_minus(const tensor& __other) const;
  tensor             neon_operator_minus(const value_type __val) const;
  tensor             neon_transpose() const;
  tensor             neon_matmul(const tensor& __other) const;
  tensor             neon_absolute(const tensor& __tensor) const;
  tensor             neon_cross_product(const tensor& __other) const;
  tensor             neon_dot(const tensor& __other) const;
  tensor             neon_argmax(index_type __dim) const;
  tensor             neon_sum(const index_type __axis) const;
  tensor             neon_slice(index_type __dim, std::optional<index_type> __start,
                                std::optional<index_type> __end, index_type __step) const;
  tensor<bool>       neon_equal(const tensor& __other) const;
  tensor<bool>       neon_equal(const value_type __val) const;
  tensor<bool>       neon_less_equal(const tensor& __other) const;
  tensor<bool>       neon_less_equal(const value_type __val) const;
  tensor<index_type> neon_argsort(index_type __d, bool __ascending) const;
  tensor<index_type> neon_argmax_(index_type __dim) const;
  index_type         neon_count_nonzero(index_type __dim) const;
  double             neon_mean() const;
#endif

 private:
  [[nodiscard]] size_t computeStride(size_t dim, const shape_type& shape) const noexcept;
  void                 printRecursive(size_t index, size_t depth, const shape_type& shape) const;
  void                 __compute_strides();
  [[nodiscard]] index_type        __compute_index(const std::vector<index_type>& __idx) const;
  [[nodiscard]] static index_type __computeSize(const shape_type& __dims) noexcept;
  index_type                      __compute_outer_size(const index_type __dim) const;
  [[nodiscard]] static _f32       __frac(const_reference __val) noexcept;
  // where the tensor is stored
  bool __is_cuda_device() const;
  bool __equal_shape(const shape_type& __x, const shape_type& __y) const;

};  // tensor class

template <class _Tp>
bool tensor<_Tp>::__equal_shape(const shape_type& __x, const shape_type& __y) const {
  size_t __size_x = __x.size();
  size_t __size_y = __y.size();

  if (__size_x == __size_y) return __x == __y;

  if (__size_x < __size_y) return __equal_shape(__y, __x);

  int __diff = __size_x - __size_y;
  for (size_t __i = 0; __i < __size_y; ++__i)
    if (__x[__i + __diff] != __y[__i] && __x[__i + __diff] != 1 && __y[__i] != 1) return false;

  return true;
}

template <class _Tp>
inline bool tensor<_Tp>::__is_cuda_device() const {
  return (this->__device_ == Device::CUDA);
}

template <class _Tp>
[[nodiscard]]
inline _f32 tensor<_Tp>::__frac(const_reference __val) noexcept {
  return std::fmod(static_cast<float32_t>(__val), 1.0f);
}

template <class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::__compute_outer_size(
    const index_type __dim) const {
  // just a placeholder for now
  return 0;
}

template <class _Tp>
[[nodiscard]]
inline
    typename tensor<_Tp>::index_type tensor<_Tp>::__computeSize(const shape_type& __dims) noexcept {
  index_type __ret = 1;
  for (const index_type& __d : __dims) __ret *= __d;
  return __ret;
}

template <class _Tp>
[[nodiscard]]
inline typename tensor<_Tp>::index_type tensor<_Tp>::__compute_index(
    const std::vector<index_type>& __idx) const {
  if (__idx.size() != this->__shape_.size())
    throw std::out_of_range("__compute_index : input indices does not match the tensor shape");

  index_type __index = 0;
  index_type __i     = 0;

  for (; __i < this->__shape_.size(); ++__i) __index += __idx[__i] * this->__strides_[__i];

  return __index;
}

template <class _Tp>
void tensor<_Tp>::__compute_strides() {
  if (this->__shape_.empty()) {
    std::cerr << "Shape must be initialized before computing strides" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  this->__strides_ = shape_type(this->__shape_.size(), 1);
  int __st = 1, __i = static_cast<int>(this->__shape_.size() - 1);

  for (; __i >= 0; __i--) {
    this->__strides_[__i] = __st;
    __st *= this->__shape_[__i];
  }
}

template <class _Tp>
void tensor<_Tp>::printRecursive(size_t __index, size_t __depth, const shape_type& __shape) const {
  if (__depth == __shape.size() - 1) {
    std::cout << "[";

    for (size_t __i = 0; __i < __shape[__depth]; ++__i) {
      if constexpr (std::is_floating_point_v<_Tp>)
        std::cout << std::fixed << std::setprecision(4) << __data_[__index + __i];
      else
        std::cout << __data_[__index + __i];

      if (__i < __shape[__depth] - 1) std::cout << ", ";
    }
    std::cout << "]";
  } else {
    std::cout << "[\n";
    size_t __stride = computeStride(__depth + 1, __shape);

    for (size_t __i = 0; __i < __shape[__depth]; ++__i) {
      if (__i > 0) std::cout << "\n";

      for (size_t __j = 0; __j < __depth + 1; ++__j) std::cout << " ";

      printRecursive(__index + __i * __stride, __depth + 1, __shape);

      if (__i < __shape[__depth] - 1) std::cout << ",";
    }
    std::cout << "\n";
    for (size_t __j = 0; __j < __depth; ++__j) std::cout << " ";
    std::cout << "]";
  }
}

template <class _Tp>
[[nodiscard]]
inline size_t tensor<_Tp>::computeStride(size_t __dim, const shape_type& __shape) const noexcept {
  size_t __stride = 1;
  for (size_t __i = __dim; __i < __shape.size(); ++__i) __stride *= __shape[__i];
  return __stride;
}

template <>
class tensor<bool>;

template <>
class tensor<bool> {
 public:
  using __self                 = tensor;
  using value_type             = bool;
  using data_t                 = std::vector<bool>;
  using index_type             = uint64_t;
  using shape_type             = std::vector<index_type>;
  using reference              = typename data_t::reference;
  using iterator               = typename data_t::iterator;
  using reverse_iterator       = typename data_t::reverse_iterator;
  using const_iterator         = typename data_t::const_iterator;
  using const_reference        = typename data_t::const_reference;
  using const_reverse_iterator = typename data_t::const_reverse_iterator;

  enum class Device { CPU, CUDA };

 private:
  mutable data_t     __data_;
  mutable shape_type __shape_;
  mutable shape_type __strides_;
  Device             __device_;
  bool               __is_cuda_tensor_ = false;

 public:
  tensor() = default;

  explicit tensor(const shape_type& __sh, value_type __v, Device __d = Device::CPU)
      : __shape_(__sh), __data_(this->__computeSize(__sh), __v), __device_(__d) {
    this->__compute_strides();
  }

  explicit tensor(const shape_type& __sh, Device __d = Device::CPU)
      : __shape_(__sh), __device_(__d) {
    index_type __s = this->__computeSize(__sh);
    this->__data_  = data_t(__s);
    this->__compute_strides();
  }

  explicit tensor(const shape_type& __sh, const data_t& __d, Device __dev = Device::CPU)
      : __shape_(__sh), __device_(__dev) {
    index_type __s = this->__computeSize(__sh);
    assert(__d.size() == static_cast<size_t>(__s) &&
           "Initial data vector must match the tensor size");
    this->__data_ = __d;
    this->__compute_strides();
  }

  tensor(const tensor& __t)
      : __data_(__t.storage()),
        __shape_(__t.shape()),
        __strides_(__t.strides()),
        __device_(__t.device()) {}

  tensor(tensor&& __t) noexcept
      : __data_(std::move(__t.storage())),
        __shape_(std::move(__t.shape())),
        __strides_(std::move(__t.strides())),
        __device_(std::move(__t.device())) {}

  tensor(const shape_type& __sh, std::initializer_list<value_type> init_list,
         Device __d = Device::CPU)
      : __shape_(__sh), __device_(__d) {
    index_type __s = this->__computeSize(__sh);
    assert(init_list.size() == static_cast<size_t>(__s) &&
           "Initializer list size must match tensor size");
    this->__data_ = data_t(init_list);
    this->__compute_strides();
  }

  tensor(const shape_type& __sh, const tensor& __other)
      : __data_(__other.storage()), __shape_(__sh), __device_(__other.device()) {
    this->__compute_strides();
  }

  data_t storage() const noexcept { return this->__data_; }

  shape_type shape() const noexcept { return this->__shape_; }

  shape_type strides() const noexcept { return this->__strides_; }

  Device device() const noexcept { return this->__device_; }

  bool operator==(const tensor& __other) const {
    return __equal_shape(this->shape(), __other.shape()) && this->__data_ == __other.storage();
  }

  bool operator!=(const tensor& __other) const { return !(*this == __other); }

  tensor<bool>& operator=(const tensor<bool>& __other) const noexcept;

  reference at(shape_type __idx) {
    if (__idx.empty())
      throw std::invalid_argument("Passing an empty vector as indices for a tensor");

    index_type __i = this->__compute_index(__idx);
    if (__i < 0 || __i >= this->__data_.size())
      throw std::invalid_argument("input indices are out of bounds");

    return this->__data_[__i];
  }

  reference operator[](const index_type __idx) {
    if (__idx < 0 || __idx >= this->__data_.size())
      throw std::invalid_argument("input index is out of bounds");

    return this->__data_[__idx];
  }

  const_reference at(const shape_type __idx) const { return this->at(__idx); }

  const_reference operator[](const index_type __idx) const { return (*this)[__idx]; }

  reference operator()(std::initializer_list<index_type> __index_list) {
    return this->at(shape_type(__index_list));
  }

  const_reference operator()(std::initializer_list<index_type> __index_list) const {
    return this->at(shape_type(__index_list));
  }

  bool empty() const { return this->__data_.empty(); }

  size_t n_dims() const noexcept { return this->__shape_.size(); }

  index_type size(const index_type __dim) const {
    if (__dim < 0 || __dim > static_cast<index_type>(this->__shape_.size()))
      throw std::invalid_argument("dimension input is out of range");

    if (__dim == 0) return this->__data_.size();

    return this->__shape_[__dim - 1];
  }

  index_type capacity() const noexcept { return this->__data_.capacity(); }

  tensor<bool> logical_not() const { return tensor<bool>(this->logical_not_()); }

  tensor<bool> logical_or(const value_type __val) const {
    return tensor<bool>(this->logical_or_(__val));
  }

  tensor<bool> logical_or(const tensor& __other) const {
    return tensor<bool>(this->logical_or_(__other));
  }

  tensor<bool> logical_and(const value_type __val) const {
    return tensor<bool>(this->logical_and_(__val));
  }

  tensor<bool> logical_and(const tensor& __other) const {
    return tensor<bool>(this->logical_and_(__other));
  }

  tensor<bool> logical_xor(const value_type __val) const {
    return tensor<bool>(this->logical_xor_(__val));
  }

  tensor<bool> logical_xor(const tensor& __other) const {
    return tensor<bool>(this->logical_xor_(__other));
  }

  tensor<bool>& logical_not_() {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = ~(this->__data_[__i]);
    return *this;
  }

  tensor<bool>& logical_or_(const value_type __val) {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = this->__data_[__i] || __val;
    return *this;
  }

  tensor<bool>& logical_or_(const tensor& __other) {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i)
      this->__data_[__i] = this->__data_[__i] || __other[__i];
    return *this;
  }

  tensor<bool>& logical_and_(const value_type __val) {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = this->__data_[__i] && __val;
    return *this;
  }

  tensor<bool>& logical_and_(const tensor& __other) {
    index_type __i = 0;
    for (; __i < this->__data_.size(); ++__i)
      this->__data_[__i] = this->__data_[__i] && __other[__i];
    return *this;
  }

  tensor<bool>& logical_xor_(const value_type __val) {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = this->__data_[__i] ^ __val;
    return *this;
  }

  tensor<bool>& logical_xor_(const tensor& __other) {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i)
      this->__data_[__i] = this->__data_[__i] ^ __other[__i];
    return *this;
  }

  const tensor<bool>& logical_not_() const {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = ~(this->__data_[__i]);
    return *this;
  }

  const tensor<bool>& logical_or_(const value_type __val) const {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = this->__data_[__i] || __val;
    return *this;
  }

  const tensor<bool>& logical_or_(const tensor& __other) const {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i)
      this->__data_[__i] = this->__data_[__i] || __other[__i];
    return *this;
  }

  const tensor<bool>& logical_and_(const value_type __val) const {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = this->__data_[__i] && __val;
    return *this;
  }

  const tensor<bool>& logical_and_(const tensor& __other) const {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i)
      this->__data_[__i] = this->__data_[__i] && __other[__i];
    return *this;
  }

  const tensor<bool>& logical_xor_(const value_type __val) const {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = this->__data_[__i] ^ __val;
    return *this;
  }

  const tensor<bool>& logical_xor_(const tensor& __other) const {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i)
      this->__data_[__i] = this->__data_[__i] ^ __other[__i];
    return *this;
  }

  tensor<bool>& operator!() {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = !(this->__data_[__i]);
    return *this;
  }

  const tensor<bool>& operator!() const {
    index_type __i = 0;
#pragma omp parallel
    for (; __i < this->__data_.size(); ++__i) this->__data_[__i] = !(this->__data_[__i]);
    return *this;
  }

  tensor<bool> slice(index_type __dim, std::optional<index_type> __start,
                     std::optional<index_type> __end, index_type __step) const {
    if (__dim < 0 || __dim >= static_cast<index_type>(this->__data_.size()))
      throw std::out_of_range("Dimension out of range");

    tensor<bool> __ret;
    index_type   __s       = this->__shape_[__dim];
    index_type   __start_i = __start.value_or(0);
    index_type   __end_i   = __end.value_or(0);

    if (__start_i < 0) __start_i += __s;
    if (__end_i < 0) __end_i += __s;

    __start_i = std::max(index_type(0), std::min(__start_i, __s));
    __end_i   = std::max(index_type(0), std::min(__end_i, __s));

    index_type __slice_size = (__end_i - __start_i + __step - 1) / __step;
    shape_type __ret_dims   = this->__shape_;

    __ret_dims[-__dim] = __slice_size;
    __ret              = __self(__ret_dims);

    index_type __i = __start_i, __j = 0;
#pragma omp parallel
    for (; __i < __end_i; __i += __step, ++__j) __ret[__j] = this->__data_[__j];

    return __ret;
  }

  tensor<bool> row(const index_type __index) const {
    if (this->__shape_.size() != 2)
      throw std::runtime_error("Cannot get a row from a non two dimensional tensor");

    if (this->__shape_[0] <= __index || __index < 0)
      throw std::invalid_argument("Index input is out of range");

    index_type __start = this->__shape_[1] * __index;
    index_type __end   = this->__shape_[1] * __index + this->__shape_[1];
    index_type __i     = __start;
    data_t     __r(__end);
#pragma omp parallel
    for (; __i < __end; ++__i) __r[__i] = this->__data_[__i];

    return __self({this->__shape_[1]}, __r);
  }

  tensor<bool> col(const index_type __index) const {
    if (this->__shape_.size() != 2)
      throw std::runtime_error("Cannot get a column from a non two dimensional tensor");

    if (this->__shape_[1] <= __index || __index < 0)
      throw std::invalid_argument("Index input out of range");

    data_t     __c(this->__shape_[0]);
    index_type __i = 0;
    for (; __i < this->__shape_[0]; ++__i)
      __c[__i] = this->__data_[this->__compute_index({__i, __index})];

    return __self({this->__shape_[0]}, __c);
  }

  tensor<bool> clone() const {
    data_t     __d = this->__data_;
    shape_type __s = this->__shape_;
    return __self(__s, __d);
  }

  tensor<bool> reshape(const shape_type __sh) const {
    data_t     __d = this->__data_;
    index_type __s = this->__computeSize(__sh);

    if (__s != this->__data_.size())
      throw std::invalid_argument(
          "input shape must have size of element equal to the current number of elements in the"
          "tensor data");

    return __self(__sh, __d);
  }

  tensor<bool> reshape_as(const tensor& __other) const { return this->reshape(__other.shape()); }

  tensor<bool> transpose() const {
    if (__equal_shape(this->__shape_, shape_type({this->__shape_[0], this->__shape_[1], 1}))
        || __equal_shape(this->__shape_, shape_type({1, this->__shape_[0], this->__shape_[1]}))) 
      throw std::invalid_argument("Matrix transposition can only be done on 2D tensors");

    tensor           __ret({this->__shape_[1], this->__shape_[0]});
    const index_type __rows = this->__shape_[0];
    const index_type __cols = this->__shape_[1];

    index_type __i = 0;
    for (; __i < __rows; ++__i) {
      index_type __j = 0;
      for (; __j < __cols; ++__j) __ret.at({__j, __i}) = this->at({__i, __j});
    }

    return __ret;
  }

  tensor<bool>& transpose_() {
    if (this->__shape_.size() != 2)
      throw std::runtime_error("Transpose operation is only valid for 2D tensors");

    const index_type __r = this->__shape_[0];
    const index_type __c = this->__shape_[1];

    if (__r != __c)
      throw std::runtime_error("In-place transpose is only supported for square tensors");

    for (index_type __i = 0; __i < __r; ++__i)
      for (index_type __j = __i + 1; __j < __c; ++__j)
        std::swap(this->__data_[__i * __c + __j], this->__data_[__j * __c + __i]);

    return *this;
  }

  const tensor<bool>& transpose_() const {
    if (this->__shape_.size() != 2)
      throw std::runtime_error("Transpose operation is only valid for 2D tensors");

    const index_type __r = this->__shape_[0];
    const index_type __c = this->__shape_[1];

    if (__r != __c)
      throw std::runtime_error("In-place transpose is only supported for square tensors");

    for (index_type __i = 0; __i < __r; ++__i)
      for (index_type __j = __i + 1; __j < __c; ++__j)
        std::swap(this->__data_[__i * __c + __j], this->__data_[__j * __c + __i]);

    return *this;
  }

  tensor<bool> resize_as(const shape_type __sh) const {
    __self __ret = this->clone();
    __ret.resize_as_(__sh);
    return __ret;
  }

  tensor<bool>& resize_as_(const shape_type __sh) { return *this; }

  tensor<bool> squeeze(const index_type __dim) const {
    __self __ret = this->clone();
    __ret.squeeze_(__dim);
    return __ret;
  }

  tensor<bool>& squeeze_(const index_type __dim) { return *this; }

  tensor<bool> repeat(const data_t& __d, int __dim) const {
    __self __ret = this->clone();
    __ret.repeat_(__d, __dim);
    return __ret;
  }

  tensor<bool>& repeat_(const data_t& __d, int __dim) {
    if (__d.empty()) throw std::invalid_argument("Cannot repeat an empty tensor");

    if (this->size(0) < __d.size()) this->__data_ = data_t(__d.begin(), __d.end());

    size_t __start      = 0;
    size_t __end        = __d.size();
    size_t __total_size = this->size(0);

    if (__total_size < __d.size()) return *this;

    unsigned int __nbatches  = __total_size / __d.size();
    size_t       __remainder = __total_size % __d.size();

    for (unsigned int __i = 0; __i < __nbatches; ++__i) {
      for (size_t __j = __start, __k = 0; __k < __d.size(); ++__j, ++__k)
        this->__data_[__j] = __d[__k];

      __start += __d.size();
    }
#pragma omp parallel
    for (size_t __j = __start, __k = 0; __j < __total_size && __k < __remainder; ++__j, ++__k)
      this->__data_[__j] = __d[__k];

    return *this;
  }

  tensor<bool> permute(const index_type __dim) const {
    // TODO : implement permute here
    data_t __d;
    return __self(this->__shape_, __d);
  }

  tensor<bool> cat(const std::vector<tensor<value_type>>& __others, index_type __dim) const {
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

    for (const tensor& __t : __others)
      __c.insert(__c.end(), __t.__data_.begin(), __t.__data_.end());

    return __self(__ret_sh, __c);
  }

  tensor<bool> unsqueeze(const index_type __dim) const {
    if (__dim < 0 || __dim > static_cast<index_type>(this->__shape_.size()))
      throw std::out_of_range("Dimension out of range in unsqueeze");

    shape_type __s = this->__shape_;
    __s.insert(__s.begin() + __dim, 1);

    tensor __ret;
    __ret.__shape_ = __s;
    __ret.__data_  = this->__data_;

    return __ret;
  }

  tensor<bool>& randomize_(const shape_type& __sh = {}) {
    if (__sh.empty() && this->__shape_.empty())
      throw std::invalid_argument("Shape must be initialized");

    if (this->__shape_.empty() || this->__shape_ != __sh) this->__shape_ = __sh;

    index_type __s = this->__computeSize(this->__shape_);
    this->__data_.resize(__s);
    this->__compute_strides();

    std::random_device                 __rd;
    std::mt19937                       __gen(__rd());
    std::uniform_int_distribution<int> __dist(0, 1);

    index_type __i = 0;
#pragma omp parallel
    for (; __i < static_cast<index_type>(__s); ++__i) this->__data_[__i] = (__dist(__gen) == 0);

    return *this;
  }

  tensor<bool>& push_back(value_type __v) {
    if (this->__shape_.size() != 1)
      throw std::range_error("push_back is only supported for one dimensional tensors");

    this->__data_.push_back(__v);
    ++(this->__shape_[0]);
    this->__compute_strides();
    return *this;
  }

  tensor<bool>& pop_back(value_type __v) {
    if (this->__shape_.size() != 1)
      throw std::range_error("push_back is only supported for one dimensional tensors");

    this->__data_.pop_back();
    --(this->__shape_[0]);
    this->__compute_strides();
    return *this;
  }

  tensor<bool>& view(std::initializer_list<index_type> __sh) {
    index_type __s = this->__computeSize(__sh);

    if (__s != this->__data_.size())
      throw std::invalid_argument("Total elements do not match for new shape");

    this->__shape_ = __sh;
    this->__compute_strides();
    return *this;
  }

  void print() const {
    this->printRecursive(0, 0, __shape_);
    std::cout << std::endl;
  }

 private:
  bool __equal_shape(const shape_type& __x, const shape_type& __y) const {
    size_t __size_x = __x.size();
    size_t __size_y = __y.size();

    if (__size_x == __size_y) return __x == __y;

    if (__size_x < __size_y) return __equal_shape(__y, __x);

    int __diff = __size_x - __size_y;
    for (size_t __i = 0; __i < __size_y; ++__i)
      if (__x[__i + __diff] != __y[__i] && __x[__i + __diff] != 1 && __y[__i] != 1) return false;

    return true;
  }

  [[nodiscard]]
  inline size_t computeStride(size_t __dim, const shape_type& __shape) const noexcept {
    size_t __stride = 1;
    for (size_t __i = __dim; __i < __shape.size(); __i++) __stride *= __shape[__i];
    return __stride;
  }

  void printRecursive(size_t __index, size_t __depth, const shape_type& __shape) const {
    if (__depth == __shape.size() - 1) {
      std::cout << "[";

      for (size_t __i = 0; __i < __shape[__depth]; ++__i) {
        std::cout << (this->__data_[__index + __i] ? "true" : "false");

        if (__i < __shape[__depth] - 1) std::cout << ", ";
      }
      std::cout << "]";
    } else {
      std::cout << "[\n";
      size_t __stride = computeStride(__depth + 1, __shape);

      for (size_t __i = 0; __i < __shape[__depth]; ++__i) {
        if (__i > 0) std::cout << "\n";

        for (size_t __j = 0; __j < __depth + 1; __j++) std::cout << " ";

        printRecursive(__index + __i * __stride, __depth + 1, __shape);

        if (__i < __shape[__depth] - 1) std::cout << ",";
      }
      std::cout << "\n";
      for (size_t __j = 0; __j < __depth; __j++) std::cout << " ";
      std::cout << "]";
    }
  }

  void __compute_strides() {
    if (this->__shape_.empty()) {
      std::cerr << "Shape must be initialized before computing strides" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    this->__strides_ = shape_type(this->__shape_.size(), 1);
    int __st = 1, __i = static_cast<int>(this->__shape_.size() - 1);
    for (; __i >= 0; __i--) {
      this->__strides_[__i] = __st;
      __st *= this->__shape_[__i];
    }
  }

  [[nodiscard]]
  index_type __compute_index(const std::vector<index_type>& __idx) const {
    if (__idx.size() != this->__shape_.size())
      throw std::out_of_range("__compute_index : input indices does not match the tensor shape");

    index_type __index = 0;
    index_type __i     = 0;
    for (; __i < this->__shape_.size(); ++__i) __index += __idx[__i] * this->__strides_[__i];

    return __index;
  }

  [[nodiscard]]
  static index_type __computeSize(const shape_type& __dims) noexcept {
    uint64_t __ret = 1;
    for (const index_type& __d : __dims) __ret *= __d;
    return __ret;
  }

  index_type __compute_outer_size(const index_type __dim) const {
    // just a placeholder for now
    return 0;
  }

  [[nodiscard]]
  static _f32 __frac(const_reference __val) noexcept {
    return std::fmod(static_cast<float32_t>(__val), 1.0f);
  }

  bool __is_cuda_device() const { return (this->__device_ == Device::CUDA); }
};