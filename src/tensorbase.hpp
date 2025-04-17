//
// tensor base
//

#pragma once

#include <__algorithm/clamp.h>
#include <__algorithm/comp.h>
#include <__algorithm/sort.h>
#include <__functional/hash.h>
#include <arm_neon.h>
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
#include <type_traits>
#include <typeinfo>
#include <valarray>
#include <vector>
#include <version>

#include "error.hpp"

#if defined(USE_CUDA)
    #include <cuda_runtime.h>
#endif

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

const int _ARM64_REG_WIDTH = 128;  // 128 bit wide register

template<class _Tp>
class tensor
{
   public:
    using self                   = tensor;
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

    enum class Device {
        CPU,
        CUDA
    };

   private:
    mutable data_t     data_;
    mutable shape_type shape_;
    mutable shape_type strides_;
    Device             device_;
    bool               is_cuda_tensor_ = false;

   public:
    tensor() = default;
    explicit tensor(const shape_type& sh, const_reference v, Device d = Device::CPU);
    explicit tensor(const shape_type& sh, Device d = Device::CPU);
    explicit tensor(const shape_type& sh, const data_t& d, Device dev = Device::CPU);
    tensor(const tensor& t);
    tensor(tensor&& t) noexcept;
    tensor(const shape_type& sh, const tensor& other);
    tensor(const shape_type& sh, std::initializer_list<value_type> init_list, Device d = Device::CPU);

   private:
    class destroy_tensor
    {
       private:
        tensor& tens_;

       public:
        explicit destroy_tensor(tensor& tens) :
            tens_(tens) {}

        void operator()() {}
    };

   public:
    ~tensor() { destroy_tensor (*this)(); }

    tensor<short>         short_() const;
    tensor<long long>     long_long_() const;
    tensor<long>          long_() const;
    tensor<int>           int_() const;
    tensor<unsigned int>  unsigned_int_() const;
    tensor<unsigned long> unsigned_long_() const;
    tensor<float>         float_() const;
    tensor<double>        double_() const;

    data_t          storage() const noexcept;
    shape_type      shape() const noexcept;
    shape_type      strides() const noexcept;
    Device          device() const noexcept;
    std::size_t     n_dims() const noexcept;
    index_type      size(const index_type dim) const;
    index_type      capacity() const noexcept;
    index_type      count_nonzero(index_type dim = 0) const;
    index_type      lcm() const;
    index_type      hash() const;
    reference       at(shape_type idx);
    reference       operator[](const index_type idx);
    const_reference at(const shape_type idx) const;
    const_reference operator[](const index_type idx) const;
    reference       operator()(std::initializer_list<index_type> index_list);
    const_reference operator()(std::initializer_list<index_type> index_list) const;

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
    /// @param val The scalar value to compute the logical OR with.
    /// @return A tensor of boolean values.
    tensor<bool> logical_or(const value_type val) const;

    /// @brief Computes the logical OR operation between corresponding elements of
    /// two tensors.
    /// @param other The tensor to compute the logical OR with.
    /// @return A tensor of boolean values.
    tensor<bool> logical_or(const tensor& other) const;

    /// @brief Checks if each element is less than or equal to the corresponding
    /// element in another tensor.
    /// @param other The tensor to compare with.
    /// @return A tensor of boolean values.
    tensor<bool> less_equal(const tensor& other) const;

    /// @brief Checks if each element is less than or equal to a scalar value.
    /// @param val The scalar value to compare with.
    /// @return A tensor of boolean values.
    tensor<bool> less_equal(const value_type val) const;

    /// @brief Checks if each element is greater than or equal to the
    /// corresponding element in another tensor.
    /// @param other The tensor to compare with.
    /// @return A tensor of boolean values.
    tensor<bool> greater_equal(const tensor& other) const;

    /// @brief Checks if each element is greater than or equal to a scalar value.
    /// @param val The scalar value to compare with.
    /// @return A tensor of boolean values.
    tensor<bool> greater_equal(const value_type val) const;

    /// @brief Checks if each element is equal to the corresponding element in
    /// another tensor.
    /// @param other The tensor to compare with.
    /// @return A tensor of boolean values.
    tensor<bool> equal(const tensor& other) const;

    /// @brief Checks if each element is equal to a scalar value.
    /// @param val The scalar value to compare with.
    /// @return A tensor of boolean values.
    tensor<bool> equal(const value_type val) const;

    /// @brief Checks if each element is not equal to the corresponding element in
    /// another tensor.
    /// @param other The tensor to compare with.
    /// @return A tensor of boolean values.
    tensor<bool> not_equal(const tensor& other) const;

    /// @brief Checks if each element is not equal to a scalar value.
    /// @param val The scalar value to compare with.
    /// @return A tensor of boolean values.
    tensor<bool> not_equal(const value_type val) const;

    /// @brief Checks if each element is less than the corresponding element in
    /// another tensor.
    /// @param other The tensor to compare with.
    /// @return A tensor of boolean values.
    tensor<bool> less(const tensor& other) const;

    /// @brief Checks if each element is less than a scalar value.
    /// @param val The scalar value to compare with.
    /// @return A tensor of boolean values.
    tensor<bool> less(const value_type val) const;

    /// @brief Checks if each element is greater than the corresponding element in
    /// another tensor.
    /// @param other The tensor to compare with.
    /// @return A tensor of boolean values.
    tensor<bool> greater(const tensor& other) const;

    /// @brief Checks if each element is greater than a scalar value.
    /// @param val The scalar value to compare with.
    /// @return A tensor of boolean values.
    tensor<bool> greater(const value_type val) const;

    /// @brief Compares this tensor with another tensor for equality.
    /// @param other The tensor to compare with.
    /// @return True if the tensors are equal, false otherwise.
    bool operator==(const tensor& other) const;

    /// @brief Compares this tensor with another tensor for inequality.
    /// @param other The tensor to compare with.
    /// @return True if the tensors are not equal, false otherwise.
    bool operator!=(const tensor& other) const;

    /// @brief Adds the elements of another tensor to this tensor.
    /// @param other The tensor to add.
    /// @return A new tensor containing the result of the addition.
    tensor operator+(const tensor& other) const;

    /// @brief Subtracts the elements of another tensor from this tensor.
    /// @param other The tensor to subtract.
    /// @return A new tensor containing the result of the subtraction.
    tensor operator-(const tensor& other) const;

    /// @brief Adds a scalar value to each element of the tensor.
    /// @param val The scalar value to add.
    /// @return A new tensor containing the result of the addition.
    tensor operator+(const value_type val) const;

    /// @brief Subtracts a scalar value from each element of the tensor.
    /// @param val The scalar value to subtract.
    /// @return A new tensor containing the result of the subtraction.
    tensor operator-(const value_type val) const;

    /// @brief Multiply a scalar value with each element of the tensor.
    /// @param val The scalar value to multiply with.
    /// @return A new tensor containing the result of the multiplication.
    tensor operator*(const value_type val) const;

    /// @brief Multiply each element of the input tensor with each element of the
    /// tensor.
    /// @param other The tensor to multiply with.
    /// @return A new tensor containing the result of the multiplication.
    tensor operator*(const tensor& other) const;

    /// @brief Subtracts the elements of another tensor from this tensor and
    /// updates the current tensor.
    /// @param other The tensor to subtract.
    /// @return A \ref to the current tensor after the subtraction.
    tensor& operator-=(const tensor& other) const;

    /// @brief Adds the elements of another tensor to this tensor and updates the
    /// current tensor.
    /// @param other The tensor to add.
    /// @return A \ref to the current tensor after the addition.
    tensor& operator+=(const tensor& other) const;

    /// @brief Multiplies the elements of this tensor by the elements of another
    /// tensor and updates the current tensor.
    /// @param other The tensor to multiply with.
    /// @return A \ref to the current tensor after the multiplication.
    tensor& operator*=(const tensor& other) const;

    /// @brief Divides the elements of this tensor by the elements of another
    /// tensor and updates the current tensor.
    /// @param other The tensor to divide by.
    /// @return A \ref to the current tensor after the division.
    tensor& operator/=(const tensor& other) const;

    /// @brief Adds a scalar value to each element of the tensor and updates the
    /// current tensor.
    /// @param val The scalar value to add.
    /// @return A \ref to the current tensor after the addition.
    tensor& operator+=(const_reference val) const;

    /// @brief Subtracts a scalar value from each element of the tensor and
    /// updates the current tensor.
    /// @param val The scalar value to subtract.
    /// @return A \ref to the current tensor after the subtraction.
    tensor& operator-=(const_reference val) const;

    /// @brief Divides each element of the tensor by a scalar value and updates
    /// the current tensor.
    /// @param val The scalar value to divide by.
    /// @return A \ref to the current tensor after the division.
    tensor& operator/=(const_reference val) const;

    /// @brief Divides each element of the tensor by the corresponding element of
    /// the other tensor
    /// @param other The other tensor to divide by (should not contain a zero)
    /// @return A tensor the contains the result of the division
    tensor operator/(const tensor& other) const;

    /// @brief Divides each element of the tensor by a scalar value
    /// @param val The scalar value to divide by (non zero)
    /// @return A tensor the contains the result of the division
    tensor operator/(const_reference val) const;

    /// @brief Multiplies each element of the tensor by a scalar value and updates
    /// the current tensor.
    /// @param val The scalar value to multiply with.
    /// @return A \ref to the current tensor after the multiplication.
    tensor& operator*=(const_reference val) const;

    /// @brief Assigns the values of another tensor to this tensor.
    /// @param other The tensor to assign from.
    /// @return A \ref to the current tensor after the assignment.
    tensor& operator=(const tensor& other) const;

    /// @brief Moves the values of another tensor to this tensor.
    /// @param other The tensor to move from.
    /// @return A \ref to the current tensor after the move.
    tensor& operator=(tensor&& other) const noexcept;

    /// @brief Assigns the values of another tensor to this tensor (default
    /// implementation).
    /// @param other The tensor to assign from.
    /// @return A \ref to the current tensor after the assignment.
    tensor& operator=(const tensor&) = default;

    tensor<bool>&       operator!();
    const tensor<bool>& operator!() const;
    /*
    /// @brief Extracts a slice of the tensor along the specified dimension.
    /// @param dim The dimension along which to slice.
    /// @param start The starting index of the slice (optional).
    /// @param end The ending index of the slice (optional).
    /// @param step The step size for the slice.
    /// @return A new tensor containing the sliced data.
    tensor slice(index_type dim, std::optional<index_type> start, std::optional<index_type> end, int64_t step) const;
*/

    /// @brief Computes the element-wise maximum between this tensor and another
    /// tensor.
    /// @param other The tensor to compare with.
    /// @return A new tensor containing the element-wise maximum values.
    tensor fmax(const tensor& other) const;

    /// @brief Computes the element-wise maximum between this tensor and a scalar
    /// value.
    /// @param val The scalar value to compare with.
    /// @return A new tensor containing the element-wise maximum values.
    tensor fmax(const value_type val) const;

    /// @brief Computes the element-wise floating-point remainder of division with
    /// another tensor.
    /// @param other The tensor to divide by.
    /// @return A new tensor containing the element-wise remainders.
    tensor fmod(const tensor& other) const;

    /// @brief Computes the element-wise floating-point remainder of division with
    /// a scalar value.
    /// @param val The scalar value to divide by.
    /// @return A new tensor containing the element-wise remainders.
    tensor fmod(const value_type val) const;

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
    /// @param axis The axis along which to compute the sum.
    /// @return A new tensor containing the sums along the specified axis.
    tensor sum(const index_type axis) const;

    /// @brief Extracts a specific row from the tensor.
    /// @param index The index of the row to extract.
    /// @return A new tensor containing the specified row.
    tensor row(const index_type index) const;

    /// @brief Extracts a specific column from the tensor.
    /// @param index The index of the column to extract.
    /// @return A new tensor containing the specified column.
    tensor col(const index_type index) const;

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
    /// @param min_val A pointer to the minimum value (optional).
    /// @param max_val A pointer to the maximum value (optional).
    /// @return A new tensor with values clamped to the specified range.
    tensor clamp(const_reference min_val = std::numeric_limits<value_type>::lowest(),
                 const_reference max_val = std::numeric_limits<value_type>::max()) const;

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
    /// @param other The tensor to apply the logical XOR operation with.
    /// @return A new tensor containing the result of the logical XOR operation.
    tensor logical_xor(const tensor& other) const;

    /// @brief Performs a logical XOR operation between the tensor and a scalar
    /// value.
    /// @param val The scalar value to apply the logical XOR operation with.
    /// @return A new tensor containing the result of the logical XOR operation.
    tensor logical_xor(const value_type val) const;

    /// @brief Performs a logical AND operation between the tensor and another
    /// tensor.
    /// @param other The tensor to apply the logical AND operation with.
    /// @return A new tensor containing the result of the logical AND operation.
    tensor logical_and(const tensor& other) const;

    /// @brief Performs a logical AND operation between the tensor and a scalar
    /// value.
    /// @param val The scalar value to apply the logical AND operation with.
    /// @return A new tensor containing the result of the logical AND operation.
    tensor logical_and(const value_type val) const;

    /// @brief Performs a bitwise NOT operation on each element of the tensor.
    /// @return A new tensor containing the result of the bitwise NOT operation.
    tensor bitwise_not() const;

    /// @brief Performs a bitwise AND operation between the tensor and a scalar
    /// value.
    /// @param val The scalar value to apply the bitwise AND operation with.
    /// @return A new tensor containing the result of the bitwise AND operation.
    tensor bitwise_and(const value_type val) const;

    /// @brief Performs a bitwise AND operation between the tensor and another
    /// tensor.
    /// @param other The tensor to apply the bitwise AND operation with.
    /// @return A new tensor containing the result of the bitwise AND operation.
    tensor bitwise_and(const tensor& other) const;

    /// @brief Performs a bitwise OR operation between the tensor and a scalar
    /// value.
    /// @param val The scalar value to apply the bitwise OR operation with.
    /// @return A new tensor containing the result of the bitwise OR operation.
    tensor bitwise_or(const value_type val) const;

    /// @brief Performs a bitwise OR operation between the tensor and another
    /// tensor.
    /// @param other The tensor to apply the bitwise OR operation with.
    /// @return A new tensor containing the result of the bitwise OR operation.
    tensor bitwise_or(const tensor& other) const;

    /// @brief Performs a bitwise XOR operation between the tensor and a scalar
    /// value.
    /// @param val The scalar value to apply the bitwise XOR operation with.
    /// @return A new tensor containing the result of the bitwise XOR operation.
    tensor bitwise_xor(const value_type val) const;

    /// @brief Performs a bitwise XOR operation between the tensor and another
    /// tensor.
    /// @param other The tensor to apply the bitwise XOR operation with.
    /// @return A new tensor containing the result of the bitwise XOR operation.
    tensor bitwise_xor(const tensor& other) const;

    /// @brief Performs a bitwise left shift operation on each element of the
    /// tensor by a specified amount.
    /// @param amount The number of bits to shift left.
    /// @return A new tensor containing the result of the bitwise left shift
    /// operation.
    tensor bitwise_left_shift(const int amount) const;

    /// @brief Performs a bitwise right shift operation on each element of the
    /// tensor by a specified amount.
    /// @param amount The number of bits to shift right.
    /// @return A new tensor containing the result of the bitwise right shift
    /// operation.
    tensor bitwise_right_shift(const int amount) const;

    /// @brief Performs matrix multiplication between the tensor and another
    /// tensor.
    /// @param other The tensor to multiply with.
    /// @return A new tensor containing the result of the matrix multiplication.
    tensor matmul(const tensor& other) const;

    /// @brief Reshapes the tensor to a specified shape.
    /// @param shape The desired shape for the tensor.
    /// @return A new tensor with the specified shape.
    tensor reshape(const shape_type shape) const;

    /// @brief Reshapes the tensor to match the shape of another tensor.
    /// @param other The tensor whose shape is to be matched.
    /// @return A new tensor reshaped to match the shape of the given tensor.
    tensor reshape_as(const tensor& other) const;

    /// @brief Computes the cross product between the tensor and another tensor.
    /// @param other The tensor to compute the cross product with.
    /// @return A new tensor containing the cross product.
    tensor cross_product(const tensor& other) const;

    /// @brief Computes the absolute value of each element in a given tensor.
    /// @param tensor The tensor to compute the absolute values of.
    /// @return A new tensor containing the absolute values of the input tensor.
    tensor absolute(const tensor& other) const;

    /// @brief Computes the dot product between the tensor and another tensor.
    /// @param other The tensor to compute the dot product with.
    /// @return A scalar tensor containing the result of the dot product.
    tensor dot(const tensor& other) const;

    /// @brief Applies the ReLU activation function to the tensor.
    /// @return A new tensor where negative values are replaced with zero.
    tensor relu() const;

    /// @brief Transposes the tensor, swapping rows and columns.
    /// @return A new tensor that is the transpose of the original tensor.
    tensor transpose() const;

    /// @brief Fills the tensor with a specified scalar value.
    /// @param val The scalar value to fill the tensor with.
    /// @return A new tensor filled with the specified value.
    tensor fill(const value_type val) const;

    /// @brief Fills the tensor with the values from another tensor.
    /// @param other The tensor whose values are to be used for filling.
    /// @return A new tensor filled with the values from the specified tensor.
    tensor fill(const tensor& other) const;

    /// @brief Resizes the tensor to a specified shape, keeping its data
    /// consistent.
    /// @param sh The desired shape for the tensor.
    /// @return A new tensor resized to the specified shape.
    tensor resize_as(const shape_type sh) const;

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
    tensor clipped_relu(const value_type clip_limit) const;

    /// @brief Sorts the elements of the tensor along a specified dimension.
    /// @param dim The dimension along which to sort the tensor.
    /// @param descending Whether to sort in descending order (default is
    /// false).
    /// @return A new tensor with the sorted elements.
    tensor sort(index_type dim, bool descending = false) const;

    /// @brief Computes the remainder of division between each element and a
    /// scalar value.
    /// @param val The scalar value to divide by.
    /// @return A new tensor containing the remainders.
    tensor remainder(const value_type val) const;

    /// @brief Computes the remainder of division between each element of the
    /// tensor and another tensor.
    /// @param other The tensor to divide by.
    /// @return A new tensor containing the remainders.
    tensor remainder(const tensor& other) const;

    /// @brief Computes the element-wise maximum between the tensor and another
    /// tensor.
    /// @param other The tensor to compare with.
    /// @return A new tensor containing the element-wise maximum values.
    tensor maximum(const tensor& other) const;

    /// @brief Computes the element-wise maximum between the tensor and a scalar
    /// value.
    /// @param val The scalar value to compare with.
    /// @return A new tensor containing the element-wise maximum values.
    tensor maximum(const_reference val) const;

    /// @brief Computes the distance between the tensor and another tensor.
    /// @param other The tensor to compute the distance from.
    /// @return A scalar tensor containing the computed distance.
    tensor dist(const tensor& other) const;

    /// @brief Computes the distance between the tensor and a scalar value.
    /// @param val The scalar value to compute the distance from.
    /// @return A scalar tensor containing the computed distance.
    tensor dist(const value_type val) const;

    /// @brief Removes dimensions of size 1 from the specified axis of the tensor.
    /// @param dim The dimension to squeeze.
    /// @return A new tensor with the specified dimension removed if its size
    /// is 1.
    tensor squeeze(index_type dim) const;

    /// @brief Computes the element-wise negation of the tensor.
    /// @return A new tensor with all elements negated.
    tensor negative() const;

    /// @brief Repeats the tensor along specified dimensions.
    /// @param d A data structure specifying the number of repetitions along
    /// each dimension.
    /// @return A new tensor with repeated elements along the specified
    /// dimensions.
    tensor repeat(const data_t& d, int dim = 0) const;

    /// @brief Permutes the dimensions of the tensor.
    /// @param dim The order of dimensions for permutation.
    /// @return A new tensor with permuted dimensions.
    tensor permute(const index_type dim) const;

    /// @brief Computes the log softmax along a specified dimension of the tensor.
    /// @param dim The dimension along which to compute the log softmax.
    /// @return A new tensor containing the log softmax values.
    tensor log_softmax(const index_type dim) const;

    /// @brief Computes the element-wise greatest common divisor (GCD) between the
    /// tensor and another tensor.
    /// @param other The tensor to compute the GCD with.
    /// @return A new tensor containing the element-wise GCD values.
    tensor gcd(const tensor& other) const;

    /// @brief Computes the element-wise greatest common divisor (GCD) between the
    /// tensor and a scalar value.
    /// @param val The scalar value to compute the GCD with.
    /// @return A new tensor containing the element-wise GCD values.
    tensor gcd(const value_type val) const;

    /// @brief Computes the element-wise power of the tensor, raising each element
    /// to the power of the corresponding element in another tensor.
    /// @param other The tensor representing the exponents.
    /// @return A new tensor containing the element-wise power values.
    tensor pow(const tensor& other) const;

    /// @brief Computes the element-wise power of the tensor, raising each element
    /// to the power of a scalar value.
    /// @param val The scalar exponent value.
    /// @return A new tensor containing the element-wise power values.
    tensor pow(const value_type val) const;

    /// @brief Computes the cumulative product of elements along a specified
    /// dimension.
    /// @param dim The dimension along which to compute the cumulative product.
    /// Default is -1 (last dimension).
    /// @return A new tensor containing the cumulative product along the specified
    /// dimension.
    tensor cumprod(index_type dim = -1) const;

    /// @brief Concatenates the tensor with a list of other tensors along a
    /// specified dimension.
    /// @param _others A vector of tensors to concatenate.
    /// @param _dim The dimension along which to concatenate the tensors.
    /// @return A new tensor resulting from the concatenation.
    tensor cat(const std::vector<tensor>& _others, index_type _dim) const;

    /// @brief Finds the indices of the maximum values along a specified
    /// dimension.
    /// @param dim The dimension along which to compute the indices of the
    /// maximum values.
    /// @return A new tensor containing the indices of the maximum values along
    /// the specified dimension.
    tensor argmax(index_type dim) const;

    /// @brief Adds a dimension of size 1 at the specified axis.
    /// @param dim The dimension where the new axis will be added.
    /// @return A new tensor with the added dimension.
    tensor unsqueeze(index_type dim) const;

    /// @brief Creates a tensor filled with zeros, with the specified shape.
    /// @param sh The shape of the tensor to be created.
    /// @return A new tensor filled with zeros.
    tensor zeros(const shape_type& sh);

    /// @brief Creates a tensor filled with ones, with the specified shape.
    /// @param sh The shape of the tensor to be created.
    /// @return A new tensor filled with ones.
    tensor ones(const shape_type& sh);

    /// @brief Creates a tensor filled with random values, with the specified
    /// shape.
    /// @param sh The shape of the tensor to be created.
    /// @param bounded Whether the random values should be bounded (default is
    /// false).
    /// @return A new tensor filled with random values.
    tensor randomize(const shape_type& sh, bool bounded = false);

    /// @brief Extracts the minor matrix by removing the specified row and column.
    /// @param a The row index to remove.
    /// @param b The column index to remove.
    /// @return A new tensor representing the minor matrix.
    tensor get_minor(index_type a, index_type b) const;

    /// @brief Expands the tensor to match the shape of another tensor along a
    /// specified dimension.
    /// @param sh The target shape for expansion.
    /// @param dim The dimension along which the tensor will be expanded.
    /// @return A new tensor expanded to the specified shape.
    tensor expand_as(shape_type sh, index_type dim) const;

    /// @brief Computes the element-wise least common multiple (LCM) with another
    /// tensor.
    /// @param other The tensor to compute the LCM with.
    /// @return A new tensor containing the element-wise LCM values.
    tensor lcm(const tensor& other) const;

    /// @brief Computes the mean (average) of all elements in the tensor.
    /// @return The mean value of all elements in the tensor.
    double mean() const;

    /// @brief Computes the median of the elements along a specified dimension.
    /// @param dim The dimension along which to compute the median.
    /// @return The median value of the elements along the specified dimension.
    double median(const index_type dim) const;

    /// @brief Computes the mode (most frequent value) of the elements along a
    /// specified dimension.
    /// @param dim The dimension along which to compute the mode.
    /// @return The mode value of the elements along the specified dimension.
    double mode(const index_type dim) const;

    /// @brief Appends a value to the end of the tensor.
    /// @param v The value to be appended to the tensor.
    /// @return A \ref to the tensor after the value has been appended.
    tensor& push_back(value_type v) const;

    /// @brief Removes the last element from the tensor.
    /// @return A \ref to the tensor after the last element has been removed.

    tensor& pop_back() const;

    /// @brief Computes the square root of each element in the tensor, modifying
    /// the tensor in place.
    /// @return A \ref to the tensor after the square root has been computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       sqrt_();
    const tensor& sqrt_() const;

    /// @brief Computes the exponential of each element in the tensor, modifying
    /// the tensor in place.
    /// @return A \ref to the tensor after the exponential has been computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       exp_();
    const tensor& exp_() const;

    /// @brief Computes the base-2 logarithm of each element in the tensor,
    /// modifying the tensor in place.
    /// @return A \ref to the tensor after the base-2 logarithm has been
    /// computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       log2_();
    const tensor& log2_() const;

    /// @brief Computes the base-10 logarithm of each element in the tensor,
    /// modifying the tensor in place.
    /// @return A \ref to the tensor after the base-10 logarithm has been
    /// computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       log10_();
    const tensor& log10_() const;

    /// @brief Computes the natural logarithm (base-e) of each element in the
    /// tensor, modifying the tensor in place.
    /// @return A \ref to the tensor after the natural logarithm has been
    /// computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       log_();
    const tensor& log_() const;

    /// @brief Computes the fractional part of each element in the tensor,
    /// modifying the tensor in place.
    /// @return A \ref to the tensor after the fractional part has been
    /// computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       frac_();
    const tensor& frac_() const;

    /// @brief Computes the element-wise modulus (remainder) of each element in
    /// the tensor with another tensor, modifying the tensor in place.
    /// @param other The tensor to compute the modulus with.
    /// @return A \ref to the tensor after the modulus operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       fmod_(const tensor& other);
    const tensor& fmod_(const tensor& other) const;

    /// @brief Computes the element-wise modulus (remainder) of each element in
    /// the tensor with a scalar value, modifying the tensor in place.
    /// @param val The scalar value to compute the modulus with.
    /// @return A \ref to the tensor after the modulus operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       fmod_(const value_type val);
    const tensor& fmod_(const value_type val) const;

    /// @brief Computes the cosine of each element in the tensor, modifying the
    /// tensor in place.
    /// @return A \ref to the tensor after the cosine has been computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       cos_();
    const tensor& cos_() const;

    /// @brief Computes the hyperbolic cosine of each element in the tensor,
    /// modifying the tensor in place.
    /// @return A \ref to the tensor after the hyperbolic cosine has been
    /// computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       cosh_();
    const tensor& cosh_() const;

    /// @brief Computes the inverse cosine (arc cosine) of each element in the
    /// tensor, modifying the tensor in place.
    /// @return A \ref to the tensor after the inverse cosine has been
    /// computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       acos_();
    const tensor& acos_() const;

    /// @brief Computes the inverse hyperbolic cosine of each element in the
    /// tensor, modifying the tensor in place.
    /// @return A \ref to the tensor after the inverse hyperbolic cosine has
    /// been computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       acosh_();
    const tensor& acosh_() const;

    /// @brief Computes the tangent of each element in the tensor, modifying the
    /// tensor in place.
    /// @return A \ref to the tensor after the tangent has been computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.tan_();`
    /// The tensor is still modified in place.
    tensor&       tan_();
    const tensor& tan_() const;

    /// @brief Computes the hyperbolic tangent of each element in the tensor,
    /// modifying the tensor in place.
    /// @return A \ref to the tensor after the hyperbolic tangent has been
    /// computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       tanh_();
    const tensor& tanh_() const;

    /// @brief Computes the inverse tangent (arc tangent) of each element in the
    /// tensor, modifying the tensor in place.
    /// @return A \ref to the tensor after the inverse tangent has been
    /// computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       atan_();
    const tensor& atan_() const;

    /// @brief Computes the inverse hyperbolic tangent of each element in the
    /// tensor, modifying the tensor in place.
    /// @return A \ref to the tensor after the inverse hyperbolic tangent has
    /// been computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       atanh_();
    const tensor& atanh_() const;

    /// @brief Computes the sine of each element in the tensor, modifying the
    /// tensor in place.
    /// @return A \ref to the tensor after the sine has been computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       sin_();
    const tensor& sin_() const;

    /// @brief Computes the hyperbolic sine of each element in the tensor,
    /// modifying the tensor in place.
    /// @return A \ref to the tensor after the hyperbolic sine has been
    /// computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       sinh_();
    const tensor& sinh_() const;

    /// @brief Computes the inverse sine (arc sine) of each element in the tensor,
    /// modifying the tensor in place.
    /// @return A \ref to the tensor after the inverse sine has been
    /// computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       asin_();
    const tensor& asin_() const;

    /// @brief Computes the inverse hyperbolic sine of each element in the tensor,
    /// modifying the tensor in place.
    /// @return A \ref to the tensor after the inverse hyperbolic sine has
    /// been computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       asinh_();
    const tensor& asinh_() const;

    /// @brief Applies the ceiling function to each element in the tensor,
    /// modifying the tensor in place.
    /// @return A \ref to the tensor after the ceiling function has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       ceil_();
    const tensor& ceil_() const;

    /// @brief Applies the floor function to each element in the tensor, modifying
    /// the tensor in place.
    /// @return A \ref to the tensor after the floor function has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       floor_();
    const tensor& floor_() const;

    /// @brief Applies the ReLU activation function to each element in the tensor,
    /// modifying the tensor in place.
    /// @return A \ref to the tensor after the ReLU function has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       relu_();
    const tensor& relu_() const;

    /// @brief Clamps each element of the tensor to the given range [min_val,
    /// max_val], modifying the tensor in place.
    /// @param min_val The minimum value for clamping, or nullptr to skip the
    /// minimum bound.
    /// @param max_val The maximum value for clamping, or nullptr to skip the
    /// maximum bound.
    /// @return A \ref to the tensor after the clamping has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       clamp_(const_reference min_val = std::numeric_limits<value_type>::lowest(),
                         const_reference max_val = std::numeric_limits<value_type>::max());
    const tensor& clamp_(const_reference min_val = std::numeric_limits<value_type>::lowest(),
                         const_reference max_val = std::numeric_limits<value_type>::max()) const;

    tensor        clamp_min(const_reference min_val) const;
    tensor&       clamp_min_(const_reference min_val);
    const tensor& clamp_min_(const_reference min_val) const;

    tensor        clamp_max(const_reference max_val) const;
    tensor&       clamp_max_(const_reference max_val);
    const tensor& clamp_max_(const_reference max_val) const;

    /// @brief Applies bitwise NOT operation to each element in the tensor,
    /// modifying the tensor in place.
    /// @return A \ref to the tensor after the bitwise NOT operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       logical_not_();
    const tensor& logical_not_() const;

    /// @brief Performs logical OR between the tensor and another tensor,
    /// modifying the tensor in place.
    /// @param other The tensor to perform logical OR with.
    /// @return A \ref to the tensor after the logical OR operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       logical_or_(const tensor& other);
    const tensor& logical_or_(const tensor& other) const;

    /// @brief Performs logical OR between the tensor and a scalar value,
    /// modifying the tensor in place.
    /// @param val The scalar value to perform logical OR with.
    /// @return A \ref to the tensor after the logical OR operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       logical_or_(const value_type val);
    const tensor& logical_or_(const value_type val) const;

    /// @brief Performs logical XOR between the tensor and another tensor,
    /// modifying the tensor in place.
    /// @param other The tensor to perform logical XOR with.
    /// @return A \ref to the tensor after the logical XOR operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       logical_xor_(const tensor& other);
    const tensor& logical_xor_(const tensor& other) const;

    /// @brief Performs logical XOR between the tensor and a scalar value,
    /// modifying the tensor in place.
    /// @param val The scalar value to perform logical XOR with.
    /// @return A \ref to the tensor after the logical XOR operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       logical_xor_(const value_type val);
    const tensor& logical_xor_(const value_type val) const;

    /// @brief Performs logical AND between the tensor and another tensor,
    /// modifying the tensor in place.
    /// @param other The tensor to perform logical AND with.
    /// @return A \ref to the tensor after the logical AND operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       logical_and_(const tensor& other);
    const tensor& logical_and_(const tensor& other) const;

    /// @brief Performs logical AND between the tensor and a scalar value,
    /// modifying the tensor in place.
    /// @param val The scalar value to perform logical AND with.
    /// @return A \ref to the tensor after the logical AND operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       logical_and_(const value_type val);
    const tensor& logical_and_(const value_type val) const;

    /// @brief Computes the absolute value of each element in the tensor,
    /// modifying the tensor in place.
    /// @return A \ref to the tensor after the absolute value has been
    /// computed.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       abs_();
    const tensor& abs_() const;

    /// @brief Applies the log softmax function along the specified dimension,
    /// modifying the tensor in place.
    /// @param dim The dimension along which the log softmax is applied.
    /// @return A \ref to the tensor after the log softmax operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       log_softmax_(const index_type dim);
    const tensor& log_softmax_(const index_type dim) const;

    /// @brief Permutes the dimensions of the tensor according to the specified
    /// dimension, modifying the tensor in place.
    /// @param dim The dimension along which the permutation is applied.
    /// @return A \ref to the tensor after the permutation has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       permute_(const index_type dim);
    const tensor& permute_(const index_type dim) const;

    /// @brief Repeats the tensor according to the specified dimensions, modifying
    /// the tensor in place.
    /// @param d The dimensions to repeat the tensor along.
    /// @return A \ref to the tensor after the repeat operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       repeat_(const data_t& d, int dim = 0);
    const tensor& repeat_(const data_t& d, int dim = 0) const;

    /// @brief Negates each element in the tensor, modifying the tensor in place.
    /// @return A \ref to the tensor after the negation has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       negative_();
    const tensor& negative_() const;

    /// @brief Transposes the tensor, modifying the tensor in place.
    /// @return A \ref to the tensor after the transposition has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       transpose_();
    const tensor& transpose_() const;

    /// @brief Adds an extra dimension at the specified index, modifying the
    /// tensor in place.
    /// @param dim The index at which to insert the new dimension.
    /// @return A \ref to the tensor after the unsqueeze operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       unsqueeze_(index_type dim);
    const tensor& unsqueeze_(index_type dim) const;

    /// @brief Removes the dimension at the specified index, modifying the tensor
    /// in place.
    /// @param dim The index of the dimension to squeeze.
    /// @return A \ref to the tensor after the squeeze operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       squeeze_(index_type dim);
    const tensor& squeeze_(index_type dim) const;

    /// @brief Resizes the tensor to the specified shape, modifying the tensor in
    /// place.
    /// @param sh The new shape to resize the tensor to.
    /// @return A \ref to the tensor after the resize operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       resize_as_(const shape_type sh);
    const tensor& resize_as_(const shape_type sh) const;

    /// @brief Computes the distance between the tensor and another tensor,
    /// modifying the tensor in place.
    /// @param other The tensor to compute the distance with.
    /// @return A \ref to the tensor after the distance operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       dist_(const tensor& other);
    const tensor& dist_(const tensor& other) const;

    /// @brief Computes the distance between the tensor and a scalar value,
    /// modifying the tensor in place.
    /// @param val The scalar value to compute the distance with.
    /// @return A \ref to the tensor after the distance operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       dist_(const value_type val);
    const tensor& dist_(const value_type val) const;

    /// @brief Computes the element-wise maximum of the tensor and another tensor,
    /// modifying the tensor in place.
    /// @param other The tensor to compare with.
    /// @return A \ref to the tensor after the element-wise maximum operation
    /// has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       maximum_(const tensor& other);
    const tensor& maximum_(const tensor& other) const;

    /// @brief Computes the element-wise maximum of the tensor and a scalar value,
    /// modifying the tensor in place.
    /// @param val The scalar value to compare with.
    /// @return A \ref to the tensor after the element-wise maximum operation
    /// has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       maximum_(const value_type val);
    const tensor& maximum_(const value_type val) const;

    /// @brief Computes the element-wise remainder of the tensor and a scalar
    /// value, modifying the tensor in place.
    /// @param val The scalar value to compute the remainder with.
    /// @return A \ref to the tensor after the element-wise remainder
    /// operation has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       remainder_(const value_type val);
    const tensor& remainder_(const value_type val) const;

    /// @brief Computes the element-wise remainder of the tensor and another
    /// tensor, modifying the tensor in place.
    /// @param other The tensor to compute the remainder with.
    /// @return A \ref to the tensor after the element-wise remainder
    /// operation has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       remainder_(const tensor& other);
    const tensor& remainder_(const tensor& other) const;

    /// @brief Fills the tensor with the specified scalar value, modifying the
    /// tensor in place.
    /// @param val The scalar value to fill the tensor with.
    /// @return A \ref to the tensor after the fill operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       fill_(const value_type val);
    const tensor& fill_(const value_type val) const;

    /// @brief Fills the tensor with the values from another tensor, modifying the
    /// tensor in place.
    /// @param other The tensor whose values are used to fill this tensor.
    /// @return A \ref to the tensor after the fill operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       fill_(const tensor& other);
    const tensor& fill_(const tensor& other) const;

    /// @brief Applies the element-wise sigmoid function to the tensor, modifying
    /// the tensor in place.
    /// @return A \ref to the tensor after the sigmoid operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       sigmoid_();
    const tensor& sigmoid_() const;

    /// @brief Applies the element-wise clipped ReLU function to the tensor,
    /// modifying the tensor in place.
    ///        Values exceeding the specified clip limit are clipped.
    /// @param clip_limit The limit to which values in the tensor are clipped.
    /// @return A \ref to the tensor after the clipped ReLU operation has
    /// been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       clipped_relu_(const value_type clip_limit);
    const tensor& clipped_relu_(const value_type clip_limit) const;

    /// @brief Applies the element-wise square function to the tensor, modifying
    /// the tensor in place.
    /// @return A \ref to the tensor after the square operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       square_();
    const tensor& square_() const;

    /// @brief Raises the tensor elements to the power of the elements of another
    /// tensor, modifying the tensor in place.
    /// @param other The tensor whose elements are used as exponents.
    /// @return A \ref to the tensor after the element-wise power operation
    /// has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       pow_(const tensor& other);
    const tensor& pow_(const tensor& other) const;

    /// @brief Raises the tensor elements to the power of a scalar value,
    /// modifying the tensor in place.
    /// @param val The scalar value to which tensor elements are raised.
    /// @return A \ref to the tensor after the element-wise power operation
    /// has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       pow_(const value_type val);
    const tensor& pow_(const value_type val) const;

    /// @brief Applies the element-wise sinc function to the tensor, modifying the
    /// tensor in place.
    /// @return A \ref to the tensor after the sinc operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       sinc_();
    const tensor& sinc_() const;

    /// @brief Performs a bitwise left shift operation on the tensor elements,
    /// modifying the tensor in place.
    /// @param amount The number of positions to shift the bits.
    /// @return A \ref to the tensor after the bitwise left shift operation
    /// has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       bitwise_left_shift_(const int amount);
    const tensor& bitwise_left_shift_(const int amount) const;

    /// @brief Performs a bitwise right shift operation on the tensor elements,
    /// modifying the tensor in place.
    /// @param amount The number of positions to shift the bits.
    /// @return A \ref to the tensor after the bitwise right shift operation
    /// has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       bitwise_right_shift_(const int amount);
    const tensor& bitwise_right_shift_(const int amount) const;

    /// @brief Performs a bitwise AND operation between the tensor and a scalar
    /// value, modifying the tensor in place.
    /// @param val The scalar value to apply the bitwise AND operation with.
    /// @return A \ref to the tensor after the bitwise AND operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       bitwise_and_(const value_type val);
    const tensor& bitwise_and_(const value_type val) const;

    /// @brief Performs a bitwise AND operation between the tensor and another
    /// tensor, modifying the tensor in place.
    /// @param other The tensor to apply the bitwise AND operation with.
    /// @return A \ref to the tensor after the bitwise AND operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       bitwise_and_(const tensor& other);
    const tensor& bitwise_and_(const tensor& other) const;

    /// @brief Performs a bitwise OR operation between the tensor and a scalar
    /// value, modifying the tensor in place.
    /// @param val The scalar value to apply the bitwise OR operation with.
    /// @return A \ref to the tensor after the bitwise OR operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       bitwise_or_(const value_type val);
    const tensor& bitwise_or_(const value_type val) const;

    /// @brief Performs a bitwise OR operation between the tensor and another
    /// tensor, modifying the tensor in place.
    /// @param other The tensor to apply the bitwise OR operation with.
    /// @return A \ref to the tensor after the bitwise OR operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       bitwise_or_(const tensor& other);
    const tensor& bitwise_or_(const tensor& other) const;

    /// @brief Performs a bitwise XOR operation between the tensor and a scalar
    /// value, modifying the tensor in place.
    /// @param val The scalar value to apply the bitwise XOR operation with.
    /// @return A \ref to the tensor after the bitwise XOR operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       bitwise_xor_(const value_type val);
    const tensor& bitwise_xor_(const value_type val) const;

    /// @brief Performs a bitwise XOR operation between the tensor and another
    /// tensor, modifying the tensor in place.
    /// @param other The tensor to apply the bitwise XOR operation with.
    /// @return A \ref to the tensor after the bitwise XOR operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       bitwise_xor_(const tensor& other);
    const tensor& bitwise_xor_(const tensor& other) const;

    /// @brief Performs a bitwise NOT operation on the tensor elements, modifying
    /// the tensor in place.
    /// @return A \ref to the tensor after the bitwise NOT operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       bitwise_not_();
    const tensor& bitwise_not_() const;

    /// @brief Reshapes the tensor to a new shape specified by an initializer list
    /// of dimensions, modifying the tensor in place.
    /// @param new_sh The new shape of the tensor.
    /// @return A \ref to the tensor after the reshape operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       view(std::initializer_list<index_type> new_sh);
    const tensor& view(std::initializer_list<index_type> new_sh) const;

    /// @brief Applies the element-wise maximum function between the tensor and
    /// another tensor, modifying the tensor in place.
    /// @param other The tensor to apply the element-wise maximum operation
    /// with.
    /// @return A \ref to the tensor after the element-wise maximum operation
    /// has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       fmax_(const tensor& other);
    const tensor& fmax_(const tensor& other) const;

    /// @brief Applies the element-wise maximum function between the tensor and a
    /// scalar value, modifying the tensor in place.
    /// @param val The scalar value to apply the element-wise maximum operation
    /// with.
    /// @return A \ref to the tensor after the element-wise maximum operation
    /// has been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       fmax_(const value_type val);
    const tensor& fmax_(const value_type val) const;

    /// @brief Randomizes the values in the tensor with the specified shape,
    /// modifying the tensor in place.
    /// @param sh The shape to which the tensor will be randomized.
    /// @param bounded If true, the random values will be bounded; otherwise,
    /// they will not.
    /// @return A \ref to the tensor after the randomization operation has
    /// been applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       randomize_(const shape_type& sh, bool bounded = false);
    const tensor& randomize_(const shape_type& sh, bool bounded = false) const;

    /// @brief Fills the tensor with zeros, modifying the tensor in place.
    /// @param sh The shape to which the tensor will be resized before being
    /// filled with zeros.
    /// @return A \ref to the tensor after the zero-fill operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       zeros_(shape_type sh = {});
    const tensor& zeros_(shape_type sh = {}) const;

    /// @brief Fills the tensor with ones, modifying the tensor in place.
    /// @param sh The shape to which the tensor will be resized before being
    /// filled with ones.
    /// @return A \ref to the tensor after the one-fill operation has been
    /// applied.
    /// @note This `const` overload allows chaining and usage like:
    /// `const auto& result = tensor.asin_();`
    /// The tensor is still modified in place.
    tensor&       ones_(shape_type sh = {});
    const tensor& ones_(shape_type sh = {}) const;

    void print() const {
        printRecursive(0, 0, shape_);
        std::cout << std::endl;
    }

    tensor<index_type> argmax_(index_type dim) const;
    tensor<index_type> argsort(index_type dim = -1, bool ascending = true) const;

   private:
    tensor&          neon_fmax_(const value_type v);
    tensor&          neon_fmax_(const tensor& other);
    tensor&          neon_fmod_(const value_type val);
    tensor&          neon_fmod_(const tensor& other);
    tensor&          neon_frac_();
    tensor&          neon_log_();
    tensor&          neon_log10_();
    tensor&          neon_log2_();
    tensor&          neon_exp_();
    tensor&          neon_sqrt_();
    tensor&          neon_cos_();
    tensor&          neon_acos_();
    tensor&          neon_sin_();
    tensor&          neon_tan_();
    tensor&          neon_tanh_();
    tensor&          neon_sinc_();
    tensor&          neon_atan_();
    tensor&          neon_atanh_();
    tensor&          neon_sinh_();
    tensor&          neon_asinh_();
    tensor&          neon_asin_();
    tensor&          neon_cosh_();
    tensor&          neon_acosh_();
    tensor&          neon_pow_(const value_type val);
    tensor&          neon_pow_(const tensor& other);
    tensor&          neon_abs_();
    tensor&          neon_dist_(const tensor& other);
    tensor&          neon_dist_(const value_type val);
    tensor&          neon_maximum_(const tensor& other);
    tensor&          neon_maximum_(const value_type val);
    tensor&          neon_bitwise_right_shift_(const int amount);
    tensor&          neon_bitwise_left_shift_(const int amount);
    tensor&          neon_bitwise_or_(const value_type val);
    tensor&          neon_bitwise_xor_(const value_type val);
    tensor&          neon_bitwise_not_();
    tensor&          neon_bitwise_and_(const value_type val);
    tensor&          neon_bitwise_and_(const tensor& other);
    tensor&          neon_bitwise_or_(const tensor& other);
    tensor&          neon_bitwise_xor_(const tensor& other);
    tensor&          neon_zeros_(shape_type sh = {});
    tensor&          neon_ones_(shape_type sh);
    tensor&          neon_randomize_(const shape_type& sh, bool bounded);
    tensor&          neon_negative_();
    tensor&          neon_relu_();
    tensor&          neon_sigmoid_();
    tensor&          neon_clipped_relu_(const value_type clip_limit);
    tensor&          neon_clamp_(const_reference min_val = std::numeric_limits<value_type>::lowest(),
                                 const_reference max_val = std::numeric_limits<value_type>::max());
    tensor&          neon_floor_();
    tensor&          neon_ceil_();
    tensor&          neon_logical_or_(const value_type val);
    tensor&          neon_logical_xor_(const value_type val);
    tensor&          neon_logical_and_(const value_type val);
    tensor&          neon_logical_or_(const tensor& other);
    tensor&          neon_logical_xor_(const tensor& other);
    tensor&          neon_logical_and_(const tensor& other);
    tensor&          neon_operator_plus_eq(const_reference val) const;
    tensor&          neon_operator_minus_eq(const tensor& other) const;
    tensor&          neon_operator_times_eq(const tensor& other) const;
    tensor&          neon_operator_minus_eq(const_reference val) const;
    tensor<_s32>     neon_int32_() const;
    tensor<_u32>     neon_uint32_() const;
    tensor<_f32>     neon_float32_() const;
    tensor<_f64>     neon_double_() const;
    tensor<uint64_t> neon_unsigned_long_() const;
    tensor<int64_t>  neon_long_() const;
    tensor           neon_operator_plus(const tensor& other) const;
    tensor           neon_operator_plus(const value_type val) const;
    tensor           neon_operator_minus(const tensor& other) const;
    tensor           neon_operator_minus(const value_type val) const;
    tensor           neon_transpose() const;
    tensor           neon_matmul(const tensor& other) const;
    tensor           neon_absolute_(const tensor& tensor) const;
    tensor           neon_cross_product(const tensor& other) const;
    tensor           neon_dot(const tensor& other) const;
    tensor           neon_argmax(index_type dim) const;
    tensor           neon_sum(const index_type axis) const;
    tensor
    neon_slice(index_type dim, std::optional<index_type> start, std::optional<index_type> end, index_type step) const;
    tensor<bool>       neon_equal(const tensor& other) const;
    tensor<bool>       neon_equal(const value_type val) const;
    tensor<bool>       neon_less_equal(const tensor& other) const;
    tensor<bool>       neon_less_equal(const value_type val) const;
    tensor<bool>       neon_less(const value_type val) const;
    tensor<bool>       neon_less(const tensor& other) const;
    tensor<bool>       neon_greater(const value_type val) const;
    tensor<bool>       neon_greater(const tensor& other) const;
    tensor<bool>       neon_greater_equal(const value_type val) const;
    tensor<bool>       neon_greater_equal(const tensor& other) const;
    tensor<index_type> neon_argsort(index_type d, bool ascending) const;
    tensor<index_type> neon_argmax_(index_type dim) const;
    index_type         neon_count_nonzero(index_type dim) const;
    double             neon_mean() const;

   private:
    [[nodiscard]] std::size_t       computeStride(std::size_t dim, const shape_type& shape) const noexcept;
    void                            printRecursive(std::size_t index, std::size_t depth, const shape_type& shape) const;
    void                            compute_strides();
    [[nodiscard]] index_type        compute_index(const std::vector<index_type>& idx) const;
    [[nodiscard]] static index_type computeSize(const shape_type& dims) noexcept;
    index_type                      compute_outer_size(const index_type dim) const;
    [[nodiscard]] static _f32       frac(const_reference val) noexcept;
    // where the tensor is stored
    bool is_cuda_device() const;
    bool equal_shape(const shape_type& x, const shape_type& y) const;

};  // tensor class

template<class _Tp>
bool tensor<_Tp>::equal_shape(const shape_type& x, const shape_type& y) const {
    std::size_t size_x = x.size();
    std::size_t size_y = y.size();

    if (size_x == size_y)
    {
        return x == y;
    }

    if (size_x < size_y)
    {
        return equal_shape(y, x);
    }

    int diff = size_x - size_y;
    for (std::size_t i = 0; i < size_y; ++i)
    {
        if (x[i + diff] != y[i] and x[i + diff] != 1 and y[i] != 1)
        {
            return false;
        }
    }

    return true;
}

template<class _Tp>
inline bool tensor<_Tp>::is_cuda_device() const {
    return (device_ == Device::CUDA);
}

template<class _Tp>
[[nodiscard]]
inline _f32 tensor<_Tp>::frac(const_reference val) noexcept {
    return std::fmod(static_cast<float32_t>(val), 1.0f);
}

template<class _Tp>
inline typename tensor<_Tp>::index_type tensor<_Tp>::compute_outer_size(const index_type dim) const {
    // just a placeholder for now
    return 0;
}

template<class _Tp>
[[nodiscard]]
inline typename tensor<_Tp>::index_type tensor<_Tp>::computeSize(const shape_type& dims) noexcept {
    if (dims.empty())
    {
        return static_cast<index_type>(0);
    }
    index_type ret = 1;
    for (const index_type& d : dims)
    {
        ret *= d;
    }
    return ret;
}

template<class _Tp>
[[nodiscard]]
inline typename tensor<_Tp>::index_type tensor<_Tp>::compute_index(const std::vector<index_type>& idx) const {
    if (idx.size() != shape_.size())
    {
        throw index_error("compute_index : input indices does not match the tensor shape");
    }

    index_type index = 0, i = 0;

    for (const auto& elem : strides_)
    {
        index += idx[i++] * elem;
    }

    return index;
}

template<class _Tp>
void tensor<_Tp>::compute_strides() {
    if (shape_.empty())
    {
        assert(this->data_.empty());
        return;
    }

    strides_ = shape_type(shape_.size(), 1);
    int st = 1, i = static_cast<int>(shape_.size() - 1);

    for (; i >= 0; i--)
    {
        strides_[i] = st;
        st *= shape_[i];
    }
}

template<class _Tp>
void tensor<_Tp>::printRecursive(std::size_t index, std::size_t depth, const shape_type& shape) const {
    if (depth == shape.size() - 1)
    {
        std::cout << "[";

        for (std::size_t i = 0; i < shape[depth]; ++i)
        {
            if constexpr (std::is_floating_point_v<_Tp>)
            {
                std::cout << std::fixed << std::setprecision(4) << data_[index + i];
            }
            else
            {
                std::cout << data_[index + i];
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
        std::size_t stride = computeStride(depth + 1, shape);

        for (std::size_t i = 0; i < shape[depth]; ++i)
        {
            if (i > 0)
            {
                std::cout << "\n";
            }

            for (std::size_t j = 0; j < depth + 1; ++j)
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
        for (std::size_t j = 0; j < depth; ++j)
        {
            std::cout << " ";
        }

        std::cout << "]";
    }
}

template<class _Tp>
[[nodiscard]]
inline std::size_t tensor<_Tp>::computeStride(std::size_t dim, const shape_type& shape) const noexcept {
    std::size_t stride = 1;

    for (const auto& elem : shape)
    {
        stride *= elem;
    }

    return stride;
}

template<>
class tensor<bool>;  // explicit instantiation

template<>
class tensor<bool>
{
   public:
    using self                   = tensor;
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

    enum class Device {
        CPU,
        CUDA
    };

   private:
    mutable data_t     data_;
    mutable shape_type shape_;
    mutable shape_type strides_;
    Device             device_;
    bool               is_cuda_tensor_ = false;

   public:
    tensor() = default;

    explicit tensor(const shape_type& sh, value_type v, Device d = Device::CPU) :
        shape_(sh),
        data_(computeSize(sh), v),
        device_(d) {
        compute_strides();
    }

    explicit tensor(const shape_type& sh, Device d = Device::CPU) :
        shape_(sh),
        device_(d) {
        index_type s = computeSize(sh);
        data_        = data_t(s);
        compute_strides();
    }

    explicit tensor(const shape_type& sh, const data_t& d, Device dev = Device::CPU) :
        shape_(sh),
        device_(dev) {
        index_type s = computeSize(sh);
        if (d.size() != static_cast<std::size_t>(s))
        {
            throw std::invalid_argument("Initial data vector must match the tensor");
        }
        data_ = d;
        compute_strides();
    }

    tensor(const tensor& t) :
        data_(t.storage()),
        shape_(t.shape()),
        strides_(t.strides()),
        device_(t.device()) {}

    tensor(tensor&& t) noexcept :
        data_(std::move(t.storage())),
        shape_(std::move(t.shape())),
        strides_(std::move(t.strides())),
        device_(std::move(t.device())) {}

    tensor(const shape_type& sh, std::initializer_list<value_type> init_list, Device d = Device::CPU) :
        shape_(sh),
        device_(d) {
        index_type s = computeSize(sh);
        if (init_list.size() != static_cast<std::size_t>(s))
        {
            throw std::invalid_argument("Initializer list size must match tensor size");
        }
        data_ = data_t(init_list);
        compute_strides();
    }

    tensor(const shape_type& sh, const tensor& other) :
        data_(other.storage()),
        shape_(sh),
        device_(other.device()) {
        compute_strides();
    }

    data_t     storage() const noexcept { return data_; }
    shape_type shape() const noexcept { return shape_; }
    shape_type strides() const noexcept { return strides_; }
    Device     device() const noexcept { return device_; }

    bool operator==(const tensor& other) const {
        return equal_shape(shape(), other.shape()) and data_ == other.storage();
    }

    bool operator!=(const tensor& other) const { return !(*this == other); }

    tensor<bool>& operator=(const tensor<bool>& other) const noexcept;

    reference at(shape_type idx) {
        if (idx.empty())
        {
            throw index_error("Passing an empty vector as indices for a tensor");
        }

        index_type i = compute_index(idx);

        if (i < 0 or i >= data_.size())
        {
            throw index_error("input indices are out of bounds");
        }

        return data_[i];
    }

    reference operator[](const index_type idx) {
        if (idx < 0 or idx >= data_.size())
        {
            throw index_error("input index is out of bounds");
        }

        return data_[idx];
    }

    const_reference at(const shape_type idx) const { return at(idx); }

    const_reference operator[](const index_type idx) const { return (*this)[idx]; }

    reference operator()(std::initializer_list<index_type> index_list) { return at(shape_type(index_list)); }

    const_reference operator()(std::initializer_list<index_type> index_list) const {
        return at(shape_type(index_list));
    }

    bool empty() const { return data_.empty(); }

    std::size_t n_dims() const noexcept { return shape_.size(); }

    index_type size(const index_type dim) const {
        if (dim < 0 or dim > static_cast<index_type>(shape_.size()))
        {
            throw std::invalid_argument("dimension input is out of range");
        }

        if (dim == 0)
        {
            return data_.size();
        }

        return shape_[dim - 1];
    }

    index_type capacity() const noexcept { return data_.capacity(); }

    tensor<bool> logical_not() const { return tensor<bool>(logical_not_()); }

    tensor<bool> logical_or(const value_type val) const {
        self t = clone();
        t.logical_or_(val);
        return t;
    }

    tensor<bool> logical_or(const tensor& other) const {
        self t = clone();
        t.logical_or_(other);
        return t;
    }

    tensor<bool> logical_and(const value_type val) const {
        self t = clone();
        t.logical_and_(val);
        return t;
    }

    tensor<bool> logical_and(const tensor& other) const {
        self t = clone();
        t.logical_and_(other);
        return t;
    }

    tensor<bool> logical_xor(const value_type val) const {
        self t = clone();
        t.logical_and_(val);
        return t;
    }

    tensor<bool> logical_xor(const tensor& other) const {
        self t = clone();
        t.logical_xor_(other);
        return t;
    }

    tensor<bool>& logical_not_() {

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = ~(data_[i]);
        }

        return *this;
    }

    tensor<bool>& logical_or_(const value_type val) {

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (data_[i] or val);
        }

        return *this;
    }

    tensor<bool>& logical_or_(const tensor& other) {

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (data_[i] or other[i]);
        }

        return *this;
    }

    tensor<bool>& logical_and_(const value_type val) {

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (data_[i] and val);
        }

        return *this;
    }

    tensor<bool>& logical_and_(const tensor& other) {

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (data_[i] and other[i]);
        }

        return *this;
    }

    tensor<bool>& logical_xor_(const value_type val) {

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (data_[i] xor val);
        }

        return *this;
    }

    tensor<bool>& logical_xor_(const tensor& other) {

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (data_[i] xor other[i]);
        }

        return *this;
    }

    const tensor<bool>& logical_not_() const {
        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = ~(data_[i]);
        }

        return *this;
    }

    const tensor<bool>& logical_or_(const value_type val) const {

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (data_[i] or val);
        }

        return *this;
    }

    const tensor<bool>& logical_or_(const tensor& other) const {

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (data_[i] or other[i]);
        }

        return *this;
    }

    const tensor<bool>& logical_and_(const value_type val) const {

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (data_[i] and val);
        }

        return *this;
    }

    const tensor<bool>& logical_and_(const tensor& other) const {
        index_type i = 0;

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (data_[i] and other[i]);
        }

        return *this;
    }

    const tensor<bool>& logical_xor_(const value_type val) const {
        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (data_[i] xor val);
        }

        return *this;
    }

    const tensor<bool>& logical_xor_(const tensor& other) const {

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (data_[i] xor other[i]);
        }

        return *this;
    }

    tensor<bool>& operator!() {
        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = !(data_[i]);
        }

        return *this;
    }

    const tensor<bool>& operator!() const {
        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = !(data_[i]);
        }

        return *this;
    }

    tensor<bool>
    slice(index_type dim, std::optional<index_type> start, std::optional<index_type> end, index_type step) const {
        if (dim < 0 or dim >= static_cast<index_type>(data_.size()))
        {
            throw shape_error("Dimension out of range");
        }

        tensor<bool> ret;
        index_type   s       = shape_[dim];
        index_type   start_i = start.value_or(0);
        index_type   end_i   = end.value_or(0);

        if (start_i < 0)
        {
            start_i += s;
        }

        if (end_i < 0)
        {
            end_i += s;
        }

        start_i = std::max(index_type(0), std::min(start_i, s));
        end_i   = std::max(index_type(0), std::min(end_i, s));

        index_type slice_size = (end_i - start_i + step - 1) / step;
        shape_type ret_dims   = shape_;

        ret_dims[-dim] = slice_size;
        ret            = self(ret_dims);

        index_type i = start_i, j = 0;

        for (; i < end_i; i += step, ++j)
        {
            ret[j] = data_[j];
        }

        return ret;
    }

    tensor<bool> row(const index_type index) const {
        if (shape_.size() != 2)
        {
            throw shape_error("Cannot get a row from a non two dimensional tensor");
        }

        if (shape_[0] <= index or index < 0)
        {
            throw index_error("Index input is out of range");
        }

        index_type start = shape_[1] * index;
        index_type end   = shape_[1] * index + shape_[1];
        index_type i     = start;
        data_t     r(end);

        for (; i < end; ++i)
        {
            r[i] = data_[i];
        }

        return self({shape_[1]}, r);
    }

    tensor<bool> col(const index_type index) const {
        if (shape_.size() != 2)
        {
            throw index_error("Cannot get a column from a non two dimensional tensor");
        }

        if (shape_[1] <= index or index < 0)
        {
            throw index_error("Index input out of range");
        }

        data_t     c(shape_[0]);
        index_type i = 0;
        for (; i < shape_[0]; ++i)
        {
            c[i] = data_[compute_index({i, index})];
        }

        return self({shape_[0]}, c);
    }

    tensor<bool> clone() const {
        data_t     d = data_;
        shape_type s = shape_;
        return self(s, d);
    }

    tensor<bool> reshape(const shape_type sh) const {
        data_t     d = data_;
        index_type s = computeSize(sh);

        if (s != data_.size())
        {
            throw shape_error("input shape must have size of element equal to the current number of elements in "
                              "the"
                              "tensor data");
        }

        return self(sh, d);
    }

    tensor<bool> reshape_as(const tensor& other) const { return reshape(other.shape()); }

    tensor<bool> transpose() const {
        if (equal_shape(shape_, shape_type({shape_[0], shape_[1], 1}))
            or equal_shape(shape_, shape_type({1, shape_[0], shape_[1]})))
        {
            throw shape_error("Matrix transposition can only be done on 2D tensors");
        }

        tensor           ret({shape_[1], shape_[0]});
        const index_type rows = shape_[0];
        const index_type cols = shape_[1];

        index_type i = 0;
        for (; i < rows; ++i)
        {
            index_type j = 0;
            for (; j < cols; ++j)
            {
                ret.at({j, i}) = at({i, j});
            }
        }

        return ret;
    }

    tensor<bool>& transpose_() {
        if (shape_.size() != 2)
        {
            throw shape_error("Transpose operation is only valid for 2D tensors");
        }

        const index_type r = shape_[0];
        const index_type c = shape_[1];

        if (r != c)
        {
            throw shape_error("In-place transpose is only supported for square tensors");
        }

        for (index_type i = 0; i < r; ++i)
        {
            for (index_type j = i + 1; j < c; ++j)
            {
                std::swap(data_[i * c + j], data_[j * c + i]);
            }
        }

        return *this;
    }

    const tensor<bool>& transpose_() const {
        if (shape_.size() != 2)
        {
            throw shape_error("Transpose operation is only valid for 2D tensors");
        }

        const index_type r = shape_[0];
        const index_type c = shape_[1];

        if (r != c)
        {
            throw shape_error("In-place transpose is only supported for square tensors");
        }

        for (index_type i = 0; i < r; ++i)
        {
            for (index_type j = i + 1; j < c; ++j)
            {
                std::swap(data_[i * c + j], data_[j * c + i]);
            }
        }

        return *this;
    }

    tensor<bool> resize_as(const shape_type sh) const {
        self ret = clone();
        ret.resize_as_(sh);
        return ret;
    }

    tensor<bool>& resize_as_(const shape_type sh) { return *this; }

    tensor<bool> squeeze(const index_type dim) const {
        self ret = clone();
        ret.squeeze_(dim);
        return ret;
    }

    tensor<bool>& squeeze_(const index_type dim) { return *this; }

    tensor<bool> repeat(const data_t& d, int dim) const {
        self ret = clone();
        ret.repeat_(d, dim);
        return ret;
    }

    tensor<bool>& repeat_(const data_t& d, int dim) {
        if (d.empty())
        {
            throw std::invalid_argument("Cannot repeat an empty tensor");
        }

        if (size(0) < d.size())
        {
            data_ = data_t(d.begin(), d.end());
        }

        std::size_t start      = 0;
        std::size_t end        = d.size();
        std::size_t total_size = size(0);

        if (total_size < d.size())
        {
            return *this;
        }

        unsigned int nbatches  = total_size / d.size();
        std::size_t  remainder = total_size % d.size();

        for (unsigned int i = 0; i < nbatches; ++i)
        {
            for (std::size_t j = start, k = 0; k < d.size(); ++j, ++k)
            {
                data_[j] = d[k];
            }

            start += d.size();
        }

        for (std::size_t j = start, k = 0; j < total_size and k < remainder; ++j, ++k)
        {
            data_[j] = d[k];
        }

        return *this;
    }

    tensor<bool> permute(const index_type dim) const {
        // TODO : implement permute here
        data_t d;
        return self(shape_, d);
    }

    tensor<bool> cat(const std::vector<tensor<value_type>>& others, index_type dim) const {
        for (const tensor& t : others)
        {
            index_type i = 0;
            for (; i < shape_.size(); ++i)
            {
                if (i != dim and shape_[i] != t.shape_[i])
                {
                    throw shape_error("Cannot concatenate tensors with different shapes along non-concatenation "
                                      "dimensions");
                }
            }
        }

        shape_type ret_sh = shape_;
        for (const tensor& t : others)
        {
            ret_sh[dim] += t.shape_[dim];
        }

        data_t c;
        c.reserve(data_.size());
        c.insert(c.end(), data_.begin(), data_.end());

        for (const tensor& t : others)
        {
            c.insert(c.end(), t.data_.begin(), t.data_.end());
        }

        return self(ret_sh, c);
    }

    tensor<bool> unsqueeze(const index_type dim) const {
        if (dim < 0 or dim > static_cast<index_type>(shape_.size()))
        {
            throw index_error("Dimension out of range in unsqueeze");
        }

        shape_type s = shape_;
        s.insert(s.begin() + dim, 1);

        tensor ret;
        ret.shape_ = s;
        ret.data_  = data_;

        return ret;
    }

    tensor<bool>& randomize_(const shape_type& sh = {}) {
        if (sh.empty() and shape_.empty())
        {
            throw shape_error("Shape must be initialized");
        }

        if (shape_.empty() or shape_ != sh)
        {
            shape_ = sh;
        }

        index_type s = computeSize(shape_);
        data_.resize(s);
        compute_strides();

        std::random_device                 rd;
        std::mt19937                       gen(rd());
        std::uniform_int_distribution<int> dist(0, 1);

        for (index_type i = 0; i < data_.size(); ++i)
        {
            data_[i] = (dist(gen) == 0);
        }

        return *this;
    }

    tensor<bool>& push_back(value_type v) {
        if (shape_.size() != 1)
        {
            throw shape_error("push_back is only supported for one dimensional tensors");
        }
        data_.push_back(v);
        ++(shape_[0]);
        compute_strides();
        return *this;
    }

    tensor<bool>& pop_back(value_type v) {
        if (shape_.size() != 1)
        {
            throw shape_error("push_back is only supported for one dimensional tensors");
        }
        data_.pop_back();
        --(shape_[0]);
        compute_strides();
        return *this;
    }

    tensor<bool>& view(std::initializer_list<index_type> sh) {
        index_type s = computeSize(sh);

        if (s != data_.size())
        {
            throw std::invalid_argument("Total elements do not match for new shape");
        }
        shape_ = sh;
        compute_strides();
        return *this;
    }

    void print() const {
        printRecursive(0, 0, shape_);
        std::cout << std::endl;
    }

   private:
    bool equal_shape(const shape_type& x, const shape_type& y) const {
        std::size_t size_x = x.size();
        std::size_t size_y = y.size();

        if (size_x == size_y)
        {
            return x == y;
        }

        if (size_x < size_y)
        {
            return equal_shape(y, x);
        }

        int diff = size_x - size_y;
        for (std::size_t i = 0; i < size_y; ++i)
        {
            if (x[i + diff] != y[i] and x[i + diff] != 1 and y[i] != 1)
            {
                return false;
            }
        }

        return true;
    }

    [[nodiscard]]
    inline std::size_t computeStride(std::size_t dim, const shape_type& shape) const noexcept {
        std::size_t stride = 1;

        for (index_type i = dim; i < shape_.size(); ++i)
        {
            stride *= shape_[i];
        }

        return stride;
    }

    void printRecursive(std::size_t index, std::size_t depth, const shape_type& shape) const {
        if (depth == shape.size() - 1)
        {
            std::cout << "[";

            for (std::size_t i = 0; i < shape[depth]; ++i)
            {
                std::cout << (data_[index + i] ? "true" : "false");

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
            std::size_t stride = computeStride(depth + 1, shape);

            for (std::size_t i = 0; i < shape[depth]; ++i)
            {
                if (i > 0)
                {
                    std::cout << "\n";
                }

                for (std::size_t j = 0; j < depth + 1; j++)
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
            for (std::size_t j = 0; j < depth; j++)
            {
                std::cout << " ";
            }

            std::cout << "]";
        }
    }

    void compute_strides() {
        if (shape_.empty())
        {
            throw shape_error("Shape must be initialized before computing strides");
        }

        strides_ = shape_type(shape_.size(), 1);
        int st = 1, i = static_cast<int>(shape_.size() - 1);
        for (; i >= 0; i--)
        {
            strides_[i] = st;
            st *= shape_[i];
        }
    }

    [[nodiscard]]
    index_type compute_index(const std::vector<index_type>& idx) const {
        if (idx.size() != shape_.size())
        {
            throw index_error("compute_index : input indices does not match the tensor shape");
        }

        index_type index = 0;
        index_type i     = 0;
        for (; i < shape_.size(); ++i)
        {
            index += idx[i] * strides_[i];
        }

        return index;
    }

    [[nodiscard]]
    static index_type computeSize(const shape_type& dims) noexcept {
        uint64_t ret = 1;
        for (const auto& d : dims)
        {
            ret *= d;
        }

        return ret;
    }

    index_type compute_outer_size(const index_type dim) const {
        // just a placeholder for now
        return 0;
    }

    [[nodiscard]]
    static _f32 frac(const_reference val) noexcept {
        return std::fmod(static_cast<float32_t>(val), 1.0f);
    }

    bool is_cuda_device() const { return (device_ == Device::CUDA); }
};  // tensor<bool>
