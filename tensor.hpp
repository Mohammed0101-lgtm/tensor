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

#include <iostream>
#include <stdexcept>
#include <random>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <array>
#include <arm_neon.h>
#include <type_traits>
#include <immintrin.h>
#include <omp.h>
#include <initializer_list>
#include <memory>
#include <climits>
#include <functional>
#include <omp-tools.h>
#include <optional>
#include <bit>
#include <array>
#include <valarray>
#include <numeric>
#include <__iterator/iterator_traits.h>
#include <__iterator/reverse_iterator.h>
#include <__iterator/wrap_iter.h>


constexpr size_t _ARM64_REG_WIDTH = 4;
constexpr size_t _AVX_REG_WIDTH   = 8;


enum class Device {
    CPU,
    CUDA
};


template<class _Tp>
class tensor
{
   public:
    typedef tensor                  __self;
    typedef _Tp                     value_type;
    typedef int64_t                 index_type;
    typedef std::vector<index_type> shape_type;
    typedef value_type&             reference;
    typedef const value_type&       const_reference;
    typedef value_type*             pointer;
    typedef const value_type*       const_pointer;
    typedef std::vector<value_type> data_t;
    typedef const data_t            const_data_container;

    typedef typename data_t::iterator               iterator;
    typedef typename data_t::const_iterator         const_iterator;
    typedef typename data_t::reverse_iterator       reverse_iterator;
    typedef typename data_t::const_reverse_iterator const_reverse_iterator;

   private:
    data_t                  __data_;
    shape_type              __shape_;
    std::vector<index_type> __strides_;
    Device                  __device_;

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

    bool __is_cuda_device = this->__device_ == Device::CUDA ? true : false;

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

    const_iterator begin() const { return this->__data_.begin(); }
    const_iterator end() const { return this->__data_.end(); }

    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

    /**
     * @brief returns the linear data as stored in memory.
     * @return vector of elements stored in this->__data_.
     */
    data_t storage() const noexcept { return this->__data_; }

    /**
     * @brief returns the shape of a tensor
     * @return vector of 64 bit integers indicating the size of each dimension 
    */
    shape_type shape() const noexcept { return this->__shape_; }

    /** 
     * @brief returns the strides of the dimensions
     * @return vector of 64 bit integers indicating the 'stride' of each dimension
    */
    shape_type strides() const noexcept { return this->__strides_; }

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
    size_t size(const index_type __dim) const;

    /**
     * @brief whats the capacity of the tensor (memory reserved)
     * @return unsigned long (aka 64 bit unsigned integer, or 32 bits depending on the system and compiler)
     */
    size_t capacity() const noexcept { return this->__data_.capacity(); }

    /**
     * @brief for direct modifiable access
     * @param __idx vector of integers that specify the multidimensional index 
     * @return reference to the element at the input index
     */
    reference at(shape_type __idx);

    /**
     * @brief for direct unmodifiable access
     * @param __idx vector of integers that specify the multidimensional index 
     * @return constant reference to the element at the input index
     */
    const_reference at(const shape_type __idx) const;

    /**
     * @brief for linear modifiable access
     * @param __in linear index in the tensor storage
     * @return reference to the element at the input index
     */
    reference operator[](const size_t __in);

    /**
     * @brief for linear unmodifiable access
     * @param __in linear index in the tensor storage
     * @return constant reference to the element at the input index
     */
    const_reference operator[](const size_t __in) const;

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
    tensor operator+(const value_type _scalar) const;

    /**
     * @brief subtracts a scalar from each element of the tensor 
     * @param __scalar a value of an arithmetic type
     * @return the difference between self and __scalar element wise
     */
    tensor operator-(const value_type __scalar) const;

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
    bool operatornot_eq(const tensor& __other) const { return !(*this == __other); }

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
    tensor<bool> logical_or(const value_type __val) const;

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
    tensor<bool> less_equal(const value_type __val) const;

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
    tensor<bool> greater_equal(const value_type __val) const;

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
    tensor<bool> equal(const value_type __val) const;

    /**
     * @brief Counts the number of non-zero elements in the tensor.
     * 
     * Optionally, can count non-zero elements along a specific dimension.
     * 
     * @param __dim (Optional) The dimension to count along. Defaults to -1, meaning all dimensions.
     * @return The count of non-zero elements.
     */
    size_t count_nonzero(index_type __dim = -1LL) const;

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
    tensor slice(index_type                __dim,
                 std::optional<index_type> __start,
                 std::optional<index_type> __end,
                 index_type                __step) const;

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
    tensor fmax(const value_type __val) const;

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
    void fmax_(const value_type __val);

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
    tensor fmod(const value_type __val) const;

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
    tensor sum(const index_type __axis) const;

    /**
     * @brief Returns the row at the specified index in the tensor.
     * 
     * @param __index The index of the row to retrieve.
     * @return A new tensor representing the row at the specified index.
     */
    tensor row(const index_type __index) const;

    /**
     * @brief Returns the column at the specified index in the tensor.
     * 
     * @param __index The index of the column to retrieve.
     * @return A new tensor representing the column at the specified index.
     */
    tensor col(const index_type __index) const;

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
    tensor logical_xor(const value_type __val) const;

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
    tensor logical_and(const value_type __val) const;

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
    tensor bitwise_and(const value_type __val) const;

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
    tensor bitwise_or(const value_type __val) const;

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
    tensor bitwise_xor(const value_type __val) const;

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
    tensor reshape(const shape_type __shape) const;

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
    tensor pow(const value_type __val) const;

    /**
     * @brief Computes the cumulative product of the elements along a specified dimension.
     * 
     * @param __dim The dimension along which to compute the cumulative product. Defaults to -1 (last dimension).
     * @return A new tensor containing the cumulative product along the specified dimension.
     */
    tensor cumprod(index_type __dim = -1) const;

    /**
     * @brief Concatenates a vector of tensors along a specified dimension.
     * 
     * @param _others A vector of tensors to concatenate. All tensors must have the same shape except for the specified dimension.
     * @param _dim The dimension along which to concatenate the tensors.
     * @return A new tensor formed by concatenating the input tensors along the specified dimension.
     * 
     * @throws std::invalid_argument If the tensors in _others have incompatible shapes for concatenation.
     */
    tensor cat(const std::vector<tensor>& _others, index_type _dim) const;

    /**
     * @brief Returns the indices of the maximum values along a specified dimension.
     * 
     * @param __dim The dimension along which to find the indices of the maximum values.
     * @return A new tensor containing the indices of the maximum values along the specified dimension.
     */
    tensor argmax(index_type __dim) const;

    /**
     * @brief Inserts a new dimension of size one at the specified index.
     * 
     * @param __dim The dimension index at which to add the new dimension.
     * @return A new tensor with the added dimension.
     */
    tensor unsqueeze(index_type __dim) const;

    /**
     * @brief computes the least common multiple for all the elements in the tensor
     * @return returns the least common multiple as a 64 bit integer
    */
    index_type lcm() const;

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
    void fmod_(const value_type __val);

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
    void logical_or_(const value_type __val);

    /**
     * @brief in place version of logical_xor(tensor) 
     */
    void logical_xor_(const tensor& __other);

    /**
     * @brief in place version of 
     */
    void logical_xor_(const value_type __val);

    /**
     * @brief in place version of logical_and(tensor)
     */
    void logical_and_(const tensor& __other);

    /**
     * @brief in place version of logical_and(value)
     */
    void logical_and_(const value_type __val);

    /**
     * @brief in place version of pow(tensor)
     */
    void pow_(const tensor& __other);

    /**
     * @brief in place version of pow(value)
     */
    void pow_(const value_type __val);

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
    void bitwise_and_(const value_type __val);

    /**
     * @brief in place version of bitwise_and(tensor)
     */
    void bitwise_and_(const tensor& __other);

    /**
     * @brief in place version of bitwise_or(value)
     */
    void bitwise_or_(const value_type __val);

    /**
     * @brief in place version of bitwise_or(tensor)
     */
    void bitwise_or_(const tensor& __other);

    /**
     * @brief in place version of bitwise_xor(value)
     */
    void bitwise_xor_(const value_type __val);

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
    void view(std::initializer_list<index_type> __new_sh);

    /**
     * @brief prints a tensor to stdout
     */
    void print() const noexcept;

    /**
     * @brief sets all the element of the tensor to zero
     * @param __sh the shape of the returned tensor
     * @return a tensor of the input shape where all elements are set to zero
     */
    static tensor zeros(const shape_type& __sh) {
        __check_is_scalar_type("template type must be a scalar : tensor.zeros()");
        data_t __d(std::accumulate(__sh.begin(), __sh.end(), 1, std::multiplies<index_type>()), value_type(0));
        return __self(__d, __sh);
    }

    /**
     * @brief sets all the element of the tensor to one
     * @param __sh the shape of the returned tensor 
     * @return a tensor of the input shape where all the elements are set to one
     */
    static tensor ones(const shape_type& __sh) {
        __check_is_scalar_type("template type must be a scalar : tensor.ones()");
        data_t __d(std::accumulate(__sh.begin(), __sh.end(), 1, std::multiplies<index_type>()), value_type(1));
        return __self(__d, __sh);
    }

    /**
     * @brief in place version of zeros(shape)
     */
    void zeros_(const shape_type& __sh = {}) {
        if (__sh.empty())
            __sh = this->__shape_;

        this->__data_ =
            data_t(std::accumulate(__sh.begin(), __sh.end(), 1, std::multiplies<index_type>()), value_type(0));
    }

    /**
     * @brief in place version of ones(shape)
     */
    void ones_(const shape_type& __sh = {}) {
        if (__sh.empty())
            __sh = this->__shape_;

        this->__data_ =
            data_t(std::accumulate(__sh.begin(), __sh.end(), 1, std::multiplies<index_type>()), value_type(0));
    }

    /**
     * @brief sets all the elements of the tensor to random values
     * @param __sh the shape of the returned tensor
     * @param __bounded wether the elements should be between zero or one or not
     * @return a tensor of the input shape where all the elements are set to random values 
     */
    tensor randomize(const shape_type& __sh, bool __bounded = false);

    /**
     * @brief in place version of randomize(shape, bounded)
     */
    void randomize_(const shape_type& __sh, bool __bounded = false);

    /**
     * @brief gets the indices of the maximum values along each dimension
     * @param __dim 64 bit integer indicating the desired dimension to be evaluated
     * @return a tensor of 64 bit integers
     */
    tensor<index_type> argmax_(index_type __dim) const;

    /**
     * @brief sorts the elements along a given dimension
     * @param __dim the dimension along which to sort the elements
     * @param __ascending true to sort in ascending order 
     * @return a tensor of 64 bit integer (indicies of the sorted elements in the sort order)
     */
    tensor<index_type> argsort(index_type __dim = -1LL, bool __ascending = true) const;

   private:
    static void __check_is_scalar_type(const std::string __msg) {
        assert(!__msg.empty());
        if (!std::is_scalar<value_type>::value)
            throw std::runtime_error(__msg);
    }

    static void __check_is_integral_type(const std::string __msg) {
        assert(!__msg.empty());
        if (!std::is_integral<value_type>::value)
            throw std::runtime_error(__msg);
    }

    template<typename __t>
    static void __check_is_same_type(const std::string __msg) {
        assert(!__msg.empty());
        if (!std::is_same<value_type, __t>::value)
            throw std::runtime_error(__msg);
    }

    static void __check_is_arithmetic_type(const std::string __msg) {
        assert(!__msg.empty());
        if (!std::is_arithmetic<value_type>::value)
            throw std::runtime_error(__msg);
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

    size_t __compute_index(const std::vector<index_type>& __idx) const {
        if (__idx.size() != this->__shape_.size())
            throw std::out_of_range("input indices does not match the tensor __shape_");

        size_t __index = 0, __i = 0;
        for (; __i < this->__shape_.size(); __i++)
            __index += __idx[__i] * this->__strides_[__i];

        return __index;
    }

    static uint64_t __computeSize(const shape_type& __dims) {
        uint64_t __ret = 1ULL;
        for (const index_type& __d : __dims)
            __ret *= __d;

        return __ret;
    }

    uint64_t __compute_outer_size(const index_type __dim) const {
        // just a placeholder for now
        return 0ULL;
    }

    float      __frac(const_reference __scalar) { return std::fmod(static_cast<float>(__scalar), 1.0f); }
    index_type __lcm(const index_type __a, const index_type __b) { return (__a * __b) / std::gcd(__a, __b); }
};  // tensor class


template<class _Tp>
size_t tensor<_Tp>::size(const index_type __dim) const {
    if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
        throw std::invalid_argument("dimension input is out of range");

    if (__dim == 0)
        return this->__computeSize(this->__shape_);

    return this->__shape_[__dim];
}


template<class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::at(tensor<_Tp>::shape_type __idx) {
    if (__idx.empty())
        throw std::invalid_argument("Passing an empty vector as indices for a tensor");

    index_type __i = this->__compute_index(__idx);
    if (__i < 0 || __i >= this->__data_.size())
        throw std::invalid_argument("input indices are out of bounds");

    return this->__data_[__i];
}


template<class _Tp>
typename tensor<_Tp>::const_reference tensor<_Tp>::at(const tensor<_Tp>::shape_type __idx) const {
    if (__idx.empty())
        throw std::invalid_argument("Passing an empty vector as indices for a tensor");

    index_type __i = this->__compute_index(__idx);
    if (__i < 0 || __i >= this->__data_.size())
        throw std::invalid_argument("input indices are out of bounds");

    return this->__data_[__i];
}


template<class _Tp>
tensor<_Tp>::reference tensor<_Tp>::operator[](const size_t __in) {
    if (__in >= this->__data_.size() || __in < 0)
        throw std::out_of_range("Access index is out of range");

    return this->__data_[__in];
}


template<class _Tp>
tensor<_Tp>::const_reference tensor<_Tp>::operator[](const size_t __in) const {
    if (__in >= this->__data_.size() || __in < 0)
        throw std::out_of_range("Access index is out of range");

    return this->__data_[__in];
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const tensor& __other) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");
    if (__other.shape() != this->__shape_)
        throw std::invalid_argument("Cannot add two tensors with different shapes");

    data_t __d(this->__data_.size());

#if defined(__ARM_NEON)
    constexpr __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    size_t __i = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __vec1 = vld1q_f32(reinterpret_cast<const float*>(&this->__data_[__i]));
        float32x4_t __vec2 = vld1q_f32(reinterpret_cast<const float*>(&__other[__i]));

        float32x4_t __result = vaddq_f32(__vec1, __vec2);

        vst1q_f32(reinterpret_cast<float*>(&__d[__i]), __result);
    }

    for (__i; __i < this->__data_.size(); __i++)
        __d[__i] = this->__data_[__i] + __other[__i];

#else
    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), __d.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v + __w); });

#endif
    return __self(__d, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const value_type __scalar) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");
    data_t __d(this->__data_.size());

#if defined(__ARM_NEON)
    constexpr __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    float32x4_t __val_vec = vdupq_n_f32(reinterpret_cast<const float32_t*>(&__scalar));

    size_t __i = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __vec = vld1q_f32(reinterpret_castr<const float32_t*>(&this->__data[__i]));
        float32x4_t __res = vaddq_f32(__vec, __val_vec);

        vst1q_f32(reinterpret_cast<float32_t*>(&__d[__i]), __res);
    }

    for (__i; __i < this->__data_.size(); __i++)
        __d[__i] = this->__data_[__i] + __scalar;

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__scalar](const_reference __v) { return static_cast<value_type>(__v + __scalar); });

#endif
    return __self(__d, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+=(const tensor& __other) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");
    assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));

#if defined(__ARM_NEON)

#else
    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v + __w); });
#endif
    return *this;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+=(const_reference __scalar) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");
#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    size_t      __i       = 0;
    float32x4_t __val_vec = vdupq_n_f32(reinterpret_cast<const float32_t*>(&__scalar));
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
        float32x4_t __add_vec  = vaddq_f32(__data_vec, __val_vec);

        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __add_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = this->__data_[__i] + __scalar;

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__scalar](const_reference __v) { return static_cast<value_type>(__v + __scalar); });
#endif
    return *this;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const tensor& __other) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");
    if (__other.shape() != this->__shape_)
        throw std::invalid_argument("Cannot add two tensors with different shapes");

    data_t __d(this->__data_.size());
#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    size_t __i = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
        float32x4_t __oth = vld1q_f32(reinterpret_cast<const float32_t*>(&__other[__i]));
        float32x4_t __sub = vsubq_f32(__vec, __oth);
        vst1q_f32(reinterpret_cast<const float32_t*>(&__d[__i]), __sub);
    }

    for (; __i < this->__data_[__i]; __i++)
        __d[__i] = this->__data_[__i] - __other[__i];

#else
    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), __d.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v - __w); });

#endif
    return __self(__d, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const value_type __scalar) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");

    data_t __d(this->__data_.size());

#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    float32x4_t __vals = vdupq_n_f32(reinterpret_cast<const float32_t*>(&__scalar));
    size_t      __i    = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
        float32x4_t __sub = vsubq_f32(__vec, __vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&__d[__i]), __sub);
    }

    for (; __i < this->__data_.size(); __i++)
        __d[__i] = this->__data_[__i] - __scalar;

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__scalar](const_reference __v) { return static_cast<value_type>(__v - __scalar); });

#endif
    return __self(*this);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-=(const tensor& __other) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");
    static_assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));

#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    size_t __i = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
        float32x4_t __oth = vld1q_f32(reinterpret_cast<const float32_t*>(&__other[__i]));
        float32x4_t __sub = vsubq_f32(__vec, __oth);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __sub);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] -= __other[__i];

#else
    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v - __w); });

#endif
    return *this;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator*=(const tensor& __other) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");
    assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));

#if defined(__ARM_NEON)

#else
    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v * __w); });

#endif
    return *this;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator*=(const_reference __scalar) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");
#if defined(__ARM_NEON)

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_type>(__v * __scalar); });
#endif
    return *this;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator/=(const tensor& __other) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");
    assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));

    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v / __w); });
    return *this;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator/=(const_reference __scalar) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");

    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_type>(__v / __scalar); });
    return *this;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-=(const_reference __scalar) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");

#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    float32x4_t __val = vld1q_f32(reinterpret_cast<const float32_t*>(&__val));
    size_t      __i   = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
        float32x4_t __sub = vsubq_f32(__vec, __val);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __sub);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] -= __scalar;

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v - __w); });

#endif
    return *this;
}


template<class _Tp>
void tensor<_Tp>::log2_() {
    this->__check_is_integral_type("Given data type must be an integral");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "log2 : operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += 4)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::log2(__vals[0]);
        __vals[1] = std::log2(__vals[1]);
        __vals[2] = std::log2(__vals[2]);
        __vals[3] = std::log2(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::log2(this->__data_[__i]));
#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_type>(std::log2(static_cast<float>(this->__data_[__i])));
    });
#endif
}


template<class _Tp>
void tensor<_Tp>::asinh_() {
    this->__check_is_scalar_type("Cannot perform asinh on non-scalar data type");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "asinh : operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += 4)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::asinh(__vals[0]);
        __vals[1] = std::asinh(__vals[1]);
        __vals[2] = std::asinh(__vals[2]);
        __vals[3] = std::asinh(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(&this->__data_[__i], __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::asinh(this->__data_[__i]));
#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::asinh(static_cast<double>(__v))) });
#endif
}


template<class _Tp>
void tensor<_Tp>::atan_() {
    this->__check_is_integral_type("template class must be integral type");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "Arctangent operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += 4)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::atan(__vals[0]);
        __vals[1] = std::atan(__vals[1]);
        __vals[2] = std::atan(__vals[2]);
        __vals[3] = std::atan(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(&this->__data_[__i], __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::atan(this->__data_[__i]));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::atan(static_cast<double>(__v))); });
#endif
}


template<class _Tp>
void tensor<_Tp>::floor_() {
    this->__check_is_scalar_type("Cannot get the floor of a non scalar value");

#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "Floor operation only supported for floating-point types.");

    size_t __i = 0;
    for (; __i < this->__data_.size(); __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec  = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
        float32x4_t __floor_vec = vrndmq_f32(__data_vec);
        vst1q_f32(&this->__data_[__i], __floor_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::floor(static_cast<double>(this->__data_[__i])));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::floor(static_cast<double>(__v))); });

#endif
}


template<class _Tp>
void tensor<_Tp>::ceil_() {
    this->__check_is_scalar_type("Cannot get the ceiling of a non scalar value");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "Ceiling operation only supported for floating-point types.");

    size_t __i = 0;
    for (; __i + _ARM64_REG_WIDTH <= this->__data_.size(); __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
        float32x4_t __ceil_vec = vrndpq_f32(__data_vec);
        vst1q_f32(&this->__data_[__i], __ceil_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::ceil(static_cast<double>(this->__data_[__i])));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::ceil(static_cast<double>(__v))) });
#endif
}


template<class _Tp>
void tensor<_Tp>::sin_() {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");

#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    size_t __i = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float43x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
        float32_t   __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::sin(__vals[0]);
        __vals[1] = std::sin(__vals[1]);
        __vals[2] = std::sin(__vals[2]);
        __vals[3] = std::sin(__vals[3]);

        float32x4_t __sin_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __sin_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::sin(this->__data_[__i]));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::sin(static_cast<double>(__v))) });
#endif
}


template<class _Tp>
void tensor<_Tp>::asin_() {
    this->__check_is_scalar_type("Cannot perform asin on non-scalar data type");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "asin : operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::asin(__vals[0]);
        __vals[1] = std::asin(__vals[1]);
        __vals[2] = std::asin(__vals[2]);
        __vals[3] = std::asin(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::asin(this->__data_[__i]));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::asin(static_cast<double>(__v))); });
#endif
}


template<class _Tp>
void tensor<_Tp>::log10_() {
    this->__check_is_integral_type("Given data type must be an integral");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "log10 : operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::log10(__vals[0]);
        __vals[1] = std::log10(__vals[1]);
        __vals[2] = std::log10(__vals[2]);
        __vals[3] = std::log10(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::log10(this->__data_[__i]));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_type>(std::log10(static_cast<double>(this->__data_[__i])));
    });
#endif
}


template<class _Tp>
void tensor<_Tp>::log_() {
    this->__check_is_integral_type("Given data type must be an integral");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "log : operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::log(__vals[0]);
        __vals[1] = std::log(__vals[1]);
        __vals[2] = std::log(__vals[2]);
        __vals[3] = std::log(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::log(this->__data_[__i]));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_type>(std::log(static_cast<double>(this->__data_[__i])));
    });
#endif
}


template<class _Tp>
void tensor<_Tp>::exp_() {
    this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value, "exp operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::exp(__vals[0]);
        __vals[1] = std::exp(__vals[1]);
        __vals[2] = std::exp(__vals[2]);
        __vals[3] = std::exp(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::exp(this->__data_[__i]));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_type>(std::exp(static_cast<double>(this->__data_[__i])));
    });
#endif
}


template<class _Tp>
void tensor<_Tp>::sqrt_() {
    this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "sqrt : operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::sqrt(__vals[0]);
        __vals[1] = std::sqrt(__vals[1]);
        __vals[2] = std::sqrt(__vals[2]);
        __vals[3] = std::sqrt(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::sqrt(this->__data_[__i]));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_type>(std::sqrt(static_cast<double>(this->__data_[__i])));
    });
#endif
}


template<class _Tp>
void tensor<_Tp>::sinh_() {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "sinh : operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::sinh(__vals[0]);
        __vals[1] = std::sinh(__vals[1]);
        __vals[2] = std::sinh(__vals[2]);
        __vals[3] = std::sinh(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::sinh(this->__data_[__i]));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::sinh(static_cast<double>(__v))) });
#endif
}


template<class _Tp>
void tensor<_Tp>::acosh_() {
    this->__check_is_scalar_type("Cannot perform a acosh on non-scalar data type");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "acosh : operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::acosh(__vals[0]);
        __vals[1] = std::acosh(__vals[1]);
        __vals[2] = std::acosh(__vals[2]);
        __vals[3] = std::acosh(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::acosh(this->__data_[__i]));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_type>(std::acosh(static_cast<double>(this->__data_[__i])));
    });
#endif
}


template<class _Tp>
void tensor<_Tp>::cosh_() {
    this->__check_is_scalar_type("Cannot perform a cosh on non-scalar data type");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "cosh : operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::cosh(__vals[0]);
        __vals[1] = std::cosh(__vals[1]);
        __vals[2] = std::cosh(__vals[2]);
        __vals[3] = std::cosh(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::cosh(this->__data_[__i]));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_type>(std::cosh(static_cast<double>(this->__data_[__i])));
    });
#endif
}


template<class _Tp>
void tensor<_Tp>::cos_() {
    this->__check_is_scalar_type("Cannot perform a cosine on non-scalar data type");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "cos : operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::cos(__vals[0]);
        __vals[1] = std::cos(__vals[1]);
        __vals[2] = std::cos(__vals[2]);
        __vals[3] = std::cos(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::cos(this->__data_[__i]));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
        return static_cast<value_type>(std::cos(static_cast<double>(this->__data_[__i])));
    });
#endif
}


template<class _Tp>
void tensor<_Tp>::frac_() {
    this->__check_is_scalar_type("Cannot get the fraction of a non-scalar type");
    size_t __i = 0;
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "frac : operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = this->__frac(__vals[0]);
        __vals[1] = this->__frac(__vals[1]);
        __vals[2] = this->__frac(__vals[2]);
        __vals[3] = this->__frac(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __atan_vec);
    }

#endif
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));
}


template<class _Tp>
void tensor<_Tp>::pow_(const value_type __val) {
    this->__check_is_integral_type("cannot get the power of a non integral value");
#if defined(__ARM_NEON)
    static_assert(std::is_floating_point<value_type>::value,
                  "pow : operation only supported for floating-point types.");

    size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t __i        = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));

        float __vals[4];
        vst1q_f32(__vals, __data_vec);
        __vals[0] = std::pow(__vals[0]);
        __vals[1] = std::pow(__vals[1]);
        __vals[2] = std::pow(__vals[2]);
        __vals[3] = std::pow(__vals[3]);

        float32x4_t __atan_vec = vld1q_f32(__vals);
        vst1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]), __atan_vec);
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::pow(this->__data_[__i]));

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [&__val](const_reference __v) {
        return static_cast<value_type>(std::pow(static_cast<double>(__v), static_cast<double>(__val)))
    });
#endif
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::clone() const {
    data_t     __d = this->__data_;
    shape_type __s = this->__shape_;
    return __self(__d, __s);
}


template<class _Tp>
void tensor<_Tp>::bitwise_right_shift_(const int __amount) {
    this->__check_is_integral_type("Cannot perform a bitwise right shift on non-integral values");

    size_t __i = 0;
#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % 4);

    if constexpr (std::is_same<value_type, int32_t>::value)
    {
        for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
        {
            int32x4_t __data_vec    = vld1q_s32(&this->__data_[__i]);
            int32x4_t __shifted_vec = vshrq_n_s32(__data_vec, __amount);
            vst1q_s32(&this->__data_[__i], __shifted_vec);
        }
    }
    else if constexpr (std::is_same<value_type, uint32_t>::value)
    {
        for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
        {
            uint32x4_t __data_vec    = vld1q_u32(&this->__data_[__i]);
            uint32x4_t __shifted_vec = vshrq_n_u32(__data_vec, __amount);
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

    size_t __i = 0;
#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    if constexpr (std::is_same<value_type, int32_t>::value)
    {
        for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
        {
            int32x4_t __data_vec    = vld1q_s32(&this->__data_[__i]);
            int32x4_t __shifted_vec = vshlq_n_s32(__data_vec, __amount);
            vst1q_s32(&this->__data_[__i], __shifted_vec);
        }
    }
    else if constexpr (std::is_same<value_type, uint32_t>::value)
    {
        for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
        {
            uint32x4_t __data_vec    = vld1q_u32(&this->__data_[__i]);
            uint32x4_t __shifted_vec = vshlq_n_u32(__data_vec, __amount);
            vst1q_u32(&this->__data_[__i], __shifted_vec);
        }
    }
#endif
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] >>= __amount;
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


template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::lcm() const {
    this->__check_is_scalar_type("Given template type must be an int");
    index_type __ret = static_cast<index_type>(this->__data_[0]);

    size_t __i = 1;
    for (; __i < this->__data_.size(); __i++)
        __ret = this->__lcm(static_cast<index_type>(this->__data_[__i]), __ret);

    return __ret;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::sinh() const {
    return this->clone().sinh_();
}

template<class _Tp>
void tensor<_Tp>::logical_or_(const value_type __val) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot perform logical OR on non-integral and non-boolean values");

#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t           __i        = 0;

    if constexpr (std::is_same<value_type, int32_t>::value || std::is_same<value_type, bool>::value)
    {
        int32x4_t __val_vec = vdupq_n_s32(static_cast<int32_t>(__val));

        for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
        {
            int32x4_t __data_vec = vld1q_s32(reinterpret_cast<const int32_t*>(&this->__data_[__i]));
            int32x4_t __or       = vorrq_s32(__data_vec, __val_vec);
            vst1q_s32(&this->__data_[__i], __or);
        }
    }
    else if constexpr (std::is_same<value_type, uint32_t>::value)
    {
        uint32x4_t __val_vec = vdupq_n_u32(static_cast<uint32_t>(__val));

        for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
        {
            uint32x4_t __data_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
            uint32x4_t __or       = vorrq_u32(__data_vec, __val_vec);
            vst1q_u32(&this->__data_[__i], __or);
        }
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(this->__data_[__i] || __val);

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__val](const_reference __v) { return static_cast<value_type>(__v || __val); });
#endif
}

template<class _Tp>
void tensor<_Tp>::logical_xor_(const value_type __val) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");

#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);
    size_t           __i        = 0;

    if constexpr (std::is_same<value_type, int32_t>::value || std::is_same<value_type, bool>::value)
    {
        int32x4_t __val_vec = vdupq_n_s32(static_cast<int32_t>(__val));
        for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
        {
            int32z4_t __data_vec = vld1q_s32(reinterpret_cast<const int32_t*>(&this->__data_[__i]));
            int32x4_t __xor      = veorq_s32(__data_vec, __val_vec);
            vst1q_s32(&this->__data_[__i], __or);
        }
    }
    else if constexpr (std::is_same<value_type, uint32_t>::value)
    {
        uint32x4_t __val_vec = vdupq_n_u32(static_cast<uint32_t>(__val));

        for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
        {
            uint32x4_t __data_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
            uint32x4_t __xor      = veorq_u32(__data_vec, __val_vec);
            vst1q_u32(&this->__data_[__i], __xor);
        }
    }

    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(this->__data_[__i] ^ __val);

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__val](const_reference __v) { return static_cast<value_type>(__v ^ __val); });

#endif
}


template<class _Tp>
void tensor<_Tp>::logical_and_(const value_type __val) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot get the element wise and of non-integral and non-boolean value");

#if defined(__ARM_NEON)


#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__val](const_reference __v) { return static_cast<value_type>(__v && __val); });

#endif
}


template<class _Tp>
void tensor<_Tp>::bitwise_or_(const value_type __val) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");

    size_t __i = 0;
#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    if constexpr (std::is_same<value_type, int32_t>::value)
    {
        int32x4_t __val_vec = vdupq_n_s32(__val);
        for (; __i < __simd_end; __i += 4)
        {
            int32x4_t __data_vec   = vld1q_s32(reinterpret_cast<int32_t*>(&this->__data_[__i]));
            int32x4_t __result_vec = vorrq_s32(__data_vec, __val_vec);
            vst1q_s32(reinterpret_cast<int32_t*>(&this->__data_[__i]), __result_vec);
        }
    }
    else if constexpr (std::is_same<value_type, uint32_t>::value)
    {
        uint32x4_t __val_vec = vdupq_n_u32(__val);
        for (; __i < __simd_end; __i += 4)
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
void tensor<_Tp>::bitwise_xor_(const value_type __val) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");

    size_t __i = 0;
#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    if constexpr (std::is_same<value_type, int32_t>::value)
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
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot perform a bitwise not on non integral or boolean value");

    size_t __i = 0;
#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    if constexpr (std::is_same<value_type, int32_t>::value)
    {
        for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
        {
            int32x4_t __data_vec   = vld1q_s32(reinterpret_cast<const int32_t*>(&this->__data_[__i]));
            int32x4_t __result_vec = vmvnq_s32(__data_vec);
            vst1q_s32(reinterpret_cast<int32_t*>(&this->__data_[__i]), __result_vec);
        }
    }
    else if constexpr (std::is_same<value_type, uint32_t>::value)
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
void tensor<_Tp>::bitwise_and_(const value_type __val) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");

    size_t __i = 0;
#if defined(__ARM_NEON)
    constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    if constexpr (std::is_same<value_type, int32_t>::value)
    {
        int32x4_t __val_vec = vdupq_n_s32(reinterpret_cast<int32_t*>(&__val));
        for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
        {
            int32x4_t __data_vec   = vld1q_s32(reinterpret_cast<const int32_t*>(&this->__data_[__i]));
            int32x4_t __result_vec = vandq_s32(__data_vec, __val_vec);
            vst1q_s32(reinterpret_cast<int32_t*>(&this->__data_[__i]), __result_vec);
        }
    }
    else if constexpr (std::is_same<value_type, uint32_t>::value)
    {
        uint32x4_t __val_vec = vdupq_n_u32(reinterpret_cast<uint32_t*>(&__val));
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
    for (size_t i = 0; i < this->__shape_.size(); ++i)
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
tensor<_Tp> tensor<_Tp>::randomize(const shape_type& __sh, bool __bounded) {
    return this->clone().randomize_(__sh, __bounded);
}


template<class _Tp>
void tensor<_Tp>::randomize_(const shape_type& __sh, bool __bounded) {
    this->__check_is_scalar_type("template class must be a scalar type");

    size_t __s;
    if (this->__shape_ != __sh)
        __s = this->__computeSize(__sh);
    else
        __s = this->__data_.size();

    if (__s != this->__data_.size())
        this->__data_.resize(__s);

    std::random_device __rd;
    std::mt19937       __gen(__rd());

    std::uniform_real_distribution<float> __bounded_dist(0.0f, static_cast<float>(RAND_MAX));
    std::uniform_real_distribution<float> __unbounded_dist(0.0f, 1.0f);

    index_type __i = 0;

#if defined(__AVX__)
    const __m256 __scale = _mm256_set1_ps(__bounded ? static_cast<float>(RAND_MAX) : 1.0f);
    for (; __i + 8 <= static_cast<index_type>(__s); __i += 8)
    {
        __m256 __random_values =
            _mm256_setr_ps(__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen),
                           __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen));
        if (!__bounded)
            __random_values = _mm256_div_ps(__random_values, __scale);

        _mm256_storeu_ps(&this->__data_[__i], __random_values);
    }
#elif defined(__SSE__)
    const __m128 __scale = _mm_set1_ps(__bounded ? static_cast<float>(RAND_MAX) : 1.0f);
    for (; __i + 4 <= static_cast<index_type>(__s); __i += 4)
    {
        __m128 __random_values =
            _mm_setr_ps(__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen));
        if (!__bounded)
            __random_values = _mm_div_ps(__random_values, __scale);

        _mm_storeu_ps(&this->__data_[__i], __random_values);
    }
#elif defined(__ARM_NEON)
    const float32x4_t __scale = vdupq_n_f32(__bounded ? static_cast<float>(RAND_MAX) : 1.0f);
    for (; __i + 4 <= static_cast<index_type>(__s); __i += 4)
    {
        float32x4_t __random_values =
            vsetq_f32(__bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen), __bounded_dist(__gen));
        if (!__bounded)
            __random_values = vmulq_f32(__random_values, vrecpeq_f32(__scale));

        vst1q_f32(&this->__data_[__i], __random_values);
    }
#endif

    for (; __i < static_cast<index_type>(__s); __i++)
        this->__data_[__i] = value_type(__bounded ? __bounded_dist(__gen) : __unbounded_dist(__gen));
}

template<class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argmax_(index_type __dim) const {
    if (__dim < 0 || __dim >= this->__shape_.size())
        throw std::out_of_range("Dimension out of range in argmax");

    shape_type __ret_sh = this->__shape_;
    __ret_sh.erase(__ret_sh.begin() + __dim);
    tensor<index_type> __ret;
    __ret.__shape_ = __ret_sh;
    __ret.__data_.resize(this->__computeSize(__ret_sh), 0);

    index_type __outer_size = 1LL;
    index_type __inner_size = 1LL;

    index_type __i = 0;
    for (; __i < __dim; __i++)
        __outer_size *= this->__shape_[__i];

    for (__i = __dim + 1; __i < this->__shape_.size(); __i++)
        __inner_size *= this->__shape_[__i];

#if defined(__AVX2__)
    if constexpr (std::is_same_v<_Tp, float>)
    {
        for (__i = 0; __i < __outer_size; __i++)
        {
            for (index_type __j = 0; __j < __inner_size; __j++)
            {
                __m256  __max_vec       = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                __m256i __index_vec     = _mm256_setzero_si256();
                __m256i __increment     = _mm256_set1_epi32(1);
                __m256i __current_index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

                index_type __k = 0;
                for (; __k + 8 <= this->__shape_[__dim]; __k += 8)
                {
                    __m256 __data_vec =
                        _mm256_loadu_ps(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
                    __m256 __mask   = _mm256_cmp_ps(__data_vec, __max_vec, _CMP_GT_OQ);
                    __max_vec       = _mm256_blendv_ps(__max_vec, __data_vec, __mask);
                    __index_vec     = _mm256_blendv_epi8(__index_vec, __current_index, _mm256_castps_si256(__mask));
                    __current_index = _mm256_add_epi32(__current_index, __increment);
                }

                float   __max_values[8];
                int32_t __indices[8];
                _mm256_storeu_ps(__max_values, __max_vec);
                _mm256_storeu_si256((__m256i*) __indices, __index_vec);

                float      __max_value = __max_values[0];
                index_type __max_index = __indices[0];
                for (int __i = 1; __i < 8; __i++)
                {
                    if (__max_values[__i] > __max_value)
                    {
                        __max_value = __max_values[__i];
                        __max_index = __indices[__i];
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
            for (index_type __j = 0; __j < __inner_size; __j++)
            {
                float32x4_t __max_vec       = vdupq_n_f32(-std::numeric_limits<float>::infinity());
                uint32x4_t  __index_vec     = vdupq_n_u32(0);
                uint32x4_t  __increment     = vdupq_n_u32(1);
                uint32x4_t  __current_index = {0, 1, 2, 3};

                index_type __k = 0;
                for (; __k + 4 <= this->__shape_[__dim]; __k += 4)
                {
                    float32x4_t __data_vec =
                        vld1q_f32(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
                    uint32x4_t __mask = vcgtq_f32(__data_vec, __max_vec);
                    __max_vec         = vbslq_f32(__mask, __data_vec, __max_vec);
                    __index_vec       = vbslq_u32(__mask, __current_index, __index_vec);
                    __current_index   = vaddq_u32(__current_index, __increment);
                }

                float    __max_values[4];
                uint32_t __indices[4];
                vst1q_f32(__max_values, __max_vec);
                vst1q_u32(__indices, __index_vec);

                float      __max_value = __max_values[0];
                index_type __max_index = __indices[0];
                for (int __i = 1; __i < 4; __i++)
                {
                    if (__max_values[__i] > __max_value)
                    {
                        __max_value = __max_values[__i];
                        __max_index = __indices[__i];
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
#endif
    {
        for (__i = 0; __i < __outer_size; __i++)
        {
            index_type __j = 0;
            for (; __j < __inner_size; __j++)
            {
                index_type __max_index = 0;
                value_type __max_value = this->__data_[__i * this->__shape_[__dim] * __inner_size + __j];

                index_type __k = 1;
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
        throw std::out_of_range("Dimension out of range in argmax");

    shape_type __ret_sh = this->__shape_;
    __ret_sh.erase(__ret_sh.begin() + __dim);

    tensor __ret;
    __ret.__shape_ = __ret_sh;
    __ret.__data_.resize(this->__computeSize(__ret_sh), value_type(0));

    index_type __outer_size = 1;
    index_type __inner_size = 1;

    index_type __i = 0;
    for (; __i < __dim; __i++)
        __outer_size *= this->__shape_[__i];

    for (__i = __dim + 1; __i < static_cast<index_type>(this->__shape_.size()); __i++)
        __inner_size *= this->__shape_[__i];

#if defined(__AVX2__)
    if constexpr (std::is_same_v<_Tp, float>)
    {
        for (__i = 0; __i < __outer_size; __i++)
        {
            for (index_type __j = 0; __j < __inner_size; __j++)
            {
                __m256     __max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                index_type __k       = 0;
                for (; __k + 8 <= this->__shape_[__dim]; __k += 8)
                {
                    __m256 __data_vec =
                        _mm256_loadu_ps(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
                    __max_vec = _mm256_max_ps(__max_vec, __data_vec);
                }
                float __max_value = _mm256_reduce_max_ps(__max_vec);
                for (; __k < this->__shape_[__dim]; __k++)
                {
                    float __v   = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
                    __max_value = std::max(__max_value, __v);
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
            for (index_type __j = 0; __j < __inner_size; __j++)
            {
                float32x4_t __max_vec = vdupq_n_f32(-std::numeric_limits<float>::infinity());
                index_type  __k       = 0;
                for (; __k + 4 <= this->__shape_[__dim]; __k += 4)
                {
                    float32x4_t __data_vec =
                        vld1q_f32(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
                    __max_vec = vmaxq_f32(__max_vec, __data_vec);
                }
                float __max_value = vmaxvq_f32(__max_vec);
                for (; __k < this->__shape_[__dim]; __k++)
                {
                    float __v   = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
                    __max_value = std::max(max_value, __v);
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

                index_type __k = 1;
                for (; __k < this->__shape_[__dim]; __k++)
                {
                    value_type __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
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
void tensor<_Tp>::view(std::initializer_list<index_type> __sh) {
    index_type __s = this->__computeSize(__sh);
    if (__s != this->__data_.size())
        throw std::invalid_argument("Total elements do not match for new shape");

    this->__shape_ = __sh;
    this->__compute_strides();
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::cat(const std::vector<tensor<_Tp>>& __others, index_type __dim) const {
    for (const tensor& __t : __others)
    {
        index_type __i = 0;
        for (; __i < this->__shape_.size(); __i++)
        {
            if (__i != __dim && this->__shape_[__i] != __t.__shape_[__i])
                throw std::invalid_argument(
                    "Cannot concatenate tensors with different shapes along non-concatenation dimensions");
        }
    }

    shape_type __ret_sh = this->__shape_;

    for (const tensor& __t : __others)
        __ret_sh[__dim] += __t.__shape_[__dim];

    data_t __c;
    __c.reserve(this->__data_.size());
    __c.insert(__c.end(), this->__data_.begin(), this->__data_.end());

    for (const tensor& __t : __others)
        __c.insert(__c.end(), __t.__data_.begin(), __t.__data_.end());

    return __self(__c, __ret_sh);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::cumprod(index_type __dim) const {
    if (__dim == -1)
    {
        data_t __flat = this->__data_;
        data_t __ret(__flat.size());
        __ret[0] = __flat[0];

#if defined(__AVX2__)
        if constexpr (std::is_same_v<_Tp, float>)
        {
            size_t __i = 1;
            for (; __i + 8 <= __flat.size(); __i += 8)
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
            size_t __i = 1;
            for (; __i < __flat.size(); __i++)
                __ret[__i] = __ret[__i - 1] * __flat[__i];
        }
#else
        size_t __i = 1;
        for (; __i < __flat.size(); __i++)
            __ret[__i] = __ret[__i - 1] * __flat[__i];
#endif

        return __self(__ret, {__flat.size()});
    }
    else
    {
        if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
            throw std::invalid_argument("Invalid dimension provided.");

        data_t __ret(this->__data_);
        // TODO : compute_outer_size() implementation
        size_t __outer_size = this->__compute_outer_size(__dim);
        size_t __inner_size = this->__shape_[__dim];
        size_t __st         = this->__strides_[__dim];

#if defined(__AVX2__)
        if constexpr (std::is_same_v<_Tp, float>)
        {
            for (size_t __i = 0; __i < __outer_size; __i++)
            {
                size_t __base = __i * __st;
                __ret[__base] = __data_[__base];

                size_t __j = 1;
                for (; __j + 8 <= __inner_size; __j += 8)
                {
                    __m256 __prev   = _mm256_loadu_ps(&__ret[__base + __j - 1]);
                    __m256 __curr   = _mm256_loadu_ps(&__data_[__base + __j]);
                    __m256 __result = _mm256_mul_ps(__prev, __curr);
                    _mm256_storeu_ps(&__ret[__base + __j], __result);
                }
                for (; __j < __inner_size; __j++)
                {
                    size_t __curr = __base + __j;
                    __ret[__curr] = __ret[__base + __j - 1] * __data_[__curr];
                }
            }
        }
        else
        {
            for (size_t __i = 0; __i < __outer_size; ++__i)
            {
                size_t __base = __i * __st;
                __ret[__base] = __data_[__base];

                for (size_t __j = 1; __j < __inner_size; __j++)
                {
                    size_t __curr = __base + __j;
                    __ret[__curr] = __ret[__base + __j - 1] * __data_[__curr];
                }
            }
        }
#else
        size_t __i = 0;
        for (; __i < __outer_size; __i++)
        {
            size_t __base = __i * __st;
            __ret[__base] = __data_[__base];

            size_t __j = 1;
            for (; __j < __inner_size; __j++)
            {
                size_t __curr = __base + __j;
                __ret[__curr] = __ret[__base + __j - 1] * __data_[__curr];
            }
        }
#endif

        return __self(__ret, this->__shape_);
    }
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::slice(index_type                __dim,
                               std::optional<index_type> __start,
                               std::optional<index_type> __end,
                               index_type                __step) const {
    if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
        throw std::out_of_range("Dimension out of range.");

    tensor __ret;

    index_type __s       = this->__shape_[__dim];
    index_type __start_i = __start.value_or(0ULL);
    index_type __end_i   = __end.value_or(__s);

    if (__start_i < 0)
        __start_i += __s;

    if (__end_i < 0)
        __end_i += __s;

    __start_i = std::max(index_type(0ULL), std::min(__start_i, __s));
    __end_i   = std::max(index_type(0ULL), std::min(__end_i, __s));

    index_type __slice_size = (__end_i - __start_i + __step - 1) / __step;
    shape_type __ret_dims   = this->__shape_;
    __ret_dims[-__dim]      = __slice_size;
    __ret                   = tensor(__ret_dims);

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
        if constexpr (std::is_same_v<_Tp, float> && __step == 1)
        {
            const int  __vector_size = 4;
            index_type __vector_end  = __start_i + ((__end_i - __start_i) / __vector_size) * __vector_size;

            for (index_type __i = __start_i, __j = 0; __i < __vector_end; __i += __vector_size, __j += __vector_size)
            {
                float32x4_t __vec = vld1q_f32(&(this->__data_[__i]));
                vst1q_f32(&(__ret.__data_[__j]), __vec);
            }

            for (index_type __i = __vector_end, __j = __vector_end - __start_i; __i < __end_i; __i++, __j++)
                __ret.__data_[__j] = this->__data_[__i];
        }
        else
        {
#endif
            index_type __i = __start_i, __j = 0ULL;
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
tensor<_Tp> tensor<_Tp>::fmod(const value_type __val) const {
    return this->clone().fmod_(__val);
}

template<class _Tp>
void tensor<_Tp>::fmod_(const value_type __val) {
    this->__check_is_scalar_type("Cannot divide a non scalar type");
#if defined(__ARM_NEON)
    constexpr size_t _ARM64_REG_WIDTH = 4;
    constexpr size_t __simd_end       = this->__data_.size() - (this->__data_.size() - _ARM64_REG_WIDTH);

    float32x4_t __b = vdupq_n_f32(static_cast<float32_t>(__val));

    size_t __i = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __a         = vld1q_f32(&this->__data_[__i]);
        float32x4_t __div       = vdivq_f32(__a, __b);
        float32x4_t __floor_div = vrndq_f32(__div);
        float32x4_t __mult      = vmulq_f32(__floor_div, __b);
        float32x4_t __mod       = vsubq_f32(__a, __mult);

        vst1q_f32(&this->__data_[__i], __mod);
    }


    __i = __simd_end;
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = std::fmod(this->__data_[__i], __val);

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] =
            static_cast<value_type>(std::fmod(static_cast<double>(this->__data_[__i]), static_cast<double>(__val)));
#endif
}


template<class _Tp>
void tensor<_Tp>::fmod_(const tensor<_Tp>& __other) {
    this->__check_is_scalar_type("Cannot divide non scalar values");
    if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
        throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");

#if defined(__ARM_NEON)
    constexpr size_t _ARM64_REG_WIDTH = 4;
    constexpr size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    size_t __i = 0;
    for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __a         = vld1q_f32(&this->__data_[__i]);
        float32x4_t __b         = vld1q_f32(&__other[__i]);
        float32x4_t __div       = vdivq_f32(__a, __b);
        float32x4_t __floor_div = vrndq_f32(__div);
        float32x4_t __mult      = vmulq_f32(__floor_div, __b);
        float32x4_t __mod       = vsubq_f32(__a, __mult);

        vst1q_f32(&this->__data_[__i], __mod);
    }
    __i = __simd_end;
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = std::fmod(this->__data_[__i], __other[__i]);

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(
            std::fmod(static_cast<double>(this->__data_[__i]), static_cast<double>(__other[__i])));

#endif
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmax(const tensor& __other) const {
    return this->clone().fmax_(__other);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmax(const value_type __val) const {
    return this->clone().fmax_(__val);
}


template<class _Tp>
void tensor<_Tp>::fmax_(const value_type __val) {
#if defined(__ARM_NEON)
    const size_t _ARM64_REG_WIDTH = 4;
    const size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    float32x4_t __scalar_val = vdupq_n_f32(__val);

    for (size_t __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __a       = vld1q_f32(&this->__data_[__i]);
        float32x4_t __max_val = vmaxq_f32(__a, __scalar_val);
        vst1q_f32(&this->__data_[__i], __max_val);
    }

    for (size_t __i = __simd_end; __i < this->__data_.size(); __i++)
        this->__data_[__i] = std::fmax(this->__data_[__i], __val);

#else
    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__val](const_reference __v) { return std::fmax(__v, __val); });
#endif
}


template<class _Tp>
void tensor<_Tp>::fmax_(const tensor<value_type>& __other) {
    static_assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));

#if defined(__ARM_NEON)
    const size_t _ARM64_REG_WIDTH = 4;
    const size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (size_t __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __a       = vld1q_f32(&this->__data_[__i]);
        float32x4_t __b       = vld1q_f32(&(__other[__i]));
        float32x4_t __max_val = vmaxq_f32(__a, __b);
        vst1q_f32(&this->__data_[__i], __max_val);
    }

    for (size_t __i = __simd_end; __i < this->__data_.size(); __i++)
        this->__data_[__i] = std::fmax(this->__data_[__i], __other[__i]);

#else
    std::transform(this->__data.begin(), this->__data_.end(), __other.begin(), this->__data.begin(),
                   [](const_reference __v, const_reference __w) { return std::fmax(__v, __w); });

#endif
}


template<class _Tp>
size_t tensor<_Tp>::count_nonzero(index_type __dim) const {
    this->__check_is_scalar_type("Cannot compare a non-scalar value to zero");

    size_t __c = 0;

    if (__dim == -1)
    {
#pragma omp parallel
        {
            size_t __local_count = 0;

#ifdef __AVX__
            if constexpr (std::is_same_v<value_type, float>)
            {
                size_t __size = this->__data_.size();
                size_t __i    = 0;

                for (; __i + 8 <= __size; __i += 8)
                {
                    __m256 __vec          = _mm256_loadu_ps(&this->__data_[__i]);
                    __m256 __nonzero_mask = _mm256_cmp_ps(__vec, _mm256_setzero_ps(), _CMP_NEQ_OQ);
                    __local_count += _mm256_movemask_ps(__nonzero_mask);
                }
            }
#endif
#if defined(__ARM_NEON)
            if constexpr (std::is_same_v<value_type, float>)
            {
                size_t __size = this->__data_.size();
                size_t __i    = 0;

                for (; __i + 4 <= __size; __i += 4)
                {
                    float32x4_t __vec          = vld1q_f32(&this->__data_[__i]);
                    uint32x4_t  __nonzero_mask = vcgtq_f32(__vec, vdupq_n_f32(0.0f));
                    __local_count += vaddvq_u32(__nonzero_mask);
                }
            }
#endif
            for (size_t __j = __i; __j < this->__data_.size(); __j++)
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
        if (__dim < 0 || __dim >= static_cast<index_type>(__shape_.size()))
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

    shape_type __ret_sh = {this->__shape_[0], __other.shape()[1]};
    data_t     __ret_d(__ret_sh[0] * __ret_sh[1], 0);

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
                    for (int __jj = __j; __jj < std::min(static_cast<index_type>(__j + __blockSize), __ret_sh[1]);
                         __jj++)
                    {
                        value_type __sum = 0;
                        for (int __kk = __k;
                             __kk < std::min(static_cast<index_type>(__k + __blockSize), this->__shape_[1]); __kk++)
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
    const int __neonBlockSize = 4;

    for (int __i = 0; __i < __ret_sh[0]; __i += __neonBlockSize)
    {
        for (int __j = 0; __j < __ret_sh[1]; __j += __neonBlockSize)
        {
            for (int __k = 0; __k < this->__shape_[1]; __k += __neonBlockSize)
            {
                for (int __ii = __i; __ii < std::min(static_cast<index_type>(__i + __neonBlockSize), __ret_sh[0]);
                     __ii++)
                {
                    for (int __jj = __j; __jj < std::min(static_cast<index_type>(__j + __neonBlockSize), __ret_sh[1]);
                         __jj++)
                    {
                        float32x4_t sum_vec = vdupq_n_f32(0);

                        for (int __kk = __k;
                             __kk < std::min(static_cast<index_type>(__k + __neonBlockSize), this->__shape_[1]);
                             __kk += 4)
                        {
                            float32x4_t __a_vec = vld1q_f32(&this->__data_[__ii * this->__shape_[1] + __kk]);
                            float32x4_t __b_vec = vld1q_f32(&__other.__data_[__kk * __other.shape()[1] + __jj]);
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
tensor<_Tp> tensor<_Tp>::reshape(const shape_type __sh) const {
    data_t __d = this->__data_;
    size_t __s = this->__computeSize(__sh);

    if (__s != this->__data_.size())
        throw std::invalid_argument(
            "input shape must have size of elements equal to the current number of elements in the tensor data");

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
    float32x4_t __a = vld1q_f32(reinterpret_cast<const float*>(this->__data_.data()));
    float32x4_t __b = vld1q_f32(reinterpret_cast<const float*>(__other.storage().data()));

    float32x4_t __a_yzx = vextq_f32(__a, __a, 1);
    float32x4_t __b_yzx = vextq_f32(__b, __b, 1);

    float32x4_t __result = vsubq_f32(vmulq_f32(__a_yzx, __b), vmulq_f32(__a, __b_yzx));

    __result = vextq_f32(__result, __result, 3);

    vst1q_f32(reinterpret_cast<float*>(__ret.storage().data()), __result);
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
    __m256 __a = _mm256_loadu_ps(reinterpret_cast<const float*>(this->__data_.data()));
    __m256 __b = _mm256_loadu_ps(reinterpret_cast<const float*>(__other.storage().data()));

    __m256 __a_yzx = _mm256_permute_ps(__a, _MM_SHUFFLE(3, 0, 2, 1));
    __m256 __b_yzx = _mm256_permute_ps(__b, _MM_SHUFFLE(3, 0, 2, 1));

    __m256 __result = _mm256_sub_ps(_mm256_mul_ps(__a_yzx, __b), _mm256_mul_ps(__a, __b_yzx));

    __result = _mm256_permute_ps(__result, _MM_SHUFFLE(3, 0, 2, 1));

    _mm256_storeu_ps(reinterpret_cast<float*>(__ret.storage().data()), __result);
#endif
#if defined(__SSE__)
    __m128 __a = _mm_loadu_ps(reinterpret_cast<const float*>(this->__data_.data()));
    __m128 __b = _mm_loadu_ps(reinterpret_cast<const float*>(__other.storage().data()));

    __m128 __a_yzx = _mm_shuffle_ps(__a, __a, _MM_SHUFFLE(3, 0, 2, 1));
    __m128 __b_yzx = _mm_shuffle_ps(__b, __b, _MM_SHUFFLE(3, 0, 2, 1));

    __m128 __result = _mm_sub_ps(_mm_mul_ps(__a_yzx, __b), _mm_mul_ps(__a, __b_yzx));

    __result = _mm_shuffle_ps(__result, __result, _MM_SHUFFLE(3, 0, 2, 1));

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

    size_t __s = __tensor.storage().size();
    data_t __a;

    __a.reserve(__s);
    size_t __i = 0;

#ifdef __AVX__
    if constexpr (std::is_same_v<value_type, float>)
    {
        for (; __i + 8 <= __s; __i += 8)
        {
            __m256 __input     = _mm256_loadu_ps(&__tensor.storage()[__i]);
            __m256 __abs_value = _mm256_abs_ps(__input);
            _mm256_storeu_ps(&__a[__i], __abs_value);
        }
    }
#endif

#if defined(__SSE__)
    if constexpr (std::is_same_v<value_type, float>)
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
        __a.push_back(static_cast<value_type>(std::fabs(float(__tensor.storage()[__i]))));

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

        value_type __ret = 0;

#ifdef __CUDACC__
        if (this->__is_cuda_tensor && __other.__is_cuda_tensor)
        {
            thrust::device_vector<value_type> __d_result(1);
            thrust::transform(thrust::device, __this_data, __this_data + __size, __other_data, __d_result.begin(),
                              thrust::multiplies<value_type>());
            __ret = thrust::reduce(__d_result.begin(), __d_result.end());
        }
        else
#endif
        {
#if defined(__AVX__)
            if constexpr (std::is_same_v<value_type, float> && __size >= 8)
            {
                __m256 __sum = _mm256_setzero_ps();
                for (size_t __i = 0; __i < __size - 7; __i += 8)
                {
                    __m256 __a = _mm256_loadu_ps(__this_data + __i);
                    __m256 __b = _mm256_loadu_ps(__other_data + __i);
                    __sum      = _mm256_add_ps(__sum, _mm256_mul_ps(__a, __b));
                }
                __ret = _mm256_reduce_add_ps(__sum);
                for (size_t __i = __size - (__size % 8); __i < __size; __i++)
                    __ret += __this_data[__i] * __other_data[__i];
            }
            else
#elif defined(__SSE__)
            if constexpr (std::is_same_v<value_type, float> && __size >= 4)
            {
                __m128 __sum = _mm_setzero_ps();
                for (size_t __i = 0; __i < __size - 3; __i += 4)
                {
                    __m128 __a = _mm_loadu_ps(__this_data + __i);
                    __m128 __b = _mm_loadu_ps(__other_data + __i);
                    __sum      = _mm_add_ps(__sum, _mm_mul_ps(__a, __b));
                }
                __ret = _mm_reduce_add_ps(__sum);
                for (size_t __i = __size - (__size % 4); __i < __size; __i++)
                    __ret += __this_data[__i] * __other_data[__i];
            }
            else
#endif
            {
                __ret = std::inner_product(__this_data, __this_data + __size, __other_data, value_type(0));
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

    size_t __s = this->__data_.size();
    size_t __i = 0;
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
    if constexpr (std::is_same_v<value_type, float>)
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
    if constexpr (std::is_same_v<value_type, float>)
    {
        __m256 __zero = _mm256_setzero_ps();
        for (; __i + 8 <= __s; __i += 8)
        {
            __m256 __x      = _mm256_loadu_ps(&this->__data_[__i]);
            __m256 __result = _mm256_max_ps(__x, __zero);
            _mm256_storeu_ps(&this->__data_[__i], __result);
        }
    }
#elif defined(__ARM_NEON)
    if constexpr (std::is_same_v<value_type, float>)
    {
        const float32x4_t __vZero = vdupq_n_f32(0.0f);
        for (; __i + 4 <= __s; __i += 4)
        {
            float32x4_t __v = vld1q_f32(&this->__data_[__i]);
            __v             = vmaxq_f32(__v, __vZero);
            vst1q_f32(&this->__data_[__i], __v);
        }
    }
    else if constexpr (std::is_same_v<value_type, int32_t>)
    {
        const int32x4_t __vZero = vdupq_n_s32(0);
        for (; __i + 4 <= __s; __i += 4)
        {
            int32x4_t __v = vld1q_s32(&this->__data_[__i]);
            __v           = vmaxq_s32(__v, __vZero);
            vst1q_s32(&this->__data_[__i], __v);
        }
    }
#else
    for (__i = 0; __i < __s; __i++)
        this->__data_[__i] = std::max(this->__data_[__i], value_type(0));
#endif
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::transpose() const {
    if (this->__shape_.size() != 2)
        throw std::invalid_argument("Matrix transposition can only be done on 2D tensors");

    tensor __ret({this->__shape_[1], this->__shape_[0]});

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
    if constexpr (std::is_same_v<_Tp, float>)
    {
        for (index_type __i = 0; __i < __rows; __i += 4)
        {
            for (index_type __j = 0; j < __cols; __j += 4)
            {
                if (__i + 4 <= __rows && __j + 4 <= __cols)
                {
                    float32x4x4_t __input;
                    for (int __k = 0; __k < 4; __k++)
                        __input.val[__k] = vld1q_f32(&this->__data_[(__i + __k) * __cols + __j]);

                    float32x4x4_t __output = vld4q_f32(reinterpret_cast<const float*>(&__input));

                    for (int __k = 0; __k < 4; __k++)
                        vst1q_f32(&__ret.__data_[(__j + __k) * __rows + i], __output.val[__k]);
                }
                else
                {
                    for (index_type __ii = __i; __ii < std::min(__i + 4, __rows); __ii++)
                    {
                        for (index_type __jj = __j; __jj < std::min(__j + 4, __cols); __jj++)
                            __ret.at({__jj, __ii}) = this->at({__ii, __jj});
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
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argsort(index_type __d, bool __ascending) const {
    index_type __adjusted = (__d < 0LL) ? __d + this->__data_.size() : __d;

    if (__adjusted != 0LL)
        throw std::out_of_range("Invalid dimension for argsort: only 1D tensors are supported");

    index_type              __size = static_cast<index_type>(this->__data_.size());
    std::vector<index_type> __indices(__size);
    std::iota(__indices.begin(), __indices.end(), 0);

#if defined(__ARM_NEON)
    constexpr size_t simd_width = 4;
    size_t           i          = 0;

    if constexpr (std::is_same<value_type, float>::value)
    {
        for (; i + simd_width <= __size; i += simd_width)
        {
            float32x4_t data_vec = vld1q_f32(&this->__data_[i]);

            float32x2_t min1 = vpmin_f32(vget_low_f32(data_vec), vget_high_f32(data_vec));
            float32x2_t min2 = vpmin_f32(min1, min1);

            float32x4_t cmp_vec = vdupq_lane_f32(min2, 0);
            uint32x4_t  cmp_result;

            if (__ascending)
                cmp_result = vcltq_f32(data_vec, cmp_vec);
            else
                cmp_result = vcgtq_f32(data_vec, cmp_vec);

            for (int j = 0; j < simd_width; ++j)
                __indices[i + j] = (cmp_result[j] ? i + j : i + j + 1);
        }
    }

    for (; i < __size; ++i)
        __indices[i] = i;
#endif
    std::sort(__indices.begin(), __indices.end(), [&](index_type __a, index_type __b) {
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
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");

    static_assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));
#if defined(__ARM_NEON)
    const size_t _ARM64_REG_WIDTH = 4;
    const size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (size_t __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        uint32x4_t __data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
        uint32x4_t __other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));

        uint32x4_t __xor_vec = vandq_u32(__data_vec, __other_vec);

        vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __xor_vec);
    }

    for (size_t __i = __simd_end; __i < this->__data_.size(); __i++)
        this->__data_[__i] &= (__other[__i]);

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] &= __other[__i];
#endif
}


template<class _Tp>
void tensor<_Tp>::bitwise_or_(const tensor& __other) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");

    static_assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));

#if defined(__ARM_NEON)
    const size_t _ARM64_REG_WIDTH = 4;
    const size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (size_t __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        uint32x4_t __data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
        uint32x4_t __other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));

        uint32x4_t __xor_vec = vornq_u32(__data_vec, __other_vec);

        vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __xor_vec);
    }

    for (size_t __i = __simd_end; __i < this->__data_.size(); __i++)
        this->__data_[__i] |= (__other[__i]);

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] |= __other[__i];

#endif
}


template<class _Tp>
void tensor<_Tp>::bitwise_xor_(const tensor& __other) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");

    static_assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));
#if defined(__ARM_NEON)
    const size_t _ARM64_REG_WIDTH = 4;
    const size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (size_t __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        uint32x4_t __data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
        uint32x4_t __other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));

        uint32x4_t __xor_vec = veorq_u32(__data_vec, __other_vec);

        vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __xor_vec);
    }

    for (size_t __i = __simd_end; __i < this->__data_.size(); __i++)
        this->__data_[__i] ^= (__other[__i]);

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] ^= __other[__i];

#endif
}


template<class _Tp>
tensor<bool> tensor<_Tp>::logical_not() const {
    return this->clone().logical_not_();
}


template<class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const value_type __val) const {
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
tensor<_Tp> tensor<_Tp>::logical_xor(const value_type __val) const {
    return this->clone().logical_xor_(__val);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const tensor& __other) const {
    return this->clone().logical_and_(__other);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const value_type __val) const {
    return this->clone().logical_and_(__val);
}


template<class _Tp>
void tensor<_Tp>::logical_or_(const tensor<_Tp>& __other) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot get the element wise not of non-integral and non-boolean value");

    static_assert(this->__shape_ == __other.shape());
#if defined(__ARM_NEON)
    const size_t _ARM64_REG_WIDTH = 4;
    const size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (size_t __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        uint32x4_t __data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
        uint32x4_t __other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__other[__i]));

        uint32x4_t __or_vec = vornq_u32(__data_vec, __other_vec);

        vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __or_vec);
    }

    for (size_t __i = __simd_end; __i < this->__data_.size(); __i++)
        this->__data_[__i] = (this->__data_[__i] || __other[__i]);

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = (this->__data_[__i] || __other[__i]);

#endif
}


template<class _Tp>
void tensor<_Tp>::logical_xor_(const tensor<_Tp>& __other) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");

    static_assert(this->__shape_ == __other.shape());
#if defined(__ARM_NEON)
    const size_t _ARM64_REG_WIDTH = 4;
    const size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (size_t __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        uint32x4_t __data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
        uint32x4_t __other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));

        uint32x4_t __xor_vec = veorq_u32(__data_vec, __other_vec);

        vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __xor_vec);
    }

    for (size_t __i = __simd_end; __i < this->__data_.size(); __i++)
        this->__data_[__i] = (this->__data_[__i] && __other[__i]);

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = (this->__data_[__i] ^ __other[__i]);

#endif
}


template<class _Tp>
void tensor<_Tp>::logical_and_(const tensor<_Tp>& __other) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
        throw std::runtime_error("Cannot get the element-wise and of non-integral and non-boolean value");

    static_assert(this->__shape_ == __other.shape());

#if defined(__ARM_NEON)
    const size_t _ARM64_REG_WIDTH = 4;
    const size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (size_t __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        uint32x4_t __data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
        uint32x4_t __other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));

        uint32x4_t __and_vec = vandq_u32(__data_vec, __other_vec);

        vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), __and_vec);
    }

    for (size_t __i = __simd_end; __i < this->__data_.size(); __i++)
        this->__data_[__i] = (this->__data_[__i] && __other[__i]);

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = (this->__data_[__i] && __other[__i]);

#endif
}


template<class _Tp>
double tensor<_Tp>::mean() const {
    this->__check_is_integral_type("input must be of integral type to get the mean average");
    double __m = 0.0f;

    if (this->empty())
        return __m;

    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        __m += this->__data_[__i];

    return static_cast<double>(__m / static_cast<double>(this->__data_.size()));
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::pow(const tensor& __other) const {
    return this->clone().pow_(__other);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::pow(const value_type __val) const {
    return this->clone().pow_(__val);
}


template<class _Tp>
void tensor<_Tp>::pow_(const tensor& __other) {
    this->__check_is_integral_type("cannot get the power of a non integral value");

    static_assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));
#if defined(__ARM_NEON)
    const size_t _ARM64_REG_WIDTH = 4;
    const size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (size_t __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __base_vec = vld1q_f32(reinterpret_cast<const float*>(&this->__data_[__i]));
        float32x4_t __exp_vec  = vld1q_f32(reinterpret_cast<const float*>(&__other[__i]));

        float32x4_t __result_vec = {std::pow(vgetq_lane_f32(base_vec, 0), vgetq_lane_f32(exp_vec, 0)),
                                    std::pow(vgetq_lane_f32(base_vec, 1), vgetq_lane_f32(exp_vec, 1)),
                                    std::pow(vgetq_lane_f32(base_vec, 2), vgetq_lane_f32(exp_vec, 2)),
                                    std::pow(vgetq_lane_f32(base_vec, 3), vgetq_lane_f32(exp_vec, 3))};

        vst1q_f32(&this->__data_[__i], __result_vec);
    }

    for (size_t __i = simd_end; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(
            std::pow(static_cast<double>(this->__data_[__i]), static_cast<double>(__other[__i])));
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        this->__data_[__i] = static_cast<value_type>(std::pow(double(this->__data_[__i]), double(__other[__i])));
#endif
}


template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const tensor& __other) const {
    if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
        throw std::runtime_error("Cannot compare non-integral or scalar value");

    static_assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));

    std::vector<bool> __ret(this->__data_.size());
#if defined(__ARM_NEON)
    if constexpr (std::is_same_v<_Tp, float>)
    {
        const size_t __vec_size    = 4;
        const size_t __num_vectors = this->__data_.size() / __vec_size;
        const size_t __remaining   = this->__data_.size() % __vec_size;

        size_t __i = 0;
        for (; __i < __num_vectors * __vec_size; __i += __vec_size)
        {
            float32x4_t __data_vec1  = vld1q_f32(&this->__data_[__i]);
            float32x4_t __data_vec2  = vld1q_f32(&__other.__data_[__i]);
            uint32x4_t  __cmp_result = vcleq_f32(__data_vec1, __data_vec2);
            uint32_t    __mask       = vaddvq_u32(__cmp_result);

            __ret[__i]     = __mask & 1;
            __ret[__i + 1] = (__mask >> 8) & 1;
            __ret[__i + 2] = (__mask >> 16) & 1;
            __ret[__i + 3] = (__mask >> 24) & 1;
        }

        for (; __i < this->__data_.size(); __i++)
            __ret[__i] = (this->__data_[__i] <= __other.__data_[__i]);
    }
    else
    {
        size_t __i = 0;
        for (; __i < this->__data_.size(); __i++)
            __ret[__i] = (this->__data_[__i] <= __other.__data_[__i]);
    }
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        __ret[__i] = (this->__data_[__i] <= __other[__i]);
#endif
    return tensor<bool>(__ret, this->__shape_);
}


template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const value_type __val) const {
    if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
        throw std::runtime_error("Cannot compare non-integral or scalar value");

    std::vector<bool> __ret(this->__data_.size());

#if defined(__ARM_NEON)
    if constexpr (std::is_same_v<_Tp, float>)
    {
        const size_t __vec_size    = 4;
        const size_t __num_vectors = this->__data_.size() / __vec_size;
        const size_t __remaining   = this->__data_.size() % __vec_size;

        size_t __i = 0;
        for (; __i < __num_vectors * __vec_size; __i += __vec_size)
        {
            float32x4_t __data_vec   = vld1q_f32(&this->__data_[__i]);
            float32x4_t __val_vec    = vdupq_n_f32(__val);
            uint32x4_t  __cmp_result = vcleq_f32(__data_vec, __val_vec);
            uint32_t    __mask       = vaddvq_u32(__cmp_result);

            __ret[__i]     = __mask & 1;
            __ret[__i + 1] = (__mask >> 8) & 1;
            __ret[__i + 2] = (__mask >> 16) & 1;
            __ret[__i + 3] = (__mask >> 24) & 1;
        }

        for (; __i < this->__data_.size(); __i++)
            __ret[__i] = (this->__data_[__i] <= __val);
    }
    else
    {
        size_t __i = 0;
        for (; __i < this->__data_.size(); __i++)
            __ret[__i] = (this->__data_[__i] <= __val);
    }

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        __ret[__i] = (this->__data_[__i] <= __val) ? true : false;

#endif
    return tensor<bool>(__ret, this->__shape_);
}


template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const tensor& __other) const {
    if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
        throw std::runtime_error("Cannot compare non-integral or scalar value");

    assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));

    std::vector<bool> __ret(this->__data_.size());
#if defined(__ARM_NEON)
    if constexpr (std::is_same_v<_Tp, float>)
    {
        const size_t __vec_size    = 4;
        const size_t __num_vectors = this->__data_.size() / __vec_size;
        const size_t __remaining   = this->__data_.size() % __vec_size;

        size_t __i = 0;
        for (; __i < __num_vectors * __vec_size; __i += __vec_size)
        {
            float32x4_t __data_vec1  = vld1q_f32(&this->__data_[__i]);
            float32x4_t __data_vec2  = vld1q_f32(&__other.__data_[__i]);
            uint32x4_t  __cmp_result = vcgeq_f32(__data_vec1, __data_vec2);
            uint32_t    __mask       = vaddvq_u32(__cmp_result);

            __ret[__i]     = __mask & 1;
            __ret[__i + 1] = (__mask >> 8) & 1;
            __ret[__i + 2] = (__mask >> 16) & 1;
            __ret[__i + 3] = (__mask >> 24) & 1;
        }

        for (; __i < this->__data_.size(); __i++)
            __ret[__i] = (this->__data_[__i] >= __other.__data_[__i]);
    }
    else
    {
        size_t __i = 0;
        for (; __i < this->__data_.size(); __i++)
            __ret[__i] = (this->__data_[__i] >= __other.__data_[__i]);
    }

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        __ret[__i] = (this->__data_[__i] >= __other[__i]) ? true : false;

#endif
    return tensor<bool>(__ret, this->__shape_);
}


template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const value_type __val) const {
    if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
        throw std::runtime_error("Cannot compare non-integral or scalar value");

    std::vector<bool> __ret(this->__data_.size());
#if defined(__ARM_NEON)
    if constexpr (std::is_same_v<_Tp, float>)
    {
        const size_t __vec_size    = 4;
        const size_t __num_vectors = this->__data_.size() / __vec_size;
        const size_t __remaining   = this->__data_.size() % __vec_size;

        size_t      __i       = 0;
        float32x4_t __val_vec = vdupq_n_f32(__val);
        for (; __i < __num_vectors * __vec_size; __i += __vec_size)
        {
            float32x4_t __data_vec   = vld1q_f32(&this->__data_[__i]);
            uint32x4_t  __cmp_result = vcgeq_f32(__data_vec, __val_vec);
            uint32_t    __mask       = vaddvq_u32(__cmp_result);

            __ret[__i]     = __mask & 1;
            __ret[__i + 1] = (__mask >> 8) & 1;
            __ret[__i + 2] = (__mask >> 16) & 1;
            __ret[__i + 3] = (__mask >> 24) & 1;
        }

        for (; __i < this->__data_.size(); __i++)
            __ret[__i] = (this->__data_[__i] >= __val);
    }
    else
    {
        size_t __i = 0;
        for (; __i < this->__data_.size(); __i++)
            __ret[__i] = (this->__data_[__i] >= __val);
    }

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        __ret[__i] = (this->__data_[__i] >= __val) ? true : false;
#endif
    return tensor<bool>(__ret, this->__shape_);
}


template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const tensor& __other) const {
    if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
        throw std::runtime_error("Cannot compare non-integral or scalar value");

    static_assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0),
                  "equal : tensor shapes && sizes must match");

    std::vector<bool> __ret(this->__data_.size());
#if defined(__ARM_NEON)
    if constexpr (std::is_same_v<_Tp, float>)
    {
        constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

        size_t __i = 0;
        for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
        {
            float32x4_t __data_vec1  = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
            float32x4_t __data_vec2  = vld1q_f32(reinterpret_cast<const float32_t*>(&__other.__data_[__i]));
            uint32x4_t  __cmp_result = vceqq_f32(__data_vec1, __data_vec2);
            uint32_t    __mask       = vaddvq_u32(__cmp_result);

            __ret[__i]     = __mask & 1;
            __ret[__i + 1] = (__mask >> 8) & 1;
            __ret[__i + 2] = (__mask >> 16) & 1;
            __ret[__i + 3] = (__mask >> 24) & 1;
        }

        for (; __i < this->__data_.size(); __i++)
            __ret[__i] = (this->__data_[__i] == __other.__data_[__i]);
    }
    else
    {
        size_t __i = 0;
        for (; __i < this->__data_.size(); __i++)
            __ret[__i] = (this->__data_[__i] == __other.__data_[__i]);
    }

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        __ret[__i] = (this->__data_[__i] == __other[__i]);

#endif
    return tensor<bool>(__ret, this->__shape_);
}


template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const value_type __val) const {
    if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
        throw std::runtime_error("Cannot compare non-integral or scalar value");

    std::vector<bool> __ret(this->__data_.size());
#if defined(__ARM_NEON)
    if constexpr (std::is_same_v<_Tp, float>)
    {
        constexpr size_t __simd_end = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

        float32x4_t __val_vec = vdupq_n_f32(__val);

        size_t __i = 0;
        for (; __i < __simd_end; __i += _ARM64_REG_WIDTH)
        {
            float32x4_t __data_vec   = vld1q_f32(reinterpret_cast<const float32_t*>(&this->__data_[__i]));
            uint32x4_t  __cmp_result = vceqq_f32(__data_vec, __val_vec);
            uint32_t    __mask       = vaddvq_u32(__cmp_result);

            __ret[__i]     = __mask & 1;
            __ret[__i + 1] = (__mask >> 8) & 1;
            __ret[__i + 2] = (__mask >> 16) & 1;
            __ret[__i + 3] = (__mask >> 24) & 1;
        }

        for (; __i < this->__data_.size(); __i++)
            __ret[__i] = (this->__data_[__i] == __val);
    }
    else
    {
        size_t __i = 0;
        for (; __i < this->__data_.size(); __i++)
            __ret[__i] = (this->__data_[__i] == __val);
    }
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        __ret[__i] = (this->__data_[__i] >= __val) ? true : false;
#endif
    return tensor<bool>(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::sum(const index_type __axis) const {
    this->__check_is_scalar_type("Cannot reduce tensor with non scalar type");
    if (__axis < 0LL || __axis >= static_cast<index_type>(this->__shape_.size()))
        throw std::invalid_argument("Invalid axis for sum");

    shape_type __ret_sh = this->__shape_;
    __ret_sh[__axis]    = 1LL;

    index_type __ret_size = std::accumulate(__ret_sh.begin(), __ret_sh.end(), 1, std::multiplies<index_type>());
    data_t     __ret_data(__ret_size, value_type(0.0f));

#if defined(__ARM_NEON)
    if constexpr (std::is_same_v<_Tp, float>)
    {
        const index_type __axis_size  = this->__shape_[__axis];
        const index_type __outer_size = this->__compute_outer_size(__axis);
        const index_type __inner_size = this->size(0) / (__outer_size * __axis_size);

        for (index_type __outer = 0; __outer < __outer_size; __outer++)
        {
            for (index_type __inner = 0; __inner < __inner_size; __inner++)
            {
                float32x4_t __sum_vec = vdupq_n_f32(0.0f);
                index_type  __i       = __outer * __axis_size * __inner_size + __inner;
                index_type  __j       = 0;
                for (; __j + 4 <= __axis_size; __j += 4)
                {
                    float32x4_t __data_vec = vld1q_f32(&this->__data_[__i]);
                    __sum_vec              = vaddq_f32(__sum_vec, __data_vec);
                    __i += __inner_size * 4;
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
        index_type __i = 0;
        for (; __i < static_cast<index_type>(this->__data_.size()); __i++)
        {
            std::vector<index_type> __orig(this->__shape_.size());
            index_type              __index = __i;

            index_type __j = static_cast<index_type>(this->__shape_.size()) - 1;
            for (; __j >= 0; __j--)
            {
                __orig[__j] = __index % this->__shape_[__j];
                __index /= this->__shape_[__j];
            }

            __orig[__axis]         = 0LL;
            index_type __ret_index = 0LL;
            index_type __st        = 1LL;

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
tensor<_Tp> tensor<_Tp>::row(const index_type __index) const {
    if (this->__shape_.size() != 2)
        throw std::runtime_error("Cannot get a row from a non two dimensional tensor");

    if (this->__shape_[0] <= __index || __index < 0LL)
        throw std::invalid_argument("Index input is out of range");

    data_t __r(this->__data_.begin() + (this->__shape_[1] * __index),
               this->__data_.begin() + (this->__shape_[1] * __index + this->__shape_[1]));

    return __self(__r, {this->__shape_[1]});
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::col(const index_type __index) const {
    if (this->__shape_.size() != 2)
        throw std::runtime_error("Cannot get a column from a non two dimensional tensor");

    if (this->__shape_[1] <= __index || __index < 0LL)
        throw std::invalid_argument("Index input out of range");

    data_t __c;

    index_type __i = 0;
    for (; __i < this->__shape_[0]; __i++)
        __c.push_back(this->__data_[this->__compute_index({__i, __index})]);

    return __self(__c, {this->__shape_[0]});
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::ceil() const {
    this->__check_is_scalar_type("Cannot get the ceiling of a non scalar value");
    data_t __ret(this->__data_.size());

#if defined(__ARM_NEON)
    const size_t _ARM64_REG_WIDTH = 4;
    const size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    for (size_t __i = 0; __i < simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data   = vld1q_f32(&this->__data_[__i]);
        float32x4_t __ceiled = vrndpq_f32(__data);
        vst1q_f32(&__ret[__i], __ceiled);
    }

    for (size_t __i = __simd_end; __i < this->__data_.size(); __i++)
        __ret[__i] = std::ceil(this->__data_[__i]);

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        __ret[__i] = std::ceil(this->__data_[__i]);

#endif
    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::floor() const {
    this->__check_is_scalar_type("Cannot get the floor of a non scalar value");
    data_t __ret(this->__data_.size());

#if defined(__ARM_NEON)
    const size_t _ARM64_REG_WIDTH = 4;
    const size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    float32x4_t zero = vdupq_n_f32(0.0f);

    for (size_t __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data    = vld1q_f32(&this->__data_[__i]);
        float32x4_t __floored = vrndmq_f32(__data);
        vst1q_f32(&__ret[__i], __floored);
    }

    for (size_t __i = __simd_end; __i < this->__data_.size(); __i++)
        __ret[__i] = std::floor(this->__data_[__i]);

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
        __ret[__i] = std::floor(this->__data_[__i]);

#endif
    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::clamp(const_pointer __min_val, const_pointer __max_val) const {
    return this->clone().clamp_(__min_val, _max_val);
}


template<class _Tp>
void tensor<_Tp>::clamp_(const_pointer __min_val, const_pointer __max_val) {
#if defined(__AVX2__)
    const size_t __simd_end = this->__data_.size() - (this->__data_.size() % _AVX_REG_WIDTH);

    __m256 __min_vec = _mm256_set1_ps(__min_val ? *__min_val : std::numeric_limits<_Tp>::lowest());
    __m256 __max_vec = _mm256_set1_ps(__max_val ? *__max_val : std::numeric_limits<_Tp>::max());

    for (size_t __i = 0; __i < __simd_end; __i += _AVX_REG_WIDTH)
    {
        __m256 __data_vec = _mm256_loadu_ps(&this->__data_[__i]);
        __m256 __clamped  = _mm256_min_ps(_mm256_max_ps(data_vec, __min_vec), __max_vec);
        _mm256_storeu_ps(&this->__data_[__i], __clamped);
    }

    for (size_t __i = __simd_end; __i < this->__data_.size(); __i++)
    {
        if (__min_val)
            this->__data_[__i] = std::max(*__min_val, this->__data_[__i]);

        if (__max_val)
            this->__data_[__i] = std::min(*__max_val, this->__data_[__i]);
    }
#elif defined(__ARM_NEON)
    const size_t _ARM64_REG_WIDTH = 4;
    const size_t __simd_end       = this->__data_.size() - (this->__data_.size() % _ARM64_REG_WIDTH);

    float32x4_t __min_vec = vdupq_n_f32(__min_val ? *__min_val : std::numeric_limits<_Tp>::lowest());
    float32x4_t __max_vec = vdupq_n_f32(__max_val ? *__max_val : std::numeric_limits<_Tp>::max());

    for (size_t __i = 0; __i < __simd_end; __i += _ARM64_REG_WIDTH)
    {
        float32x4_t __data_vec = vld1q_f32(&this->__data_[__i]);
        float32x4_t __clamped  = vminq_f32(vmaxq_f32(__data_vec, __min_vec), __max_vec);
        vst1q_f32(&this->__data_[__i], __clamped);
    }

    for (size_t __i = __simd_end; __i < this->__data_.size(); __i++)
    {
        if (__min_val)
            this->__data_[__i] = std::max(*__min_val, this->__data_[__i]);

        if (__max_val)
            this->__data_[__i] = std::min(*__max_val, this->__data_[__i]);
    }
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        if (__min_val)
            this->__data_[__i] = std::max(*__min_val, this->__data_[__i]);

        if (__max_val)
            this->__data_[__i] = std::min(*__max_val, this->__data_[__i]);
    }
#endif
}
