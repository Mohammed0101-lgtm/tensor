# Tensor Library

A lightweight, header-only C++ Tensor library designed for efficient numerical computations and matrix operations. This library provides a flexible and intuitive interface for handling multi-dimensional arrays, enabling users to perform mathematical operations with ease. No gradient tracking or auto-differentiation is supported.

## Features

Header-Only: Implemented entirely in .hpp files, making it easy to integrate into any project without compilation.

Multi-Dimensional Support: Work with tensors of arbitrary dimensions.

Element-Wise Operations: Perform addition, subtraction, multiplication, and division on tensors.

Matrix Operations: Includes dot product, matrix multiplication, transposition, and reshaping.

Efficient Memory Management: Optimized for performance, minimizing unnecessary copies.

Intuitive API: Simple and expressive syntax for ease of use.

No External Dependencies: Works with standard C++ libraries.

## Installation & Usage

Since this is a header-only library, no separate compilation is required. Simply include the necessary header files in your project and start using the tensor functionalities.

1. Clone the Repository
```bash
 git clone https://github.com/Mohammed0101-lgtm/tensor
```

2. Include the Headers in Your Project
```cpp
#include "src/tensor.hpp"
```

3. Example Usage

Creating a Tensor

```cpp
#include <iostream>
#include "src/tensor.hpp"

int main() {
    Tensor<float> t({2, 3}, {1, 2, 3, 4, 5, 6}); // 2x3 Tensor
    t.print();
    return 0;
}

Performing Operations

tensor<float> A({2, 2}, {1, 2, 3, 4});
tensor<float> B({2, 2}, {5, 6, 7, 8});
tensor<float> C = A + B; // Element-wise addition
C.print();
```

## API Reference

Tensor Class

```cpp
template<typename T>
class tensor {
public:
    Tensor(std::vector<size_t> shape, std::vector<T> data);
    void print() const;
    tensor<T> operator+(const tensor<T>& other) const;
    tensor<T> operator-(const tensor<T>& other) const;
    tensor<T> operator*(const tensor<T>& other) const;
    tensor<T> operator/(const tensor<T>& other) const;
    tensor<T> matmul(const tensor<T>& other) const;
    tensor<T> transpose() const;
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
  mutable data_t                  __data_;
  mutable shape_type              __shape_;
  mutable std::vector<index_type> __strides_;
  Device                          __device_;
  bool                            __is_cuda_tensor_ = false;

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
  bool operator==(const tensor& __other) const;
  bool operator!=(const tensor& __other) const;
  tensor<bool>& operator!() const;
  tensor operator+(const tensor& __other) const;
  tensor operator-(const tensor& __other) const;
  tensor operator+(const value_type __val) const;
  tensor operator-(const value_type __val) const;
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
  tensor& operator=(const tensor<_Tp>&) = default; 
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
  tensor get_minor(index_type __a, index_type __b) const;
  tensor expand_as(shape_type __sh, index_type __dim) const;
  tensor lcm(const tensor& __other) const;
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
  tensor& asinh_();
  const tensor& asinh_() const;
  tensor& ceil_() const;
  tensor& floor_() const;
  tensor& relu_() const;
  tensor& clamp_(const_pointer __min_val = nullptr, const_pointer __max_val = nullptr) const;
  tensor& logical_not_() const { return this->bitwise_not_(); }
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
  tensor& resize_as_(const shape_type __sh) const;
  tensor& dist_(const tensor& __other) const;
  tensor& dist_(const value_type __val) const;
  tensor& maximum_(const tensor& __other) const;
  tensor& maximum_(const value_type __val) const;
  tensor& remainder_(const value_type __val) const;
  tensor& remainder_(const tensor& __other) const;
  tensor& fill_(const value_type __val) const;
  tensor& fill_(const tensor& __other) const;
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
  void print() const;
  tensor<index_type> argmax_(index_type __dim) const;
  tensor<index_type> argsort(index_type __dim = -1, bool __ascending = true) const;
};  

```

## Key Methods

```cpp
Tensor(std::vector<size_t> shape, std::vector<T> data)
```
Initializes a tensor with a given shape and data.

```cpp
void print() const
```
Prints the tensor contents.

operator+, operator-, operator*, operator/: Perform element-wise operations.

```cpp
matmul(const Tensor<T>& other) 
```
Performs matrix multiplication.

```cpp
transpose()
``` 
Returns the transposed tensor.

* Performance Considerations

- Uses row-major storage for efficient cache utilization.

- Avoids unnecessary copies by using move semantics.

- Operations are optimized for multi-dimensional computations.

* Roadmap

- Support for slicing and indexing operations.

- Addition of broadcasting rules for element-wise operations.

- More optimized parallelized computations using multi-threading.

* Contributing

Contributions are welcome! Feel free to submit pull requests, report issues, or suggest enhancements.

## License

MIT License


