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
template<class _Tp>
class tensor
{
 public:
  tensor() = default;

  explicit tensor(const std::vector<long long>& __sh, const _Tp& __v, Device __d = Device::CPU);
  explicit tensor(const std::vector<long long>& __sh, Device __d = Device::CPU);
  explicit tensor(const std::vector<_Tp>& __d, const std::vector<long long>& __sh, Device __dev = Device::CPU);

  tensor(const tensor& __t);
  tensor(tensor&& __t) noexcept;
  tensor(const std::vector<long long>& __sh, std::initializer_list<_Tp> init_list, Device __d = Device::CPU);
  tensor(const std::vector<long long>& __sh, const tensor& __other);

 public:
  ~tensor();
  std::vector<_Tp> storage() const noexcept;
  std::__1::__wrap_iter<_Tp *>  begin() noexcept;
  std::__1::__wrap_iter<_Tp *>  end() noexcept;
  std::__1::__wrap_iter<const _Tp *> begin() const noexcept;
  std::__1::__wrap_iter<const _Tp *> end() const noexcept;
  std::reverse_iterator<std::__wrap_iter<const _Tp *>> rbegin() noexcept;
  std::reverse_iterator<std::__wrap_iter<const _Tp *>> rend() noexcept;
  std::reverse_iterator<const std::__wrap_iter<const _Tp *>> rbegin() const noexcept;
  std::reverse_iterator<const std::__wrap_iter<const _Tp *>> rend() const noexcept;
  tensor<long long>   long_() const;
  tensor<int32_t>   int32_() const;
  tensor<uint32_t>  uint32_() const;
  tensor<ulong long>  unsigned_long_() const;
  tensor<float32_t> float32_() const;
  tensor<float64_t> double_() const;
  std::vector<long long> shape() const noexcept;
  std::vector<long long> strides() const noexcept;
  Device device() const noexcept { return this->__device_; }
  size_t n_dims() const noexcept;
  long long size(const long long __dim) const;
  long long capacity() const noexcept;
  long long count_nonzero(long long __dim = -1) const;
  long long lcm() const;
  long long hash() const;
  _Tp& at(std::vector<long long> __idx);
  _Tp& operator[](const long long __in);
  const _Tp& at(const std::vector<long long> __idx) const;
  const _Tp& operator[](const long long __in) const;
  bool empty() const;
  tensor<bool> bool_() const;
  tensor<bool> logical_not() const;
  tensor<bool> logical_or(const _Tp __val) const;
  tensor<bool> logical_or(const tensor<_Tp>& __other) const;
  tensor<bool> less_equal(const tensor<_Tp>& __other) const;
  tensor<bool> less_equal(const _Tp __val) const;
  tensor<bool> greater_equal(const tensor<_Tp>::tensor_reference __other) const;
  tensor<bool> greater_equal(const _Tp __val) const;
  tensor<bool> equal(const tensor<_Tp>& __other) const;
  tensor<bool> equal(const _Tp __val) const;
  tensor<bool> not_equal(const tensor<_Tp>& __other) const;
  tensor<bool> not_equal(const _Tp __val) const;
  tensor<bool> less(const tensor<_Tp>& __other) const;
  tensor<bool> less(const _Tp __val) const;
  tensor<bool> greater(const tensor<_Tp>& __other) const;
  tensor<bool> greater(const _Tp __val) const;
  bool operator==(const tensor<_Tp>& __other) const;
  bool operator!=(const tensor<_Tp>& __other) const;
  tensor<bool>& operator!() const;
  tensor<_Tp> operator+(const tensor<_Tp>& __other) const;
  tensor<_Tp> operator-(const tensor<_Tp>& __other) const;
  tensor<_Tp> operator+(const _Tp __val) const;
  tensor<_Tp> operator-(const _Tp __val) const;
  tensor<_Tp>& operator-=(const tensor<_Tp>& __other) const;
  tensor<_Tp>& operator+=(const tensor<_Tp>& __other) const;
  tensor<_Tp>& operator*=(const tensor<_Tp>& __other) const;
  tensor<_Tp>& operator/=(const tensor<_Tp>& __other) const;
  tensor<_Tp>& operator+=(const _Tp& __val) const;
  tensor<_Tp>& operator-=(const _Tp& __val) const;
  tensor<_Tp>& operator/=(const _Tp& __val) const;
  tensor<_Tp>& operator*=(const _Tp& __val) const;
  tensor<_Tp>& operator=(const tensor<_Tp>& __other) const;
  tensor<_Tp>& operator=(tensor<_Tp>&& __other) const noexcept;
  tensor<_Tp>& operator=(const tensor<_Tp><_Tp>&) = default; 
  tensor<_Tp>
  slice(long long __dim, std::optional<long long> __start, std::optional<long long> __end, long long __step) const;
  tensor<_Tp> fmax(const tensor<_Tp>& __other) const;
  tensor<_Tp> fmax(const _Tp __val) const;
  tensor<_Tp> fmod(const tensor<_Tp>& __other) const;
  tensor<_Tp> fmod(const _Tp __val) const;
  tensor<_Tp> frac() const;
  tensor<_Tp> log() const;
  tensor<_Tp> log10() const;
  tensor<_Tp> log2() const;
  tensor<_Tp> exp() const;
  tensor<_Tp> sqrt() const;
  tensor<_Tp> sum(const long long __axis) const;
  tensor<_Tp> row(const long long __index) const;
  tensor<_Tp> col(const long long __index) const;
  tensor<_Tp> ceil() const;
  tensor<_Tp> floor() const;
  tensor<_Tp> clone() const;
  tensor<_Tp> clamp(const _Tp* __min_val = nullptr, const _Tp* __max_val = nullptr) const;
  tensor<_Tp> cos() const;
  tensor<_Tp> cosh() const;
  tensor<_Tp> acos() const;
  tensor<_Tp> acosh() const;
  tensor<_Tp> tan() const;
  tensor<_Tp> tanh() const;
  tensor<_Tp> atan() const;
  tensor<_Tp> atanh() const;
  tensor<_Tp> sin() const;
  tensor<_Tp> sinc() const;
  tensor<_Tp> sinh() const;
  tensor<_Tp> asin() const;
  tensor<_Tp> asinh() const;
  tensor<_Tp> abs() const;
   logical_xor(const tensor<_Tp>& __other) const;
  tensor<_Tp> logical_xor(const _Tp __val) const;
  tensor<_Tp> logical_and(const tensor<_Tp>& __other) const;
  tensor<_Tp> logical_and(const _Tp __val) const;
  tensor<_Tp> bitwise_not() const;
  tensor<_Tp> bitwise_and(const _Tp __val) const;
  tensor<_Tp> bitwise_and(const tensor<_Tp>& __other) const;
  tensor<_Tp> bitwise_or(const _Tp __val) const;
  tensor<_Tp> bitwise_or(const tensor<_Tp>& __other) const;
  tensor<_Tp> bitwise_xor(const _Tp __val) const;
  tensor<_Tp> bitwise_xor(const tensor<_Tp>& __other) const;
  tensor<_Tp> bitwise_left_shift(const int __amount) const;
  tensor<_Tp> bitwise_right_shift(const int __amount) const;
  tensor<_Tp> matmul(const tensor<_Tp>& __other) const;
  tensor<_Tp> reshape(const std::vector<long long> __shape) const;
  tensor<_Tp> reshape_as(const tensor<_Tp>& __other) const;
  tensor<_Tp> cross_product(const tensor<_Tp>& __other) const;
  tensor<_Tp> absolute(const tensor<_Tp>& __tensor) const;
  tensor<_Tp> dot(const tensor<_Tp>& __other) const;
  tensor<_Tp> relu() const;
  tensor<_Tp> transpose() const;
  tensor<_Tp> fill(const _Tp __val) const;
  tensor<_Tp> fill(const tensor<_Tp>& __other) const;
  tensor<_Tp> resize_as(const std::vector<long long> __sh) const;
  tensor<_Tp> all() const;
  tensor<_Tp> any() const;
  tensor<_Tp> det() const;
  tensor<_Tp> square() const;
  tensor<_Tp> sigmoid() const;
  tensor<_Tp> clipped_relu() const;
  tensor<_Tp> sort(long long __dim, bool __descending = false) const;
  tensor<_Tp> remainder(const _Tp __val) const;
  tensor<_Tp> remainder(const tensor<_Tp>& __other) const;
  tensor<_Tp> maximum(const tensor<_Tp>& __other) const;
  tensor<_Tp> maximum(const _Tp& __val) const;
  tensor<_Tp> dist(const tensor<_Tp>& __other) const;
  tensor<_Tp> dist(const _Tp __val) const;
  tensor<_Tp> squeeze(long long __dim) const;
  tensor<_Tp> negative() const;
  tensor<_Tp> repeat(const std::vector<_Tp>& __d) const;
  tensor<_Tp> permute(const long long __dim) const;
  tensor<_Tp> log_softmax(const long long __dim) const;
  tensor<_Tp> gcd(const tensor<_Tp>& __other) const;
  tensor<_Tp> gcd(const _Tp __val) const;
  tensor<_Tp> pow(const tensor<_Tp>& __other) const;
  tensor<_Tp> pow(const _Tp __val) const;
  tensor<_Tp> cumprod(long long __dim = -1) const;
  tensor<_Tp> cat(const std::vector<tensor<_Tp>>& _others, long long _dim) const;
  tensor<_Tp> argmax(long long __dim) const;
  tensor<_Tp> unsqueeze(long long __dim) const;
  tensor<_Tp> zeros(const std::vector<long long>& __sh);
  tensor<_Tp> ones(const std::vector<long long>& __sh);
  tensor<_Tp> randomize(const std::vector<long long>& __sh, bool __bounded = false);
  tensor<_Tp> get_minor(long long __a, long long __b) const;
  tensor<_Tp> expand_as(std::vector<long long> __sh, long long __dim) const;
  tensor<_Tp> lcm(const tensor<_Tp>& __other) const;
  double mean() const;
  double median(const long long __dim) const;
  double mode(const long long __dim) const;
  tensor<_Tp>& push_back(_Tp __v) const;
  tensor<_Tp>& pop_back() const;
  tensor<_Tp>& sqrt_() const;
  tensor<_Tp>& exp_() const;
  tensor<_Tp>& log2_() const;
  tensor<_Tp>& log10_() const;
  tensor<_Tp>& log_() const;
  tensor<_Tp>& frac_() const;
  tensor<_Tp>& fmod_(const tensor<_Tp>& __other) const;
  tensor<_Tp>& fmod_(const _Tp __val) const;
  tensor<_Tp>& cos_() const;
  tensor<_Tp>& cosh_() const;
  tensor<_Tp>& acos_() const;
  tensor<_Tp>& acosh_() const;
  tensor<_Tp>& tan_() const;
  tensor<_Tp>& tanh_() const;
  tensor<_Tp>& atan_() const;
  tensor<_Tp>& atanh_() const;
  tensor<_Tp>& sin_() const;
  tensor<_Tp>& sinh_() const;
  tensor<_Tp>& asin_() const;
  tensor<_Tp>& asinh_();
  const tensor<_Tp>& asinh_() const;
  tensor<_Tp>& ceil_() const;
  tensor<_Tp>& floor_() const;
  tensor<_Tp>& relu_() const;
  tensor<_Tp>& clamp_(const _Tp* __min_val = nullptr, const _Tp* __max_val = nullptr) const;
  tensor<_Tp>& logical_not_() const { return this->bitwise_not_(); }
  tensor<_Tp>& logical_or_(const tensor<_Tp>& __other) const;
  tensor<_Tp>& logical_or_(const _Tp __val) const;
  tensor<_Tp>& logical_xor_(const tensor<_Tp>& __other) const;
  tensor<_Tp>& logical_xor_(const _Tp __val) const;
  tensor<_Tp>& logical_and_(const tensor<_Tp>& __other) const;
  tensor<_Tp>& logical_and_(const _Tp __val) const;
  tensor<_Tp>& abs_() const;
  tensor<_Tp>& log_softmax_(const long long __dim) const;
  tensor<_Tp>& permute_(const long long __dim) const;
  tensor<_Tp>& repeat_(const std::vector<_Tp>& __d) const;
  tensor<_Tp>& negative_() const;
  tensor<_Tp>& transpose_() const;
  tensor<_Tp>& unsqueeze_(long long __dim) const;
  tensor<_Tp>& squeeze_(long long __dim) const;
  tensor<_Tp>& resize_as_(const std::vector<long long> __sh) const;
  tensor<_Tp>& dist_(const tensor<_Tp>& __other) const;
  tensor<_Tp>& dist_(const _Tp __val) const;
  tensor<_Tp>& maximum_(const tensor<_Tp>& __other) const;
  tensor<_Tp>& maximum_(const _Tp __val) const;
  tensor<_Tp>& remainder_(const _Tp __val) const;
  tensor<_Tp>& remainder_(const tensor<_Tp>& __other) const;
  tensor<_Tp>& fill_(const _Tp __val) const;
  tensor<_Tp>& fill_(const tensor<_Tp>& __other) const;
  tensor<_Tp>& sigmoid_() const;
  tensor<_Tp>& clipped_relu_(const _Tp __clip_limit) const;
  tensor<_Tp>& square_() const;
  tensor<_Tp>& pow_(const tensor<_Tp>& __other);
  tensor<_Tp>& pow_(const _Tp __val);
  tensor<_Tp>& sinc_() const;
  tensor<_Tp>& bitwise_left_shift_(const int __amount);
  tensor<_Tp>& bitwise_right_shift_(const int __amount);
  tensor<_Tp>& bitwise_and_(const _Tp __val);
  tensor<_Tp>& bitwise_and_(const tensor<_Tp>& __other);
  tensor<_Tp>& bitwise_or_(const _Tp __val);
  tensor<_Tp>& bitwise_or_(const tensor<_Tp>& __other);
  tensor<_Tp>& bitwise_xor_(const _Tp __val);
  tensor<_Tp>& bitwise_xor_(const tensor<_Tp>& __other);
  tensor<_Tp>& bitwise_not_();
  tensor<_Tp>& view(std::initializer_list<long long> __new_sh);
  tensor<_Tp>& fmax_(const tensor<_Tp>& __other);
  tensor<_Tp>& fmax_(const _Tp __val);
  tensor<_Tp>& randomize_(const std::vector<long long>& __sh, bool __bounded = false);
  tensor<_Tp>& zeros_(std::vector<long long> __sh = {});
  tensor<_Tp>& ones_(std::vector<long long> __sh = {});
  void print() const;
  tensor<long long> argmax_(long long __dim) const;
  tensor<long long> argsort(long long __dim = -1, bool __ascending = true) const;
};  

```

## Key Methods

```cpp
tensor(std::vector<size_t> shape, std::vector<T> data)
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
Performs matrix transposition.

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


