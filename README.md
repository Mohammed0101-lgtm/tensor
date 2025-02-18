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


