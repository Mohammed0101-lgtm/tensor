#pragma once

#include "tensorbase.hpp"

template <>
class tensor<bool>;  // explicit instantiation

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
    if (__equal_shape(this->__shape_, shape_type({this->__shape_[0], this->__shape_[1], 1})) ||
        __equal_shape(this->__shape_, shape_type({1, this->__shape_[0], this->__shape_[1]})))
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
};  // tensor<bool>