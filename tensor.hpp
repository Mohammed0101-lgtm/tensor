#include <iostream>
#include <stdexcept>
#include <random>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <array>
#include <type_traits>

#include <__iterator/iterator_traits.h>
#include <__iterator/reverse_iterator.h>
#include <__iterator/wrap_iter.h>


/*
  tensor synopsis

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
  typedef const pointer           const_pointer;
  typedef std::vector<value_type> data_container;
  typedef const data_container    const_data_container;

  typedef std::__wrap_iter<typename std::allocator_traits<std::allocator<_Tp>>::pointer>       __iterator;
  typedef std::__wrap_iter<typename std::allocator_traits<std::allocator<_Tp>>::const_pointer> __const_iterator;
  typedef std::reverse_iterator<__iterator>                                                    reverse_iterator;
  typedef std::reverse_iterator<__const_iterator>                                              const_reverse_iterator;


 private:
  data_container          __data_;
  shape_type              __shape_;
  std::vector<index_type> __strides_;

 public:
  tensor() = default;

  explicit tensor(const shape_type __sh, const value_type __v);
  explicit tensor(const shape_type __sh);
  explicit tensor(const data_container __d, const shape_type __sh);
  explicit tensor(const tensor& __t);

  tensor(tensor&& __t) noexcept;
  tensor(const shape_type __sh, std::initializer_list<value_type> init_list);
  tensor(const shape_type& __sh, const tensor& __other);

 private:
  class __destroy_tensor
  {
   public:
    __destroy_tensor(tensor& __tens);
    void operator()();

   private:
    tensor& __tens_;
  };

 public:
  ~tensor();

  tensor& operator=(const tensor& __other);
  tensor& operator=(tensor&& __other) noexcept;

  __const_iterator begin() const;
  __const_iterator end() const;

  const_reverse_iterator rbegin() const;
  const_reverse_iterator rend() const;

  data_container storage() const;
  shape_type     shape() const;
  shape_type     strides() const;
  size_t         n_dims() const;

  size_t size(const index_type __dim) const;

  size_t          capacity() const noexcept;
  reference       at(shape_type __idx);
  const_reference at(const shape_type __idx) const;

  reference operator[](const size_t __in);

  const_reference operator[](const size_t __in) const;

  tensor operator+(const tensor& __other) const;
  tensor operator-(const tensor& __other) const;
  tensor operator-=(const tensor& __other) const;
  tensor operator+=(const tensor& __other) const;
  tensor operator+(const value_type _scalar) const;
  tensor operator-(const value_type _scalar) const;
  tensor operator+=(const_reference __scalar) const;
  tensor operator-=(const_reference __scalar) const;

  bool operator==(const tensor& __other) const;
  bool operator!=(const tensor& __other) const { return !(*this == __other); }

  bool empty() const { return this->__data_.empty(); }

  double mean() const;

  tensor<bool> logical_not() const;
  tensor<bool> logical_or(const value_type __val) const;
  tensor<bool> logical_or(const tensor& __other) const;

  tensor<bool> less_equal(const tensor& __other) const;
  tensor<bool> less_equal(const value_type __val) const;

  tensor<bool> greater_equal(const tensor& __other) const;
  tensor<bool> greater_equal(const value_type __val) const;

  tensor<bool> equal(const tensor& __other) const;
  tensor<bool> equal(const value_type __val) const;

  size_t count_nonzero(index_type __dim = -1LL) const;

  tensor
  slice(index_type __dim, std::optional<index_type> __start, std::optional<index_type> __end, index_type __step) const;

  tensor fmax(const tensor& __other) const;
  tensor fmod(const tensor& __other) const;

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
  tensor sin() const;
  tensor sinh() const;
  tensor asinh() const;
  tensor cosh() const;
  tensor acosh() const;

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
  tensor reshape_as(const tensor& __other) { return this->reshape(__other.__shape_); }

  tensor cross_product(const tensor& __other) const;

  tensor absolute(const tensor& __tensor) const;

  tensor dot(const tensor& __other) const;

  tensor relu() const;

  tensor transpose() const;

  tensor pow(const tensor& __other) const;
  tensor pow(const value_type __val) const;

  tensor cumprod(index_type __dim = -1) const;

  tensor cat(const std::vector<tensor>& _others, index_type _dim) const;

  tensor argmax(index_type __dim) const;

  tensor unsqueeze(index_type __dim) const;

  index_type lcm() const;

  void sqrt_();

  void exp_();

  void log2_();
  void log10_();
  void log_();

  void frac_();

  void fmod_(const tensor& __other);

  void cos_();
  void cosh_();
  void acosh_();
  void sin_() const;
  void sinh_();
  void asinh_();

  void ceil_();
  void floor_();

  void relu_();

  void clamp_(const_pointer __min_val = nullptr, const_pointer __max_val = nullptr);

  void logical_not_() const;

  void logical_or_(const tensor& __other);
  void logical_or_(const value_type __val);

  void logical_xor_(const tensor& __other);
  void logical_xor_(const value_type __val);

  void logical_and_(const tensor& __other);
  void logical_and_(const value_type __val);

  void pow_(const tensor& __other) const;
  void pow_(const value_type __val) const;

  void bitwise_left_shift_(const int __amount);
  void bitwise_right_shift_(const int __amount);

  void bitwise_and_(const value_type __val);
  void bitwise_and_(const tensor& __other);

  void bitwise_or_(const value_type __val);
  void bitwise_or_(const tensor& __other);

  void bitwise_xor_(const value_type __val);
  void bitwise_xor_(const tensor& __other);

  void bitwise_not_();

  void view(std::initializer_list<index_type> __new_sh);

  void print() const noexcept;

  static tensor zeros(const shape_type& __sh);

  static tensor ones(const shape_type& __sh);

  static tensor randomize(const shape_type& __sh, bool __bounded = false);

  tensor<index_type> argmax_(index_type __dim) const;
  tensor<index_type> argsort(index_type __dim = -1LL, bool __ascending = true) const;

 private:
  static void __check_is_scalar_type(const std::string __msg);

  static void __check_is_integral_type(const std::string __msg);

  template<typename __t>
  static void __check_is_same_type(const std::string __msg);
  static void __check_is_arithmetic_type(const std::string __msg);
  void        __compute_strides();
  size_t      __compute_index(const std::vector<index_type>& __idx) const;
  uint64_t    __computeSize(const shape_type& __dims) const;
  uint64_t    __compute_outer_size(const index_type __dim) const;
  float       __frac(const_reference __scalar);
  index_type  __lcm(const index_type __a, const index_type __b);
};  // tensor class
*/

/*tensor implementation*/

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
  typedef const pointer           const_pointer;
  typedef std::vector<value_type> data_container;
  typedef const data_container    const_data_container;

  typedef std::__wrap_iter<typename std::allocator_traits<std::allocator<_Tp>>::pointer>       __iterator;
  typedef std::__wrap_iter<typename std::allocator_traits<std::allocator<_Tp>>::const_pointer> __const_iterator;
  typedef std::reverse_iterator<__iterator>                                                    reverse_iterator;
  typedef std::reverse_iterator<__const_iterator>                                              const_reverse_iterator;


 private:
  data_container          __data_;
  shape_type              __shape_;
  std::vector<index_type> __strides_;

 public:
  tensor() = default;

  explicit tensor(const shape_type __sh, const value_type __v) :
      __shape_(__sh) {
    index_type __s = this->__computeSize(__sh);
    this->__data_  = data_container(__s, __v);
    this->__compute_strides();
  }

  explicit tensor(const shape_type __sh) :
      __shape_(__sh) {
    index_type __s = this->__computeSize(__sh);
    this->__data_  = data_container(__s);
    this->__compute_strides();
  }

  explicit tensor(const data_container __d, const shape_type __sh) :
      __data_(__d),
      __shape_(__sh) {
    this->__compute_strides();
  }

  explicit tensor(const tensor& __t) :
      __data_(__t.storage()),
      __shape_(__t.shape()),
      __strides_(__t.strides()) {}

  tensor(tensor&& __t) noexcept :
      __data_(std::move(__t.storage())),
      __shape_(std::move(__t.shape())),
      __strides_(std::move(__t.strides())) {}

  tensor(const shape_type __sh, std::initializer_list<value_type> init_list) :
      __shape_(__sh) {
    index_type __s = this->__computeSize(__sh);
    assert(init_list.size() == __s && "Initializer list size must match tensor size");
    this->__data_ = data_container(init_list);
    this->__compute_strides();
  }

  tensor(const shape_type& __sh, const tensor& __other) :
      __data_(__other.storage()),
      __shape_(__sh) {
    this->__compute_strides();
  }

 private:
  class __destroy_tensor
  {
   public:
    _LIBCPP_CONSTEXPR __destroy_tensor(tensor& __tens) :
        __tens_(__tens) {}
    void operator()() {
      if (__tens_.begin() != __tens_.end())
      {
        __tens_.__data_.~vector();
        __tens_.__shape_.~vector();
        __tens_.__strides_.~vector();
      }
    }

   private:
    tensor& __tens_;
  };

 public:
  ~tensor() { __destroy_tensor (*this)(); }

  tensor& operator=(const tensor& __other);
  tensor& operator=(tensor&& __other) noexcept;

  __const_iterator begin() const { return this->__data_.begin(); }
  __const_iterator end() const { return this->__data_.end(); }

  const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
  const_reverse_iterator rend() const { return const_reverse_iterator(begin()); }

  data_container storage() const { return this->__data_; }
  shape_type     shape() const { return this->__shape_; }
  shape_type     strides() const { return this->__strides_; }
  size_t         n_dims() const { return this->__shape_.size(); }

  size_t size(const index_type __dim) const {
    if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
    {
      throw std::invalid_argument("dimension input is out of range");
    }

    if (__dim == 0)
    {
      return this->__computeSize(this->__shape_);
    }

    return this->__shape_[__dim];
  }

  size_t          capacity() const noexcept { return this->__data_.capacity(); }
  reference       at(shape_type __idx);
  const_reference at(const shape_type __idx) const;

  reference operator[](const size_t __in) {
    if (__in >= this->__data_.size() || __in < 0)
    {
      throw std::out_of_range("Access index is out of range");
    }

    return this->__data_[__in];
  }

  const_reference operator[](const size_t __in) const {
    if (__in >= this->__data_.size() || __in < 0)
    {
      throw std::out_of_range("Access index is out of range");
    }

    return this->__data_[__in];
  }

  tensor operator+(const tensor& __other) const;
  tensor operator-(const tensor& __other) const;

  tensor operator-=(const tensor& __other) const;
  tensor operator+=(const tensor& __other) const;

  tensor operator+(const value_type _scalar) const;
  tensor operator-(const value_type _scalar) const;

  tensor operator+=(const_reference __scalar) const;
  tensor operator-=(const_reference __scalar) const;

  bool operator==(const tensor& __other) const {
    if ((this->__shape_ != __other.shape()) && (this->__strides_ != __other.strides()))
    {
      return false;
    }

    return this->__data_ == __other.storage();
  }

  bool operator!=(const tensor& __other) const { return !(*this == __other); }

  bool empty() const { return this->__data_.empty(); }

  double mean() const;

  tensor<bool> logical_not() const;
  tensor<bool> logical_or(const value_type __val) const;
  tensor<bool> logical_or(const tensor& __other) const;

  tensor<bool> less_equal(const tensor& __other) const;
  tensor<bool> less_equal(const value_type __val) const;

  tensor<bool> greater_equal(const tensor& __other) const;
  tensor<bool> greater_equal(const value_type __val) const;

  tensor<bool> equal(const tensor& __other) const;
  tensor<bool> equal(const value_type __val) const;

  size_t count_nonzero(index_type __dim = -1LL) const;

  tensor
  slice(index_type __dim, std::optional<index_type> __start, std::optional<index_type> __end, index_type __step) const;

  tensor fmax(const tensor& __other) const;
  tensor fmod(const tensor& __other) const;

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

  tensor clone() const {
    data_container __d = this->__data_;
    shape_type     __s = this->__shape_;
    return __self(__d, __s);
  }

  tensor clamp(const_pointer __min_val = nullptr, const_pointer __max_val = nullptr) const;

  tensor cos() const;
  tensor sin() const;
  tensor sinh() const;
  tensor asinh() const;
  tensor cosh() const;
  tensor acosh() const;

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
  tensor reshape_as(const tensor& __other) { return this->reshape(__other.__shape_); }

  tensor cross_product(const tensor& __other) const;

  tensor absolute(const tensor& __tensor) const;

  tensor dot(const tensor& __other) const;

  tensor relu() const;

  tensor transpose() const;

  tensor pow(const tensor& __other) const;
  tensor pow(const value_type __val) const;

  tensor cumprod(index_type __dim = -1) const;

  tensor cat(const std::vector<tensor>& _others, index_type _dim) const;

  tensor argmax(index_type __dim) const;

  tensor unsqueeze(index_type __dim) const;

  index_type lcm() const;
  void       sqrt_() {
    this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(std::exp(double(this->__data_[__i])));
    }
  }

  void exp_() {
    this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(std::exp(double(this->__data_[__i])));
    }
  }
  void log2_() {
    this->__check_is_integral_type("Given data type must be an integral");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(std::log2(double(this->__data_[__i])));
    }
  }

  void log10_() {
    this->__check_is_integral_type("Given data type must be an integral");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(std::log10(double(this->__data_[__i])));
    }
  }

  void log_() {
    this->__check_is_integral_type("Given data type must be an integral");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(std::log(double(this->__data_[__i])));
    }
  }

  void frac_() {
    this->__check_is_scalar_type("Cannot get the fraction of a non-scalar type");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));
    }
  }

  void fmod_(const tensor& __other);

  void cos_() {
    this->__check_is_scalar_type("Cannot perform a cosine on non-scalar data type");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(std::cos(static_cast<double>(this->__data_[__i])));
    }
  }

  void cosh_() {
    this->__check_is_scalar_type("Cannot perform a cosh on non-scalar data type");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(std::cosh(static_cast<double>(this->__data_[__i])));
    }
  }

  void acosh_() {
    this->__check_is_scalar_type("Cannot perform a acosh on non-scalar data type");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(std::acosh(static_cast<double>(this->__data_[__i])));
    }
  }

  void sinh_() {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(std::sinh(static_cast<double>(this->__data_[__i])));
    }
  }

  void asinh_() {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(std::asinh(static_cast<double>(this->__data_[__i])));
    }
  }

  void ceil_() {
    this->__check_is_scalar_type("Cannot get the ceiling of a non scalar value");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = std::ceil(this->__data_[__i]);
    }
  }

  void floor_() {
    this->__check_is_scalar_type("Cannot get the floor of a non scalar value");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = std::floor(this->__data_[__i]);
    }
  }

  void sin_() const {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(std::sin(static_cast<double>(this->__data_[__i])));
    }
  }

  void relu_();

  void clamp_(const_pointer __min_val = nullptr, const_pointer __max_val = nullptr);

  void logical_not_() const { this->bitwise_not_(); }

  void logical_or_(const tensor& __other);
  void logical_or_(const value_type __val);

  void logical_xor_(const tensor& __other);
  void logical_xor_(const value_type __val);

  void logical_and_(const tensor& __other);
  void logical_and_(const value_type __val);

  void pow_(const tensor& __other) const;

  void pow_(const value_type __val) const {
    this->__check_is_integral_type("cannot get the power of a non integral value");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] = static_cast<value_type>(std::pow(double(this->__data_[__i]), double(__val)));
    }
  }

  void bitwise_left_shift_(const int __amount) {
    this->__check_is_integral_type("Cannot perform a bitwise left shift on non-integral values");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] <<= __amount;
    }
  }

  void bitwise_right_shift_(const int __amount) {
    this->__check_is_integral_type("Cannot perform a bitwise right shift on non-integral values");
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      this->__data_[__i] >>= __amount;
    }
  }

  void bitwise_and_(const value_type __val);
  void bitwise_and_(const tensor& __other);

  void bitwise_or_(const value_type __val);
  void bitwise_or_(const tensor& __other);

  void bitwise_xor_(const value_type __val);
  void bitwise_xor_(const tensor& __other);

  void bitwise_not_();

  void view(std::initializer_list<index_type> __new_sh);
  void print() const noexcept;

  static tensor zeros(const shape_type& __sh) {
    __check_is_scalar_type("template type must be a scalar : tensor.zeros()");
    data_container __d(std::accumulate(__sh.begin(), __sh.end(), 1, std::multiplies<index_type>()), value_type(0));
    return __self(__d, __sh);
  }

  static tensor ones(const shape_type& __sh) {
    __check_is_scalar_type("template type must be a scalar : tensor.ones()");
    data_container __d(std::accumulate(__sh.begin(), __sh.end(), 1, std::multiplies<index_type>()), value_type(1));
    return __self(__d, __sh);
  }

  static tensor randomize(const shape_type& __sh, bool __bounded = false);

  tensor<index_type> argmax_(index_type __dim) const;

  tensor<index_type> argsort(index_type __dim = -1LL, bool __ascending = true) const;

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
    {
      throw std::out_of_range("input indices does not match the tensor __shape_");
    }

    size_t __index = 0, __i = 0;
    for (; __i < this->__shape_.size(); __i++)
    {
      __index += __idx[__i] * this->__strides_[__i];
    }

    return __index;
  }

  uint64_t __computeSize(const shape_type& __dims) const {
    uint64_t __ret = 0ULL;
    for (const index_type& __d : __dims)
    {
      __ret *= __d;
    }

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
tensor<_Tp> tensor<_Tp>::sinh() const {
  this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::sinh(static_cast<double>(this->__data_[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sin() const {
  this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::sin(static_cast<double>(this->__data_[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::frac() const {
  this->__check_is_scalar_type("Cannot get the fraction of a non-scalar type");
  std::vector<value_type> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::frac() const {
  this->__check_is_scalar_type("Cannot get the fraction of a non-scalar type");
  std::vector<value_type> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cos() const {
  this->__check_is_scalar_type("Cannot perform a cosine on non-scalar data type");
  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::cos(static_cast<double>(this->__data_[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log() const {
  this->__check_is_integral_type("Given data type must be an integral");
  std::vector<value_type> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::log(double(this->__data_[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::asinh() const {
  this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::asinh(static_cast<double>(this->__data_[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cosh() const {
  this->__check_is_scalar_type("Cannot perform a cosh on non-scalar data type");
  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::cosh(static_cast<double>(this->__data_[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sqrt() const {
  this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
  std::vector<value_type> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::sqrt(double(this->__data_[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::acosh() const {
  this->__check_is_scalar_type("Cannot perform a acosh on non-scalar data type");
  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::acosh(static_cast<double>(this->__data_[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log10() const {
  this->__check_is_integral_type("Given data type must be an integral");
  std::vector<value_type> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::log10(double(this->__data_[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::log2() const {
  this->__check_is_integral_type("Given data type must be an integral");
  std::vector<value_type> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::log2(double(this->__data_[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::exp() const {
  this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
  std::vector<value_type> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::exp(double(this->__data_[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp>::index_type tensor<_Tp>::lcm() const {
  this->__check_is_scalar_type("Given template type must be an int");
  index_type __ret = static_cast<index_type>(this->__data_[0]);

  size_t __i = 1;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret = this->__lcm(static_cast<index_type>(this->__data_[__i]), __ret);
  }

  return __ret;
}

template<class _Tp>
void tensor<_Tp>::logical_or_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise not of non-integral and non-boolean value");
  }

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = (this->__data_[__i] || __val);
  }
}

template<class _Tp>
void tensor<_Tp>::logical_xor_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
  }

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = (this->__data_[__i] ^ __val);
  }
}

template<class _Tp>
void tensor<_Tp>::logical_and_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise and of non-integral and non-boolean value");
  }

  for (size_t __i = 0; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = (this->__data_[__i] && __val);
  }
}

template<class _Tp>
void tensor<_Tp>::bitwise_or_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");
  }

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] |= __val;
  }
}

template<class _Tp>
void tensor<_Tp>::bitwise_xor_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");
  }

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] ^= __val;
  }
}

template<class _Tp>
void tensor<_Tp>::bitwise_not_() {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise not on non integral or boolean value");
  }

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = ~this->__data_[__i];
  }
}

template<class _Tp>
void tensor<_Tp>::bitwise_and_(const value_type __val) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");
  }

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] &= __val;
  }
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
  for (size_t i = 0; i < this->__shape_.size(); i++)
  {
    std::cout << this->__shape_[i] << (i < this->__shape_.size() - 1 ? ", " : "");
  }

  std::cout << "]\nData: ";
  for (const value_type& val : this->__data_)
  {
    std::cout << val << " ";
  }

  std::cout << std::endl;
}

template<class _Tp>
typename tensor<_Tp>::reference tensor<_Tp>::at(tensor<_Tp>::shape_type __idx) {
  if (__idx.empty())
  {
    throw std::invalid_argument("Passing an empty vector as indices for a tensor");
  }

  index_type __i = this->__compute_index(__idx);
  if (__i < 0 || __i >= this->__data_.size())
  {
    throw std::invalid_argument("input indices are out of bounds");
  }

  return this->__data_[__i];
}

template<class _Tp>
typename tensor<_Tp>::const_reference tensor<_Tp>::at(const tensor<_Tp>::shape_type __idx) const {
  if (__idx.empty())
  {
    throw std::invalid_argument("Passing an empty vector as indices for a tensor");
  }

  index_type __i = this->__compute_index(__idx);
  if (__i < 0 || __i >= this->__data_.size())
  {
    throw std::invalid_argument("input indices are out of bounds");
  }

  return this->__data_[__i];
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  if (__other.shape() != this->__shape_)
  {
    throw std::invalid_argument("Cannot add two tensors with different shapes");
  }

  data_container __d(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __d[__i] = this->__data_[__i] + __other[__i];
  }

  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  if (__other.shape() != this->__shape_)
  {
    throw std::invalid_argument("Cannot add two tensors with different shapes");
  }

  data_container __d(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __d[__i] = this->__data_[__i] - __other[__i];
  }

  return __self(__d, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const value_type __scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] += __scalar;
  }

  return __self(*this);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const value_type __scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] -= __scalar;
  }

  return __self(*this);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+=(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = this->__data_[__i] + __other[__i];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+=(const_reference __scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = this->__data_[__i] + __scalar;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-=(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = this->__data_[__i] - __other[__i];
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-=(const_reference __scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = this->__data_[__i] - __scalar;
  }

  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::randomize(const shape_type& __sh, bool __bounded) {
  __check_is_scalar_type("template class must be a scalar type");

  std::srand(time(nullptr));
  index_type __s = 1;
  for (index_type __d : __sh)
  {
    __s *= __d;
  }

  data_container __d(__s);

  index_type __i = 0;
  for (; __i < __s; __i++)
  {
    __d[__i] = value_type(__bounded ? static_cast<float>(rand() % RAND_MAX) : static_cast<float>(rand()));
  }

  return __self(__d, __sh);
}

template<class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argmax_(index_type __dim) const {
  if (__dim < 0 || __dim >= this->__shape_.size())
  {
    throw std::out_of_range("Dimension out of range in argmax");
  }

  shape_type __ret_sh = this->__shape_;
  __ret_sh.erase(__ret_sh.begin() + __dim);
  tensor<index_type> __ret;
  __ret.__shape_ = __ret_sh;
  __ret.__data_.resize(this->__computeSize(__ret_sh), 0);

  index_type __outer_size = 1LL;
  index_type __inner_size = 1LL;

  index_type __i = 0;
  for (; __i < __dim; __i++)
  {
    __outer_size *= this->__shape_[__i];
  }

  for (__i = __dim + 1; __i < this->__shape_.size(); __i++)
  {
    __inner_size *= this->__shape_[__i];
  }

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

  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::argmax(index_type __dim) const {
  if (__dim < 0 || __dim >= this->__shape_.size())
  {
    throw std::out_of_range("Dimension out of range in argmax");
  }

  shape_type __ret_sh = this->__shape_;
  __ret_sh.erase(__ret_sh.begin() + __dim);
  tensor __ret;
  __ret.__shape_ = __ret_sh;
  __ret.__data_.resize(this->__computeSize(__ret_sh), value_type(0));

  index_type __outer_size = 1;
  index_type __inner_size = 1;

  index_type __i = 0;
  for (; __i < __dim; __i++)
  {
    __outer_size *= this->__shape_[__i];
  }

  for (__i = __dim + 1; __i < static_cast<index_type>(this->__shape_.size()); __i++)
  {
    __inner_size *= this->__shape_[__i];
  }

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
        {
          __max_value = __v;
        }
      }
      __ret.__data_[__i * __inner_size + __j] = __max_value;
    }
  }

  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::unsqueeze(index_type __dim) const {
  if (__dim < 0 || __dim > static_cast<index_type>(this->__shape_.size()))
  {
    throw std::out_of_range("Dimension out of range in unsqueeze");
  }

  shape_type __s = this->__shape_;
  __s.insert(__s.begin() + __dim, 1);
  tensor __ret;
  __ret.__shape_ = __s;
  __ret.__data_  = this->__data_;

  return __ret;
}

template<class _Tp>
void tensor<_Tp>::view(std::initializer_list<index_type> __sh) {
  index_type __s = 1LL;
  for (index_type __d : __sh)
  {
    __s *= __d;
  }

  if (__s != this->__data_.size())
  {
    throw std::invalid_argument("_Tpotal elements do not match for new shape");
  }

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
      {
        throw std::invalid_argument(
          "Cannot concatenate tensors with different shapes along non-concatenation dimensions");
      }
    }
  }

  shape_type __ret_sh = this->__shape_;

  for (const tensor& __t : __others)
  {
    __ret_sh[__dim] += __t.__shape_[__dim];
  }

  data_container __c;
  __c.reserve(this->__data_.size());
  __c.insert(__c.end(), this->__data_.begin(), this->__data_.end());

  for (const tensor& __t : __others)
  {
    __c.insert(__c.end(), __t.__data_.begin(), __t.__data_.end());
  }

  return __self(__c, __ret_sh);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cumprod(index_type __dim) const {
  if (__dim == -1)
  {
    data_container __flat = this->__data_;
    data_container __ret(__flat.size());
    __ret[0] = __flat[0];

    size_t __i = 1;
    for (; __i < __flat.size(); __i++)
    {
      __ret[__i] = __ret[__i - 1] * __flat[__i];
    }

    return __self(__ret, {__flat.size()});
  } else
  {
    if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
    {
      throw std::invalid_argument("Invalid dimension provided.");
    }

    data_container __ret(this->__data_);
    // TODO : compute_outer_size() implementation
    size_t __outer_size = this->__compute_outer_size(__dim);
    size_t __inner_size = this->__shape_[__dim];

    size_t __st = this->__strides_[__dim];

    size_t __i = 0;
    for (; __i < __outer_size; ++__i)
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

    return __self(__ret, this->__shape_);
  }
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::slice(index_type                __dim,
                               std::optional<index_type> __start,
                               std::optional<index_type> __end,
                               index_type                __step) const {
  if (__dim < 0ULL || __dim >= static_cast<index_type>(this->__shape_.size()))
  {
    throw std::out_of_range("Dimension out of range.");
  }

  tensor __ret;

  index_type __s       = this->__shape_[__dim];
  index_type __start_i = __start.value_or(0ULL);
  index_type __end_i   = __end.value_or(__s);

  if (__start_i < 0ULL)
  {
    __start_i += __s;
  }

  if (__end_i < 0ULL)
  {
    __end_i += __s;
  }

  __start_i = std::max(index_type(0ULL), std::min(__start_i, __s));
  __end_i   = std::max(index_type(0ULL), std::min(__end_i, __s));

  index_type __slice_size = (__end_i - __start_i + __step - 1) / __step;
  shape_type __ret_dims   = this->__shape_;
  __ret_dims[-__dim]      = __slice_size;
  __ret                   = tensor(__ret_dims);

  index_type __i = __start_i, __j = 0ULL;
  for (; __i < __end_i; __i += __step, __j++)
  {
    __ret({__j}) = this->at({__i});
  }

  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmod(const tensor<_Tp>& __other) const {
  this->__check_is_scalar_type("Cannot divide non scalar values");
  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
  {
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");
  }

  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::fmod(double(this->__data_[__i]), double(__other[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
void tensor<_Tp>::fmod_(const tensor<_Tp>& __other) {
  this->__check_is_scalar_type("Cannot divide non scalar values");
  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
  {
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");
  }

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::fmax(double(this->__data_[__i]), double(__other[__i])));
  }
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::fmax(const tensor& __other) const {
  this->__check_is_scalar_type("Cannot deduce the maximum of non scalar values");
  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
  {
    throw std::invalid_argument("Cannot compare two tensors of different shapes : fmax");
  }

  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::fmax(double(this->__data_[__i]), double(__other[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
size_t tensor<_Tp>::count_nonzero(index_type __dim) const {
  this->__check_is_scalar_type("Cannot compare a non scalar value to zero");

  size_t __c = 0;

  if (__dim == -1)
  {
    for (const value_type& __el : this->__data_)
    {
      if (__el != 0)
      {
        __c++;
      }
    }
  } else
  {
    if (__dim < 0 || __dim >= static_cast<index_type>(__shape_.size()))
    {
      throw std::invalid_argument("Invalid dimension provided.");
    }
    throw std::runtime_error("Dimension-specific non-zero counting is not implemented yet.");
  }

  return __c;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::matmul(const tensor& __other) const {
  if (this->__shape_.size() != 2 || __other.shape().size() != 2)
  {
    throw std::invalid_argument("matmul is only supported for 2D tensors");
  }

  if (this->__shape_[1] != __other.shape()[0])
  {
    throw std::invalid_argument("Shape mismatch for matrix multiplication");
  }

  shape_type     __ret_sh = {this->__shape_[0], __other.shape()[1]};
  data_container __ret_d(__ret_sh[0] * __ret_sh[1], 0);

  const int blockSize = 64;  // Example block size, can be tuned for performance

  for (int i = 0; i < __ret_sh[0]; i += blockSize)
  {
    for (int j = 0; j < __ret_sh[1]; j += blockSize)
    {
      for (int k = 0; k < this->__shape_[1]; k += blockSize)
      {
        for (int ii = i; ii < std::min(i + blockSize, __ret_sh[0]); ++ii)
        {
          for (int jj = j; jj < std::min(j + blockSize, __ret_sh[1]); ++jj)
          {
            value_type __sum = 0;
            for (int kk = k; kk < std::min(k + blockSize, this->__shape_[1]); ++kk)
            {
              __sum += this->at({ii, kk}) * __other.at({kk, jj});
            }
            __ret_d[ii * __ret_sh[1] + jj] += __sum;
          }
        }
      }
    }
  }
  return __self(__ret_d, __ret_sh);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::reshape(const shape_type __sh) const {
  data_container __d = this->__data_;
  size_t         __s = this->__computeSize(__sh);

  if (__s != this->__data_.size())
  {
    throw std::invalid_argument(
      "input shape must have size of elements equal to the current number of elements in the tensor data");
  }

  return __self(__d, __sh);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::cross_product(const tensor& __other) const {
  this->__check_is_arithmetic_type("Cannot perform a cross product on non-scalar data types");
  if (this->empty() || __other.empty())
  {
    throw std::invalid_argument("Cannot cross product an empty vector");
  }

  if (this->shape() != std::vector<int>{3} || __other.shape() != std::vector<int>{3})
  {
    throw std::invalid_argument("Cross product can only be performed on 3-element vectors");
  }

  tensor __ret({3});

  tensor<_Tp>::const_reference __a1 = this->__data_[0];
  tensor<_Tp>::const_reference __a2 = this->__data_[1];
  tensor<_Tp>::const_reference __a3 = this->__data_[2];

  tensor<_Tp>::const_reference __b1 = __other[0];
  tensor<_Tp>::const_reference __b2 = __other[1];
  tensor<_Tp>::const_reference __b3 = __other[2];

  __ret[0] = __a2 * __b3 - __a3 * __b2;
  __ret[1] = __a3 * __b1 - __a1 * __b3;
  __ret[2] = __a1 * __b2 - __a2 * __b1;

  return __ret;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::absolute(const tensor& __tensor) const {
  this->__check_is_scalar_type("Cannot call absolute on non scalar value");

  data_container __a;

  for (tensor<_Tp>::const_reference __v : __tensor.storage())
  {
    __a.push_back(static_cast<value_type>(std::fabs(float(__v))));
  }

  return __self(__a, __tensor.__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::dot(const tensor& __other) const {
  this->__check_is_scalar_type("Cannot perform a dot product on non scalar data types");
  if (this->empty() || __other.empty())
  {
    throw std::invalid_argument("Cannot dot product an empty vector");
  }

  if (this->__shape_.size() == 1 && __other.shape().size())
  {
    assert(this->__shape_[0] == __other.shape()[0]);
    value_type __ret = 0;

    index_type __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
      __ret += this->__data_[__i] * __other[__i];
    }

    return __self({__ret}, {1});
  }

  if (this->__shape_.size() == 2 && __other.shape().size())
  {
    return this->matmul(__other);
  }

  if (this->__shape_.size() == 3 && __other.shape().size())
  {
    return this->cross_product(__other);
  }

  return __self();
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::relu() const {
  this->__check_is_scalar_type("Cannot relu non-scalar type");
  size_t         __s = this->__data_.size();
  data_container __d(__s);

  size_t __i = 0;
  for (; __i < __s; __i++)
  {
    __d[__i] = std::max(this->__data_[__i], value_type(0));
  }

  return __self(__d, this->__shape_);
}

template<class _Tp>
void tensor<_Tp>::relu_() {
  this->__check_is_scalar_type("Cannot relu non-scalar type");

  size_t __s = this->__data_.size(), __i = 0;
  for (; __i < __s; __i++)
  {
    this->__data_[__i] = std::max(this->__data_[__i], value_type(0));
  }
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::transpose() const {
  if (this->__shape_.size() != 2)
  {
    std::cerr << "Matrix transposition can only be done on 2D tensors" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  tensor __ret({this->__shape_[1], this->__shape_[0]});

  index_type __i = 0;
  for (; __i < this->__shape_[0]; __i++)
  {
    index_type __j = 0;
    for (; __j < this->__shape_[1]; __j++)
    {
      __ret.at({__j, __i}) = this->at({__i, __j});
    }
  }
  return __ret;
}

/*I did this wrong and forgot to deal with one dimensional __data_*/
/*
template<class _Tp>
tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argsort(index_type __d, bool __ascending) const {
  index_type __adjusted = (__d < 0LL) ? __d + this->__data_.size() : __d;
  if (__adjusted < 0LL || __adjusted >= static_cast<index_type>(this->__data_.size()))
    throw std::out_of_range("Invalid dimension for argsort");

  std::vector<std::vector<index_type>> __indices = this->__data_;

  size_t __i = 0;
  for (; __i < __data_.size(); __i++) {
    std::vector<index_type> __idx(this->__data_[__i].size());
    std::iota(__idx.begin(), __idx.end(), 0);

    std::sort(__idx.begin(), __idx.end(), [&](index_type __a, index_type __b) {
      return __ascending ? this->__data_[__i][__a] < this->__data_[__i][__b]
                         : this->__data_[__i][__a] > this->__data_[__i][__b];
    });

    __indices[__i] = __idx;
  }

  return tensor<index_type>(__indices);
}
*/

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_left_shift(const int __amount) const {
  this->__check_is_integral_type("Cannot perform a bitwise left shift on non-integral values");

  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = this->__data_[__i] << __amount;
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_xor(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");

  assert(this->shape() == __other.shape() && this->size(0) == __other.size(0));

  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = this->__data_[__i] ^ __other[__i];
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::bitwise_right_shift(const int _amount) const {
  this->__check_is_integral_type("Cannot perform a bitwise right shift on non-integral values");

  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = this->__data_[__i] >> _amount;
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
void tensor<_Tp>::bitwise_and_(const tensor& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");
  }

  assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] &= __other[__i];
  }
}

template<class _Tp>
void tensor<_Tp>::bitwise_or_(const tensor& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");
  }

  assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] |= __other[__i];
  }
}

template<class _Tp>
void tensor<_Tp>::bitwise_xor_(const tensor& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");
  }

  assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] ^= __other[__i];
  }
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_not() const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise not of non-integral and non-boolean value");
  }

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = ~(this->__data_[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise or of non-integral and non-boolean value");
  }

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] || __val);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const tensor<_Tp>& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise or of non-integral and non-boolean value");
  }

  assert(this->__shape_ == __other.shape());

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] || __other[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
  }

  assert(this->__shape_ == __other.shape());

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] ^ __other[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
  }

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] ^ __val);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise and of non-integral and non-boolean value");
  }

  assert(this->__shape_ == __other.shape());

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] && __other[__i]);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise and of non-integral and non-boolean value");
  }

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] && __val);
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
void tensor<_Tp>::logical_or_(const tensor<_Tp>& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise not of non-integral and non-boolean value");
  }

  assert(this->__shape_ == __other.shape());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = (this->__data_[__i] || __other[__i]);
  }
}

template<class _Tp>
void tensor<_Tp>::logical_xor_(const tensor<_Tp>& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
  }

  assert(this->__shape_ == __other.shape());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = (this->__data_[__i] ^ __other[__i]);
  }
}

template<class _Tp>
void tensor<_Tp>::logical_and_(const tensor<_Tp>& __other) {
  if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
  {
    throw std::runtime_error("Cannot get the element wise and of non-integral and non-boolean value");
  }

  assert(this->__shape_ == __other.shape());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = (this->__data_[__i] && __other[__i]);
  }
}

template<class _Tp>
double tensor<_Tp>::mean() const {
  this->__check_is_integral_type("input must be of integral type to get the mean average");
  double __m = 0.0f;

  if (this->empty())
  {
    return 0.0f;
  }

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __m += this->__data_[__i];
  }

  return static_cast<double>(__m / double(this->__data_.size()));
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::pow(const tensor& __other) const {
  this->__check_is_integral_type("cannot get the power of a non integral value");

  assert(this->__shape_ == __other.shape() && this->size(0) == __other.shape());

  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::pow(double(this->__data_[__i]), double(__other[__i])));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::pow(const value_type __val) const {
  this->__check_is_integral_type("cannot get the power of a non integral value");

  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = static_cast<value_type>(std::pow(double(this->__data_[__i]), double(__val)));
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
void tensor<_Tp>::pow_(const tensor& __other) const {
  this->__check_is_integral_type("cannot get the power of a non integral value");

  assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    this->__data_[__i] = static_cast<value_type>(std::pow(double(this->__data_[__i]), double(__other[__i])));
  }
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] <= __other[__i]) ? true : false;
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] <= __val) ? true : false;
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] >= __other[__i]) ? true : false;
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] >= __val) ? true : false;
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const tensor& __other) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] == __other[__i]) ? true : false;
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const value_type __val) const {
  if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
  {
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  }

  std::vector<bool> __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = (this->__data_[__i] >= __val) ? true : false;
  }

  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::sum(const index_type __axis) const {
  this->__check_is_scalar_type("Cannot reduce tensor with non scalar type");
  if (__axis < 0LL || __axis >= static_cast<index_type>(this->__shape_.size()))
  {
    throw std::invalid_argument("Invalid axis for sum");
  }

  shape_type __ret_sh = this->__shape_;
  __ret_sh[__axis]    = 1LL;

  index_type     __ret_size = std::accumulate(__ret_sh.begin(), __ret_sh.end(), 1, std::multiplies<index_type>());
  data_container __ret_data(__ret_size, value_type(0.0f));

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

  return __self(__ret_data, __ret_sh);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::row(const index_type __index) const {
  if (this->__shape_.size() != 2)
  {
    throw std::runtime_error("Cannot get a row from a non two dimensional tensor");
  }

  if (this->__shape_[0] <= __index || __index < 0LL)
  {
    throw std::invalid_argument("Index input is out of range");
  }

  data_container __r(this->__data_.begin() + (this->__shape_[1] * __index),
                     this->__data_.begin() + (this->__shape_[1] * __index + this->__shape_[1]));

  return __self(__r, {this->__shape_[1]});
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::col(const index_type __index) const {
  if (this->__shape_.size() != 2)
  {
    throw std::runtime_error("Cannot get a column from a non two dimensional tensor");
  }

  if (this->__shape_[1] <= __index || __index < 0LL)
  {
    throw std::invalid_argument("Index input out of range");
  }

  data_container __c;

  index_type __i = 0;
  for (; __i < this->__shape_[0]; __i++)
  {
    __c.push_back(this->__data_[this->__compute_index({__i, __index})]);
  }

  return __self(__c, {this->__shape_[0]});
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::ceil() const {
  this->__check_is_scalar_type("Cannot get the ceiling of a non scalar value");
  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = std::ceil(this->__data_[__i]);
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::floor() const {
  this->__check_is_scalar_type("Cannot get the floor of a non scalar value");
  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    __ret[__i] = std::floor(this->__data_[__i]);
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::clamp(const_pointer __min_val, const_pointer __max_val) const {
  data_container __ret(this->__data_.size());

  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    value_type __v = this->__data_[__i];
    if (__min_val)
    {
      __v = std::max(*__min_val, __v);
    }

    if (__max_val)
    {
      __v = std::min(*__max_val, __v);
    }

    __ret[__i] = __v;
  }

  return __self(__ret, this->__shape_);
}

template<class _Tp>
void tensor<_Tp>::clamp_(const_pointer __min_val, const_pointer __max_val) {
  size_t __i = 0;
  for (; __i < this->__data_.size(); __i++)
  {
    if (__min_val)
    {
      this->__data_[__i] = std::max(*__min_val, this->__data_[__i]);
    }

    if (__max_val)
    {
      this->__data_[__i] = std::min(*__max_val, this->__data_[__i]);
    }
  }
}
