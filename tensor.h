/*tensor synopsis*/

template<class _Tp>
class tensor {
 public:
  typedef int64_t                                index_type;
  typedef std::vector<_Tp>                       data_container;
  typedef const data_container                   const_data_container;
  typedef std::vector<index_type>                shape_type;
  typedef _Tp                                    value_type;
  typedef value_type&                            reference;
  typedef const value_type&                      const_reference;
  typedef allocator<value_type>                  allocator_type;
  typedef allocator_traits<allocator_type>       __alloc_traits;
  typedef typename __alloc_traits::pointer       pointer;
  typedef typename __alloc_traits::const_pointer const_pointer;
  typedef __wrap_iter<pointer>                   iterator;
  typedef __wrap_iter<const_pointer>             const_iterator;
  typedef std::reverse_iterator<iterator>        reverse_iterator;
  typedef const reverse_iterator                 const_reverse_iterator;

 private:
  data_container          __data_;
  shape_type              __shape_;
  std::vector<index_type> __strides_;

 public:
  tensor() = default;

  explicit tensor(const shape_type __sh, const value_type __v);

  explicit tensor(const shape_type __sh);

  explicit tensor(const data_container __d, const shape_type __sh);

 private:
  class __destroy_tensor {
   public:
    __destroy_tensor(tensor& __tens);
    void operator()();

   private:
    tensor& __tens_;
  };

 public:
  ~tensor() { __destroy_tensor (*this)(); }
  const_iterator begin() const;
  const_iterator end() const;

  const_reverse_iterator rbegin() const;
  const_reverse_iterator rend() const;

  data_container storage() const _NOEXCEPT { return this->__data_; }
  shape_type     shape() const _NOEXCEPT { return this->__shape_; }
  shape_type     strides() const _NOEXCEPT { return this->__strides_; }

  size_t n_dims() const _NOEXCEPT { return this->__shape_.size(); }
  size_t size(const index_type __d) const;
  size_t capacity() const _NOEXCEPT;

  reference       at(shape_type __idx);
  const_reference at(const shape_type __idx) const;

  reference       operator[](const size_t __in);
  const_reference operator[](const size_t __in) const;

  tensor<value_type> operator+(const tensor<value_type>& __other) const;
  tensor<value_type> operator-(const tensor<value_type>& __other) const;

  tensor<value_type> operator-=(const tensor<value_type>& __other) const;
  tensor<value_type> operator+=(const tensor<value_type>& __other) const;

  tensor<value_type> operator+(const value_type _scalar) const;
  tensor<value_type> operator-(const value_type _scalar) const;

  tensor<value_type> operator+=(const_reference _scalar) const;
  tensor<value_type> operator-=(const_reference __scalar) const;

  bool operator==(const tensor<value_type>& __other) const;
  bool operator!=(const tensor<value_type>& other) const;

  bool empty() const;

  double mean() const;

  tensor<bool> logical_not() const;
  tensor<bool> logical_or(const value_type value) const;
  tensor<bool> logical_or(const tensor<value_type>& __other) const;

  tensor<bool> less_equal(const tensor<value_type>& __other) const;
  tensor<bool> less_equal(const value_type value) const;

  tensor<bool> greater_equal(const tensor<value_type>& __other) const;
  tensor<bool> greater_equal(const value_type value) const;

  tensor<bool> equal(const tensor<value_type>& __other) const;
  tensor<bool> equal(const value_type value) const;

  size_t count_nonzero(index_type dim = -1LL) const;
  tensor<value_type>
  slice(index_type _dim, std::optional<index_type> _start, std::optional<index_type> _end, index_type _step) const;

  tensor<value_type> fmax(const tensor<value_type>& __other) const;
  tensor<value_type> fmod(const tensor<value_type>& __other) const;
  tensor<value_type> frac() const;
  tensor<value_type> log() const;
  tensor<value_type> log10() const;
  tensor<value_type> log2() const;
  tensor<value_type> exp() const;
  tensor<value_type> sqrt() const;

  tensor<value_type> sum(const index_type _axis) const;

  tensor<value_type> row(const index_type _index) const;
  tensor<value_type> col(const index_type _index) const;

  tensor<value_type> ceil() const;
  tensor<value_type> floor() const;

  tensor<value_type> clone() const;

  tensor<value_type> clamp(const value_type* _min_val = nullptr, const value_type* _max_val = nullptr) const;

  tensor<value_type> cos() const;
  tensor<value_type> sin() const;
  tensor<value_type> sinh() const;
  tensor<value_type> asinh() const;
  tensor<value_type> cosh() const;
  tensor<value_type> acosh() const;

  tensor<value_type> logical_xor(const tensor<value_type>& __other) const;
  tensor<value_type> logical_xor(const value_type value) const;
  tensor<value_type> logical_and(const tensor<value_type>& __other) const;
  tensor<value_type> logical_and(const value_type value) const;

  tensor<value_type> bitwise_not() const;
  tensor<value_type> bitwise_and(const value_type value) const;
  tensor<value_type> bitwise_and(const tensor<value_type>& __other) const;
  tensor<value_type> bitwise_or(const value_type value) const;
  tensor<value_type> bitwise_or(const tensor<value_type>& __other) const;
  tensor<value_type> bitwise_xor(const value_type value) const;
  tensor<value_type> bitwise_xor(const tensor<value_type>& __other) const;

  tensor<value_type> bitwise_left_shift(const int _amount) const;
  tensor<value_type> bitwise_right_shift(const int _amount) const;

  tensor<value_type> matmul(const tensor<value_type>& __other) const;

  tensor<value_type> reshape(const shape_type _shape) const;
  tensor<value_type> reshape_as(const tensor<value_type>& __other) { return this->reshape(__other.__shape_); }

  tensor<value_type> cross_product(const tensor<value_type>& __other) const;

  tensor<value_type> absolute(const tensor<value_type>& _tensor) const;

  tensor<value_type> dot(const tensor<value_type>& __other) const;

  tensor<value_type> relu() const;

  tensor<value_type> transpose() const;

  tensor<value_type> pow(const tensor<value_type>& __other) const;
  tensor<value_type> pow(const value_type value) const;

  tensor<value_type> cumprod(index_type _dim = -1) const;

  tensor<value_type> cat(const std::vector<tensor<value_type>>& _others, index_type _dim) const;

  tensor<value_type> argmax(index_type _dim) const;

  tensor<value_type> unsqueeze(index_type _dim) const;

  index_type lcm() const;

  void sqrt_();
  void exp_();
  void log2_();
  void log10_();
  void log_();
  void frac_();
  void fmod_(const tensor<value_type>& __other);
  void cos_();
  void cosh_();
  void acosh_();
  void sinh_();
  void asinh_();
  void ceil_();
  void floor_();
  void sin_() const;

  void relu_();

  void clamp_(const value_type* _min_val = nullptr, const value_type* _max_val = nullptr);

  void logical_not_() const { this->bitwise_not_(); }
  void logical_or_(const tensor<value_type>& __other);
  void logical_or_(const value_type value);
  void logical_xor_(const tensor<value_type>& __other);
  void logical_xor_(const value_type value);
  void logical_and_(const tensor<value_type>& __other);
  void logical_and_(const value_type value);

  void pow_(const tensor<value_type>& __other) const;
  void pow_(const value_type value) const;

  void bitwise_left_shift_(const int _amount);
  void bitwise_right_shift_(const int _amount);

  void bitwise_and_(const value_type value);
  void bitwise_and_(const tensor<value_type>& __other);
  void bitwise_or_(const value_type value);
  void bitwise_or_(const tensor<value_type>& __other);
  void bitwise_xor_(const value_type value);
  void bitwise_xor_(const tensor<value_type>& __other);
  void bitwise_not_();

  void view(std::initializer_list<index_type> _new_sh);

  void print() const _NOEXCEPT;

  static tensor<value_type> zeros(const shape_type& _sh);
  static tensor<value_type> ones(const shape_type& _sh);

  static tensor<value_type> randomize(const shape_type& _sh, bool _bounded = false) _NOEXCEPT;

  tensor<index_type> argmax_(index_type _dim) const;

  tensor<index_type> argsort(index_type _dim = -1LL, bool _ascending = true) const;

 private:
  void __check_is_scalar_type(const std::string __msg);
  void __check_is_integral_type(const std::string __msg);
  void __check_is_same_type(class __t, const std::string __msg);
  void __check_is_arithmetic_type(const std::string __msg);

  void     __compute_strides();
  size_t   __compute_index(const std::vector<index_type>& _idx) const;
  uint64_t __computeSize(const std::vector<index_type>& _dims) const;

  float      __frac(const reference scalar);
  index_type __lcm(const index_type a, const index_type b);
};


#include <iostream>
#include <stdexcept>
#include <random>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <type_traits>


_LIBCPP_BEGIN_NAMESPACE_STD
template<class _Tp>
class _LIBCPP_TEMPLATE_VIS tensor {
 public:
  typedef int64_t                                index_type;
  typedef std::vector<_Tp>                       data_container;
  typedef const data_container                   const_data_container;
  typedef std::vector<index_type>                shape_type;
  typedef _Tp                                    value_type;
  typedef value_type&                            reference;
  typedef const value_type&                      const_reference;
  typedef allocator<value_type>                  allocator_type;
  typedef allocator_traits<allocator_type>       __alloc_traits;
  typedef typename __alloc_traits::pointer       pointer;
  typedef typename __alloc_traits::const_pointer const_pointer;
  typedef __wrap_iter<pointer>                   iterator;
  typedef __wrap_iter<const_pointer>             const_iterator;
  typedef std::reverse_iterator<iterator>        reverse_iterator;
  typedef const reverse_iterator                 const_reverse_iterator;

 private:
  data_container          __data_;
  shape_type              __shape_;
  std::vector<index_type> __strides_;

 public:
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor() = default;
  _LIBCPP_CONSTEXPR_SINCE_CXX20
  _LIBCPP_HIDE_FROM_ABI explicit tensor(const shape_type __sh, const value_type __v) :
      __shape_(__sh) {
    index_type __s = this->__computeSize(__sh);
    this->__data_  = data_container(__s, __v);
    this->__compute_strides();
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20
  _LIBCPP_HIDE_FROM_ABI explicit tensor(const shape_type __sh) :
      __shape_(__sh) {
    index_type __s = this->__computeSize(__sh);
    this->__data_  = data_container(__s);
    this->__compute_strides();
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20
  _LIBCPP_HIDE_FROM_ABI explicit tensor(const data_container __d, const shape_type __sh) :
      __data_(__d),
      __shape_(__sh) {
    this->__compute_strides();
  }

 private:
  class __destroy_tensor {
   public:
    _LIBCPP_CONSTEXPR _LIBCPP_HIDE_FROM_ABI __destroy_tensor(tensor& __tens) :
        __tens_(__tens) {}
    _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void operator()() {
      if (__tens_.begin() != nullptr) {
        __tens_.__data_.~vector();
        __tens_.__shape_.~vector();
        __tens_.__strides_.~vector();
      }
    }

   private:
    tensor& __tens_;
  };

 public:
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI ~tensor() { __destroy_tensor (*this)(); }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI const_iterator begin() const _NOEXCEPT {
    return this->__data_.begin();
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI const_iterator end() const _NOEXCEPT {
    return this->__data_.end();
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI const_reverse_iterator rbegin() const _NOEXCEPT {
    return const_reverse_iterator(end());
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI const_reverse_iterator rend() const _NOEXCEPT {
    return const_reverse_iterator(begin());
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI data_container storage() const _NOEXCEPT { return this->__data_; }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI shape_type     shape() const _NOEXCEPT { return this->__shape_; }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI shape_type strides() const _NOEXCEPT { return this->__strides_; }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI size_t n_dims() const _NOEXCEPT { return this->__shape_.size(); }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI size_t size(const index_type __dim) const {
    if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
      throw std::invalid_argument("dimension input is out of range");
    if (__dim == 0)
      return this->__computeSize(this->__shape_);
    return this->__shape_[__dim];
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI size_t capacity() const _NOEXCEPT {
    return this->__data_.capacity();
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI reference       at(shape_type __idx);
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI const_reference at(const shape_type __idx) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI reference       operator[](const size_t __in) {
    if (__in >= this->__data_.size() || __in < 0)
      throw std::out_of_range("Access index is out of range");
    return this->__data_[__in];
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI const_reference operator[](const size_t __in) const {
    if (__in >= this->__data_.size() || __in < 0)
      throw std::out_of_range("Access index is out of range");
    return this->__data_[__in];
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type>
                                                      operator+(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type>
                                                      operator-(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type>
                                                      operator-=(const tensor<value_type>& __other) const _NOEXCEPT {
    return tensor<value_type>(*this - __other);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type>
                                                      operator+=(const tensor<value_type>& __other) const _NOEXCEPT {
    return tensor<value_type>(*this + __other);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> operator+(const value_type _scalar) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> operator-(const value_type _scalar) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI                    tensor<value_type>
  operator+=(const_reference __scalar) const _NOEXCEPT {
    return tensor<value_type>(*this + __scalar);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type>
                                                      operator-=(const_reference __scalar) const _NOEXCEPT {
    return tensor<value_type>(*this - __scalar);
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI bool
  operator==(const tensor<value_type>& __other) const _NOEXCEPT {
    if ((this->__shape_ != __other.shape()) && (this->__strides_ != __other.strides()))
      return false;
    return this->__data_ == __other.storage();
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI bool
  operator!=(const tensor<value_type>& __other) const _NOEXCEPT {
    return !(*this == __other);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI bool   empty() const _NOEXCEPT { return this->__data_.empty(); }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI double mean() const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<bool> logical_not() const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<bool> logical_or(const value_type __val) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<bool> logical_or(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<bool> less_equal(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<bool> less_equal(const value_type __val) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI              tensor<bool>
                                                      greater_equal(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<bool> greater_equal(const value_type __val) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<bool> equal(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<bool> equal(const value_type __val) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI size_t       count_nonzero(index_type __dim = -1LL) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI              tensor<value_type>
  slice(index_type __dim, std::optional<index_type> __start, std::optional<index_type> __end, index_type __step) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> fmax(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> fmod(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> frac() const {
    this->__check_is_scalar_type("Cannot get the fraction of a non-scalar type");
    std::vector<value_type> __ret(this->__data_.size());
    size_t                  __i = 0;
    for (; __i < this->__data_.size(); __i++)
      __ret[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));
    return tensor<value_type>(__ret, this->__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> log() const {
    this->__check_is_integral_type("Given data type must be an integral");
    std::vector<value_type> __ret(this->__data_.size());
    size_t                  __i = 0;
    for (; __i < this->__data_.size(); __i++)
      __ret[__i] = static_cast<value_type>(std::log(double(this->__data_[__i])));
    return tensor<value_type>(__ret, this->__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> log10() const {
    this->__check_is_integral_type("Given data type must be an integral");
    std::vector<value_type> __ret(this->__data_.size());
    size_t                  __i = 0;
    for (; __i < this->__data_.size(); __i++)
      __ret[__i] = static_cast<value_type>(std::log10(double(this->__data_[__i])));
    return tensor<value_type>(__ret, this->__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> log2() const {
    this->__check_is_integral_type("Given data type must be an integral");
    std::vector<value_type> __ret(this->__data_.size());
    size_t                  __i = 0;
    for (; __i < this->__data_.size(); __i++)
      __ret[__i] = static_cast<value_type>(std::log2(double(this->__data_[__i])));
    return tensor<value_type>(__ret, this->__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> exp() const {
    this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
    std::vector<value_type> __ret(this->__data_.size());
    size_t                  __i = 0;
    for (; __i < this->__data_.size(); __i++)
      __ret[__i] = static_cast<value_type>(std::exp(double(this->__data_[__i])));
    return tensor<value_type>(__ret, this->__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> sqrt() const {
    this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
    std::vector<value_type> __ret(this->__data_.size());
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      __ret[__i] = static_cast<value_type>(std::sqrt(double(this->__data_[__i])));
    return tensor<value_type>(__ret, this->__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> sum(const index_type _axis) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> row(const index_type _index) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> col(const index_type _index) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> ceil() const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> floor() const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> clone() const {
    std::vector<value_type> new_data  = this->__data_;
    std::vector<size_t>     new_shape = this->__shape_;
    return tensor<value_type>(new_data, new_shape);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type>
  clamp(const value_type* _min_val = nullptr, const value_type* _max_val = nullptr) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> cos() const {
    this->__check_is_scalar_type("Cannot perform a cosine on non-scalar data type");
    data_container __ret(this->__data_.size());
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      __ret[__i] = static_cast<value_type>(std::cos(static_cast<double>(this->__data_[__i])));
    return tensor<value_type>(__ret, this->__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> sin() const {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
    std::vector<value_type> __ret(this->__data_.size());
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      __ret[__i] = static_cast<value_type>(std::sin(static_cast<double>(this->__data_[__i])));
    return tensor<value_type>(__ret, this->__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> sinh() const {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
    std::vector<value_type> __ret(this->__data_.size());
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      __ret[__i] = static_cast<value_type>(std::sinh(static_cast<double>(this->__data_[__i])));
    return tensor<value_type>(__ret, this->__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> asinh() const {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
    std::vector<value_type> __ret(this->__data_.size());
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      __ret[__i] = static_cast<value_type>(std::asinh(static_cast<double>(this->__data_[__i])));
    return tensor<value_type>(__ret, this->__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> cosh() const {
    this->__check_is_scalar_type("Cannot perform a cosh on non-scalar data type");
    std::vector<value_type> __ret(this->__data_.size());
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      __ret[__i] = static_cast<value_type>(std::cosh(static_cast<double>(this->__data_[__i])));
    return tensor<value_type>(__ret, this->__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> acosh() const {
    this->__check_is_scalar_type("Cannot perform a acosh on non-scalar data type");
    std::vector<value_type> __ret(this->__data_.size());
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      __ret[__i] = static_cast<value_type>(std::acosh(static_cast<double>(this->__data_[__i])));
    return tensor<value_type>(__ret, this->__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type>
                                                      logical_xor(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> logical_xor(const value_type value) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI                    tensor<value_type>
                                                      logical_and(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> logical_and(const value_type value) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> bitwise_not() const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> bitwise_and(const value_type value) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI                    tensor<value_type>
                                                      bitwise_and(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> bitwise_or(const value_type value) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI                    tensor<value_type>
                                                      bitwise_or(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> bitwise_xor(const value_type value) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI                    tensor<value_type>
                                                      bitwise_xor(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> bitwise_left_shift(const int _amount) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> bitwise_right_shift(const int _amount) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI                    tensor<value_type>
                                                      matmul(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> reshape(const shape_type _shape) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> reshape_as(const tensor<value_type>& __other) {
    return this->reshape(__other.__shape_);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type>
                                                      cross_product(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type>
                                                      absolute(const tensor<value_type>& _tensor) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> dot(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> relu() const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> transpose() const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> pow(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> pow(const value_type value) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> cumprod(index_type _dim = -1) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI                    tensor<value_type>
  cat(const std::vector<tensor<value_type>>& _others, index_type _dim) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> argmax(index_type _dim) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<value_type> unsqueeze(index_type _dim) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI index_type         lcm() const {
    this->__check_is_scalar_type("Given template type must be an int");
    index_type __ret = static_cast<index_type>(this->__data_[0]);
    for (size_t __i = 1; __i < this->__data_.size(); __i++)
      __ret = this->__lcm(static_cast<index_type>(this->__data_[__i]), __ret);
    return __ret;
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void sqrt_() {
    this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(std::exp(double(this->__data_[__i])));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void exp_() {
    this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(std::exp(double(this->__data_[__i])));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void log2_() {
    this->__check_is_integral_type("Given data type must be an integral");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(std::log2(double(this->__data_[__i])));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void log10_() {
    this->__check_is_integral_type("Given data type must be an integral");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(std::log10(double(this->__data_[__i])));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void log_() {
    this->__check_is_integral_type("Given data type must be an integral");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(std::log(double(this->__data_[__i])));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void frac_() {
    this->__check_is_scalar_type("Cannot get the fraction of a non-scalar type");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(this->__frac(this->__data_[__i]));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void fmod_(const tensor<value_type>& __other);
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void cos_() {
    this->__check_is_scalar_type("Cannot perform a cosine on non-scalar data type");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(std::cos(static_cast<double>(this->__data_[__i])));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void cosh_() {
    this->__check_is_scalar_type("Cannot perform a cosh on non-scalar data type");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(std::cosh(static_cast<double>(this->__data_[__i])));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void acosh_() {
    this->__check_is_scalar_type("Cannot perform a acosh on non-scalar data type");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(std::acosh(static_cast<double>(this->__data_[__i])));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void sinh_() {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(std::sinh(static_cast<double>(this->__data_[__i])));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void asinh_() {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(std::asinh(static_cast<double>(this->__data_[__i])));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void ceil_() {
    this->__check_is_scalar_type("Cannot get the ceiling of a non scalar value");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = std::ceil(this->__data_[__i]);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void floor_() {
    this->__check_is_scalar_type("Cannot get the floor of a non scalar value");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = std::floor(this->__data_[__i]);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void sin_() const {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(std::sin(static_cast<double>(this->__data_[__i])));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 void                       relu_();
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void clamp_(const value_type* _min_val = nullptr,
                                                                  const value_type* _max_val = nullptr);
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void logical_not_() const { this->bitwise_not_(); }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void logical_or_(const tensor<value_type>& __other);
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void logical_or_(const value_type value) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
      throw std::runtime_error("Cannot get the element wise not of non-integral and non-boolean value");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = (this->__data_[__i] || value);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void logical_xor_(const tensor<value_type>& __other);
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void logical_xor_(const value_type value) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
      throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = (this->__data_[__i] ^ value);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void logical_and_(const tensor<value_type>& __other);
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void logical_and_(const value_type value) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
      throw std::runtime_error("Cannot get the element wise and of non-integral and non-boolean value");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = (this->__data_[__i] && value);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void pow_(const tensor<value_type>& __other) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void pow_(const value_type value) const {
    this->__check_is_integral_type("cannot get the power of a non integral value");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = static_cast<value_type>(std::pow(double(this->__data_[__i]), double(value)));
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void bitwise_left_shift_(const int _amount) {
    this->__check_is_integral_type("Cannot perform a bitwise left shift on non-integral values");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] <<= _amount;
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void bitwise_right_shift_(const int _amount) {
    this->__check_is_integral_type("Cannot perform a bitwise right shift on non-integral values");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] >>= amount;
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void bitwise_and_(const value_type value) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
      throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] &= value;
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void bitwise_and_(const tensor<value_type>& __other);
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void bitwise_or_(const value_type value) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
      throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] |= value;
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void bitwise_or_(const tensor<value_type>& __other);
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void bitwise_xor_(const value_type value) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
      throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] ^= value;
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void bitwise_xor_(const tensor<value_type>& __other);
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void bitwise_not_() {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
      throw std::runtime_error("Cannot perform a bitwise not on non integral or boolean value");
    for (size_t __i = 0; __i < this->__data_.size(); __i++)
      this->__data_[__i] = ~this->__data_[__i];
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void view(std::initializer_list<index_type> _new_sh);
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void print() const _NOEXCEPT;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI static tensor<value_type> zeros(const shape_type& _sh) {
    this->__check_is_scalar_type("template type must be a scalar : tensor.zeros()");
    std::vector<value_type> d(std::accumulate(_sh.begin(), _sh.end(), 1, std::multiplies<index_type>()), value_type(0));
    return tensor<value_type>(d, _sh);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI static tensor<value_type> ones(const shape_type& _sh) {
    this->__check_is_scalar_type("template type must be a scalar : tensor.ones()");
    std::vector<value_type> d(std::accumulate(_sh.begin(), _sh.end(), 1, std::multiplies<index_type>()), value_type(1));
    return tensor<value_type>(d, _sh);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20                       _LIBCPP_HIDE_FROM_ABI static tensor<value_type>
                                                      randomize(const shape_type& _sh, bool _bounded = false) _NOEXCEPT;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<index_type> argmax_(index_type _dim) const;
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI tensor<index_type> argsort(index_type _dim       = -1LL,
                                                                                 bool       _ascending = true) const;

 private:
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __check_is_scalar_type(const std::string __msg) {
    assert(!__msg.empty());
    if (!std::is_scalar<value_type>::value)
      throw std::runtime_error(__msg);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __check_is_integral_type(const std::string __msg) {
    assert(!__msg.empty());
    if (!std::is_integral<value_type>::value)
      throw std::runtime_error(__msg);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __check_is_same_type(class __t, const std::string __msg) {
    assert(!__msg.empty());
    if (!std::is_same<value_type, __t>::value)
      throw std::runtime_error(__msg);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __check_is_arithmetic_type(const std::string __msg) {
    assert(!__msg.empty());
    if (!std::is_arithmetic<value_type, __t>::value)
      throw std::runtime_error(__msg);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI void __compute_strides() {
    if (this->__shape_.empty()) {
      std::cerr << "Shape must be initialized before computing strides" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    this->__strides_  = std::vector<index_type>(this->__shape_.size(), 1);
    index_type stride = 1;
    for (index_type __i = this->__shape_.size() - 1; __i >= 0; __i--) {
      this->__strides_[__i] = stride;
      stride *= this->__shape_[__i];
    }
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI size_t
  __compute_index(const std::vector<index_type>& _idx) const {
    if (_idx.size() != this->__shape_.size())
      throw std::out_of_range("input indices does not match the tensor __shape_");
    size_t index = 0;
    for (size_t __i = 0; __i < this->__shape_.size(); __i++)
      index += _idx[__i] * this->__strides_[__i];
    return index;
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI uint64_t
  __computeSize(const std::vector<index_type>& _dims) const {
    uint64_t __ret = 0ULL;
    for (const index_type& d : _dims)
      __ret *= d;
    return __ret;
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI float __frac(const reference scalar) {
    return std::fmod(static_cast<float>(scalar), 1.0lf);
  }
  _LIBCPP_CONSTEXPR_SINCE_CXX20 _LIBCPP_HIDE_FROM_ABI index_type __lcm(const index_type a, const index_type b) {
    return (a * b) / std::gcd(a, b);
  }
};  // tensor class

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename tensor<_Tp>::reference tensor<_Tp>::at(tensor<_Tp>::shape_type __idx) {
  if (__idx.empty())
    throw std::invalid_argument("Passing an empty vector as indices for a tensor");
  index_type __i = this->__compute_index(__idx);
  if (__i < 0 || __i >= this->__data_.size())
    throw std::invalid_argument("input indices are out of bounds");
  return this->__data_[__i];
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 typename tensor<_Tp>::const_reference
tensor<_Tp>::at(const tensor<_Tp>::shape_type __idx) const {
  if (__idx.empty())
    throw std::invalid_argument("Passing an empty vector as indices for a tensor");
  index_type __i = this->__compute_index(__idx);
  if (__i < 0 || __i >= this->__data_.size())
    throw std::invalid_argument("input indices are out of bounds");
  return this->__data_[__i];
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::operator+(const tensor<tensor<_Tp>::value_type>& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  if (__other.shape() != this->__shape_)
    throw std::invalid_argument("Cannot add two tensors with different shapes");
  data_container new_data(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    new_data[__i] = this->__data_[__i] + __other[__i];
  return tensor<tensor<_Tp>::value_type>(new_data, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::operator-(const tensor<tensor<_Tp>::value_type>& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  if (__other.shape() != this->__shape_) {
    throw std::invalid_argument("Cannot add two tensors with different shapes");
  }
  data_container other_data = __other.storage();
  data_container new_data(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    new_data[__i] = this->__data_[__i] - other_data[__i];
  return tensor<tensor<_Tp>::value_type>(new_data, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::operator+(const tensor<_Tp>::value_type _scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    this->__data_[__i] += _scalar;
  return tensor<tensor<_Tp>::value_type>(*this);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::operator-(const tensor<_Tp>::value_type _scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    this->__data_[__i] -= _scalar;
  return tensor<tensor<_Tp>::value_type>(*this);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::randomize(const shape_type& _sh,
                                                                 bool              _bounded = false) _NOEXCEPT {
  this->__check_is_scalar_type("template class must be a scalar type");
  std::srand(time(nullptr));
  index_type _size = 1;
  for (index_type __d : _sh) {
    _size *= __d;
  }
  data_container d(_size);
  for (index_type __i = 0; __i < _size; __i++)
    d[__i] = _bounded ? tensor<_Tp>::value_type(static_cast<float>(rand() % RAND_MAX) : static_cast<float>(
                                                  rand());) return tensor<tensor<_Tp>::value_type>(d, _sh);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<typename tensor<_Tp>::index_type> tensor<_Tp>::argmax_(index_type _dim) const {
  if (_dim < 0 || _dim >= this->__shape_.size()) {
    throw std::out_of_range("Dimension out of range in argmax");
  }
  shape_type ret_shape = this->__shape_;
  ret_shape.erase(ret_shape.begin() + _dim);
  tensor<index_type> __ret;
  __ret.__shape_ = ret_shape;
  __ret.__data_.resize(this->computeSize(ret_shape), 0);
  index_type outer_size = 1LL;
  index_type inner_size = 1LL;
  for (index_type __i = 0; __i < _dim; ++__i)
    outer_size *= this->__shape_[__i];
  for (index_type __i = _dim + 1; __i < this->__shape_.size(); ++__i)
    inner_size *= this->__shape_[__i];
  for (index_type __i = 0; __i < outer_size; ++__i) {
    for (index_type j = 0; j < inner_size; ++j) {
      index_type              max_index = 0;
      tensor<_Tp>::value_type max_value = this->__data_[__i * this->__shape_[_dim] * inner_size + j];
      for (index_type k = 1; k < this->__shape_[_dim]; ++k) {
        tensor<_Tp>::value_type value = this->__data_[(__i * this->__shape_[_dim] + k) * inner_size + j];
        if (value > max_value) {
          max_value = value;
          max_index = k;
        }
      }
      __ret.__data_[__i * inner_size + j] = max_index;
    }
  }
  return __ret;
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::argmax(index_type _dim) const {
  if (_dim < 0 || _dim >= this->__shape_.size())
    throw std::out_of_range("Dimension out of range in argmax");
  shape_type ret_shape = this->__shape_;
  ret_shape.erase(ret_shape.begin() + _dim);
  tensor<tensor<_Tp>::value_type> __ret;
  __ret.__shape_ = ret_shape;
  __ret.__data_.resize(this->computeSize(ret_shape), tensor<_Tp>::value_type(0));
  index_type outer_size = 1;
  index_type inner_size = 1;
  for (index_type __i = 0; __i < _dim; __i++)
    outer_size *= this->__shape_[__i];
  for (index_type __i = _dim + 1; __i < static_cast<index_type>(this->__shape_.size()); __i++)
    inner_size *= this->__shape_[__i];
  for (index_type __i = 0; __i < outer_size; __i++) {
    for (index_type j = 0; j < inner_size; j++) {
      tensor<_Tp>::value_type max_value = this->__data_[__i * this->__shape_[_dim] * inner_size + j];
      for (index_type k = 1; k < this->__shape_[_dim]; k++) {
        tensor<_Tp>::value_type value = this->__data_[(__i * this->__shape_[_dim] + k) * inner_size + j];
        if (value > max_value)
          max_value = value;
      }
      __ret.__data_[__i * inner_size + j] = max_value;
    }
  }
  return __ret;
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::unsqueeze(index_type _dim) const {
  if (_dim < 0 || _dim > static_cast<index_type>(this->__shape_.size()))
    throw std::out_of_range("Dimension out of range in unsqueeze");
  shape_type new_shape = this->__shape_;
  new_shape.insert(new_shape.begin() + _dim, 1);
  tensor<tensor<_Tp>::value_type> __ret;
  __ret.__shape_ = new_shape;
  __ret.__data_  = this->__data_;
  return __ret;
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void tensor<_Tp>::view(std::initializer_list<index_type> _new_sh) {
  index_type new_size = 1LL;
  for (index_type __d : _new_sh)
    new_size *= __d;
  if (new_size != this->__data_.size())
    throw std::invalid_argument("_Tpotal elements do not match for new shape");
  this->__shape_ = _new_sh;
  this->__compute_strides();
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::cat(const std::vector<tensor<_Tp>>& _others,
                                                           index_type                      _dim) const {
  for (const tensor<tensor<_Tp>::value_type>& t : _others)
    for (index_type __i = 0; __i < this->__shape_.size(); __i++)
      if (__i != _dim && this->__shape_[__i] != t.__shape_[__i])
        throw std::invalid_argument(
          "Cannot concatenate tensors with different shapes along non-concatenation dimensions");
  shape_type ret_sh = this->__shape_;
  for (const tensor<tensor<_Tp>::value_type>& t : _others)
    ret_sh[_dim] += t.__shape_[_dim];
  data_container c_data;
  c_data.reserve(this->__data_.size());
  c_data.insert(c_data.end(), this->__data_.begin(), this->__data_.end());
  for (const tensor<tensor<_Tp>::value_type>& t : _others)
    c_data.insert(c_data.end(), t.__data_.begin(), t.__data_.end());
  tensor<tensor<_Tp>::value_type> __ret(c_data, ret_sh);
  return __ret;
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::cumprod(index_type _dim = -1) const {
  if (_dim == -1) {
    data_container flattened_data = this->__data_;
    data_container __ret(flattened_data.size());
    __ret[0] = flattened_data[0];
    for (size_t __i = 1; __i < flattened_data.size(); __i++)
      __ret[__i] = __ret[__i - 1] * flattened_data[__i];
    return tensor<tensor<_Tp>::value_type>(__ret, {flattened_data.size()});
  }
  else {
    if (_dim < 0 || _dim >= static_cast<index_type>(this->__shape_.size()))
      throw std::invalid_argument("Invalid dimension provided.");
    data_container __ret(this->__data_);
    // TODO : compute_outer_size() implementation
    size_t outer_size = this->compute_outer_size(_dim);
    size_t inner_size = this->__shape_[_dim];
    size_t stride     = this->__strides_[_dim];
    for (size_t __i = 0; __i < outer_size; ++__i) {
      size_t base_index = __i * stride;
      __ret[base_index] = __data_[base_index];

      for (size_t j = 1; j < inner_size; ++j) {
        size_t current_index = base_index + j;
        __ret[current_index] = __ret[base_index + j - 1] * __data_[current_index];
      }
    }
    return tensor<tensor<_Tp>::value_type>(__ret, this->__shape_);
  }
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::slice(index_type                _dim,
                                                             std::optional<index_type> _start,
                                                             std::optional<index_type> _end,
                                                             index_type                _step) const {
  tensor<tensor<_Tp>::value_type> __ret;
  if (_dim < 0ULL || _dim >= static_cast<index_type>(this->__shape_.size()))
    throw std::out_of_range("Dimension out of range.");
  index_type size      = this->__shape_[_dim];
  index_type start_idx = _start.value_or(0ULL);
  index_type end_idx   = _end.value_or(size);
  if (start_idx < 0ULL) {
    start_idx += size;
  }
  if (end_idx < 0ULL) {
    end_idx += size;
  }
  start_idx             = std::max(index_type(0ULL), std::min(start_idx, size));
  end_idx               = std::max(index_type(0ULL), std::min(end_idx, size));
  index_type slice_size = (end_idx - start_idx + _step - 1) / _step;
  shape_type ret_dims   = this->__shape_;
  ret_dims[_dim]        = slice_size;
  __ret                 = tensor<tensor<_Tp>::value_type>(ret_dims);
  for (index_type __i = start_idx, j = 0ULL; __i < end_idx; __i += _step, j++)
    __ret({j}) = this->__data_[compute_index({__i})];
  return __ret;
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::fmod(const tensor<_Tp>& __other) const {
  this->__check_is_scalar_type("Cannot divide non scalar values");
  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");
  data_container __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = static_cast<tensor<_Tp>::value_type>(std::fmod(double(this->__data_[__i]), double(__other[__i])));
  return tensor<tensor<_Tp>::value_type>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void tensor<_Tp>::fmod_(const tensor<_Tp>& __other) {
  this->__check_is_scalar_type("Cannot divide non scalar values");
  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
    throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    this->__data_[__i] =
      static_cast<tensor<_Tp>::value_type>(std::fmax(double(this->__data_[__i]), double(__other[__i])));
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::fmax(const tensor<tensor<_Tp>::value_type>& __other) const {
  this->__check_is_scalar_type("Cannot deduce the maximum of non scalar values");
  if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
    throw std::invalid_argument("Cannot compare two tensors of different shapes : fmax");
  data_container __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = static_cast<tensor<_Tp>::value_type>(std::fmax(double(this->__data_[__i]), double(__other[__i])));
  return tensor<tensor<_Tp>::value_type>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 size_t tensor<_Tp>::count_nonzero(index_type dim = -1LL) const {
  this->__check_is_scalar_type("Cannot compare a non scalar value to zero");
  size_t count = 0;
  if (dim == -1) {
    for (const tensor<_Tp>::value_type& elem : __data_)
      if (elem != 0) {
        count++;
      }
  }
  else {
    if (dim < 0 || dim >= static_cast<index_type>(__shape_.size()))
      throw std::invalid_argument("Invalid dimension provided.");
    throw std::runtime_error("Dimension-specific non-zero counting is not implemented yet.");
  }
  return count;
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::matmul(const tensor<tensor<_Tp>::value_type>& __other) const {
  if (this->__shape_.size() != 2 || __other.shape().size() != 2)
    throw std::invalid_argument("matmul is only supported for 2D tensors");

  if (this->__shape_[1] != __other.shape()[0])
    throw std::invalid_argument("Shape mismatch for matrix multiplication");

  shape_type     ret_sh = {this->__shape_[0], __other.shape()[1]};
  data_container ret_data(ret_sh[0] * ret_sh[1]);

  for (index_type __i = 0; __i < ret_sh[0]; __i++)
    for (index_type j = 0; j < ret_sh[1]; j++) {
      tensor<_Tp>::value_type sum = 0.0f;
      for (index_type k = 0; k < this->__shape_[1]; k++)
        sum += this->at({__i, k}) * __other.at({k, j});
      ret_data[__i * ret_sh[1] + j] = sum;
    }

  return tensor<tensor<_Tp>::value_type>(ret_data, ret_sh);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::reshape(const shape_type _shape) const {
  data_container d     = this->__data_;
  size_t         _size = this->computeSize(_shape);
  if (_size != this->__data_.size())
    throw std::invalid_argument(
      "input shape must have size of elements equal to the current number of elements in the tensor data");
  return tensor<_T>(d, _shape);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp>
                              tensor<_Tp>::cross_product(const tensor<tensor<_Tp>::value_type>& __other) const {
  if (this->empty() || __other.empty()) {
    throw std::invalid_argument("Cannot cross product an empty vector");
  }
  this->__check_is_arithmetic_type("Cannot perform a cross product on non-scalar data types");
  if (this->shape() != std::vector<int>{3} || __other.shape() != std::vector<int>{3})
    throw std::invalid_argument("Cross product can only be performed on 3-element vectors");
  tensor<tensor<_Tp>::value_type> __ret({3});
  const tensor<_Tp>::reference    a1 = this->storage()[0];
  const tensor<_Tp>::reference    a2 = this->storage()[1];
  const tensor<_Tp>::reference    a3 = this->storage()[2];
  const tensor<_Tp>::reference    b1 = __other.storage()[0];
  const tensor<_Tp>::reference    b2 = __other.storage()[1];
  const tensor<_Tp>::reference    b3 = __other.storage()[2];
  __ret.storage()[0]                 = a2 * b3 - a3 * b2;
  __ret.storage()[1]                 = a3 * b1 - a1 * b3;
  __ret.storage()[2]                 = a1 * b2 - a2 * b1;
  return __ret;
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::absolute(const tensor<tensor<_Tp>::value_type>& _tensor) const {
  this->__check_is_scalar_type("Cannot call absolute on non scalar value");
  data_container a;
  for (const tensor<_Tp>::reference v : _tensor.storage())
    a.push_back(static_cast<tensor<_Tp>::value_type>(std::fabs(float(v))));
  return tensor<tensor<_Tp>::value_type>(a, _tensor.__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::dot(const tensor<tensor<_Tp>::value_type>& __other) const {
  if (this->empty() || __other.empty())
    throw std::invalid_argument("Cannot dot product an empty vector");
  this->__check_is_scalar_type("Cannot perform a dot product on non scalar data types");
  if (this->shape().size() == 1 && __other.shape().size()) {
    assert(this->shape()[0] == __other.shape()[0]);
    tensor<_Tp>::value_type __ret = 0;
    for (index_type __i = 0; __i < this->__data_.size(); __i++)
      ret_data += this->__data_[__i] * __other[__i];
    return tensor<tensor<_Tp>::value_type>({__ret}, {1});
  }
  if (this->shape().size() == 2 && __other.shape().size())
    return this->matmul(__other);
  if (this->shape().size() == 3 && __other.shape().size())
    return this->cross_product(__other);
  return tensor<tensor<_Tp>::value_type>();
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::relu() const {
  this->__check_is_scalar_type("Cannot relu non-scalar type");
  size_t         _size = this->__data_.size();
  data_container new_data(_size);
  for (size_t __i = 0; __i < _size; __i++)
    new_data[__i] = std::max(this->__data_[__i], tensor<_Tp>::value_type(0));
  return tensor<tensor<_Tp>::value_type>(new_data, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void tensor<_Tp>::relu_() {
  this->__check_is_scalar_type("Cannot relu non-scalar type");
  size_t _size = this->__data_.size();
  for (size_t __i = 0; __i < _size; __i++)
    this->__data_[__i] = std::max(this->__data_[__i], tensor<_Tp>::value_type(0));
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::transpose() const {
  if (this->__shape_.size() != 2) {
    std::cerr << "Matrix transposition can only be done on 2D tensors" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  tensor<tensor<_Tp>::value_type> transposed({this->__shape_[1], this->__shape_[0]});
  for (index_type __i = 0; __i < this->__shape_[0]; __i++)
    for (index_type j = 0; j < this->__shape_[1]; j++)
      transposed.at({j, __i}) = this->at({__i, j});
  return transposed;
}

/*I did this wrong and forgot to deal with one dimensional __data_*/
template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<typename tensor<_Tp>::index_type>
                              tensor<_Tp>::argsort(index_type __d = -1LL, bool __ascending = true) const {
  index_type __adjusted = (__d < 0LL) ? __d + this->__data_.size() : __d;
  if (__adjusted < 0LL || __adjusted >= static_cast<index_type>(this->__data_.size()))
    throw std::out_of_range("Invalid dimension for argsort");
  std::vector<std::vector<index_type>> __indices = this->__data_;
  size_t                               __i       = 0;
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

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::bitwise_left_shift(const int _amount) const {
  this->__check_is_integral_type("Cannot perform a bitwise left shift on non-integral values");
  data_container __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = this->__data_[__i] << _amount;
  return tensor<tensor<_Tp>::value_type>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp>
                              tensor<_Tp>::bitwise_xor(const tensor<tensor<_Tp>::value_type>& __other) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");
  assert(this->shape() == __other.shape() && this->size(0) == __other.size(0));
  data_container __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = this->__data_[__i] ^ __other[__i];
  return tensor<tensor<_Tp>::value_type>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::bitwise_right_shift(const int _amount) const {
  this->__check_is_integral_type("Cannot perform a bitwise right shift on non-integral values");
  data_container __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = this->__data_[__i] >> _amount;
  return tensor<tensor<_Tp>::value_type>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void tensor<_Tp>::bitwise_and_(const tensor<tensor<_Tp>::value_type>& __other) {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise AND on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    this->__data_[__i] &= __other[__i];
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void tensor<_Tp>::bitwise_or_(const tensor<tensor<_Tp>::value_type>& __other) {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise OR on non-integral or non-boolean values");

  assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    this->__data_[__i] |= __other[__i];
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void tensor<_Tp>::bitwise_xor_(const tensor<tensor<_Tp>::value_type>& __other) {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");
  assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    this->__data_[__i] ^= __other[__i];
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<bool> tensor<_Tp>::logical_not() const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot get the element wise not of non-integral and non-boolean value");
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = ~(this->__data_[__i]);
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<bool> tensor<_Tp>::logical_or(const tensor<_Tp>::value_type value) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot get the element wise or of non-integral and non-boolean value");
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] || value);
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<bool> tensor<_Tp>::logical_or(const tensor<_Tp>& __other) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot get the element wise or of non-integral and non-boolean value");
  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] || __other[__i]);
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp>
                              tensor<_Tp>::logical_xor(const tensor<tensor<_Tp>::value_type>& __other) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] ^ __other[__i]);
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::logical_xor(const _Tp value) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] ^ value);
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp>
                              tensor<_Tp>::logical_and(const tensor<tensor<_Tp>::value_type>& __other) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot get the element wise and of non-integral and non-boolean value");
  assert(this->__shape_ == __other.shape());
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] && __other[__i]);
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::logical_and(const _Tp value) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot get the element wise and of non-integral and non-boolean value");
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] && __other[__i]);
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void tensor<_Tp>::logical_or_(const tensor<_Tp>& __other) {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot get the element wise not of non-integral and non-boolean value");
  assert(this->__shape_ == __other.shape());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    this->__data_[__i] = (this->__data_[__i] || __other[__i]);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void tensor<_Tp>::logical_xor_(const tensor<_Tp>& __other) {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
  assert(this->__shape_ == __other.shape());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    this->__data_[__i] = (this->__data_[__i] ^ __other[__i]);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void tensor<_Tp>::logical_and_(const tensor<_Tp>& __other) {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_same<tensor<_Tp>::value_type, bool>::value)
    throw std::runtime_error("Cannot get the element wise and of non-integral and non-boolean value");
  assert(this->__shape_ == __other.shape());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    this->__data_[__i] = (this->__data_[__i] && __other[__i]);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 double tensor<_Tp>::mean() const {
  this->__check_is_integral_type("input must be of integral type to get the mean average");
  double mean = 0.0lf;
  if (this->empty())
    return 0.0lf;
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    mean += this->__data_[__i];
  return static_cast<double>(mean / double(this->__data_.size()));
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::pow(const tensor<tensor<_Tp>::value_type>& __other) const {
  this->__check_is_integral_type("cannot get the power of a non integral value");
  assert(this->__shape_ == __other.shape() && this->size(0) == __other.shape());
  data_container __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = static_cast<tensor<_Tp>::value_type>(std::pow(double(this->__data_[__i]), double(__other[__i])));
  return tensor<tensor<_Tp>::value_type>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::pow(const tensor<_Tp>::value_type value) const {
  this->__check_is_integral_type("cannot get the power of a non integral value");
  data_container __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = static_cast<tensor<_Tp>::value_type>(std::pow(double(this->__data_[__i]), double(value)));
  return tensor<tensor<_Tp>::value_type>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void tensor<_Tp>::pow_(const tensor<tensor<_Tp>::value_type>& __other) const {
  this->__check_is_integral_type("cannot get the power of a non integral value");
  assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    this->__data_[__i] =
      static_cast<tensor<_Tp>::value_type>(std::pow(double(this->__data_[__i]), double(__other[__i])));
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<bool>
                              tensor<_Tp>::less_equal(const tensor<tensor<_Tp>::value_type>& __other) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_scalar<tensor<_Tp>::value_type>::value)
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] <= __other[__i]) ? true : false;
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<bool> tensor<_Tp>::less_equal(const tensor<_Tp>::value_type value) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_scalar<tensor<_Tp>::value_type>::value)
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] <= value) ? true : false;
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<bool>
                              tensor<_Tp>::greater_equal(const tensor<tensor<_Tp>::value_type>& __other) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_scalar<tensor<_Tp>::value_type>::value)
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] >= __other[__i]) ? true : false;
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<bool> tensor<_Tp>::greater_equal(const tensor<_Tp>::value_type value) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_scalar<tensor<_Tp>::value_type>::value)
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] >= value) ? true : false;
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<bool> tensor<_Tp>::equal(const tensor<tensor<_Tp>::value_type>& __other) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_scalar<tensor<_Tp>::value_type>::value)
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] == __other[__i]) ? true : false;
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<bool> tensor<_Tp>::equal(const tensor<_Tp>::value_type value) const {
  if (!std::is_integral<tensor<_Tp>::value_type>::value && !std::is_scalar<tensor<_Tp>::value_type>::value)
    throw std::runtime_error("Cannot compare non-integral or scalar value");
  std::vector<bool> __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = (this->__data_[__i] >= value) ? true : false;
  return tensor<bool>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::sum(const index_type _axis) const {
  this->__check_is_scalar_type("Cannot reduce tensor with non scalar type");
  if (_axis < 0LL || _axis >= static_cast<index_type>(this->__shape_.size()))
    throw std::invalid_argument("Invalid axis for sum");
  shape_type ret_shape    = this->__shape_;
  ret_shape[_axis]        = 1LL;
  index_type     ret_size = std::accumulate(ret_shape.begin(), ret_shape.end(), 1, std::multiplies<index_type>());
  data_container ret_data(ret_size, tensor<_Tp>::value_type(0.0f));
  for (index_type __i = 0; __i < static_cast<index_type>(this->__data_.size()); ++__i) {
    std::vector<index_type> original_indices(this->__shape_.size());
    index_type              index = __i;
    for (index_type j = static_cast<index_type>(this->__shape_.size()) - 1LL; j >= 0LL; j--) {
      original_indices[j] = index % this->__shape_[j];
      index /= this->__shape_[j];
    }
    original_indices[_axis] = 0LL;
    index_type ret_index    = 0LL;
    index_type stride       = 1LL;
    for (index_type j = static_cast<index_type>(this->__shape_.size()) - 1LL; j >= 0LL; j--) {
      ret_index += original_indices[j] * stride;
      stride *= ret_shape[j];
    }
    ret_data[ret_index] += this->__data_[__i];
  }
  return tensor<tensor<_Tp>::value_type>(ret_data, ret_shape);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::row(const index_type _index) const {
  if (this->__shape_.size() != 2)
    throw std::runtime_error("Cannot get a row from a non two dimensional tensor");
  if (this->__shape_[0] <= _index || _index < 0LL)
    throw std::invalid_argument("Index input is out of range");
  data_container R(this->__data_.begin() + (this->__shape_[1] * _index),
                   this->__data_.begin() + (this->__shape_[1] * _index + this->__shape_[1]));
  return tensor<tensor<_Tp>::value_type>(R, {this->__shape_[1]});
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::col(const index_type _index) const {
  if (this->__shape_.size() != 2)
    throw std::runtime_error("Cannot get a column from a non two dimensional tensor");
  if (this->__shape_[1] <= _index || _index < 0LL)
    throw std::invalid_argument("Index input out of range");
  data_container C;
  for (index_type __i = 0LL; __i < this->__shape_[0]; __i++)
    C.push_back(this->__data_[this->compute_index({__i, _index})]);
  return tensor<tensor<_Tp>::value_type>(C, {this->__shape_[0]});
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::ceil() const {
  this->__check_is_scalar_type("Cannot get the ceiling of a non scalar value");
  data_container __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = std::ceil(this->__data_[__i]);
  return tensor<tensor<_Tp>::value_type>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::floor() const {
  this->__check_is_scalar_type("Cannot get the floor of a non scalar value");
  data_container __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++)
    __ret[__i] = std::floor(this->__data_[__i]);
  return tensor<tensor<_Tp>::value_type>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 tensor<_Tp> tensor<_Tp>::clamp(const tensor<_Tp>::value_type* _min_val = nullptr,
                                                             const tensor<_Tp>::value_type* _max_val = nullptr) const {
  data_container __ret(this->__data_.size());
  for (size_t __i = 0; __i < this->__data_.size(); __i++) {
    tensor<_Tp>::value_type value = this->__data_[__i];
    if (_min_val)
      value = std::max(*_min_val, value);
    if (_max_val)
      value = std::min(*_max_val, value);
    __ret[__i] = value;
  }
  return tensor<tensor<_Tp>::value_type>(__ret, this->__shape_);
}

template<class _Tp>
_LIBCPP_CONSTEXPR_SINCE_CXX20 void tensor<_Tp>::clamp_(const tensor<_Tp>::value_type* _min_val = nullptr,
                                                       const tensor<_Tp>::value_type* _max_val = nullptr) {
  for (size_t __i = 0; __i < this->__data_.size(); __i++) {
    if (_min_val)
      this->__data_[__i] = std::max(*_min_val, this->__data_[__i]);
    if (_max_val)
      this->__data_[__i] = std::min(*_max_val, this->__data_[__i]);
  }
}

_LIBCPP_END_NAMESPACE_STD


void test() { std::tens }