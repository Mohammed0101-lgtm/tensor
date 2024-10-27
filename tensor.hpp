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
  typedef std::vector<value_type> data_t;
  typedef const data_t    const_data_container;

  typedef std::__wrap_iter<typename std::allocator_traits<std::allocator<_Tp>>::pointer>       __iterator;
  typedef std::__wrap_iter<typename std::allocator_traits<std::allocator<_Tp>>::const_pointer> __const_iterator;
  typedef std::reverse_iterator<__iterator>                                                    reverse_iterator;
  typedef std::reverse_iterator<__const_iterator>                                              const_reverse_iterator;


 private:
  data_t          __data_;
  shape_type              __shape_;
  std::vector<index_type> __strides_;

 public:
  tensor() = default;

  explicit tensor(const shape_type __sh, const value_type __v);
  explicit tensor(const shape_type __sh);
  explicit tensor(const data_t __d, const shape_type __sh);
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

  data_t storage() const;
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

enum class Device {
    CPU,
    CUDA
};

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

    data_t     storage() const noexcept { return this->__data_; }
    shape_type shape() const noexcept { return this->__shape_; }
    shape_type strides() const noexcept { return this->__strides_; }
    Device     device() const noexcept { return this->__device_; }
    size_t     n_dims() const noexcept { return this->__shape_.size(); }

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

    tensor slice(index_type                __dim,
                 std::optional<index_type> __start,
                 std::optional<index_type> __end,
                 index_type                __step) const;

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
        data_t     __d = this->__data_;
        shape_type __s = this->__shape_;
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

        std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                       [](const_reference __v) { return static_cast<value_type>(std::sinh(static_cast<double>(__v))) });
    }

    void asinh_() {
        this->__check_is_scalar_type("Cannot perform asinh on non-scalar data type");

        std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
            return static_cast<value_type>(std::asinh(static_cast<double>(__v)))
        });
    }

    void ceil_() {
        this->__check_is_scalar_type("Cannot get the ceiling of a non scalar value");

        std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                       [](const_reference __v) { return static_cast<value_type>(std::ceil(static_cast<double>(__v))) });
    }

    void floor_() {
        this->__check_is_scalar_type("Cannot get the floor of a non scalar value");

        std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [](const_reference __v) {
            return static_cast<value_type>(std::floor(static_cast<double>(__v)))
        });
    }

    void sin_() const {
        this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");

        std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                       [](const_reference __v) { return static_cast<value_type>(std::sin(static_cast<double>(__v))) });
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

        std::transform(
            this->__data_.begin(), this->__data_.end(), this->__data_.begin(), [&__val](const_reference __v) {
                return static_cast<value_type>(std::pow(static_cast<double>(__v), static_cast<double>(__val)))
            });
    }

    void bitwise_left_shift_(const int __amount) {
        this->__check_is_integral_type("Cannot perform a bitwise left shift on non-integral values");

        std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                       [&__amount](const_reference __v) { return __v <<= __amount; });
    }

    void bitwise_right_shift_(const int __amount) {
        this->__check_is_integral_type("Cannot perform a bitwise right shift on non-integral values");

        std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                       [&__amount](const_reference __v) { return __v >>= amount; });
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
        data_t __d(std::accumulate(__sh.begin(), __sh.end(), 1, std::multiplies<index_type>()), value_type(0));
        return __self(__d, __sh);
    }

    static tensor ones(const shape_type& __sh) {
        __check_is_scalar_type("template type must be a scalar : tensor.ones()");
        data_t __d(std::accumulate(__sh.begin(), __sh.end(), 1, std::multiplies<index_type>()), value_type(1));
        return __self(__d, __sh);
    }

    void zeros_(const shape_type& __sh = {}) {
        if (__sh.empty())
        {
            __sh = this->__shape_;
        }

        this->__data_ =
            data_t(std::accumulate(__sh.begin(), __sh.end(), 1, std::multiplies<index_type>()), value_type(0));
    }

    void ones_(const shape_type& __sh = {}) {
        if (__sh.empty())
        {
            __sh = this->__shape_;
        }

        this->__data_ =
            data_t(std::accumulate(__sh.begin(), __sh.end(), 1, std::multiplies<index_type>()), value_type(0));
    }

    tensor randomize(const shape_type& __sh, bool __bounded = false);
    void   randomize_(const shape_type& __sh, bool __bounded = false);

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

    static uint64_t __computeSize(const shape_type& __dims) {
        uint64_t __ret = 1ULL;
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
    data_t __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __ret.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::sinh(static_cast<double>(__v))); });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::sin() const {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
    data_t __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __ret.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::sin(static_cast<double>(__v))); });

    return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::frac() const {
    this->__check_is_scalar_type("Cannot get the fraction of a non-scalar type");
    data_t __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __ret.begin(),
                   [this](const_reference __v) { return static_cast<value_type>(this->__frac(__v)); });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::cos() const {
    this->__check_is_scalar_type("Cannot perform a cosine on non-scalar data type");
    data_t __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __ret.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::cos(static_cast<double>(__v))); });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::log() const {
    this->__check_is_integral_type("Given data type must be an integral");
    std::vector<value_type> __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __ret.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::log(static_cast<double>(__v))); });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::asinh() const {
    this->__check_is_scalar_type("Cannot perform a sin on non-scalar data type");
    data_t __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __ret.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::asinh(static_cast<double>(__v))); });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::cosh() const {
    this->__check_is_scalar_type("Cannot perform a cosh on non-scalar data type");
    data_t __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __ret.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::cosh(static_cast<double>(__v))); });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::sqrt() const {
    this->__check_is_scalar_type("Cannot get the square root of non scalar values");
    data_t __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __ret.begin(),
                   [](const auto& val) { return static_cast<value_type>(std::sqrt(static_cast<double>(val))); });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::acosh() const {
    this->__check_is_scalar_type("Cannot perform a acosh on non-scalar data type");
    data_t __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __ret.begin(),
                   [](const auto& val) { return static_cast<value_type>(std::acosh(static_cast<double>(val))); });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::log10() const {
    this->__check_is_integral_type("Given data type must be an integral");
    data_t __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __ret.begin(),
                   [](const auto& val) { return static_cast<value_type>(std::acosh(static_cast<double>(val))); });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::log2() const {
    this->__check_is_integral_type("Given data type must be an integral");
    std::vector<value_type> __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __ret.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::log2(static_cast<double>(__v))); });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::exp() const {
    this->__check_is_scalar_type("Cannot get the exponential of non scalar values");
    std::vector<value_type> __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __ret.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::exp(static_cast<double>(__v))); });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
typename tensor<_Tp>::index_type tensor<_Tp>::lcm() const {
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
tensor<_Tp> tensor<_Tp>::sinh() const {
    this->__check_is_integral_type("Given template type must be integral");
    data_t __ret(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), this->__ret.begin(),
                   [](const_reference __v) { return static_cast<value_type>(std::sinh(static_cast<double>(__v))); });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
void tensor<_Tp>::logical_or_(const value_type __val) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    {
        throw std::runtime_error("Cannot get the element wise not of non-integral and non-boolean value");
    }

    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__val](const_reference __v) { return static_cast<value_type>(__v || __val); });
}


template<class _Tp>
void tensor<_Tp>::logical_xor_(const value_type __val) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    {
        throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
    }

    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__val](const_reference __v) { return static_cast<value_type>(__v ^ __val); });
}


template<class _Tp>
void tensor<_Tp>::logical_and_(const value_type __val) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    {
        throw std::runtime_error("Cannot get the element wise and of non-integral and non-boolean value");
    }

    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__val](const_reference __v) { return static_cast<value_type>(__v && __val); });
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
    for (size_t i = 0; i < this->__shape_.size(); ++i)
    {
        std::cout << this->__shape_[i];
        if (i < this->__shape_.size() - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]\nData: ";
    if (!this->__data_.empty())
    {
        for (const auto& __v : this->__data_)
        {
            std::cout << __v << " ";
        }
    }
    else
    {
        std::cout << "Empty";
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

    data_t __d(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), __d.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v + __w); });

    return __self(__d, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const tensor& __other) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");
    if (__other.shape() != this->__shape_)
    {
        throw std::invalid_argument("Cannot add two tensors with different shapes");
    }

    data_t __d(this->__data_.size());

    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), __d.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v - __w); });

    return __self(__d, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+(const value_type __scalar) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");

    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__scalar](const_reference __v) { return static_cast<value_type>(__v + __scalar); });

    return __self(*this);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-(const value_type __scalar) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");

    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__scalar](const_reference __v) { return static_cast<value_type>(__v - __scalar); });

    return __self(*this);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+=(const tensor& __other) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");
    assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));

    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v + __w); });

    return *this;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator+=(const_reference __scalar) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");

    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [&__scalar](const_reference __v) { return static_cast<value_type>(__v + __scalar); });
    return *this;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-=(const tensor& __other) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");
    assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));

    std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v - __w); });
    return *this;
}


/*
template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator*=(const tensor& __other) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");
  assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));

  std::transform(this->__data_.begin(), this->__data_.end(), __other.storage().begin(), this->__data_.begin(),
                 [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v * __w); });
  return *this;
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator*=(const_reference __scalar) const {
  this->__check_is_arithmetic_type("template class must be an arithmetic type");

  std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                 [](const_reference __v) { return static_cast<value_type>(__v * __scalar); });
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
*/


template<class _Tp>
tensor<_Tp> tensor<_Tp>::operator-=(const_reference __scalar) const {
    this->__check_is_arithmetic_type("template class must be an arithmetic type");

    std::transform(this->__data_.begin(), this->__data_.end(), this->__data_.begin(),
                   [](const_reference __v, const_reference __w) { return static_cast<value_type>(__v - __w); });
    return *this;
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::randomize(const shape_type& __sh, bool __bounded) {
    this->__check_is_scalar_type("template class must be a scalar type");

    size_t __s = this->__computeSize(__sh);
    data_t __d(__s);

    std::random_device rd;
    std::mt19937       gen(rd());

    std::uniform_real_distribution<float> bounded_dist(0.0f, static_cast<float>(RAND_MAX));
    std::uniform_real_distribution<float> unbounded_dist(0.0f, 1.0f);

    index_type __i = 0;
    for (; __i < static_cast<index_type>(__s); __i++)
    {
        __d[__i] = value_type(__bounded ? bounded_dist(gen) : unbounded_dist(gen));
    }

    return tensor(__d, __sh);
}


template<class _Tp>
void tensor<_Tp>::randomize_(const shape_type& __sh, bool __bounded) {
    this->__check_is_scalar_type("template class must be a scalar type");

    size_t __s;
    if (this->__shape_ != __sh)
    {
        __s = this->__computeSize(__sh);
    }
    else
    {
        __s = this->__data_.size();
    }

    if (__s != this->__data_.size())
    {
        this->__data_.resize(__s);
    }

    std::random_device rd;
    std::mt19937       gen(rd());

    std::uniform_real_distribution<float> bounded_dist(0.0f, static_cast<float>(RAND_MAX));
    std::uniform_real_distribution<float> unbounded_dist(0.0f, 1.0f);

    index_type __i = 0;
    for (; __i < static_cast<index_type>(__s); __i++)
    {
        this->__data_[__i] = value_type(__bounded ? bounded_dist(gen) : unbounded_dist(gen));
    }
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
                        __max_value = __max_values[i];
                        __max_index = __indices[i];
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

                __ret.__data_[__i * __inner_size + __j] = max_index;
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
                float32x4_t max_vec       = vdupq_n_f32(-std::numeric_limits<float>::infinity());
                uint32x4_t  index_vec     = vdupq_n_u32(0);
                uint32x4_t  increment     = vdupq_n_u32(1);
                uint32x4_t  current_index = {0, 1, 2, 3};

                index_type __k = 0;
                for (; __k + 4 <= this->__shape_[__dim]; __k += 4)
                {
                    float32x4_t data_vec =
                        vld1q_f32(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
                    uint32x4_t mask = vcgtq_f32(data_vec, max_vec);
                    max_vec         = vbslq_f32(mask, data_vec, max_vec);
                    index_vec       = vbslq_u32(mask, current_index, index_vec);
                    current_index   = vaddq_u32(current_index, increment);
                }

                float    max_values[4];
                uint32_t indices[4];
                vst1q_f32(max_values, max_vec);
                vst1q_u32(indices, index_vec);

                float      max_value = max_values[0];
                index_type max_index = indices[0];
                for (int i = 1; i < 4; ++i)
                {
                    if (max_values[i] > max_value)
                    {
                        max_value = max_values[i];
                        max_index = indices[i];
                    }
                }

                for (; __k < this->__shape_[__dim]; __k++)
                {
                    float __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
                    if (__v > max_value)
                    {
                        max_value = __v;
                        max_index = __k;
                    }
                }

                __ret.__data_[__i * __inner_size + __j] = max_index;
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

#if defined(__AVX2__)
    if constexpr (std::is_same_v<_Tp, float>)
    {
        for (__i = 0; __i < __outer_size; __i++)
        {
            for (index_type __j = 0; __j < __inner_size; __j++)
            {
                __m256     max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
                index_type __k     = 0;
                for (; __k + 8 <= this->__shape_[__dim]; __k += 8)
                {
                    __m256 data_vec =
                        _mm256_loadu_ps(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
                    max_vec = _mm256_max_ps(max_vec, data_vec);
                }
                float max_value = _mm256_reduce_max_ps(max_vec);
                for (; __k < this->__shape_[__dim]; __k++)
                {
                    float __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
                    max_value = std::max(max_value, __v);
                }
                __ret.__data_[__i * __inner_size + __j] = max_value;
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
                float32x4_t max_vec = vdupq_n_f32(-std::numeric_limits<float>::infinity());
                index_type  __k     = 0;
                for (; __k + 4 <= this->__shape_[__dim]; __k += 4)
                {
                    float32x4_t data_vec =
                        vld1q_f32(&this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j]);
                    max_vec = vmaxq_f32(max_vec, data_vec);
                }
                float max_value = vmaxvq_f32(max_vec);
                for (; __k < this->__shape_[__dim]; __k++)
                {
                    float __v = this->__data_[(__i * this->__shape_[__dim] + __k) * __inner_size + __j];
                    max_value = std::max(max_value, __v);
                }
                __ret.__data_[__i * __inner_size + __j] = max_value;
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
                    {
                        __max_value = __v;
                    }
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
    index_type __s = this->__computeSize(__sh);
    if (__s != this->__data_.size())
    {
        throw std::invalid_argument("Total elements do not match for new shape");
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

    data_t __c;
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
        data_t __flat = this->__data_;
        data_t __ret(__flat.size());
        __ret[0] = __flat[0];

#if defined(__AVX2__)
        if constexpr (std::is_same_v<_Tp, float>)
        {
            size_t __i = 1;
            for (; __i + 8 <= __flat.size(); __i += 8)
            {
                __m256 prev   = _mm256_loadu_ps(&__ret[__i - 1]);
                __m256 curr   = _mm256_loadu_ps(&__flat[__i]);
                __m256 result = _mm256_mul_ps(prev, curr);
                _mm256_storeu_ps(&__ret[__i], result);
            }
            for (; __i < __flat.size(); __i++)
            {
                __ret[__i] = __ret[__i - 1] * __flat[__i];
            }
        }
        else
        {
            size_t __i = 1;
            for (; __i < __flat.size(); __i++)
            {
                __ret[__i] = __ret[__i - 1] * __flat[__i];
            }
        }
#else
        size_t __i = 1;
        for (; __i < __flat.size(); __i++)
        {
            __ret[__i] = __ret[__i - 1] * __flat[__i];
        }
#endif

        return __self(__ret, {__flat.size()});
    }
    else
    {
        if (__dim < 0 || __dim >= static_cast<index_type>(this->__shape_.size()))
        {
            throw std::invalid_argument("Invalid dimension provided.");
        }

        data_t __ret(this->__data_);
        // TODO : compute_outer_size() implementation
        size_t __outer_size = this->__compute_outer_size(__dim);
        size_t __inner_size = this->__shape_[__dim];

        size_t __st = this->__strides_[__dim];

#if defined(__AVX2__)
        if constexpr (std::is_same_v<_Tp, float>)
        {
            for (size_t __i = 0; __i < __outer_size; ++__i)
            {
                size_t __base = __i * __st;
                __ret[__base] = __data_[__base];

                size_t __j = 1;
                for (; __j + 8 <= __inner_size; __j += 8)
                {
                    __m256 prev   = _mm256_loadu_ps(&__ret[__base + __j - 1]);
                    __m256 curr   = _mm256_loadu_ps(&__data_[__base + __j]);
                    __m256 result = _mm256_mul_ps(prev, curr);
                    _mm256_storeu_ps(&__ret[__base + __j], result);
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
#endif

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

#if defined(__CUDACC__)
    if (this->__data_.size() >= 1024)
    {
        _Tp* d_input;
        _Tp* d_output;
        cudaMalloc(&d_input, this->__data_.size() * sizeof(_Tp));
        cudaMalloc(&d_output, __ret.__data_.size() * sizeof(_Tp));

        cudaMemcpy(d_input, this->__data_.data(), this->__data_.size() * sizeof(_Tp), cudaMemcpyHostToDevice);

        dim3 block(256);
        dim3 grid(((__slice_size + block.x - 1) / block.x));
        slice_kernel<<<grid, block>>>(d_input, d_output, __start_i, __end_i, __step, __slice_size);

        cudaMemcpy(__ret.__data_.data(), d_output, __ret.__data_.size() * sizeof(_Tp), cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);
    }
    else
    {
#endif

#if defined(__ARM_NEON)
        if constexpr (std::is_same_v<_Tp, float> && __step == 1)
        {
            const int  vector_size = 4;
            index_type vector_end  = __start_i + ((__end_i - __start_i) / vector_size) * vector_size;

            for (index_type i = __start_i, j = 0; i < vector_end; i += vector_size, j += vector_size)
            {
                float32x4_t vec = vld1q_f32(&(this->__data_[i]));
                vst1q_f32(&(__ret.__data_[j]), vec);
            }

            for (index_type i = vector_end, j = vector_end - __start_i; i < __end_i; ++i, ++j)
            {
                __ret.__data_[j] = this->__data_[i];
            }
        }
        else
        {
#endif
            index_type __i = __start_i, __j = 0ULL;
            for (; __i < __end_i; __i += __step, __j++)
            {
                __ret({__j}) = this->at({__i});
            }
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
    this->__check_is_scalar_type("Cannot divide non scalar values");
    if (this->__shape_ != __other.shape() || this->__data_.size() != __other.size(0))
    {
        throw std::invalid_argument("Cannot divide two tensors of different shapes : fmax");
    }

    data_t __ret(this->__data_.size());

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

    data_t __ret(this->__data_.size());

    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = static_cast<value_type>(std::fmax(double(this->__data_[__i]), double(__other[__i])));
    }

    std::transform(this->__data_.begin(), this->__data_.end(), __other.begin(), __ret.begin(),
                   [](const_reference __v, const_reference __w) {
                       return static_cast<value_type>(std::fmax(static_cast<double>(__v), static_cast<double>(__w)));
                   });

    return __self(__ret, this->__shape_);
}


template<class _Tp>
size_t tensor<_Tp>::count_nonzero(index_type __dim) const {
    this->__check_is_scalar_type("Cannot compare a non scalar value to zero");

    size_t __c = 0;

    if (__dim == -1)
    {
        for (const_reference __el : this->__data_)
        {
            if (__el != 0)
            {
                __c++;
            }
        }
    }
    else
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

    shape_type __ret_sh = {this->__shape_[0], __other.shape()[1]};
    data_t     __ret_d(__ret_sh[0] * __ret_sh[1], 0);

    const int blockSize = 64;

    for (int i = 0; i < __ret_sh[0]; i += blockSize)
    {
        for (int j = 0; j < __ret_sh[1]; j += blockSize)
        {
            for (int k = 0; k < this->__shape_[1]; k += blockSize)
            {
                for (int ii = i; ii < std::min(static_cast<index_type>(i + blockSize), __ret_sh[0]); ++ii)
                {
                    for (int jj = j; jj < std::min(static_cast<index_type>(j + blockSize), __ret_sh[1]); ++jj)
                    {
                        value_type __sum = 0;
                        for (int kk = k; kk < std::min(static_cast<index_type>(k + blockSize), this->__shape_[1]); ++kk)
                        {
                            __sum += this->at({ii, kk}) * __other.at({kk, jj});
                        }
                        __ret_d[ii * __ret_sh[1] + jj] += __sum;
                    }
                }
            }
        }
    }

#ifdef __ARM_NEON
    const int neonBlockSize = 4;

    for (int i = 0; i < __ret_sh[0]; i += neonBlockSize)
    {
        for (int j = 0; j < __ret_sh[1]; j += neonBlockSize)
        {
            for (int k = 0; k < this->__shape_[1]; k += neonBlockSize)
            {
                for (int ii = i; ii < std::min(static_cast<index_type>(i + neonBlockSize), __ret_sh[0]); ++ii)
                {
                    for (int jj = j; jj < std::min(static_cast<index_type>(j + neonBlockSize), __ret_sh[1]); ++jj)
                    {
                        float32x4_t sum_vec = vdupq_n_f32(0);

                        for (int kk = k; kk < std::min(static_cast<index_type>(k + neonBlockSize), this->__shape_[1]);
                             kk += 4)
                        {
                            float32x4_t a_vec = vld1q_f32(&this->__data_[ii * this->__shape_[1] + kk]);
                            float32x4_t b_vec = vld1q_f32(&__other.__data_[kk * __other.shape()[1] + jj]);
                            sum_vec           = vmlaq_f32(sum_vec, a_vec, b_vec);
                        }

                        float32x2_t sum_low  = vget_low_f32(sum_vec);
                        float32x2_t sum_high = vget_high_f32(sum_vec);
                        sum_low              = vadd_f32(sum_low, sum_high);
                        float32x2_t sum_dup  = vpadd_f32(sum_low, sum_low);

                        __ret_d[ii * __ret_sh[1] + jj] += vget_lane_f32(sum_dup, 0);
                    }
                }
            }
        }
    }
#endif

#ifdef __CUDACC__
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (__ret_sh[0] * __ret_sh[1] + threadsPerBlock - 1) / threadsPerBlock;

    _Tp *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, this->__data_.size() * sizeof(_Tp));
    cudaMalloc(&d_b, __other.__data_.size() * sizeof(_Tp));
    cudaMalloc(&d_c, __ret_d.size() * sizeof(_Tp));

    cudaMemcpy(d_a, this->__data_.data(), this->__data_.size() * sizeof(_Tp), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, __other.__data_.data(), __other.__data_.size() * sizeof(_Tp), cudaMemcpyHostToDevice);

    matmul_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, this->__shape_[0], this->__shape_[1],
                                                      __other.shape()[1]);

    cudaMemcpy(__ret_d.data(), d_c, __ret_d.size() * sizeof(_Tp), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
#endif

    return __self(__ret_d, __ret_sh);
}


#ifdef __CUDACC__
template<class _Tp>
__global__ void matmul_kernel(_Tp* a, _Tp* b, _Tp* c, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k)
    {
        _Tp sum = 0;
        for (int i = 0; i < n; ++i)
        {
            sum += a[row * n + i] * b[i * k + col];
        }

        c[row * k + col] = sum;
    }
}
#endif


template<class _Tp>
tensor<_Tp> tensor<_Tp>::reshape(const shape_type __sh) const {
    data_t __d = this->__data_;
    size_t __s = this->__computeSize(__sh);

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
__global__ void cross_product_kernel(_Tp* a, _Tp* b, _Tp* c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}
#endif


template<class _Tp>
tensor<_Tp> tensor<_Tp>::absolute(const tensor& __tensor) const {
    this->__check_is_scalar_type("Cannot call absolute on non scalar value");

    data_t __a;

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

    if (this->__shape_.size() == 1 && __other.shape().size() == 1)
    {
        if (this->__shape_[0] != __other.shape()[0])
        {
            throw std::invalid_argument("Vectors must have the same size for dot product");
        }

        const auto* __this_data  = this->__data_.data();
        const auto* __other_data = __other.storage().data();
        const auto  __size       = this->__data_.size();

        value_type __ret = 0;

#ifdef __CUDACC__
        if (this->__is_cuda_tensor && __other.__is_cuda_tensor)
        {
            thrust::device_vector<value_type> d_result(1);
            thrust::transform(thrust::device, __this_data, __this_data + __size, __other_data, d_result.begin(),
                              thrust::multiplies<value_type>());
            __ret = thrust::reduce(d_result.begin(), d_result.end());
        }
        else
#endif
        {
#if defined(__AVX__)
            if constexpr (std::is_same_v<value_type, float> && __size >= 8)
            {
                __m256 sum = _mm256_setzero_ps();
                for (size_t i = 0; i < __size - 7; i += 8)
                {
                    __m256 a = _mm256_loadu_ps(__this_data + i);
                    __m256 b = _mm256_loadu_ps(__other_data + i);
                    sum      = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
                }
                __ret = _mm256_reduce_add_ps(sum);
                for (size_t i = __size - (__size % 8); i < __size; ++i)
                {
                    __ret += __this_data[i] * __other_data[i];
                }
            }
            else
#elif defined(__SSE__)
            if constexpr (std::is_same_v<value_type, float> && __size >= 4)
            {
                __m128 sum = _mm_setzero_ps();
                for (size_t i = 0; i < __size - 3; i += 4)
                {
                    __m128 a = _mm_loadu_ps(__this_data + i);
                    __m128 b = _mm_loadu_ps(__other_data + i);
                    sum      = _mm_add_ps(sum, _mm_mul_ps(a, b));
                }
                __ret = _mm_reduce_add_ps(sum);
                for (size_t i = __size - (__size % 4); i < __size; ++i)
                {
                    __ret += __this_data[i] * __other_data[i];
                }
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
    {
        return this->matmul(__other);
    }

    if (this->__shape_.size() == 3 && __other.shape().size() == 3)
    {
        return this->cross_product(__other);
    }

    return __self();
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::relu() const {
    this->__check_is_scalar_type("Cannot relu non-scalar type");
    size_t __s = this->__data_.size();
    data_t __d(__s);

#ifdef __CUDACC__
    if (this->__is_cuda_tensor)
    {
        thrust::device_vector<value_type> d_input(this->__data_);
        thrust::device_vector<value_type> d_output(__s);
        thrust::transform(d_input.begin(), d_input.end(), d_output.begin(),
                          [] __device__(value_type x) { return max(x, value_type(0)); });
        thrust::copy(d_output.begin(), d_output.end(), __d.begin());
        return __self(__d, this->__shape_);
    }
#endif

#if defined(__AVX__)
    if constexpr (std::is_same_v<value_type, float>)
    {
        size_t __i  = 0;
        __m256 zero = _mm256_setzero_ps();
        for (; __i + 8 <= __s; __i += 8)
        {
            __m256 x      = _mm256_loadu_ps(&this->__data_[__i]);
            __m256 result = _mm256_max_ps(x, zero);
            _mm256_storeu_ps(&__d[__i], result);
        }
        for (; __i < __s; __i++)
        {
            __d[__i] = std::max(this->__data_[__i], value_type(0));
        }
    }
    else
#elif defined(__SSE__)
    if constexpr (std::is_same_v<value_type, float>)
    {
        size_t __i  = 0;
        __m128 zero = _mm_setzero_ps();
        for (; __i + 4 <= __s; __i += 4)
        {
            __m128 x      = _mm_loadu_ps(&this->__data_[__i]);
            __m128 result = _mm_max_ps(x, zero);
            _mm_storeu_ps(&__d[__i], result);
        }
        for (; __i < __s; __i++)
        {
            __d[__i] = std::max(this->__data_[__i], value_type(0));
        }
    }
    else
#endif
    {
        for (size_t __i = 0; __i < __s; __i++)
        {
            __d[__i] = std::max(this->__data_[__i], value_type(0));
        }
    }

    return __self(__d, this->__shape_);
}


template<class _Tp>
void tensor<_Tp>::relu_() {
    this->__check_is_scalar_type("Cannot relu non-scalar type");

    size_t __s = this->__data_.size();
    size_t __i = 0;

#ifdef __CUDACC__
    if (this->__is_cuda_tensor)
    {
        value_type* d_data = thrust::raw_pointer_cast(this->__data_.data());
        thrust::transform(thrust::device, d_data, d_data + __s, d_data,
                          [] __device__(value_type x) { return max(x, value_type(0)); });
        return;
    }
#endif

#ifdef __ARM_NEON
    if constexpr (std::is_same_v<value_type, float>)
    {
        const float32x4_t vZero = vdupq_n_f32(0.0f);
        for (; __i + 4 <= __s; __i += 4)
        {
            float32x4_t v = vld1q_f32(&this->__data_[__i]);
            v             = vmaxq_f32(v, vZero);
            vst1q_f32(&this->__data_[__i], v);
        }
    }
    else if constexpr (std::is_same_v<value_type, int32_t>)
    {
        const int32x4_t vZero = vdupq_n_s32(0);
        for (; __i + 4 <= __s; __i += 4)
        {
            int32x4_t v = vld1q_s32(&this->__data_[__i]);
            v           = vmaxq_s32(v, vZero);
            vst1q_s32(&this->__data_[__i], v);
        }
    }


#endif

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

    const index_type rows = this->__shape_[0];
    const index_type cols = this->__shape_[1];

#ifdef __CUDACC__
    if (this->__is_cuda_tensor)
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

        transpose_kernel<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(this->__data_.data()),
                                                thrust::raw_pointer_cast(__ret.__data_.data()), rows, cols);
        cudaDeviceSynchronize();
        return __ret;
    }
#endif

#ifdef __ARM_NEON
    if constexpr (std::is_same_v<_Tp, float>)
    {
        for (index_type i = 0; i < rows; i += 4)
        {
            for (index_type j = 0; j < cols; j += 4)
            {
                if (i + 4 <= rows && j + 4 <= cols)
                {
                    float32x4x4_t input;
                    for (int k = 0; k < 4; ++k)
                    {
                        input.val[k] = vld1q_f32(&this->__data_[(i + k) * cols + j]);
                    }

                    float32x4x4_t output = vld4q_f32(reinterpret_cast<const float*>(&input));

                    for (int k = 0; k < 4; ++k)
                    {
                        vst1q_f32(&__ret.__data_[(j + k) * rows + i], output.val[k]);
                    }
                }
                else
                {
                    for (index_type ii = i; ii < std::min(i + 4, rows); ++ii)
                    {
                        for (index_type jj = j; jj < std::min(j + 4, cols); ++jj)
                        {
                            __ret.at({jj, ii}) = this->at({ii, jj});
                        }
                    }
                }
            }
        }
    }
    else
#endif
    {
        index_type __i = 0;
        for (; __i < rows; __i++)
        {
            index_type __j = 0;
            for (; __j < cols; __j++)
            {
                __ret.at({__j, __i}) = this->at({__i, __j});
            }
        }
    }
    return __ret;
}

#ifdef __CUDACC__
template<class _Tp>
__global__ void transpose_kernel(_Tp* input, _Tp* output, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols)
    {
        output[j * rows + i] = input[i * cols + j];
    }
}
#endif


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

    data_t __ret(this->__data_.size());

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

    data_t __ret(this->__data_.size());

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

    data_t __ret(this->__data_.size());

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
#ifdef __ARM_NEON
    
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        this->__data_[__i] |= __other[__i];
    }
#endif
}


template<class _Tp>
void tensor<_Tp>::bitwise_xor_(const tensor& __other) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    {
        throw std::runtime_error("Cannot perform a bitwise XOR on non-integral or non-boolean values");
    }

    assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));
#ifdef __ARM_NEON
    const size_t simd_size = 4;
    const size_t simd_end  = this->__data_.size() - (this->__data_.size() % simd_size);

    for (size_t __i = 0; __i < simd_end; __i += simd_size)
    {
        uint32x4_t data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
        uint32x4_t other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));

        uint32x4_t xor_vec = veorq_u32(data_vec, other_vec);

        vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), xor_vec);
    }

    for (size_t __i = simd_end; __i < this->__data_.size(); __i++)
    {
        this->__data_[__i] = (this->__data_[__i] ^ __other[__i]);
    }
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        this->__data_[__i] ^= __other[__i];
    }
#endif
}


template<class _Tp>
tensor<bool> tensor<_Tp>::logical_not() const {
    return __self(*this).logical_not_();
}


template<class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const value_type __val) const {
    return __self(*this).logical_or_(__val);
}


template<class _Tp>
tensor<bool> tensor<_Tp>::logical_or(const tensor<_Tp>& __other) const {
    return __self(*this).logical_or_(__other);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const tensor& __other) const {
    return __self(*this).logical_xor_(__other);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_xor(const value_type __val) const {
    return __self(*this).logical_xor_(__val);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const tensor& __other) const {
    return __self(*this).logical_and_(__other);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::logical_and(const value_type __val) const {
    return __self(*this).logical_and_(__val);
}


template<class _Tp>
void tensor<_Tp>::logical_or_(const tensor<_Tp>& __other) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    {
        throw std::runtime_error("Cannot get the element wise not of non-integral and non-boolean value");
    }

    assert(this->__shape_ == __other.shape());
#ifdef __ARM_NEON
    const size_t simd_size = 4;
    const size_t simd_end  = this->__data_.size() - (this->__data_.size() % simd_size);

    for (size_t __i = 0; __i < simd_end; __i += simd_size)
    {
        uint32x4_t data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
        uint32x4_t other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__other[__i]));

        uint32x4_t or_vec = vornq_u32(data_vec, other_vec);

        vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), or_vec);
    }

    for (size_t __i = simd_end; __i < this->__data_.size(); __i++)
    {
        this->__data_[__i] = (this->__data_[__i] || __other[__i]);
    }
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        this->__data_[__i] = (this->__data_[__i] || __other[__i]);
    }
#endif
}


template<class _Tp>
void tensor<_Tp>::logical_xor_(const tensor<_Tp>& __other) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    {
        throw std::runtime_error("Cannot get the element wise xor of non-integral and non-boolean value");
    }

    assert(this->__shape_ == __other.shape());
#ifdef __ARM_NEON
    const size_t simd_size = 4;
    const size_t simd_end  = this->__data_.size() - (this->__data_.size() % simd_size);

    for (size_t __i = 0; __i < simd_end; __i += simd_size)
    {
        uint32x4_t data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
        uint32x4_t other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));

        uint32x4_t xor_vec = veorq_u32(data_vec, other_vec);

        vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), xor_vec);
    }

    for (size_t __i = simd_end; __i < this->__data_.size(); __i++)
    {
        this->__data_[__i] = (this->__data_[__i] && __other[__i]);
    }
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        this->__data_[__i] = (this->__data_[__i] ^ __other[__i]);
    }
#endif
}


template<class _Tp>
void tensor<_Tp>::logical_and_(const tensor<_Tp>& __other) {
    if (!std::is_integral<value_type>::value && !std::is_same<value_type, bool>::value)
    {
        throw std::runtime_error("Cannot get the element-wise and of non-integral and non-boolean value");
    }

    assert(this->__shape_ == __other.shape());

#ifdef __ARM_NEON
    const size_t simd_size = 4;
    const size_t simd_end  = this->__data_.size() - (this->__data_.size() % simd_size);

    for (size_t __i = 0; __i < simd_end; __i += simd_size)
    {
        uint32x4_t data_vec  = vld1q_u32(reinterpret_cast<const uint32_t*>(&this->__data_[__i]));
        uint32x4_t other_vec = vld1q_u32(reinterpret_cast<const uint32_t*>(&__other[__i]));

        uint32x4_t and_vec = vandq_u32(data_vec, other_vec);

        vst1q_u32(reinterpret_cast<uint32_t*>(&this->__data_[__i]), and_vec);
    }

    for (size_t __i = simd_end; __i < this->__data_.size(); __i++)
    {
        this->__data_[__i] = (this->__data_[__i] && __other[__i]);
    }

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        this->__data_[__i] = (this->__data_[__i] && __other[__i]);
    }
#endif
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

    data_t __ret(this->__data_.size());

#ifdef __ARM_NEON
    const size_t simd_size = 4;
    const size_t simd_end  = this->__data_.size() - (this->__data_.size() % simd_size);

    for (size_t __i = 0; __i < simd_end; __i += simd_size)
    {
        float32x4_t base_vec = vld1q_f32(reinterpret_cast<const float*>(&this->__data_[__i]));
        float32x4_t exp_vec  = vld1q_f32(reinterpret_cast<const float*>(&__other[__i]));

        float32x4_t result_vec = {std::pow(vgetq_lane_f32(base_vec, 0), vgetq_lane_f32(exp_vec, 0)),
                                  std::pow(vgetq_lane_f32(base_vec, 1), vgetq_lane_f32(exp_vec, 1)),
                                  std::pow(vgetq_lane_f32(base_vec, 2), vgetq_lane_f32(exp_vec, 2)),
                                  std::pow(vgetq_lane_f32(base_vec, 3), vgetq_lane_f32(exp_vec, 3))};

        vst1q_f32(&this->__data_[__i], result_vec);
    }

    for (size_t __i = simd_end; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = static_cast<value_type>(
            std::pow(static_cast<double>(this->__data_[__i]), static_cast<double>(__other[__i])));
    }

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = static_cast<value_type>(
            std::pow(static_cast<double>(this->__data_[__i]), static_cast<double>(__other[__i])));
    }
#endif
    return __self(__ret, this->__shape_);
}

template<class _Tp>
tensor<_Tp> tensor<_Tp>::pow(const value_type __val) const {
    this->__check_is_integral_type("cannot get the power of a non-integral value");

    data_t __ret(this->__data_.size());
#ifdef __ARM_NEON
    const size_t simd_size = 4;
    const size_t simd_end  = this->__data_.size() - (this->__data_.size() % simd_size);

    float32x4_t val_vec = vdupq_n_f32(static_cast<float>(__val));

    for (size_t __i = 0; __i < simd_end; __i += simd_size)
    {
        float32x4_t data_vec = vld1q_f32(reinterpret_cast<const float*>(&this->__data_[__i]));

        float32x4_t result_vec = {std::pow(vgetq_lane_f32(data_vec, 0), vgetq_lane_f32(val_vec, 0)),
                                  std::pow(vgetq_lane_f32(data_vec, 1), vgetq_lane_f32(val_vec, 1)),
                                  std::pow(vgetq_lane_f32(data_vec, 2), vgetq_lane_f32(val_vec, 2)),
                                  std::pow(vgetq_lane_f32(data_vec, 3), vgetq_lane_f32(val_vec, 3))};

        vst1q_f32(reinterpret_cast<float*>(&__ret[__i]), result_vec);
    }

    for (size_t __i = simd_end; __i < this->__data_.size(); __i++)
    {
        __ret[__i] =
            static_cast<value_type>(std::pow(static_cast<double>(this->__data_[__i]), static_cast<double>(__val)));
    }
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        __ret[__i] =
            static_cast<value_type>(std::pow(static_cast<double>(this->__data_[__i]), static_cast<double>(__val)));
    }
#endif
    return __self(__ret, this->__shape_);
}

template<class _Tp>
void tensor<_Tp>::pow_(const tensor& __other) const {
    this->__check_is_integral_type("cannot get the power of a non integral value");

    assert(this->__shape_ == __other.shape() && this->__data_.size() == __other.size(0));
#ifdef __ARM_NEON
    const size_t simd_size = 4;
    const size_t simd_end  = this->__data_.size() - (this->__data_.size() % simd_size);

    for (size_t __i = 0; __i < simd_end; __i += simd_size)
    {
        float32x4_t base_vec = vld1q_f32(reinterpret_cast<const float*>(&this->__data_[__i]));
        float32x4_t exp_vec  = vld1q_f32(reinterpret_cast<const float*>(&__other[__i]));

        float32x4_t result_vec = {std::pow(vgetq_lane_f32(base_vec, 0), vgetq_lane_f32(exp_vec, 0)),
                                  std::pow(vgetq_lane_f32(base_vec, 1), vgetq_lane_f32(exp_vec, 1)),
                                  std::pow(vgetq_lane_f32(base_vec, 2), vgetq_lane_f32(exp_vec, 2)),
                                  std::pow(vgetq_lane_f32(base_vec, 3), vgetq_lane_f32(exp_vec, 3))};

        vst1q_f32(&this->__data_[__i], result_vec);
    }

    for (size_t __i = simd_end; __i < this->__data_.size(); __i++)
    {
        this->__data_[__i] = static_cast<value_type>(
            std::pow(static_cast<double>(this->__data_[__i]), static_cast<double>(__other[__i])));
    }
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        this->__data_[__i] = static_cast<value_type>(std::pow(double(this->__data_[__i]), double(__other[__i])));
    }
#endif
}


template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const tensor& __other) const {
    if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
    {
        throw std::runtime_error("Cannot compare non-integral or scalar value");
    }

    assert(this->__shape_ == __other.shape() && this->size(0) == __other.size(0));

    std::vector<bool> __ret(this->__data_.size());
#ifdef __ARM_NEON
    if constexpr (std::is_same_v<_Tp, float>)
    {
        const size_t vec_size    = 4;
        const size_t num_vectors = this->__data_.size() / vec_size;
        const size_t remaining   = this->__data_.size() % vec_size;

        size_t __i = 0;
        for (; __i < num_vectors * vec_size; __i += vec_size)
        {
            float32x4_t data_vec1  = vld1q_f32(&this->__data_[__i]);
            float32x4_t data_vec2  = vld1q_f32(&__other.__data_[__i]);
            uint32x4_t  cmp_result = vcleq_f32(data_vec1, data_vec2);
            uint32_t    mask       = vaddvq_u32(cmp_result);

            __ret[__i]     = mask & 1;
            __ret[__i + 1] = (mask >> 8) & 1;
            __ret[__i + 2] = (mask >> 16) & 1;
            __ret[__i + 3] = (mask >> 24) & 1;
        }

        for (; __i < this->__data_.size(); __i++)
        {
            __ret[__i] = (this->__data_[__i] <= __other.__data_[__i]);
        }
    }
    else
    {
        size_t __i = 0;
        for (; __i < this->__data_.size(); __i++)
        {
            __ret[__i] = (this->__data_[__i] <= __other.__data_[__i]);
        }
    }
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = (this->__data_[__i] <= __other[__i]) ? true : false;
    }
#endif
    return tensor<bool>(__ret, this->__shape_);
}


template<class _Tp>
tensor<bool> tensor<_Tp>::less_equal(const value_type __val) const {
    if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
    {
        throw std::runtime_error("Cannot compare non-integral or scalar value");
    }

    std::vector<bool> __ret(this->__data_.size());

#ifdef __ARM_NEON
    if constexpr (std::is_same_v<_Tp, float>)
    {
        const size_t vec_size    = 4;
        const size_t num_vectors = this->__data_.size() / vec_size;
        const size_t remaining   = this->__data_.size() % vec_size;

        size_t __i = 0;
        for (; __i < num_vectors * vec_size; __i += vec_size)
        {
            float32x4_t data_vec   = vld1q_f32(&this->__data_[__i]);
            float32x4_t val_vec    = vdupq_n_f32(__val);
            uint32x4_t  cmp_result = vcleq_f32(data_vec, val_vec);
            uint32_t    mask       = vaddvq_u32(cmp_result);

            __ret[__i]     = mask & 1;
            __ret[__i + 1] = (mask >> 8) & 1;
            __ret[__i + 2] = (mask >> 16) & 1;
            __ret[__i + 3] = (mask >> 24) & 1;
        }

        for (; __i < this->__data_.size(); __i++)
        {
            __ret[__i] = (this->__data_[__i] <= __val);
        }
    }
    else
    {
        size_t __i = 0;
        for (; __i < this->__data_.size(); __i++)
        {
            __ret[__i] = (this->__data_[__i] <= __val);
        }
    }
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = (this->__data_[__i] <= __val) ? true : false;
    }
#endif
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
#ifdef __ARM_NEON
    if constexpr (std::is_same_v<_Tp, float>)
    {
        const size_t vec_size    = 4;
        const size_t num_vectors = this->__data_.size() / vec_size;
        const size_t remaining   = this->__data_.size() % vec_size;

        size_t __i = 0;
        for (; __i < num_vectors * vec_size; __i += vec_size)
        {
            float32x4_t data_vec1  = vld1q_f32(&this->__data_[__i]);
            float32x4_t data_vec2  = vld1q_f32(&__other.__data_[__i]);
            uint32x4_t  cmp_result = vcgeq_f32(data_vec1, data_vec2);
            uint32_t    mask       = vaddvq_u32(cmp_result);

            __ret[__i]     = mask & 1;
            __ret[__i + 1] = (mask >> 8) & 1;
            __ret[__i + 2] = (mask >> 16) & 1;
            __ret[__i + 3] = (mask >> 24) & 1;
        }

        for (; __i < this->__data_.size(); __i++)
        {
            __ret[__i] = (this->__data_[__i] >= __other.__data_[__i]);
        }
    }
    else
    {
        size_t __i = 0;
        for (; __i < this->__data_.size(); __i++)
        {
            __ret[__i] = (this->__data_[__i] >= __other.__data_[__i]);
        }
    }

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = (this->__data_[__i] >= __other[__i]) ? true : false;
    }
#endif
    return tensor<bool>(__ret, this->__shape_);
}


template<class _Tp>
tensor<bool> tensor<_Tp>::greater_equal(const value_type __val) const {
    if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
    {
        throw std::runtime_error("Cannot compare non-integral or scalar value");
    }

    std::vector<bool> __ret(this->__data_.size());
#ifdef __ARM_NEON
    if constexpr (std::is_same_v<_Tp, float>)
    {
        const size_t vec_size    = 4;
        const size_t num_vectors = this->__data_.size() / vec_size;
        const size_t remaining   = this->__data_.size() % vec_size;

        size_t      __i     = 0;
        float32x4_t val_vec = vdupq_n_f32(__val);
        for (; __i < num_vectors * vec_size; __i += vec_size)
        {
            float32x4_t data_vec   = vld1q_f32(&this->__data_[__i]);
            uint32x4_t  cmp_result = vcgeq_f32(data_vec, val_vec);
            uint32_t    mask       = vaddvq_u32(cmp_result);

            __ret[__i]     = mask & 1;
            __ret[__i + 1] = (mask >> 8) & 1;
            __ret[__i + 2] = (mask >> 16) & 1;
            __ret[__i + 3] = (mask >> 24) & 1;
        }

        for (; __i < this->__data_.size(); __i++)
        {
            __ret[__i] = (this->__data_[__i] >= __val);
        }
    }
    else
    {
        size_t __i = 0;
        for (; __i < this->__data_.size(); __i++)
        {
            __ret[__i] = (this->__data_[__i] >= __val);
        }
    }

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = (this->__data_[__i] >= __val) ? true : false;
    }
#endif
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
#ifdef __ARM_NEON
    if constexpr (std::is_same_v<_Tp, float>)
    {
        const size_t vec_size    = 4;
        const size_t num_vectors = this->__data_.size() / vec_size;
        const size_t remaining   = this->__data_.size() % vec_size;

        size_t __i = 0;
        for (; __i < num_vectors * vec_size; __i += vec_size)
        {
            float32x4_t data_vec1  = vld1q_f32(&this->__data_[__i]);
            float32x4_t data_vec2  = vld1q_f32(&__other.__data_[__i]);
            uint32x4_t  cmp_result = vceqq_f32(data_vec1, data_vec2);
            uint32_t    mask       = vaddvq_u32(cmp_result);

            __ret[__i]     = mask & 1;
            __ret[__i + 1] = (mask >> 8) & 1;
            __ret[__i + 2] = (mask >> 16) & 1;
            __ret[__i + 3] = (mask >> 24) & 1;
        }

        for (; __i < this->__data_.size(); __i++)
        {
            __ret[__i] = (this->__data_[__i] == __other.__data_[__i]);
        }
    }
    else
    {
        size_t __i = 0;
        for (; __i < this->__data_.size(); __i++)
        {
            __ret[__i] = (this->__data_[__i] == __other.__data_[__i]);
        }
    }

#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = (this->__data_[__i] == __other[__i]) ? true : false;
    }

#endif
    return tensor<bool>(__ret, this->__shape_);
}


template<class _Tp>
tensor<bool> tensor<_Tp>::equal(const value_type __val) const {
    if (!std::is_integral<value_type>::value && !std::is_scalar<value_type>::value)
    {
        throw std::runtime_error("Cannot compare non-integral or scalar value");
    }
    std::vector<bool> __ret(this->__data_.size());
#ifdef __ARM_NEON
    if constexpr (std::is_same_v<_Tp, float>)
    {
        const size_t vec_size    = 4;
        const size_t num_vectors = this->__data_.size() / vec_size;
        const size_t remaining   = this->__data_.size() % vec_size;

        float32x4_t val_vec = vdupq_n_f32(__val);

        size_t __i = 0;
        for (; __i < num_vectors * vec_size; __i += vec_size)
        {
            float32x4_t data_vec   = vld1q_f32(&this->__data_[__i]);
            uint32x4_t  cmp_result = vceqq_f32(data_vec, val_vec);
            uint32_t    mask       = vaddvq_u32(cmp_result);

            __ret[__i]     = mask & 1;
            __ret[__i + 1] = (mask >> 8) & 1;
            __ret[__i + 2] = (mask >> 16) & 1;
            __ret[__i + 3] = (mask >> 24) & 1;
        }

        for (; __i < this->__data_.size(); __i++)
        {
            __ret[__i] = (this->__data_[__i] == __val);
        }
    }
    else
    {
        size_t __i = 0;
        for (; __i < this->__data_.size(); __i++)
        {
            __ret[__i] = (this->__data_[__i] == __val);
        }
    }
#else
    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = (this->__data_[__i] >= __val) ? true : false;
    }
#endif
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

    index_type __ret_size = std::accumulate(__ret_sh.begin(), __ret_sh.end(), 1, std::multiplies<index_type>());
    data_t     __ret_data(__ret_size, value_type(0.0f));

#if defined(__ARM_NEON)
    if constexpr (std::is_same_v<_Tp, float>)
    {
        const index_type __axis_size  = this->__shape_[__axis];
        const index_type __outer_size = this->__compute_outer_size(__axis);
        const index_type __inner_size = this->size(0) / (__outer_size * __axis_size);

        for (index_type __outer = 0; __outer < __outer_size; ++__outer)
        {
            for (index_type __inner = 0; __inner < __inner_size; ++__inner)
            {
                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                index_type  __i     = __outer * __axis_size * __inner_size + __inner;
                index_type  __j     = 0;

                for (; __j + 4 <= __axis_size; __j += 4)
                {
                    float32x4_t data_vec = vld1q_f32(&this->__data_[__i]);
                    sum_vec              = vaddq_f32(sum_vec, data_vec);
                    __i += __inner_size * 4;
                }

                float sum = vaddvq_f32(sum_vec);

                for (; __j < __axis_size; ++__j)
                {
                    sum += this->__data_[__i];
                    __i += __inner_size;
                }

                __ret_data[__outer * __inner_size + __inner] = sum;
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
    {
        throw std::runtime_error("Cannot get a row from a non two dimensional tensor");
    }

    if (this->__shape_[0] <= __index || __index < 0LL)
    {
        throw std::invalid_argument("Index input is out of range");
    }

    data_t __r(this->__data_.begin() + (this->__shape_[1] * __index),
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

    data_t __c;

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
    data_t __ret(this->__data_.size());

#ifdef __ARM_NEON
    const size_t simd_size = 4;
    const size_t simd_end  = this->__data_.size() - (this->__data_.size() % simd_size);

    for (size_t __i = 0; __i < simd_end; __i += simd_size)
    {
        float32x4_t data   = vld1q_f32(&this->__data_[__i]);
        float32x4_t ceiled = vrndpq_f32(data);
        vst1q_f32(&__ret[__i], ceiled);
    }

    for (size_t __i = simd_end; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = std::ceil(this->__data_[__i]);
    }
#else

    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = std::ceil(this->__data_[__i]);
    }

#endif
    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::floor() const {
    this->__check_is_scalar_type("Cannot get the floor of a non scalar value");
    data_t __ret(this->__data_.size());

#ifdef __ARM_NEON
    const size_t simd_size = 4;
    const size_t simd_end  = this->__data_.size() - (this->__data_.size() % simd_size);

    float32x4_t zero = vdupq_n_f32(0.0f);

    for (size_t __i = 0; __i < simd_end; __i += simd_size)
    {
        float32x4_t data    = vld1q_f32(&this->__data_[__i]);
        float32x4_t floored = vrndmq_f32(data);
        vst1q_f32(&__ret[__i], floored);
    }

    for (size_t __i = simd_end; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = std::floor(this->__data_[__i]);
    }

#else

    size_t __i = 0;
    for (; __i < this->__data_.size(); __i++)
    {
        __ret[__i] = std::floor(this->__data_[__i]);
    }
#endif
    return __self(__ret, this->__shape_);
}


template<class _Tp>
tensor<_Tp> tensor<_Tp>::clamp(const_pointer __min_val, const_pointer __max_val) const {
    tensor<value_type> __t = __self(*this);
    __t.clamp_(__min_val, __max_val);
    return __t;
}


template<class _Tp>
void tensor<_Tp>::clamp_(const_pointer __min_val, const_pointer __max_val) {
#if defined(__AVX2__)
    const size_t simd_size = 8;
    const size_t simd_end  = this->__data_.size() - (this->__data_.size() % simd_size);

    __m256 min_vec = _mm256_set1_ps(__min_val ? *__min_val : std::numeric_limits<_Tp>::lowest());
    __m256 max_vec = _mm256_set1_ps(__max_val ? *__max_val : std::numeric_limits<_Tp>::max());

    for (size_t __i = 0; __i < simd_end; __i += simd_size)
    {
        __m256 data_vec = _mm256_loadu_ps(&this->__data_[__i]);
        __m256 clamped  = _mm256_min_ps(_mm256_max_ps(data_vec, min_vec), max_vec);
        _mm256_storeu_ps(&this->__data_[__i], clamped);
    }

    for (size_t __i = simd_end; __i < this->__data_.size(); __i++)
    {
        if (__min_val)
            this->__data_[__i] = std::max(*__min_val, this->__data_[__i]);
        if (__max_val)
            this->__data_[__i] = std::min(*__max_val, this->__data_[__i]);
    }
#elif defined(__ARM_NEON)
    const size_t simd_size = 4;
    const size_t simd_end  = this->__data_.size() - (this->__data_.size() % simd_size);

    float32x4_t min_vec = vdupq_n_f32(__min_val ? *__min_val : std::numeric_limits<_Tp>::lowest());
    float32x4_t max_vec = vdupq_n_f32(__max_val ? *__max_val : std::numeric_limits<_Tp>::max());

    for (size_t __i = 0; __i < simd_end; __i += simd_size)
    {
        float32x4_t data_vec = vld1q_f32(&this->__data_[__i]);
        float32x4_t clamped  = vminq_f32(vmaxq_f32(data_vec, min_vec), max_vec);
        vst1q_f32(&this->__data_[__i], clamped);
    }

    for (size_t __i = simd_end; __i < this->__data_.size(); __i++)
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
        {
            this->__data_[__i] = std::max(*__min_val, this->__data_[__i]);
        }

        if (__max_val)
        {
            this->__data_[__i] = std::min(*__max_val, this->__data_[__i]);
        }
    }
#endif
}
