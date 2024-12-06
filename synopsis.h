//
//  tensor synopsis
//

template<class _Tp>
class tensor
{
 public:
  using __self                 = tensor;
  using value_t                = _Tp;
  using data_t                 = std::vector<value_t>;
  using index_t                = int64_t;
  using shape_t                = std::vector<index_t>;
  using reference              = value_t&;
  using const_reference        = const value_t&;
  using pointer                = value_t*;
  using const_pointer          = const value_t*;
  using iterator               = std::__wrap_iter<pointer>;
  using const_iterator         = std::__wrap_iter<const_pointer>;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

 private:
  data_t               __data_;
  shape_t              __shape_;
  std::vector<index_t> __strides_;
  Device               __device_;

 public:
  tensor() = default;

  explicit tensor(const shape_t& __sh, const value_t& __v, Device __d = Device::CPU);
  explicit tensor(const shape_t& __sh, Device __d = Device::CPU);
  explicit tensor(const data_t& __d, const shape_t& __sh, Device __dev = Device::CPU);
  tensor(const tensor& __t);
  tensor(tensor&& __t) noexcept;
  tensor(const shape_t& __sh, std::initializer_list<value_t> init_list, Device __d = Device::CPU);
  tensor(const shape_t& __sh, const tensor& __other);

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

  bool          operator==(const tensor& __other) const;
  bool          operator!=(const tensor& __other) const;
  tensor        operator+(const tensor& __other) const;
  tensor        operator-(const tensor& __other) const;
  tensor        operator+(const value_type __val) const;
  tensor        operator-(const value_type __val) const;
  tensor&       operator-=(const tensor& __other) const;
  tensor&       operator+=(const tensor& __other) const;
  tensor&       operator*=(const tensor& __other) const;
  tensor&       operator/=(const tensor& __other) const;
  tensor&       operator+=(const_reference __val) const;
  tensor&       operator-=(const_reference __val) const;
  tensor&       operator/=(const_reference __val) const;
  tensor&       operator*=(const_reference __val) const;
  tensor&       operator=(const tensor& __other) const;
  tensor&       operator=(tensor&& __other) const noexcept;
  tensor<bool>& operator!() const;


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
  tensor expand_as(shape_type __sh, index_type __dim) const;

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
  tensor& asinh_() const;
  tensor& ceil_() const;
  tensor& floor_() const;
  tensor& relu_() const;
  tensor& clamp_(const_pointer __min_val = nullptr, const_pointer __max_val = nullptr) const;
  tensor& logical_not_() const { this->bitwise_not_(); }
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
  tensor& dist_(const tensor& __other) const;
  tensor& dist_(const value_type __val) const;
  tensor& maximum_(const tensor& __other) const;
  tensor& maximum_(const value_type __val) const;
  tensor& remainder_(const value_type __val) const;
  tensor& remainder_(const tensor& __other) const;
  tensor& fill_(const value_type __val) const;
  tensor& fill_(const tensor& __other) const;
  tensor& less_(const tensor& __other) const;
  tensor& less_(const value_type __val) const;
  tensor& greater_(const tensor& __other) const;
  tensor& greater_(const value_type __val) const;
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
  tensor& print() const;

  tensor<index_type> argmax_(index_type __dim) const;
  tensor<index_type> argsort(index_type __dim = -1, bool __ascending = true) const;

 private:
  static uint64_t  __computeSize(const shape_t& __dims);
  static float32_t __frac(const_reference __val);
  static void      __check_is_same_type(const std::string __msg);
  static void      __check_is_arithmetic_type(const std::string __msg);
  static void      __check_is_scalar_type(const std::string __msg);
  static void      __check_is_integral_type(const std::string __msg);
  template<typename __t>
  size_t   computeStride(size_t dim, const shape_t& shape) const;
  void     printRecursive(size_t index, size_t depth, const shape_t& shape) const;
  void     __compute_strides();
  index_t  __compute_index(const std::vector<index_t>& __idx) const;
  uint64_t __compute_outer_size(const index_t __dim) const;
  bool     __is_cuda_device() const { return (this->__device_ == Device::CUDA); }
};  // tensor class
