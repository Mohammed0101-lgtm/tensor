


template<class _Tp>
class tensor
{
   public:
    typedef tensor               __self;
    typedef _Tp                  value_t;
    typedef int64_t              index_t;
    typedef std::vector<index_t> shape_t;
    typedef value_t&             reference;
    typedef const value_t&       const_reference;
    typedef value_t*             pointer;
    typedef const value_t*       const_pointer;
    typedef std::vector<value_t> data_t;
    typedef const data_t         const_data_container;

    typedef typename data_t::iterator               iterator;
    typedef typename data_t::const_iterator         const_iterator;
    typedef typename data_t::reverse_iterator       reverse_iterator;
    typedef typename data_t::const_reverse_iterator const_reverse_iterator;

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

    bool __is_cuda_device;

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

    data_t storage() const noexcept { return this->__data_; }

    shape_t shape() const noexcept { return this->__shape_; }

    shape_t strides() const noexcept { return this->__strides_; }

    Device device() const noexcept { return this->__device_; }

    size_t n_dims() const noexcept { return this->__shape_.size(); }

    size_t size(const index_t __dim) const;

    size_t capacity() const noexcept { return this->__data_.capacity(); }

    reference at(shape_t __idx);

    const_reference at(const shape_t __idx) const;

    reference operator[](const size_t __in);

    const_reference operator[](const size_t __in) const;

    tensor operator+(const tensor& __other) const;

    tensor operator-(const tensor& __other) const;

    tensor operator-=(const tensor& __other) const;

    tensor operator+=(const tensor& __other) const;

    tensor operator*=(const tensor& __other) const;

    tensor operator/=(const tensor& __other) const;

    tensor operator+(const value_t _scalar) const;

    tensor operator-(const value_t __scalar) const;

    tensor operator+=(const_reference __scalar) const;

    tensor operator-=(const_reference __scalar) const;

    tensor operator/=(const_reference __scalar) const;

    tensor operator*=(const_reference __scalar) const;

    bool operator==(const tensor& __other) const;

    bool operatornot_eq(const tensor& __other) const { return !(*this == __other); }

    bool empty() const { return this->__data_.empty(); }

    double mean() const;

    tensor<bool> logical_not() const;

    tensor<bool> logical_or(const value_t __val) const;

    tensor<bool> logical_or(const tensor& __other) const;

    tensor<bool> less_equal(const tensor& __other) const;

    tensor<bool> less_equal(const value_t __val) const;

    tensor<bool> greater_equal(const tensor& __other) const;

    tensor<bool> greater_equal(const value_t __val) const;

    tensor<bool> equal(const tensor& __other) const;

    tensor<bool> equal(const value_t __val) const;

    size_t count_nonzero(index_t __dim = -1) const;

    tensor slice(index_t __dim, std::optional<index_t> __start, std::optional<index_t> __end, index_t __step) const;

    tensor fmax(const tensor& __other) const;

    tensor fmax(const value_t __val) const;

    void fmax_(const tensor& __other);

    void fmax_(const value_t __val);

    tensor fmod(const tensor& __other) const;

    tensor fmod(const value_t __val) const;

    tensor frac() const;

    tensor log() const;

    tensor log10() const;

    tensor log2() const;

    tensor exp() const;

    tensor sqrt() const;

    tensor sum(const index_t __axis) const;

    tensor row(const index_t __index) const;

    tensor col(const index_t __index) const;

    tensor ceil() const;

    tensor floor() const;

    tensor clone() const;

    tensor clamp(const_pointer __min_val = nullptr, const_pointer __max_val = nullptr) const;

    tensor cos() const;

    tensor sin() const;

    tensor atan() const;

    tensor sinh() const;

    tensor asinh() const;

    tensor asin() const;

    tensor cosh() const;

    tensor acosh() const;

    tensor logical_xor(const tensor& __other) const;

    tensor logical_xor(const value_t __val) const;

    tensor logical_and(const tensor& __other) const;

    tensor logical_and(const value_t __val) const;

    tensor bitwise_not() const;

    tensor bitwise_and(const value_t __val) const;

    tensor bitwise_and(const tensor& __other) const;

    tensor bitwise_or(const value_t __val) const;

    tensor bitwise_or(const tensor& __other) const;

    tensor bitwise_xor(const value_t __val) const;

    tensor bitwise_xor(const tensor& __other) const;

    tensor bitwise_left_shift(const int __amount) const;

    tensor bitwise_right_shift(const int __amount) const;

    tensor matmul(const tensor& __other) const;

    tensor reshape(const shape_t __shape) const;

    tensor reshape_as(const tensor& __other) { return this->reshape(__other.__shape_); }

    tensor cross_product(const tensor& __other) const;

    tensor absolute(const tensor& __tensor) const;

    tensor dot(const tensor& __other) const;

    tensor relu() const;

    tensor transpose() const;

    tensor pow(const tensor& __other) const;

    tensor pow(const value_t __val) const;

    tensor cumprod(index_t __dim = -1) const;

    tensor cat(const std::vector<tensor>& _others, index_t _dim) const;

    tensor argmax(index_t __dim) const;

    tensor unsqueeze(index_t __dim) const;

    index_t lcm() const;

    void sqrt_();

    void exp_();

    void log2_();

    void log10_();

    void log_();

    void frac_();

    void fmod_(const tensor& __other);

    void fmod_(const value_t __val);

    void cos_();

    void cosh_();

    void atan_();

    void acosh_();

    void sinh_();

    void asinh_();

    void asin_();

    void ceil_();

    void floor_();

    void sin_();

    void relu_();

    void clamp_(const_pointer __min_val = nullptr, const_pointer __max_val = nullptr);

    void logical_not_() const { this->bitwise_not_(); }

    void logical_or_(const tensor& __other);

    void logical_or_(const value_t __val);

    void logical_xor_(const tensor& __other);

    void logical_xor_(const value_t __val);

    void logical_and_(const tensor& __other);

    void logical_and_(const value_t __val);

    void pow_(const tensor& __other);

    void pow_(const value_t __val);

    void bitwise_left_shift_(const int __amount);

    void bitwise_right_shift_(const int __amount);

    void bitwise_and_(const value_t __val);

    void bitwise_and_(const tensor& __other);

    void bitwise_or_(const value_t __val);

    void bitwise_or_(const tensor& __other);

    void bitwise_xor_(const value_t __val);

    void bitwise_xor_(const tensor& __other);

    void bitwise_not_();

    void view(std::initializer_list<index_t> __new_sh);

    void print() const noexcept;

    static tensor zeros(const shape_t& __sh);

    static tensor ones(const shape_t& __sh);

    void zeros_(const shape_t& __sh = {});

    void ones_(const shape_t& __sh = {});

    tensor randomize(const shape_t& __sh, bool __bounded = false);

    void randomize_(const shape_t& __sh, bool __bounded = false);

    tensor<index_t> argmax_(index_t __dim) const;

    tensor<index_t> argsort(index_t __dim = -1, bool __ascending = true) const;

   private:
    static void __check_is_scalar_type(const std::string __msg);

    static void __check_is_integral_type(const std::string __msg);

    template<typename __t>
    static void __check_is_same_type(const std::string __msg);

    static void __check_is_arithmetic_type(const std::string __msg);

    void __compute_strides();

    size_t __compute_index(const std::vector<index_t>& __idx) const;

    static uint64_t __computeSize(const shape_t& __dims);

    uint64_t __compute_outer_size(const index_t __dim) const;

    float __frac(const_reference __scalar);

    index_t __lcm(const index_t __a, const index_t __b);
};  // tensor class