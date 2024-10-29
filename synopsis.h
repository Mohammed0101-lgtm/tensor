



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
    typedef const data_t            const_data_container;

    typedef std::__wrap_iter<typename std::allocator_traits<std::allocator<_Tp>>::pointer>
        __iterator;
    typedef std::__wrap_iter<typename std::allocator_traits<std::allocator<_Tp>>::const_pointer>
                                                    __const_iterator;
    typedef std::reverse_iterator<__iterator>       reverse_iterator;
    typedef std::reverse_iterator<__const_iterator> const_reverse_iterator;


   private:
    data_t                  __data_;
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

    data_t     storage() const;
    shape_type shape() const;
    shape_type strides() const;
    size_t     n_dims() const;

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

