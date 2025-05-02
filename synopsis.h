//
// tensor base
//

#pragma once

#include <__algorithm/clamp.h>
#include <__algorithm/comp.h>
#include <__algorithm/sort.h>
#include <__functional/hash.h>
#include <arm_neon.h>
#include <omp-tools.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cmath>
#include <compare>
#include <complex>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stack>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <valarray>
#include <vector>
#include <version>

#include "error.hpp"

#if defined(USE_CUDA)
    #include <cuda_runtime.h>
#endif

using _s8  = int8_t;
using _s16 = int16_t;
using _s32 = int32_t;
using _s64 = int64_t;

using _u8  = uint8_t;
using _u16 = uint16_t;
using _u32 = uint32_t;
using _u64 = uint64_t;

using _f16 = float16_t;
using _f32 = float32_t;
using _f64 = float64_t;

using neon_s8  = int8x16_t;
using neon_s16 = int16x8_t;
using neon_s32 = int32x4_t;
using neon_s64 = int64x2_t;

using neon_u8  = uint8x16_t;
using neon_u16 = uint16x8_t;
using neon_u32 = uint32x4_t;
using neon_u64 = uint64x2_t;

using neon_f16 = float16x8_t;
using neon_f32 = float32x4_t;
using neon_f64 = float64x2_t;

constexpr int _ARM64_REG_WIDTH = 128;  // 128 bit wide register

namespace aten {

template<class _Tp>
class tensor
{
   public:
    using self                   = tensor;
    using value_type             = _Tp;
    using data_t                 = std::vector<value_type>;
    using index_type             = uint64_t;
    using shape_type             = std::vector<index_type>;
    using reference              = value_type&;
    using const_reference        = const value_type&;
    using pointer                = value_type*;
    using const_pointer          = const value_type*;
    using iterator               = typename data_t::iterator;
    using const_iterator         = typename data_t::const_iterator;
    using reverse_iterator       = typename data_t::reverse_iterator;
    using const_reverse_iterator = typename data_t::const_reverse_iterator;

    enum class Device {
        CPU,
        CUDA
    };

   protected:
    static const std::size_t simd_width = _ARM64_REG_WIDTH / sizeof(value_type);
    static_assert(simd_width % 2 == 0, "register width must divide the size of the data type evenly");

   private:
    mutable data_t data_;
    mutable shape_type shape_;
    mutable shape_type strides_;
    Device device_;
    bool is_cuda_tensor_ = false;

   public:
    tensor() = default;
    explicit tensor(const shape_type& shape_, const_reference v, Device d = Device::CPU);
    explicit tensor(const shape_type& shape_, Device d = Device::CPU);
    explicit tensor(const shape_type& shape_, const data_t& d, Device dev = Device::CPU);
    tensor(const aten::tensor& t);
    tensor(aten::tensor&& t) noexcept;
    tensor(const shape_type& shape_, const aten::tensor& other);
    tensor(const shape_type& shape_, std::initializer_list<value_type> init_list, Device d = Device::CPU);

   private:
    class destroy_tensor
    {
       private:
        aten::tensor& tens_;

       public:
        explicit destroy_tensor(aten::tensor& tens) :
            tens_(tens) {}

        void operator()() {}
    };

   public:
    ~tensor() { destroy_tensor (*this)(); }

    aten::tensor<short> short_() const;
    aten::tensor<long long> long_long_() const;
    aten::tensor<long> long_() const;
    aten::tensor<int> int_() const;
    aten::tensor<unsigned int> unsigned_int_() const;
    aten::tensor<unsigned long> unsigned_long_() const;
    aten::tensor<float> float_() const;
    aten::tensor<double> double_() const;
    data_t storage() const noexcept;
    shape_type shape() const noexcept;
    shape_type strides() const noexcept;
    Device device() const noexcept;
    std::size_t n_dims() const noexcept;
    index_type size(const index_type dimension) const;
    index_type capacity() const noexcept;
    index_type count_nonzero(index_type dimension = 0) const;
    index_type lcm() const;
    index_type hash() const;
    reference at(shape_type idx);
    reference operator[](const index_type idx);
    const_reference at(const shape_type idx) const;
    const_reference operator[](const index_type idx) const;
    reference operator()(std::initializer_list<index_type> index_list);
    const_reference operator()(std::initializer_list<index_type> index_list) const;
    bool empty() const;
    aten::tensor<bool> bool_() const;
    aten::tensor<bool> logical_not() const;
    aten::tensor<bool> logical_or(const value_type value) const;
    aten::tensor<bool> logical_or(const aten::tensor& other) const;
    aten::tensor<bool> less_equal(const aten::tensor& other) const;
    aten::tensor<bool> less_equal(const value_type value) const;
    aten::tensor<bool> greater_equal(const aten::tensor& other) const;
    aten::tensor<bool> greater_equal(const value_type value) const;
    aten::tensor<bool> equal(const aten::tensor& other) const;
    aten::tensor<bool> equal(const value_type value) const;
    aten::tensor<bool> not_equal(const aten::tensor& other) const;
    aten::tensor<bool> not_equal(const value_type value) const;
    aten::tensor<bool> less(const aten::tensor& other) const;
    aten::tensor<bool> less(const value_type value) const;
    aten::tensor<bool> greater(const aten::tensor& other) const;
    aten::tensor<bool> greater(const value_type value) const;
    bool operator==(const aten::tensor& other) const;
    bool operator!=(const aten::tensor& other) const;
    aten::tensor operator+(const aten::tensor& other) const;
    aten::tensor operator-(const aten::tensor& other) const;
    aten::tensor operator+(const value_type value) const;
    aten::tensor operator-(const value_type value) const;
    aten::tensor operator*(const value_type value) const;
    aten::tensor operator/(const aten::tensor& other) const;
    aten::tensor operator/(const_reference value) const;
    aten::tensor operator*(const aten::tensor& other) const;
    aten::tensor& operator-=(const aten::tensor& other) const;
    aten::tensor& operator+=(const aten::tensor& other) const;
    aten::tensor& operator*=(const aten::tensor& other) const;
    aten::tensor& operator/=(const aten::tensor& other) const;
    aten::tensor& operator+=(const_reference value) const;
    aten::tensor& operator*=(const_reference value) const;
    aten::tensor& operator-=(const_reference value) const;
    aten::tensor& operator/=(const_reference value) const;
    aten::tensor& operator=(aten::tensor&& other) const noexcept;
    aten::tensor& operator=(const aten::tensor&);
    aten::tensor<bool>& operator!();
    const aten::tensor<bool>& operator!() const;
    aten::tensor
    slice(index_type dimension, std::optional<index_type> start, std::optional<index_type> end, int64_t step) const;
    aten::tensor fmax(const aten::tensor& other) const;
    aten::tensor fmax(const value_type value) const;
    aten::tensor fmod(const aten::tensor& other) const;
    aten::tensor fmod(const value_type value) const;
    aten::tensor frac() const;
    aten::tensor log() const;
    aten::tensor log10() const;
    aten::tensor log2() const;
    aten::tensor exp() const;
    aten::tensor sqrt() const;
    aten::tensor sum(const index_type axis) const;
    aten::tensor row(const index_type index) const;
    aten::tensor col(const index_type index) const;
    aten::tensor ceil() const;
    aten::tensor floor() const;
    aten::tensor clone() const;
    aten::tensor clamp(const_reference min_val = std::numeric_limits<value_type>::lowest(),
                       const_reference max_val = std::numeric_limits<value_type>::max()) const;
    aten::tensor cos() const;
    aten::tensor cosh() const;
    aten::tensor acos() const;
    aten::tensor acosh() const;
    aten::tensor tan() const;
    aten::tensor tanh() const;
    aten::tensor atan() const;
    aten::tensor atanh() const;
    aten::tensor sin() const;
    aten::tensor sinc() const;
    aten::tensor sinh() const;
    aten::tensor asin() const;
    aten::tensor asinh() const;
    aten::tensor abs() const;
    aten::tensor logical_xor(const aten::tensor& other) const;
    aten::tensor logical_xor(const value_type value) const;
    aten::tensor logical_and(const aten::tensor& other) const;
    aten::tensor logical_and(const value_type value) const;
    aten::tensor bitwise_not() const;
    aten::tensor bitwise_and(const value_type value) const;
    aten::tensor bitwise_and(const aten::tensor& other) const;
    aten::tensor bitwise_or(const value_type value) const;
    aten::tensor bitwise_or(const aten::tensor& other) const;
    aten::tensor bitwise_xor(const value_type value) const;
    aten::tensor bitwise_xor(const aten::tensor& other) const;
    aten::tensor bitwise_left_shift(const int amount) const;
    aten::tensor bitwise_right_shift(const int amount) const;
    aten::tensor matmul(const aten::tensor& other) const;
    aten::tensor reshape(const shape_type shape) const;
    aten::tensor reshape_as(const aten::tensor& other) const;
    aten::tensor cross_product(const aten::tensor& other) const;
    aten::tensor absolute(const aten::tensor& other) const;
    aten::tensor dot(const aten::tensor& other) const;
    aten::tensor relu() const;
    aten::tensor transpose() const;
    aten::tensor fill(const value_type value) const;
    aten::tensor resize_as(const shape_type shape_) const;
    aten::tensor all() const;
    aten::tensor any() const;
    aten::tensor det() const;
    aten::tensor square() const;
    aten::tensor clipped_relu(const value_type clip_limit) const;
    aten::tensor sort(index_type dimension, bool descending = false) const;
    aten::tensor remainder(const value_type value) const;
    aten::tensor remainder(const aten::tensor& other) const;
    aten::tensor maximum(const aten::tensor& other) const;
    aten::tensor maximum(const_reference value) const;
    aten::tensor dist(const aten::tensor& other) const;
    aten::tensor dist(const value_type value) const;
    aten::tensor squeeze(index_type dimension) const;
    aten::tensor negative() const;
    aten::tensor repeat(const data_t& d, int dimension = 0) const;
    aten::tensor permute(const index_type dimension) const;
    aten::tensor log_softmax(const index_type dimension) const;
    aten::tensor gcd(const aten::tensor& other) const;
    aten::tensor gcd(const value_type value) const;
    aten::tensor pow(const aten::tensor& other) const;
    aten::tensor pow(const value_type value) const;
    aten::tensor cumprod(index_type dimension = -1) const;
    aten::tensor cat(const std::vector<aten::tensor>& _others, index_type _dim) const;
    aten::tensor argmax(index_type dimension) const;
    aten::tensor unsqueeze(index_type dimension) const;
    aten::tensor zeros(const shape_type& shape_);
    aten::tensor ones(const shape_type& shape_);
    aten::tensor randomize(const shape_type& shape_, bool bounded = false);
    aten::tensor get_minor(index_type a, index_type b) const;
    aten::tensor expand_as(shape_type shape_, index_type dimension) const;
    aten::tensor lcm(const aten::tensor& other) const;
    double mean() const;
    double median(const index_type dimension) const;
    double mode(const index_type dimension) const;
    aten::tensor& push_back(value_type v) const;
    aten::tensor& pop_back() const;
    aten::tensor& sqrt_();
    aten::tensor& exp_();
    aten::tensor& log2_();
    aten::tensor& log10_();
    aten::tensor& log_();
    aten::tensor& frac_();
    aten::tensor& fmod_(const aten::tensor& other);
    aten::tensor& fmod_(const value_type value);
    aten::tensor& cos_();
    aten::tensor& cosh_();
    aten::tensor& acos_();
    aten::tensor& acosh_();
    aten::tensor& tan_();
    aten::tensor& tanh_();
    aten::tensor& atan_();
    aten::tensor& atanh_();
    aten::tensor& sin_();
    aten::tensor& sinh_();
    aten::tensor& asin_();
    aten::tensor& asinh_();
    aten::tensor& ceil_();
    aten::tensor& floor_();
    aten::tensor& relu_();
    aten::tensor& clamp_(const_reference min_val = std::numeric_limits<value_type>::lowest(),
                         const_reference max_val = std::numeric_limits<value_type>::max());
    aten::tensor clamp_min(const_reference min_val) const;
    aten::tensor& clamp_min_(const_reference min_val);
    aten::tensor clamp_max(const_reference max_val) const;
    aten::tensor& clamp_max_(const_reference max_val);
    aten::tensor& logical_not_();
    aten::tensor& logical_or_(const aten::tensor& other);
    aten::tensor& logical_or_(const value_type value);
    aten::tensor& logical_xor_(const aten::tensor& other);
    aten::tensor& logical_xor_(const value_type value);
    aten::tensor& logical_and_(const aten::tensor& other);
    aten::tensor& logical_and_(const value_type value);
    aten::tensor& abs_();
    aten::tensor& log_softmax_(const index_type dimension);
    aten::tensor& permute_(const index_type dimension);
    aten::tensor& repeat_(const data_t& d, int dimension = 0);
    aten::tensor& negative_();
    aten::tensor& transpose_();
    aten::tensor& unsqueeze_(index_type dimension);
    aten::tensor& squeeze_(index_type dimension);
    aten::tensor& resize_as_(const shape_type shape_);
    aten::tensor& dist_(const aten::tensor& other);
    aten::tensor& dist_(const value_type value);
    aten::tensor& maximum_(const aten::tensor& other);
    aten::tensor& maximum_(const value_type value);
    aten::tensor& remainder_(const value_type value);
    aten::tensor& remainder_(const aten::tensor& other);
    aten::tensor& fill_(const value_type value);
    aten::tensor& fill_(const aten::tensor& other);
    aten::tensor& sigmoid_();
    aten::tensor& clipped_relu_(const value_type clip_limit);
    aten::tensor& square_();
    aten::tensor& pow_(const aten::tensor& other);
    aten::tensor& pow_(const value_type value);
    aten::tensor& sinc_();
    aten::tensor& bitwise_left_shift_(const int amount);
    aten::tensor& bitwise_right_shift_(const int amount);
    aten::tensor& bitwise_and_(const value_type value);
    aten::tensor& bitwise_and_(const aten::tensor& other);
    aten::tensor& bitwise_or_(const value_type value);
    aten::tensor& bitwise_or_(const aten::tensor& other);
    aten::tensor& bitwise_xor_(const value_type value);
    aten::tensor& bitwise_xor_(const aten::tensor& other);
    aten::tensor& bitwise_not_();
    aten::tensor& view(std::initializer_list<index_type> new_shape);
    aten::tensor& fmax_(const aten::tensor& other);
    aten::tensor& fmax_(const value_type value);
    aten::tensor& randomize_(const shape_type& shape_, bool bounded = false);
    aten::tensor& zeros_(shape_type shape_ = {});
    aten::tensor& ones_(shape_type shape_ = {});
    const aten::tensor& sqrt_() const;
    const aten::tensor& exp_() const;
    const aten::tensor& log2_() const;
    const aten::tensor& log10_() const;
    const aten::tensor& log_() const;
    const aten::tensor& frac_() const;
    const aten::tensor& fmod_(const aten::tensor& other) const;
    const aten::tensor& fmod_(const value_type value) const;
    const aten::tensor& cos_() const;
    const aten::tensor& cosh_() const;
    const aten::tensor& acos_() const;
    const aten::tensor& acosh_() const;
    const aten::tensor& tan_() const;
    const aten::tensor& tanh_() const;
    const aten::tensor& atan_() const;
    const aten::tensor& atanh_() const;
    const aten::tensor& sin_() const;
    const aten::tensor& sinh_() const;
    const aten::tensor& asin_() const;
    const aten::tensor& asinh_() const;
    const aten::tensor& ceil_() const;
    const aten::tensor& floor_() const;
    const aten::tensor& relu_() const;
    const aten::tensor& clamp_(const_reference min_val = std::numeric_limits<value_type>::lowest(),
                               const_reference max_val = std::numeric_limits<value_type>::max()) const;
    const aten::tensor& clamp_min_(const_reference min_val) const;
    const aten::tensor& clamp_max_(const_reference max_val) const;
    const aten::tensor& logical_not_() const;
    const aten::tensor& logical_or_(const aten::tensor& other) const;
    const aten::tensor& logical_or_(const value_type value) const;
    const aten::tensor& logical_xor_(const aten::tensor& other) const;
    const aten::tensor& logical_xor_(const value_type value) const;
    const aten::tensor& logical_and_(const aten::tensor& other) const;
    const aten::tensor& logical_and_(const value_type value) const;
    const aten::tensor& abs_() const;
    const aten::tensor& log_softmax_(const index_type dimension) const;
    const aten::tensor& permute_(const index_type dimension) const;
    const aten::tensor& repeat_(const data_t& d, int dimension = 0) const;
    const aten::tensor& negative_() const;
    const aten::tensor& transpose_() const;
    const aten::tensor& unsqueeze_(index_type dimension) const;
    const aten::tensor& squeeze_(index_type dimension) const;
    const aten::tensor& resize_as_(const shape_type shape_) const;
    const aten::tensor& dist_(const aten::tensor& other) const;
    const aten::tensor& dist_(const value_type value) const;
    const aten::tensor& maximum_(const aten::tensor& other) const;
    const aten::tensor& maximum_(const value_type value) const;
    const aten::tensor& remainder_(const value_type value) const;
    const aten::tensor& remainder_(const aten::tensor& other) const;
    const aten::tensor& fill_(const value_type value) const;
    const aten::tensor& fill_(const aten::tensor& other) const;
    const aten::tensor& sigmoid_() const;
    const aten::tensor& clipped_relu_(const value_type clip_limit) const;
    const aten::tensor& square_() const;
    const aten::tensor& pow_(const aten::tensor& other) const;
    const aten::tensor& pow_(const value_type value) const;
    const aten::tensor& sinc_() const;
    const aten::tensor& bitwise_left_shift_(const int amount) const;
    const aten::tensor& bitwise_right_shift_(const int amount) const;
    const aten::tensor& bitwise_and_(const value_type value) const;
    const aten::tensor& bitwise_and_(const aten::tensor& other) const;
    const aten::tensor& bitwise_or_(const value_type value) const;
    const aten::tensor& bitwise_or_(const aten::tensor& other) const;
    const aten::tensor& bitwise_xor_(const value_type value) const;
    const aten::tensor& bitwise_xor_(const aten::tensor& other) const;
    const aten::tensor& bitwise_not_() const;
    const aten::tensor& view(std::initializer_list<index_type> new_shape) const;
    const aten::tensor& fmax_(const aten::tensor& other) const;
    const aten::tensor& fmax_(const value_type value) const;
    const aten::tensor& randomize_(const shape_type& shape_, bool bounded = false) const;
    const aten::tensor& zeros_(shape_type shape_ = {}) const;
    const aten::tensor& ones_(shape_type shape_ = {}) const;
    aten::tensor<index_type> argmax_(index_type dimension) const;
    aten::tensor<index_type> argsort(index_type dimension = -1, bool ascending = true) const;

    void print() const;

   private:
    aten::tensor& neon_fmax_(const value_type v);
    aten::tensor& neon_fmax_(const aten::tensor& other);
    aten::tensor& neon_fmod_(const value_type value);
    aten::tensor& neon_fmod_(const aten::tensor& other);
    aten::tensor& neon_frac_();
    aten::tensor& neon_log_();
    aten::tensor& neon_log10_();
    aten::tensor& neon_log2_();
    aten::tensor& neon_exp_();
    aten::tensor& neon_sqrt_();
    aten::tensor& neon_cos_();
    aten::tensor& neon_acos_();
    aten::tensor& neon_sin_();
    aten::tensor& neon_tan_();
    aten::tensor& neon_tanh_();
    aten::tensor& neon_sinc_();
    aten::tensor& neon_atan_();
    aten::tensor& neon_atanh_();
    aten::tensor& neon_sinh_();
    aten::tensor& neon_asinh_();
    aten::tensor& neon_asin_();
    aten::tensor& neon_cosh_();
    aten::tensor& neon_acosh_();
    aten::tensor& neon_pow_(const value_type value);
    aten::tensor& neon_pow_(const aten::tensor& other);
    aten::tensor& neon_abs_();
    aten::tensor& neon_dist_(const aten::tensor& other);
    aten::tensor& neon_dist_(const value_type value);
    aten::tensor& neon_maximum_(const aten::tensor& other);
    aten::tensor& neon_maximum_(const value_type value);
    aten::tensor& neon_bitwise_right_shift_(const int amount);
    aten::tensor& neon_bitwise_left_shift_(const int amount);
    aten::tensor& neon_bitwise_or_(const value_type value);
    aten::tensor& neon_bitwise_xor_(const value_type value);
    aten::tensor& neon_bitwise_not_();
    aten::tensor& neon_bitwise_and_(const value_type value);
    aten::tensor& neon_bitwise_and_(const aten::tensor& other);
    aten::tensor& neon_bitwise_or_(const aten::tensor& other);
    aten::tensor& neon_bitwise_xor_(const aten::tensor& other);
    aten::tensor& neon_zeros_(shape_type shape_ = {});
    aten::tensor& neon_ones_(shape_type shape_);
    aten::tensor& neon_randomize_(const shape_type& shape_, bool bounded);
    aten::tensor& neon_negative_();
    aten::tensor& neon_relu_();
    aten::tensor& neon_sigmoid_();
    aten::tensor& neon_clipped_relu_(const value_type clip_limit);
    aten::tensor& neon_clamp_(const_reference min_val = std::numeric_limits<value_type>::lowest(),
                              const_reference max_val = std::numeric_limits<value_type>::max());
    aten::tensor& neon_floor_();
    aten::tensor& neon_ceil_();
    aten::tensor& neon_logical_or_(const value_type value);
    aten::tensor& neon_logical_xor_(const value_type value);
    aten::tensor& neon_logical_and_(const value_type value);
    aten::tensor& neon_logical_or_(const aten::tensor& other);
    aten::tensor& neon_logical_xor_(const aten::tensor& other);
    aten::tensor& neon_logical_and_(const aten::tensor& other);
    aten::tensor& neon_operator_plus_eq(const_reference value) const;
    aten::tensor& neon_operator_minus_eq(const aten::tensor& other) const;
    aten::tensor& neon_operator_times_eq(const aten::tensor& other) const;
    aten::tensor& neon_operator_minus_eq(const_reference value) const;
    aten::tensor<_s32> neon_int32_() const;
    aten::tensor<_u32> neon_uint32_() const;
    aten::tensor<_f32> neon_float32_() const;
    aten::tensor<_f64> neon_double_() const;
    aten::tensor<uint64_t> neon_unsigned_long_() const;
    aten::tensor<int64_t> neon_long_() const;
    aten::tensor neon_operator_plus(const aten::tensor& other) const;
    aten::tensor neon_operator_plus(const value_type value) const;
    aten::tensor neon_operator_minus(const aten::tensor& other) const;
    aten::tensor neon_operator_minus(const value_type value) const;
    aten::tensor neon_transpose() const;
    aten::tensor neon_matmul(const aten::tensor& other) const;
    aten::tensor neon_absolute_(const aten::tensor& aten::tensor) const;
    aten::tensor neon_cross_product(const aten::tensor& other) const;
    aten::tensor neon_dot(const aten::tensor& other) const;
    aten::tensor neon_argmax(index_type dimension) const;
    aten::tensor neon_sum(const index_type axis) const;
    aten::tensor neon_slice(index_type dimension,
                            std::optional<index_type> start,
                            std::optional<index_type> end,
                            index_type step) const;
    aten::tensor<bool> neon_equal(const aten::tensor& other) const;
    aten::tensor<bool> neon_equal(const value_type value) const;
    aten::tensor<bool> neon_less_equal(const aten::tensor& other) const;
    aten::tensor<bool> neon_less_equal(const value_type value) const;
    aten::tensor<bool> neon_less(const value_type value) const;
    aten::tensor<bool> neon_less(const aten::tensor& other) const;
    aten::tensor<bool> neon_greater(const value_type value) const;
    aten::tensor<bool> neon_greater(const aten::tensor& other) const;
    aten::tensor<bool> neon_greater_equal(const value_type value) const;
    aten::tensor<bool> neon_greater_equal(const aten::tensor& other) const;
    aten::tensor<index_type> neon_argsort(index_type d, bool ascending) const;
    aten::tensor<index_type> neon_argmax_(index_type dimension) const;
    index_type neon_count_nonzero(index_type dimension) const;
    double neon_mean() const;

   private:
    [[nodiscard]] std::size_t computeStride(std::size_t dimension, const shape_type& shape) const noexcept;
    void printRecursive(std::size_t index, std::size_t depth, const shape_type& shape) const;
    void compute_strides();
    [[nodiscard]] index_type compute_index(const std::vector<index_type>& idx) const;
    [[nodiscard]] static index_type computeSize(const shape_type& dims) noexcept;
    index_type compute_outer_size(const index_type dimension) const;
    [[nodiscard]] static _f32 frac(const_reference value) noexcept;
    // where the aten::tensor is stored
    bool is_cuda_device() const;
    bool equal_shape(const shape_type& x, const shape_type& y) const;

};  // aten::tensor class


template<>
class tensor<bool>;  // explicit instantiation

template<>
class tensor<bool>
{
   public:
    using self                   = tensor;
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

    enum class Device {
        CPU,
        CUDA
    };

   private:
    mutable data_t data_;
    mutable shape_type shape_;
    mutable shape_type strides_;
    Device device_;
    bool is_cuda_tensor_ = false;

   public:
    tensor() = default;
    explicit tensor(const shape_type& shape_, value_type v, Device d = Device::CPU);
    explicit tensor(const shape_type& shape_, Device d = Device::CPU);
    explicit tensor(const shape_type& shape_, const data_t& d, Device dev = Device::CPU);
    tensor(const tensor& t);
    tensor(tensor&& t) noexcept;
    tensor(const shape_type& shape_, std::initializer_list<value_type> init_list, Device d = Device::CPU);
    tensor(const shape_type& shape_, const tensor& other);

    data_t storage() const noexcept;
    shape_type shape() const noexcept;
    shape_type strides() const noexcept;
    Device device() const noexcept;
    bool operator==(const tensor& other) const;
    bool operator!=(const tensor& other) const;
    aten::tensor<bool>& operator=(const aten::tensor<bool>& other) const noexcept;
    reference at(shape_type idx);
    reference operator[](const index_type idx);
    const_reference at(const shape_type idx) const;
    const_reference operator[](const index_type idx) const;
    reference operator()(std::initializer_list<index_type> index_list);
    const_reference operator()(std::initializer_list<index_type> index_list) const;
    bool empty() const;
    std::size_t n_dims() const noexcept;
    index_type size(const index_type dimension) const;
    index_type capacity() const noexcept;
    aten::tensor<bool> logical_not() const;
    aten::tensor<bool> logical_or(const value_type value) const;
    aten::tensor<bool> logical_or(const tensor& other) const;
    aten::tensor<bool> logical_and(const value_type value) const;
    aten::tensor<bool> logical_and(const tensor& other) const;
    aten::tensor<bool> logical_xor(const value_type value) const;
    aten::tensor<bool> logical_xor(const tensor& other) const;
    aten::tensor<bool>& logical_not_();
    aten::tensor<bool>& logical_or_(const value_type value);
    aten::tensor<bool>& logical_or_(const tensor& other);
    aten::tensor<bool>& logical_and_(const value_type value);
    aten::tensor<bool>& logical_and_(const tensor& other);
    aten::tensor<bool>& logical_xor_(const value_type value);
    aten::tensor<bool>& logical_xor_(const tensor& other);
    const aten::tensor<bool>& logical_not_() const;
    const aten::tensor<bool>& logical_or_(const value_type value) const;
    const aten::tensor<bool>& logical_or_(const tensor& other) const;
    const aten::tensor<bool>& logical_and_(const value_type value) const;
    const aten::tensor<bool>& logical_and_(const tensor& other) const;
    const aten::tensor<bool>& logical_xor_(const value_type value) const;
    const aten::tensor<bool>& logical_xor_(const tensor& other) const;
    aten::tensor<bool>& operator!();
    const aten::tensor<bool>& operator!() const;
    aten::tensor<bool>
    slice(index_type dimension, std::optional<index_type> start, std::optional<index_type> end, index_type step) const;
    aten::tensor<bool> row(const index_type index) const;
    aten::tensor<bool> col(const index_type index) const;
    aten::tensor<bool> clone() const;
    aten::tensor<bool> reshape(const shape_type shape_) const;
    aten::tensor<bool> reshape_as(const tensor& other) const;
    aten::tensor<bool> transpose() const;
    aten::tensor<bool>& transpose_();
    const aten::tensor<bool>& transpose_() const;
    aten::tensor<bool> resize_as(const shape_type shape_) const;
    aten::tensor<bool>& resize_as_(const shape_type shape_);
    aten::tensor<bool> squeeze(const index_type dimension) const;
    aten::tensor<bool>& squeeze_(const index_type dimension);
    aten::tensor<bool> repeat(const data_t& d, int dimension) const;
    aten::tensor<bool>& repeat_(const data_t& d, int dimension);
    aten::tensor<bool> permute(const index_type dimension) const;
    aten::tensor<bool> cat(const std::vector<tensor<value_type>>& others, index_type dimension) const;
    aten::tensor<bool> unsqueeze(const index_type dimension) const;
    aten::tensor<bool>& randomize_(const shape_type& shape_ = {});
    aten::tensor<bool>& push_back(value_type v);
    aten::tensor<bool>& pop_back(value_type v);
    aten::tensor<bool>& view(std::initializer_list<index_type> shape_);
    void print() const;

   private:
    bool equal_shape(const shape_type& x, const shape_type& y) const;
    [[nodiscard]]
    inline std::size_t computeStride(std::size_t dimension, const shape_type& shape) const noexcept;
    void printRecursive(std::size_t index, std::size_t depth, const shape_type& shape) const;
    void compute_strides();
    [[nodiscard]]
    index_type compute_index(const std::vector<index_type>& idx) const;
    [[nodiscard]]
    static index_type computeSize(const shape_type& dims) noexcept;
    index_type compute_outer_size(const index_type dimension) const;
    [[nodiscard]]
    static _f32 frac(const_reference value) noexcept;
    bool is_cuda_device() const;
};  // aten::tensor<bool>

};  // namespace at