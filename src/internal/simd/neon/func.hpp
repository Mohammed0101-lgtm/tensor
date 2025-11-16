#pragma once

#include "tensor.hpp"


// forward declaration
template<class _Tp>
class tensor;

namespace internal {
namespace neon {

template<class _Tp>
arch::tensor<_Tp>& operator_plus_eq(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp> operator_times(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp> operator_divide(const arch::tensor<_Tp>& t, const _Tp& value);

template<class _Tp>
arch::tensor<_Tp> operator_divide(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& operator_divide_eq(arch::tensor<_Tp>& t, const _Tp& value);

template<class _Tp>
arch::tensor<_Tp>& operator_divide_eq(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& fill_(arch::tensor<_Tp>& t, const _Tp& value);

template<class _Tp>
arch::tensor<_Tp>& fill_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_s16> int16_(const arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_s32> int32_(const arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_u32> uint32_(const arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_f32> float32_(const arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_f64> float64_(const arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_u64> uint64_(const arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_s64> int64_(const arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& fmax_(arch::tensor<_Tp>& t, const _Tp v);

template<class _Tp>
arch::tensor<_Tp>& fmax_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& fmod_(arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<_Tp>& fmod_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& frac_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& log10_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& log2_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& exp_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& sqrt_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& cos_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& acos_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& sin_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& tan_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& tanh_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& sinc_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& atan_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& atanh_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& sinh_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& asinh_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& asin_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& cosh_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& acosh_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& pow_(arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<_Tp>& pow_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& abs_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& dist_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& dist_(arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<_Tp>& maximum_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& operator_times_eq(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& operator_times_eq(arch::tensor<_Tp>& t, const _Tp& value);

template<class _Tp>
arch::tensor<_Tp>& maximum_(arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<_Tp>& bitwise_right_shift_(arch::tensor<_Tp>& t, const int amount);

template<class _Tp>
arch::tensor<_Tp>& bitwise_left_shift_(arch::tensor<_Tp>& t, const int amount);

template<class _Tp>
arch::tensor<_Tp>& bitwise_or_(arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<_Tp>& bitwise_xor_(arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<_Tp>& bitwise_not_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& bitwise_and_(arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<_Tp>& bitwise_and_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& bitwise_or_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& bitwise_xor_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& zeros_(arch::tensor<_Tp>& t, shape::Shape shape_ = {});

template<class _Tp>
arch::tensor<_Tp>& ones_(arch::tensor<_Tp>& t, shape::Shape shape_);

template<class _Tp>
arch::tensor<_Tp>& randomize_(arch::tensor<_Tp>& t, const shape::Shape& shape_, bool bounded);

template<class _Tp>
arch::tensor<_Tp>& negative_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& relu_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& sigmoid_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& clipped_relu_(arch::tensor<_Tp>& t, const _Tp clip_limit);

template<class _Tp>
arch::tensor<_Tp>& clamp_(arch::tensor<_Tp>& t,
                          const _Tp&         min_val = std::numeric_limits<_Tp>::lowest(),
                          const _Tp&         max_val = std::numeric_limits<_Tp>::max());

template<class _Tp>
arch::tensor<_Tp>& floor_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& ceil_(arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp>& logical_or_(arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<_Tp>& logical_xor_(arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<_Tp>& logical_and_(arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<_Tp>& logical_or_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& logical_xor_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& logical_and_(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& operator_plus_eq(arch::tensor<_Tp>& t, const _Tp& value);

template<class _Tp>
arch::tensor<_Tp>& operator_minus_eq(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& operator_times_eq(arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& operator_minus_eq(arch::tensor<_Tp>& t, const _Tp& value);

template<class _Tp>
arch::tensor<_Tp> operator_plus(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp> operator_plus(const arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<_Tp> operator_minus(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp> operator_minus(const arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<_Tp> transpose(const arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<_Tp> matmul(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp> absolute_(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp> cross_product(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp> dot(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp> argmax(const arch::tensor<_Tp>& t, const _u64 dimension);

template<class _Tp>
arch::tensor<_Tp> sum(const arch::tensor<_Tp>& t, const _u64 axis);

template<class _Tp>
arch::tensor<_Tp> slice(const arch::tensor<_Tp>& t,
                        const _u64               dimension,
                        std::optional<_u64>      start,
                        std::optional<_u64>      end,
                        const _u64               step);

template<class _Tp>
arch::tensor<bool> equal(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<bool> equal(const arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<bool> less_equal(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<bool> less_equal(const arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<bool> less(const arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<bool> less(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<bool> greater(const arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<bool> greater(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<bool> greater_equal(const arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<bool> greater_equal(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_u64> argsort(const arch::tensor<_Tp>& t, _u64 d, bool ascending);

template<class _Tp>
arch::tensor<_u64> argmax_(const arch::tensor<_Tp>& t, _u64 dimension);

template<class _Tp>
_u64 count_nonzero(const arch::tensor<_Tp>& t, _u64 dimension);

template<class _Tp>
double mean(const arch::tensor<_Tp>& t);

template<class _Tp>
arch::tensor<bool> not_equal(const arch::tensor<_Tp>& t, const _Tp value);

template<class _Tp>
arch::tensor<bool> not_equal(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp> absolute(const arch::tensor<_Tp>& t, const arch::tensor<_Tp>& other);

template<class _Tp>
arch::tensor<_Tp>& clamp_min_(arch::tensor<_Tp>& t, const _Tp& min_val);

}  // namespace neon
}  // namespace internal
