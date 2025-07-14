#include "tensorbase.hpp"


// forward declaration
template<class _Tp>
class tensor;

namespace internal {
namespace neon {

template<class _Tp>
tensor<_Tp> operator_times(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp> operator_divide(const tensor<_Tp>& t, const _Tp& value);

template<class _Tp>
tensor<_Tp> operator_divide(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp> operator_divide_eq(const tensor<_Tp>& t, const _Tp& value);

template<class _Tp>
tensor<_Tp> operator_divide_eq(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& fill_(tensor<_Tp>& t, const _Tp& value);

template<class _Tp>
tensor<_Tp>& fill_(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_s16> int16_(const tensor<_Tp>& t);

template<class _Tp>
tensor<_s32> int32_(const tensor<_Tp>& t);

template<class _Tp>
tensor<_u32> uint32_(const tensor<_Tp>& t);

template<class _Tp>
tensor<_f32> float32_(const tensor<_Tp>& t);

template<class _Tp>
tensor<_f64> float64_(const tensor<_Tp>& t);

template<class _Tp>
tensor<_u64> uint64_(const tensor<_Tp>& t);

template<class _Tp>
tensor<_s64> int64_(const tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& fmax_(tensor<_Tp>& t, const _Tp v);

template<class _Tp>
tensor<_Tp>& fmax_(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& fmod_(tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<_Tp>& fmod_(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& frac_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& log10_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& log2_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& exp_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& sqrt_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& cos_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& acos_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& sin_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& tan_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& tanh_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& sinc_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& atan_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& atanh_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& sinh_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& asinh_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& asin_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& cosh_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& acosh_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& pow_(tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<_Tp>& pow_(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& abs_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& dist_(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& dist_(tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<_Tp>& maximum_(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& maximum_(tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<_Tp>& bitwise_right_shift_(tensor<_Tp>& t, const int amount);

template<class _Tp>
tensor<_Tp>& bitwise_left_shift_(tensor<_Tp>& t, const int amount);

template<class _Tp>
tensor<_Tp>& bitwise_or_(tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<_Tp>& bitwise_xor_(tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<_Tp>& bitwise_not_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& bitwise_and_(tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<_Tp>& bitwise_and_(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& bitwise_or_(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& bitwise_xor_(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& zeros_(tensor<_Tp>& t, shape::Shape shape_ = {});

template<class _Tp>
tensor<_Tp>& ones_(tensor<_Tp>& t, shape::Shape shape_);

template<class _Tp>
tensor<_Tp>& randomize_(tensor<_Tp>& t, const shape::Shape& shape_, bool bounded);

template<class _Tp>
tensor<_Tp>& negative_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& relu_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& sigmoid_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& clipped_relu_(tensor<_Tp>& t, const _Tp clip_limit);

template<class _Tp>
tensor<_Tp>& clamp_(tensor<_Tp>& t,
                    const _Tp&   min_val = std::numeric_limits<_Tp>::lowest(),
                    const _Tp&   max_val = std::numeric_limits<_Tp>::max());

template<class _Tp>
tensor<_Tp>& floor_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& ceil_(tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp>& logical_or_(tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<_Tp>& logical_xor_(tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<_Tp>& logical_and_(tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<_Tp>& logical_or_(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& logical_xor_(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& logical_and_(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& operator_plus_eq(tensor<_Tp>& t, const _Tp& value);

template<class _Tp>
tensor<_Tp>& operator_minus_eq(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& operator_times_eq(tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& operator_minus_eq(tensor<_Tp>& t, const _Tp& value);

template<class _Tp>
tensor<_Tp> operator_plus(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp> operator_plus(const tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<_Tp> operator_minus(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp> operator_minus(const tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<_Tp> transpose(const tensor<_Tp>& t);

template<class _Tp>
tensor<_Tp> matmul(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp> absolute_(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp> cross_product(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp> dot(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp> argmax(const tensor<_Tp>& t, const _u64 dimension);

template<class _Tp>
tensor<_Tp> sum(const tensor<_Tp>& t, const _u64 axis);

template<class _Tp>
tensor<_Tp>
slice(const tensor<_Tp>& t, const _u64 dimension, std::optional<_u64> start, std::optional<_u64> end, const _u64 step);

template<class _Tp>
tensor<bool> equal(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<bool> equal(const tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<bool> less_equal(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<bool> less_equal(const tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<bool> less(const tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<bool> less(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<bool> greater(const tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<bool> greater(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<bool> greater_equal(const tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<bool> greater_equal(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_u64> argsort(const tensor<_Tp>& t, _u64 d, bool ascending);

template<class _Tp>
tensor<_u64> argmax_(const tensor<_Tp>& t, _u64 dimension);

template<class _Tp>
_u64 count_nonzero(const tensor<_Tp>& t, _u64 dimension);

template<class _Tp>
double mean(const tensor<_Tp>& t);

template<class _Tp>
tensor<bool> not_equal(const tensor<_Tp>& t, const _Tp value);

template<class _Tp>
tensor<bool> not_equal(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp> absolute(const tensor<_Tp>& t, const tensor<_Tp>& other);

template<class _Tp>
tensor<_Tp>& clamp_min_(tensor<_Tp>& t, const _Tp& min_val);

}  // namespace neon
}  // namespace internal
