#pragma once

#include "tensorbase.hpp"


template<typename T>
struct neon_type_selector;

template<typename T>
struct wide_neon_type_selector;

template<>
struct neon_type_selector<_s8>
{
    using type = neon_s8;
};

template<>
struct neon_type_selector<_s16>
{
    using type = neon_s16;
};

template<>
struct neon_type_selector<_s32>
{
    using type = neon_s32;
};

template<>
struct neon_type_selector<_s64>
{
    using type = neon_s64;
};

template<>
struct neon_type_selector<_f16>
{
    using type = neon_f16;
};

template<>
struct neon_type_selector<_f32>
{
    using type = neon_f32;
};

template<>
struct neon_type_selector<_f64>
{
    using type = neon_f64;
};

template<>
struct neon_type_selector<_u8>
{
    using type = neon_u8;
};

template<>
struct neon_type_selector<_u16>
{
    using type = neon_u16;
};

template<>
struct neon_type_selector<_u32>
{
    using type = neon_u32;
};

template<>
struct neon_type_selector<_u64>
{
    using type = neon_u64;
};

template<>
struct wide_neon_type_selector<_s8>
{
    using type = int8x16x4_t;
};

template<>
struct wide_neon_type_selector<_s16>
{
    using type = int16x8x4_t;
};

template<>
struct wide_neon_type_selector<_s32>
{
    using type = int32x4x4_t;
};

template<>
struct wide_neon_type_selector<_s64>
{
    using type = int64x2x4_t;
};

template<>
struct wide_neon_type_selector<_f16>
{
    using type = float16x8x4_t;
};

template<>
struct wide_neon_type_selector<_f32>
{
    using type = float32x4x4_t;
};

template<>
struct wide_neon_type_selector<_f64>
{
    using type = float64x2x4_t;
};

template<>
struct wide_neon_type_selector<_u8>
{
    using type = uint8x16x4_t;
};

template<>
struct wide_neon_type_selector<_u16>
{
    using type = uint16x8x4_t;
};

template<>
struct wide_neon_type_selector<_u32>
{
    using type = uint32x4x4_t;
};

template<>
struct wide_neon_type_selector<_u64>
{
    using type = uint64x2x4_t;
};

template<typename T>
using neon_type = typename neon_type_selector<T>::type;

template<typename T>
using wide_neon_type = typename wide_neon_type_selector<T>::type;

template<typename T>
neon_type<T> neon_load(const T* load_from) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vld1q_f16(reinterpret_cast<const _f16*>(load_from));
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vld1q_f32(reinterpret_cast<const _f32*>(load_from));
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vld1q_f64(reinterpret_cast<const _f64*>(load_from));
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vld1q_s8(reinterpret_cast<const _s8*>(load_from));
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vld1q_s16(reinterpret_cast<const _s16*>(load_from));
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vld1q_s32(reinterpret_cast<const _s32*>(load_from));
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vld1q_s64(reinterpret_cast<const _s64*>(load_from));
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vld1q_u8(reinterpret_cast<const _u8*>(load_from));
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vld1q_u16(reinterpret_cast<const _u16*>(load_from));
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vld1q_u32(reinterpret_cast<const _u32*>(load_from));
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vld1q_u64(reinterpret_cast<const _u64*>(load_from));
    }
}

template<typename T>
void neon_store(T* store_in, neon_type<T>& store_from) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        vst1q_f16(reinterpret_cast<_f16*>(store_in), store_from);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        vst1q_f32(reinterpret_cast<_f32*>(store_in), store_from);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        vst1q_f64(reinterpret_cast<_f64*>(store_in), store_from);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        vst1q_s8(reinterpret_cast<_s8*>(store_in), store_from);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        vst1q_s16(reinterpret_cast<_s16*>(store_in), store_from);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        vst1q_s32(reinterpret_cast<_s32*>(store_in), store_from);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        vst1q_s64(reinterpret_cast<_s64*>(store_in), store_from);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        vst1q_u8(reinterpret_cast<_u8*>(store_in), store_from);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        vst1q_u16(reinterpret_cast<_u16*>(store_in), store_from);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        vst1q_u32(reinterpret_cast<_u32*>(store_in), store_from);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        vst1q_u64(reinterpret_cast<_u64*>(store_in), store_from);
    }
}

template<typename T>
neon_type<T> neon_vabdq(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vabdq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vabdq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vabdq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vabdq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vabdq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vabdq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vabdq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vabdq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vabdq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vabdq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vabdq_u64(a, b);
    }
}

template<typename T>
neon_type<T> neon_add(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vaddq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vaddq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vaddq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vaddq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vaddq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vaddq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vaddq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vaddq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vaddq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vaddq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vaddq_u64(a, b);
    }
}

template<typename T>
neon_type<T> neon_sub(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vsubq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vsubq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vsubq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vsubq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vsubq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vsubq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vsubq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vsubq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vsubq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vsubq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vsubq_u64(a, b);
    }
}

template<typename T>
neon_type<T> neon_mul(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vmulq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vmulq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vmulq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vmulq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vmulq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vmulq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vmulq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vmulq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vmulq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vmulq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vmulq_u64(a, b);
    }
}

template<typename T>
neon_type<T> neon_dup(T v) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vdupq_n_f16(v);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vdupq_n_f32(v);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vdupq_n_f64(v);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vdupq_n_s8(v);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vdupq_n_s16(v);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vdupq_n_s32(v);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vdupq_n_s64(v);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vdupq_n_u8(v);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vdupq_n_u16(v);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vdupq_n_u32(v);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vdupq_n_u64(v);
    }
}

template<typename T>
neon_type<T> neon_abs(neon_type<T>& a) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vabsq_f16(a);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vabsq_f32(a);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vabsq_f64(a);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vabsq_s8(a);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vabsq_s16(a);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vabsq_s32(a);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vabsq_s64(a);
    }
}

template<typename T>
neon_type<T> neon_vcvtq(neon_type<T>& a) {}

template<typename T>
neon_type<T> neon_cgt(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vcgtq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vcgtq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vcgtq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vcgtq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vcgtq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vcgtq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vcgtq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vcgtq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vcgtq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vcgtq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vcgtq_u64(a, b);
    }
}

template<typename T>
neon_type<T> neon_vbslq(neon_type<T>& a, neon_type<T>& b, neon_type<T>& c) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vbslq_f16(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vbslq_f32(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vbslq_f64(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vbslq_s8(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vbslq_s16(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vbslq_s32(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vbslq_s64(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vbslq_u8(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vbslq_u16(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vbslq_u32(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vbslq_u64(a, b, c);
    }
}

template<typename T>
neon_type<T> neon_max(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vmaxq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vmaxq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vmaxq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vmaxq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vmaxq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vmaxq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vmaxq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vmaxq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vmaxq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vmaxq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vmaxq_u64(a, b);
    }
}

template<typename T>
T neon_maxv(neon_type<T>& a) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vmaxvq_f16(a);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vmaxvq_f32(a);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vmaxvq_f64(a);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vmaxvq_s8(a);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vmaxvq_s16(a);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vmaxvq_s32(a);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vmaxvq_s64(a);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vmaxvq_u8(a);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vmaxvq_u16(a);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vmaxvq_u32(a);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vmaxvq_u64(a);
    }
}

template<typename T>
neon_type<T> neon_min(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vminq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vminq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vminq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vminq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vminq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vminq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vminq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vminq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vminq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vminq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vminq_u64(a, b);
    }
}

template<typename T>
T neon_get_lane(neon_type<T>& a, T& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vgetq_lane_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vgetq_lane_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vgetq_lane_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vgetq_lane_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vgetq_lane_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vgetq_lane_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vgetq_lane_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vgetq_lane_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vgetq_lane_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vgetq_lane_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vgetq_lane_u64(a, b);
    }
}

template<typename T>
neon_type<T> neon_shl(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vshlq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vshlq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vshlq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vshlq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vshlq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vshlq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vshlq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vshlq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vshlq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vshlq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vshlq_u64(a, b);
    }
}

template<typename T>
neon_type<T> neon_or(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vorrq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vorrq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vorrq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vorrq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vorrq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vorrq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vorrq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vorrq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vorrq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vorrq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vorrq_u64(a, b);
    }
}

template<typename T>
neon_type<T> neon_xor(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return veorq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return veorq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return veorq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return veorq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return veorq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return veorq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return veorq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return veorq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return veorq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return veorq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return veorq_u64(a, b);
    }
}

template<typename T>
neon_type<T> neon_and(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vandq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vandq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vandq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vandq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vandq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vandq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vandq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vandq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vandq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vandq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vandq_u64(a, b);
    }
}

template<typename T>
neon_type<_u32> neon_ceq(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vceqq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vceqq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vceqq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vceqq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vceqq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vceqq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vceqq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vceqq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vceqq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vceqq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vceqq_u64(a, b);
    }
}

template<typename T>
neon_type<T> neon_ext(neon_type<T>& a, neon_type<T>& b, T& c) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vextq_f16(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vextq_f32(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vextq_f64(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vextq_s8(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vextq_s16(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vextq_s32(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vextq_s64(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vextq_u8(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vextq_u16(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vextq_u32(a, b, c);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vextq_u64(a, b, c);
    }
}

template<typename T>
T neon_addv(neon_type<T>& a) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vaddvq_f16(a);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vaddvq_f32(a);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vaddvq_f64(a);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vaddvq_s8(a);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vaddvq_s16(a);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vaddvq_s32(a);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vaddvq_s64(a);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vaddvq_u8(a);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vaddvq_u16(a);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vaddvq_u32(a);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vaddvq_u64(a);
    }
}

template<typename T>
wide_neon_type<T> wide_neon_load(const T* load_from) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vld4q_f16(load_from);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vld4q_f32(load_from);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vld4q_f64(load_from);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vld4q_s8(load_from);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vld4q_s16(load_from);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vld4q_s32(load_from);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vld4q_s64(load_from);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vld4q_u8(load_from);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vld4q_u16(load_from);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vld4q_u32(load_from);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vld4q_u64(load_from);
    }
}

template<typename T>
neon_type<T> neon_cleq(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
    {
        return vcleq_f16(a, b);
    }
    else if constexpr (std::is_same_v<T, _f32>)
    {
        return vcleq_f32(a, b);
    }
    else if constexpr (std::is_same_v<T, _f64>)
    {
        return vcleq_f64(a, b);
    }
    else if constexpr (std::is_same_v<T, _s8>)
    {
        return vcleq_s8(a, b);
    }
    else if constexpr (std::is_same_v<T, _s16>)
    {
        return vcleq_s16(a, b);
    }
    else if constexpr (std::is_same_v<T, _s32>)
    {
        return vcleq_s32(a, b);
    }
    else if constexpr (std::is_same_v<T, _s64>)
    {
        return vcleq_s64(a, b);
    }
    else if constexpr (std::is_same_v<T, _u8>)
    {
        return vcleq_u8(a, b);
    }
    else if constexpr (std::is_same_v<T, _u16>)
    {
        return vcleq_u16(a, b);
    }
    else if constexpr (std::is_same_v<T, _u32>)
    {
        return vcleq_u32(a, b);
    }
    else if constexpr (std::is_same_v<T, _u64>)
    {
        return vcleq_u64(a, b);
    }
}