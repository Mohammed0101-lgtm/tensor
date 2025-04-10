#include "tensorbase.hpp"

template <typename T>
struct neon_type_selector;

template <>
struct neon_type_selector<_s8> {
    using type = neon_s8;
};

template <>
struct neon_type_selector<_s16> {
    using type = neon_s16;
};

template <>
struct neon_type_selector<_s32> {
    using type = neon_s32;
};

template <>
struct neon_type_selector<_s64> {
    using type = neon_s64;
};

template <>
struct neon_type_selector<_f16> {
    using type = neon_f16;
};

template <>
struct neon_type_selector<_f32> {
    using type = neon_f32;
};

template <>
struct neon_type_selector<_f64> {
    using type = neon_f64;
};

template <>
struct neon_type_selector<_u8> {
    using type = neon_u8;
};

template <>
struct neon_type_selector<_u16> {
    using type = neon_u16;
};

template <>
struct neon_type_selector<_u32> {
    using type = neon_u32;
};

template <>
struct neon_type_selector<_u64> {
    using type = neon_u64;
};

template <typename T>
using neon_type = typename neon_type_selector<T>::type;

template <typename T>
neon_type<T> neon_load(const T* load_from) {
    if constexpr (std::is_same_v<T, _f16>)
        return vld1q_f16(reinterpret_cast<const _f16*>(load_from));

    else if constexpr (std::is_same_v<T, _f32>)
        return vld1q_f32(reinterpret_cast<const _f32*>(load_from));

    else if constexpr (std::is_same_v<T, _f64>)
        return vld1q_f64(reinterpret_cast<const _f64*>(load_from));

    else if constexpr (std::is_same_v<T, _s8>)
        return vld1q_s8(reinterpret_cast<const _s8*>(load_from));

    else if constexpr (std::is_same_v<T, _s16>)
        return vld1q_s16(reinterpret_cast<const _s16*>(load_from));

    else if constexpr (std::is_same_v<T, _s32>)
        return vld1q_s32(reinterpret_cast<const _s32*>(load_from));

    else if constexpr (std::is_same_v<T, _s64>)
        return vld1q_s64(reinterpret_cast<const _s64*>(load_from));

    else if constexpr (std::is_same_v<T, _u8>)
        return vld1q_u8(reinterpret_cast<const _u8*>(load_from));

    else if constexpr (std::is_same_v<T, _u16>)
        return vld1q_u16(reinterpret_cast<const _u16*>(load_from));

    else if constexpr (std::is_same_v<T, _u32>)
        return vld1q_u32(reinterpret_cast<const _u32*>(load_from));

    else if constexpr (std::is_same_v<T, _u64>)
        return vld1q_u64(reinterpret_cast<const _u64*>(load_from));
}

template <typename T>
void neon_store(T* store_in, neon_type<T>& store_from) {
    if constexpr (std::is_same_v<T, _f16>)
        vst1q_f16(reinterpret_cast<_f16*>(store_in), store_from);

    else if constexpr (std::is_same_v<T, _f32>)
        vst1q_f32(reinterpret_cast<_f32*>(store_in), store_from);

    else if constexpr (std::is_same_v<T, _f64>)
        vst1q_f64(reinterpret_cast<_f64*>(store_in), store_from);

    else if constexpr (std::is_same_v<T, _s8>)
        vst1q_s8(reinterpret_cast<_s8*>(store_in), store_from);

    else if constexpr (std::is_same_v<T, _s16>)
        vst1q_s16(reinterpret_cast<_s16*>(store_in), store_from);

    else if constexpr (std::is_same_v<T, _s32>)
        vst1q_s32(reinterpret_cast<_s32*>(store_in), store_from);

    else if constexpr (std::is_same_v<T, _s64>)
        vst1q_s64(reinterpret_cast<_s64*>(store_in), store_from);
    else if constexpr (std::is_same_v<T, _u8>)

        vst1q_u8(reinterpret_cast<_u8*>(store_in), store_from);
    else if constexpr (std::is_same_v<T, _u16>)

        vst1q_u16(reinterpret_cast<_u16*>(store_in), store_from);
    else if constexpr (std::is_same_v<T, _u32>)

        vst1q_u32(reinterpret_cast<_u32*>(store_in), store_from);

    else if constexpr (std::is_same_v<T, _u64>)
        vst1q_u64(reinterpret_cast<_u64*>(store_in), store_from);
}

template <typename T>
neon_type<T> neon_vabdq(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
        return vabdq_f16(a, b);

    else if constexpr (std::is_same_v<T, _f32>)
        return vabdq_f32(a, b);

    else if constexpr (std::is_same_v<T, _f64>)
        return vabdq_f64(a, b);

    else if constexpr (std::is_same_v<T, _s8>)
        return vabdq_s8(a, b);

    else if constexpr (std::is_same_v<T, _s16>)
        return vabdq_s16(a, b);

    else if constexpr (std::is_same_v<T, _s32>)
        return vabdq_s32(a, b);

    else if constexpr (std::is_same_v<T, _s64>)
        return vabdq_s64(a, b);

    else if constexpr (std::is_same_v<T, _u8>)
        return vabdq_u8(a, b);

    else if constexpr (std::is_same_v<T, _u16>)
        return vabdq_u16(a, b);

    else if constexpr (std::is_same_v<T, _u32>)
        return vabdq_u32(a, b);

    else if constexpr (std::is_same_v<T, _u64>)
        return vabdq_u64(a, b);
}

template <typename T>
neon_type<T> neon_add(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
        return vaddq_f16(a, b);

    else if constexpr (std::is_same_v<T, _f32>)
        return vaddq_f32(a, b);

    else if constexpr (std::is_same_v<T, _f64>)
        return vaddq_f64(a, b);

    else if constexpr (std::is_same_v<T, _s8>)
        return vaddq_s8(a, b);

    else if constexpr (std::is_same_v<T, _s16>)
        return vaddq_s16(a, b);

    else if constexpr (std::is_same_v<T, _s32>)
        return vaddq_s32(a, b);

    else if constexpr (std::is_same_v<T, _s64>)
        return vaddq_s64(a, b);

    else if constexpr (std::is_same_v<T, _u8>)
        return vaddq_u8(a, b);

    else if constexpr (std::is_same_v<T, _u16>)
        return vaddq_u16(a, b);

    else if constexpr (std::is_same_v<T, _u32>)
        return vaddq_u32(a, b);

    else if constexpr (std::is_same_v<T, _u64>)
        return vaddq_u64(a, b);
}

template <typename T>
neon_type<T> neon_sub(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
        return vsubq_f16(a, b);

    else if constexpr (std::is_same_v<T, _f32>)
        return vsubq_f32(a, b);

    else if constexpr (std::is_same_v<T, _f64>)
        return vsubq_f64(a, b);

    else if constexpr (std::is_same_v<T, _s8>)
        return vsubq_s8(a, b);

    else if constexpr (std::is_same_v<T, _s16>)
        return vsubq_s16(a, b);

    else if constexpr (std::is_same_v<T, _s32>)
        return vsubq_s32(a, b);

    else if constexpr (std::is_same_v<T, _s64>)
        return vsubq_s64(a, b);

    else if constexpr (std::is_same_v<T, _u8>)
        return vsubq_u8(a, b);

    else if constexpr (std::is_same_v<T, _u16>)
        return vsubq_u16(a, b);

    else if constexpr (std::is_same_v<T, _u32>)
        return vsubq_u32(a, b);

    else if constexpr (std::is_same_v<T, _u64>)
        return vsubq_u64(a, b);
}

template <typename T>
neon_type<T> neon_mul(neon_type<T>& a, neon_type<T>& b) {
    if constexpr (std::is_same_v<T, _f16>)
        return vmulq_f16(a, b);

    else if constexpr (std::is_same_v<T, _f32>)
        return vmulq_f32(a, b);

    else if constexpr (std::is_same_v<T, _f64>)
        return vmulq_f64(a, b);

    else if constexpr (std::is_same_v<T, _s8>)
        return vmulq_s8(a, b);

    else if constexpr (std::is_same_v<T, _s16>)
        return vmulq_s16(a, b);

    else if constexpr (std::is_same_v<T, _s32>)
        return vmulq_s32(a, b);

    else if constexpr (std::is_same_v<T, _s64>)
        return vmulq_s64(a, b);

    else if constexpr (std::is_same_v<T, _u8>)
        return vmulq_u8(a, b);

    else if constexpr (std::is_same_v<T, _u16>)
        return vmulq_u16(a, b);

    else if constexpr (std::is_same_v<T, _u32>)
        return vmulq_u32(a, b);

    else if constexpr (std::is_same_v<T, _u64>)
        return vmulq_u64(a, b);
}

template <typename T>
neon_type<T> neon_dup(T* v) {
    if constexpr (std::is_same_v<T, _f16>) {
        return vdupq_n_f16(reinterpret_cast<_f16*>(v));
    }

    else if constexpr (std::is_same_v<T, _f32>) {
        return vdupq_n_f32(reinterpret_cast<_f32*>(v));
    } else if constexpr (std::is_same_v<T, _f64>) {
        return vdupq_n_f64(reinterpret_cast<_f64*>(v));
    } else if constexpr (std::is_same_v<T, _s8>) {
        return vdupq_n_s8(reinterpret_cast<_s8*>(v));
    } else if constexpr (std::is_same_v<T, _s16>) {
        return vdupq_n_s16(reinterpret_cast<_s16*>(v));
    } else if constexpr (std::is_same_v<T, _s32>) {
        return vdupq_n_s32(reinterpret_cast<_s32*>(v));
    } else if constexpr (std::is_same_v<T, _s64>) {
        return vdupq_n_s64(reinterpret_cast<_s64*>(v));
    } else if constexpr (std::is_same_v<T, _u8>) {
        return vdupq_n_u8(reinterpret_cast<_u8*>(v));
    } else if constexpr (std::is_same_v<T, _u16>) {
        return vdupq_n_u16(reinterpret_cast<_u16*>(v));
    } else if constexpr (std::is_same_v<T, _u32>) {
        return vdupq_n_u32(reinterpret_cast<_u32*>(v));
    } else if constexpr (std::is_same_v<T, _u64>) {
        return vdupq_n_u64(reinterpret_cast<_u64*>(v));
    }
}

template <typename T>
neon_type<T> neon_abs(neon_type<T>& a) {
    if constexpr (std::is_same_v<T, _f16>)
        return vabsq_f16(a);

    else if constexpr (std::is_same_v<T, _f32>)
        return vabsq_f32(a);

    else if constexpr (std::is_same_v<T, _f64>)
        return vabsq_f64(a);

    else if constexpr (std::is_same_v<T, _s8>)
        return vabsq_s8(a);

    else if constexpr (std::is_same_v<T, _s16>)
        return vabsq_s16(a);

    else if constexpr (std::is_same_v<T, _s32>)
        return vabsq_s32(a);

    else if constexpr (std::is_same_v<T, _s64>)
        return vabsq_s64(a);
}
