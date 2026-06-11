#pragma once
// Numeric formats of the hot path (Spec E §E.1):
//   cf32  — compute-domain complex float (8 B)
//   ci16  — fronthaul wire complex int16 (4 B, Spec B U-plane)
//   h16/half2c — host shim for the CUDA half2 FP16-complex H storage (4 B)
// plus the K0 (ci16→cf32) and K5 (cf32→ci16, saturating round) converts
// (Spec E §E.7), shared by the CPU golden model and the host build.

#include <cmath>
#include <cstdint>
#include <cstring>

namespace orca {

struct cf32 {
    float re, im;
};
static_assert(sizeof(cf32) == 8, "cf32 is 8 B");

struct ci16 {
    int16_t re, im;
};
static_assert(sizeof(ci16) == 4, "ci16 is 4 B");

// ---- K0: ci16 → cf32 -------------------------------------------------------

constexpr cf32 toCf32(ci16 v) {
    return cf32{static_cast<float>(v.re), static_cast<float>(v.im)};
}

// ---- K5: cf32 → ci16, saturating round -------------------------------------

// Round-to-nearest-even, independent of the process FP rounding mode (the
// GPU K5 kernel and the CPU golden model must agree bit-exactly).
inline float roundNearestEven(float x) {
    const float fl = std::floor(x);
    const float diff = x - fl;
    if (diff > 0.5f) return fl + 1.0f;
    if (diff < 0.5f) return fl;
    return (std::fmod(fl, 2.0f) == 0.0f) ? fl : fl + 1.0f;  // tie → even
}

inline int16_t satRoundI16(float x) {
    if (std::isnan(x)) return 0;
    const float r = roundNearestEven(x);
    if (r >= 32767.0f) return INT16_MAX;
    if (r <= -32768.0f) return INT16_MIN;
    return static_cast<int16_t>(r);
}

inline ci16 toCi16(cf32 v) {
    return ci16{satRoundI16(v.re), satRoundI16(v.im)};
}

// ---- IEEE binary16 host shim (CUDA half2 stand-in) --------------------------
// H is stored as FP16 complex on the GPU (Spec E §E.1); the host build and the
// golden model use this bit-exact software conversion.

struct h16 {
    uint16_t bits;
};

struct half2c {
    h16 re, im;
};
static_assert(sizeof(half2c) == 4, "half2c mirrors CUDA half2 (4 B)");

inline uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mant = x & 0x007FFFFFu;
    const uint32_t exp8 = (x >> 23) & 0xFFu;

    if (exp8 == 0xFFu)  // Inf / NaN
        return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x0200u : 0u));

    const int32_t exp5 = static_cast<int32_t>(exp8) - 127 + 15;
    if (exp5 >= 31)  // overflow → Inf
        return static_cast<uint16_t>(sign | 0x7C00u);

    if (exp5 <= 0) {  // subnormal half / underflow
        if (exp5 < -10) return static_cast<uint16_t>(sign);  // → ±0
        mant |= 0x00800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp5);
        uint32_t half = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }

    uint32_t half = (static_cast<uint32_t>(exp5) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
        ++half;  // a carry here correctly rolls into the exponent
    return static_cast<uint16_t>(sign | half);
}

inline float halfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp5 = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t x;
    if (exp5 == 0) {
        if (mant == 0) {
            x = sign;  // ±0
        } else {       // subnormal half → normal float
            uint32_t e = 127 - 15 + 1;
            while (!(mant & 0x400u)) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3FFu;
            x = sign | (e << 23) | (mant << 13);
        }
    } else if (exp5 == 31) {  // Inf / NaN
        x = sign | 0x7F800000u | (mant << 13);
    } else {
        x = sign | ((exp5 - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, sizeof f);
    return f;
}

inline half2c toHalf2(cf32 v) {
    return half2c{{floatToHalf(v.re)}, {floatToHalf(v.im)}};
}

inline cf32 toCf32(half2c v) {
    return cf32{halfToFloat(v.re.bits), halfToFloat(v.im.bits)};
}

}  // namespace orca
