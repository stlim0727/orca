#pragma once
// Philox4x32-10 counter-based RNG (Salmon et al., SC'11 / Random123) — the
// CPU side of the Spec E §E.7 noise contract. The GPU kernels use cuRAND's
// curandStatePhilox4_32_10_t; this implementation reproduces the same
// keyed-counter → 4×u32 block function, so the contract is:
//
//   * INTEGER LAYER (hard guarantee): block(key=seed, counter) is bit-exact
//     between this code and cuRAND/Random123 (verified against the published
//     known-answer vectors in tests).
//   * NORMAL LAYER: normal2() applies the documented Box–Muller below in
//     double precision. The CUDA kernel must implement the same formula;
//     cross-device agreement is then within libm rounding (a few ULP), to be
//     verified with tolerance in the GPU phase — bit-exactness is anchored
//     at the integer layer.
//
// Keying (refines the Spec E §E.7 sketch): one whole 4-output counter block
// per sample pair — counter = {lo64 = sampleIdx, hi64 = subsequence}, where
// subsequence = linearIdx(dir, ue, rx) and sampleIdx = symbolCtr·numScP + sc.
// Equivalent cuRAND call: curand_init(seed, subsequence, 4·sampleIdx, &st).
// (The sketch's offset = sc would overlap adjacent subcarriers' draw pairs;
// block granularity keeps every (stream, sample) draw pair disjoint.)

#include <cmath>
#include <cstdint>

namespace orca {

struct PhiloxBlock {
    uint32_t x[4];
};

namespace detail {

constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;  // Weyl constants
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;

inline uint32_t mulhi32(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 32);
}
inline uint32_t mullo32(uint32_t a, uint32_t b) { return a * b; }

}  // namespace detail

// One Philox4x32-10 block: 128-bit counter + 64-bit key → 4×u32.
inline PhiloxBlock philox4x32_10(uint32_t c0, uint32_t c1, uint32_t c2,
                                 uint32_t c3, uint32_t k0, uint32_t k1) {
    using namespace detail;
    for (int round = 0; round < 10; ++round) {
        const uint32_t hi0 = mulhi32(kPhiloxM0, c0);
        const uint32_t lo0 = mullo32(kPhiloxM0, c0);
        const uint32_t hi1 = mulhi32(kPhiloxM1, c2);
        const uint32_t lo1 = mullo32(kPhiloxM1, c2);
        const uint32_t n0 = hi1 ^ c1 ^ k0;
        const uint32_t n1 = lo1;
        const uint32_t n2 = hi0 ^ c3 ^ k1;
        const uint32_t n3 = lo0;
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
        k0 += kPhiloxW0;
        k1 += kPhiloxW1;
    }
    return PhiloxBlock{{c0, c1, c2, c3}};
}

// The keyed block of the noise contract: seed = 64-bit run seed (key),
// counter = {lo64 = sampleIdx, hi64 = subsequence}.
inline PhiloxBlock philoxBlock(uint64_t seed, uint64_t subsequence,
                               uint64_t sampleIdx) {
    return philox4x32_10(static_cast<uint32_t>(sampleIdx),
                         static_cast<uint32_t>(sampleIdx >> 32),
                         static_cast<uint32_t>(subsequence),
                         static_cast<uint32_t>(subsequence >> 32),
                         static_cast<uint32_t>(seed),
                         static_cast<uint32_t>(seed >> 32));
}

// u32 → uniform in (0, 1]: (x + 1) · 2^-32 — never 0, so log() is safe.
// This exact mapping is part of the contract: the CUDA path must apply the
// same integer→uniform formula to the raw Philox outputs (NOT
// curand_uniform, whose mapping differs).
inline double philoxUniform(uint32_t x) {
    return (static_cast<double>(x) + 1.0) * (1.0 / 4294967296.0);
}

// Box–Muller on outputs 0 and 1 of the block (documented contract): returns
// a standard-normal pair. Double intermediates; results cast to float.
inline void philoxNormal2(const PhiloxBlock& b, float& n0, float& n1) {
    const double u1 = philoxUniform(b.x[0]);
    const double u2 = philoxUniform(b.x[1]);
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double t = 6.283185307179586476925286766559 * u2;  // 2π·u2
    n0 = static_cast<float>(r * std::cos(t));
    n1 = static_cast<float>(r * std::sin(t));
}

}  // namespace orca
