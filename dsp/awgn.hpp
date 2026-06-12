#pragma once
// AWGN, CPU golden model (Spec E §E.7; Stage 2). Deterministic, stateless:
// every complex noise sample is a pure function of
//   (seed, subsequence = linearIdx(dir, ue, rx), sampleIdx = symbolCtr·numScP + sc)
// via one Philox block per sample (common/philox.hpp). The CUDA kernels add
// the identical draw with curand_init(seed, subsequence, 4·sampleIdx).

#include <cstdint>

#include "common/complex.hpp"
#include "common/philox.hpp"

namespace orca {
namespace dsp {

// The per-stream subsequence of the §E.7 keying. Field widths: rx < 256,
// ue < 2^24 — ample for numRx=4 / U=32; dir occupies bits 32+ so the three
// fields never collide.
constexpr uint64_t awgnSubsequence(uint8_t dir, uint32_t ue, uint32_t rx) {
    return (uint64_t{dir} << 32) | (uint64_t{ue} << 8) | uint64_t{rx};
}

// One complex N(0, std²) draw for (seed, subsequence, sampleIdx).
inline cf32 awgnSample(uint64_t seed, uint64_t subsequence, uint64_t sampleIdx,
                       float noiseStd) {
    float n0, n1;
    philoxNormal2(philoxBlock(seed, subsequence, sampleIdx), n0, n1);
    return cf32{n0 * noiseStd, n1 * noiseStd};
}

// buf[i] += noise for i in [0, n): sampleIdx = sampleBase + i. Stateless and
// restartable at any i (golden-model contract).
void addAwgnGolden(cf32* buf, uint32_t n, uint64_t seed, uint64_t subsequence,
                   uint64_t sampleBase, float noiseStd);

}  // namespace dsp
}  // namespace orca
