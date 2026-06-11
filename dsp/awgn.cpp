#include "dsp/awgn.hpp"

namespace orca {
namespace dsp {

void addAwgnGolden(cf32* buf, uint32_t n, uint64_t seed, uint64_t subsequence,
                   uint64_t sampleBase, float noiseStd) {
    if (noiseStd == 0.0f) return;
    for (uint32_t i = 0; i < n; ++i) {
        const cf32 nz = awgnSample(seed, subsequence, sampleBase + i, noiseStd);
        buf[i].re += nz.re;
        buf[i].im += nz.im;
    }
}

}  // namespace dsp
}  // namespace orca
