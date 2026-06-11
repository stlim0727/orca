#include "dsp/channel_apply.hpp"

#include <cstring>

#include "channel/doppler.hpp"
#include "common/dims.hpp"
#include "common/layout.hpp"
#include "dsp/awgn.hpp"
#include "scenario/victim_map.hpp"

namespace orca {
namespace dsp {

void channelApplyDlGolden(const half2c* H, const cf32* y,
                          const uint16_t* victim, const cf32* rot,
                          uint64_t seed, uint64_t symbolCtr, float noiseStd,
                          cf32* r_dl) {
    std::memset(r_dl, 0, sizeof(cf32) * layout::elemsRdl);

    for (uint32_t c = 0; c < dims::C; ++c) {
        const uint16_t* vrow = victim + size_t{c} * dims::numScP;
        for (uint32_t sc = 0; sc < dims::numSc; ++sc) {
            const uint16_t u = vrow[sc];
            if (u == kNoUe || u >= dims::U) continue;
            for (uint32_t rx = 0; rx < dims::numRx; ++rx) {
                cf32 acc =
                    noiseStd == 0.0f
                        ? cf32{0.0f, 0.0f}
                        : awgnSample(seed, awgnSubsequence(0, u, rx),
                                     symbolCtr * dims::numScP + sc, noiseStd);
                for (uint32_t c2 = 0; c2 < dims::C; ++c2) {
                    cf32 partial{0.0f, 0.0f};
                    for (uint32_t tx = 0; tx < dims::numTx; ++tx) {
                        const cf32 h =
                            toCf32(H[layout::idxH(c2, u, rx, tx, sc)]);
                        partial = cmac(partial, h,
                                       y[layout::idxY(c2, tx, sc)]);
                    }
                    acc = cmac(acc, rot[dopplerIdx(c2, u)], partial);
                }
                r_dl[layout::idxRdl(u, rx, sc)] = acc;
            }
        }
    }
}

bool channelApplyUlGolden(const half2c* H, const cf32* x_ul,
                          const uint16_t* victim, const cf32* rot,
                          const uint8_t* ueTxToRx, uint64_t seed,
                          uint64_t symbolCtr, float noiseStd, cf32* r_ul) {
    // A bad antenna mapping would index outside the H row — refuse up front.
    for (uint32_t t = 0; t < dims::numUeTx; ++t) {
        if (ueTxToRx[t] >= dims::numRx) {
            std::memset(r_ul, 0, sizeof(cf32) * layout::elemsRul);
            return false;
        }
    }
    for (uint32_t c = 0; c < dims::C; ++c) {
        for (uint32_t rxRu = 0; rxRu < dims::numTx; ++rxRu) {
            for (uint32_t sc = 0; sc < dims::numSc; ++sc) {
                cf32 acc =
                    noiseStd == 0.0f
                        ? cf32{0.0f, 0.0f}
                        : awgnSample(seed, awgnSubsequence(1, c, rxRu),
                                     symbolCtr * dims::numScP + sc, noiseStd);
                // Contributors on sc: each cell's scheduled UE (SU all-to-all).
                for (uint32_t c2 = 0; c2 < dims::C; ++c2) {
                    const uint16_t u = victim[size_t{c2} * dims::numScP + sc];
                    if (u == kNoUe || u >= dims::U) continue;
                    cf32 partial{0.0f, 0.0f};
                    for (uint32_t t = 0; t < dims::numUeTx; ++t) {
                        // Reciprocity: H_ul[u][c][rxRu][t][sc] =
                        // H_dl[c][u][ueTxToRx[t]][rxRu][sc] (Spec E §E.6).
                        const cf32 h = toCf32(
                            H[layout::idxH(c, u, ueTxToRx[t], rxRu, sc)]);
                        partial = cmac(
                            partial, h,
                            x_ul[layout::idxXul(u, t, sc)]);
                    }
                    acc = cmac(acc, rot[dopplerIdx(c, u)], partial);
                }
                r_ul[layout::idxRul(c, rxRu, sc)] = acc;
            }
        }
    }
    return true;
}

}  // namespace dsp
}  // namespace orca
