#include "channel/cir_expand.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "common/dims.hpp"
#include "common/layout.hpp"

namespace orca {
namespace channel {

namespace {

constexpr double kPi           = 3.14159265358979323846;
constexpr double kSpeedOfLight = 299792458.0;  // m/s
constexpr uint32_t kMaxPaths   = 64;           // generous cap for stack arrays

// ULA per-element phasor: exp(-j·2π·(d/λ)·sin(az)).
inline cf32 ulaSteerUnit(double az, double dOverLambda) {
    const double phi = 2.0 * kPi * dOverLambda * std::sin(az);
    return cf32{static_cast<float>(std::cos(phi)),
                static_cast<float>(-std::sin(phi))};
}

// Build a steering vector: out[k] = unit^k (iterative cmul from out[0]=1).
inline void buildUla(cf32* out, uint32_t n, cf32 unit) {
    cf32 acc{1.0f, 0.0f};
    for (uint32_t i = 0; i < n; ++i) {
        out[i] = acc;
        acc = orca::cmul(acc, unit);
    }
}

// exp(-j·2π·f_sc·tau).
inline cf32 delayPhase(double f_sc, double tau) {
    const double phi = 2.0 * kPi * f_sc * tau;
    return cf32{static_cast<float>(std::cos(phi)),
                static_cast<float>(-std::sin(phi))};
}

}  // namespace

void expandLink(half2c* H_dl, uint32_t c, uint32_t u,
                const CirHeader& hdr, const CellDesc& cell,
                const UeArrayDesc& ue,
                const LinkBlockHeader* blk, const PathRecord* paths) {
    // Always zero the [c][u] slice first (handles noCoverage and fresh expansion).
    for (uint32_t rx = 0; rx < dims::numRx; ++rx)
        for (uint32_t tx = 0; tx < dims::numTx; ++tx)
            for (uint32_t sc = 0; sc < dims::numSc; ++sc)
                H_dl[layout::idxH(c, u, rx, tx, sc)] = half2c{{0}, {0}};

    if (!blk || (blk->flags & kLinkNoCoverage)) return;

    const uint32_t P = std::min({uint32_t{blk->numPaths}, hdr.pMax, kMaxPaths});
    if (P == 0) return;

    const double lambda   = kSpeedOfLight / hdr.carrierHz;
    const double dCellOvL = cell.elemSpacing / lambda;
    const double dUeOvL   = ue.elemSpacing   / lambda;

    // --- Precompute per-path quantities (stack, bounded by kMaxPaths) ----------
    cf32  gainAll[kMaxPaths];
    cf32  aTxAll[kMaxPaths][dims::numTx];  // kMaxPaths*64*8 = 32 KB
    cf32  aRxAll[kMaxPaths][dims::numRx];  // kMaxPaths*4*8  =  2 KB
    float tauAll[kMaxPaths];

    for (uint32_t p = 0; p < P; ++p) {
        const PathRecord& pr = paths[p];
        gainAll[p] = cf32{pr.gainRe, pr.gainIm};
        tauAll[p]  = pr.tau;
        buildUla(aTxAll[p], dims::numTx,
                 ulaSteerUnit(double{pr.aodAz}, dCellOvL));
        buildUla(aRxAll[p], dims::numRx,
                 ulaSteerUnit(double{pr.aoaAz}, dUeOvL));
    }

    // Precompute delay phases: dpAll[p][sc] = exp(-j·2π·f_sc·tau_p).
    // Heap (P * numSc * 8 B ≈ 16*3276*8 ≈ 419 KB for P=16).
    std::vector<cf32> dpAll(P * dims::numSc);
    for (uint32_t p = 0; p < P; ++p) {
        const double tau = double{tauAll[p]};
        for (uint32_t sc = 0; sc < dims::numSc; ++sc) {
            const double f_sc =
                hdr.carrierHz +
                (double{sc} - double{dims::numSc} / 2.0) * kScSpacingHz;
            dpAll[p * dims::numSc + sc] = delayPhase(f_sc, tau);
        }
    }

    // --- Accumulate H in cf32 per (rx,tx), convert to half2c once ------------
    // Per-(rx,tx) accumulator over sc: 3276 * 8 B = 26 KB stack.
    cf32 accSc[dims::numSc];

    for (uint32_t rx = 0; rx < dims::numRx; ++rx) {
        for (uint32_t tx = 0; tx < dims::numTx; ++tx) {
            // base_p = gain_p * aRx_p[rx] * aTx_p[tx]  — constant for all sc.
            cf32 baseArr[kMaxPaths];
            for (uint32_t p = 0; p < P; ++p)
                baseArr[p] = orca::cmul(gainAll[p],
                                        orca::cmul(aRxAll[p][rx], aTxAll[p][tx]));

            // Accumulate all path contributions per sc in float.
            for (uint32_t sc = 0; sc < dims::numSc; ++sc) {
                cf32 acc{0.0f, 0.0f};
                for (uint32_t p = 0; p < P; ++p)
                    acc = orca::cadd(acc,
                                     orca::cmul(baseArr[p],
                                                dpAll[p * dims::numSc + sc]));
                accSc[sc] = acc;
            }

            // Write to H_dl once — single half2c quantization per element.
            for (uint32_t sc = 0; sc < dims::numSc; ++sc)
                H_dl[layout::idxH(c, u, rx, tx, sc)] = toHalf2(accSc[sc]);
        }
    }
}

}  // namespace channel
}  // namespace orca
