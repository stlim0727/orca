// Stage-7-shape golden pipeline e2e (A4): one 2-cell SU symbol through the
// full DSP chain with consistent allocations and victim maps —
//   DL: x_dl → K1 precode → K2 channel-apply (+interference, rotors) → r_dl
//   UL: x_ul → K3 channel-apply (reciprocity) → K4 combine → z
// All values are powers of two and beam 0 of the DFT book is exactly 0.125
// per antenna, so every expected number is float-exact (CHECK ==, no tol).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "channel/doppler.hpp"
#include "common/dims.hpp"
#include "common/layout.hpp"
#include "dsp/awgn.hpp"
#include "dsp/channel_apply.hpp"
#include "dsp/combine.hpp"
#include "dsp/precode.hpp"
#include "scenario/beam_codebook.hpp"
#include "scenario/victim_map.hpp"

using namespace orca;

static int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

namespace {

struct Pipeline {
    BeamCodebook book = makeDftCodebook(64);
    std::vector<half2c> H =
        std::vector<half2c>(layout::elemsH, half2c{{0}, {0}});
    std::vector<cf32> xdl = std::vector<cf32>(layout::elemsXdl, cf32{0, 0});
    std::vector<cf32> y = std::vector<cf32>(layout::elemsY);
    std::vector<cf32> rdl = std::vector<cf32>(layout::elemsRdl);
    std::vector<cf32> xul = std::vector<cf32>(layout::elemsXul, cf32{0, 0});
    std::vector<cf32> rul = std::vector<cf32>(layout::elemsRul);
    std::vector<cf32> z = std::vector<cf32>(layout::elemsZ);
    std::vector<uint16_t> vDl =
        std::vector<uint16_t>(size_t{dims::C} * dims::numScP);
    std::vector<uint16_t> vUl =
        std::vector<uint16_t>(size_t{dims::C} * dims::numScP);
    std::vector<cf32> rot = std::vector<cf32>(dims::C * dims::U);
};

// Fill one (cell→ue) link with a constant gain over every (rx, tx, sc).
void fillLink(std::vector<half2c>& H, uint32_t c, uint32_t u, float gain) {
    const half2c g = toHalf2(cf32{gain, 0.0f});
    for (uint32_t rx = 0; rx < dims::numRx; ++rx)
        for (uint32_t tx = 0; tx < dims::numTx; ++tx)
            for (uint32_t sc = 0; sc < dims::numSc; ++sc)
                H[layout::idxH(c, u, rx, tx, sc)] = g;
}

}  // namespace

static void testDlPipeline(Pipeline& p) {
    // DL scheduling: cell0 → ue3 on sc [0, 1200); cell1 → ue17 on [600, 1800).
    Alloc allocs[2] = {};
    allocs[0] = Alloc{0, 3, 0, 1200, 0, 1, {0, 0, 0, 0}};     // beam 0
    allocs[1] = Alloc{1, 17, 600, 1200, 0, 1, {0, 0, 0, 0}};  // beam 0
    CHECK(buildVictimMap(allocs, 2, 0, p.vDl.data()) == 2);

    // Channel gains (all powers of two; exact in half and float).
    fillLink(p.H, 0, 3, 0.5f);    // serving cell0 → ue3
    fillLink(p.H, 1, 3, 0.25f);   // interferer cell1 → ue3
    fillLink(p.H, 0, 17, 0.125f); // interferer cell0 → ue17
    fillLink(p.H, 1, 17, 1.0f);   // serving cell1 → ue17

    // Rotors: every cell0 link 1; every cell1 link j (Δφ=π/2 at symbol 1).
    for (uint32_t u = 0; u < dims::U; ++u) {
        p.rot[dopplerIdx(0, u)] = cf32{1.0f, 0.0f};
        p.rot[dopplerIdx(1, u)] = cf32{0.0f, 1.0f};
    }

    // Layer IQ: cell0 layer0 = 2, cell1 layer0 = 4 (constant over sc).
    for (uint32_t sc = 0; sc < dims::numSc; ++sc) {
        p.xdl[layout::idxXdl(0, 0, sc)] = cf32{2.0f, 0.0f};
        p.xdl[layout::idxXdl(1, 0, sc)] = cf32{4.0f, 0.0f};
    }

    // K1: beam 0 is exactly 0.125 per antenna → y[tx] = 0.25 / 0.5.
    CHECK(dsp::precodeGolden(p.xdl.data(), p.book, allocs, 2, p.y.data()) == 0);
    CHECK(p.y[layout::idxY(0, 11, 100)].re == 0.25f);
    CHECK(p.y[layout::idxY(1, 50, 700)].re == 0.5f);
    CHECK(p.y[layout::idxY(1, 50, 100)].re == 0.0f);  // cell1 idle below 600

    // K2, noiseless.
    dsp::channelApplyDlGolden(p.H.data(), p.y.data(), p.vDl.data(),
                              p.rot.data(), 0, 0, 0.0f, p.rdl.data());

    // ue3 in cell0-only region (sc<600): only its serving cell radiates →
    //   Σ_tx 0.5·0.25 ·64 = 8, rotor 1 → 8 + 0j.
    CHECK(p.rdl[layout::idxRdl(3, 0, 100)].re == 8.0f);
    CHECK(p.rdl[layout::idxRdl(3, 0, 100)].im == 0.0f);

    // ue3 in the overlap (600 ≤ sc < 1200): + interference from cell1:
    //   Σ_tx 0.25·0.5 ·64 = 8, rotor j → +8j. Total 8 + 8j.
    CHECK(p.rdl[layout::idxRdl(3, 2, 800)].re == 8.0f);
    CHECK(p.rdl[layout::idxRdl(3, 2, 800)].im == 8.0f);

    // ue17 in the overlap: serving cell1 (1.0·0.5·64 = 32, rotor j → 32j)
    //   + interference from cell0 (0.125·0.25·64 = 2, rotor 1 → 2).
    CHECK(p.rdl[layout::idxRdl(17, 1, 800)].re == 2.0f);
    CHECK(p.rdl[layout::idxRdl(17, 1, 800)].im == 32.0f);

    // ue17 where only cell1 schedules (1200 ≤ sc < 1800): no cell0 radiation
    // on those sc → pure serving term 32j.
    CHECK(p.rdl[layout::idxRdl(17, 1, 1500)].re == 0.0f);
    CHECK(p.rdl[layout::idxRdl(17, 1, 1500)].im == 32.0f);

    // Nobody scheduled past 1800 → exact zero.
    CHECK(p.rdl[layout::idxRdl(3, 0, 2000)].re == 0.0f);
    CHECK(p.rdl[layout::idxRdl(17, 0, 2000)].re == 0.0f);
}

static void testUlPipeline(Pipeline& p) {
    // UL scheduling: only cell0 serves ue3 on sc [0, 600). Under all-to-all
    // UL channel-apply, every cell hears that UE through its cross-link;
    // only cell0 is combined below because it is the sole UL allocation.
    Alloc ul{0, 3, 0, 600, 1, 1, {0, 0, 0, 0}};
    CHECK(buildVictimMap(&ul, 1, 1, p.vUl.data()) == 1);

    // UE transmit IQ: ue3 antennas {1, 2j} (constant over sc).
    for (uint32_t sc = 0; sc < dims::numSc; ++sc) {
        p.xul[layout::idxXul(3, 0, sc)] = cf32{1.0f, 0.0f};
        p.xul[layout::idxXul(3, 1, sc)] = cf32{0.0f, 2.0f};
    }

    // K3 (reciprocity):
    //   r_ul[0][rxRu][sc] = 0.5·1 + 0.5·2j = 0.5 + 1j on every RU antenna.
    //   r_ul[1][rxRu][sc] = j·(0.25·1 + 0.25·2j) = -0.5 + 0.25j,
    //   the all-to-all cross-link heard at cell1.
    const uint8_t ueTxToRx[2] = {0, 1};
    CHECK(dsp::channelApplyUlGolden(p.H.data(), p.xul.data(), p.vUl.data(),
                                    p.rot.data(), ueTxToRx, 0, 0, 0.0f,
                                    p.rul.data()));
    CHECK(p.rul[layout::idxRul(0, 33, 100)].re == 0.5f);
    CHECK(p.rul[layout::idxRul(0, 33, 100)].im == 1.0f);
    CHECK(p.rul[layout::idxRul(1, 33, 100)].re == -0.5f);
    CHECK(p.rul[layout::idxRul(1, 33, 100)].im == 0.25f);

    // K4 with beam 0: z = Σ_rx 0.125·(0.5 + 1j) = 64·0.125·(0.5+1j) = 4 + 8j.
    CHECK(dsp::combineGolden(p.rul.data(), p.book, &ul, 1, p.z.data()) == 0);
    CHECK(p.z[layout::idxZ(0, 0, 100)].re == 4.0f);
    CHECK(p.z[layout::idxZ(0, 0, 100)].im == 8.0f);
    CHECK(p.z[layout::idxZ(0, 0, 700)].re == 0.0f);  // outside the UL alloc
    CHECK(p.z[layout::idxZ(1, 0, 100)].re == 0.0f);
}

static void testNoiseDeterminism(Pipeline& p) {
    // The full noisy DL leg is bit-reproducible, and equals the noiseless
    // result plus exactly the keyed AWGN draw.
    std::vector<cf32> r1(layout::elemsRdl), r2(layout::elemsRdl);
    dsp::channelApplyDlGolden(p.H.data(), p.y.data(), p.vDl.data(),
                              p.rot.data(), 777, 42, 0.25f, r1.data());
    dsp::channelApplyDlGolden(p.H.data(), p.y.data(), p.vDl.data(),
                              p.rot.data(), 777, 42, 0.25f, r2.data());
    CHECK(std::memcmp(r1.data(), r2.data(),
                      sizeof(cf32) * layout::elemsRdl) == 0);

    const cf32 nz = dsp::awgnSample(777, dsp::awgnSubsequence(0, 3, 0),
                                    42ull * dims::numScP + 100, 0.25f);
    CHECK(r1[layout::idxRdl(3, 0, 100)].re == 8.0f + nz.re);
    CHECK(r1[layout::idxRdl(3, 0, 100)].im == 0.0f + nz.im);
}

int main() {
    Pipeline p;
    testDlPipeline(p);
    testUlPipeline(p);
    testNoiseDeterminism(p);

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("test_golden_pipeline: all checks passed");
    return EXIT_SUCCESS;
}
