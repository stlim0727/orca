// Stage-3 golden test: victim map build, Doppler rotor, K2 (DL channel-apply
// with all-to-all interference) and K3 (UL via reciprocity) against direct
// hand computation on small sc ranges, plus noise determinism/keying.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "channel/doppler.hpp"
#include "common/dims.hpp"
#include "common/layout.hpp"
#include "dsp/awgn.hpp"
#include "dsp/channel_apply.hpp"
#include "scenario/victim_map.hpp"
#include "tests/check.hpp"

using namespace orca;

static bool close(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) <= tol;
}

static void testVictimMap() {
    std::vector<uint16_t> v(size_t{dims::C} * dims::numScP);

    Alloc a[3] = {};
    a[0] = Alloc{0, 7, 0, 96, 0, 2, {1, 2, 0, 0}};       // cell0 DL sc[0,96)
    a[1] = Alloc{1, 20, 48, 60, 0, 1, {3, 0, 0, 0}};     // cell1 DL sc[48,108)
    a[2] = Alloc{0, 9, 200, 12, 1, 1, {4, 0, 0, 0}};     // cell0 UL — filtered
    CHECK(buildVictimMap(a, 3, /*dir=*/0, v.data()) == 2);

    CHECK(v[0] == 7 && v[95] == 7 && v[96] == kNoUe);
    CHECK(v[size_t{1} * dims::numScP + 47] == kNoUe);
    CHECK(v[size_t{1} * dims::numScP + 48] == 20);
    CHECK(v[size_t{1} * dims::numScP + 107] == 20);
    CHECK(v[200] == kNoUe);  // the UL alloc did not land in the DL map

    // UL direction picks up only the UL alloc.
    CHECK(buildVictimMap(a, 3, /*dir=*/1, v.data()) == 1);
    CHECK(v[200] == 9 && v[211] == 9 && v[0] == kNoUe);

    // Malformed allocations are skipped.
    Alloc bad{2 /*cell OOR*/, 1, 0, 12, 0, 1, {0, 0, 0, 0}};
    CHECK(buildVictimMap(&bad, 1, 0, v.data()) == 0);
}

static void testDoppler() {
    // symbolCtr 0 → unit rotor; phase accumulates linearly; magnitude 1.
    CHECK(dopplerRotor(0.1, 0).re == 1.0f && dopplerRotor(0.1, 0).im == 0.0f);
    const cf32 r1 = dopplerRotor(0.1, 1);
    CHECK(close(r1.re, std::cos(0.1f), 1e-6f) &&
          close(r1.im, std::sin(0.1f), 1e-6f));
    const cf32 r100 = dopplerRotor(0.1, 100);
    CHECK(close(r100.re * r100.re + r100.im * r100.im, 1.0f, 1e-5f));
    // Long-counter accumulation stays exact in double (10 rad ≫ float ulp).
    const cf32 big = dopplerRotor(1e-6, 10000000);
    CHECK(close(big.re, std::cos(10.0f), 1e-5f));
}

static void testChannelApplyDl() {
    // Full tensors, but only a small sc window is scheduled/verified.
    std::vector<half2c> H(layout::elemsH, half2c{{0}, {0}});
    std::vector<cf32> y(layout::elemsY, cf32{0, 0});
    std::vector<cf32> rdl(layout::elemsRdl);
    std::vector<uint16_t> victim(size_t{dims::C} * dims::numScP, kNoUe);
    std::vector<cf32> rot(dims::C * dims::U, cf32{1.0f, 0.0f});

    // Schedule UE 3 on cell 0 and UE 17 on cell 1, sc [100, 112).
    for (uint32_t sc = 100; sc < 112; ++sc) {
        victim[sc] = 3;
        victim[size_t{1} * dims::numScP + sc] = 17;
    }

    // Sparse H/y so the expected sum is hand-computable. Half-exact values.
    auto setH = [&](uint32_t c, uint32_t u, uint32_t rx, uint32_t tx,
                    uint32_t sc, float re, float im) {
        H[layout::idxH(c, u, rx, tx, sc)] = toHalf2(cf32{re, im});
    };
    const uint32_t sc0 = 105;
    setH(0, 3, 0, 5, sc0, 0.5f, 0.0f);    // serving link, cell0→ue3
    setH(0, 3, 0, 9, sc0, 0.25f, 0.0f);
    setH(1, 3, 0, 2, sc0, 0.0f, 1.0f);    // interferer link, cell1→ue3
    y[layout::idxY(0, 5, sc0)] = cf32{2.0f, 0.0f};
    y[layout::idxY(0, 9, sc0)] = cf32{4.0f, 0.0f};
    y[layout::idxY(1, 2, sc0)] = cf32{1.0f, 1.0f};

    // Distinct rotors per (cell, ue): serving rotor 1, interferer rotor j.
    rot[dopplerIdx(1, 3)] = cf32{0.0f, 1.0f};

    dsp::channelApplyDlGolden(H.data(), y.data(), victim.data(), rot.data(),
                              /*seed=*/0, /*symbolCtr=*/0, /*noiseStd=*/0.0f,
                              rdl.data());

    // Expected at (ue3, rx0, sc0):
    //   serving:    0.5·2 + 0.25·4 = 2.0           (rotor 1)
    //   interferer: j·(1+j) = -1 + j, rotor j →  j·(-1+j) = -1 - j ... compute:
    //   partial_c1 = H·y = j·(1+j) = (-1 + j); rot j → j·(-1+j) = (-j -1)
    const cf32 got = rdl[layout::idxRdl(3, 0, sc0)];
    CHECK(close(got.re, 2.0f - 1.0f));
    CHECK(close(got.im, -1.0f));

    // Other rx of the same victim: zero links → exactly 0 (no noise).
    CHECK(rdl[layout::idxRdl(3, 1, sc0)].re == 0.0f);
    // Unscheduled UEs and sc outside the window stay zero.
    CHECK(rdl[layout::idxRdl(0, 0, sc0)].re == 0.0f);
    CHECK(rdl[layout::idxRdl(3, 0, 99)].re == 0.0f);

    // With noise: the scheduled victim's sample is the zero-noise value plus
    // exactly the keyed AWGN draw; deterministic across calls.
    std::vector<cf32> rn1(layout::elemsRdl), rn2(layout::elemsRdl);
    dsp::channelApplyDlGolden(H.data(), y.data(), victim.data(), rot.data(),
                              1234, 77, 0.1f, rn1.data());
    dsp::channelApplyDlGolden(H.data(), y.data(), victim.data(), rot.data(),
                              1234, 77, 0.1f, rn2.data());
    const cf32 nz = dsp::awgnSample(1234, dsp::awgnSubsequence(0, 3, 0),
                                    77ull * dims::numScP + sc0, 0.1f);
    CHECK(rn1[layout::idxRdl(3, 0, sc0)].re == got.re + nz.re);
    CHECK(rn1[layout::idxRdl(3, 0, sc0)].im == got.im + nz.im);
    CHECK(rn1[layout::idxRdl(3, 0, sc0)].re == rn2[layout::idxRdl(3, 0, sc0)].re);
}

static void testChannelApplyUl() {
    std::vector<half2c> H(layout::elemsH, half2c{{0}, {0}});
    std::vector<cf32> xul(layout::elemsXul, cf32{0, 0});
    std::vector<cf32> rul(layout::elemsRul);
    std::vector<uint16_t> victim(size_t{dims::C} * dims::numScP, kNoUe);
    std::vector<cf32> rot(dims::C * dims::U, cf32{1.0f, 0.0f});
    const uint8_t ueTxToRx[2] = {0, 1};

    const uint32_t sc0 = 40;
    victim[sc0] = 5;  // cell 0 hears UE 5 (cell 1 idle on sc0)

    // Reciprocity wiring: H_ul[u=5][c=0][rxRu=12][t=0][sc0] must read
    // H_dl[c=0][u=5][rx=ueTxToRx[0]=0][tx=12][sc0].
    H[layout::idxH(0, 5, 0, 12, sc0)] = toHalf2(cf32{0.5f, 0.0f});
    H[layout::idxH(0, 5, 1, 12, sc0)] = toHalf2(cf32{0.0f, 0.25f});  // t=1
    xul[layout::idxXul(5, 0, sc0)] = cf32{2.0f, 0.0f};
    xul[layout::idxXul(5, 1, sc0)] = cf32{0.0f, 4.0f};

    CHECK(dsp::channelApplyUlGolden(H.data(), xul.data(), victim.data(),
                                    rot.data(), ueTxToRx, 0, 0, 0.0f,
                                    rul.data()));

    // Expected at (cell0, rxRu=12, sc0): 0.5·2 + (0.25j)·(4j) = 1 − 1 = 0…
    // compute: (0.25j)·(4j) = j·j·1 = −1 → total = 1 − 1 + j·0 = 0.
    const cf32 got = rul[layout::idxRul(0, 12, sc0)];
    CHECK(close(got.re, 1.0f - 1.0f));
    CHECK(close(got.im, 0.0f));

    // Unwired RU antenna: zero (no noise in this run).
    CHECK(rul[layout::idxRul(0, 13, sc0)].re == 0.0f);
    // Cell 1 has no contributor on sc0 → exactly zero.
    CHECK(rul[layout::idxRul(1, 12, sc0)].re == 0.0f);

    // Noise lands on every RU antenna stream, even with no contributor.
    std::vector<cf32> rn(layout::elemsRul);
    CHECK(dsp::channelApplyUlGolden(H.data(), xul.data(), victim.data(),
                                    rot.data(), ueTxToRx, 99, 5, 0.2f,
                                    rn.data()));
    const cf32 nz = dsp::awgnSample(99, dsp::awgnSubsequence(1, 1, 30),
                                    5ull * dims::numScP + 7, 0.2f);
    CHECK(rn[layout::idxRul(1, 30, 7)].re == nz.re &&
          rn[layout::idxRul(1, 30, 7)].im == nz.im);

    // A bad ueTxToRx mapping is refused (false, output zeroed) — never an
    // out-of-row H read.
    const uint8_t badMap[2] = {0, dims::numRx};
    CHECK(!dsp::channelApplyUlGolden(H.data(), xul.data(), victim.data(),
                                     rot.data(), badMap, 0, 0, 0.5f,
                                     rn.data()));
    CHECK(rn[layout::idxRul(0, 12, sc0)].re == 0.0f &&
          rn[layout::idxRul(1, 30, 7)].re == 0.0f);
}

int main() {
    testVictimMap();
    testDoppler();
    testChannelApplyDl();
    testChannelApplyUl();

    return orca::test::report("test_channel_apply");
}
