// Stage-2 golden test: Philox4x32-10 against the published Random123
// known-answer vectors (the hard bit-exactness anchor of the Spec E §E.7
// noise contract), keyed-block mapping, Box–Muller normal statistics, and
// the AWGN golden (determinism, additivity, restartability).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "common/philox.hpp"
#include "dsp/awgn.hpp"

using namespace orca;

static int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

static void testKnownAnswers() {
    // Random123 v1.x kat_vectors, philox4x32 (10 rounds).
    {
        const PhiloxBlock b = philox4x32_10(0, 0, 0, 0, 0, 0);
        CHECK(b.x[0] == 0x6627e8d5u && b.x[1] == 0xe169c58du &&
              b.x[2] == 0xbc57ac4cu && b.x[3] == 0x9b00dbd8u);
    }
    {
        const PhiloxBlock b =
            philox4x32_10(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
                          0xffffffffu, 0xffffffffu);
        CHECK(b.x[0] == 0x408f276du && b.x[1] == 0x41c83b0eu &&
              b.x[2] == 0xa20bc7c6u && b.x[3] == 0x6d5451fdu);
    }
    {
        // π-digit counter/key vector.
        const PhiloxBlock b =
            philox4x32_10(0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u,
                          0xa4093822u, 0x299f31d0u);
        CHECK(b.x[0] == 0xd16cfe09u && b.x[1] == 0x94fdccebu &&
              b.x[2] == 0x5001e420u && b.x[3] == 0x24126ea1u);
    }
}

static void testKeyedBlock() {
    // philoxBlock maps (seed, subseq, sampleIdx) onto (key, counter-hi,
    // counter-lo) exactly as documented.
    const uint64_t seed = 0x0123456789ABCDEFull;
    const uint64_t sub = 0xFEDCBA9876543210ull;
    const uint64_t idx = 0x1122334455667788ull;
    const PhiloxBlock a = philoxBlock(seed, sub, idx);
    const PhiloxBlock b =
        philox4x32_10(0x55667788u, 0x11223344u, 0x76543210u, 0xFEDCBA98u,
                      0x89ABCDEFu, 0x01234567u);
    CHECK(a.x[0] == b.x[0] && a.x[1] == b.x[1] && a.x[2] == b.x[2] &&
          a.x[3] == b.x[3]);

    // Reproducible; distinct inputs give distinct blocks.
    const PhiloxBlock a2 = philoxBlock(seed, sub, idx);
    CHECK(a.x[0] == a2.x[0] && a.x[3] == a2.x[3]);
    const PhiloxBlock c = philoxBlock(seed, sub, idx + 1);
    const PhiloxBlock d = philoxBlock(seed, sub + 1, idx);
    const PhiloxBlock e = philoxBlock(seed + 1, sub, idx);
    CHECK(c.x[0] != a.x[0] || c.x[1] != a.x[1]);
    CHECK(d.x[0] != a.x[0] || d.x[1] != a.x[1]);
    CHECK(e.x[0] != a.x[0] || e.x[1] != a.x[1]);
}

static void testUniformAndNormal() {
    // philoxUniform is in (0, 1]: never 0 (log-safe), 0xFFFFFFFF → 1.0.
    CHECK(philoxUniform(0u) > 0.0);
    CHECK(philoxUniform(0xFFFFFFFFu) == 1.0);

    // Normal pair statistics over 100k draws: mean ≈ 0, var ≈ 1.
    const uint64_t seed = 42, sub = 7;
    double sum = 0.0, sumSq = 0.0;
    const int kN = 100000;
    for (int i = 0; i < kN; ++i) {
        float n0, n1;
        philoxNormal2(philoxBlock(seed, sub, static_cast<uint64_t>(i)), n0, n1);
        sum += n0 + n1;
        sumSq += double{n0} * n0 + double{n1} * n1;
    }
    const double mean = sum / (2.0 * kN);
    const double var = sumSq / (2.0 * kN) - mean * mean;
    CHECK(std::fabs(mean) < 0.01);
    CHECK(std::fabs(var - 1.0) < 0.02);
}

static void testAwgnGolden() {
    const uint64_t seed = 1234;
    const uint64_t sub = dsp::awgnSubsequence(0, 5, 2);
    constexpr uint32_t kN = 256;

    // Deterministic: same key → identical floats.
    cf32 a[kN] = {}, b[kN] = {};
    dsp::addAwgnGolden(a, kN, seed, sub, 1000, 0.5f);
    dsp::addAwgnGolden(b, kN, seed, sub, 1000, 0.5f);
    for (uint32_t i = 0; i < kN; ++i)
        CHECK(a[i].re == b[i].re && a[i].im == b[i].im);

    // Additive on top of existing data.
    cf32 c[kN];
    for (uint32_t i = 0; i < kN; ++i) c[i] = cf32{1.0f, -2.0f};
    dsp::addAwgnGolden(c, kN, seed, sub, 1000, 0.5f);
    for (uint32_t i = 0; i < kN; ++i) {
        CHECK(c[i].re == 1.0f + a[i].re);
        CHECK(c[i].im == -2.0f + a[i].im);
    }

    // Stateless restart: generating [0,N) equals [0,k) then [k,N).
    cf32 whole[kN] = {}, split[kN] = {};
    dsp::addAwgnGolden(whole, kN, seed, sub, 0, 1.0f);
    dsp::addAwgnGolden(split, 100, seed, sub, 0, 1.0f);
    dsp::addAwgnGolden(split + 100, kN - 100, seed, sub, 100, 1.0f);
    for (uint32_t i = 0; i < kN; ++i)
        CHECK(whole[i].re == split[i].re && whole[i].im == split[i].im);

    // std = 0 → untouched.
    cf32 z[4] = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    dsp::addAwgnGolden(z, 4, seed, sub, 0, 0.0f);
    CHECK(z[0].re == 1.0f && z[3].im == 8.0f);

    // Subsequence separates streams (different (dir,ue,rx) → different noise).
    cf32 s2[kN] = {};
    dsp::addAwgnGolden(s2, kN, seed, dsp::awgnSubsequence(1, 5, 2), 1000, 0.5f);
    int same = 0;
    for (uint32_t i = 0; i < kN; ++i)
        if (s2[i].re == a[i].re && s2[i].im == a[i].im) ++same;
    CHECK(same == 0);
}

int main() {
    testKnownAnswers();
    testKeyedBlock();
    testUniformAndNormal();
    testAwgnGolden();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("test_philox: all checks passed");
    return EXIT_SUCCESS;
}
