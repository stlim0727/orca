// Stage-2 golden test: beam codebook properties and the K1 precode golden —
// gather + per-allocation GEMV against direct hand computation, rank
// accumulation, range zeroing, and malformed-allocation rejection.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/dims.hpp"
#include "common/layout.hpp"
#include "dsp/precode.hpp"
#include "scenario/beam_codebook.hpp"
#include "tests/check.hpp"

using namespace orca;

static bool close(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

static void testCodebook() {
    const BeamCodebook book = makeDftCodebook(1024);
    CHECK(book.numBeams() == 1024);
    CHECK(book.beam(0) != nullptr);
    CHECK(book.beam(1023) != nullptr);
    CHECK(book.beam(1024) == nullptr);  // out of range

    // Beam 0 is the uniform broadside beam: every element 1/√numTx.
    const float inv = 1.0f / std::sqrt(static_cast<float>(dims::numTx));
    for (uint32_t n = 0; n < dims::numTx; ++n) {
        CHECK(close(book.beam(0)[n].re, inv));
        CHECK(close(book.beam(0)[n].im, 0.0f));
    }

    // Every beam has unit L2 norm.
    const uint16_t samples[] = {1, 511, 1023};
    for (uint16_t b : samples) {
        double norm = 0.0;
        for (uint32_t n = 0; n < dims::numTx; ++n) {
            const cf32 w = book.beam(b)[n];
            norm += double{w.re} * w.re + double{w.im} * w.im;
        }
        CHECK(std::fabs(norm - 1.0) < 1e-6);
    }

    // Deterministic regeneration.
    const BeamCodebook again = makeDftCodebook(1024);
    CHECK(again.beam(777)[33].re == book.beam(777)[33].re);
    CHECK(again.beam(777)[33].im == book.beam(777)[33].im);
}

static void testPrecodeGolden() {
    const BeamCodebook book = makeDftCodebook(256);
    std::vector<cf32> x(layout::elemsXdl, cf32{0, 0});
    std::vector<cf32> y(layout::elemsY, cf32{99, 99});  // proves zeroing

    // Deterministic x pattern.
    for (uint32_t c = 0; c < dims::C; ++c)
        for (uint32_t l = 0; l < dims::rankMax; ++l)
            for (uint32_t sc = 0; sc < dims::numSc; ++sc)
                x[layout::idxXdl(c, l, sc)] =
                    cf32{static_cast<float>((sc % 13) + l), static_cast<float>(c) - 0.5f};

    // One rank-2 DL allocation on cell 1, sc [1200, 1296).
    Alloc a{};
    a.cell = 1;
    a.ueId = 3;
    a.scStart = 1200;
    a.scLen = 96;
    a.dir = 0;
    a.rank = 2;
    a.beamId[0] = 10;
    a.beamId[1] = 200;

    CHECK(dsp::precodeGolden(x.data(), book, &a, 1, y.data()) == 0);

    // Direct check: y[c][tx][sc] == Σ_l W_l[tx]·x[c][l][sc].
    const uint32_t txs[] = {0, 17, 63};
    const uint32_t scs[] = {1200, 1250, 1295};
    for (uint32_t tx : txs) {
        for (uint32_t sc : scs) {
            cf32 want{0, 0};
            want = cmac(want, book.beam(10)[tx], x[layout::idxXdl(1, 0, sc)]);
            want = cmac(want, book.beam(200)[tx], x[layout::idxXdl(1, 1, sc)]);
            const cf32 got = y[layout::idxY(1, tx, sc)];
            CHECK(close(got.re, want.re) && close(got.im, want.im));
        }
    }

    // Outside the allocation (and the whole other cell): exactly zero.
    CHECK(y[layout::idxY(1, 0, 1199)].re == 0.0f);
    CHECK(y[layout::idxY(1, 0, 1296)].re == 0.0f);
    CHECK(y[layout::idxY(0, 32, 1250)].re == 0.0f &&
          y[layout::idxY(0, 32, 1250)].im == 0.0f);

    // Rank-1: single-layer product, exact (one cmul, no accumulation).
    Alloc r1 = a;
    r1.rank = 1;
    r1.scStart = 0;
    r1.scLen = 12;
    CHECK(dsp::precodeGolden(x.data(), book, &r1, 1, y.data()) == 0);
    const cf32 w = book.beam(10)[5];
    const cf32 xv = x[layout::idxXdl(1, 0, 7)];
    const cf32 want = cmul(w, xv);
    CHECK(y[layout::idxY(1, 5, 7)].re == want.re &&
          y[layout::idxY(1, 5, 7)].im == want.im);

    // UL allocations are ignored (not precoded, not rejected).
    Alloc ul = a;
    ul.dir = 1;
    CHECK(dsp::precodeGolden(x.data(), book, &ul, 1, y.data()) == 0);
    CHECK(y[layout::idxY(1, 0, 1250)].re == 0.0f);

    // Malformed allocations are rejected, never written.
    Alloc bad[4] = {a, a, a, a};
    bad[0].cell = dims::C;                  // cell out of range
    bad[1].rank = 5;                        // rank out of range
    bad[2].scStart = dims::numSc;           // sc out of range
    bad[3].beamId[0] = 256;                 // beam out of range (256-beam book)
    CHECK(dsp::precodeGolden(x.data(), book, bad, 4, y.data()) == 4);
    for (size_t i = 0; i < layout::elemsY; ++i)
        CHECK(y[i].re == 0.0f && y[i].im == 0.0f);
}

int main() {
    testCodebook();
    testPrecodeGolden();

    return orca::test::report("test_precode");
}
