// Stage-5 golden test: K4 combine (64 → rank) against direct computation,
// zeroing contract, DL-ignore, and malformed-allocation rejection.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "common/dims.hpp"
#include "common/layout.hpp"
#include "dsp/combine.hpp"
#include "scenario/beam_codebook.hpp"
#include "tests/check.hpp"

using namespace orca;

static bool close(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

int main() {
    const BeamCodebook book = makeDftCodebook(256);
    std::vector<cf32> rul(layout::elemsRul, cf32{0, 0});
    std::vector<cf32> z(layout::elemsZ, cf32{55, 55});  // proves zeroing

    // Deterministic r_ul pattern.
    for (uint32_t c = 0; c < dims::C; ++c)
        for (uint32_t rx = 0; rx < dims::numTx; ++rx)
            for (uint32_t sc = 0; sc < dims::numSc; ++sc)
                rul[layout::idxRul(c, rx, sc)] =
                    cf32{static_cast<float>((rx + sc) % 7) - 3.0f,
                         static_cast<float>(c + (rx % 5)) - 2.0f};

    // Rank-2 UL allocation on cell 1, sc [600, 648).
    Alloc a{};
    a.cell = 1;
    a.ueId = 9;
    a.scStart = 600;
    a.scLen = 48;
    a.dir = 1;
    a.rank = 2;
    a.beamId[0] = 30;
    a.beamId[1] = 31;

    CHECK(dsp::combineGolden(rul.data(), book, &a, 1, z.data()) == 0);

    // Direct check at sampled (l, sc).
    const uint32_t scs[] = {600, 620, 647};
    for (uint8_t l = 0; l < 2; ++l) {
        for (uint32_t sc : scs) {
            cf32 want{0, 0};
            for (uint32_t rx = 0; rx < dims::numTx; ++rx)
                want = cmac(want, book.beam(a.beamId[l])[rx],
                            rul[layout::idxRul(1, rx, sc)]);
            const cf32 got = z[layout::idxZ(1, l, sc)];
            CHECK(close(got.re, want.re) && close(got.im, want.im));
        }
    }

    // Outside the allocation, unused layers, and the other cell: zero.
    CHECK(z[layout::idxZ(1, 0, 599)].re == 0.0f);
    CHECK(z[layout::idxZ(1, 0, 648)].re == 0.0f);
    CHECK(z[layout::idxZ(1, 2, 620)].re == 0.0f);  // rank 2 → layer 2 unused
    CHECK(z[layout::idxZ(0, 0, 620)].re == 0.0f);

    // Valid boundary: the highest beam id of the book is accepted.
    Alloc top = a;
    top.beamId[0] = top.beamId[1] = 255;
    CHECK(dsp::combineGolden(rul.data(), book, &top, 1, z.data()) == 0);
    CHECK(z[layout::idxZ(1, 0, 620)].re != 55.0f);  // written (not stale)

    // DL allocations are ignored (not combined, not rejected).
    Alloc dl = a;
    dl.dir = 0;
    CHECK(dsp::combineGolden(rul.data(), book, &dl, 1, z.data()) == 0);
    CHECK(z[layout::idxZ(1, 0, 620)].re == 0.0f);

    // Malformed allocations are rejected; z stays fully zeroed.
    Alloc bad[4] = {a, a, a, a};
    bad[0].cell = dims::C;
    bad[1].rank = 0;
    bad[2].scLen = dims::numSc;  // scStart 600 + numSc > carrier
    bad[3].beamId[1] = 256;      // out of the 256-beam book
    CHECK(dsp::combineGolden(rul.data(), book, bad, 4, z.data()) == 4);
    for (size_t i = 0; i < layout::elemsZ; ++i)
        CHECK(z[i].re == 0.0f && z[i].im == 0.0f);

    return orca::test::report("test_combine");
}
