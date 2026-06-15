// Sub-stage 1a test (AGENT.md): layout strides/alignment per Spec E §E.11,
// dims sanity, ci16↔cf32 round-trip + K5 saturation, half shim round-trip,
// symbol-counter continuity.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "common/complex.hpp"
#include "common/dims.hpp"
#include "common/layout.hpp"
#include "common/symbol_id.hpp"
#include "tests/check.hpp"

using namespace orca;

static void testDims() {
    CHECK(roundUp(3276, 32) == 3296);
    CHECK(dims::numSc == 3276);
    CHECK(dims::numScP == 3296);
    CHECK(dims::U == 32);
    CHECK(dims::slotsPerFrame(1) == 20);
}

static void testLayout() {
    using namespace layout;

    // Row strides and 128-B alignment are static_asserted in layout.hpp;
    // re-check the values the spec quotes.
    CHECK(rowBytesHalf2 == 13184);
    CHECK(rowBytesCf32 == 26368);

    // sc is unit-stride on every tensor.
    CHECK(idxH(0, 0, 0, 0, 1) - idxH(0, 0, 0, 0, 0) == 1);
    CHECK(idxXdl(0, 0, 1) - idxXdl(0, 0, 0) == 1);
    CHECK(idxY(0, 0, 1) - idxY(0, 0, 0) == 1);
    CHECK(idxRdl(0, 0, 1) - idxRdl(0, 0, 0) == 1);
    CHECK(idxXul(0, 0, 1) - idxXul(0, 0, 0) == 1);
    CHECK(idxRul(0, 0, 1) - idxRul(0, 0, 0) == 1);
    CHECK(idxZ(0, 0, 1) - idxZ(0, 0, 0) == 1);

    // Axis strides match the Spec E §E.11 index formulas.
    CHECK(idxH(0, 0, 0, 1, 0) == dims::numScP);                       // tx
    CHECK(idxH(0, 0, 1, 0, 0) == size_t{dims::numTx} * dims::numScP); // rx
    CHECK(idxH(0, 1, 0, 0, 0) ==
          size_t{dims::numRx} * dims::numTx * dims::numScP);          // u
    CHECK(idxH(1, 0, 0, 0, 0) ==
          size_t{dims::U} * dims::numRx * dims::numTx * dims::numScP);  // c
    CHECK(idxY(0, 1, 0) == dims::numScP);
    CHECK(idxY(1, 0, 0) == size_t{dims::numTx} * dims::numScP);
    CHECK(idxXdl(1, 0, 0) == size_t{dims::rankMax} * dims::numScP);
    CHECK(idxRdl(1, 0, 0) == size_t{dims::numRx} * dims::numScP);
    CHECK(idxXul(1, 0, 0) == size_t{dims::numUeTx} * dims::numScP);

    // Last valid element is within the allocation extent.
    CHECK(idxH(dims::C - 1, dims::U - 1, dims::numRx - 1, dims::numTx - 1,
               dims::numSc - 1) < elemsH);
    CHECK(idxRdl(dims::U - 1, dims::numRx - 1, dims::numSc - 1) < elemsRdl);
    CHECK(idxZ(dims::C - 1, dims::rankMax - 1, dims::numSc - 1) < elemsZ);
}

static void testCi16Cf32RoundTrip() {
    const int16_t samples[] = {0, 1, -1, 1234, -4321, INT16_MAX, INT16_MIN};
    for (int16_t re : samples) {
        for (int16_t im : samples) {
            ci16 in{re, im};
            ci16 out = toCi16(toCf32(in));
            CHECK(out.re == in.re && out.im == in.im);
        }
    }

    // K5 saturating round.
    CHECK(satRoundI16(40000.0f) == INT16_MAX);
    CHECK(satRoundI16(-40000.0f) == INT16_MIN);
    CHECK(satRoundI16(32767.4f) == INT16_MAX);
    CHECK(satRoundI16(-32768.4f) == INT16_MIN);
    CHECK(satRoundI16(0.25f) == 0);
    CHECK(satRoundI16(-0.25f) == 0);
    CHECK(satRoundI16(2.0f) == 2);

    // Ties round to even, independent of the FP rounding mode.
    CHECK(satRoundI16(0.5f) == 0);
    CHECK(satRoundI16(1.5f) == 2);
    CHECK(satRoundI16(2.5f) == 2);
    CHECK(satRoundI16(-0.5f) == 0);
    CHECK(satRoundI16(-1.5f) == -2);

    // NaN policy: maps to 0.
    CHECK(satRoundI16(std::numeric_limits<float>::quiet_NaN()) == 0);
}

static void testHalfShim() {
    // Values exactly representable in binary16 round-trip bit-exactly.
    const float exact[] = {0.0f, 1.0f, -1.0f, 0.5f, -2.0f, 0.099975586f, 65504.0f};
    for (float f : exact) {
        CHECK(halfToFloat(floatToHalf(f)) == f);
    }
    // Overflow → Inf; sign preserved on ±0.
    CHECK(halfToFloat(floatToHalf(1.0e6f)) > 65504.0f);   // +Inf
    CHECK(floatToHalf(-0.0f) == 0x8000u);
    // Complex shim round-trip.
    cf32 v{0.5f, -1.5f};
    cf32 back = toCf32(toHalf2(v));
    CHECK(back.re == v.re && back.im == v.im);
}

static void testSymbolCounter() {
    CHECK(symbolCounter(0, 0, 0) == 0);
    CHECK(symbolCounter(0, 0, 13) == 13);
    CHECK(symbolCounter(0, 1, 0) == 14);
    CHECK(symbolCounter(1, 0, 0) == 20ull * 14);  // µ=1: 20 slots/frame
    CHECK(symbolsPerSfnPeriod() == 1024ull * 20 * 14);
    CHECK(symbolCounter(1024, 0, 0) == symbolCounter(0, 0, 0));  // SFN mod 1024
    CHECK(symbolCounter(1030, 2, 3) == symbolCounter(6, 2, 3));

    SymbolId a{0, Dir::DL, 7, 3, 5};
    SymbolId b{0, Dir::DL, 7, 3, 5};
    SymbolId c = a;
    c.dir = Dir::UL;
    CHECK(a == b);
    CHECK(a != c);
}

int main() {
    testDims();
    testLayout();
    testCi16Cf32RoundTrip();
    testHalfShim();
    testSymbolCounter();

    return orca::test::report("test_layout");
}
