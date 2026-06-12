// Sub-stage 1b test: Spec A §A.1/§A.2 timing — exact µ=1 symbol geometry in
// Tc, Tc→ns conversion, T_air monotonicity, §A.3 deadline formulas, and the
// jitter histogram.

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "orchestr/jitter.hpp"
#include "orchestr/timing.hpp"

using namespace orca;

static int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

static void testSymbolGeometry() {
    // µ=1 exact values (38.211 normal CP).
    CHECK(usefulTc(1) == 65536);
    CHECK(cpNormalTc(1) == 4608);
    CHECK(cpLongExtraTc() == 1024);
    CHECK(symTc(1) == 70144);
    CHECK(slotDurTc(1) == 983040);

    // Tc→ns: slot is exactly 500000 ns; normal symbol ≈ 35.68 µs (Spec A §A.2
    // quotes ~35.7 µs).
    CHECK(tcToNs(slotDurTc(1)) == 500000);
    CHECK(tcToNs(symTc(1)) == 35677);
    CHECK(tcToNs(kTcPerSec) == 1000000000ull);  // 1 s of Tc is exactly 1e9 ns

    // symStart: CP-adjusted cumulative offsets; symbol 0 carries the long CP.
    CHECK(symStartTc(0) == 0);
    CHECK(symStartTc(1) == 71168);                 // long-CP symbol span
    CHECK(symStartTc(2) - symStartTc(1) == 70144); // normal span
    CHECK(symStartTc(1) - symStartTc(0) > symStartTc(2) - symStartTc(1));
    // Last symbol ends exactly at the slot boundary.
    CHECK(symStartTc(13) + symTc(1) == slotDurTc(1));
}

static void testTAir() {
    const uint64_t t0 = 1'000'000;  // arbitrary S-plane bootstrap (Tc)

    CHECK(tAirTc(t0, 0, 0) == t0);
    // Monotonic across symbol and slot boundaries.
    uint64_t prev = 0;
    for (uint64_t slot = 0; slot < 3; ++slot) {
        for (uint32_t sym = 0; sym < 14; ++sym) {
            const uint64_t t = tAirTc(t0, slot, sym);
            CHECK(slot == 0 && sym == 0 ? t == t0 : t > prev);
            prev = t;
        }
    }
    // Slot boundary: last symbol of slot 0 ends where slot 1 begins.
    CHECK(tAirTc(t0, 0, 13) + symTc(1) == tAirTc(t0, 1, 0));

    // SymbolId overload: epoch 0 matches the wrapped slot index; epoch k adds
    // k full 1024-frame periods (no aliasing for long runs).
    SymbolId id{0, Dir::DL, 2, 5, 7};
    CHECK(tAirTc(t0, 0, id) == tAirTc(t0, 2ull * 20 + 5, 7));
    CHECK(tAirTc(t0, 1, id) == tAirTc(t0, 1024ull * 20 + 2ull * 20 + 5, 7));
}

static void testDeadlines() {
    const DeadlineConfig d{5000, 6000, 7000, 30000};  // egress/proc/margin/ulOff ns
    const int64_t tAirNs = 10'000'000;

    CHECK(deadlineDlNs(tAirNs, d) == tAirNs - 5000 - 6000 - 7000);
    CHECK(deadlineUlNs(tAirNs, d) == tAirNs + 30000 - 6000 - 5000 - 7000);
    CHECK(deadlineDlNs(tAirNs, d) < tAirNs);   // D_r before air time
    CHECK(deadlineUlNs(tAirNs, d) > tAirNs);   // UL window after air time here

    // Signed arithmetic: a deadline before the clock origin stays negative
    // instead of wrapping to a huge unsigned future.
    CHECK(deadlineDlNs(0, d) == -(5000 + 6000 + 7000));
}

static void testJitter() {
    JitterHistogram h;
    CHECK(h.count() == 0);
    CHECK(h.percentileNs(0.5) == 0);

    // 1000 samples at 10 µs, 10 at 100 µs, 1 overflow at 2 ms.
    for (int i = 0; i < 1000; ++i) h.record(10'000);
    for (int i = 0; i < 10; ++i) h.record(100'000);
    h.record(2'000'000);

    CHECK(h.count() == 1011);
    CHECK(h.minNs() == 10'000);
    CHECK(h.maxNs() == 2'000'000);
    CHECK(h.overflow() == 1);
    // p50 lands in the 10 µs bin (upper edge 10.1 µs).
    CHECK(h.percentileNs(0.5) == 10'100);
    // p99.9 still inside the recorded range, ≥ the 100 µs population.
    CHECK(h.percentileNs(0.999) >= 100'000);

    h.reset();
    CHECK(h.count() == 0 && h.maxNs() == 0 && h.percentileNs(0.99) == 0);
}

int main() {
    testSymbolGeometry();
    testTAir();
    testDeadlines();
    testJitter();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("test_timing: all checks passed");
    return EXIT_SUCCESS;
}
