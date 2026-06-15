// Sub-stage 1b test: Spec A §A.4/§A.5 — ring lifecycle, coverage bitmap,
// out-of-order/duplicate sections, deadline-forced partial (late_drop),
// oldest-Filling eviction, and never-block claim rejection.

#include <cstdint>

#include "orchestr/symbol_ring.hpp"
#include "tests/check.hpp"

using namespace orca;

static SymbolId sym(uint16_t sfn, uint8_t slot, uint8_t s) {
    return SymbolId{0, Dir::DL, sfn, slot, s};
}

static void testCoverage() {
    PrbCoverage c;
    CHECK(!c.complete() && c.covered() == 0);

    CHECK(c.mark(0, 100) == 100);
    CHECK(c.mark(50, 100) == 50);   // overlap counts only new PRBs
    CHECK(c.mark(0, 150) == 0);     // duplicate is idempotent
    CHECK(c.covered() == 150);
    CHECK(c.test(0) && c.test(149) && !c.test(150));

    CHECK(c.mark(150, 1000) == dims::numPrb - 150);  // clamps at numPrb
    CHECK(c.complete());
    CHECK(!c.test(dims::numPrb));  // out-of-range is false, not UB

    c.clear();
    CHECK(c.covered() == 0 && !c.test(0));
}

static void testLifecycle() {
    SymbolRing<4> ring;
    const auto id = sym(0, 0, 0);

    const uint32_t i = ring.claim(id);
    CHECK(i != ring.kNone);
    CHECK(ring.at(i).state == SlotState::Filling);
    CHECK(ring.claim(id) == i);  // re-claim of the same symbol accumulates

    // Out-of-order sections complete the coverage.
    CHECK(!ring.addSection(i, 100, 173));
    CHECK(ring.addSection(i, 0, 100));
    ring.markReady(i);
    CHECK(ring.at(i).state == SlotState::Ready);
    CHECK(!ring.at(i).partial);
    CHECK(ring.stats().completed == 1);

    CHECK(ring.nextReady() == i);
    ring.markComputing(i);
    CHECK(ring.at(i).state == SlotState::Computing);
    CHECK(ring.nextReady() == ring.kNone);
    ring.markEgressing(i);
    ring.release(i);
    CHECK(ring.at(i).state == SlotState::Free);
    CHECK(ring.find(id) == ring.kNone);
}

static void testDeadlinePartial() {
    SymbolRing<4> ring;
    const uint32_t i = ring.claim(sym(0, 0, 1));
    ring.addSection(i, 0, 100);  // 173 PRBs missing at the deadline

    ring.forceReadyAtDeadline(i);
    CHECK(ring.at(i).state == SlotState::Ready);
    CHECK(ring.at(i).partial);
    CHECK(ring.stats().lateDrops == 1);
    // Missing PRBs are queryable so the owner can zero-fill the IQ.
    CHECK(ring.at(i).coverage.test(50) && !ring.at(i).coverage.test(200));

    // Forcing a complete slot is a normal completion, not a late drop.
    const uint32_t j = ring.claim(sym(0, 0, 2));
    ring.addSection(j, 0, dims::numPrb);
    ring.forceReadyAtDeadline(j);
    CHECK(!ring.at(j).partial);
    CHECK(ring.stats().lateDrops == 1);
    CHECK(ring.stats().completed == 1);
}

static void testEvictionAndRejection() {
    SymbolRing<3> ring;

    // Fill all three slots with unfinished symbols.
    const uint32_t a = ring.claim(sym(1, 0, 0));
    const uint32_t b = ring.claim(sym(1, 0, 1));
    const uint32_t c = ring.claim(sym(1, 0, 2));
    CHECK(a != ring.kNone && b != ring.kNone && c != ring.kNone);

    // A fourth symbol evicts the oldest Filling slot (a) — never stalls.
    const uint32_t d = ring.claim(sym(1, 0, 3));
    CHECK(d == a);
    CHECK(ring.at(d).id == sym(1, 0, 3));
    CHECK(ring.stats().overwrites == 1);
    CHECK(ring.stats().lateDrops == 1);
    CHECK(ring.find(sym(1, 0, 0)) == ring.kNone);

    // All slots busy beyond Filling → claim is rejected (input dropped).
    ring.addSection(d, 0, dims::numPrb);
    ring.markReady(d);
    const uint32_t rest[] = {b, c};
    for (uint32_t i : rest) {
        ring.addSection(i, 0, dims::numPrb);
        ring.markReady(i);
    }
    CHECK(ring.claim(sym(1, 0, 4)) == ring.kNone);
    CHECK(ring.stats().rejected == 1);

    // FIFO readiness: the oldest Ready slot comes out first.
    CHECK(ring.nextReady() == b);
}

int main() {
    testCoverage();
    testLifecycle();
    testDeadlinePartial();
    testEvictionAndRejection();

    return orca::test::report("test_ring");
}
