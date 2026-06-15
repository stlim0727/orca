// Sub-stage 1c test: OruTransport/VueTransport loopback backends —
// produce/consume in each direction, bulk-slot integrity, credit recycling
// over many cycles, and backpressure → push fails (never blocks).

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "common/spsc_ring.hpp"
#include "oru/oru_loopback.hpp"
#include "vue/vue_loopback.hpp"
#include "tests/check.hpp"

using namespace orca;

static void testSpscRing() {
    SpscRing<uint32_t, 4> r;
    CHECK(r.empty());

    uint32_t v = 0;
    CHECK(!r.pop(v));
    for (uint32_t i = 0; i < 4; ++i) CHECK(r.push(i));
    CHECK(!r.push(99));  // full → false, no block
    CHECK(r.size() == 4);
    for (uint32_t i = 0; i < 4; ++i) {
        CHECK(r.pop(v));
        CHECK(v == i);  // FIFO
    }
    CHECK(!r.pop(v));

    // Wraparound across the index space.
    for (uint32_t i = 0; i < 100; ++i) {
        CHECK(r.push(i));
        CHECK(r.pop(v));
        CHECK(v == i);
    }
}

static void testOruLoopback() {
    OruLoopback t;
    OruDoorbell d{};
    CHECK(!t.pollDl(d));  // not attached yet
    CHECK(t.attach());

    // DL: ORU side fills slot + allocs, rings the doorbell; ORCA consumes.
    ci16* dl = t.dlSlot(1);
    CHECK(dl != nullptr);
    for (uint32_t i = 0; i < 16; ++i) dl[i] = ci16{int16_t(i), int16_t(-int(i))};
    AllocBlock* ab = t.allocBlock(1);
    CHECK(ab != nullptr);
    ab->numAllocs = 1;
    ab->allocs[0] = Alloc{0, 7, 0, 48, 0, 2, {3, 4, 0, 0}};
    CHECK(t.oruPublishDl(OruDoorbell{1, 100, 2, 3, 1, 0, 42}));

    CHECK(t.pollDl(d));
    CHECK(d.slotIdx == 1 && d.sfn == 100 && d.slot == 2 && d.sym == 3);
    CHECK(d.numAllocs == 1 && d.seq == 42);
    const ci16* rd = t.dlSlot(d.slotIdx);
    CHECK(rd[5].re == 5 && rd[5].im == -5);
    CHECK(t.allocBlock(d.slotIdx)->allocs[0].ueId == 7);
    CHECK(t.returnDl(d.slotIdx, d.seq));

    OruCredit c{};
    CHECK(t.oruReclaimDl(c));
    CHECK(c.slotIdx == 1 && c.seq == 42);

    // UL: ORCA fills z slot, publishes; ORU side consumes and credits back.
    // The explicit slotIdx parameter overrides whatever meta carries.
    ci16* ul = t.ulSlot(2);
    ul[0] = ci16{1234, -4321};
    CHECK(t.publishUl(2, OruDoorbell{99, 100, 2, 3, 0, 0, 43}));
    CHECK(t.oruPollUl(d));
    CHECK(d.slotIdx == 2 && d.seq == 43);  // 99 was overwritten
    CHECK(t.ulSlot(d.slotIdx)[0].re == 1234);
    CHECK(t.oruReturnUl(OruCredit{d.slotIdx, d.seq}));
    CHECK(t.reclaimUl(c));
    CHECK(c.slotIdx == 2 && c.seq == 43);

    // Out-of-range slot access is null, not UB.
    CHECK(t.dlSlot(dims::N_ring) == nullptr);
    CHECK(t.allocBlock(0xFFFF) == nullptr);

    // Backpressure: the doorbell ring fills and then refuses (never blocks).
    uint32_t pushed = 0;
    while (t.oruPublishDl(OruDoorbell{0, 0, 0, 0, 0, 0, pushed})) ++pushed;
    CHECK(pushed == OruLoopback::kRingDepth);

    t.detach();
    CHECK(!t.pollDl(d));
    CHECK(t.dlSlot(0) == nullptr);  // detached → no slot pointers
}

static void testVueLoopback() {
    VueLoopback t;
    CHECK(t.attach());

    // DL: ORCA produces r_dl, vUE consumes and credits back (§D.7 DL-1..6).
    cf32* dl = t.dlSlot(0);
    CHECK(dl != nullptr);
    dl[0] = cf32{1.5f, -2.5f};
    CHECK(t.publishDl(0, VueDoorbell{0, 200, 4, 5, 0, 7}));

    VueDoorbell d{};
    CHECK(t.pollDl(d));
    CHECK(d.slotIdx == 0 && d.sfn == 200 && d.slot == 4 && d.sym == 5 && d.seq == 7);
    CHECK(t.dlSlot(d.slotIdx)[0].re == 1.5f);
    CHECK(t.returnDl(d.slotIdx, d.seq));
    VueCredit c{};
    CHECK(t.reclaimDl(c));
    CHECK(c.slotIdx == 0 && c.seq == 7);

    // UL: vUE produces x_ul, ORCA consumes and credits back; explicit slotIdx
    // wins over meta.
    t.ulSlot(3)[1] = cf32{-0.5f, 0.25f};
    CHECK(t.submitUl(3, VueDoorbell{77, 200, 4, 5, kVueFlagPartial, 8}));
    CHECK(t.pollUl(d));
    CHECK(d.slotIdx == 3 && (d.flags & kVueFlagPartial));  // 77 overwritten
    CHECK(t.ulSlot(d.slotIdx)[1].im == 0.25f);
    CHECK(t.returnUl(d.slotIdx, d.seq));
    CHECK(t.reclaimUl(c));
    CHECK(c.slotIdx == 3 && c.seq == 8);

    // Credit recycling: cycle every slot many times through the full
    // publish→poll→return→reclaim loop; rings must never wedge.
    for (uint32_t cycle = 0; cycle < 100; ++cycle) {
        const uint16_t slot = uint16_t(cycle % dims::N_ring);
        CHECK(t.publishDl(slot, VueDoorbell{0, 0, 0, 0, 0, cycle}));
        CHECK(t.pollDl(d));
        CHECK(d.slotIdx == slot);
        CHECK(t.returnDl(d.slotIdx, d.seq));
        CHECK(t.reclaimDl(c));
        CHECK(c.seq == cycle);
    }

    t.detach();
    CHECK(t.ulSlot(0) == nullptr);
}

int main() {
    testSpscRing();
    testOruLoopback();
    testVueLoopback();

    return orca::test::report("test_transport_loopback");
}
