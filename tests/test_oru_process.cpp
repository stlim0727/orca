// Sub-stage 1e test: vDU stub → OruEngine (Spec B terminate + reassemble +
// C-plane parse) → OruTransport → fake-ORCA identity echo → OruEngine UL
// packetize → vDU stub bit-exact verification. Plus: out-of-order sections,
// deadline partial publish, malformed-packet rejection, eviction.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "oru_process/oru_engine.hpp"
#include "tests/vdu_stub.hpp"

using namespace orca;
using testutil::VduStub;

static int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

// Fake ORCA: drain DL doorbells, copy x_dl slot → z slot (identity), publish
// UL with the same symbol key, recycle credits.
static void fakeOrcaIdentity(OruLoopback& t) {
    OruDoorbell d{};
    while (t.pollDl(d)) {
        const ci16* x = t.dlSlot(d.slotIdx);
        ci16* z = t.ulSlot(d.slotIdx);
        CHECK(x && z);
        std::memcpy(z, x, sizeof(ci16) * OruTransport::kSlotElems);
        CHECK(t.publishUl(d.slotIdx, d));  // same key/seq/allocs back
        CHECK(t.returnDl(d.slotIdx, d.seq));
    }
}

static void runSymbol(OruLoopback& t, OruEngine& e, VduStub& vdu, uint16_t sfn,
                      uint8_t slot, uint8_t sym, uint8_t rank,
                      uint32_t numSections, bool shuffle) {
    auto pkts = vdu.buildDlSymbol(sfn, slot, sym, rank, numSections);
    if (shuffle && pkts.size() > 2) {
        // Deliver U-plane sections in reverse (C-plane still first).
        std::vector<std::vector<uint8_t>> rev(pkts.begin() + 1, pkts.end());
        for (size_t i = 0; i < rev.size(); ++i)
            pkts[1 + i] = rev[rev.size() - 1 - i];
    }
    for (const auto& p : pkts)
        CHECK(e.onPacket(p.data(), static_cast<uint32_t>(p.size())));

    fakeOrcaIdentity(t);

    int ulPackets = 0, mismatches = 0;
    e.pollUlAndSend([&](const uint8_t* data, uint32_t len) {
        ++ulPackets;
        const int bad = vdu.verifyUlPacket(data, len);
        CHECK(bad >= 0);
        mismatches += bad > 0 ? bad : 0;
    });
    CHECK(ulPackets == rank);  // one full-band UL packet per layer
    CHECK(mismatches == 0);    // bit-exact identity loopback
}

static void testIdentityLoopback() {
    OruLoopback t;
    CHECK(t.attach());
    OruEngine e(t);
    VduStub vdu;

    // Whole-band single section, rank 1.
    runSymbol(t, e, vdu, 0, 0, 0, 1, 1, false);
    // Multi-section reassembly (7 sections), rank 1, out-of-order delivery.
    runSymbol(t, e, vdu, 0, 0, 1, 1, 7, true);
    // Rank 4 — completion requires all four layer flows.
    runSymbol(t, e, vdu, 0, 1, 2, 4, 3, false);

    CHECK(e.stats().published == 3);
    CHECK(e.stats().partial == 0);
    CHECK(e.stats().rxRejected == 0);
}

static void testPartialDeadline() {
    OruLoopback t;
    CHECK(t.attach());
    OruEngine e(t);
    VduStub vdu;

    // Drop the last U-plane section → incomplete at the deadline.
    auto pkts = vdu.buildDlSymbol(1, 2, 3, 1, 4);
    pkts.pop_back();
    for (const auto& p : pkts)
        CHECK(e.onPacket(p.data(), static_cast<uint32_t>(p.size())));
    CHECK(e.stats().published == 0);  // not complete yet

    e.onDeadline(1, 2, 3);
    CHECK(e.stats().published == 1);
    CHECK(e.stats().partial == 1);

    // The partial flag propagates; missing PRBs come back as zeros.
    OruDoorbell d{};
    CHECK(t.pollDl(d));
    CHECK(d.flags & kOruFlagPartial);
    const ci16* x = t.dlSlot(d.slotIdx);
    // Last quarter of the band was dropped → zero-filled.
    const uint32_t lastSc = dims::numSc - 1;
    CHECK(x[layout::idxXdl(0, 0, lastSc)].re == 0 &&
          x[layout::idxXdl(0, 0, lastSc)].im == 0);
    // First section's data is present.
    const ci16 e0 = VduStub::patternIq(1, 2, 3, 0, 0);
    CHECK(x[layout::idxXdl(0, 0, 0)].re == e0.re);

    // Deadline on an unknown symbol is a no-op.
    e.onDeadline(9, 9, 9);
    CHECK(e.stats().published == 1);
}

static void testRejects() {
    OruLoopback t;
    CHECK(t.attach());
    OruEngine e(t);
    VduStub vdu;

    auto pkts = vdu.buildDlSymbol(2, 0, 0, 1, 1);
    // Truncated header.
    CHECK(!e.onPacket(pkts[1].data(), 10));
    // Truncated U-plane payload.
    CHECK(!e.onPacket(pkts[1].data(), static_cast<uint32_t>(pkts[1].size() - 1)));
    // Bad version.
    std::vector<uint8_t> bad = pkts[1];
    bad[0] = 0x20;  // ver=2
    CHECK(!e.onPacket(bad.data(), static_cast<uint32_t>(bad.size())));
    // Cell beyond numActiveCells (default 1): eAxC with ccId=1 → cell 1.
    bad = pkts[1];
    fh::FhHeader h = fh::unpack(bad.data());
    h.eAxC = fh::encodeEaxc(fh::Eaxc{0, 0, 1, 0});
    fh::pack(h, bad.data());
    CHECK(!e.onPacket(bad.data(), static_cast<uint32_t>(bad.size())));
    CHECK(e.stats().rxRejected == 4);
    CHECK(e.stats().published == 0);
}

static void testDuplicateAfterPublish() {
    OruLoopback t;
    CHECK(t.attach());
    OruEngine e(t);
    VduStub vdu;

    auto pkts = vdu.buildDlSymbol(4, 0, 0, 1, 1);
    for (const auto& p : pkts)
        CHECK(e.onPacket(p.data(), static_cast<uint32_t>(p.size())));
    CHECK(e.stats().published == 1);

    // ORCA has not credited the slot yet; a late duplicate section for the
    // same symbol must not re-claim a slot and double-publish.
    auto dup = vdu.buildDlSymbol(4, 0, 0, 1, 1);
    CHECK(!e.onPacket(dup[1].data(), static_cast<uint32_t>(dup[1].size())));
    CHECK(e.stats().published == 1);
}

static void testEviction() {
    OruLoopback t;
    CHECK(t.attach());
    OruEngine e(t);
    VduStub vdu;

    // Start N_ring+1 incomplete symbols; the oldest is evicted, never a stall.
    for (uint8_t sym = 0; sym < dims::N_ring + 1; ++sym) {
        auto pkts = vdu.buildDlSymbol(3, 0, sym, 1, 2);
        pkts.pop_back();  // keep incomplete
        for (const auto& p : pkts)
            CHECK(e.onPacket(p.data(), static_cast<uint32_t>(p.size())));
    }
    CHECK(e.stats().lateDrops == 1);
    CHECK(e.stats().published == 0);
}

int main() {
    testIdentityLoopback();
    testPartialDeadline();
    testRejects();
    testDuplicateAfterPublish();
    testEviction();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("test_oru_process: all checks passed");
    return EXIT_SUCCESS;
}
