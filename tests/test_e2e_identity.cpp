// Sub-stage 1f test: full host-config identity pipeline —
// vDU stub → ORU engine → OruTransport → ORCA (K0 → identity → K5) →
// VueTransport → vUE stub → back → vDU bit-exact (vDU-in == vDU-out),
// including a partial (deadline) symbol and the K0/K5 round-trip property.
// Prints jitter percentiles (Stage-1 deliverable).

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "app/identity_app.hpp"
#include "dsp/convert.hpp"
#include "oru/oru_loopback.hpp"
#include "oru_process/oru_engine.hpp"
#include "tests/vdu_stub.hpp"
#include "vue/vue_loopback.hpp"
#include "vue/vue_stub.hpp"

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

static void testConvertRoundTrip() {
    // K0 → K5 is bit-exact for every representable wire value (sampled).
    constexpr size_t kN = 1024;
    ci16 in[kN], out[kN];
    cf32 mid[kN];
    for (size_t i = 0; i < kN; ++i)
        in[i] = ci16{static_cast<int16_t>(i * 64 - 32768),
                     static_cast<int16_t>(32767 - i * 64)};
    dsp::convertK0(in, mid, kN);
    dsp::convertK5(mid, out, kN);
    for (size_t i = 0; i < kN; ++i)
        CHECK(out[i].re == in[i].re && out[i].im == in[i].im);
}

struct Harness {
    OruLoopback north;
    VueLoopback south;
    OruEngine* oru = nullptr;
    IdentityApp* orca = nullptr;
    VueStub* vue = nullptr;
    VduStub vdu;

    bool init() {
        if (!north.attach() || !south.attach()) return false;
        static OruEngine e{north};
        static IdentityApp a{north, south};
        static VueStub v{south};
        oru = &e;
        orca = &a;
        vue = &v;
        return true;
    }

    // Runs one symbol end to end; returns (ulPackets, mismatches).
    void roundTrip(uint16_t sfn, uint8_t slot, uint8_t sym, uint8_t rank,
                   uint32_t sections, int& ulPackets, int& mismatches,
                   bool dropLastSection = false) {
        auto pkts = vdu.buildDlSymbol(sfn, slot, sym, rank, sections);
        if (dropLastSection) pkts.pop_back();
        for (const auto& p : pkts)
            oru->onPacket(p.data(), static_cast<uint32_t>(p.size()));
        if (dropLastSection) oru->onDeadline(sfn, slot, sym);

        orca->step();
        vue->step();
        orca->step();

        ulPackets = 0;
        mismatches = 0;
        oru->pollUlAndSend([&](const uint8_t* data, uint32_t len) {
            ++ulPackets;
            const int bad = vdu.verifyUlPacket(data, len);
            CHECK(bad >= 0);
            if (bad > 0) mismatches += bad;
        });
    }
};

static Harness h;

static void testEndToEndIdentity() {
    CHECK(h.init());

    // 56 symbols (4 slots), mixed ranks/section counts — all bit-exact.
    int totalUl = 0;
    for (int s = 0; s < 56; ++s) {
        const uint16_t sfn = 0;
        const uint8_t slot = static_cast<uint8_t>(s / 14);
        const uint8_t sym = static_cast<uint8_t>(s % 14);
        const uint8_t rank = static_cast<uint8_t>(1 + (s % 4));
        const uint32_t sections = 1 + (s % 5);
        int ul = 0, bad = 0;
        h.roundTrip(sfn, slot, sym, rank, sections, ul, bad);
        CHECK(ul == rank);  // one full-band UL packet per layer
        CHECK(bad == 0);    // vDU-in == vDU-out, bit-exact
        totalUl += ul;
    }
    CHECK(h.oru->stats().published == 56);
    CHECK(h.oru->stats().publishDrops == 0);
    CHECK(static_cast<uint64_t>(totalUl) == h.oru->stats().ulPackets);
    CHECK(h.vue->processed() == 56);
    CHECK(h.vue->submitDrops() == 0);
    CHECK(h.orca->stats().delivered == 56);
    CHECK(h.orca->stats().southDlDrops == 0);
    CHECK(h.orca->stats().northUlDrops == 0);
    CHECK(h.orca->stats().staleUl == 0);

    // Jitter recorded for every symbol; print the Stage-1 percentiles.
    const auto& j = h.orca->jitter();
    CHECK(j.count() == 56);
    std::printf(
        "e2e jitter (K0→K5 span): n=%llu p50=%llu ns p99=%llu ns p99.9=%llu "
        "ns max=%llu ns\n",
        (unsigned long long)j.count(),
        (unsigned long long)j.percentileNs(0.5),
        (unsigned long long)j.percentileNs(0.99),
        (unsigned long long)j.percentileNs(0.999),
        (unsigned long long)j.maxNs());
}

static void testPartialSymbolEndToEnd() {
    // A deadline-forced partial symbol still flows end to end; the present
    // PRBs are bit-exact and the partial flag propagates to the vUE side.
    int ul = 0, bad = 0;
    h.roundTrip(1, 0, 0, /*rank=*/1, /*sections=*/4, ul, bad,
                /*dropLastSection=*/true);
    CHECK(ul == 1);
    // The dropped quarter comes back as zeros — those samples mismatch the
    // pattern by design; the verifier counts them, so bad > 0 here. The
    // first three quarters must be exact: bound the mismatch count.
    const int quarterSc = (dims::numPrb / 4 + 1) * dims::scPerPrb;
    CHECK(bad > 0 && bad <= quarterSc);
    CHECK(h.oru->stats().partial == 1);
}

int main() {
    testConvertRoundTrip();
    testEndToEndIdentity();
    testPartialSymbolEndToEnd();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("test_e2e_identity: all checks passed");
    return EXIT_SUCCESS;
}
