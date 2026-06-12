// ORCA application — Stage-1 identity hot path over the real transport
// seams (AGENT.md 1f). Host config: in-process loopback backends + CPU
// converts; the full chain is
//   vDU stub → ORU engine → OruTransport → ORCA (K0 → identity → K5)
//   → VueTransport → vUE stub → back → vDU stub bit-exact check,
// with per-symbol jitter percentiles printed (Stage-1 deliverable).

#include <cstdio>
#include <cstdlib>

#include "app/identity_app.hpp"
#include "oru/oru_loopback.hpp"
#include "oru_process/oru_engine.hpp"
#include "tests/vdu_stub.hpp"
#include "vue/vue_loopback.hpp"
#include "vue/vue_stub.hpp"

using namespace orca;

int main(int argc, char** argv) {
    const int numSymbols = argc > 1 ? std::atoi(argv[1]) : 280;  // 10 slots
    if (numSymbols <= 0) {
        std::fprintf(stderr, "usage: %s [numSymbols>0]\n", argv[0]);
        return 1;
    }

    OruLoopback north;
    VueLoopback south;
    if (!north.attach() || !south.attach()) {
        std::fprintf(stderr, "transport attach failed\n");
        return 1;
    }
    OruEngine oru(north);
    IdentityApp orca(north, south);
    VueStub vue(south);
    testutil::VduStub vdu;

    int verified = 0, mismatched = 0;
    for (int s = 0; s < numSymbols; ++s) {
        const uint16_t sfn = static_cast<uint16_t>(s / (20 * 14));
        const uint8_t slot = static_cast<uint8_t>((s / 14) % 20);
        const uint8_t sym = static_cast<uint8_t>(s % 14);

        for (const auto& p : vdu.buildDlSymbol(sfn, slot, sym, /*rank=*/2,
                                               /*numSections=*/4))
            oru.onPacket(p.data(), static_cast<uint32_t>(p.size()));

        orca.step();  // DL: north → K0 → south
        vue.step();   // vUE identity PHY
        orca.step();  // UL: south → K5 → north

        oru.pollUlAndSend([&](const uint8_t* data, uint32_t len) {
            if (vdu.verifyUlPacket(data, len) == 0)
                ++verified;
            else
                ++mismatched;
        });
    }

    const auto& st = oru.stats();
    const auto& j = orca.jitter();
    std::printf(
        "orca identity e2e: %d symbols | published %llu (%llu partial, %llu "
        "late, %llu pubDrops) | ul %llu pkts | verified %d, mismatched %d\n",
        numSymbols, (unsigned long long)st.published,
        (unsigned long long)st.partial, (unsigned long long)st.lateDrops,
        (unsigned long long)st.publishDrops, (unsigned long long)st.ulPackets,
        verified, mismatched);
    std::printf(
        "per-symbol latency (K0→K5 span): n=%llu min=%llu ns p50=%llu ns "
        "p99=%llu ns p99.9=%llu ns max=%llu ns\n",
        (unsigned long long)j.count(), (unsigned long long)j.minNs(),
        (unsigned long long)j.percentileNs(0.5),
        (unsigned long long)j.percentileNs(0.99),
        (unsigned long long)j.percentileNs(0.999),
        (unsigned long long)j.maxNs());

    const bool ok = mismatched == 0 && verified > 0 &&
                    st.published == static_cast<uint64_t>(numSymbols);
    return ok ? 0 : 1;
}
