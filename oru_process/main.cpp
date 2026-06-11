// ORU process — standalone program (ADR 0007: separate from ORCA).
//
// Phase-1 host build: no NIC yet (kernel/DPDK socket I/O lands at Linux
// bring-up; DOCA deferred). This main wires the engine to an in-memory
// transport and runs a self-contained smoke loop (synthetic vDU → engine →
// identity echo → UL verification) so the program exists, builds, and
// exercises the real engine path end-to-end.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "oru_process/oru_engine.hpp"
#include "tests/vdu_stub.hpp"

using namespace orca;

int main(int argc, char** argv) {
    const int numSymbols = argc > 1 ? std::atoi(argv[1]) : 28;  // 1 slot pair
    if (numSymbols <= 0) {
        std::fprintf(stderr, "usage: %s [numSymbols>0]\n", argv[0]);
        return 1;
    }

    OruLoopback transport;
    if (!transport.attach()) {
        std::fprintf(stderr, "transport attach failed\n");
        return 1;
    }
    OruEngine engine(transport);
    testutil::VduStub vdu;

    int verified = 0, mismatched = 0;
    for (int s = 0; s < numSymbols; ++s) {
        const uint16_t sfn = static_cast<uint16_t>(s / (20 * 14));
        const uint8_t slot = static_cast<uint8_t>((s / 14) % 20);
        const uint8_t sym = static_cast<uint8_t>(s % 14);

        for (const auto& p : vdu.buildDlSymbol(sfn, slot, sym, /*rank=*/2,
                                               /*numSections=*/4))
            engine.onPacket(p.data(), static_cast<uint32_t>(p.size()));

        // Identity echo (stands in for ORCA until 1f wires the real app).
        OruDoorbell d{};
        while (transport.pollDl(d)) {
            std::memcpy(transport.ulSlot(d.slotIdx), transport.dlSlot(d.slotIdx),
                        sizeof(ci16) * OruTransport::kSlotElems);
            transport.publishUl(d.slotIdx, d);
            transport.returnDl(d.slotIdx, d.seq);
        }

        engine.pollUlAndSend([&](const uint8_t* data, uint32_t len) {
            const int bad = vdu.verifyUlPacket(data, len);
            if (bad == 0)
                ++verified;
            else
                ++mismatched;
        });
    }

    const auto& st = engine.stats();
    std::printf(
        "oru_process: %d symbols | rx %llu pkts (%llu rejected) | published "
        "%llu (%llu partial, %llu late) | ul %llu pkts | verified %d, "
        "mismatched %d\n",
        numSymbols, (unsigned long long)st.rxPackets,
        (unsigned long long)st.rxRejected, (unsigned long long)st.published,
        (unsigned long long)st.partial, (unsigned long long)st.lateDrops,
        (unsigned long long)st.ulPackets, verified, mismatched);

    return (mismatched == 0 && st.published > 0) ? 0 : 1;
}
