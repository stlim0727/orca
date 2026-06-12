#pragma once
// ORCA Stage-1 identity hot path (AGENT.md 1f): per symbol
//   north pollDl → K0 (ci16→cf32) → south publishDl   (DL leg)
//   south pollUl → K5 (cf32→ci16) → north publishUl   (UL leg)
// No K1–K4 yet — the "channel" is the vUE identity copy. Host config runs
// the CPU converts; the target config swaps CUDA kernels behind dsp/convert
// (stream-ordered first, graph capture later — ADR 0001 §5).
//
// Stage-1 identity mapping: the cf32 image of x_dl[C][rank][numScP] occupies
// the head of the vUE DL slot (r_dl is strictly larger); the vUE copies its
// slot head to the UL slot (x_ul, also larger than the image), and K5
// converts the head back into z_host. Real tensor semantics arrive with the
// Stage-2+ kernels; the seams and slot protocol are exercised for real here.
//
// Slot identity: the north slot index is reused as the south slot index
// (both rings are N_ring deep). The north DL credit is held until the
// matching UL has been published north (the allocBlock of that slot must
// stay valid for the ORU process's UL packetization — see returnDl note).

#include <chrono>
#include <cstdint>

#include "common/dims.hpp"
#include "dsp/convert.hpp"
#include "orchestr/jitter.hpp"
#include "oru/oru_transport.hpp"
#include "vue/vue_transport.hpp"

namespace orca {

class IdentityApp {
  public:
    static constexpr uint32_t kImageElems = OruTransport::kSlotElems;
    static_assert(kImageElems == dims::identityImageElems,
                  "one shared definition of the Stage-1 image extent");
    static_assert(kImageElems <= VueTransport::kDlSlotElems,
                  "x_dl image must fit the vUE DL slot");
    static_assert(kImageElems <= VueTransport::kUlSlotElems,
                  "x_dl image must fit the vUE UL slot");

    struct Stats {
        uint64_t delivered = 0;     // UL published north successfully
        uint64_t southDlDrops = 0;  // vUE DL ring full — symbol dropped
        uint64_t northUlDrops = 0;  // ORU UL ring full — symbol dropped
        uint64_t staleUl = 0;       // UL doorbell with no/mismatched pending
    };

    IdentityApp(OruTransport& north, VueTransport& south)
        : north_(north), south_(south) {
        for (auto& p : pending_) p.valid = false;
    }

    // One scheduling pass over both directions. Returns DL symbols ingested.
    // Backpressure anywhere → drop + count, never block (Spec A §A.4).
    uint32_t step() {
        uint32_t n = 0;

        // DL leg: north → K0 → south.
        OruDoorbell d{};
        while (north_.pollDl(d)) {
            const ci16* x = north_.dlSlot(d.slotIdx);
            cf32* r = south_.dlSlot(d.slotIdx);
            if (x && r && d.slotIdx < dims::N_ring) {
                dsp::convertK0(x, r, kImageElems);
                VueDoorbell v{};
                v.sfn = d.sfn;
                v.slot = d.slot;
                v.sym = d.sym;
                v.flags = (d.flags & kOruFlagPartial) ? kVueFlagPartial : 0;
                v.seq = d.seq;
                if (south_.publishDl(d.slotIdx, v)) {
                    pending_[d.slotIdx].valid = true;
                    pending_[d.slotIdx].dl = d;
                    pending_[d.slotIdx].t0 = Clock::now();
                    ++n;
                } else {
                    // vUE ring full: the symbol is lost south — credit the
                    // north slot back so the pipeline keeps moving.
                    ++stats_.southDlDrops;
                    north_.returnDl(d.slotIdx, d.seq);
                }
            } else {
                // Nothing usable — credit the slot straight back.
                north_.returnDl(d.slotIdx, d.seq);
            }
        }

        // UL leg: south → K5 → north (then release the held DL credit).
        VueDoorbell u{};
        while (south_.pollUl(u)) {
            const cf32* xu = south_.ulSlot(u.slotIdx);
            ci16* z = north_.ulSlot(u.slotIdx);
            const bool match = u.slotIdx < dims::N_ring &&
                               pending_[u.slotIdx].valid &&
                               u.seq == pending_[u.slotIdx].dl.seq;
            if (xu && z && match) {
                dsp::convertK5(xu, z, kImageElems);
                const OruDoorbell orig = pending_[u.slotIdx].dl;
                if (north_.publishUl(u.slotIdx, orig)) {
                    ++stats_.delivered;
                    jitter_.record(static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            Clock::now() - pending_[u.slotIdx].t0)
                            .count()));
                } else {
                    ++stats_.northUlDrops;  // ORU ring full — UL lost
                }
                // UL sits in the north slot; the ORU process still needs the
                // slot's allocBlock — only now is the DL credit released
                // (on publish failure too: the symbol is accounted dropped
                // and the slot must recycle).
                north_.returnDl(u.slotIdx, orig.seq);
                pending_[u.slotIdx].valid = false;
            } else if (u.slotIdx >= dims::N_ring || !match) {
                ++stats_.staleUl;  // duplicate/unknown UL doorbell
            }
            south_.returnUl(u.slotIdx, u.seq);
        }

        // South DL credits (vUE finished reading r_dl).
        VueCredit c{};
        while (south_.reclaimDl(c)) {
        }
        // North UL credits (ORU process finished packetizing z).
        OruCredit oc{};
        while (north_.reclaimUl(oc)) {
        }
        return n;
    }

    const JitterHistogram& jitter() const { return jitter_; }
    const Stats& stats() const { return stats_; }

  private:
    using Clock = std::chrono::steady_clock;

    struct Pending {
        bool valid = false;
        OruDoorbell dl{};
        Clock::time_point t0{};
    };

    OruTransport& north_;
    VueTransport& south_;
    Pending pending_[dims::N_ring];
    JitterHistogram jitter_;
    Stats stats_;
};

}  // namespace orca
