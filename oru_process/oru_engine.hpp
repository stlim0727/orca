#pragma once
// The ORU process core (AGENT.md 1e; ADR 0007 / Spec F §F.6). A separate
// program from ORCA: terminates Spec B framing from the vDU (here over a
// software/in-memory transport — kernel/DPDK sockets arrive at Linux
// bring-up), reassembles U-plane sections per symbol, parses the C-plane
// into the allocBlock, and relays via the OruTransport producer side.
//
// Reassembly model: one DL slot per symbol holding ALL cells (the Spec F
// slot tensor is x_dl_host[C][rank][numScP]). Coverage is tracked per
// (cell, layer) PRB bitmap; a cell is complete when its C-plane arrived and
// every layer 0..rank-1 has full PRB coverage; the symbol publishes when all
// active cells are complete, or at the deadline (§A.4: missing PRBs stay
// zero-filled — slots are zeroed on claim — flags.partial set, late_drop).

#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

#include "common/dims.hpp"
#include "common/layout.hpp"
#include "fh/cplane.hpp"
#include "fh/eaxc.hpp"
#include "fh/fh_header.hpp"
#include "fh/uplane.hpp"
#include "orchestr/symbol_ring.hpp"
#include "oru/oru_loopback.hpp"

namespace orca {

class OruEngine {
  public:
    struct Config {
        uint32_t numActiveCells = 1;  // cells this vDU drives (≤ dims::C)
        fh::EaxcConfig eaxc{};
    };

    struct Stats {
        uint64_t rxPackets = 0;
        uint64_t rxRejected = 0;   // failed validate()/malformed payload
        uint64_t published = 0;    // DL symbols handed to ORCA
        uint64_t partial = 0;      // §A.4 deadline-forced publishes
        uint64_t lateDrops = 0;    // evicted unfinished symbols
        uint64_t publishDrops = 0; // doorbell ring full — symbol lost
        uint64_t ulPackets = 0;    // UL packets sent toward the vDU
    };

    using TxFn = std::function<void(const uint8_t* data, uint32_t len)>;

    explicit OruEngine(OruLoopback& transport, Config cfg = {})
        : t_(transport), cfg_(cfg) {
        // An invalid eAxC width config would poison every encode/decode;
        // clamp to the deployment default (documented behavior).
        if (!cfg_.eaxc.valid()) cfg_.eaxc = fh::EaxcConfig{};
        if (cfg_.numActiveCells == 0 || cfg_.numActiveCells > dims::C)
            cfg_.numActiveCells = 1;
        txBuf_.resize(fh::kHeaderBytes + fh::uplaneBytes(dims::numPrb));
        for (auto& s : slots_) s = SlotTrack{};
    }

    // One Spec B packet from the vDU. Returns false on rejected input.
    // All validation happens BEFORE a slot is claimed, so a malformed packet
    // never allocates/zeroes a slot.
    bool onPacket(const uint8_t* data, uint32_t len) {
        ++stats_.rxPackets;
        if (len < fh::kHeaderBytes) return reject();
        const fh::FhHeader h = fh::unpack(data);
        if (!fh::validate(h) || h.dir != fh::FhDir::DL) return reject();
        const uint16_t cell = fh::cellOf(h.eAxC, cfg_.eaxc);
        if (cell >= cfg_.numActiveCells) return reject();

        switch (h.msgtyp) {
            case fh::MsgType::UPlane:
                return onUplane(h, cell, data, len);
            case fh::MsgType::CPlane:
                return onCplane(h, cell, data, len);
            default:
                // S-plane/telemetry handling is out of 1e scope.
                return true;
        }
    }

    // Deadline D_r(s) for a symbol key (§A.4): publish what arrived; missing
    // PRBs are already zero (slot zeroed on claim).
    void onDeadline(uint16_t sfn, uint8_t slot, uint8_t sym) {
        const uint32_t idx = find(sfn, slot, sym);
        if (idx == kNone) return;
        ++stats_.partial;
        publish(idx, /*partialFlag=*/true);
    }

    // UL drain (§F.6 UL-3): packetize each published z slot back to the vDU
    // per its allocBlock, then credit the slot back to ORCA.
    void pollUlAndSend(const TxFn& tx) {
        OruDoorbell d{};
        while (t_.oruPollUl(d)) {
            const ci16* z = t_.ulSlot(d.slotIdx);
            const AllocBlock* ab = t_.allocBlock(d.slotIdx);
            if (z && ab) sendUlSlot(*ab, z, d, tx);
            t_.oruReturnUl(OruCredit{d.slotIdx, d.seq});
        }
    }

    const Stats& stats() const { return stats_; }

  private:
    static constexpr uint32_t kNone = UINT32_MAX;

    struct SlotTrack {
        bool inUse = false;
        uint16_t sfn = 0;
        uint8_t slot = 0, sym = 0;
        uint64_t seqClaim = 0;
        bool cplaneSeen[dims::C] = {};
        uint8_t rank[dims::C] = {};
        PrbCoverage cov[dims::C][dims::rankMax];
    };

    bool reject() {
        ++stats_.rxRejected;
        return false;
    }

    bool onUplane(const fh::FhHeader& h, uint16_t cell, const uint8_t* data,
                  uint32_t len) {
        // Fully validate before claiming.
        const uint16_t layer = fh::decodeEaxc(h.eAxC, cfg_.eaxc).ruPort;
        if (layer >= dims::rankMax) return reject();
        const uint32_t want = fh::uplaneBytes(h.numPrb);
        if (want == 0 || len != fh::kHeaderBytes + want) return reject();

        const uint32_t idx = claim(h.sfn, h.slot, h.sym);
        if (idx == kNone) return reject();  // mid-flight everywhere → drop
        SlotTrack& s = slots_[idx];
        ci16* dst = t_.dlSlot(static_cast<uint16_t>(idx));
        if (!dst) return reject();
        const uint32_t scStart = h.startPrb * dims::scPerPrb;
        const uint32_t numSc = h.numPrb * dims::scPerPrb;
        fh::unpackUplane(data + fh::kHeaderBytes, numSc,
                         dst + layout::idxXdl(cell, layer, scStart));
        s.cov[cell][layer].mark(h.startPrb, h.numPrb);
        if (symbolComplete(s)) publish(idx, /*partialFlag=*/false);
        return true;
    }

    bool onCplane(const fh::FhHeader& h, uint16_t cell, const uint8_t* data,
                  uint32_t len) {
        // Parse and validate the whole payload before claiming/committing
        // (engine is single-threaded; member scratch avoids a 9 KB stack hit).
        Alloc* pending = pendingScratch_;
        uint32_t numPending = 0;
        uint32_t off = fh::kHeaderBytes;
        while (off < len) {
            fh::CplaneSection sec{};
            const uint32_t used = fh::parseCplane(data + off, len - off, sec);
            if (used == 0) return reject();
            off += used;
            Alloc a{};
            if (!fh::toAlloc(sec, cell, a)) return reject();
            if (numPending >= dims::MAX_ALLOCS) return reject();
            pending[numPending++] = a;
        }
        if (numPending == 0) return reject();

        const uint32_t idx = claim(h.sfn, h.slot, h.sym);
        if (idx == kNone) return reject();
        SlotTrack& s = slots_[idx];
        AllocBlock* ab = t_.allocBlock(static_cast<uint16_t>(idx));
        if (!ab) return reject();
        for (uint32_t i = 0; i < numPending; ++i) {
            if (ab->numAllocs >= dims::MAX_ALLOCS) break;
            ab->allocs[ab->numAllocs++] = pending[i];
            s.cplaneSeen[cell] = true;
            if (pending[i].rank > s.rank[cell]) s.rank[cell] = pending[i].rank;
        }
        if (symbolComplete(s)) publish(idx, /*partialFlag=*/false);
        return true;
    }

    uint32_t find(uint16_t sfn, uint8_t slot, uint8_t sym) const {
        for (uint32_t i = 0; i < dims::N_ring; ++i) {
            const SlotTrack& s = slots_[i];
            if (s.inUse && s.sfn == sfn && s.slot == slot && s.sym == sym)
                return i;
        }
        return kNone;
    }

    // A published slot stays ORCA-owned until its dlReturn credit comes back
    // (§F.6 DL-5) — never reuse it earlier.
    void drainCredits() {
        OruCredit c{};
        while (t_.oruReclaimDl(c))
            if (c.slotIdx < dims::N_ring) busy_[c.slotIdx] = false;
    }

    uint32_t claim(uint16_t sfn, uint8_t slot, uint8_t sym) {
        uint32_t idx = find(sfn, slot, sym);
        if (idx != kNone) return idx;
        drainCredits();
        // A symbol already published and still ORCA-owned must not be
        // re-claimed by late duplicate sections (duplicate publish hazard).
        for (uint32_t i = 0; i < dims::N_ring; ++i) {
            if (busy_[i] && pubSfn_[i] == sfn && pubSlot_[i] == slot &&
                pubSym_[i] == sym)
                return kNone;
        }
        uint32_t freeIdx = kNone, oldest = kNone;
        uint64_t oldestSeq = UINT64_MAX;
        for (uint32_t i = 0; i < dims::N_ring; ++i) {
            if (!slots_[i].inUse) {
                if (!busy_[i] && freeIdx == kNone) freeIdx = i;
            } else if (slots_[i].seqClaim < oldestSeq) {
                oldestSeq = slots_[i].seqClaim;
                oldest = i;
            }
        }
        idx = freeIdx;
        if (idx == kNone) {
            // Evict the oldest unfinished symbol — never stall (§A.4).
            idx = oldest;
            if (idx == kNone) return kNone;
            ++stats_.lateDrops;
        }
        SlotTrack& s = slots_[idx];
        s = SlotTrack{};
        s.inUse = true;
        s.sfn = sfn;
        s.slot = slot;
        s.sym = sym;
        s.seqClaim = nextClaim_++;
        // Zero the bulk + alloc map so missing PRBs publish as zeros (§A.4).
        ci16* dst = t_.dlSlot(static_cast<uint16_t>(idx));
        if (dst) std::memset(dst, 0, sizeof(ci16) * OruTransport::kSlotElems);
        AllocBlock* ab = t_.allocBlock(static_cast<uint16_t>(idx));
        if (ab) ab->numAllocs = 0;
        return idx;
    }

    bool symbolComplete(const SlotTrack& s) const {
        for (uint32_t c = 0; c < cfg_.numActiveCells; ++c) {
            if (!s.cplaneSeen[c] || s.rank[c] == 0) return false;
            for (uint8_t l = 0; l < s.rank[c]; ++l)
                if (!s.cov[c][l].complete()) return false;
        }
        return true;
    }

    void publish(uint32_t idx, bool partialFlag) {
        SlotTrack& s = slots_[idx];
        if (!s.inUse) return;
        OruDoorbell d{};
        d.slotIdx = static_cast<uint16_t>(idx);
        d.sfn = s.sfn;
        d.slot = s.slot;
        d.sym = s.sym;
        const AllocBlock* ab = t_.allocBlock(d.slotIdx);
        d.numAllocs = ab ? ab->numAllocs : 0;
        d.flags = partialFlag ? kOruFlagPartial : 0;
        d.seq = static_cast<uint32_t>(dlSeq_++);
        if (t_.oruPublishDl(d)) {
            busy_[idx] = true;  // ORCA-owned until the dlReturn credit
            pubSfn_[idx] = s.sfn;
            pubSlot_[idx] = s.slot;
            pubSym_[idx] = s.sym;
            ++stats_.published;
        } else {
            // Doorbell ring full: the symbol is lost to ORCA — account it as
            // a drop, leave the slot reusable (never report it published).
            ++stats_.publishDrops;
        }
        s.inUse = false;
    }

    void sendUlSlot(const AllocBlock& ab, const ci16* z, const OruDoorbell& d,
                    const TxFn& tx) {
        uint8_t* pkt = txBuf_.data();  // sized once in the constructor
        for (uint16_t i = 0; i < ab.numAllocs; ++i) {
            const Alloc& a = ab.allocs[i];
            for (uint8_t l = 0; l < a.rank; ++l) {
                fh::FhHeader h{};
                h.ver = fh::kFhVersion;
                h.msgtyp = fh::MsgType::UPlane;
                h.dir = fh::FhDir::UL;
                h.cmp = fh::Cmp::Int16;
                h.iqWidth = 16;
                h.eAxC = fh::encodeEaxc(
                    fh::Eaxc{0,
                             static_cast<uint16_t>(a.cell >> cfg_.eaxc.ccIdBits),
                             static_cast<uint16_t>(
                                 a.cell & fh::mask(cfg_.eaxc.ccIdBits)),
                             l},
                    cfg_.eaxc);
                h.sectionId = static_cast<uint8_t>(i);
                h.numSections = static_cast<uint8_t>(ab.numAllocs);
                h.sfn = d.sfn;
                h.slot = d.slot;
                h.sym = d.sym;
                h.startPrb = a.scStart / dims::scPerPrb;
                h.numPrb = a.scLen / dims::scPerPrb;
                h.seqNum = static_cast<uint16_t>(ulSeq_++);
                h.udCompParam = 0;
                fh::pack(h, pkt);
                fh::packUplane(z + layout::idxXdl(a.cell, l, a.scStart),
                               a.scLen, pkt + fh::kHeaderBytes);
                tx(pkt, fh::kHeaderBytes + fh::uplaneBytes(h.numPrb));
                ++stats_.ulPackets;
            }
        }
    }

    OruLoopback& t_;
    Config cfg_;
    Stats stats_;
    SlotTrack slots_[dims::N_ring];
    bool busy_[dims::N_ring] = {};  // published, awaiting ORCA credit
    uint16_t pubSfn_[dims::N_ring] = {};  // key of the busy publication
    uint8_t pubSlot_[dims::N_ring] = {};
    uint8_t pubSym_[dims::N_ring] = {};
    Alloc pendingScratch_[dims::MAX_ALLOCS];
    std::vector<uint8_t> txBuf_;
    uint64_t nextClaim_ = 1;
    uint64_t dlSeq_ = 0;
    uint64_t ulSeq_ = 0;
};

}  // namespace orca
