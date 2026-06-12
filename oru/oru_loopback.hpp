#pragma once
// Host-only, in-process OruTransport backend (AGENT.md 1c). Stands in for the
// Spec F host-shm + DPDK rings so ORCA's hot path can run end-to-end on any
// C++17 box. Uses the *exact* §F.4 descriptor schemas — the Linux backend is
// a drop-in swap. The `oru*()` methods are the ORU-process-side endpoint,
// used by the in-process ORU stub / tests (a real ORU process owns them via
// the shm mapping instead).

#include <cstdint>
#include <vector>

#include "common/spsc_ring.hpp"
#include "oru/oru_transport.hpp"

namespace orca {

class OruLoopback final : public OruTransport {
  public:
    static constexpr uint32_t kRingDepth = 8;  // ≥ N_ring, power of 2

    bool attach() override {
        dl_.assign(size_t{dims::N_ring} * kSlotElems, ci16{0, 0});
        ul_.assign(size_t{dims::N_ring} * kSlotElems, ci16{0, 0});
        allocs_.assign(dims::N_ring, AllocBlock{});
        attached_ = true;
        return true;
    }

    void detach() override {
        attached_ = false;
        dl_.clear();
        ul_.clear();
        allocs_.clear();
    }

    // --- ORCA side (OruTransport, §F.8) ---
    bool pollDl(OruDoorbell& out) override { return attached_ && dlDoorbell_.pop(out); }
    bool returnDl(uint16_t slotIdx, uint32_t seq) override {
        return attached_ && dlReturn_.push(OruCredit{slotIdx, seq});
    }
    bool publishUl(uint16_t slotIdx, OruDoorbell meta) override {
        if (!attached_) return false;
        meta.slotIdx = slotIdx;  // the parameter is the source of truth
        return ulDoorbell_.push(meta);
    }
    bool reclaimUl(OruCredit& out) override { return attached_ && ulReturn_.pop(out); }

    ci16* dlSlot(uint16_t slotIdx) override {
        return attached_ && slotIdx < dims::N_ring
                   ? dl_.data() + size_t{slotIdx} * kSlotElems
                   : nullptr;
    }
    ci16* ulSlot(uint16_t slotIdx) override {
        return attached_ && slotIdx < dims::N_ring
                   ? ul_.data() + size_t{slotIdx} * kSlotElems
                   : nullptr;
    }
    AllocBlock* allocBlock(uint16_t slotIdx) override {
        return attached_ && slotIdx < dims::N_ring ? &allocs_[slotIdx] : nullptr;
    }

    // --- ORU-process side (loopback counterpart, §F.6 producer/consumer) ---
    bool oruPublishDl(const OruDoorbell& d) { return attached_ && dlDoorbell_.push(d); }
    bool oruReclaimDl(OruCredit& out) { return attached_ && dlReturn_.pop(out); }
    bool oruPollUl(OruDoorbell& out) { return attached_ && ulDoorbell_.pop(out); }
    bool oruReturnUl(const OruCredit& c) { return attached_ && ulReturn_.push(c); }

  private:
    bool attached_ = false;
    std::vector<ci16> dl_, ul_;
    std::vector<AllocBlock> allocs_;
    SpscRing<OruDoorbell, kRingDepth> dlDoorbell_, ulDoorbell_;
    SpscRing<OruCredit, kRingDepth> dlReturn_, ulReturn_;
};

}  // namespace orca
