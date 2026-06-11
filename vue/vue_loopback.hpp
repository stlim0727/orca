#pragma once
// Host-only, in-process VueTransport backend (AGENT.md 1c). Stands in for the
// Spec D CUDA-IPC + DPDK-shm mechanism with the *exact* §D.3 descriptor
// schemas; the Linux/CUDA backend is a drop-in swap (ADR 0004 §4). Bulk slots
// live in host memory here (HBM on target); descriptors and credits flow
// through the same four rings.

#include <cstdint>
#include <vector>

#include "common/spsc_ring.hpp"
#include "vue/vue_transport.hpp"

namespace orca {

class VueLoopback final : public VueTransport {
  public:
    static constexpr uint32_t kRingDepth = 8;  // ≥ N_ring, power of 2

    bool attach() override {
        dl_.assign(size_t{dims::N_ring} * kDlSlotElems, cf32{0.0f, 0.0f});
        ul_.assign(size_t{dims::N_ring} * kUlSlotElems, cf32{0.0f, 0.0f});
        attached_ = true;
        return true;
    }

    void detach() override {
        attached_ = false;
        dl_.clear();
        ul_.clear();
    }

    // --- ORCA side (§D.7) ---
    bool publishDl(uint16_t slotIdx, VueDoorbell meta) override {
        if (!attached_) return false;
        meta.slotIdx = slotIdx;  // the parameter is the source of truth
        return dlDoorbell_.push(meta);
    }
    bool reclaimDl(VueCredit& out) override { return attached_ && dlReturn_.pop(out); }
    bool pollUl(VueDoorbell& out) override { return attached_ && ulDoorbell_.pop(out); }
    bool returnUl(uint16_t slotIdx, uint32_t seq) override {
        return attached_ && ulReturn_.push(VueCredit{slotIdx, seq});
    }

    // --- vUE side (§D.7) ---
    bool pollDl(VueDoorbell& out) override { return attached_ && dlDoorbell_.pop(out); }
    bool returnDl(uint16_t slotIdx, uint32_t seq) override {
        return attached_ && dlReturn_.push(VueCredit{slotIdx, seq});
    }
    bool submitUl(uint16_t slotIdx, VueDoorbell meta) override {
        if (!attached_) return false;
        meta.slotIdx = slotIdx;
        return ulDoorbell_.push(meta);
    }
    bool reclaimUl(VueCredit& out) override { return attached_ && ulReturn_.pop(out); }

    cf32* dlSlot(uint16_t slotIdx) override {
        return attached_ && slotIdx < dims::N_ring
                   ? dl_.data() + size_t{slotIdx} * kDlSlotElems
                   : nullptr;
    }
    cf32* ulSlot(uint16_t slotIdx) override {
        return attached_ && slotIdx < dims::N_ring
                   ? ul_.data() + size_t{slotIdx} * kUlSlotElems
                   : nullptr;
    }

  private:
    bool attached_ = false;
    std::vector<cf32> dl_, ul_;
    SpscRing<VueDoorbell, kRingDepth> dlDoorbell_, ulDoorbell_;
    SpscRing<VueCredit, kRingDepth> dlReturn_, ulReturn_;
};

}  // namespace orca
