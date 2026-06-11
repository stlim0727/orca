#pragma once
// Minimal in-process vUE (AGENT.md 1f): consumes DL slots, copies them to UL
// slots (identity PHY), and signals per Spec D §D.7 — pollDl → process →
// returnDl credit; submitUl → (ORCA consumes) → reclaimUl credit.

#include <cstdint>
#include <cstring>

#include "common/dims.hpp"
#include "vue/vue_transport.hpp"

namespace orca {

class VueStub {
  public:
    explicit VueStub(VueTransport& t) : t_(t) {}

    // Drains pending DL doorbells; for each: identity-copy the Stage-1 image
    // into the UL slot, submit UL with the same symbol key, credit the DL
    // slot back. Backpressure → drop + count, never block (§D.7). Returns DL
    // symbols processed.
    uint32_t step() {
        uint32_t n = 0;
        VueDoorbell d{};
        while (t_.pollDl(d)) {
            const cf32* dl = t_.dlSlot(d.slotIdx);
            cf32* ul = t_.ulSlot(d.slotIdx);
            if (dl && ul) {
                std::memcpy(ul, dl,
                            sizeof(cf32) * dims::identityImageElems);
                if (t_.submitUl(d.slotIdx, d))  // same key/seq back
                    ++processed_;
                else
                    ++submitDrops_;  // UL ring full — dropped, not blocked
            }
            t_.returnDl(d.slotIdx, d.seq);
            ++n;
        }
        VueCredit c{};
        while (t_.reclaimUl(c)) {
        }  // UL slots recycled
        return n;
    }

    uint64_t processed() const { return processed_; }
    uint64_t submitDrops() const { return submitDrops_; }

  private:
    VueTransport& t_;
    uint64_t processed_ = 0;
    uint64_t submitDrops_ = 0;
};

}  // namespace orca
