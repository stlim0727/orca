#pragma once
// Spec D — ORCA ↔ vUE south interface: the VueTransport seam (§D.8) and the
// control-ring doorbell/credit descriptors (§D.3). ORCA hot-path code calls
// only this interface; the Phase-1 Linux backend (CUDA IPC + DPDK shm) and
// the Phase-2 NVLink backend swap behind it with the same slot layout
// (ADR 0004 §4).

#include <cstddef>
#include <cstdint>

#include "common/complex.hpp"
#include "common/dims.hpp"

namespace orca {

// Doorbell descriptor (§D.3) — fits a cache line.
struct VueDoorbell {
    uint16_t slotIdx;  // ring slot (0..N-1)
    uint16_t sfn;      // symbol key
    uint8_t slot;
    uint8_t sym;
    uint8_t flags;     // bit0 = partial/zero-filled (late)
    uint32_t seq;      // monotonic per-direction sequence
};
constexpr uint8_t kVueFlagPartial = 0x01;

// Credit (return-ring entry): {slotIdx, seq}.
struct VueCredit {
    uint16_t slotIdx;
    uint32_t seq;
};

// Shared-memory schemas — lock layout against compiler padding drift.
static_assert(sizeof(VueDoorbell) == 12, "VueDoorbell schema (Spec D §D.3)");
static_assert(offsetof(VueDoorbell, flags) == 6 && offsetof(VueDoorbell, seq) == 8,
              "VueDoorbell field offsets");
static_assert(sizeof(VueCredit) == 8 && offsetof(VueCredit, seq) == 4,
              "VueCredit schema");

// §D.8 — implemented by vue_loopback (host) and the CUDA-IPC backend.
// Non-blocking throughout: publish/submit return false when the ring is full
// (producer applies the §D.7 drop policy — overwrite oldest / mark late,
// never block); polls return false when nothing is pending.
class VueTransport {
  public:
    virtual ~VueTransport() = default;

    virtual bool attach() = 0;
    virtual void detach() = 0;

    // The slot index is an explicit parameter (Spec D §D.8 publishDl(slotIdx,
    // meta), submitUl(slotIdx, meta), returnUl(slotIdx)); meta.slotIdx is
    // overwritten by the backend — one source of truth for the bulk slot.

    // --- ORCA side ---
    virtual bool publishDl(uint16_t slotIdx, VueDoorbell meta) = 0;  // §D.7 DL-2,3
    virtual bool reclaimDl(VueCredit& out) = 0;                      // §D.7 DL-6
    virtual bool pollUl(VueDoorbell& out) = 0;                       // §D.7 UL consume
    virtual bool returnUl(uint16_t slotIdx, uint32_t seq) = 0;       // §D.7 UL credit

    // --- vUE side (used by the in-process stub / vUE program) ---
    virtual bool pollDl(VueDoorbell& out) = 0;                       // §D.7 DL-4
    virtual bool returnDl(uint16_t slotIdx, uint32_t seq) = 0;       // §D.7 DL-5
    virtual bool submitUl(uint16_t slotIdx, VueDoorbell meta) = 0;   // §D.7 UL produce
    virtual bool reclaimUl(VueCredit& out) = 0;                      // UL credit drain

    // --- bulk slot access (§D.2; HBM via CUDA IPC on target, host here) ---
    // Logical tensors r_dl[U][numRx][numSc] / x_ul[U][numUeTx][numSc], stored
    // with the padded numScP row stride (Spec E §E.11).
    virtual cf32* dlSlot(uint16_t slotIdx) = 0;
    virtual cf32* ulSlot(uint16_t slotIdx) = 0;

    static constexpr uint32_t kDlSlotElems = dims::U * dims::numRx * dims::numScP;
    static constexpr uint32_t kUlSlotElems = dims::U * dims::numUeTx * dims::numScP;
};

}  // namespace orca
