#pragma once
// Spec F — ORU ↔ ORCA north interface: the OruTransport seam (§F.8), the
// control-ring doorbell/credit descriptors (§F.4), and the per-symbol
// allocation block (§F.3 / Spec E §E.2). ORCA hot-path code calls only this
// interface; the Phase-1 Linux backend (host shm + DPDK + H2D/D2H) and the
// deferred DOCA backend are drop-in swaps (ADR 0007 §5).

#include <cstddef>
#include <cstdint>

#include "common/complex.hpp"
#include "common/dims.hpp"

namespace orca {

// One scheduled SU resource (Spec E §E.2 / Spec B §B.5 section).
struct Alloc {
    uint16_t cell;
    uint16_t ueId;
    uint16_t scStart;
    uint16_t scLen;
    uint8_t dir;   // 0 = DL, 1 = UL (Spec B §B.2)
    uint8_t rank;  // ≤ rankMax
    uint16_t beamId[4];
};

// Per-slot allocation map the ORU process fills from the C-plane (§F.3).
struct AllocBlock {
    uint16_t numAllocs;
    Alloc allocs[dims::MAX_ALLOCS];
};

// Doorbell descriptor (§F.4) — fits a cache line.
struct OruDoorbell {
    uint16_t slotIdx;    // host-ring slot
    uint16_t sfn;        // symbol key
    uint8_t slot;
    uint8_t sym;
    uint16_t numAllocs;  // entries in allocBlock[slotIdx]
    uint8_t flags;       // bit0 = partial/late
    uint32_t seq;        // per-direction sequence
};
constexpr uint8_t kOruFlagPartial = 0x01;

// Credit (return-ring entry).
struct OruCredit {
    uint16_t slotIdx;
    uint32_t seq;
};

// These structs are shared-memory schemas (the Linux backend maps them
// verbatim); lock their layout so padding drift across compilers is caught
// at build time.
static_assert(sizeof(Alloc) == 18, "Alloc schema (Spec F §F.3)");
static_assert(offsetof(Alloc, dir) == 8 && offsetof(Alloc, beamId) == 10,
              "Alloc field offsets");
static_assert(sizeof(AllocBlock) == 2 + sizeof(Alloc) * dims::MAX_ALLOCS,
              "AllocBlock schema (Spec F §F.3)");
static_assert(sizeof(OruDoorbell) == 16, "OruDoorbell schema (Spec F §F.4)");
static_assert(offsetof(OruDoorbell, numAllocs) == 6 &&
                  offsetof(OruDoorbell, flags) == 8 &&
                  offsetof(OruDoorbell, seq) == 12,
              "OruDoorbell field offsets");
static_assert(sizeof(OruCredit) == 8 && offsetof(OruCredit, seq) == 4,
              "OruCredit schema");

// §F.8 — implemented by oru_loopback (host) and the Linux backends.
// All calls are non-blocking: polls return false when nothing is pending,
// publishes return false when the ring is full (caller applies the Spec A
// §A.4 drop policy — never stall).
class OruTransport {
  public:
    virtual ~OruTransport() = default;

    virtual bool attach() = 0;
    virtual void detach() = 0;

    // --- ORCA side (consumer of DL, producer of UL) ---
    // The slot index is an explicit parameter (Spec F §F.8 returnDl(slotIdx),
    // publishUl(slotIdx, meta)) — it is the single source of truth for which
    // bulk slot is meant; meta.slotIdx is overwritten by the backend.
    virtual bool pollDl(OruDoorbell& out) = 0;                       // §F.6 DL-3
    virtual bool returnDl(uint16_t slotIdx, uint32_t seq) = 0;       // §F.6 DL-5
    virtual bool publishUl(uint16_t slotIdx, OruDoorbell meta) = 0;  // §F.6 UL-2
    virtual bool reclaimUl(OruCredit& out) = 0;                      // drain ulReturn

    // --- bulk slot access (host-staged ci16 layer IQ, §F.2) ---
    // Logical tensor [C][rank][numSc]; stored with the padded numScP row
    // stride (Spec E §E.11) so it mirrors the GPU x_dl/z layout exactly.
    virtual ci16* dlSlot(uint16_t slotIdx) = 0;  // x_dl_host
    virtual ci16* ulSlot(uint16_t slotIdx) = 0;  // z_host
    virtual AllocBlock* allocBlock(uint16_t slotIdx) = 0;

    // Elements per bulk slot (padded stride).
    static constexpr uint32_t kSlotElems = dims::C * dims::rankMax * dims::numScP;
};

}  // namespace orca
