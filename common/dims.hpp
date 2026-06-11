#pragma once
// Phase-1 reference dimensions (Spec E §E.1) and derived layout constants.
// The cell dimension is present from Stage 1 (MILESTONES invariant) even while
// early stages run effectively single-flow.

#include <cstddef>
#include <cstdint>

namespace orca {

// Precondition: multiple > 0. Avoids the v + multiple - 1 addition overflow;
// the result itself must still fit uint32_t.
constexpr uint32_t roundUp(uint32_t v, uint32_t multiple) {
    return (v / multiple + (v % multiple != 0u ? 1u : 0u)) * multiple;
}

namespace dims {

constexpr uint32_t C        = 2;   // cells
constexpr uint32_t U_c      = 16;  // UEs per cell
constexpr uint32_t U        = C * U_c;  // total UEs = 32
constexpr uint32_t numTx    = 64;  // cell TRX
constexpr uint32_t numRx    = 4;   // UE rx antennas
constexpr uint32_t numUeTx  = 2;   // UE tx antennas (⊆ rx, Spec E §E.6)
constexpr uint32_t numPrb   = 273;
constexpr uint32_t scPerPrb = 12;
constexpr uint32_t numSc    = numPrb * scPerPrb;  // 3276
constexpr uint32_t rankMax  = 4;   // layers per SU resource (ADR 0005)

// Innermost-axis padding so every tensor row starts 128-B aligned (Spec E §E.11).
constexpr uint32_t numScP = roundUp(numSc, 32);  // 3296

constexpr uint32_t N_ring     = 4;    // symbol ring slots (Spec A §A.5, N ≥ 3)
constexpr uint32_t MAX_ALLOCS = 512;  // static-grid bound (Spec E §E.8)

// Stage-1 identity image: the cf32 image of x_dl[C][rankMax][numScP] carried
// at the head of the vUE DL/UL slots until the real K1–K4 kernels land.
constexpr uint32_t identityImageElems = C * rankMax * numScP;

// Numerology (Spec A §A.2): µ=1 target.
constexpr uint32_t mu          = 1;
constexpr uint32_t slotsPerFrame(uint32_t mu_) { return 10u << mu_; }
constexpr uint32_t symsPerSlot = 14;
constexpr uint32_t sfnPeriod   = 1024;  // SFN wraps mod 1024

static_assert(numSc == 3276, "Phase-1 numSc");
static_assert(numScP == 3296, "padded sc extent (Spec E §E.11)");
static_assert(numUeTx <= numRx, "reciprocity requires UE tx ⊆ rx (Spec E §E.6)");

}  // namespace dims
}  // namespace orca
