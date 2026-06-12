#pragma once
// Spec G §G.8 — slow-plane ray→H expansion.
// Fills the H_dl back buffer for one (cell c, UE u) link from the stored ray
// list, then the caller swaps via the ADR 0001 §4 indirection cell.

#include <cstdint>

#include "channel/cir_table.hpp"
#include "common/complex.hpp"

namespace orca {
namespace channel {

// Subcarrier spacing for µ=1 (30 kHz), Phase-1 constant (Spec A §A.2).
constexpr double kScSpacingHz = 30'000.0;

// Expand one link (cell c, UE u at its current grid point) into H_dl.
// H_dl is the Spec E §E.11 flat half2c tensor indexed by layout::idxH.
// The (c, u) slice is zeroed first, then all paths are accumulated.
//
// Phase-1 constraints honoured here:
//   - ULA steering model (arrayType=0) for both cell and UE arrays.
//   - Cell numTx must equal dims::numTx (64); UE numRx must equal dims::numRx (4).
//   - noCoverage links → zeroed slice, no further work.
//   - f_sc = carrierHz + (sc - numSc/2) * kScSpacingHz (µ=1 absolute frequency).
void expandLink(half2c* H_dl, uint32_t c, uint32_t u,
                const CirHeader& hdr, const CellDesc& cell,
                const UeArrayDesc& ue,
                const LinkBlockHeader* blk, const PathRecord* paths);

}  // namespace channel
}  // namespace orca
