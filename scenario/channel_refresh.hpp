#pragma once
// Spec G §G.8 — slow-plane channel refresh driver (scenario orchestration).
//
// On a UE move, this looks up the CIR table at the UE's grid point (scenario/
// grid.hpp), expands the affected (cell, ue) links' rays into the H_dl back
// buffer (channel/cir_expand.hpp), and forms the per-link Doppler phase
// increment (scenario/doppler_handoff.hpp). The caller then publishes H_dl via
// the ADR 0001 §4 indirection cell. This is slow-plane only — never the hot path
// (ADR 0002 §5) — so plain host loops over the affected links are fine.
//
// Layering: scenario → channel (the table + expansion primitive); the hot path
// reads only the resident H_dl, never this driver or the table.

#include <cstdint>

#include "channel/cir_expand.hpp"
#include "channel/cir_loader.hpp"
#include "channel/doppler.hpp"  // dopplerIdx
#include "common/complex.hpp"
#include "common/dims.hpp"
#include "scenario/doppler_handoff.hpp"
#include "scenario/grid.hpp"

namespace orca {
namespace scenario {

// Refresh one (cell c, UE u) link at grid point `gp`: expand H_dl[c][u] from the
// table's rays and set phaseInc[dopplerIdx(c,u)] from the dominant path + UE
// velocity. A missing/out-of-range block zeroes the slice (expandLink) and the
// Doppler (no paths). `tbl` must be valid() and cover cell c.
inline void refreshLink(half2c* H_dl, double* phaseInc,
                        const channel::CirTable& tbl,
                        uint32_t c, uint32_t u, uint64_t gp,
                        const Vec3& vel, double tSymSec) {
    const channel::LinkBlockHeader* blk = tbl.linkBlock(c, gp);
    // A covered link has a block AND no noCoverage flag. Gate BOTH the H
    // expansion and the Doppler on coverage so they never disagree (expandLink
    // zeroes the slice for a missing/noCoverage block; the Doppler must then be
    // zero too, regardless of any stale numPaths in the block header).
    const bool covered = blk != nullptr &&
                         !(blk->flags & channel::kLinkNoCoverage);
    const channel::PathRecord* paths = covered ? tbl.paths(blk) : nullptr;
    const uint16_t numPaths          = covered ? blk->numPaths : uint16_t{0};

    channel::expandLink(H_dl, c, u, tbl.header(), tbl.cell(c), tbl.ueArray(),
                        blk, paths);
    phaseInc[dopplerIdx(c, u)] = dopplerPhaseIncFromPaths(
        paths, numPaths, vel, tbl.header().carrierHz, tSymSec);
}

// Refresh all C links for one UE at world position `pos` (velocity `vel`). This
// is the "only the affected links" path for a single UE move (ADR 0002 §5): a UE
// occupies one grid point, seen by each cell c via the table's (c, gp) entry.
inline void refreshUe(half2c* H_dl, double* phaseInc,
                      const channel::CirTable& tbl, uint32_t u,
                      const Vec3& pos, const Vec3& vel, double tSymSec) {
    const uint64_t gp = posToGpIndex(tbl.header(), pos);
    for (uint32_t c = 0; c < dims::C; ++c)
        refreshLink(H_dl, phaseInc, tbl, c, u, gp, vel, tSymSec);
}

// Full rebuild: refresh every (cell, UE) link from per-UE positions/velocities
// (e.g. at startup). `pos` and `vel` are length-dims::U arrays.
inline void refreshAll(half2c* H_dl, double* phaseInc,
                       const channel::CirTable& tbl,
                       const Vec3* pos, const Vec3* vel, double tSymSec) {
    for (uint32_t u = 0; u < dims::U; ++u)
        refreshUe(H_dl, phaseInc, tbl, u, pos[u], vel[u], tSymSec);
}

}  // namespace scenario
}  // namespace orca
