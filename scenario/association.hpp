#pragma once
// ADR 0002 §3/§4 — serving-cell + interferer association (scenario layer).
//
// Each UE has one SERVING cell (its desired DL signal / UL grant target) and a
// CONTRIBUTOR set (serving + interferers) that the channel-apply sums over. The
// association is derived from the per-link aggregate path loss the CIR table
// stores per (cell, grid-point) (Spec G §G.5/§G.7): the strongest link (lowest
// pathlossDb) serves; the top-K by path loss are the contributors.
//
// This is slow-plane state (ADR 0002 §4): recomputed on a UE grid move and
// republished, never on the hot path. A change in servingCell across a move IS
// a handover. Phase 1 runs all-to-all (maxK ≥ C, ADR 0005), so the contributor
// set is every covered cell; top-K (maxK < C) bounds it for scale.

#include <cmath>
#include <cstdint>
#include <limits>

#include "channel/cir_loader.hpp"
#include "channel/cir_table.hpp"
#include "common/dims.hpp"

namespace orca {
namespace scenario {

constexpr uint32_t kNoCell = 0xFFFFFFFFu;  // no covered serving cell (outage)

struct UeAssoc {
    uint32_t servingCell;          // strongest link, or kNoCell if all outage
    uint32_t numContrib;           // entries in contrib[]
    uint32_t contrib[dims::C];     // contributor cells, strongest-first
};

// Aggregate path loss (dB) of the (cell, gp) link; +inf for a missing/noCoverage
// link (never serves, excluded from contributors).
inline float linkPathLossDb(const channel::CirTable& tbl, uint32_t c,
                            uint64_t gp) {
    const channel::LinkBlockHeader* blk = tbl.linkBlock(c, gp);
    const bool covered =
        blk != nullptr && !(blk->flags & channel::kLinkNoCoverage);
    return covered ? blk->pathlossDb
                   : std::numeric_limits<float>::infinity();
}

// Compute the association for a UE at grid point `gp`: rank cells by (path loss
// ascending, cell index) — strongest first, ties broken by lower cell index —
// serving = strongest covered cell, contributors = the top-K covered cells
// (K = min(maxK, C)). Deterministic for any C. All links outage → servingCell =
// kNoCell, numContrib = 0.
inline UeAssoc computeAssoc(const channel::CirTable& tbl, uint64_t gp,
                            uint32_t maxK = dims::C) {
    struct CellLoss {
        uint32_t cell;
        float    lossDb;
        bool     covered;
    };
    CellLoss cl[dims::C];
    for (uint32_t c = 0; c < dims::C; ++c) {
        const float loss = linkPathLossDb(tbl, c, gp);
        cl[c] = CellLoss{c, loss, std::isfinite(loss)};
    }

    // Sort ascending by (lossDb, cell): path loss first, cell index as the tie
    // break so equal-loss cells keep ascending order. Deterministic for any C
    // (a plain min-only swap would not be stable once C > 2). C is tiny.
    for (uint32_t i = 0; i < dims::C; ++i)
        for (uint32_t j = i + 1; j < dims::C; ++j)
            if (cl[j].lossDb < cl[i].lossDb ||
                (cl[j].lossDb == cl[i].lossDb && cl[j].cell < cl[i].cell)) {
                CellLoss t = cl[i];
                cl[i] = cl[j];
                cl[j] = t;
            }

    UeAssoc a{};
    a.servingCell = cl[0].covered ? cl[0].cell : kNoCell;
    const uint32_t k = (maxK < dims::C) ? maxK : dims::C;
    uint32_t n = 0;
    for (uint32_t i = 0; i < k; ++i)
        if (cl[i].covered) a.contrib[n++] = cl[i].cell;
    a.numContrib = n;
    return a;
}

// Handover predicate: did the serving cell change between two associations?
inline bool isHandover(const UeAssoc& prev, const UeAssoc& cur) {
    return prev.servingCell != cur.servingCell;
}

}  // namespace scenario
}  // namespace orca
