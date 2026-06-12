#pragma once
// Scenario runtime state (ADR 0002 §4/§5, Spec G §G.8) — the slow-plane object
// that owns per-UE kinematics and keeps the resident channel in sync as UEs move.
//
// It ties together the host slow-plane pieces:
//   - grid.hpp            position → grid point
//   - channel_refresh.hpp grid point → expand H_dl[c][u] + Doppler phaseInc
//   - association.hpp     serving cell + interferer set (+ handover)
//   - channel/doppler.hpp per-symbol rotor build
//
// On a UE move it refreshes ONLY that UE's links and re-associates (ADR 0002 §5):
// a grid-point change triggers a slow-plane update; a serving-cell change is a
// handover. The hot path reads only the resident H_dl and the per-symbol rotors;
// it never touches this object. The caller owns H_dl (the indirection-cell back
// buffer) and publishes it after a batch of moves.

#include <cstdint>

#include "channel/cir_loader.hpp"
#include "channel/doppler.hpp"  // buildDopplerRotors, dopplerIdx
#include "common/complex.hpp"
#include "common/dims.hpp"
#include "scenario/association.hpp"
#include "scenario/channel_refresh.hpp"
#include "scenario/grid.hpp"
#include "scenario/vec3.hpp"

namespace orca {
namespace scenario {

class ScenarioState {
  public:
    static constexpr uint64_t kNoGp = ~uint64_t{0};  // "not yet placed"

    // Bind to a loaded CIR table and the H_dl back buffer the slow plane fills.
    // tSymSec = symbol period (s) for the Doppler increment (orchestr/timing.hpp).
    void init(const channel::CirTable* tbl, half2c* H_dl, double tSymSec) {
        tbl_  = tbl;
        H_dl_ = H_dl;
        tSym_ = tSymSec;
        for (uint32_t u = 0; u < dims::U; ++u) {
            pos_[u] = Vec3{0, 0, 0};
            vel_[u] = Vec3{0, 0, 0};
            gp_[u]  = kNoGp;
            assoc_[u] = UeAssoc{kNoCell, 0, {}};
        }
        for (double& v : phaseInc_) v = 0.0;
    }

    // Place / update a UE's kinematics and refresh its links unconditionally
    // (position and/or velocity may have changed → Doppler must be recomputed).
    void setUe(uint32_t u, const Vec3& pos, const Vec3& vel) {
        pos_[u] = pos;
        vel_[u] = vel;
        refreshUeState(u);
    }

    // Move a UE (position only, velocity unchanged). Refreshes its links + re-
    // associates ONLY if the grid point changed (ADR 0002 §5). Returns true when
    // a slow-plane update occurred. Within one grid cell, the channel + Doppler
    // are unchanged (fixed dominant-path geometry, fixed velocity), so nothing
    // is recomputed — the per-symbol rotor still advances the phase each symbol.
    // To change a UE's velocity (and hence its Doppler) WITHOUT a grid move, call
    // setUe() — moveUe() will not pick up a velocity change inside one cell.
    bool moveUe(uint32_t u, const Vec3& pos) {
        const uint64_t newGp = posToGpIndex(tbl_->header(), pos);
        pos_[u] = pos;
        if (newGp == gp_[u]) return false;
        refreshUeState(u);
        return true;
    }

    // Full rebuild from current per-UE positions/velocities (startup / after a
    // table swap). The caller sets kinematics via setUe first, or accepts the
    // init() defaults (all UEs at the origin).
    void rebuildAll() {
        for (uint32_t u = 0; u < dims::U; ++u) refreshUeState(u);
    }

    // Per-symbol fast-plane hook: rot[c*U+u] = e^{j·phaseInc[c][u]·symbolCtr}.
    void buildRotors(uint64_t symbolCtr, cf32* rot) const {
        buildDopplerRotors(phaseInc_, symbolCtr, rot);
    }

    // Accessors.
    const UeAssoc& assoc(uint32_t u) const { return assoc_[u]; }
    uint32_t servingCell(uint32_t u) const { return assoc_[u].servingCell; }
    uint64_t gridPoint(uint32_t u) const { return gp_[u]; }
    const Vec3& position(uint32_t u) const { return pos_[u]; }
    const double* phaseInc() const { return phaseInc_; }

  private:
    // Recompute everything that depends on UE u's grid point + velocity: cache
    // gp, expand H_dl[c][u] + Doppler phaseInc (channel_refresh), and the
    // serving-cell / interferer association.
    void refreshUeState(uint32_t u) {
        gp_[u] = posToGpIndex(tbl_->header(), pos_[u]);
        refreshUe(H_dl_, phaseInc_, *tbl_, u, pos_[u], vel_[u], tSym_);
        assoc_[u] = computeAssoc(*tbl_, gp_[u]);
    }

    const channel::CirTable* tbl_  = nullptr;
    half2c*                  H_dl_ = nullptr;
    double                   tSym_ = 0.0;

    Vec3     pos_[dims::U]{};
    Vec3     vel_[dims::U]{};
    uint64_t gp_[dims::U];
    UeAssoc  assoc_[dims::U]{};
    double   phaseInc_[dims::C * dims::U]{};
};

}  // namespace scenario
}  // namespace orca
