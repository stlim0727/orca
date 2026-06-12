#pragma once
// Spec G §G.3 — UE grid model. Maps between grid indices (ix,iy[,iz]), the flat
// grid-point index `gp` the CIR table is keyed on (cir_table.hpp linkOffset),
// and world positions (m). Phase 1 default is a 2-D horizontal grid at a fixed
// `ueHeight`; 3-D (gridDims==3) is format-compatible.
//
//   2-D: gp = iy·nx + ix          pos = (originX+ix·Δx, originY+iy·Δy, ueHeight)
//   3-D: gp = (iz·ny+iy)·nx + ix  pos = origin + (ix·Δx, iy·Δy, iz·Δz)
//
// A UE move is a change in `gp` (slow-plane event, Spec G §G.8). posToGp uses the
// NEAREST grid point — no interpolation (Spec G §G.9) — clamped to grid bounds.

#include <cmath>
#include <cstdint>

#include "channel/cir_table.hpp"  // CirHeader
#include "scenario/vec3.hpp"

namespace orca {
namespace scenario {

struct GridIndex {
    uint32_t ix, iy, iz;
};

// Flat grid-point index — matches the `gp` used by cir_table.hpp linkOffset.
inline uint64_t gpIndex(const channel::CirHeader& h, const GridIndex& g) {
    return (h.gridDims == 3)
               ? (uint64_t{g.iz} * h.ny + g.iy) * h.nx + g.ix
               : uint64_t{g.iy} * h.nx + g.ix;
}

// Grid index → world position (m).
inline Vec3 gpToPos(const channel::CirHeader& h, const GridIndex& g) {
    const double x = double{h.originX} + double{g.ix} * double{h.spacingX};
    const double y = double{h.originY} + double{g.iy} * double{h.spacingY};
    const double z = (h.gridDims == 3)
                         ? double{h.originZ} + double{g.iz} * double{h.spacingZ}
                         : double{h.ueHeight};
    return Vec3{x, y, z};
}

// Round to nearest grid index along one axis, clamped to [0, n-1].
// (n ≥ 1 is guaranteed by the loader's zero-extent check, cir_loader.cpp.)
inline uint32_t nearestIdx(double coord, uint32_t n) {
    if (coord <= 0.0) return 0;
    const double r = std::floor(coord + 0.5);  // round half up
    if (r >= double{n}) return n - 1;
    return static_cast<uint32_t>(r);
}

// World position → nearest grid index (clamped, Spec G §G.9). Axes with zero
// spacing (e.g. z on a 2-D grid) collapse to index 0.
inline GridIndex posToGp(const channel::CirHeader& h, const Vec3& p) {
    const uint32_t ix =
        (h.spacingX > 0.0f)
            ? nearestIdx((p.x - double{h.originX}) / double{h.spacingX}, h.nx)
            : 0u;
    const uint32_t iy =
        (h.spacingY > 0.0f)
            ? nearestIdx((p.y - double{h.originY}) / double{h.spacingY}, h.ny)
            : 0u;
    const uint32_t iz =
        (h.gridDims == 3 && h.spacingZ > 0.0f)
            ? nearestIdx((p.z - double{h.originZ}) / double{h.spacingZ}, h.nz)
            : 0u;
    return GridIndex{ix, iy, iz};
}

// World position → flat gp index (nearest grid point, clamped).
inline uint64_t posToGpIndex(const channel::CirHeader& h, const Vec3& p) {
    return gpIndex(h, posToGp(h, p));
}

}  // namespace scenario
}  // namespace orca
