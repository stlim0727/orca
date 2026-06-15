// Spec G §G.3 golden test: UE grid model (scenario/grid.hpp).
//   1. gpToPos at the origin and at interior points.
//   2. gpIndex flattening (2-D and 3-D conventions).
//   3. posToGp round-trips a grid index back to itself.
//   4. nearest-grid-point rounding (±0.4Δ stays, ±0.6Δ steps).
//   5. clamping out-of-bounds positions to the grid edge.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "channel/cir_table.hpp"
#include "scenario/grid.hpp"
#include "tests/check.hpp"

using namespace orca;
using namespace orca::scenario;
using orca::channel::CirHeader;

static bool nearD(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol;
}

// 2-D grid: 5×4, origin (10,20,0), spacing (2,3,0), ueHeight 1.5.
static CirHeader makeHdr2D() {
    CirHeader h{};
    std::memcpy(h.magic, "ORCACIR1", 8);
    h.version  = orca::channel::kCirVersion;
    h.numCells = 1;
    h.gridDims = 2;
    h.nx = 5; h.ny = 4; h.nz = 1;
    h.originX = 10.0f; h.originY = 20.0f; h.originZ = 0.0f;
    h.spacingX = 2.0f; h.spacingY = 3.0f; h.spacingZ = 0.0f;
    h.ueHeight = 1.5f;
    h.carrierHz = 3.5e9;
    return h;
}

// 3-D grid: 3×3×2, origin (0,0,0), spacing (1,1,4).
static CirHeader makeHdr3D() {
    CirHeader h = makeHdr2D();
    h.gridDims = 3;
    h.nx = 3; h.ny = 3; h.nz = 2;
    h.originX = 0.0f; h.originY = 0.0f; h.originZ = 0.0f;
    h.spacingX = 1.0f; h.spacingY = 1.0f; h.spacingZ = 4.0f;
    return h;
}

int main() {
    const CirHeader h = makeHdr2D();

    // --- 1. gpToPos ----------------------------------------------------------
    Vec3 p00 = gpToPos(h, GridIndex{0, 0, 0});
    CHECK(nearD(p00.x, 10.0) && nearD(p00.y, 20.0) && nearD(p00.z, 1.5));
    Vec3 p32 = gpToPos(h, GridIndex{3, 2, 0});
    CHECK(nearD(p32.x, 10.0 + 3 * 2.0) && nearD(p32.y, 20.0 + 2 * 3.0) &&
          nearD(p32.z, 1.5));

    // --- 2. gpIndex (2-D: iy·nx + ix) ----------------------------------------
    CHECK(gpIndex(h, GridIndex{0, 0, 0}) == 0u);
    CHECK(gpIndex(h, GridIndex{4, 0, 0}) == 4u);
    CHECK(gpIndex(h, GridIndex{0, 1, 0}) == 5u);          // iy=1 → nx
    CHECK(gpIndex(h, GridIndex{3, 2, 0}) == 2u * 5 + 3);  // = 13
    CHECK(gpIndex(h, GridIndex{4, 3, 0}) == h.nx * h.ny - 1);  // last gp = 19

    // --- 3. round-trip every grid point --------------------------------------
    for (uint32_t iy = 0; iy < h.ny; ++iy)
        for (uint32_t ix = 0; ix < h.nx; ++ix) {
            GridIndex g{ix, iy, 0};
            GridIndex back = posToGp(h, gpToPos(h, g));
            CHECK(back.ix == ix && back.iy == iy && back.iz == 0);
            CHECK(posToGpIndex(h, gpToPos(h, g)) == gpIndex(h, g));
        }

    // --- 4. nearest-grid-point rounding --------------------------------------
    // Around grid point (2,1): x0=14, y0=23. Δx=2, Δy=3.
    CHECK(posToGp(h, Vec3{14.0 + 0.4 * 2.0, 23.0, 1.5}).ix == 2);  // stays
    CHECK(posToGp(h, Vec3{14.0 + 0.6 * 2.0, 23.0, 1.5}).ix == 3);  // steps up
    CHECK(posToGp(h, Vec3{14.0 - 0.4 * 2.0, 23.0, 1.5}).ix == 2);  // stays
    CHECK(posToGp(h, Vec3{14.0 - 0.6 * 2.0, 23.0, 1.5}).ix == 1);  // steps down
    CHECK(posToGp(h, Vec3{14.0, 23.0 + 0.6 * 3.0, 1.5}).iy == 2);

    // --- 5. clamping ---------------------------------------------------------
    CHECK(posToGp(h, Vec3{-1000.0, -1000.0, 0.0}).ix == 0);
    CHECK(posToGp(h, Vec3{-1000.0, -1000.0, 0.0}).iy == 0);
    GridIndex hi = posToGp(h, Vec3{1e6, 1e6, 0.0});
    CHECK(hi.ix == h.nx - 1 && hi.iy == h.ny - 1);   // 4, 3
    // z is ignored on a 2-D grid (spacingZ==0) → iz stays 0.
    CHECK(posToGp(h, Vec3{12.0, 20.0, 9999.0}).iz == 0);

    // --- 6. 3-D grid ---------------------------------------------------------
    const CirHeader h3 = makeHdr3D();
    CHECK(gpIndex(h3, GridIndex{1, 2, 1}) == (1u * 3 + 2) * 3 + 1);  // 16
    Vec3 q = gpToPos(h3, GridIndex{2, 1, 1});
    CHECK(nearD(q.x, 2.0) && nearD(q.y, 1.0) && nearD(q.z, 4.0));  // iz·Δz=4
    for (uint32_t iz = 0; iz < h3.nz; ++iz)
        for (uint32_t iy = 0; iy < h3.ny; ++iy)
            for (uint32_t ix = 0; ix < h3.nx; ++ix) {
                GridIndex g{ix, iy, iz};
                GridIndex back = posToGp(h3, gpToPos(h3, g));
                CHECK(back.ix == ix && back.iy == iy && back.iz == iz);
            }
    // 3-D z clamps high.
    CHECK(posToGp(h3, Vec3{0.0, 0.0, 1e6}).iz == h3.nz - 1);  // 1

    return orca::test::report("test_grid");
}
