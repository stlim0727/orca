// Spec G §G.8 golden test: slow-plane channel refresh driver
// (scenario/channel_refresh.hpp). Verifies the wiring grid → CIR lookup →
// expandLink + Doppler handoff, across a 2-cell 2×2-grid table.
//
// All test paths are LoS broadside (tau=0, AoD=AoA=0), so the expanded channel
// is H[c][u] = gain + 0j for every (rx,tx,sc) — a closed-form spot-check. AoA=0
// also lets a +x velocity produce a known nonzero Doppler.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "channel/cir_loader.hpp"
#include "channel/cir_table.hpp"
#include "channel/cir_writer.hpp"
#include "channel/doppler.hpp"
#include "common/complex.hpp"
#include "common/dims.hpp"
#include "common/layout.hpp"
#include "scenario/channel_refresh.hpp"
#include "scenario/doppler_handoff.hpp"

using namespace orca;
using namespace orca::scenario;
using namespace orca::channel;

static int failures = 0;

#define CHECK(cond) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++failures; \
    } } while (0)

static bool near(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) <= tol;
}
static bool nearD(double a, double b, double tol = 1e-12) {
    return std::fabs(a - b) <= tol;
}

static const double kFc   = 3.5e9;
static const double kTSym = 35.7e-6;  // value irrelevant to wiring; kept fixed

// 2-cell, 2×2 grid (gp: 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1)), origin 0, spacing 1.
static CirHeader makeHdr() {
    CirHeader h{};
    std::memcpy(h.magic, "ORCACIR1", 8);
    h.version  = kCirVersion;
    h.numCells = dims::C;        // 2 — must cover all runtime cells
    h.gridDims = 2;
    h.nx = 2; h.ny = 2; h.nz = 1;
    h.originX = h.originY = h.originZ = 0.0f;
    h.spacingX = h.spacingY = 1.0f; h.spacingZ = 0.0f;
    h.ueHeight = 1.5f;
    h.carrierHz = kFc;
    h.pMax = 4;
    h.pathRecBytes   = sizeof(PathRecord);
    h.linkBlockBytes = linkBlockStride(4);
    return h;
}
static CellDesc makeCell() {
    CellDesc c{};
    c.numTx = dims::numTx; c.arrayType = 0; c.elemSpacing = 0.0428f;
    return c;
}
static UeArrayDesc makeUe() {
    UeArrayDesc u{};
    u.numRx = dims::numRx; u.numUeTx = dims::numUeTx;
    u.ueTxToRx[0] = 0; u.ueTxToRx[1] = 1; u.arrayType = 0; u.elemSpacing = 0.0428f;
    return u;
}
static PathRecord losPath(float gain) {
    PathRecord p{};
    p.gainRe = gain;  // tau=0, all angles 0 → broadside, H = gain everywhere
    p.flags  = kPathLoS;
    return p;
}

// Spot-check that the H[c][u] slice equals (re,im) at a few (rx,tx,sc) points.
static void checkSlice(const half2c* H, uint32_t c, uint32_t u,
                       float re, float im) {
    const size_t pts[][3] = {{0,0,0}, {1,32,1000}, {3,63,3275}};
    for (const auto& q : pts) {
        const cf32 h = toCf32(H[layout::idxH(c, u,
                                             (uint32_t)q[0], (uint32_t)q[1],
                                             (uint32_t)q[2])]);
        CHECK(near(h.re, re));
        CHECK(near(h.im, im));
    }
}

int main() {
    const std::string fname = "refresh_test.cir";

    // --- build a 2-cell table -------------------------------------------------
    CirHeader hdr = makeHdr();
    CellDesc cells[dims::C] = {makeCell(), makeCell()};
    CirWriter w(hdr, cells, makeUe());
    PathRecord p;
    p = losPath(1.00f); w.setLink(0, 0, 1, 0, -80.f, &p);   // cell0 gp0 = 1.00
    p = losPath(0.50f); w.setLink(1, 0, 1, 0, -80.f, &p);   // cell1 gp0 = 0.50
    p = losPath(0.25f); w.setLink(0, 3, 1, 0, -80.f, &p);   // cell0 gp3 = 0.25
    p = losPath(0.75f); w.setLink(1, 3, 1, 0, -80.f, &p);   // cell1 gp3 = 0.75
    // cell0 gp1: noCoverage, but with a STALE nonzero path in the block header.
    // expandLink zeroes H here; the refresh driver must also force zero Doppler.
    p = losPath(0.90f); w.setLink(0, 1, 1, kLinkNoCoverage, -80.f, &p);
    p = losPath(0.50f); w.setLink(1, 1, 1, 0, -80.f, &p);   // cell1 gp1 = 0.50
    CHECK(w.write(fname));

    CirTable tbl;
    std::string err;
    CHECK(tbl.load(fname, err));
    if (!tbl.valid()) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }

    std::vector<half2c> H(layout::elemsH, half2c{{0}, {0}});
    double phaseInc[dims::C * dims::U];
    for (double& v : phaseInc) v = -999.0;  // sentinel

    const Vec3 velX{30.0, 0.0, 0.0};  // +x → closing on AoA az=0
    const double expDphi =
        dopplerPhaseInc(0.0, 0.0, velX, kFc, kTSym);  // aoa=0 path

    // --- 1. UE0 at gp0 → cell0=1.0, cell1=0.5; Doppler set on both links ------
    refreshUe(H.data(), phaseInc, tbl, 0, Vec3{0, 0, 1.5}, velX, kTSym);
    checkSlice(H.data(), 0, 0, 1.0f, 0.0f);
    checkSlice(H.data(), 1, 0, 0.5f, 0.0f);
    CHECK(nearD(phaseInc[dopplerIdx(0, 0)], expDphi));
    CHECK(nearD(phaseInc[dopplerIdx(1, 0)], expDphi));

    // --- 1b. perpendicular velocity → zero Doppler (velocity flows through) ---
    const Vec3 velY{0.0, 30.0, 0.0};  // ⊥ to AoA az=0 → dot=0 → Δφ=0
    refreshUe(H.data(), phaseInc, tbl, 0, Vec3{0, 0, 1.5}, velY, kTSym);
    CHECK(nearD(phaseInc[dopplerIdx(0, 0)], 0.0));   // distinct from expDphi
    checkSlice(H.data(), 0, 0, 1.0f, 0.0f);          // H unchanged at gp0

    // --- 2. move UE0 to gp3 → cell0=0.25, cell1=0.75 -------------------------
    refreshUe(H.data(), phaseInc, tbl, 0, Vec3{1.0, 1.0, 1.5}, velX, kTSym);
    checkSlice(H.data(), 0, 0, 0.25f, 0.0f);
    checkSlice(H.data(), 1, 0, 0.75f, 0.0f);

    // --- 3. UE0 at gp1: cell0 noCoverage → zeroed + zero Doppler; cell1=0.5 ---
    // The cell0/gp1 block has noCoverage set AND a stale path; coverage gating
    // must force BOTH H and Doppler to zero (velX would otherwise be nonzero).
    refreshUe(H.data(), phaseInc, tbl, 0, Vec3{1.0, 0.0, 1.5}, velX, kTSym);
    checkSlice(H.data(), 0, 0, 0.0f, 0.0f);              // zeroed slice
    CHECK(nearD(phaseInc[dopplerIdx(0, 0)], 0.0));       // noCoverage → 0 Doppler
    checkSlice(H.data(), 1, 0, 0.5f, 0.0f);

    // --- 4. clamp: huge position → nearest grid point gp3 --------------------
    refreshUe(H.data(), phaseInc, tbl, 0, Vec3{1e6, 1e6, 1.5}, velX, kTSym);
    checkSlice(H.data(), 0, 0, 0.25f, 0.0f);             // gp3 cell0
    checkSlice(H.data(), 1, 0, 0.75f, 0.0f);

    // --- 5. refreshAll: every UE at gp0 --------------------------------------
    std::vector<Vec3> pos(dims::U, Vec3{0, 0, 1.5});
    std::vector<Vec3> vel(dims::U, velX);
    refreshAll(H.data(), phaseInc, tbl, pos.data(), vel.data(), kTSym);
    checkSlice(H.data(), 0, 5, 1.0f, 0.0f);              // UE5 cell0 = 1.0
    checkSlice(H.data(), 1, 5, 0.5f, 0.0f);              // UE5 cell1 = 0.5
    CHECK(nearD(phaseInc[dopplerIdx(0, dims::U - 1)], expDphi));  // last UE set

    std::remove(fname.c_str());

    if (failures == 0) {
        std::printf("test_channel_refresh: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_channel_refresh: %d failure(s)\n", failures);
    return 1;
}
