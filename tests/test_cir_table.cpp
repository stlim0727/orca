// Spec G golden test: CIR table write/read roundtrip + ray→H expansion.
//   1. Build a 1-cell 2×2-grid table with 2 link entries (synthetic paths).
//   2. Serialize and reload; verify every header and link field.
//   3. Expand a LoS broadside path (tau=0, az=0): expect H=1+0j everywhere.
//   4. Expand a non-broadside path: verify steering phase varies across tx.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "channel/cir_expand.hpp"
#include "channel/cir_loader.hpp"
#include "channel/cir_table.hpp"
#include "channel/cir_writer.hpp"
#include "common/complex.hpp"
#include "common/dims.hpp"
#include "common/layout.hpp"

using namespace orca;
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

// ---- helpers ----------------------------------------------------------------

static CirHeader makeHdr(uint32_t pMax) {
    CirHeader h{};
    std::memcpy(h.magic, "ORCACIR1", 8);
    h.version         = kCirVersion;
    h.numCells        = 1;
    h.gridDims        = 2;
    h.nx = 2; h.ny = 2; h.nz = 1;
    h.originX = h.originY = h.originZ = 0.0f;
    h.spacingX = h.spacingY = 1.0f; h.spacingZ = 0.0f;
    h.ueHeight        = 1.5f;
    h.carrierHz       = 3.5e9;
    h.pMax            = pMax;
    h.pathRecBytes    = sizeof(PathRecord);
    h.linkBlockBytes  = linkBlockStride(pMax);
    h.normRefDb       = std::numeric_limits<float>::quiet_NaN();
    h.pruneThreshDb   = -30.0f;
    h.maxExcessDelayS = 2e-6f;
    return h;
}

static CellDesc makeCell() {
    CellDesc c{};
    c.pos[0] = 0; c.pos[1] = 0; c.pos[2] = 30.0f;
    c.boresight[0] = 0; c.boresight[1] = 1; c.boresight[2] = 0;
    c.numTx      = dims::numTx;
    c.arrayType  = 0;       // ULA
    c.elemSpacing = 0.0428f; // ≈ λ/2 at 3.5 GHz
    return c;
}

static UeArrayDesc makeUe() {
    UeArrayDesc u{};
    u.numRx       = dims::numRx;
    u.numUeTx     = dims::numUeTx;
    u.ueTxToRx[0] = 0; u.ueTxToRx[1] = 1;
    u.ueTxToRx[2] = 0; u.ueTxToRx[3] = 0;
    u.arrayType   = 0;
    u.elemSpacing = 0.0428f;
    return u;
}

// ---- test: roundtrip --------------------------------------------------------

static void testRoundtrip(const std::string& fname) {
    // --- write ---
    CirHeader hdr  = makeHdr(4);
    CellDesc  cell = makeCell();
    UeArrayDesc ue = makeUe();
    CirWriter writer(hdr, &cell, ue);

    // Link (0, gp=0): LoS broadside, tau=0, gain=1+0j.
    PathRecord p0{};
    p0.gainRe = 1.0f;
    p0.flags  = kPathLoS;
    writer.setLink(0, 0, 1, 0, -80.0f, &p0);

    // Link (0, gp=1): 1 path, 45-deg AoD, tau=100 ns.
    PathRecord p1{};
    p1.tau    = 100e-9f;
    p1.gainRe = 0.5f;
    p1.aodAz  = static_cast<float>(M_PI / 4.0);
    p1.aoaAz  = static_cast<float>(-M_PI / 6.0);
    writer.setLink(0, 1, 1, 0, -90.0f, &p1);

    // Link (0, gp=2): no coverage.
    writer.setLink(0, 2, 0, kLinkNoCoverage, 0.0f, nullptr);

    CHECK(writer.write(fname));

    // --- read ---
    CirTable table;
    std::string err;
    CHECK(table.load(fname, err));
    if (!table.valid()) {
        std::fprintf(stderr, "  load error: %s\n", err.c_str());
        return;
    }

    // Header fields.
    CHECK(table.header().numCells == 1);
    CHECK(table.header().nx == 2 && table.header().ny == 2 && table.header().nz == 1);
    CHECK(table.header().pMax == 4);
    CHECK(near(static_cast<float>(table.header().carrierHz), 3.5e9f, 1e3f));

    // Cell descriptor.
    CHECK(near(table.cell(0).elemSpacing, 0.0428f));
    CHECK(table.cell(0).numTx == dims::numTx);

    // UE descriptor.
    CHECK(table.ueArray().numRx == dims::numRx);
    CHECK(table.ueArray().ueTxToRx[0] == 0 && table.ueArray().ueTxToRx[1] == 1);

    // Link (0,0): LoS broadside.
    const LinkBlockHeader* blk0 = table.linkBlock(0, 0);
    CHECK(blk0 != nullptr);
    CHECK(blk0->numPaths == 1);
    CHECK(!(blk0->flags & kLinkNoCoverage));
    const PathRecord* rp0 = table.paths(blk0);
    CHECK(near(rp0[0].tau, 0.0f));
    CHECK(near(rp0[0].gainRe, 1.0f));
    CHECK(rp0[0].flags == kPathLoS);

    // Link (0,1): 45-deg path.
    const LinkBlockHeader* blk1 = table.linkBlock(0, 1);
    CHECK(blk1 != nullptr);
    CHECK(blk1->numPaths == 1);
    CHECK(near(table.paths(blk1)[0].tau, 100e-9f));

    // Link (0,2): no coverage.
    const LinkBlockHeader* blk2 = table.linkBlock(0, 2);
    CHECK(blk2 != nullptr);
    CHECK(blk2->flags & kLinkNoCoverage);

    // OOB gp=4 (grid is 2×2=4 points, so gp=4 is out-of-range).
    CHECK(table.linkBlock(0, 4) == nullptr);
}

// ---- test: expansion --------------------------------------------------------

static void testExpansion(const std::string& fname) {
    CirTable table;
    std::string err;
    if (!table.load(fname, err)) {
        std::fprintf(stderr, "  expansion test skipped: %s\n", err.c_str());
        ++failures;
        return;
    }

    // Allocate the full Spec E H_dl tensor.
    std::vector<half2c> H(layout::elemsH, half2c{{0}, {0}});

    // Expand LoS broadside link (c=0, u=0) into gp=0.
    // tau=0, az=0: all steering vectors = 1, delay phase = exp(0) = 1.
    // → H[0][0][rx][tx][sc] = 1+0j for all (rx, tx, sc).
    const LinkBlockHeader* blk0 = table.linkBlock(0, 0);
    expandLink(H.data(), 0, 0, table.header(), table.cell(0), table.ueArray(),
               blk0, table.paths(blk0));

    // Spot-check several (rx,tx,sc) combinations — all should be ≈ 1+0j.
    const size_t checks[][3] = {{0,0,0}, {1,32,1000}, {3,63,3275}};
    for (const auto& idx : checks) {
        const cf32 h = toCf32(H[layout::idxH(0, 0,
                                              static_cast<uint32_t>(idx[0]),
                                              static_cast<uint32_t>(idx[1]),
                                              static_cast<uint32_t>(idx[2]))]);
        CHECK(near(h.re, 1.0f, 1e-3f));
        CHECK(near(h.im, 0.0f, 1e-3f));
    }

    // H[1][0] (cell 1, UE 0): not touched by c=0 expansion → still zero.
    CHECK(near(toCf32(H[layout::idxH(1, 0, 0, 0, 0)]).re, 0.0f));

    // Expand the 45-degree path (c=0, u=1) from gp=1.
    // AoD=45°, dOverLambda ≈ 0.5, sin(45°) ≈ 0.7071
    // Phase per element ≈ 2π·0.5·0.7071 ≈ 2.22 rad → strong phase variation.
    const LinkBlockHeader* blk1 = table.linkBlock(0, 1);
    expandLink(H.data(), 0, 1, table.header(), table.cell(0), table.ueArray(),
               blk1, table.paths(blk1));

    // Elements at different tx must differ in phase.
    const cf32 h_tx0 = toCf32(H[layout::idxH(0, 1, 0, 0, 0)]);
    const cf32 h_tx1 = toCf32(H[layout::idxH(0, 1, 0, 1, 0)]);
    // Not identical — the steering phase rotates by ~2.22 rad per element.
    CHECK(!(near(h_tx0.re, h_tx1.re, 1e-3f) && near(h_tx0.im, h_tx1.im, 1e-3f)));

    // The LoS link (u=0) slice should be unchanged after expanding u=1.
    const cf32 h_los = toCf32(H[layout::idxH(0, 0, 0, 0, 0)]);
    CHECK(near(h_los.re, 1.0f, 1e-3f));

    // Expand a noCoverage link (gp=2): H[0][2] slice should be zeroed.
    const LinkBlockHeader* blk2 = table.linkBlock(0, 2);
    // Pre-fill with junk.
    H[layout::idxH(0, 2, 0, 0, 0)] = toHalf2(cf32{99.0f, 99.0f});
    expandLink(H.data(), 0, 2, table.header(), table.cell(0), table.ueArray(),
               blk2, table.paths(blk2));
    CHECK(near(toCf32(H[layout::idxH(0, 2, 0, 0, 0)]).re, 0.0f));
}

// ---- main -------------------------------------------------------------------

int main() {
    const std::string fname = "test_cir_roundtrip.bin";

    testRoundtrip(fname);
    testExpansion(fname);

    std::remove(fname.c_str());

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("test_cir_table: all checks passed");
    return EXIT_SUCCESS;
}
