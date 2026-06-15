// Golden test: ScenarioState (scenario/scenario_state.hpp) — the slow-plane
// container that keeps H_dl, Doppler phaseInc, and serving-cell association in
// sync as UEs move, and builds per-symbol rotors. 2-cell 2×2 table, LoS-broadside
// paths (H==gain), distinct per-(cell,gp) gains + path loss.

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
#include "scenario/doppler_handoff.hpp"
#include "scenario/scenario_state.hpp"
#include "tests/check.hpp"

using namespace orca;
using namespace orca::scenario;
using namespace orca::channel;

static bool near(float a, float b, float tol = 1e-3f) { return std::fabs(a - b) <= tol; }
static bool nearD(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

static const double kFc   = 3.5e9;
static const double kTSym = 35.7e-6;

static PathRecord los(float gain) {
    PathRecord p{};
    p.gainRe = gain; p.flags = kPathLoS;  // tau=0, angles 0 → H = gain everywhere
    return p;
}
static void checkSlice(const half2c* H, uint32_t c, uint32_t u, float re) {
    const size_t pts[][3] = {{0,0,0}, {2,40,2000}, {3,63,3275}};
    for (const auto& q : pts) {
        const cf32 h = toCf32(H[layout::idxH(c, u, (uint32_t)q[0], (uint32_t)q[1],
                                             (uint32_t)q[2])]);
        CHECK(near(h.re, re) && near(h.im, 0.0f));
    }
}

static CirHeader makeHdr() {
    CirHeader h{};
    std::memcpy(h.magic, "ORCACIR1", 8);
    h.version = kCirVersion; h.numCells = dims::C; h.gridDims = 2;
    h.nx = 2; h.ny = 2; h.nz = 1;
    h.spacingX = h.spacingY = 1.0f; h.ueHeight = 1.5f; h.carrierHz = kFc;
    h.pMax = 4; h.pathRecBytes = sizeof(PathRecord); h.linkBlockBytes = linkBlockStride(4);
    return h;
}
static CellDesc makeCell() { CellDesc c{}; c.numTx = dims::numTx; c.arrayType = 0; c.elemSpacing = 0.0428f; return c; }
static UeArrayDesc makeUe() {
    UeArrayDesc u{}; u.numRx = dims::numRx; u.numUeTx = dims::numUeTx;
    u.ueTxToRx[0] = 0; u.ueTxToRx[1] = 1; u.arrayType = 0; u.elemSpacing = 0.0428f; return u;
}

int main() {
    const std::string fname = "scenstate_test.cir";

    CirHeader hdr = makeHdr();
    CellDesc cells[dims::C] = {makeCell(), makeCell()};
    CirWriter w(hdr, cells, makeUe());
    PathRecord p;
    // gp0: cell0 gain1.0 loss80; cell1 gain0.5 loss70 (serving=1)
    p = los(1.00f); w.setLink(0, 0, 1, 0, 80.f, &p);
    p = los(0.50f); w.setLink(1, 0, 1, 0, 70.f, &p);
    // gp3: cell0 gain0.25 loss60 (serving=0); cell1 gain0.75 loss90
    p = los(0.25f); w.setLink(0, 3, 1, 0, 60.f, &p);
    p = los(0.75f); w.setLink(1, 3, 1, 0, 90.f, &p);
    CHECK(w.write(fname));

    CirTable tbl;
    std::string err;
    CHECK(tbl.load(fname, err));
    if (!tbl.valid()) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }

    std::vector<half2c> H(layout::elemsH, half2c{{0}, {0}});
    ScenarioState st;
    st.init(&tbl, H.data(), kTSym);

    const Vec3 velX{30.0, 0.0, 0.0};
    const double expDphi = dopplerPhaseInc(0.0, 0.0, velX, kFc, kTSym);

    // --- place UE0 at gp0 -----------------------------------------------------
    st.setUe(0, Vec3{0, 0, 1.5}, velX);
    CHECK(st.gridPoint(0) == 0);
    CHECK(st.servingCell(0) == 1);                  // cell1 stronger at gp0
    checkSlice(H.data(), 0, 0, 1.00f);
    checkSlice(H.data(), 1, 0, 0.50f);
    CHECK(nearD(st.phaseInc()[dopplerIdx(0, 0)], expDphi));

    // --- per-symbol rotor build ----------------------------------------------
    cf32 rot[dims::C * dims::U];
    st.buildRotors(/*symbolCtr=*/0, rot);
    CHECK(near(rot[dopplerIdx(0, 0)].re, 1.0f) && near(rot[dopplerIdx(0, 0)].im, 0.0f));
    st.buildRotors(/*symbolCtr=*/2, rot);
    CHECK(near(rot[dopplerIdx(0, 0)].re, (float)std::cos(2 * expDphi)));
    CHECK(near(rot[dopplerIdx(0, 0)].im, (float)std::sin(2 * expDphi)));

    // --- intra-cell move: same grid point → no slow-plane work ---------------
    CHECK(st.moveUe(0, Vec3{0.3, 0.3, 1.5}) == false);  // still nearest gp0
    CHECK(st.gridPoint(0) == 0);
    checkSlice(H.data(), 0, 0, 1.00f);                  // unchanged

    // --- move to gp3 → refresh + handover (serving 1 → 0) --------------------
    CHECK(st.moveUe(0, Vec3{1.0, 1.0, 1.5}) == true);
    CHECK(st.gridPoint(0) == 3);
    CHECK(st.servingCell(0) == 0);                      // handover
    checkSlice(H.data(), 0, 0, 0.25f);
    checkSlice(H.data(), 1, 0, 0.75f);

    // --- rebuildAll: UE7 at gp0, UE9 at gp3 ----------------------------------
    st.setUe(7, Vec3{0, 0, 1.5}, velX);
    st.setUe(9, Vec3{1.0, 1.0, 1.5}, velX);
    st.rebuildAll();
    checkSlice(H.data(), 0, 7, 1.00f);                  // UE7 cell0 gp0
    checkSlice(H.data(), 1, 9, 0.75f);                  // UE9 cell1 gp3
    CHECK(st.servingCell(7) == 1);
    CHECK(st.servingCell(9) == 0);
    CHECK(nearD(st.phaseInc()[dopplerIdx(0, 7)], expDphi));

    std::remove(fname.c_str());

    return orca::test::report("test_scenario_state");
}
