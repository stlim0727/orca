// Integration golden test: the scenario slow-plane (ScenarioState) drives the
// DSP golden channel-apply. Proves the cross-layer seam — a real CIR table →
// ScenarioState fills H_dl + builds Doppler rotors → dsp::channelApplyDlGolden
// consumes them with consistent layout / dopplerIdx / victim-map conventions.
//
// Broadside LoS paths → H[c][u] = gain everywhere; power-of-two gains/IQ → the
// noiseless, zero-Doppler result is float-exact.

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
#include "dsp/channel_apply.hpp"
#include "scenario/doppler_handoff.hpp"
#include "scenario/scenario_state.hpp"
#include "scenario/victim_map.hpp"
#include "tests/check.hpp"

using namespace orca;
using namespace orca::scenario;
using namespace orca::channel;

static bool near(float a, float b, float tol = 1e-3f) { return std::fabs(a - b) <= tol; }

static const double kFc   = 3.5e9;
static const double kTSym = 35.7e-6;

static PathRecord los(float gain) {
    PathRecord p{};
    p.gainRe = gain; p.flags = kPathLoS;  // tau=0, angles 0 → H = gain everywhere
    return p;
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
    const std::string fname = "slowplane_test.cir";

    // gp0: cell0 gain 0.5, cell1 gain 0.25.  gp3: cell0 0.125, cell1 1.0.
    CirHeader hdr = makeHdr();
    CellDesc cells[dims::C] = {makeCell(), makeCell()};
    CirWriter w(hdr, cells, makeUe());
    PathRecord p;
    p = los(0.5f);   w.setLink(0, 0, 1, 0, 70.f, &p);
    p = los(0.25f);  w.setLink(1, 0, 1, 0, 80.f, &p);
    p = los(0.125f); w.setLink(0, 3, 1, 0, 80.f, &p);
    p = los(1.0f);   w.setLink(1, 3, 1, 0, 70.f, &p);
    CHECK(w.write(fname));

    CirTable tbl;
    std::string err;
    CHECK(tbl.load(fname, err));
    if (!tbl.valid()) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }

    std::vector<half2c> H(layout::elemsH, half2c{{0}, {0}});
    std::vector<cf32>   y(layout::elemsY, cf32{0, 0});
    std::vector<cf32>   rdl(layout::elemsRdl);
    std::vector<uint16_t> vDl(size_t{dims::C} * dims::numScP);
    cf32 rot[dims::C * dims::U];

    ScenarioState st;
    st.init(&tbl, H.data(), kTSym);

    // DL: cell0 schedules ue3 on sc [0, 1500). y[0]=2, y[1]=4 over that range
    // (cell1 radiates as interference). Both constant over (tx, sc).
    Alloc alloc{0, 3, 0, 1500, 0, 1, {0, 0, 0, 0}};
    CHECK(buildVictimMap(&alloc, 1, 0, vDl.data()) == 1);
    for (uint32_t sc = 0; sc < 1500; ++sc)
        for (uint32_t tx = 0; tx < dims::numTx; ++tx) {
            y[layout::idxY(0, tx, sc)] = cf32{2.0f, 0.0f};
            y[layout::idxY(1, tx, sc)] = cf32{4.0f, 0.0f};
        }

    // --- 1. UE3 at gp0, zero velocity → rotor 1, float-exact channel-apply ----
    // r_dl[3] = Σ_c2 1·(Σ_tx H[c2][3]·y[c2]) = 64·0.5·2 + 64·0.25·4 = 64 + 64 = 128
    st.setUe(3, Vec3{0, 0, 1.5}, Vec3{0, 0, 0});
    st.buildRotors(/*symbolCtr=*/0, rot);
    dsp::channelApplyDlGolden(H.data(), y.data(), vDl.data(), rot, 0, 0, 0.0f,
                              rdl.data());
    CHECK(near(rdl[layout::idxRdl(3, 0, 100)].re, 128.0f) &&
          near(rdl[layout::idxRdl(3, 0, 100)].im, 0.0f));
    CHECK(near(rdl[layout::idxRdl(3, 2, 1400)].re, 128.0f));  // another rx/sc

    // --- 2. move UE3 to gp3 → channel changes → output changes ----------------
    // r_dl[3] = 64·0.125·2 + 64·1.0·4 = 16 + 256 = 272
    CHECK(st.moveUe(3, Vec3{1.0, 1.0, 1.5}) == true);
    st.buildRotors(0, rot);
    dsp::channelApplyDlGolden(H.data(), y.data(), vDl.data(), rot, 0, 0, 0.0f,
                              rdl.data());
    CHECK(near(rdl[layout::idxRdl(3, 0, 100)].re, 272.0f) &&
          near(rdl[layout::idxRdl(3, 0, 100)].im, 0.0f));

    // --- 3. Doppler flows through: UE3 at gp0 with +x velocity, symbol 1 ------
    // base 128 (real) times the rotor e^{jΔφ}; both cells' links share Δφ
    // (same UE, same velocity, AoA=0) → r_dl = 128·(cosΔφ + j sinΔφ).
    const Vec3 velX{30.0, 0.0, 0.0};
    const double dphi = dopplerPhaseInc(0.0, 0.0, velX, kFc, kTSym);
    st.setUe(3, Vec3{0, 0, 1.5}, velX);
    st.buildRotors(/*symbolCtr=*/1, rot);
    dsp::channelApplyDlGolden(H.data(), y.data(), vDl.data(), rot, 0, 0, 0.0f,
                              rdl.data());
    CHECK(near(rdl[layout::idxRdl(3, 0, 100)].re, 128.0f * (float)std::cos(dphi)));
    CHECK(near(rdl[layout::idxRdl(3, 0, 100)].im, 128.0f * (float)std::sin(dphi)));

    std::remove(fname.c_str());

    return orca::test::report("test_slow_plane_pipeline");
}
