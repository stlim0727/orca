// Spec G §G.8 golden test: Doppler handoff (scenario/doppler_handoff.hpp).
//   1. aoaUnit maps (az,el) to the right unit vectors.
//   2. f_d sign/magnitude: closing (+), receding (-), broadside (0).
//   3. Δφ = 2π·f_d·T_sym matches the closed form.
//   4. dominant-path convenience reads paths[0]; 0 paths → 0 Doppler.
//   5. integration: phaseInc → buildDopplerRotors gives e^{jΔφ·symbolCtr}.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "channel/cir_table.hpp"
#include "channel/doppler.hpp"
#include "common/complex.hpp"
#include "common/dims.hpp"
#include "orchestr/timing.hpp"
#include "scenario/doppler_handoff.hpp"

using namespace orca;
using namespace orca::scenario;

static int failures = 0;

#define CHECK(cond) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++failures; \
    } } while (0)

static bool nearD(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol;
}

int main() {
    constexpr double kPi = 3.14159265358979323846;
    const double fc      = 3.5e9;
    // T_sym (s) from the exact µ=1 symbol geometry (orchestr/timing.hpp).
    const double tSym = static_cast<double>(symTc()) /
                        static_cast<double>(kTcPerSec);

    // --- 1. aoaUnit ----------------------------------------------------------
    Vec3 u0 = aoaUnit(0.0, 0.0);           // +x
    CHECK(nearD(u0.x, 1.0) && nearD(u0.y, 0.0) && nearD(u0.z, 0.0));
    Vec3 u1 = aoaUnit(kPi / 2.0, 0.0);     // +y
    CHECK(nearD(u1.x, 0.0) && nearD(u1.y, 1.0) && nearD(u1.z, 0.0));
    Vec3 u2 = aoaUnit(0.0, kPi / 2.0);     // +z
    CHECK(nearD(u2.x, 0.0) && nearD(u2.y, 0.0) && nearD(u2.z, 1.0));

    // --- 2. Doppler shift sign & magnitude -----------------------------------
    const double speed   = 30.0;                  // m/s
    const double fdClose = (fc / kSpeedOfLight) * speed;  // expected closing f_d

    // UE moving toward an arrival at (az=0,el=0) → +x: closing, +f_d.
    CHECK(nearD(dopplerShiftHz(0.0, 0.0, Vec3{speed, 0, 0}, fc), fdClose, 1e-6));
    // Receding (−x velocity): −f_d.
    CHECK(nearD(dopplerShiftHz(0.0, 0.0, Vec3{-speed, 0, 0}, fc), -fdClose, 1e-6));
    // Perpendicular (broadside) velocity → 0.
    CHECK(nearD(dopplerShiftHz(0.0, 0.0, Vec3{0, speed, 0}, fc), 0.0, 1e-9));
    // Arrival along +y, velocity +y → closing.
    CHECK(nearD(dopplerShiftHz(kPi / 2.0, 0.0, Vec3{0, speed, 0}, fc),
                fdClose, 1e-6));

    // --- 3. Phase increment closed form --------------------------------------
    const double dphi = dopplerPhaseInc(0.0, 0.0, Vec3{speed, 0, 0}, fc, tSym);
    CHECK(nearD(dphi, kTwoPi * fdClose * tSym, 1e-12));
    // Broadside → zero phase increment.
    CHECK(nearD(dopplerPhaseInc(0.0, 0.0, Vec3{0, speed, 0}, fc, tSym), 0.0));

    // --- 4. dominant-path convenience ----------------------------------------
    channel::PathRecord p{};
    p.aoaAz = static_cast<float>(kPi / 2.0);  // +y
    p.aoaEl = 0.0f;
    const double dphiPath = dopplerPhaseIncFromPaths(&p, 1, Vec3{0, speed, 0},
                                                     fc, tSym);
    CHECK(nearD(dphiPath,
                dopplerPhaseInc(kPi / 2.0, 0.0, Vec3{0, speed, 0}, fc, tSym),
                1e-12));
    // No paths → no Doppler (outage / noCoverage link).
    CHECK(nearD(dopplerPhaseIncFromPaths(nullptr, 0, Vec3{speed, 0, 0}, fc, tSym),
                0.0));
    CHECK(nearD(dopplerPhaseIncFromPaths(&p, 0, Vec3{speed, 0, 0}, fc, tSym),
                0.0));

    // --- 5. integration with buildDopplerRotors (channel/doppler.hpp) --------
    // Put dphi on link (cell 0, ue 0); all others zero.
    double phaseInc[dims::C * dims::U] = {0.0};
    phaseInc[dopplerIdx(0, 0)] = dphi;
    cf32 rot[dims::C * dims::U];

    buildDopplerRotors(phaseInc, /*symbolCtr=*/0, rot);
    CHECK(nearD(rot[dopplerIdx(0, 0)].re, 1.0, 1e-6));   // e^{j·0} = 1
    CHECK(nearD(rot[dopplerIdx(0, 0)].im, 0.0, 1e-6));

    buildDopplerRotors(phaseInc, /*symbolCtr=*/1, rot);
    CHECK(nearD(rot[dopplerIdx(0, 0)].re, std::cos(dphi), 1e-6));
    CHECK(nearD(rot[dopplerIdx(0, 0)].im, std::sin(dphi), 1e-6));
    // A zero-Doppler link stays at 1+0j for every symbol.
    CHECK(nearD(rot[dopplerIdx(1, 5)].re, 1.0, 1e-6));
    CHECK(nearD(rot[dopplerIdx(1, 5)].im, 0.0, 1e-6));

    // symbolCtr=3 → angle triples.
    buildDopplerRotors(phaseInc, /*symbolCtr=*/3, rot);
    CHECK(nearD(rot[dopplerIdx(0, 0)].re, std::cos(3.0 * dphi), 1e-6));
    CHECK(nearD(rot[dopplerIdx(0, 0)].im, std::sin(3.0 * dphi), 1e-6));

    if (failures == 0) {
        std::printf("test_doppler_handoff: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_doppler_handoff: %d failure(s)\n", failures);
    return 1;
}
