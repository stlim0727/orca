#pragma once
// Spec G §G.8 — Doppler handoff (scenario layer).
//
// Forms the per-(cell, ue) link phase increment Δφ that the fast plane feeds to
// the Doppler rotor (channel/doppler.hpp → buildDopplerRotors). Phase 1 uses a
// SINGLE dominant-path rotor per link (Spec G §G.8); per-path Doppler /
// time-selectivity is deferred (Spec G §G.12).
//
//   û_AoA = (cos el·cos az, cos el·sin az, sin el)     unit arrival dir, UE frame
//   f_d   = (f_c / c) · (v · û_AoA)                     Doppler shift [Hz]
//   Δφ    = 2π · f_d · T_sym                            rad per symbol
//
// `v` is the UE velocity (m/s) expressed in the SAME frame as the stored arrival
// angles (the UE-array frame, Spec G §G.5/§G.7). The carrier `f_c` and `T_sym`
// come from the table header (CirHeader.carrierHz) and the symbol geometry
// (orchestr/timing.hpp: double(symTc())/kTcPerSec). The result is consumed by
// buildDopplerRotors(phaseInc[C·U], symbolCtr, rot) in channel/doppler.hpp.

#include <cmath>
#include <cstdint>

#include "channel/cir_table.hpp"  // PathRecord
#include "common/dims.hpp"
#include "scenario/vec3.hpp"

namespace orca {
namespace scenario {

constexpr double kSpeedOfLight = 299792458.0;  // m/s
constexpr double kTwoPi        = 6.283185307179586476925286766559;

// `Vec3` and `dot3` live in scenario/vec3.hpp (shared with the grid model).
// Here a Vec3 is a velocity (m/s) or unit direction in the UE-array frame.

// Unit arrival direction from (az, el), UE-array frame (Spec G §G.8).
inline Vec3 aoaUnit(double azRad, double elRad) {
    const double ce = std::cos(elRad);
    return Vec3{ce * std::cos(azRad), ce * std::sin(azRad), std::sin(elRad)};
}

// Doppler shift f_d (Hz) for one link: positive when the UE moves toward the
// arrival direction (closing), negative when receding.
inline double dopplerShiftHz(double aoaAz, double aoaEl, const Vec3& vel,
                             double carrierHz) {
    return (carrierHz / kSpeedOfLight) * dot3(vel, aoaUnit(aoaAz, aoaEl));
}

// Per-symbol phase increment Δφ (rad) for the fast-plane rotor.
inline double dopplerPhaseInc(double aoaAz, double aoaEl, const Vec3& vel,
                              double carrierHz, double tSymSec) {
    return kTwoPi * dopplerShiftHz(aoaAz, aoaEl, vel, carrierHz) * tSymSec;
}

// Convenience: Δφ from a link's DOMINANT path. After the Spec G §G.6 power sort,
// paths[0] is the strongest, so it drives the single Phase-1 rotor. A link with
// no surviving path (outage / noCoverage) has zero Doppler.
inline double dopplerPhaseIncFromPaths(const channel::PathRecord* paths,
                                       uint16_t numPaths, const Vec3& vel,
                                       double carrierHz, double tSymSec) {
    if (paths == nullptr || numPaths == 0) return 0.0;
    return dopplerPhaseInc(static_cast<double>(paths[0].aoaAz),
                           static_cast<double>(paths[0].aoaEl),
                           vel, carrierHz, tSymSec);
}

}  // namespace scenario
}  // namespace orca
