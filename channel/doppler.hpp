#pragma once
// Per-link Doppler rotor (ADR 0002 §5 / Spec E §E.5): the fast plane applies
// e^{j·Δφ[c][u]·symbolCtr} per (cell, ue) link, one step per symbol — the
// channel evolves per-symbol between slow-plane H updates without touching
// H itself (Spec E §E.10). Δφ = 2π·f_d·T_sym comes from the scenario layer
// (Spec G §G.8 Doppler handoff).

#include <cmath>
#include <cstdint>

#include "common/complex.hpp"
#include "common/dims.hpp"

namespace orca {

// One rotor value: e^{j·phaseInc·symbolCtr}. Double-precision phase
// accumulation so long symbol counters don't lose the angle (the golden
// contract; the GPU fast plane accumulates the same product).
inline cf32 dopplerRotor(double phaseIncRad, uint64_t symbolCtr) {
    const double phi = phaseIncRad * static_cast<double>(symbolCtr);
    return cf32{static_cast<float>(std::cos(phi)),
                static_cast<float>(std::sin(phi))};
}

// Fills rot[c][u] (flat C·U, u innermost) for one symbol from the per-link
// phase increments (same layout).
inline void buildDopplerRotors(const double* phaseIncRad, uint64_t symbolCtr,
                               cf32* rot) {
    for (uint32_t i = 0; i < dims::C * dims::U; ++i)
        rot[i] = dopplerRotor(phaseIncRad[i], symbolCtr);
}

constexpr size_t dopplerIdx(uint32_t cell, uint32_t ue) {
    return size_t{cell} * dims::U + ue;
}

}  // namespace orca
