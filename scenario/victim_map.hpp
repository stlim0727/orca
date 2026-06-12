#pragma once
// Per-symbol scheduling lookup (Spec E §E.2): d_victim[cell][sc] → the UE
// scheduled on that subcarrier by that cell (SU-MIMO, ADR 0005 — one UE per
// time-frequency resource per cell), derived from the Alloc list once per
// symbol. K2 reads it as the DL victim map; K3 reads the same table as the
// UL contributor map (the UE transmitting on sc toward cell c).

#include <cstdint>
#include <cstring>

#include "common/dims.hpp"
#include "oru/oru_transport.hpp"

namespace orca {

constexpr uint16_t kNoUe = 0xFFFF;

// victim must hold C·numScP entries (row stride numScP, sc innermost —
// matching the device table). Fills every entry (kNoUe where unscheduled),
// then marks each valid allocation of the requested direction. Later
// allocations overwrite earlier ones on overlap (vDU scheduling is expected
// to be disjoint per cell; the golden model needs *a* deterministic rule —
// overlap *detection*/policing is the scheduler-facing caller's job, not
// this helper's). Returns the number of allocations applied.
inline uint32_t buildVictimMap(const Alloc* allocs, uint32_t numAllocs,
                               uint8_t dir, uint16_t* victim) {
    for (size_t i = 0; i < size_t{dims::C} * dims::numScP; ++i)
        victim[i] = kNoUe;

    uint32_t applied = 0;
    for (uint32_t i = 0; i < numAllocs; ++i) {
        const Alloc& a = allocs[i];
        if (a.dir != dir) continue;
        const bool ok = a.cell < dims::C && a.scStart < dims::numSc &&
                        a.scLen >= 1 && a.scLen <= dims::numSc - a.scStart;
        if (!ok) continue;
        uint16_t* row = victim + size_t{a.cell} * dims::numScP;
        for (uint32_t sc = a.scStart; sc < uint32_t{a.scStart} + a.scLen; ++sc)
            row[sc] = a.ueId;
        ++applied;
    }
    return applied;
}

}  // namespace orca
