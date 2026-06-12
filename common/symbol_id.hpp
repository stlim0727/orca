#pragma once
// Symbol identity: the reassembly/scheduling key (cell, dir, sfn, slot, sym)
// (Spec A §A.5 / Spec B §B.2) and the continuous symbol counter derived from
// the S-plane time reference (Spec A §A.1).

#include <cstdint>

#include "common/dims.hpp"

namespace orca {

enum class Dir : uint8_t { DL = 0, UL = 1 };  // Spec B §B.2 enumeration

struct SymbolId {
    uint16_t cell;
    Dir dir;
    uint16_t sfn;   // 0..1023
    uint8_t slot;   // 0..slotsPerFrame(µ)-1
    uint8_t sym;    // 0..13
};

constexpr bool operator==(SymbolId a, SymbolId b) {
    return a.cell == b.cell && a.dir == b.dir && a.sfn == b.sfn &&
           a.slot == b.slot && a.sym == b.sym;
}

constexpr bool operator!=(SymbolId a, SymbolId b) { return !(a == b); }

// Continuous slot index within one SFN period (Spec A §A.1). The SFN is
// normalized mod 1024 (Spec A §A.2 / Spec B u16 header field).
constexpr uint64_t slotIndex(uint16_t sfn, uint8_t slot, uint32_t mu = dims::mu) {
    return static_cast<uint64_t>(sfn % dims::sfnPeriod) * dims::slotsPerFrame(mu) +
           slot;
}

// Continuous symbol counter within one SFN period; wraps with the SFN (mod 1024
// frames). Callers tracking longer spans add their own epoch.
constexpr uint64_t symbolCounter(uint16_t sfn, uint8_t slot, uint8_t sym,
                                 uint32_t mu = dims::mu) {
    return slotIndex(sfn, slot, mu) * dims::symsPerSlot + sym;
}

constexpr uint64_t symbolCounter(SymbolId id, uint32_t mu = dims::mu) {
    return symbolCounter(id.sfn, id.slot, id.sym, mu);
}

constexpr uint64_t symbolsPerSfnPeriod(uint32_t mu = dims::mu) {
    return static_cast<uint64_t>(dims::sfnPeriod) * dims::slotsPerFrame(mu) *
           dims::symsPerSlot;
}

}  // namespace orca
