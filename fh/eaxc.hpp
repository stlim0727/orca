#pragma once
// Spec B §B.3 — eAxC encoding. 16-bit {DU_Port, BandSector, CC_ID, RU/UE_Port},
// bit widths configurable at deployment (default 4/2/2/8), packed MSB→LSB in
// that order. The cell is identified by {BandSector, CC_ID} (multi-cell
// addressing, §B.3).

#include <cassert>
#include <cstdint>

namespace orca {
namespace fh {

struct EaxcConfig {
    uint8_t duPortBits = 4;
    uint8_t bandSectorBits = 2;
    uint8_t ccIdBits = 2;
    uint8_t ruPortBits = 8;

    // Widths must each fit the 16-bit field and sum to exactly 16.
    constexpr bool valid() const {
        return duPortBits <= 16 && bandSectorBits <= 16 && ccIdBits <= 16 &&
               ruPortBits <= 16 &&
               duPortBits + bandSectorBits + ccIdBits + ruPortBits == 16;
    }
};

struct Eaxc {
    uint16_t duPort;
    uint16_t bandSector;
    uint16_t ccId;
    uint16_t ruPort;  // RU antenna / UE port within the cell
};

// bits ≤ 16 by EaxcConfig::valid(); the 16 case is explicit so the shift
// never reaches UB territory.
constexpr uint16_t mask(uint8_t bits) {
    return bits >= 16 ? uint16_t{0xFFFF}
                      : static_cast<uint16_t>((1u << bits) - 1u);
}

// Packs MSB→LSB: [duPort | bandSector | ccId | ruPort]. Fields are masked to
// their configured widths. Precondition: c.valid() (asserted — an invalid
// config would produce overlapping fields).
constexpr uint16_t encodeEaxc(const Eaxc& e, const EaxcConfig& c = {}) {
    assert(c.valid());
    return static_cast<uint16_t>(
        ((e.duPort & mask(c.duPortBits))
         << (c.bandSectorBits + c.ccIdBits + c.ruPortBits)) |
        ((e.bandSector & mask(c.bandSectorBits)) << (c.ccIdBits + c.ruPortBits)) |
        ((e.ccId & mask(c.ccIdBits)) << c.ruPortBits) |
        (e.ruPort & mask(c.ruPortBits)));
}

// Precondition: c.valid() (asserted).
constexpr Eaxc decodeEaxc(uint16_t v, const EaxcConfig& c = {}) {
    assert(c.valid());
    return Eaxc{
        static_cast<uint16_t>(
            (v >> (c.bandSectorBits + c.ccIdBits + c.ruPortBits)) &
            mask(c.duPortBits)),
        static_cast<uint16_t>((v >> (c.ccIdBits + c.ruPortBits)) &
                              mask(c.bandSectorBits)),
        static_cast<uint16_t>((v >> c.ruPortBits) & mask(c.ccIdBits)),
        static_cast<uint16_t>(v & mask(c.ruPortBits)),
    };
}

// Cell id from {BandSector, CC_ID} (§B.3 multi-cell addressing).
constexpr uint16_t cellOf(uint16_t eaxc, const EaxcConfig& c = {}) {
    const Eaxc e = decodeEaxc(eaxc, c);
    return static_cast<uint16_t>((e.bandSector << c.ccIdBits) | e.ccId);
}

}  // namespace fh
}  // namespace orca
