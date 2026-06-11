#pragma once
// Spec B §B.5 — C-plane payload (msgtyp=1): per-resource allocation +
// beam_id, carrying the SU-MIMO scheduling map (ADR 0005) and the codebook
// indices (ADR 0006). Wire section:
//   prbStart u16 | numPrb u16 | ueId u16 | numLayers u8 | dir u8 |
//   beamId[numLayers] u16[]            (all multi-byte fields big-endian)
// Variable length: 8 + 2·numLayers bytes. Parses into the Spec E `Alloc`
// (PRB range → subcarrier range; cell comes from the eAxC, not this payload).

#include <cstdint>

#include "common/dims.hpp"
#include "fh/fh_header.hpp"
#include "oru/oru_transport.hpp"

namespace orca {
namespace fh {

struct CplaneSection {
    uint16_t prbStart;
    uint16_t numPrb;
    uint16_t ueId;
    uint8_t numLayers;  // rank, 1..rankMax
    uint8_t dir;        // 0 = DL, 1 = UL
    uint16_t beamId[dims::rankMax];
};

constexpr uint32_t kCplaneFixedBytes = 8;

constexpr uint32_t cplaneBytes(uint8_t numLayers) {
    return kCplaneFixedBytes + 2u * numLayers;
}

// Serializes one section; returns bytes written, or 0 if numLayers is out of
// range or `cap` is too small.
inline uint32_t packCplane(const CplaneSection& s, uint8_t* out, uint32_t cap) {
    if (s.numLayers == 0 || s.numLayers > dims::rankMax) return 0;
    const uint32_t need = cplaneBytes(s.numLayers);
    if (cap < need) return 0;
    storeBe16(out, s.prbStart);
    storeBe16(out + 2, s.numPrb);
    storeBe16(out + 4, s.ueId);
    out[6] = s.numLayers;
    out[7] = s.dir;
    for (uint8_t l = 0; l < s.numLayers; ++l)
        storeBe16(out + 8 + 2 * size_t{l}, s.beamId[l]);
    return need;
}

// Parses one section; returns bytes consumed, or 0 on malformed/truncated
// input (never reads past `len`).
inline uint32_t parseCplane(const uint8_t* in, uint32_t len, CplaneSection& s) {
    if (len < kCplaneFixedBytes) return 0;
    s.prbStart = loadBe16(in);
    s.numPrb = loadBe16(in + 2);
    s.ueId = loadBe16(in + 4);
    s.numLayers = in[6];
    s.dir = in[7];
    if (s.numLayers == 0 || s.numLayers > dims::rankMax) return 0;
    const uint32_t need = cplaneBytes(s.numLayers);
    if (len < need) return 0;
    for (uint8_t l = 0; l < s.numLayers; ++l)
        s.beamId[l] = loadBe16(in + 8 + 2 * size_t{l});
    for (uint8_t l = s.numLayers; l < dims::rankMax; ++l) s.beamId[l] = 0;
    return need;
}

// CplaneSection → Spec E Alloc (Spec F §F.3): PRB range → subcarrier range;
// the cell comes from the packet's eAxC. Returns false (out untouched) when
// the PRB range exceeds the carrier — a malformed section must not become an
// out-of-bounds allocation.
inline bool toAlloc(const CplaneSection& s, uint16_t cell, Alloc& out) {
    if (s.numPrb == 0 || s.prbStart >= dims::numPrb ||
        s.numPrb > dims::numPrb - s.prbStart)
        return false;
    if (s.numLayers == 0 || s.numLayers > dims::rankMax) return false;
    Alloc a{};
    a.cell = cell;
    a.ueId = s.ueId;
    a.scStart = static_cast<uint16_t>(s.prbStart * dims::scPerPrb);
    a.scLen = static_cast<uint16_t>(s.numPrb * dims::scPerPrb);
    a.dir = s.dir;
    a.rank = s.numLayers;
    for (uint8_t l = 0; l < dims::rankMax; ++l) a.beamId[l] = s.beamId[l];
    out = a;
    return true;
}

}  // namespace fh
}  // namespace orca
