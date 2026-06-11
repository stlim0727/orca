#pragma once
// Spec B §B.4 — U-plane payload: frequency-domain IQ for numPrb PRBs starting
// at startPrb → numPrb·12 subcarriers, complex. Phase 1 uses cmp=0 (int16,
// 4 B/SC, bit-exact); each I/Q is int16 in network byte order. BFP (cmp=1)
// is deferred (Spec B §B.4 / deferred-goals #10).

#include <cstdint>

#include "common/complex.hpp"
#include "common/dims.hpp"
#include "fh/fh_header.hpp"

namespace orca {
namespace fh {

constexpr uint32_t kBytesPerSc = 4;  // int16 I + int16 Q

// Wire bytes for a numPrb-PRB section; 0 for an out-of-range numPrb (callers
// must treat 0 as malformed input, mirroring the C-plane parse contract).
constexpr uint32_t uplaneBytes(uint32_t numPrb) {
    return numPrb <= dims::numPrb ? numPrb * dims::scPerPrb * kBytesPerSc : 0;
}

// Serializes numSc complex samples to the wire (be16 I, be16 Q per SC).
// `out` must hold numSc·kBytesPerSc bytes.
inline void packUplane(const ci16* src, uint32_t numSc, uint8_t* out) {
    for (uint32_t i = 0; i < numSc; ++i) {
        storeBe16(out + size_t{i} * 4, static_cast<uint16_t>(src[i].re));
        storeBe16(out + size_t{i} * 4 + 2, static_cast<uint16_t>(src[i].im));
    }
}

// Deserializes numSc complex samples from the wire.
inline void unpackUplane(const uint8_t* in, uint32_t numSc, ci16* dst) {
    for (uint32_t i = 0; i < numSc; ++i) {
        dst[i].re = static_cast<int16_t>(loadBe16(in + size_t{i} * 4));
        dst[i].im = static_cast<int16_t>(loadBe16(in + size_t{i} * 4 + 2));
    }
}

}  // namespace fh
}  // namespace orca
