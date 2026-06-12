#pragma once
// Resident beam codebook (ADR 0006 / Spec E §E.2): precoding/combining
// vectors indexed by beam_id, loaded once at startup, gathered on the hot
// path. Until a deployment artifact exists, makeDftCodebook() generates the
// deterministic grid-of-beams (oversampled DFT) codebook — also the golden
// reference the GPU gather is validated against.

#include <cmath>
#include <cstdint>
#include <vector>

#include "common/complex.hpp"
#include "common/dims.hpp"

namespace orca {

class BeamCodebook {
  public:
    BeamCodebook() = default;
    BeamCodebook(std::vector<cf32> data, uint32_t numBeams)
        : data_(std::move(data)), numBeams_(numBeams) {}

    // Beam vector over numTx antennas, or nullptr for an out-of-range id.
    const cf32* beam(uint16_t beamId) const {
        return beamId < numBeams_ ? data_.data() + size_t{beamId} * dims::numTx
                                  : nullptr;
    }

    uint32_t numBeams() const { return numBeams_; }

  private:
    std::vector<cf32> data_;  // [numBeams][numTx]
    uint32_t numBeams_ = 0;
};

// Deterministic grid-of-beams: beam b, antenna n →
//   w[n] = (1/√numTx) · e^(−j·2π·n·b/numBeams)
// (oversampled DFT over the array; b = 0 is the broadside uniform beam).
// Each beam has unit L2 norm. Double-precision phase, cast to cf32.
inline BeamCodebook makeDftCodebook(uint32_t numBeams = 1024) {
    std::vector<cf32> data(size_t{numBeams} * dims::numTx);
    const double scale = 1.0 / std::sqrt(static_cast<double>(dims::numTx));
    for (uint32_t b = 0; b < numBeams; ++b) {
        for (uint32_t n = 0; n < dims::numTx; ++n) {
            const double phase = -2.0 * 3.14159265358979323846 *
                                 static_cast<double>(n) * b / numBeams;
            data[size_t{b} * dims::numTx + n] =
                cf32{static_cast<float>(scale * std::cos(phase)),
                     static_cast<float>(scale * std::sin(phase))};
        }
    }
    return BeamCodebook(std::move(data), numBeams);
}

}  // namespace orca
