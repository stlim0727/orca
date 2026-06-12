#pragma once
// K0 / K5 converts behind one signature (AGENT.md 1f; Spec E §E.7).
// Host config: CPU loops (this TU). Target config: CUDA kernels with the
// same signatures land behind EMU_WITH_CUDA — callers never change.

#include <cstddef>

#include "common/complex.hpp"

namespace orca {
namespace dsp {

// K0 — ingress convert: ci16 → cf32, flat n elements.
void convertK0(const ci16* src, cf32* dst, size_t n);

// K5 — egress pack: cf32 → ci16, saturating round-to-nearest-even, flat.
void convertK5(const cf32* src, ci16* dst, size_t n);

}  // namespace dsp
}  // namespace orca
