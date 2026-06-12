#pragma once
// K1 — DL precode, CPU golden model (Spec E §E.4; Stage 2).
// Per DL allocation: y[c][tx][sc] = Σ_{l<rank} W[tx][l] · x_dl[c][l][sc],
// W[tx][l] = precodeBook[beamId[l]][tx] (gathered). Subcarriers outside any
// DL allocation stay zero. The CUDA K1 kernel is validated against this.

#include <cstdint>

#include "common/complex.hpp"
#include "oru/oru_transport.hpp"
#include "scenario/beam_codebook.hpp"

namespace orca {
namespace dsp {

// x_dl, y are the Spec E §E.11 flat tensors ([C][rankMax][numScP] cf32 and
// [C][numTx][numScP] cf32). Zeroes all of y, then applies every valid DL
// allocation. Allocations with out-of-range cell/rank/sc-range/beam ids are
// skipped and counted in the returned reject count — y is always fully
// written (zeros where nothing applied), so callers distinguishing "all
// rejected" from "valid all-zero" must compare the return value against
// their DL-allocation count.
uint32_t precodeGolden(const cf32* x_dl, const BeamCodebook& book,
                       const Alloc* allocs, uint32_t numAllocs, cf32* y);

}  // namespace dsp
}  // namespace orca
