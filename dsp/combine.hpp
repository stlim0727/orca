#pragma once
// K4 — UL combine, CPU golden model (Spec E §E.6; Stage 5; ADR 0006 §3).
// Per UL allocation: z[c][l][sc] = Σ_{rx<numTx} C[l][rx] · r_ul[c][rx][sc],
// C[l] = combineBook[beamId[l]] (gathered; 64 → rank, "precombining on
// behalf of the RU", symmetric with K1). Subcarriers outside any UL
// allocation stay zero.
//
// NO conjugation is applied here — Spec E §E.6's formula is a plain product,
// and the combine book is a *separate* resident codebook from the precode
// book exactly so its entries are stored already in combine orientation
// (i.e. if matched combining w^H is desired, the deployment artifact stores
// the conjugated vectors). The golden model must not second-guess that.

#include <cstdint>

#include "common/complex.hpp"
#include "oru/oru_transport.hpp"
#include "scenario/beam_codebook.hpp"

namespace orca {
namespace dsp {

// r_ul, z are the Spec E §E.11 flat tensors ([C][numTx][numScP] cf32 and
// [C][rankMax][numScP] cf32). Zeroes all of z, then applies every valid UL
// allocation; malformed ones (cell/rank/sc-range/beam out of range) are
// skipped and counted in the returned reject count (same contract as
// precodeGolden: z is always fully written).
uint32_t combineGolden(const cf32* r_ul, const BeamCodebook& book,
                       const Alloc* allocs, uint32_t numAllocs, cf32* z);

}  // namespace dsp
}  // namespace orca
