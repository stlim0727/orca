#pragma once
// K2 / K3 — channel-apply, CPU golden models (Spec E §E.5/§E.6; Stage 3).
//
// K2 (DL), per cell c and victim u = victim[c][sc]:
//   r_dl[u][rx][sc] = awgn + Σ_{c'<C} rot[c'][u] · Σ_{tx} H[c'][u][rx][tx][sc] · y[c'][tx][sc]
// (all-to-all interference, ADR 0002 §2; unscheduled (c,sc) writes nothing —
// r_dl is zeroed first).
//
// K3 (UL), per cell c, RU antenna rxRu:
//   r_ul[c][rxRu][sc] = awgn + Σ_{u sched on sc} rot[c][u] · Σ_{ueTx} H_ul[u][c][rxRu][ueTx][sc] · x_ul[u][ueTx][sc]
// with reciprocity H_ul[u][c][rxRu][ueTx][sc] = H_dl[c][u][ueTxToRx[ueTx]][rxRu][sc]
// (Spec E §E.6 — no H_ul table). Noise lands on every RU antenna stream.
//
// Noise keying (Spec E §E.7 / dsp/awgn.hpp): DL subsequence = (dir=0, ue=u,
// rx), UL subsequence = (dir=1, "ue"=cell, rx=rxRu); sampleIdx =
// symbolCtr·numScP + sc. H is stored half2c and converted toCf32 on load —
// exactly what the GPU kernel does.

#include <cstdint>

#include "common/complex.hpp"

namespace orca {
namespace dsp {

// All tensors are the Spec E §E.11 flat layouts:
//   H      [C][U][numRx][numTx][numScP]  half2c
//   y      [C][numTx][numScP]            cf32
//   victim [C][numScP]                   uint16 (kNoUe = unscheduled)
//   rot    [C][U]                        cf32 (this symbol's rotors)
//   r_dl   [U][numRx][numScP]            cf32 (fully zeroed first)
void channelApplyDlGolden(const half2c* H, const cf32* y,
                          const uint16_t* victim, const cf32* rot,
                          uint64_t seed, uint64_t symbolCtr, float noiseStd,
                          cf32* r_dl);

// DL noise note (§A.4-of-§E.5 semantics): AWGN exists only on *scheduled*
// victim streams — unscheduled (cell, sc) entries are skipped and stay zero
// (normative "u == NONE → return"), so no noise is drawn for them.
//
// UL contributor cardinality: under SU-MIMO (ADR 0005) each cell schedules
// exactly ONE UE per subcarrier, so the same victim-map table enumerates all
// UL contributors on sc (one per cell, all-to-all). An MU-MIMO world needs a
// contributor *list* here — deferred with MU (deferred-goals #1).
//
//   x_ul   [U][numUeTx][numScP]          cf32
//   r_ul   [C][numTx][numScP]            cf32 (fully written: noise + sums)
//   ueTxToRx: which UE rx elements transmit (default {0,1}, Spec E §E.6).
// Returns false (r_ul zeroed, nothing else done) if any ueTxToRx[t] is not a
// valid UE rx index — a bad mapping must not read outside the H row.
bool channelApplyUlGolden(const half2c* H, const cf32* x_ul,
                          const uint16_t* victim, const cf32* rot,
                          const uint8_t* ueTxToRx, uint64_t seed,
                          uint64_t symbolCtr, float noiseStd, cf32* r_ul);

}  // namespace dsp
}  // namespace orca
