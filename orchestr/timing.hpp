#pragma once
// Spec A §A.1–§A.3 — S-plane time reference, per-symbol air time, deadlines.
//
// Times are exact in Tc units (TS 38.211: Tc = 1/(480 kHz · 4096), i.e.
// 1,966,080,000 Tc per second); nanoseconds derive via ns = Tc·3125/6144
// (the reduced ratio of 1e9 / 1966080000), exact at slot boundaries.
// Scope: µ=1 normal CP (the project target; µ≥2 is out of scope, ADR 0001).
// At µ=1 a slot is 0.5 ms, so the long CP lands on symbol 0 of every slot.

#include <cstdint>

#include "common/dims.hpp"
#include "common/symbol_id.hpp"

namespace orca {

constexpr uint64_t kTcPerSec = 1966080000ull;  // 480e3 · 4096
constexpr uint32_t kKappa = 64;

// Exact Tc→ns conversion (truncating): ns = tc·3125/6144.
// Valid while tc·3125 fits uint64 (tc < ~5.9e15 ≈ 34 days) — ample for a run.
constexpr uint64_t tcToNs(uint64_t tc) { return tc * 3125ull / 6144ull; }

// --- µ=1 symbol geometry (38.211 normal CP) --------------------------------

constexpr uint64_t usefulTc(uint32_t mu = dims::mu) {
    return (2048ull * kKappa) >> mu;  // 65536 @ µ=1
}
constexpr uint64_t cpNormalTc(uint32_t mu = dims::mu) {
    return (144ull * kKappa) >> mu;   // 4608 @ µ=1
}
constexpr uint64_t cpLongExtraTc() { return 16ull * kKappa; }  // 1024 (not scaled)

// Normal-CP symbol period (CP + useful): 70144 Tc ≈ 35.68 µs @ µ=1.
constexpr uint64_t symTc(uint32_t mu = dims::mu) {
    return cpNormalTc(mu) + usefulTc(mu);
}

// Slot duration: 14 symbols, symbol 0 carries the long CP (µ=1: every slot
// starts a 0.5 ms half-subframe). 983040 Tc = exactly 500000 ns @ µ=1.
constexpr uint64_t slotDurTc(uint32_t mu = dims::mu) {
    return dims::symsPerSlot * symTc(mu) + cpLongExtraTc();
}

// CP-adjusted cumulative offset of symbol `sym` within the slot (§A.1
// symStart). Symbol 0 spans cpLong+useful = 71168 Tc; symbols 1..13 span
// 70144 Tc each.
constexpr uint64_t symStartTc(uint32_t sym, uint32_t mu = dims::mu) {
    return sym == 0
               ? 0
               : (cpLongExtraTc() + symTc(mu)) + (uint64_t{sym} - 1) * symTc(mu);
}

// --- T_air (§A.1) -----------------------------------------------------------
// T0 = air time of (SFN 0, slot 0, sym 0), bootstrapped from the S-plane.
// `slotIdx` is the continuous slot counter (callers add their own SFN epoch
// beyond the 1024-frame wrap; see common/symbol_id.hpp).

constexpr uint64_t tAirTc(uint64_t t0Tc, uint64_t slotIdx, uint32_t sym,
                          uint32_t mu = dims::mu) {
    return t0Tc + slotIdx * slotDurTc(mu) + symStartTc(sym, mu);
}

// SymbolId carries only the wrapped SFN (mod 1024); a long-running
// orchestrator must supply the SFN epoch (count of completed 1024-frame
// periods) to obtain a non-aliasing absolute air time.
constexpr uint64_t tAirTc(uint64_t t0Tc, uint64_t sfnEpoch, SymbolId id,
                          uint32_t mu = dims::mu) {
    return tAirTc(t0Tc,
                  (sfnEpoch * dims::sfnPeriod) * dims::slotsPerFrame(mu) +
                      slotIndex(id.sfn, id.slot, mu),
                  id.sym, mu);
}

// --- Deadlines (§A.3) -------------------------------------------------------
// Absolute times are signed int64 ns so a deadline earlier than the clock
// origin stays representable (no unsigned underflow); offsets are durations.

struct DeadlineConfig {
    int64_t tEgressNs;    // GPU → peer serialization + NIC
    int64_t tProcNs;      // compute span (latency contribution, ADR 0003)
    int64_t tMarginNs;    // jitter guard
    int64_t tUlOffsetNs;  // vDU UL receive-window offset from T_air
};

// DL reassembly deadline: D_r(s) = T_air − T_egress − T_proc − T_margin.
constexpr int64_t deadlineDlNs(int64_t tAirNs, const DeadlineConfig& d) {
    return tAirNs - d.tEgressNs - d.tProcNs - d.tMarginNs;
}

// UL delivery deadline: D_ul(s) = T_air + T_ul_offset − T_proc − T_egress − T_margin.
constexpr int64_t deadlineUlNs(int64_t tAirNs, const DeadlineConfig& d) {
    return tAirNs + d.tUlOffsetNs - d.tProcNs - d.tEgressNs - d.tMarginNs;
}

}  // namespace orca
