# ADR 0005 — SU-MIMO for Phase 1; MU-MIMO deferred

- **Status:** Accepted
- **Date:** 2026-06-09
- **Context tags:** spatial multiplexing, SU-MIMO, MU-MIMO, bandwidth, scheduling, phasing
- **Builds on:** [ADR 0002 §6](0002-multi-cell-interference-mobility.md) (the `H`-bandwidth
  roofline) and [ADR 0003](0003-throughput-latency-pipeline.md) (throughput clock).
- **Feeds:** [deferred-goals.md](../deferred-goals.md) (MU-MIMO is the flagship deferral).

## Context

The Phase-1 target is **2 cells, all-to-all interference, grid mobility, all `H` resident
in GPU memory, per-subcarrier FP16 `H`**. Under **MU-MIMO** that does **not** close the
throughput clock: channel-apply reads ~214.7 MB of `H` per symbol → **~6.0 TB/s vs
~3.35 TB/s HBM (1.8× over)** (full derivation in ADR 0002 §6).

MU-MIMO stacks **all** `U_c` UEs of a cell on **every** subcarrier (spatial separation),
so active `(UE, SC)` occupancies = `U · numSc`. **SU-MIMO** serves **one UE per
time-frequency resource** (UEs separated by OFDMA scheduling), so occupancies =
`C · numSc`. That single change divides the per-symbol `H` read by `U_c`.

## Decision

### 1. Phase 1 is SU-MIMO only

One UE per time-frequency resource per cell; UEs are separated by **frequency/time
scheduling**, not spatial multiplexing. **Inter-cell, all-to-all interference is
retained** — on each subcarrier, a cell's scheduled UE still takes interference from the
other cell via the cross-link `H[otherCell][thisUE]`.

### 2. MU-MIMO is deferred to a very later phase

See [deferred-goals.md → MU-MIMO](../deferred-goals.md#mu-mimo) for what it costs to bring
back. In short: it re-introduces the 16× `H`-bandwidth blow-up and therefore **requires
Spec C** (per-PRB-group or tap-domain `H`) or aggressive precision/scale cuts.

### 3. Hot-path consequences under SU-MIMO

- **`H`-read bandwidth** (all-to-all, per-SC FP16):
  ```
  H bytes/symbol = C² · numRx · numTx · numSc · b
                 = 2² · 4 · 64 · 3276 · 4 B = 13.4 MB
  BW ≈ 13.4 MB / 35.7 µs ≈ 0.38 TB/s  →  ~11% of HBM  ✅
  ```
  (`C²` = `C` victim UEs/SC × `C` contributors/UE.) Per-subcarrier FP16 `H` now fits with
  large margin, so **Spec C stays deferred** and Phase 1 keeps full frequency selectivity.
- **Precode / combine** operate per active resource on **UE-rank** layers, not the MU
  aggregate: `rank ≤ min(numTx, numRx) = 4`. `W` per SC is `64 × rank`; combine is
  `rank × 64`. Compute drops ~`U_c×` versus MU.
- **New requirement — per-symbol scheduling map.** Channel-apply must know
  `sched(cell, sc)` (which UE occupies each subcarrier per cell) to gather the correct
  `H`. This allocation comes from the vDU scheduler (DCI). It is a small per-symbol
  control input that **must be carried** — candidate: extend the Spec B C-plane
  (`msgtyp=1`) with a per-section UE/allocation id, or a dedicated allocation message.
  (MU did not need this — every UE was on every SC.) **Open → refine in Spec B.**
- **Resident `H` table is unchanged** — still `H[cell][grid-point]` for all positions.
  SU-MIMO cuts the per-symbol **read**, not storage.

### 4. Fidelity note

The emulated system-under-test is **SU-MIMO** — lower spectral efficiency / aggregate
throughput than MU-MIMO. This is an explicit modeling compromise for Phase 1, not a
free simplification.

## Consequences

### Positive
- The Phase-1 bandwidth blocker dissolves: 2-cell all-to-all, per-SC FP16 `H`, all
  resident → ~11% of HBM read BW. **Spec C, per-PRB-group, tap-domain, and INT8 all stay
  out of Phase 1.**
- Precode/combine compute also drops ~16×; the whole hot path is lighter.
- Full frequency selectivity retained (per-subcarrier `H`).

### Negative / costs
- Must carry and consume a **per-symbol scheduling/allocation map** (new control input).
- Emulates an SU-MIMO system only — different, lower-capacity SUT than MU.
- The relief is **conditional on staying SU**: re-enabling MU-MIMO or growing cells with
  per-SC `H` reopens the ADR 0002 §6 wall (→ deferred-goals).

## Revisit if…
- MU-MIMO is required → see [deferred-goals.md → MU-MIMO](../deferred-goals.md#mu-mimo);
  pull in Spec C (channel coherence granularity) as the enabling work.
- Cell count grows enough that even SU per-SC `H` approaches HBM → apply the same Spec C
  levers, or replicate across GPUs (ADR 0003 §6).
