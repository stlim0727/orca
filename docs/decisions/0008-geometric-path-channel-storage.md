# ADR 0008 — Geometric-path channel storage (rays, not antenna-CIR or per-SC `H`)

- **Status:** Accepted
- **Date:** 2026-06-11
- **Context tags:** channel, CIR, ray tracing, storage, mobility, slow-plane, golden-model, µ=1
- **Refines:** [ADR 0002 §5](0002-multi-cell-interference-mobility.md) (precomputed
  per-`(cell, grid-point)` CIR grid + slow-plane lookup), which decided *that* the channel
  is precomputed offline but left open *at what level* it is persisted.
- **Feeds:** [Spec G](../specs/cir-table-toolchain.md) (the toolchain/format that realizes
  this decision). **Distinct from** the deferred Spec C
  ([deferred-goals #2](../deferred-goals.md#spec-c)).

## Context

[ADR 0002 §5](0002-multi-cell-interference-mobility.md) established the offline
per-`(cell, grid-point)` CIR table and the slow-plane lookup→`H`→indirection-cell publish
flow. It did **not** pin the **storage domain** of that table — and that single choice
determines the on-disk format, the table footprint, what changes force an offline re-trace,
where the table lives (host vs GPU), and how much work the slow plane does per UE move. Spec G
cannot be written without it, so it is promoted here to a relitigation-protected decision.

A link's channel — cell `c`'s 64-TRX array ↔ a UE's 4 rx antennas at a grid point — can be
persisted at three levels:

| Storage domain | Per-`(cell,gp)` size¹ | Array-baked? | Slow-plane work to get `H` |
|---|---|---|---|
| **Geometric paths** (τ, AoD, AoA, complex gain) | `P·40 B` ≈ **640 B** | **no** | array steering + delay DFT over `P` paths |
| Antenna-domain CIR (`[rx][tx][tap]` complex) | `4·64·P·8 B` ≈ **512 KB** | yes | FFT taps→`H` only |
| Per-subcarrier `H` (`[rx][tx][sc]`) | `4·64·3276·4 B` ≈ **3.4 MB** | yes | none (already `H`) |

¹ `P` = max paths/link (default 16). The per-SC `H` form *is* the GPU-resident `H_dl`
(Spec E §E.2) — storing the whole grid that way would be ~5000× the geometric size.

## Decision

### 1. Persist the geometric multipath (ray) list per link

The offline OptiX tracer emits, per `(cell, grid-point)`, the set of significant propagation
paths: complex gain `g_p`, excess delay `τ_p`, departure angles `AoD_p` (cell-array frame),
arrival angles `AoA_p` (UE-array frame), and flags. **That ray list is what the table
stores** — not antenna-domain CIR, not per-subcarrier `H`.

### 2. Expand rays → `H` at runtime on the slow plane

On a UE move, the slow plane reads the affected links' ray lists and expands them into the
resident `H_dl` back buffer via **array steering + a direct delay DFT**:
```
H[c][u][rx][tx][sc] = Σ_{p<P} g_p · a_tx(AoD_p)[tx] · a_rx(AoA_p)[rx] · exp(-j 2π f_sc τ_p)
```
then publishes via the [ADR 0001 §4](0001-hot-path-synchronization.md) indirection cell. The
"CIR→`H` FFT" of ADR 0002 §5 is realized as this direct DFT over `P≤16` paths (exact for
arbitrary non-sample-aligned delays). **The hot path never reads the table** — only `H_dl`.

### 3. One ray set serves both directions

The traced link is reciprocal; the same rays produce DL (`cell→UE`) and UL (`UE→cell`), so
**only one ray set per `(cell, gp)` is stored**. UL reuses the DL expansion per
[Spec E §E.6](../specs/gpu-kernel-design.md) (UE tx ⊆ rx) — no separate UL table.

### 4. The table is host-resident; only expanded `H` reaches the GPU

The compact geometric table (host RAM, ~324 MB Phase 1, Spec G §G.11) stays on the **host**;
the GPU holds only the **active** expanded `H_dl` (215 MB, Spec E §E.2). Table size scales
with `cells · gridpoints · P`, never with the GPU budget.

## Consequences

### Positive
- **Compact & host-resident** — the whole scenario grid fits in host RAM; the GPU footprint
  is unchanged from Spec E (active `H_dl` only). Per-SC storage would be infeasible (~5000×).
- **Array-independent physics** — changing the 64-element cell array geometry, the 4-element
  UE array, or `numSc` is a *runtime expansion* change, **not** an offline re-trace. The rays
  are captured once per scenario.
- **Carries the angles** the per-symbol Doppler rotor needs (`f_d` from `AoA·v`, Spec E §E.5)
  and that any future SRS/beam-reciprocity work (deferred, ADR 0006) would consume.
- **Golden-model anchor** — the ray list + the §2 expansion math is a precise, immutable
  contract both the GPU `channel/` path and the CPU golden model implement → bit-reproducible
  `H` (Spec G §G.10). Partially closes the bit-exactness validation gap.

### Negative / costs
- The slow plane must run the **ray→`H` expansion** (array steering + DFT) rather than a bare
  table copy. Bounded (~sub-ms for a full 64-link refresh, Spec G §G.8) and off the hot path,
  but it is real slow-plane compute and a kernel to write.
- **Spatial interpolation is harder** in the path domain (path association / birth-death /
  angle-wrap across adjacent grid points) than it would be on a stored `H`. Phase 1 uses
  **nearest grid point** (interpolation deferred, Spec G §G.9 /
  [deferred-goals #9](../deferred-goals.md#continuous-mobility)).
- Tap-count `P` and prune threshold become **modeling parameters** (Spec G §G.6/§G.12); too
  small under-models delay spread, too large bloats the table and the DFT.

### Doc impact
- **New:** [Spec G](../specs/cir-table-toolchain.md) realizes this ADR (storage decision in
  §G.2 now points here).
- **ADR 0002 §5** → forward-ref to Spec G / this ADR for the storage level.
- **architecture.md** (slow/fast split, `channel/` module), **README**, **CLAUDE.md** (doc
  list + locked decisions) → reference ADR 0008 / Spec G.

## Rejected alternatives

- **Antenna-domain CIR** (`[rx][tx][tap]`): no runtime steering needed, but ~800× larger and
  **bakes the array geometry** into the table — any array change forces a re-trace. Rejected.
- **Per-subcarrier `H`** (the GPU-resident form, persisted for every grid point): zero
  slow-plane expansion, but ~5000× larger (per-SC, array- *and* `numSc`-baked) — infeasible
  as a stored grid. Rejected.

## Revisit if…
- The slow-plane expansion cost becomes material (very fine grids + very frequent moves) →
  consider caching expanded `H` for hot grid points, or an antenna-domain hybrid for a fixed
  array.
- **Spatial interpolation** becomes required (continuous mobility,
  [deferred-goals #9](../deferred-goals.md#continuous-mobility)) → reopen path-association vs
  `H`-domain interpolation; may favor a different stored domain.
- A **non-reciprocal** link model is needed (separate UL geometry) → §3 no longer holds; store
  per-direction ray sets.
