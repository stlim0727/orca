# ADR 0002 — Multi-cell, inter-cell interference, and UE mobility

- **Status:** Accepted
- **Date:** 2026-06-09
- **Context tags:** multi-cell, interference, mobility, channel model, real-time, µ=1
- **Supersedes scope of:** the single-cell assumption implicit in
  [architecture.md](../architecture.md) and [ADR 0001](0001-hot-path-synchronization.md).
  ADR 0001's synchronization model (CUDA Graph + CPU-controlled DOCA + indirection
  cell) is **unchanged** and still applies per symbol.

## Context

The emulator must now model **multiple cells** with **inter-cell interference**
between **multiple UEs**, where the channel and Doppler evolve **per symbol** and
**per location** (cell site and UE position), and UE positions change **dynamically on
a grid**. This is a data-model change (a new *cell* dimension and a *cross-link*
channel) and, more consequentially, it reopens the ADR-0001-era assumption that
"compute is light, timing is hard" — full interference can make the hot path
**compute-bound**.

## Decisions

### 1. Single GPU box hosts all cells

One emulator instance holds **all cells** and the **full cross-channel tensor** in one
GPU's memory. Cross-cell interference is then a local memory read, with no inter-box
IQ/state exchange. The box terminates one custom-fronthaul flow set **per cell** toward
that cell's (real) vDU; all cells share the same PTP/S-plane time reference (Spec A),
so a single per-symbol deadline covers every cell.

*Rejected:* one box/vDU per cell. More horizontally scalable but requires exchanging
per-symbol IQ or channel state across boxes inside the symbol budget — far more complex
synchronization. Revisit only if cell count exceeds one GPU's memory/compute.

### 2. Cross-link channel model

The channel is no longer per-UE; it is a **cross-link** tensor between every modeled
`(cell, ue)` pair:

```
H[cell][ue][rx][tx][sc]      # cell's TRX (tx) ↔ UE rx-port (rx), per subcarrier
```

- **DL receive at UE u** (per symbol):
  `r_u = Σ_{c ∈ contributors(u)} ( H[c][u] · y_c ) · doppler[c][u] + noise`
  where `y_c` is cell `c`'s precoded 64-Tx output and the sum runs over `u`'s serving
  cell **plus its interferers**.
- **UL receive at cell c's RU** (64 antennas, per symbol):
  `r_c = Σ_{u ∈ contributors(c)} ( H[u][c] · x_u ) · doppler[u][c] + noise`,
  then combine **64 → 16** (ADR-era spatial symmetry) to deliver layers to vDU `c`.

The desired signal is the serving-cell term; every other term is interference. Noise is
added once per receive stream, not per contributor.

### 3. Interference set is configurable (serving + interferers)

Each UE has **one serving cell** and an **interferer list**. The set of contributing
links is a **runtime config**, not hardcoded:

- **Default: neighbor-limited (top-K).** Each UE sees its serving cell + the `K`
  dominant interferers (`K` configurable, e.g. 3–6). This bounds the per-symbol GEMM
  and matches real-world interference dominance.
- **Optional: all-to-all.** Every cell contributes to every UE, both directions —
  enabled for small scenarios or accuracy studies.

The active interference set defines a **sparse contribution list** consumed by the
channel-apply stage; changing `K` or going all-to-all does not change the kernels, only
the link list (and the captured graph's batch extent — see §6).

### 4. UE ↔ cell association and handover

A UE belongs to a **serving cell** (its desired DL signal / its UL grant target) and is
interfered by others. As a UE moves across the grid, its serving cell and interferer
list may change (**handover**). Association is slow-plane state, republished to the hot
path with the channel buffer; the hot path never recomputes association.

### 5. Mobility: precomputed CIR grid + slow-plane lookup

UE positions live on a **discrete location grid**. The channel is generated **offline**
by ray tracing into a **per-`(cell, grid-point)` CIR table**. At runtime:

- A UE move (grid-cell change) is a **slow-plane lookup** into the table (with optional
  spatial interpolation between adjacent grid points), which refills the **back** CIR
  buffer's affected `(cell, ue)` links and **republishes via the indirection cell**
  (ADR 0001 §4). No live ray tracing on the hot path; no per-move OptiX launch.
- **Per-symbol Doppler** is applied on the **fast plane** as a phase rotor per
  `(cell, ue)` link, advanced one step per symbol index — so the channel evolves
  per-symbol *between* slow CIR updates, parametrized by both endpoints' locations.

Live OptiX ray tracing is retained only as the **offline table generator** (and as an
optional source for arbitrary geometry), never on the per-move runtime path.

### 6. The binding constraint is memory bandwidth, not FLOPs

With interference the per-symbol channel-apply is a batched operation over the
contribution list:

```
work/symbol ≈ Σ_u |contributors(u)| · numTx · numRxPort · numSc   complex MACs   (UL symmetric)
```

It is tempting to read this as "compute-bound, use a tensor-core GEMM." A roofline
analysis shows that is wrong in the as-drafted (per-subcarrier `H`) form: **channel-apply
is memory-bandwidth bound.**

**Reference scenario.** 7 cells × 16 UE = 112 UEs, `numTx=64`, `numRxPort=4`,
`numSc=273·12=3276`, neighbor-limited `K=3` (4 contributors/UE), FP16-complex `H`
(4 B/coefficient), 8 FLOP per complex MAC, µ=1 → 28,011 symbols/s.

| Quantity | DL value | Note |
|---|---|---|
| Compute | `112·4·4·64·3276` ≈ 3.76e8 cMAC ≈ **3.0 GFLOP/symbol** | → ~84 TFLOP/s DL, ~168 TFLOP/s both dirs |
| vs H100 FP16 tensor (~990 TFLOP/s) | comfortable | this is why "GEMM" looks fine |
| **Arithmetic intensity** | **≈ 2 FLOP/byte** | `H` is read once, reused for one MAC → a batched **GEMV**, not a GEMM |
| H100 roofline ridge | ≈ 295 FLOP/byte | AI≈2 is ~150× below → **bandwidth-bound** |
| `H` traffic, per-SC | ≈ 1.5 GB/symbol → **~42 TB/s** | vs HBM3 **≈ 3.35 TB/s** → ~12× over, *neighbor-limited* |
| all-to-all, 21 cells | ~100× over HBM | infeasible on one GPU |

The dominant cost is **streaming `H` from HBM every symbol** (`H` ≈ 1.5 GB ≫ 50 MB L2,
so no cross-symbol reuse). FLOPs are not the wall; `H` footprint and bandwidth are.

**Resolution — cut `H` traffic, two options (decide per delay-spread target):**

1. **Store `H` per PRB-group, not per subcarrier** (the channel is the FFT of a few
   taps → smooth over a coherence bandwidth). A 4-PRB group cuts `numSc` ~48× →
   active `H` ≈ 31 MB, which **fits in L2 (50 MB)** and stays resident across the
   slow-plane window; HBM then sees `H` only on the few-ms CIR update, and
   channel-apply becomes L2-/compute-bound where tensor cores genuinely help. **Cost:**
   loses sub-group frequency selectivity.
2. **Apply from CIR taps on the fly** — `H[sc] = Σ_p g_p·e^{-j2πf_sc τ_p}`. The `P`
   per-link tap gains are tiny and reused across all `numSc`, raising AI well past the
   ridge → genuinely **compute-bound** with full frequency selectivity, at more math.

**Levers, in priority order:** (a) reduce `H`'s effective `numSc` footprint via option 1
or 2; (b) **neighbor-limit** the contribution list (§3) to bound `Σ|contributors|`;
(c) store only **active links**; (d) lower `H` precision (FP16/BF16/INT8). Only after
(a) is the "tensor-core batched GEMM" framing accurate.

**Variables that pin the data layout** (decide before committing channel-apply): `H`
**precision** (FP16 / BF16 / INT8), **PRB-group size** (frequency granularity), and
**tap count `P`**. These trade accuracy against bandwidth and should be set with a
target delay spread / coherence bandwidth in mind — see the open question below.

The ADR-0001 "compute is light, timing is hard" framing holds **only** for the
no/low-interference, low-bandwidth case. The corrected general statement: **timing is
still the deadline that matters, but with multi-cell interference the channel-apply is
HBM-bandwidth bound, and reducing `H`'s footprint — not adding FLOPs/tensor cores — is
the primary design lever.**

> **Open question (→ Spec C).** Channel **coherence granularity**: per-subcarrier vs
> per-PRB-group `H` vs tap-domain application. This is the single decision that
> determines whether channel-apply is feasible on one GPU and what the `dsp`/`channel`
> tensor layouts look like. To be resolved with the offline CIR-table model.

## Consequences

### Positive
- A single, local cross-channel tensor — no inter-box exchange in the symbol loop.
- The interference set is data, not code: scale from single-cell to all-to-all without
  touching kernels or the captured graph topology.
- Mobility is deterministic and cheap at runtime (table lookup), keeping the slow plane
  off the hot path exactly as ADR 0001 requires.

### Negative / costs
- Memory **bandwidth** (not FLOPs) is the binding constraint on channel-apply; the
  per-subcarrier `H` form is infeasible on one GPU and forces the per-PRB-group or
  tap-domain resolution in §6 — a real data-layout commitment, not a tuning detail.
- Memory and compute still scale with cells × interferers; all-to-all at large scale
  will not fit one GPU — that is the explicit boundary of this decision.
- Graph batch extent depends on the **maximum** contribution count; dynamic
  handover/interferer changes must stay within the captured batch bound (size for the
  configured `K`, not the instantaneous count).
- An offline CIR-table build/generation step and storage are now part of the toolchain.

## Revisit if…
- Cell count or all-to-all interference exceeds a single GPU's memory/compute → reopen
  **one-box-per-cell** (§1) with an inter-box channel-state exchange.
- Per-symbol interference batch cannot meet `T_proc` even neighbor-limited → consider
  per-PRB-group interference pruning or coarser interferer cadence.
- `H` bandwidth still dominates after the §6 resolution (e.g. tap-domain `P` too large,
  or per-PRB-group still exceeds L2) → revisit `H` precision (INT8), coarser frequency
  granularity, or fewer active links.
- Arbitrary continuous mobility (not grid-quantized) is required → revisit live ray
  tracing or finer-grid interpolation.
