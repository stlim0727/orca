# Spec E — GPU kernel & memory design (Phase 1 hot path)

**Status:** Settled design (source of truth). No implementation yet. Drives the `dsp/` and `channel/`
GPU code. Realizes the CUDA-graph hot path of [ADR 0001](../decisions/0001-hot-path-synchronization.md)
under the Phase-1 scope: **SU-MIMO, 2 cells, all-to-all, per-subcarrier FP16 `H` resident**
([ADR 0005](../decisions/0005-su-mimo-phase1.md)), **beam-indexed precoding**
([ADR 0006](../decisions/0006-beam-indexed-precoding.md)), µ=1.

Cross-refs: [ADR 0002 §6](../decisions/0002-multi-cell-interference-mobility.md) (bandwidth
roofline — the design driver), [Spec A](timing-and-deadlines.md) (budget),
[Spec D](vue-interface-contract.md) (the `r_dl`/`x_ul` buffers are shared with vUE).

---

## E.1 Conventions & numeric formats

**Reference dims (Phase 1):**

| Symbol | Meaning | Value |
|---|---|---|
| `C` | cells | 2 |
| `U_c`, `U` | UE/cell, total UE | 16, 32 |
| `numTx` | cell TRX (Tx) | 64 |
| `numRx` | UE rx antennas | 4 |
| `numUeTx` | UE tx antennas (UL) | 2 |
| `numSc` | subcarriers (273 PRB·12) | 3276 |
| `rank` | layers per SU resource | ≤ 4 |
| contributors/UE | all-to-all | `C` = 2 |

**Numeric formats — chosen against the binding resource (HBM bandwidth, ADR 0002 §6):**

| Tensor | Storage | Rationale |
|---|---|---|
| `H` (channel) | **`half2` (FP16 complex, 4 B)** | dominates HBM traffic → store narrow; converted to `cf32` in-register on load |
| `y, x, r, z` (IQ) | **`cf32` (8 B)** | compute-domain; small footprint; no repack for the vUE GPU receiver |
| codebooks, doppler | `cf32` | tiny, resident |
| fronthaul wire | `ci16` (Spec B) | converted `ci16→cf32` at ingress, `cf32→ci16` at egress |

**Golden rule for every multi-dim tensor: `sc` is the innermost (unit-stride) axis.**
The bandwidth-bound kernel (channel-apply) tiles threads across `sc`, so unit-stride `sc`
makes the dominant `H` and `y` reads **coalesced**. Every layout below obeys this.

`cf32 = {float re, im}` (8 B). `half2` holds `{__half re, im}` (4 B).

---

## E.2 GPU memory layout & allocation

### Resident buffers (slow-plane owned; double-buffered via the ADR 0001 indirection cell)

| Buffer | Shape (row-major, `sc` innermost) | Elem | Size (Phase 1) |
|---|---|---|---|
| `H_dl` | `[C][U][numRx][numTx][numSc]` | half2 | 2·32·4·64·3276·4 B ≈ **215 MB** |
| `H_ul` | **not stored** — reciprocity from `H_dl` (UE tx=2 ⊆ rx=4; E.6) | — | **0** |
| `precodeBook` | `[numBeams][numTx]` | cf32 | 1024·64·8 B = 512 KB |
| `combineBook` | `[numBeams][numTx]` | cf32 | 512 KB |
| `doppler` | `[C][U]` rotor (phase incr + accum) | 2×float | tiny |

- **Double buffer:** `H_dl`/`H_ul` each have an active + back copy (≈ 2×). The slow plane
  fills `back`, then publishes by an atomic pointer write into the device indirection cell
  `H_active` (ADR 0001 §4). Kernels read `const half2* H = *H_active;` once at launch-time
  of each graph replay.
- `H` is indexed by `[cell][ue]` (UE → its grid position channel). SU-MIMO reads only the
  *scheduled* `(ue, sc)` subset per symbol (~13.4 MB), but the **table holds all UEs**.

### Per-symbol working buffers (ring of `N = 4` slots, E.3 dataflow)

| Buffer | Shape | Elem | Dir | Size/slot |
|---|---|---|---|---|
| `x_dl` | `[C][rank][numSc]` | cf32 | in (from vDU) | 210 KB |
| `y` | `[C][numTx][numSc]` | cf32 | precode→channel | 3.35 MB |
| `r_dl` | `[U][numRx][numSc]` | cf32 | →**vUE (Spec D)** | 3.35 MB |
| `x_ul` | `[U][numUeTx][numSc]` | cf32 | ←**vUE (Spec D)** | 1.68 MB |
| `r_ul` | `[C][64][numSc]` | cf32 | channel→combine | 3.35 MB |
| `z` | `[C][rank][numSc]` | cf32 | →vDU | 210 KB |

- `r_dl` and `x_ul` slots are the **CUDA-IPC-shared** buffers from [Spec D](vue-interface-contract.md);
  the others are ORCA-private.
- **Total** ≈ 12 MB/slot × 4 ≈ **48 MB**; with resident `H_dl` only (~215 MB × 2 double
  buffer ≈ 430 MB; **no `H_ul`** — reciprocity) the whole hot path is **< 0.5 GB** of an
  80 GB H100.

### Per-symbol control (small, uploaded from the C-plane each symbol)

```
struct Alloc {            // one scheduled SU resource (Spec B §B.5 section)
  uint16 cell, ueId;
  uint16 scStart, scLen; // contiguous subcarrier range
  uint8  dir, rank;
  uint16 beamId[4];      // per layer
};
Alloc  d_allocs[MAX_ALLOCS];   // ~32 entries
uint16 d_victim[C][numSc];     // victimMap: sc → scheduled ueId per cell (derived from d_allocs)
```

`d_victim` is built once per symbol (tiny kernel or host) so channel-apply can look up the
scheduled UE per `sc` in O(1) without scanning allocations.

---

## E.3 Dataflow / CUDA-graph nodes

The captured graph (ADR 0001) per symbol, DL then UL (TDD → one direction live per symbol,
but both node-sets are in the graph; the inactive branch is a no-op via empty alloc list):

```
 ingress(DOCA) ─► [K0 convert/scatter] ─► x_dl
                                    │
 d_allocs,d_victim ────────────────┤
                                    ▼
                        [K1 precode] ─► y ─► [K2 channel-apply DL] ─► r_dl ─► (Spec D → vUE)
                                                        ▲
                                              H_active, doppler
 (UL)  x_ul (Spec D ← vUE) ─► [K3 channel-apply UL] ─► r_ul ─► [K4 combine] ─► z ─► [K5 pack] ─► egress(DOCA→vDU)
```

Node dependencies are graph edges (no flags). `K1→K2`, `K3→K4→K5`. `K0` and the vUE handoff
follow the Spec D event protocol. Launch configs per kernel below; all are plain
`cudaGraphAddKernelNode` with fixed params (buffers are stable; `H` via indirection cell).

---

## E.4 K1 — Precode kernel

**Computes** (per cell `c`, per allocation): `y[c][tx][sc] = Σ_{l<rank} W[tx][l] · x_dl[c][l][sc]`
where `W[tx][l] = precodeBook[beamId[l]][tx]` (gathered).

| | |
|---|---|
| **Input** | `x_dl[C][rank][numSc]` cf32; `precodeBook`; `d_allocs` |
| **Output** | `y[C][numTx][numSc]` cf32 |
| **Grid** | one block per `(allocation, sc-tile)`; `gridDim = (numAllocs, ceil(scLen/TILE))` |
| **Block** | `TILE = 256` threads (one per `sc` in the tile) |
| **Shared** | `W[numTx][rank]` cf32 = 64·4·8 = 2 KB — gathered once per block |

**Thread/warp map:**
1. Block reads its `Alloc` → `cell, scStart, rank, beamId[]`.
2. Threads cooperatively gather `W[tx][l] = precodeBook[beamId[l]][tx]` into shared (64·rank
   loads), `__syncthreads()`.
3. Thread `t` owns `sc = scStart + tileBase + t`. Loop `tx = 0..63`:
   `acc = Σ_l W_smem[tx][l] · x_dl[c][l][sc]`; write `y[c][tx][sc] = acc`.

**Coalescing:** `x_dl[c][l][sc]` and `y[c][tx][sc]` are `sc`-innermost → a warp (32 consecutive
`sc`) reads/writes 32 contiguous cf32 = coalesced. `W` is broadcast from shared. Compute is
tiny (`numTx·rank` MACs/sc); this kernel is latency/occupancy-trivial.

---

## E.5 K2 — Channel-apply DL (the bandwidth-critical kernel)

**Computes** the per-symbol cross-link sum (ADR 0002 §2), SU victims:
```
for scheduled victim u = d_victim[c][sc], for each rx:
  r_dl[u][rx][sc] = noise
                  + Σ_{c'∈cells} doppler[c'][u] · Σ_{tx} H_active[c'][u][rx][tx][sc] · y[c'][tx][sc]
```
The `Σ_{c'}` is the all-to-all interference sum (serving `c'=c` + interferer). This is the
~13.4 MB/symbol `H` read = ~0.38 TB/s (ADR 0005); **getting it coalesced is the whole game.**

| | |
|---|---|
| **Input** | `H_active[C][U][numRx][numTx][numSc]` half2; `y[C][numTx][numSc]` cf32; `doppler`; `d_victim` |
| **Output** | `r_dl[U][numRx][numSc]` cf32 |
| **Grid** | `gridDim = (C, numRx, ceil(numSc/TILE))` → 2·4·⌈3276/256⌉ = 2·4·13 = **104 blocks** |
| **Block** | `TILE = 256` threads, one per `sc` |

**Thread/warp map (thread `t` = one `sc`, fixed `(c, rx)` from `blockIdx`):**
```
sc = blockIdx.z*TILE + t;            if (sc >= numSc) return;
u  = d_victim[c][sc];                if (u == NONE) { r_dl skipped; return; }
cf32 acc = awgn(seed, u, rx, sc);    // E.7 deterministic noise
#pragma unroll 1
for (c2 = 0; c2 < C; ++c2) {
   cf32 rot = doppler[c2][u];                  // 1 load, reused over tx
   cf32 partial = 0;
   for (tx = 0; tx < numTx; ++tx)              // 64 iters
       partial += toCf32(H[c2][u][rx][tx][sc]) * y[c2][tx][sc];
   acc += rot * partial;
}
r_dl[u][rx][sc] = acc;
```

**Why this is coalesced (the key point):** in the `tx` loop, for fixed `(c2,u,rx,tx)` the 32
threads of a warp hold 32 consecutive `sc` → `H[...][tx][sc]` and `y[...][tx][sc]` are
contiguous half2/cf32 lines → fully coalesced 128 B/64 B transactions. The strided axis
(`tx`, stride `numSc`) is the **loop**, not the thread axis. Victim `u` is constant within an
allocation's PRB range, so a warp mostly shares `u` (minor divergence only at PRB-range
boundaries).

**Reuse:** `y[c2][tx][sc]` is loaded once per `(c2,tx)` and reused across the `numRx` outputs
— but here each block fixes one `rx`, so cross-`rx` reuse is *not* captured. Two options:
- **(a) keep `rx` in the block** (block = one `(c, sc-tile)`, each thread computes all 4 `rx`):
  loads `y` once per `(c2,tx)`, reuses for 4 `rx` → cuts `y` traffic 4× (y is 1/3 of bytes).
  Costs 4 accumulators/thread. **Recommended.**
- **(b) keep `rx` in the grid** (above): simpler, more blocks (better occupancy), more `y` reads.

Phase-1 default: **(a)** — `gridDim=(C, ceil(numSc/256))=2·13=26 blocks`, each thread does 4
`rx`. If profiling shows HBM under-saturated (too few blocks → 26 is low for 132 SMs), apply
**split-K over `tx`**: add a `gridDim.w = numTx/16` axis, each block sums a 16-`tx` slice into
a partial, then a tiny reduce kernel (or `atomicAdd` on cf32 via 2×float) finalizes — raises
block count ~4× and recovers occupancy. (The kernel is ~4 µs at HBM peak; the budget is the
full `T_sym` on the throughput clock, so even 2–3× off-peak is fine — measure first.)

**Bandwidth check:** `H` 13.4 MB + `y` ~0.84 MB (with reuse) + `r_dl` 3.35 MB write ≈ 17.6
MB/symbol → ~0.49 TB/s, ~15 % of HBM. Comfortable.

---

## E.6 K3 — Channel-apply UL + K4 — Combine

### K3 — Channel-apply UL (RU receive)

**Computes** the signal received at cell `c`'s 64 RU antennas (ADR 0002 §2, UL):
```
r_ul[c][rx][sc] = noise + Σ_{u sched on sc} doppler[c][u] · Σ_{ueTx} H_ul[u][c][rx][ueTx][sc] · x_ul[u][ueTx][sc]
```
Under SU all-to-all, contributors on `sc` = the UE each cell scheduled there (`C` UEs), each
with `numUeTx` tx antennas → `C·numUeTx = 4` terms per `(rx,sc)`.

| | |
|---|---|
| **Input** | `H_dl` (via reciprocity, below); `x_ul[U][numUeTx][numSc]` cf32; `doppler`; `d_ulContrib[C][numSc]` (UEs scheduled on `sc`) |
| **Output** | `r_ul[C][64][numSc]` cf32 |
| **Grid** | `(C, 64/RXT, ceil(numSc/256))` — `RXT = 8` rx per block in registers → 2·8·13 = **208 blocks** |
| **Block** | 256 threads, one per `sc`; each holds `RXT=8` rx accumulators |

**Map:** thread `t`→`sc`; loop contributor `(u,ueTx)` [4 terms]: load `x = x_ul[u][ueTx][sc]`
once, reuse across the 8 rx; load `H_ul[u][c][rx][ueTx][sc]` for the block's 8 rx; `acc[r] +=
dopp·H·x`. **Coalesced** on `sc` (innermost) for both `H_ul` and `x_ul`. `x_ul` reused 8×.

**Reciprocity — DECIDED (UE tx=2 ⊆ rx=4): use it; do not store `H_ul`.** Because the UE's
2 tx antennas are a subset of its 4 rx antennas, the UL path equals the DL path on those
elements:
```
H_ul[u][c][rx_ru][ueTx][sc]  =  H_dl[c][u][ ueTxToRx[ueTx] ][ rx_ru ][sc]
```
with a small config map `ueTxToRx[numUeTx]` (which 2 of the 4 rx elements transmit; default
`{0,1}`). K3 indexes the **existing `H_dl`** — no second table (saves ~107 MB and keeps DL/UL
channels physically consistent).

**Coalescing is preserved:** the read is `H_dl[c][u][ueTxToRx[ueTx]][rx_ru][sc]` with the
thread/`sc` axis still innermost and `rx_ru` (the block's 8-wide rx tile) mapped onto `H_dl`'s
`tx` axis — a warp of consecutive `sc` is still contiguous. `ueTx` (only 2 iters) is the
non-unit inner loop. A separate-table fallback remains possible if a future UE has tx ⊄ rx.

### K4 — Combine (UL receive combining, `64 → rank`)

**Computes** per allocation `(cell c, UE u, sc range, rank, beamId_ul[])`:
```
z[c][l][sc] = Σ_{rx<64} combineBook[beamId_ul[l]][rx] · r_ul[c][rx][sc]
```

| | |
|---|---|
| **Input** | `r_ul[C][64][numSc]` cf32; `combineBook`; `d_allocs` (UL) |
| **Output** | `z[C][rank][numSc]` cf32 |
| **Grid** | `(MAX_ALLOCS, ceil(maxScLen/256))` static (E.3 note); idle blocks early-return |
| **Block** | 256 threads (one per `sc`); shared `Comb[rank][64]` cf32 = 2 KB |

**Map:** block gathers `Comb[l][rx] = combineBook[beamId_ul[l]][rx]` (64·rank loads) into
shared, `__syncthreads()`; thread `t`→`sc`: for `l<rank`, `z[c][l][sc] = Σ_{rx} Comb[l][rx] ·
r_ul[c][rx][sc]`. `r_ul[c][rx][sc]` and `z[c][l][sc]` are `sc`-innermost → coalesced; `rx` is
the reduction loop. `64·rank` MACs/sc — light.

---

## E.7 K0 — Ingress convert/scatter, K5 — Egress pack, and noise

### K0 — ci16 → cf32 convert (ingress)

Phase 1 is **host-staged** ([ADR 0007](../decisions/0007-process-topology-doca-deferral.md)):
the **ORU process** already de-frames Spec B sections into the host buffer
`x_dl_host[C][rank][numSc]` (`ci16`, [Spec F](oru-interface-contract.md)). ORCA `cudaMemcpyAsync`
H2D-copies it to a GPU staging slot `x_dl_raw` (`ci16`); **K0 is a flat convert** —
`x_dl[c][l][sc] = toCf32(x_dl_raw[c][l][sc])` (no scatter; the section layout was done host-side).

- **Input:** `x_dl_raw[C][rank][numSc]` ci16 (H2D-staged). **Output:** `x_dl[C][rank][numSc]` cf32.
- **Grid:** static, flat over `C·rank·numSc` elements (sc innermost); thread per element →
  coalesced `ci16` read (4 B) / `cf32` write (8 B).
- *(When DOCA returns — deferred-goals — the NIC DMAs straight to GPU and K0 reabsorbs the
  per-section scatter; the host H2D disappears.)*

### K5 — cf32 → ci16 pack (egress to vDU)

Inverse of K0: `z[c][l][sc]` cf32 → `z_host_dev[C][rank][numSc]` `ci16` (saturating round,
scale per Spec B `udCompParam`), flat per-element, sc-innermost → coalesced. ORCA then D2H-copies
`z_host_dev` → host `ulOutRing` ([Spec F](oru-interface-contract.md)); the **ORU process**
packetizes it per Spec B (eAxC `{cell,layer}`, PRB ranges) and sends to the vDU (DOCA deferred,
ADR 0007). BFP packing is deferred (Spec B.4).

### Noise (AWGN) — used inside K2/K3

Deterministic and stateless for golden-model reproducibility (no per-element `curand_init`
cost): **Philox counter-based** —
```
curandStatePhilox4_32_10_t st;
curand_init(noiseSeed, /*subseq*/ linearIdx(dir,u,rx), /*offset*/ sc, &st);
float2 n = curand_normal2(&st);  acc += cf32(n.x, n.y) * noiseStd;
```
`noiseSeed` is per-run config; the `(subsequence, offset)` keying makes the draw a pure
function of `(symbol, stream, sc)` → identical on GPU and the CPU golden model.

---

## E.8 Occupancy, launch tuning & open items

**CUDA-graph constraint — grids must be static (important).** A captured graph bakes each
node's `gridDim` at capture time, but `numAllocs`, `scLen`, and `numSections` **vary per
symbol** with scheduling. Therefore **every hot kernel uses a static grid** sized to a
worst-case bound, and guards at runtime:
- **Full-band-tiled kernels** (K2 DL, K3 UL): grid from `numSc`/`numRx`/`C` (all fixed) +
  per-`sc` lookup maps (`d_victim`, `d_ulContrib`) — naturally static; gaps → thread returns.
- **Per-allocation kernels** (K1 precode, K4 combine, K0/K5): grid = `MAX_ALLOCS` (or
  `MAX_SECTIONS`) × sc-tiles; blocks with `blockIdx.x ≥ numAllocs` **early-return**. Set
  `MAX_ALLOCS` generously (e.g. **512**) and `MAX_SECTIONS` to the FH worst case.
- Per-symbol the host updates only **device scalars/tables** (`numAllocs`, `d_allocs`,
  `d_victim`, …) — never grid dims — so the same graph replays every symbol.

**Occupancy / HBM saturation (the bandwidth-bound K2/K3 matter most).** 13.4 MB at HBM peak ≈
4 µs; saturating 3.35 TB/s needs many resident warps. Block counts: K2(a) 26, K2(b)/UL 104–208,
K1/K4 ≤ MAX_ALLOCS. If profiling shows HBM under-saturated:
- **Split-K over `tx`** (K2) / over `rx` (K3): add a reduction axis (16-wide slices) → ~4×
  blocks, finalize with a tiny reduce (or 2×`float` `atomicAdd`).
- Prefer K2 **variant (b)** (rx in grid) for raw block count, or **(a)** (rx in block) for less
  `y` traffic — measure both. Budget is the full `T_sym` (throughput clock), so even 2–3× off
  peak closes; tune only if measured.

**Memory/vectorization:** load `H` as `half2` (4 B/thread, warp = 128 B, 1 transaction); `y`/`r`
as `float2` (cf32). Codebooks gathered to **shared** per block (or `__ldg`/constant — they're
< 1 MB). `doppler[c][u]` loaded once per `(c2,u)` and reused over the `tx`/`rx` loop.

**Open items:**
- ~~Reciprocity for `H_ul`~~ → **DECIDED**: UE tx=2 ⊆ rx=4 → reuse `H_dl`, no `H_ul` table (E.6).
- ~~`numUeTx` confirmation~~ → **`numUeTx = 2`, `numRx = 4`** (also fixes Spec D UL slot size).
- `MAX_ALLOCS` / `MAX_SECTIONS` worst-case sizing vs the FH scheduler.
- K0 as a graph node vs ingest-thread pre-step (CPU-controlled DOCA).
- Split-K threshold — gate on measured HBM utilization on the target H100.
- `ueTxToRx[2]` element mapping (default `{0,1}`) — confirm against the UE antenna config.

---

### Status — **complete (steps 1–8).**
Covers numeric formats, full GPU memory layout & allocation, the CUDA-graph dataflow, and all
six hot-path kernels (K0–K5) with grid/block/warp/thread maps, coalescing, shared-mem use, and
launch-config rationale. Open items are tuning/decision points, not gaps. Feeds the `dsp/` and
`channel/` implementation (MILESTONES Stages 2–7).
