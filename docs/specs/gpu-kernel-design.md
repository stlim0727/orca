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

> Shapes below are **logical**. The implementation-accurate form — flat 1-D arrays, the
> `numScP` (=3296) alignment padding on `sc`, exact index macros, and the
> `threadIdx.x→sc` mapping that *forces* `sc`-innermost — is in **[§E.11](#e11-exact-array-layouts-strides--the-layoutdecomposition-coupling)**.

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
| **Grid** | default `(C, ⌈numScP/256⌉, SPLITK)` = `(2, 13, 16)` = **416 blocks** (variant (a) + split-K, E.12); `rx` in-thread |
| **Block** | `TILE = 256` threads, one per `sc` (lane = `sc`) |

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

**Phase-1 default: variant (a) + split-K over `tx` (`SPLITK≈16`)** — `gridDim=(C,
ceil(numScP/256), SPLITK)` = `(2, 13, 16)` = **416 blocks**, each thread does 4 `rx` and a
`64/SPLITK`-tx slice; a tiny reduce (or `atomicAdd` 2×float) finalizes partials. Split-K is
**required, not optional** for the cold symbol: without it the 26-block launch under-saturates
HBM (~320 GB/s) and K2 takes ~63 µs ≫ `T_sym`; with it, ~7 µs cold / ~2 µs steady. Full
derivation: [§E.12](#e12-worked-example--k2-coalescing--occupancy-the-numbers).

**Bandwidth & occupancy:** ~20.3 MB/symbol cold (`H` 13.4 + `y` 6.7 + write 0.2). This is
*not* automatically "comfortable" — the naive 26-block launch under-saturates HBM and blows
`T_sym`. See the worked numbers and the required **split-K** fix in **[§E.12](#e12-worked-example--k2-coalescing--occupancy-the-numbers)**.

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

**Occupancy / HBM saturation (the bandwidth-bound K2/K3 matter most).** The trap: too **few
blocks** (not registers, not coalescing) leaves HBM under-saturated and blows `T_sym` — the
naive K2 launch is 26 blocks → ~63 µs (worked out in [§E.12](#e12-worked-example--k2-coalescing--occupancy-the-numbers)).
**Resolution — split-K is part of the default, not a tuning afterthought:**
- **K2:** variant (a) + **split-K over `tx` (`SPLITK≈16`)** → ~416 blocks, ~7 µs cold, ~2 µs
  steady (L2). **K3:** same idea, split-K over the 64 `rx_ru`.
- Finalize partials with a tiny reduce or 2×`float` `atomicAdd`.
- Steady-state both kernels are **L2-bound** (working set < 50 MB) → far under budget; the
  cold (post-swap / schedule-change) symbol governs the deadline.

**Memory/vectorization:** load `H` as `half2` (4 B/thread, warp = 128 B, 1 transaction); `y`/`r`
as `float2` (cf32). Codebooks gathered to **shared** per block (or `__ldg`/constant — they're
< 1 MB). `doppler[c][u]` loaded once per `(c2,u)` and reused over the `tx`/`rx` loop.

**Open items:**
- ~~Reciprocity for `H_ul`~~ → **DECIDED**: UE tx=2 ⊆ rx=4 → reuse `H_dl`, no `H_ul` table (E.6).
- ~~`numUeTx` confirmation~~ → **`numUeTx = 2`, `numRx = 4`** (also fixes Spec D UL slot size).
- `MAX_ALLOCS` / `MAX_SECTIONS` worst-case sizing vs the FH scheduler.
- K0 as a graph node vs ingest-thread pre-step (CPU-controlled DOCA).
- Split-K factor — `SPLITK≈16` derived for K2 (E.12); confirm on the actual H100 and for K3.
- `ueTxToRx[2]` element mapping (default `{0,1}`) — confirm against the UE antenna config.

---

## E.9 Tensor lifecycle, layout & cache behavior

### Notation map (the `y = H·P·x` view ↔ Spec E buffers)

`P` = precode/combine (beam codebook gather), `H` = channel, `x` = layer IQ in.

| Role in `y = H·P·x` (DL) / `y = P·H_h·x` (UL) | DL buffer | UL buffer |
|---|---|---|
| `x` — layer IQ in | `x_dl[C][rank][sc]` | `x_ul[U][numUeTx][sc]` |
| `P` — operator (gathered from codebook by `beam_id`) | `precodeBook` → `W[64][rank]` | `combineBook` → `C[rank][64]` |
| `H` — channel | `H_dl[C][U][rx][tx][sc]` | `H_dl` transposed (reciprocity, E.6) |
| **mid** = `P·x` (DL) / `H_h·x` (UL) | `y[C][64][sc]` | `r_ul[C][64][sc]` |
| `y` — final out | `r_dl[U][rx][sc]` (→vUE) | `z[C][rank][sc]` (→vDU) |

- **DL** applies `P` then `H`:  `x_dl ─K1(P·)→ y ─K2(H·)→ r_dl`.
- **UL** applies `H_h` then `P`: `x_ul ─K3(H_h·)→ r_ul ─K4(P·)→ z`.
  (Spec E's `y` is the DL *intermediate* `P·x`, **not** your equation's final `y` = `r_dl`.)

### Lifecycle of each tensor

| Tensor | Created by | Update rate | Read by | Lifetime / residence | Size | Reuse |
|---|---|---|---|---|---|---|
| **`H`** | offline CIR table → slow plane fills active HBM buffer | **slow** (few ms / on UE move) | K2, K3 every symbol | whole run; **resident HBM, double-buffered** | 215 MB | **high** (see cache) |
| **`P`** (codebook) | loaded once at startup | **static** | K1, K4 every symbol (→ shared) | whole run; resident HBM | 512 KB | **high** (tiny, reused over all `sc`) |
| **`x`** | ingress (K0 / vUE) | **per symbol** | K1 / K3 once | ~`N` symbols in ring, then recycled | 0.1–1.7 MB | none (stream once) |
| **mid** (`y`/`r_ul`) | K1 / K3 | per symbol | K2 / K4 once | one symbol; ring slot | 3.35 MB | none |
| **out** (`r_dl`/`z`) | K2 / K4 | per symbol | egress (vUE / K5) once | one symbol; ring slot | 0.2–3.35 MB | none |

### Most cache-friendly layout (the rules, and why)

1. **`sc` innermost on every tensor** → threads tile `sc` → the dominant `H` and the `x/y`
   streams are **coalesced** (128 B/64 B transactions). (E.1 golden rule.)
2. **`H` in `half2`** → halves the bottleneck traffic; converted to `cf32` in-register.
3. **Contraction axes are loops, not thread axes** — `tx` (K2), `rx` (K3/K4) — so the
   strided access is amortized in registers, never in the coalesced load.
4. **`P` lives in shared memory** (gathered once per block) — reused across all `sc` of the
   allocation; never re-streamed from HBM.
5. **`x`/`y`/out are pure streams** — written once, read once, no temporal reuse → they just
   need coalescing (rule 1), not caching.

### The cache win on `H` (why SU is comfortable)

Under SU 2-cell the **per-symbol `H` working set is ~13.4 MB**, which **fits H100 L2
(50 MB)**. Scheduling is stable over many symbols (a UE holds its PRBs for ≥ a slot), so the
*same* `H[cell][ue][sc]` lines are re-read each symbol → **served from L2, not HBM**.
Effective HBM `H` traffic → ~0 in steady state; HBM is touched only on a **buffer swap**
(address changes → L2 refills) or a slow-plane update. So `H` is "read every symbol" but
**mostly an L2 hit** — the 0.38 TB/s figure is the *cold* worst case, not the steady state.

---

## E.10 Dynamic `H` update

### What actually needs to change, and how fast

| Channel effect | Handled by | Rewrites `H`? | Rate |
|---|---|---|---|
| **Doppler / fast phase** within a position | per-symbol **phase rotor** on the fast plane (E.5) | **no** | per symbol, free |
| **UE grid move** (geometry change) | slow-plane lookup refills affected `(cell,ue)` links | yes (partial) | few ms / on move |
| **Fast / continuous mobility, time-varying taps** | more frequent slow-plane regeneration | yes | "in case we need it" |

**Key point:** most "dynamic channel" behavior is **Doppler**, which the per-symbol rotor
applies *without touching `H`*. `H` itself only changes on **geometry** change (position),
which is slow. So dynamic-`H` rewriting is rarely on the critical path.

### Update mechanism (tear-free, graph-safe — baseline)

Double buffer + indirection cell (ADR 0001 §4):
1. Slow plane writes the **back** `H` buffer (only the changed `(cell,ue)` links).
2. Publish = a single **atomic pointer write** to `d_H_active`.
3. The captured graph reads `const half2* H = *d_H_active;` once per replay → sees one
   consistent generation (possibly one update stale — the intended slow/fast decoupling).
No locks, no graph re-capture, no hot-path cost beyond the pointer read.

### Partial-update options (avoid a 215 MB full copy when updates are frequent)

After a swap the back buffer is one generation stale, so incremental updates need care:

| Option | How | Cost | When |
|---|---|---|---|
| **A. Delta-to-both** (recommended) | apply each changed link to **both** buffers (back now, the other after swap) | 2× the *sparse* delta writes | mobility (few links/update) |
| **B. Full copy + deltas + swap** | D2D copy active→back, apply deltas, swap | ~215 MB D2D ≈ 64 µs | simplest; fine at ms rates, not per-symbol |
| **C. Link-granular double buffer** | double-buffer per link with a generation tag | more bookkeeping | very frequent, sparse |

The `H` update (incl. CIR→`H` FFT for changed links) runs on a **separate slow-plane stream**,
never competing on the hot path; only the pointer swap is observed by the graph.

### Feasibility: SU tolerates even fully-dynamic `H`

Worst case — `H` changes **every symbol** so there is **no cross-symbol L2 reuse** — the SU
per-symbol read is still only **~0.38 TB/s (~11 % of HBM)** (ADR 0005). So a fully-dynamic
channel is **not a feasibility risk under SU**; it is purely the consistency-mechanism choice
above. (Under MU it would reopen the ADR 0002 §6 wall — another reason MU is deferred.) If a
future need demands per-symbol *geometry* updates, escalate to **tap-domain** application
(Spec C) rather than rewriting per-`sc` `H`.

---

## E.11 Exact array layouts, strides & the layout↔decomposition coupling

The logical tensors are **multidimensional**, but each is allocated as a **flat 1-D device
array + an index macro** (CUDA kernel params can't carry dynamic multidim extents; flat is
also what makes the stride explicit). The **innermost axis is `sc`** for every hot tensor —
this is forced, not stylistic (see "the coupling" below).

### Alignment padding

A warp tiles **32 consecutive `sc`**. For each tensor *row* (a fixed choice of all outer
indices) to start on a 128-byte boundary — so a warp's load is **one** aligned transaction,
not two straddling a cache line — pad the innermost dimension:

```
numScP = roundUp(numSc, 32) = roundUp(3276, 32) = 3296      // 20 padding subcarriers
```

`numScP·sizeof(half2) = 3296·4 = 13184 = 103·128` (aligned); `numScP·sizeof(cf32) = 26368 =
206·128` (aligned). Kernels **guard `if (sc >= numSc) return;`** — the padding exists only to
align row starts, never carries data.

### Flat index macros (row-major, `sc` innermost)

With `RANK=4, RX=numRx=4, TX=numTx=64, UETX=numUeTx=2, U, C`:

| Tensor | Element | `idx(...)` into the flat array | Bytes/row |
|---|---|---|---|
| `H_dl[c][u][rx][tx][sc]` | half2 | `((((c*U+u)*RX+rx)*TX+tx)*numScP)+sc` | 13184 |
| `x_dl[c][l][sc]` | cf32 | `((c*RANK+l)*numScP)+sc` | 26368 |
| `y[c][tx][sc]` (DL mid `P·x`) | cf32 | `((c*TX+tx)*numScP)+sc` | 26368 |
| `r_dl[u][rx][sc]` (DL out) | cf32 | `((u*RX+rx)*numScP)+sc` | 26368 |
| `x_ul[u][uetx][sc]` | cf32 | `((u*UETX+uetx)*numScP)+sc` | 26368 |
| `r_ul[c][rx_ru][sc]` (UL mid `H_h·x`) | cf32 | `((c*TX+rx_ru)*numScP)+sc` | 26368 |
| `z[c][l][sc]` (UL out) | cf32 | `((c*RANK+l)*numScP)+sc` | 26368 |

**Outer-dim order is free** (chosen for logical clarity); **`sc` innermost is mandatory.**

### The coupling: decomposition ⇒ lane axis ⇒ innermost dimension

A warp's 32 lanes (`threadIdx.x & 31`) must hit 32 **consecutive addresses** to coalesce.
Whatever tensor axis is mapped to `threadIdx.x` is therefore forced to be the array's
unit-stride axis. ORCA maps **`threadIdx.x → sc`** in every hot kernel, which is *why* every
tensor is `sc`-innermost. Concretely (`TILE=256`, 8 warps):

| Kernel | `blockIdx` → | `threadIdx.x` → | per-thread loop |
|---|---|---|---|
| K1 precode | `(allocIdx, sc-tile)` | `sc` | `tx` (64), `l` (rank) |
| K2 channel DL | `(c, sc-tile)` *(rx in-thread)* | `sc` | `c2` (2), `tx` (64), `rx` (4) |
| K3 channel UL | `(c, rx_ru-tile, sc-tile)` | `sc` | contributor `u`, `uetx` (2) |
| K4 combine | `(allocIdx, sc-tile)` | `sc` | `rx` (64), `l` (rank) |

Because the lane axis is `sc`, **both the dominant `H`/`y` reads and the `r`/`z` writes are
coalesced** (a warp = 32 contiguous `sc` = one aligned 128 B / 256 B transaction). The
contraction axes (`tx` in K2, `rx` in K4) are the *per-thread loops*, where strided access is
amortized in registers and never hits the coalesced path. (Victim `u` is constant within a
PRB allocation, so a warp shares `u` except at the few allocation boundaries.)

### The one real fork (and why we pick `sc`-as-lane)

| | **A — `sc`-as-lane, thread-per-output, contract in a loop** *(chosen)* | **B — contraction-as-lane, warp-per-output, warp-reduce** |
|---|---|---|
| Innermost axis | `sc` (all tensors) | `tx`/`rx` |
| Reads coalesced | yes | yes |
| **Output writes** | **coalesced** (32 `sc`/warp) | **scattered** (1 elem/warp) |
| Reduction | none (register accumulate) | warp-shuffle (log₂32 ×2) |
| Parallelism | C·numRx·numSc threads | C·numRx·numSc **warps** (≫) |
| Best when | default; simple, all-coalesced | HBM under-saturated → need more warps |

**Default = A** — it coalesces reads *and* writes, needs no reduction, and the SU per-symbol
volume is small enough that its occupancy (~100s of blocks, boosted by **split-K over `tx`**,
E.8) saturates HBM. **B** is the escalation if profiling shows the bandwidth-bound K2/K3
under-utilizing HBM at scale; switching to B flips the innermost axis to the contraction
dimension *for `H` and `y` only* and adds a warp-reduction — a layout change, hence flagged
here rather than chosen blind.

> **E.2's tables show logical shapes; this section (E.11) is the implementation-accurate
> form** — flat arrays, `numScP` padding, index macros, and the `threadIdx.x→sc` mapping.

---

## E.12 Worked example — K2 coalescing & occupancy (the numbers)

Phase-1 SU 2-cell, `H` half2, `y`/`r_dl` cf32, block = 256 (8 warps), µ=1 (T_sym = 35.7 µs).
H100: 132 SMs, HBM 3.35 TB/s, L2 50 MB, 65536 regs/SM, HBM latency ≈ 500 ns.

### Coalescing (per access, Scheme A, lane = `sc`)

A warp = 32 lanes = 32 consecutive `sc`. With `numScP` padding, each row start is 128-B
aligned, so:

| Access | Per-warp request | Transactions | Bus efficiency |
|---|---|---|---|
| `H_dl[c2][u][rx][tx][sc]` (half2) | 32·4 B = **128 B** | **1 sector** | 100 % |
| `y[c2][tx][sc]` (cf32) | 32·8 B = 256 B | 2 sectors | 100 % |
| `r_dl[u][rx][sc]` write (cf32) | 256 B | 2 sectors | 100 % |

No wasted bytes. (Only warps straddling a PRB-allocation boundary — where victim `u` changes
mid-warp — split into ≤2 segments; rare, since allocations span ≥ 1 PRB = 12 `sc`.)

### Per-symbol HBM volume (variant (a): `rx` in-thread → `y` reused across the 4 rx)

```
H  read = C·numSc·numRx·(C·numTx)·4B = 2·3276·4·128·4   = 13.4 MB   (each H coeff once)
y  read = C·(C·numTx)·numSc·8B        = 2·128·3276·8      =  6.7 MB   (y read by both victims/sc, reused over 4 rx)
r_dl wr = C·numSc·numRx·8B            = 2·3276·4·8        =  0.2 MB
                                                          ──────────
                                                  total ≈ 20.3 MB / symbol  (cold, no L2 reuse)
```
(Variant (b), `rx` in-grid, loses the y-reuse → `y` read = 26.8 MB → total ≈ 40.4 MB.)

### The occupancy trap (why the naive launch fails)

Variant (a) grid = `(C, ⌈numScP/256⌉)` = `(2, 13)` = **26 blocks** = 208 warps. Registers
(~40/thread → ~6 blocks/SM possible) are *not* the limit — **block count is**: 26 blocks
light up only 26 of 132 SMs.

Saturating HBM needs ≈ `BW · latency = 3.35 TB/s · 500 ns ≈ 1.68 MB` of loads in flight
(Little's law). 208 warps × ~6 outstanding 128-B loads ≈ **0.16 MB in flight → ~320 GB/s
achieved**, so:
```
20.3 MB / 320 GB/s ≈ 63 µs  ≫  T_sym 35.7 µs      ❌ throughput clock BLOWN
```
Variant (b)'s 104 blocks → ~1.28 TB/s → 40.4 MB / 1.28 TB/s ≈ **31 µs** — fits, but tight and
with 2× the traffic.

### The fix — split-K over `tx` (required, not optional, for the cold symbol)

Add a reduction axis over the 64 `tx`: `gridDim.z = SPLITK`, each block sums a `64/SPLITK`-tx
slice into a partial; a tiny follow-up reduces partials (or `atomicAdd` 2×float). **`SPLITK =
16`** on variant (a): grid `(2, 13, 16)` = **416 blocks** = 3328 warps → ~2.5 MB in flight →
**saturates ~3.35 TB/s**, keeping variant (a)'s low 20.3 MB:
```
20.3 MB / ~3.0 TB/s ≈ 6.8 µs   ✅  (cold)   ·   reduction adds < 1 µs
```

### Steady state: it's actually L2-bound

With stable scheduling the 13.4 MB `H` + 3.4 MB `y` working set is **L2-resident** (< 50 MB),
so HBM sees only the 0.2 MB `r_dl` write. K2 then runs at **L2 bandwidth (several× HBM)** →
**≈ 1.7 µs**. The 6.8 µs cold figure governs the per-symbol deadline (cold recurs on H-swap /
schedule change); the steady state is far under it.

### Verdict

**Default = variant (a) + split-K over `tx` (`SPLITK≈16`).** Cold ≈ 7 µs, steady ≈ 2 µs —
both comfortably inside the compute stage's `T_sym` budget. The naive 26-block launch
(my earlier "~15 % HBM" estimate) **does not** meet the budget — occupancy, not coalescing,
was the gap. K3 (UL) is analogous; K1/K4 are compute-trivial and untouched by this.

---

### Status — **complete (steps 1–8 + lifecycle E.9 + dynamic-`H` E.10 + exact layouts E.11 + worked example E.12).**
Covers numeric formats, full GPU memory layout & allocation, the CUDA-graph dataflow, all six
hot-path kernels (K0–K5) with grid/block/warp/thread maps, coalescing and launch configs, the
`H`/`P`/`x`/`y` lifecycle + cache-friendly layout, and the dynamic-`H` update path. Open items
are tuning/decision points, not gaps. Feeds the `dsp/` and `channel/` implementation.
