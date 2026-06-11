# Spec G — Offline CIR-table toolchain & format

**Status:** Draft (source of truth once accepted). No implementation yet.

Defines the **offline channel toolchain** and the **on-disk CIR-table format** that feeds
ORCA's slow plane. It is the concrete realization of
[ADR 0002 §5](../decisions/0002-multi-cell-interference-mobility.md) (precomputed
per-`(cell, grid-point)` CIR table + slow-plane lookup). The OptiX ray tracer is the
**offline generator** of this table; **no ray tracing ever runs on the per-symbol or
per-move runtime path** (ADR 0002 §5).

> **Distinct from Spec C.** Spec C (deferred, [deferred-goals #2](../deferred-goals.md#spec-c))
> decides **how `H` is applied on the hot path** (per-SC vs per-PRB-group vs tap-domain,
> precision). **Spec G generates the offline channel table** the slow plane expands into the
> resident `H`. Spec G is **Phase-1 work** (MILESTONES Stages 4 & 8); Spec C is not.

Related: [ADR 0002](../decisions/0002-multi-cell-interference-mobility.md) (multi-cell /
mobility), [ADR 0001 §4](../decisions/0001-hot-path-synchronization.md) (indirection-cell
publish), [Spec E §E.2/§E.9/§E.10](gpu-kernel-design.md) (the `H_dl` tensor this feeds, its
lifecycle and dynamic-update path).

---

## G.1 Role & boundary — what is offline, what is runtime

```
   OFFLINE (build time, once per scenario)            RUNTIME (ORCA, slow plane only)
 ┌───────────────────────────────────────┐   file   ┌───────────────────────────────────┐
 │ scene + cell sites + UE grid (G.3/G.4) │  ──────► │ load table (host-resident, G.8)   │
 │   → OptiX ray trace per (cell, gp)(G.5)│  CIR     │ on UE move: lookup rays[cell][gp] │
 │   → prune / cap P / normalize    (G.6) │  table   │   → expand rays → per-antenna CIR │
 │   → serialize                    (G.7) │  (G.7)   │   → DFT taps→H[c][u][rx][tx][sc]  │
 └───────────────────────────────────────┘          │   → write H back buffer, publish  │
                                                     │     via indirection cell (ADR 1)  │
                                                     └───────────────────────────────────┘
```

- **Offline output = a geometric multipath (ray) table**, indexed by `(cell, grid-point)`,
  storing the *physics* of each link (path delays, angles, complex gains) — **not** the
  expanded per-subcarrier `H`. The expansion to `H[cell][ue][rx][tx][sc]` happens at runtime
  on the **slow plane** (G.8), so the table is array-geometry-independent and compact (G.2).
- **The hot path never reads this table.** It reads only the resident `H_dl`
  ([Spec E §E.2](gpu-kernel-design.md)), which the slow plane fills from the table on a UE
  move and publishes via the ADR 0001 §4 indirection cell.
- **Per-symbol Doppler is *not* in `H`.** Spec G supplies the per-path **angle** geometry
  the fast plane needs to form the per-`(cell,ue)` Doppler rotor (`doppler[c][u]`,
  Spec E §E.5); the rotor itself is fast-plane (ADR 0002 §5), never rewritten into the table.

---

## G.2 The central decision — store geometric paths, not antenna-CIR or per-SC `H`

A link's channel can be persisted at three levels. Spec G stores the **geometric path
(ray) list** and expands at runtime.

| Storage domain | Per-`(cell,gp)` size¹ | Array-dependent? | Runtime cost to get `H` | Verdict |
|---|---|---|---|---|
| **Geometric paths** (τ, AoD, AoA, gain) | `P·40 B` ≈ **640 B** | **no** | steering + DFT over `P` paths | **chosen** |
| Per-antenna CIR (`[rx][tx][tap]` complex) | `4·64·P·8 B` ≈ **512 KB** | yes (bakes array) | FFT taps→`H` only | rejected — 800× larger, re-trace on array change |
| Per-subcarrier `H` (`[rx][tx][sc]`) | `4·64·3276·4 B` ≈ **3.4 MB** | yes | none | rejected — 5000× larger; this *is* the resident GPU tensor, not a table |

¹ `P` = max paths/link (default **16**).

**Why geometric paths win:**
- **Compact** — the whole grid table fits in host RAM (G.11), so the table is **host-resident**
  and only the *active* links' expanded `H` (215 MB, Spec E) live on the GPU.
- **Array-independent** — changing the cell's 64-element array geometry, the UE's 4-element
  array, or `numSc` is a *runtime expansion* change, **not** a re-trace. The physics
  (rays) is captured once.
- **Carries the angles** the Doppler rotor and any future SRS/beam work need.
- **Cost is on the slow plane**, which has milliseconds (G.8) — the expansion is cheap.

The compromise: the slow plane must run the **ray→`H` expansion** (G.8) rather than a bare
table copy. That cost is bounded and off the hot path.

---

## G.3 Coordinate & grid model

- **World frame:** right-handed metric (meters); `+z` up. One scenario origin shared by all
  cells and the grid.
- **UE grid:** a uniform lattice of candidate UE positions. **Phase 1 = 2-D horizontal grid
  at a fixed UE height** (`ueHeight`), indexed `gp = iy·nx + ix`. A 3-D grid
  (`gp = (iz·ny + iy)·nx + ix`) is a format-compatible extension (`gridDims=3`); see G.12.
- **Spacing** `Δ` is uniform (default; per-axis spacing allowed in the header). The grid
  captures **large-scale geometry** (path delays/angles that drift over *meters*); **small-
  scale phase evolution between grid points is handled by the per-symbol Doppler rotor**
  (ADR 0002 §5), *not* by fine grid spacing. So `Δ` is set to the distance over which the
  **path geometry** changes materially (a path delay shifts ≳ one delay bin, or an angle by
  ≳ an array beamwidth), **not** to `λ/2`.
  - **Default `Δ = 1 m`** (FR1, ~3.5 GHz). Tighten only if a scenario's geometry decorrelates
    faster; finer `Δ` trades table size (G.11) for grid-quantization error
    ([deferred-goals #9](../deferred-goals.md#continuous-mobility)).
- **Grid ⇄ position:** `pos(ix,iy) = origin + (ix·Δx, iy·Δy, ueHeight)`. A UE's current `gp`
  is `round((pos − origin)/Δ)`; movement = a change in `gp` (slow-plane event, G.8).

---

## G.4 Scene & cell-site inputs (toolchain)

The offline tracer consumes:

| Input | Content |
|---|---|
| **Environment geometry** | triangulated scene (e.g. glTF/OBJ) — buildings, terrain, reflectors |
| **Material map** | per-surface EM properties (permittivity, conductivity / reflection model) |
| **Cell sites** | per cell: position, array orientation (boresight), **antenna array geometry** (default 64-TRX, e.g. 8×8 dual-pol UPA; element spacing), `numTx` |
| **UE array** | the emulated UE antenna geometry (`numRx = 4`; `numUeTx = 2 ⊆ numRx`, ADR 0006/Spec E §E.6), one config for all UEs in Phase 1 |
| **Grid spec** | `origin`, `Δ`, `nx, ny[, nz]`, `ueHeight`, carrier frequency `f_c` |
| **Trace params** | max bounces, `P_MAX`, weak-path prune threshold, max excess delay |

**Antenna geometry is recorded in the table header** (G.7) but **applied at runtime** (G.8),
keeping rays array-independent (G.2). The tracer needs the cell **position/orientation** to
emit AoD in the cell-array frame and the UE position to emit AoA in the UE-array frame.

---

## G.5 Ray-tracing stage (OptiX)

For every `(cell c, grid-point gp)` pair the tracer emits the set of significant propagation
paths between cell `c`'s array phase-center and the UE position `pos(gp)`:

- **Method:** OptiX path tracing / shooting-and-bouncing rays with image-method refinement
  for specular paths; LoS test + up to `maxBounces` reflections (diffraction optional, G.12).
- **Per path, the tracer yields:** complex gain `g_p` (linear, includes path loss + reflection
  coefficients + phase), excess delay `τ_p` (relative to first arrival), departure angles
  `(AoD_az, AoD_el)` in the **cell-array frame**, arrival angles `(AoA_az, AoA_el)` in the
  **UE-array frame**, and flags (LoS / reflection order).
- **Reciprocity:** the traced link is reciprocal; the same rays serve DL (cell→UE) and UL
  (UE→cell) — UL reuses DL via Spec E §E.6 (UE tx ⊆ rx), so **only one ray set per
  `(cell, gp)` is stored**, never a separate UL table.
- **Aggregate path loss** per link is recorded (G.7) so the scenario layer can rank
  interferers for the neighbor-limited top-K set (ADR 0002 §3) without expanding `H`.

The trace is **embarrassingly parallel** over `(cell, gp)` and runs once per scenario; wall
time is a build-time concern, not a runtime budget.

---

## G.6 Path post-processing

Before serialization, per link:

1. **Prune** paths below `pruneThresh` dB relative to the strongest path (default −30 dB).
2. **Cap** to `P_MAX` strongest (default 16); record the actual `numPaths ≤ P_MAX`.
3. **Sort** by descending power (so `paths[0]` is dominant — convenient for single-rotor
   Doppler, G.8, and for truncation).
4. **Bound delay** — drop paths with `τ_p > maxExcessDelay` (default ≤ CP length so they stay
   within the OFDM guard; µ=1 normal CP ≈ 2.34 µs long / ~1.17 µs).
5. **Normalize** (optional) to a target average link gain for controlled SNR studies; record
   the normalization in the header for reproducibility.
6. **Mark outage** (`flags.noCoverage`) if no path survives — the slow plane then fills that
   link's `H` with zeros (a covered-but-silent link).

---

## G.7 On-disk format

Single binary file, little-endian, mmap-friendly, **fixed-stride link blocks** for O(1)
`(cell, gp)` lookup. Layout: header → cell descriptors → UE-array descriptor → link blocks.

### Header (fixed)

| Field | Type | Meaning |
|---|---|---|
| `magic` | char[8] | `"ORCACIR1"` |
| `version` | u32 | format version (current = 1) |
| `numCells` | u32 | physical cell sites |
| `gridDims` | u32 | 2 or 3 |
| `nx,ny,nz` | u32×3 | grid extents (`nz=1` if 2-D) |
| `originX/Y/Z` | f32×3 | world origin (m) |
| `spacingX/Y/Z` | f32×3 | grid spacing (m) |
| `ueHeight` | f32 | UE height (2-D grids) |
| `carrierHz` | f64 | carrier frequency `f_c` |
| `pMax` | u32 | `P_MAX` (paths/link stride) |
| `pathRecBytes` | u32 | `sizeof(PathRecord)` (= 40) |
| `linkBlockBytes` | u32 | `sizeof(LinkBlock)` (stride) |
| `normRefDb` | f32 | normalization reference (G.6.5), or `NaN` |
| `pruneThreshDb` / `maxExcessDelayS` | f32×2 | provenance (G.6) |

### Cell descriptor (`numCells` of them)

`{ f32 pos[3]; f32 boresight[3]; u32 numTx; u32 arrayType; f32 elemSpacing; f32 _rsv; }`
(`arrayType` enumerates UPA/ULA/dual-pol; the runtime steering kernel, G.8, interprets it.)

### UE-array descriptor (one)

`{ u32 numRx; u32 numUeTx; u8 ueTxToRx[4]; u32 arrayType; f32 elemSpacing; }` — `ueTxToRx`
matches Spec E §E.6 (which rx elements transmit; default `{0,1}`).

### Path record (`PathRecord`, 40 B)

| Field | Type | Meaning |
|---|---|---|
| `tau` | f32 | excess delay (s), relative to first arrival |
| `gainRe,gainIm` | f32×2 | complex path gain (linear) |
| `aodAz,aodEl` | f32×2 | departure angle, cell-array frame (rad) |
| `aoaAz,aoaEl` | f32×2 | arrival angle, UE-array frame (rad) |
| `flags` | u8 | bit0 LoS, bit1 ground-reflect, … |
| `_pad` | u8×3 | alignment |

### Link block (`LinkBlock`, indexed `[cell][gp]`)

`{ u16 numPaths; u16 flags; f32 pathlossDb; PathRecord paths[pMax]; }`
Stride = `linkBlockBytes`; **index** = `(cell·(nx·ny·nz) + gp)·linkBlockBytes`. `flags.bit0`
= `noCoverage` (G.6.6). `pathlossDb` feeds interferer ranking (G.5).

---

## G.8 Runtime loader & slow-plane expansion

**Load (ORCA startup):** map the file **host-resident** (read-only mmap or pinned read).
Validate `magic`/`version` and that `numCells`, the cell/UE array descriptors, `numSc`-implied
`f_c`, and `ueTxToRx` match ORCA's config — **hard stop on mismatch** (mirrors the Spec D/F
handshake discipline). The full table stays on the **host**; only expanded `H` reaches the GPU.

**On a UE move (slow plane, off the hot path):** for each affected active link
`(cell c, UE u now at grid `gp`)`:

```
block = table[c][gp]                                  // O(1) host lookup
for rx in 0..numRx, tx in 0..numTx, sc in 0..numSc:   // → fills H back buffer
   H[c][u][rx][tx][sc] = Σ_{p<block.numPaths}
        block.gain_p
      · a_tx(AoD_p)[tx]                                // cell 64-elem array steering
      · a_rx(AoA_p)[rx]                                // UE 4-elem array steering
      · exp(-j·2π·f_sc·τ_p)                            // delay → per-SC response (direct DFT)
   // stored as half2 (FP16 complex), Spec E §E.1
```

- The "CIR→`H` FFT" of ADR 0002 §5 is realized as a **direct DFT over the `P≤16` paths**
  (exact for arbitrary non-sample-aligned `τ_p`); a bin-to-sample-grid + cuFFT variant is an
  option only if `P` ever grows large (G.12).
- **Cost:** ≈ `P·numRx·numTx·numSc` ≈ 13.4 M cMAC per link; a full 64-link refresh
  (32 UE × 2 cells) ≈ 0.9 GcMAC — **sub-millisecond on the GPU slow-plane stream**, which
  never competes with the hot path (Spec E §E.10). Only the **changed** links are recomputed.
- **Publish:** write into the **back** `H` buffer, then the ADR 0001 §4 atomic pointer swap;
  partial updates follow Spec E §E.10 (delta-to-both recommended).

**Doppler handoff:** the scenario/fast plane forms `doppler[c][u]` from `block.paths[0]`'s
`AoA` and the UE velocity vector `v`: `f_d = (f_c/c)·(v·û_AoA)`, rotor increment
`Δφ = 2π f_d T_sym`. **Phase 1 = single dominant-path rotor per link** (Spec E `doppler[c][u]`);
true per-path Doppler / time-selectivity is deferred (G.12).

---

## G.9 Interpolation & handover

- **Default = nearest grid point** (no interpolation) — keeps mobility **grid-quantized and
  deterministic** (ADR 0002 §5). `Δ` (G.3) is chosen so quantization error is acceptable.
- **Optional spatial interpolation** (ADR 0002 §5 "optional") is **deferred**: interpolating
  geometric paths needs path **association** across adjacent grid points (birth/death, angle
  wrap), and interpolating the expanded `H` risks spatial amplitude cancellation. Either is a
  later refinement tied to [deferred-goals #9](../deferred-goals.md#continuous-mobility). When
  added, it is a slow-plane change only — **no table-format change**.
- **Handover / association** (serving cell + interferer top-K) is **scenario-layer** state, not
  Spec G; it consumes `pathlossDb` (G.7) to rank links. Spec G only supplies per-link physics.

---

## G.10 Determinism & the golden model

The table is **precomputed, versioned, and immutable at runtime**. Given the same table + the
same UE trajectory + the same noise seed (Spec E §E.7), the produced `H` — and therefore the
whole hot-path output — is **bit-reproducible**. This makes Spec G the channel half of the
golden-model reference: the CPU golden model expands the *same* rays with the *same* steering/
DFT to validate the GPU `dsp/`/`channel/` path. (The expansion math, G.8, is the contract both
sides implement.)

---

## G.11 Sizing (Phase 1)

Per link block: `4 + pMax·40 = 4 + 16·40 = 644 B` (→ 648 B padded).

| Scenario | Grid | Cells | Table size |
|---|---|---|---|
| Phase-1 default | 500 m × 500 m @ `Δ=1 m` = 250 k gp | 2 | 250e3·2·648 B ≈ **324 MB** |
| Fine grid | same area @ `Δ=0.5 m` = 1 M gp | 2 | ≈ **1.3 GB** |
| 7-cell ref (later) | 1 km × 1 km @ `Δ=1 m` = 1 M gp | 7 | ≈ **4.5 GB** |

All **host-resident** and comfortable; the GPU only ever holds the active expanded `H_dl`
(215 MB, Spec E §E.2). Table size scales with `cells · gridpoints · P` — the lever if a
scenario grows is `Δ` (G.3) or `P_MAX` (G.6), never the GPU budget.

---

## G.12 Open / implementation-time

- **Grid dimensionality** — 2-D fixed-height default; 3-D (`gridDims=3`) when UE altitude
  varies (drones, multi-floor). Format already carries it.
- **Interpolation** (G.9) — path-association vs `H`-domain; deferred (#9).
- **Per-path Doppler / time-selectivity** — Phase 1 is a single dominant-path rotor (G.8);
  multi-path Doppler is a fast-plane extension (no table change — angles are already stored).
- **Diffraction / scattering** in the tracer (G.5) — specular + ground-reflection first;
  UTD diffraction and diffuse scattering later.
- **Polarization / dual-pol** — `arrayType` reserves it; Phase 1 may model single-pol.
- **`P_MAX`, `pruneThresh`, `maxExcessDelay`** — set against a target delay-spread profile
  (the same target that, for the *hot path*, pins Spec C — kept consistent here).
- **Direct-DFT vs bin+cuFFT** expansion (G.8) — direct DFT for `P≤16`; revisit only if `P`
  grows.
- **Scene/material file format** (G.4) — glTF/OBJ + a material table; exact schema TBD.
- **Normalization convention** (G.6.5) — average-gain target vs raw physical gain.

---

### Status — **draft.** Specifies the offline OptiX→CIR-table pipeline, the geometric-path
storage decision (G.2), the coordinate/grid model, the on-disk format (G.7), and the runtime
slow-plane ray→`H` expansion (G.8) that feeds the resident `H_dl` (Spec E). Realizes
ADR 0002 §5; needed by MILESTONES Stages 4 & 8. Open items (G.12) are toolchain/scenario
parameters, not boundary gaps.
