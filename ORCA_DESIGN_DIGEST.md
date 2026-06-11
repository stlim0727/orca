# ORCA — condensed design digest

Distilled from `docs/` for LLM context. Facts only; rationale/alternatives/revisit-triggers
dropped. Authoritative source remains `docs/`. Status: design complete, **no code yet**, µ=1.

## 1. What ORCA is

Real-time GPU app standing in for the **O-RU** between a real 5G NR **vDU** and **emulated
UEs (vUE)**. Per OFDM symbol: precode → channel-apply → combine, with a ray-traced channel,
multi-cell interference, grid UE mobility. Covers the RU role (precoding/combining); never
sees Ethernet.

**Three in-box processes** (ADR 0007): `ORU process ↔ ORCA ↔ vUE`.
- **North (vDU):** separate **ORU process** owns NIC + fronthaul framing (Spec B) over
  Ethernet (**DOCA deferred**; kernel/DPDK), relays layer IQ to ORCA via **host shm + H2D/D2H**
  (Spec F).
- **South (vUE), in-box:** **DPDK shm** control; bulk per-antenna IQ in HBM via **CUDA IPC**
  (Spec D).

## 2. Phase-1 locked scope

- **SU-MIMO** (one UE per time-freq resource; OFDMA separates UEs; per-resource layers = UE
  rank ≤4). MU-MIMO deferred.
- **2 cells**, **all-to-all** inter-cell interference, grid mobility.
- All `H` resident as **per-subcarrier FP16** (~0.38 TB/s, ~11% HBM → fits; Spec C deferred).
- **µ=1** (T_sym=35.7µs). µ≥2 out of scope.
- **Beam-indexed precoding** (resident codebook, `beam_id` from vDU C-plane); SRS/ZF/MMSE/SVD
  deferred (no cuSOLVER).
- Single PCIe H100; vUE in-box GPU PHY.
- **Hot-path sync:** CUDA Graph + indirection-cell double buffering; stream-ordering + one
  event/symbol. **No** persistent kernels/doorbells/system-scope flags at µ=1 (ADR 0001).
  Stepping stone: plain stream-ordered launches first, then wrap identical kernels in a graph.
- **Slow plane never touches hot path**; publishes via atomic write to a device indirection
  cell (`H_active`).
- **Never stall on a late symbol** — zero-fill missing PRBs, advance (Spec A §A.4).

## 3. Two-clock budget (ADR 0003 / Spec A)

- **Throughput (rate):** one symbol per T_sym=35.7µs; bottleneck stage ≤ T_sym. Binding =
  compute stage HBM bandwidth for `H`. Pipelining does NOT relax it.
- **Latency (deadline):** Σ stage latencies ≤ L_max (working ~70µs ≈ 2·T_sym; real value TBD).
- `T_proc ≤ 3µs` is **retired**. Representative Σ ≈ 60–65µs cold / ~50µs steady → closes.
- N≥3 symbol ring: ingest(n+2) ∥ compute(n+1) ∥ egress(n).
- **Symbol period by µ:** µ0=71.4µs, **µ1=30kHz SCS=35.7µs (target)**, µ2=17.8µs, µ3=8.9µs(FR2).
- `T_air(SFN,slot,sym) = T0 + (SFN·slotsPerFrame+slot)·T_slot + symStart[sym]`;
  `slotsPerFrame = 10·2^µ`.
- DL reassembly deadline `D_r = T_air − T_egress − T_proc − T_margin`;
  UL `D_ul = T_air + T_ul_offset − T_proc − T_egress − T_margin`.
- Drop: DL incomplete by D_r → zero-fill missing PRBs + `late_drop`; UL missing → noise-only.

## 4. Dimensions & numeric formats (Spec E.1)

| Symbol | Meaning | Phase-1 value |
|---|---|---|
| C | cells | 2 |
| U_c, U | UE/cell, total | 16, 32 |
| numTx | cell TRX (Tx) | 64 |
| numRx | UE rx antennas | 4 |
| numUeTx | UE tx antennas (UL) | 2 (⊆ rx; `ueTxToRx` default {0,1}) |
| numSc | subcarriers (273 PRB·12) | 3276 |
| numScP | sc padded to 32 | 3296 |
| rank | layers per SU resource | ≤4 |
| numBeams | codebook size | 1024 |
| prbGroupSize | | 4 |
| N_ring | symbol slots | 4 |
| MAX_ALLOCS | | 512 |

**Formats:** `H` = **half2** (FP16 complex, 4B; →cf32 in-register). IQ (`y,x,r,z`) = **cf32**
(8B). codebooks/doppler = cf32. fronthaul wire = **ci16** (convert at K0/K5).
`cf32={float re,im}`; `half2={__half re,im}`.

**Golden rule:** `sc` is innermost (unit-stride) on EVERY tensor → coalesced.

## 5. Tensor layouts & flat index macros (Spec E.2/E.11)

`numScP = roundUp(numSc,32) = 3296` (rows 128B-aligned; kernels guard `if(sc>=numSc)return`).
Row-major, sc innermost. `RANK=4,RX=4,TX=64,UETX=2`.

| Tensor | Elem | idx(...) | bytes/row |
|---|---|---|---|
| `H_dl[c][u][rx][tx][sc]` | half2 | `((((c*U+u)*RX+rx)*TX+tx)*numScP)+sc` | 13184 |
| `x_dl[c][l][sc]` | cf32 | `((c*RANK+l)*numScP)+sc` | 26368 |
| `y[c][tx][sc]` (DL mid P·x) | cf32 | `((c*TX+tx)*numScP)+sc` | 26368 |
| `r_dl[u][rx][sc]` (DL out→vUE) | cf32 | `((u*RX+rx)*numScP)+sc` | 26368 |
| `x_ul[u][uetx][sc]` (←vUE) | cf32 | `((u*UETX+uetx)*numScP)+sc` | 26368 |
| `r_ul[c][rx_ru][sc]` (UL mid) | cf32 | `((c*TX+rx_ru)*numScP)+sc` | 26368 |
| `z[c][l][sc]` (UL out→vDU) | cf32 | `((c*RANK+l)*numScP)+sc` | 26368 |

**Resident (slow-plane, double-buffered via indirection cell):** `H_dl` ≈215MB (×2 ≈430MB);
**no `H_ul`** (reciprocity); `precodeBook`/`combineBook` `[numBeams][numTx]` cf32 512KB each;
`doppler[C][U]`. Whole hot path < 0.5GB of 80GB.
Per-symbol control: `Alloc{u16 cell,ueId,scStart,scLen; u8 dir,rank; u16 beamId[4]}`,
`d_allocs[MAX_ALLOCS]`, `d_victim[C][numSc]` (sc→scheduled ueId, built once/symbol).

## 6. Hot-path kernels K0–K5 (Spec E)

Graph per symbol (TDD; inactive dir = empty alloc list → early-return). `threadIdx.x→sc` lane
in all hot kernels; TILE=256. **Static grids** (graph bakes gridDim); per-symbol host updates
device tables only. Contraction axes are per-thread loops.

- **K0 ingress convert:** `x_dl = toCf32(x_dl_raw ci16)`, flat per-element. (H2D-staged; with
  DOCA later, NIC DMAs to GPU and K0 reabsorbs scatter.)
- **K1 precode:** `y[c][tx][sc] = Σ_{l<rank} W[tx][l]·x_dl[c][l][sc]`, `W=precodeBook[beamId[l]]`
  gathered to shared. grid `(numAllocs, sc-tile)`. Compute-trivial.
- **K2 channel-apply DL (bandwidth-critical):**
  `r_dl[u][rx][sc] = noise + Σ_{c'∈C} doppler[c'][u]·Σ_tx H[c'][u][rx][tx][sc]·y[c'][tx][sc]`,
  victim `u=d_victim[c][sc]`. Default = **variant (a) rx-in-thread + split-K over tx
  (SPLITK=16)** → grid `(2,13,16)`=416 blocks. ~20.3MB/symbol cold; **split-K required** (naive
  26 blocks → ~63µs ≫ T_sym; with it ~7µs cold, ~2µs steady L2-bound). loads: H half2 1 sector,
  y/r cf32 2 sectors, 100% bus eff.
- **K3 channel-apply UL:** `r_ul[c][rx][sc]=noise+Σ_{u sched} doppler[c][u]·Σ_ueTx
  H_ul·x_ul[u][ueTx][sc]`. **Reciprocity:** `H_ul[u][c][rx_ru][ueTx][sc] =
  H_dl[c][u][ueTxToRx[ueTx]][rx_ru][sc]` — no H_ul table. grid `(C,64/RXT=8,sc-tile)`; split-K
  over rx.
- **K4 combine (64→rank):** `z[c][l][sc]=Σ_{rx<64} combineBook[beamId_ul[l]][rx]·r_ul[c][rx][sc]`,
  Comb gathered to shared. grid `(MAX_ALLOCS, sc-tile)`. Compute-light.
- **K5 egress pack:** `z cf32 → z_host_dev ci16` (saturating round), flat. Then D2H → host shm.
- **Noise (AWGN, in K2/K3):** Philox counter-based, `curand_init(seed, subseq=linearIdx(dir,u,
  rx), offset=sc)` → pure function of (symbol,stream,sc); matches CPU golden model.

Lifecycle: `H` slow/resident/high-reuse (L2-resident steady, 13.4MB working set < 50MB L2);
`P` static/shared; `x`/mid/out are per-symbol streams (no reuse, just coalesce).

## 7. Fronthaul packet format — Spec B (vDU↔ORU wire)

Ethernet + UDP/IPv4 (RoCEv2-friendly); one UDP flow per eAxC; network byte order.

**20-byte fixed header** (byte-aligned):

| Off | Field | Sz | |
|---|---|---|---|
|0|`ver_msgtyp`|u8|ver:4 \| msgtyp:4|
|1|`dir_cmp`|u8|dir:4 \| cmp:4|
|2|`iqWidth`|u8|bits/I or Q|
|3|`reserved0`|u8||
|4|`eAxC`|u16|antenna-carrier id|
|6|`sectionId`|u8||
|7|`numSections`|u8|0=unknown|
|8|`sfn`|u16||
|10|`slot`|u8||
|11|`sym`|u8|0..13|
|12|`startPrb`|u16||
|14|`numPrb`|u16||
|16|`seqNum`|u16|per-eAxC loss detect|
|18|`udCompParam`|u16|BFP exp/scale|

Enums: `msgtyp` 0=U-plane,1=C-plane,2=S-plane,3=Telemetry; `dir` 0=DL,1=UL; `cmp`
0=int16,1=BFP,2=int12,3=rsv; `ver`=1.
**eAxC** 16-bit `{DU_Port,BandSector,CC_ID,RU/UE_Port}` default 4/2/2/8 MSB→LSB; **cell** =
`{BandSector,CC_ID}`. Reassembly key `(cell,dir,sfn,slot,sym)` + `(startPrb,numPrb)` coverage.
- **U-plane** (msgtyp0): freq-domain IQ, `numPrb·12` SC; int16 default (4B/SC) or BFP.
- **C-plane** (msgtyp1): per section `{prbStart u16, numPrb u16, ueId u16, numLayers u8 (≤4),
  dir u8, beamId[numLayers] u16}` — allocation + beam_id (no W matrices). Must arrive before D_r.
- **S-plane** (msgtyp2): PTP timestamp + sfn/slot/sym; bootstraps T0.
- **Telemetry** (msgtyp3): format TBD.

## 8. ORU↔ORCA interface — Spec F (north, host-staged)

Bulk = host shm (ci16); control = DPDK shm; GPU boundary via cudaMemcpyAsync H2D/D2H. ORCA
pins shm (`cudaHostRegister`). ~3GB/s → ~1µs/symbol copy, no GPUDirect.
- **Bulk rings** (N≥3): `dlInRing` ORU→ORCA `x_dl_host[C][rank][numSc]` ci16 ≈105KB;
  `ulOutRing` ORCA→ORU `z_host[C][rank][numSc]` ci16 ≈105KB.
- **allocBlock[slot]** = `{u16 numAllocs; Alloc allocs[MAX_ALLOCS]}` → device `d_allocs`,
  build `d_victim`/`d_ulContrib`.
- **Control memzone** `"orca.oru.ctrl.v1"`: 4 SPSC rte_rings (dlDoorbell, dlReturn, ulDoorbell,
  ulReturn) + stats. Doorbell `{slotIdx u16, sfn u16, slot u8, sym u8, numAllocs u16, flags u8,
  seq u32}`.
- **Handshake:** ORU writes magic/version/config `(C,rank,numSc,iqElem,N,MAX_ALLOCS)` →
  ORU_READY; ORCA validates+pins → ORCA_ATTACHED. Mismatch = hard stop.
- **Sync:** H2D gates K0 (stream order/event); D2H completion gates ulDoorbell (event or
  `cudaLaunchHostFunc`); control ring never orders GPU memory.
- **API `OruTransport`:** `attach/detach, pollDl()->[(slotIdx,sfn,slot,sym,allocBlock*)],
  returnDl(slotIdx), publishUl(slotIdx,meta), reclaimUl()->[slotIdx]`. DOCA backend swap later
  removes the copies.

## 9. vUE↔ORCA interface — Spec D (south, in-box)

Two processes same host+GPU. Bulk IQ → shared HBM (CUDA IPC); control → DPDK shm; GPU ordering
→ CUDA IPC events. ORCA owns/allocates all regions; vUE attaches. Bulk never traverses control
ring.
- **Bulk rings** (N≥3, cudaMalloc): `dlRing` ORCA→vUE `r_dl[ue][rx][sc]`; `ulRing` vUE→ORCA
  `x_ul[ue][ueTx][sc]`. iqElem default cf32 (DL slot 3.35MB, UL 1.68MB; N=4≈20MB). SU leaves
  unscheduled SC zero-filled.
- **Control memzone** `"orca.vue.ctrl.v1"`: header+config, handshake (mem+event handles), 4
  SPSC rte_rings (dlDoorbell, dlReturn, ulDoorbell, ulReturn), stats. Doorbell `{slotIdx u16,
  sfn u16, slot u8, sym u8, flags u8, seq u32}`; return rings `{slotIdx, seq}`.
- **Config block:** `{numUe,numRx,numUeTx,numSc,iqElem,N_dl,N_ul,layoutId,cellCount}`; vUE
  refuses on mismatch.
- **Handshake:** ORCA creates memzone→INIT; cudaMalloc rings + IPC events
  (`cudaEventInterprocess|DisableTiming`) + get mem/event handles → ORCA_READY; vUE validates,
  `cudaIpcOpenMemHandle`/`OpenEventHandle` → VUE_ATTACHED; ORCA begins steady state.
- **Per-symbol DL:** ORCA writes r_dl→`cudaEventRecord(dlProduced[i])`→enqueue dlDoorbell; vUE
  dequeues→`cudaStreamWaitEvent(dlProduced[i])`→reads→`cudaEventRecord(dlConsumed[i])`→dlReturn;
  ORCA reuses slot after dlReturn credit + waits dlConsumed. UL symmetric. Backpressure →
  drop oldest, set flags.late, never block.
- **API `VueTransport`:** `attach/detach, publishDl(slotIdx,meta), reclaimDl()->[slotIdx],
  submitUl(slotIdx,meta), pollUl()->[(slotIdx,meta)], returnUl(slotIdx)`.
- **Phase 2 (deferred):** CPU PHY on Grace-Hopper; bulk→host over NVLink-C2C via DPDK; backend
  swap only (layout/rings/protocol unchanged).

## 10. Channel storage & CIR toolchain — ADR 0008 / Spec G

**Store geometric paths (rays), not antenna-CIR or per-SC H.** Offline OptiX tracer emits per
`(cell, grid-point)` the significant paths; slow plane expands rays→H on a UE move. Host-resident
table (~324MB Phase 1); GPU holds only active expanded `H_dl`. One ray set serves DL+UL
(reciprocity). Distinct from deferred Spec C (which decides hot-path H apply).

- **Grid:** uniform 2-D lattice at fixed `ueHeight` (3-D format-compatible). `gp=iy·nx+ix`.
  Default Δ=1m (FR1). Δ set by geometry decorrelation, not λ/2 (small-scale → Doppler rotor).
- **Per path:** complex gain g_p, excess delay τ_p, AoD (cell frame), AoA (UE frame), flags.
  Post-process: prune <−30dB, cap P_MAX=16, sort by power, bound τ ≤ maxExcessDelay (≤CP),
  optional normalize, mark noCoverage.
- **Expansion (slow plane, off hot path):**
  `H[c][u][rx][tx][sc] = Σ_{p<P} g_p·a_tx(AoD_p)[tx]·a_rx(AoA_p)[rx]·exp(−j2π f_sc τ_p)` (direct
  DFT, exact for non-aligned τ). ~13.4M cMAC/link; full 64-link refresh ~0.9GcMAC, sub-ms. Write
  back buffer → atomic pointer swap (indirection cell). Partial update = delta-to-both.
- **Doppler:** `doppler[c][u]` from paths[0] AoA + velocity: `f_d=(f_c/c)(v·û_AoA)`,
  `Δφ=2π f_d T_sym`. Phase 1 = single dominant-path rotor.
- **On-disk:** binary LE, mmap, fixed-stride link blocks. Header (magic `"ORCACIR1"`, version,
  numCells, gridDims, nx/ny/nz, origin, spacing, ueHeight, carrierHz, pMax, pathRecBytes=40,
  linkBlockBytes, norm/prune/maxDelay provenance) → cell descriptors `{pos[3],boresight[3],
  numTx,arrayType,elemSpacing}` → UE-array desc `{numRx,numUeTx,ueTxToRx[4],arrayType,
  elemSpacing}` → link blocks `{u16 numPaths; u16 flags; f32 pathlossDb; PathRecord paths[pMax]}`.
  PathRecord(40B): `tau f32; gainRe/Im f32×2; aodAz/El f32×2; aoaAz/El f32×2; flags u8; pad×3`.
  Index = `(cell·(nx·ny·nz)+gp)·linkBlockBytes`. `pathlossDb` feeds interferer ranking.
- **Loader:** host mmap, validate vs ORCA config (hard stop on mismatch). Nearest grid point
  (interpolation deferred). Bit-reproducible H → golden-model anchor.

## 11. Cross-link interference model (ADR 0002)

`H[cell][ue][rx][tx][sc]` for every modeled (cell,ue). Each receive stream sums a configurable
contributor set (serving + interferers): default neighbor-limited top-K, optional all-to-all
(Phase 1). Noise once per receive stream. Association (serving cell + interferers) is slow-plane
state; handover republished, never recomputed on hot path. **Interference set is data, not
topology** — graph sized for configured max K.

Bandwidth roofline (why layout matters): per-SC H is AI≈2 FLOP/byte (batched GEMV), ~150× below
H100 ridge → memory-bound. 7-cell per-SC ≈42 TB/s vs 3.35 TB/s HBM. SU 2-cell per-SC FP16 =
`C²·numRx·numTx·numSc·4B = 13.4MB/symbol ≈ 0.38 TB/s` (~11% HBM) → fits, Spec C not needed.

## 12. Module structure (to rebuild)

`common/` (types, layouts, config, telemetry) · `fh/` (Spec B wire, eAxC, DOCA/loopback) ·
`orchestr/` (symbol ring, coverage, deadline, T_air) · `dsp/` (precode/channel/combine kernels)
· `channel/` (CIR table gen + lookup, ray→H expansion, indirection cell) · `estim/` (SRS/cuSOLVER
— **dormant Phase 1**) · `scenario/` (cells, UE grid+mobility, association, beam codebook) ·
`oru/` (`OruTransport` inside ORCA; ORU process is a separate program) · `vue/` (`VueTransport`)
· `app/` (wiring, graph capture/launch, egress) · `tests/` (golden-model, loopback, jitter).
**Carry the `cell` dimension from Stage 1.**

## 13. Milestones

1 Loopback/identity (ORU process, oru/, vue/, orchestr/, app/) · 2 DL precode (beam codebook)
+AWGN vs golden · 3 static multipath + Doppler · 4 ray tracer/CIR table + loader + ray→H
(Spec G) · 5 UL combine (codebook) · 6 scale-out many vUE, deadline scheduler, soak · 7
multi-cell + interference (2 cells, all-to-all, SU) · 8 dynamic grid mobility + handover · 9
(deferred) MU-MIMO (needs Spec C).

## 14. Deferred (with re-enable cost)

| # | Goal | Enabling work | Compromise |
|---|---|---|---|
|1|MU-MIMO|Spec C|16× H read; sub-PRB selectivity or precision or scale|
|2|Spec C (H granularity)|—|per-PRB-group loses selectivity / tap-domain adds compute|
|3|More cells (>2)|Spec C / replication|interference pruning or more GPUs|
|4|µ≥2 / FR2|persistent kernels, v2 reassembly|tighter timing, GPU-controlled DOCA|
|5|Multi-box + inter-box|cell partitioning, state exchange|partition so interferers co-reside|
|6|vUE Phase 2 (CPU PHY/GH200)|NVLink-C2C, DPDK-over-NVLink|coupled HW+PHY migration|
|7|GPU-computed precoding (SRS)|SRS est + cuSOLVER|codebook-only until then|
|8|v2 streaming reassembly|scheduler change|none on wire|
|9|Continuous mobility|live RT / fine interp|grid quantization now|
|11|DOCA/GPUDirect|DOCA GPUNetIO backend|host-staged H2D copy until then|

## 15. ADR index

0001 hot-path sync (CUDA Graph + indirection cell; no persistent kernels at µ=1) · 0002
multi-cell/interference/mobility + bandwidth roofline · 0003 throughput/latency two-clocks,
vUE in-box · 0004 vUE IPC (CUDA IPC HBM + DPDK shm now; NVLink later) · 0005 SU-MIMO Phase 1 ·
0006 beam-indexed precoding, SRS deferred · 0007 3-process topology, DOCA deferred, host-staged
north · 0008 geometric-path channel storage. (Next ADR = 0009.)
