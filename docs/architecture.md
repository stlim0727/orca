# Architecture overview

**Status:** Settled design (source of truth). No implementation yet — code was
removed after we locked the design (see commit history / ADR 0001).

**ORCA** (*O-RU Channel Applier*) is a real-time GPU application that
**stands in for the O-RU** between a real 5G NR **vDU** and a set of **emulated UEs
(vUE)**. It terminates the vDU's fronthaul as the O-RU, transforms frequency-domain IQ,
applies a ray-traced channel in real time, and performs precoding / receive-combining **on
behalf of the radio unit**.

**Three in-box processes** ([ADR 0007](decisions/0007-process-topology-doca-deferral.md)):
`ORU process ↔ ORCA ↔ vUE`. ORCA holds the GPU; it never touches Ethernet.

- **North — vDU side:** a **separate ORU process** terminates the **ORU fronthaul packet
  format** (Spec B) over Ethernet (**DOCA deferred** — NIC via kernel/DPDK) and relays
  de-framed **layer IQ** to ORCA over **host shared memory + H2D copy**
  ([Spec F](specs/oru-interface-contract.md)). ORCA covers the **RU role** (precoding/
  combining on behalf of the RU). The vDU-side volume is small (~3 GB/s SU 2-cell), so the
  host-staged copy (~1 µs/sym) suffices — hence DOCA can be deferred.
- **South — vUE side:** ORCA ↔ in-box **vUE** over **DPDK shared memory** (control +
  handles); bulk **per-antenna** IQ (~160 GB/s) stays in HBM, shared via **CUDA IPC**
  ([ADR 0004](decisions/0004-vue-interface-ipc.md) / [Spec D](specs/vue-interface-contract.md)).

## Requirements (locked)

| Dimension | Decision |
|---|---|
| Fronthaul | **Custom packet format** (real vDU on the network side, custom framing). See [Spec B](specs/fronthaul-packet-format.md). |
| GPU / ingress | **NVIDIA datacenter (e.g. H100).** Phase 1: **host-staged** north ingress — separate **ORU process** (NIC via kernel/DPDK) → host shm → **H2D** (Spec F). **DOCA GPUNetIO + GPUDirect deferred** ([ADR 0007](decisions/0007-process-topology-doca-deferral.md), [deferred-goals](deferred-goals.md#doca)). |
| Scale | **Massive MIMO, 32–64+ TRX** (default 64T). |
| Spatial multiplexing | **Phase 1 = SU-MIMO** (one UE per time-frequency resource; OFDMA scheduling; per-resource layers = UE rank ≤4). **MU-MIMO deferred** (16× `H`-read blow-up → needs Spec C). See [ADR 0005](decisions/0005-su-mimo-phase1.md), [deferred-goals](deferred-goals.md#mu-mimo). |
| Channel | **Ray-traced**, with **slow CIR update + per-symbol apply** (heavy ray tracing decoupled from the hot path). |
| vUE | **Many UEs**, full per-symbol budget. |
| Precoding | **Both** vDU-supplied (C-plane) **and** GPU-computed (ZF/MMSE/SVD from SRS). |
| Cadence | **Per OFDM symbol**, **µ=1** (35.7 µs). See [Spec A](specs/timing-and-deadlines.md). |
| Hot-path sync | **CUDA Graph + CPU-controlled DOCA + indirection-cell double buffering.** See [ADR 0001](decisions/0001-hot-path-synchronization.md). |
| Budget model | **Two clocks** — *throughput* (one symbol/`T_sym`=35.7 µs; bottleneck stage ≤ `T_sym`) and *latency* (deliver within `L_max` ≈ 70 µs working, < real fronthaul tolerance). `T_proc≤3µs` retired. See [ADR 0003](decisions/0003-throughput-latency-pipeline.md). |
| vUE location | **In-box, GPU-resident.** Only the vDU side crosses the NIC fronthaul; vUE side is an in-HBM handoff. NIC budget = vDU side only. See [ADR 0003 §5](decisions/0003-throughput-latency-pipeline.md). |
| vUE IPC | **Separate process.** Phase 1: bulk IQ in HBM via **CUDA IPC**, **DPDK shm** control plane (GPU PHY, PCIe H100). Phase 2: CPU PHY on **Grace-Hopper**, bulk to host over NVLink-C2C. See [ADR 0004](decisions/0004-vue-interface-ipc.md) + [Spec D](specs/vue-interface-contract.md) (full contract). |
| Scaling | **Cells-per-box** bounded by the throughput clock (cost lever, minimize boxes); more boxes = replication (later phase); inter-box comms only if interfering cells split across boxes. See [ADR 0003 §6](decisions/0003-throughput-latency-pipeline.md). |
| Cells | **Multiple cells, single GPU box.** One emulator instance hosts all cells (one fronthaul flow set per vDU); shared PTP/S-plane time. See [ADR 0002](decisions/0002-multi-cell-interference-mobility.md). |
| Interference | **Inter-cell, multi-UE.** Cross-link channel per `(cell, ue)`; each UE = serving cell + interferers. **Configurable**: neighbor-limited top-K (default) → all-to-all. See [ADR 0002](decisions/0002-multi-cell-interference-mobility.md). |
| Mobility | **Dynamic, grid-wise.** UE positions on a discrete grid; **precomputed per-`(cell, grid-point)` CIR table**, slow-plane lookup on move; per-symbol Doppler per link. Handover supported. See [ADR 0002](decisions/0002-multi-cell-interference-mobility.md). |

## Topology & data flow

```
            Ethernet (Spec B)        host shm + DPDK         ORCA (GPU)        DPDK + CUDA IPC
  ┌──────────┐              ┌─────────────┐  H2D/D2H  ┌──────────────────┐   (HBM bulk)  ┌──────────┐
  │ Real vDU │◄────────────►│ ORU process │◄─(Spec F)►│ precode → channel │◄──(Spec D)──►│   vUE    │
  │ (3rd pty)│  DL/UL IQ    │ NIC+framing │           │ → combine         │              │ (N UEs)  │
  └──────────┘              └─────────────┘           │ ⇒ O-RU/RU role    │              └──────────┘
                          (DOCA deferred)             └──────────────────┘
```

- **DL:** vDU(s) → per-cell precode (per-PRB-group W) → channel-apply summed over each
  UE's serving cell **+ interferers** (cross-link H) → +noise → vUE
- **UL:** vUE → channel-apply → per-cell RU receives the sum over its served **+
  interfering** UEs → combine (MMSE/MRC) → vDU(s)

The two directions are spatially symmetric around each RU's **64 TRX** antennas: DL
precodes **16 layers → 64 Tx**; UL combines **64 Rx → 16 layers**. With multiple cells,
each receive stream is a **sum over a configurable contributor set** (serving cell +
interferers). See [§ Spatial dimensions](#spatial-dimensions) and
[§ Multi-cell, interference & mobility](#multi-cell-interference--mobility).
- **Slow plane:** ray tracer updates per-UE CIR every few ms; SRS estimation updates
  W per PRB-group.
- **Hot path:** per-symbol apply within the symbol deadline (Spec A).

## Key insight: timing is the constraint; interference scale is what threatens compute

For a **single cell** (or low interference), 64T64R, 100 MHz precode + channel-apply is
only a few hundred MFLOP per slot (sub-µs of math per symbol on an H100), so the
engineering challenge is **deterministic, low-jitter packet movement and per-symbol
deadline scheduling**, not FLOPS.

**With multi-cell interference this changes — but the wall is memory bandwidth, not
FLOPs.** The per-symbol channel-apply sums a cross-link `H` over each receive stream's
contributor set, scaling as `Σ_u |contributors(u)| · numTx · numRxPort · numSc`. Raw
compute (~168 TFLOP/s both directions at 7 cells, neighbor-limited) is comfortable for
H100 tensor cores — but `H` is read once per symbol with ~no reuse, so the operation is
a batched **GEMV** with **arithmetic intensity ≈ 2 FLOP/byte**, ~150× below the H100
roofline ridge. Materialized per-subcarrier, `H` is ≈ 1.5 GB/symbol → **~42 TB/s
needed vs ~3.35 TB/s HBM** (≈12× over, even neighbor-limited). The fix is to cut `H`'s
footprint — store it **per PRB-group** (≈31 MB, fits L2) or apply it **from CIR taps**
(high reuse) — *then* it becomes compute-bound and tensor cores help. Timing is still
the deadline; reducing `H` bandwidth is the primary lever. Full roofline and the
per-PRB-group vs tap-domain decision: [ADR 0002 §6](decisions/0002-multi-cell-interference-mobility.md).

## Slow / fast channel split

Two decoupled rates communicating through a double-buffered coefficient store
(published via an **indirection cell**, ADR 0001 §4):

- **Slow plane (every few ms / on UE movement):** a **precomputed per-`(cell,
  grid-point)` CIR table** (built offline by the OptiX ray tracer, **[Spec G](specs/cir-table-toolchain.md)**)
  is looked up — nearest grid point, interpolation deferred — for each UE's current grid
  position → geometric paths → ray→`H` expansion (array steering + delay DFT) →
  per-subcarrier cross-link frequency response `H[cell][ue][rx][tx][sc]`. Only the
  links affected by a move are refilled in the back buffer, then published. UE moves
  may also update **serving-cell / interferer association** (handover).
- **Fast plane (every symbol):** the apply kernel reads the active cross-link `H` and a
  per-symbol **Doppler phase rotor per `(cell, ue)` link** (one step per symbol index),
  parametrized by both endpoints' locations, so the channel evolves per-symbol within
  the slow window without re-tracing.

Mobility is **grid-quantized and deterministic**: no live ray tracing on the runtime
path; the ray tracer is the offline table generator. See
[ADR 0002 §5](decisions/0002-multi-cell-interference-mobility.md).

## Precoding / combining

**Phase 1 — beam-indexed codebook ([ADR 0006](decisions/0006-beam-indexed-precoding.md)).**
A codebook of precoding/combining vectors **indexed by `beam_id`** is **resident in GPU
memory in advance**. The vDU supplies `beam_id` per resource at runtime via the C-plane;
the hot path **gathers** the vector(s) by `beam_id` to form `W` (`64 × rank`) and applies
`y = W·x` (DL) / combines `64 → rank` (UL). No matrices on the wire, no weight computation.

| | DL precode | UL combine |
|---|---|---|
| **beam_id codebook** (Phase 1) | gather `precodeBook[beam_id]` → `W` (`64×rank`) → `y = Wx` | gather `combineBook[beam_id]` → `64 → rank` |
| **GPU-computed (SRS)** — *deferred* | estimate H from SRS → ZF/MMSE/SVD (cuSOLVER) | MMSE / MRC |

SRS-based, channel-matched precoding is **deferred** (→
[deferred-goals](deferred-goals.md#gpu-precoding)); Phase 1 needs no cuSOLVER and no
slow-plane weight estimation. An explicit-`W` C-plane variant remains possible but is not
the Phase-1 path.

## Spatial dimensions

The RU has **64 TRX** antennas (default; 32–64+ supported). The same physical
antenna count is the Tx dimension on DL and the Rx dimension on UL, so the two
directions are mirror images:

| | DL (precode) | UL (combine) |
|---|---|---|
| Input | layer-domain IQ `x[sc][layer]`, **16 layers** | RU-received IQ `r[sc][rx]`, **64 Rx** |
| Operator | `W[group][tx][layer]`, **64 × 16** per PRB-group | `C[group][layer][rx]`, **16 × 64** per PRB-group |
| Output | radiated `y[sc][tx]`, **64 Tx** | `z[sc][layer]`, **16 layers** → vDU |
| Direction | 16 layers → 64 antennas | 64 antennas → 16 layers |

So **UL combining reduces the 64 RU-Rx antenna streams to 16 layers** ("precombining
on behalf of the RU"), exactly inverting the DL precode. The combiner weight matrix
`C` is `numLayers × numTx` per PRB-group (the conjugate-transpose shape of `W`),
whether vDU-supplied or GPU-computed from SRS.

Defaults (`µ=1`, 100 MHz): `numTx = 64`, `numLayers = 16`, `numPrb = 273`,
`prbGroupSize = 4`. **Phase 1 (SU-MIMO, [ADR 0005](decisions/0005-su-mimo-phase1.md)):**
each time-frequency resource carries **one UE** with `layers = UE rank ≤ 4`, so `W` per SC
is `64 × rank`; `numLayers = 16` is the **MU aggregate** and applies only when MU-MIMO is
restored (deferred). The combine stage operates in **RU-antenna space** (`64 → rank`), not
per-UE. With multiple cells the channel tensor gains a **cell** dimension and becomes a
**cross-link** between cell TRX and UE rx-port — `H[cell][ue][rx][tx][sc]` — covering
each UE's serving cell and its interferers (next section).

## Multi-cell, interference & mobility

See [ADR 0002](decisions/0002-multi-cell-interference-mobility.md) for the full
decision and rationale. Summary:

- **One GPU box, all cells.** A single emulator instance holds every cell and the full
  cross-channel tensor in one GPU's memory; it terminates one fronthaul flow set per
  vDU. All cells share the PTP/S-plane clock, so one per-symbol deadline (Spec A)
  covers them all.
- **Cross-link channel + interference.** `H[cell][ue][rx][tx][sc]` is defined for every
  modeled `(cell, ue)` pair. Each receive stream sums over a **contributor set**:
  - DL at UE `u`: `r_u = Σ_{c} H[c][u]·y_c · doppler[c][u] + noise`, over `u`'s serving
    cell + interferers.
  - UL at cell `c` (64 antennas): `r_c = Σ_{u} H[u][c]·x_u · doppler[u][c] + noise`,
    then combine `64 → 16`.

  The desired signal is the serving-cell term; all others are interference. Noise is
  added once per receive stream.
- **Configurable interference set.** The contributor list is runtime config: default
  **neighbor-limited top-K** (bounds the per-symbol GEMM), optionally **all-to-all**
  for small scenarios. Kernels and graph topology are unchanged by the choice — only
  the link list and the graph's batch extent.
- **Serving cell + handover.** Each UE has one serving cell and an interferer list; both
  may change as it moves across the grid. Association is slow-plane state, never
  recomputed on the hot path.
- **Grid mobility.** UE positions are grid points; an offline ray-traced
  per-`(cell, grid-point)` CIR table is looked up (+interpolated) on a move, refilling
  only affected links and republishing via the indirection cell. Per-symbol Doppler is
  per `(cell, ue)` link.
- **Bandwidth caveat.** Channel-apply is **memory-bandwidth bound** (AI ≈ 2 FLOP/byte),
  not FLOP-bound: streaming per-subcarrier `H` needs ~42 TB/s vs ~3.35 TB/s HBM. The
  resolution is per-PRB-group `H` (fits L2) or tap-domain application, plus
  neighbor-limiting; only then is it a tensor-core GEMM. See ADR 0002 §6.

## Intended module structure (to rebuild from clean)

| Dir | Responsibility |
|---|---|
| `common/` | complex types, tensor layouts, config, telemetry |
| `fh/` | custom fronthaul wire format, eAxC, DOCA GPUNetIO rx/tx + loopback |
| `orchestr/` | symbol ring, coverage, deadline scheduler, `T_air` timing |
| `dsp/` | precode / channel-apply (cross-link, batched GEMV→GEMM per Spec C / ADR 0002 §6) / combine kernels (CUDA graph nodes) |
| `channel/` | offline CIR-table generator (ray tracer) + grid lookup, ray→`H` expansion, indirection-cell double buffer — **Spec G** (table/format/expansion) |
| `estim/` | SRS channel estimation + weight computation (cuSOLVER) — **deferred (ADR 0006)**, dormant in Phase 1 |
| `scenario/` | cells, UE grid + mobility, serving-cell/interferer association, contribution lists, **resident beam codebook** (ADR 0006) |
| `oru/` | `OruTransport` (host-shm + H2D/D2H, Spec F) inside ORCA + handshake. The **ORU process** (NIC + Spec B framing) is a *separate program* (ADR 0007), not part of ORCA. |
| `vue/` | UE-facing endpoint + `VueTransport` (Phase-1 CUDA-IPC / Phase-2 DPDK-over-NVLink backends), ADR 0004 |
| `app/` | wiring, graph capture/launch, per-symbol event-gated egress |
| `tests/` | golden-model bit-exactness, loopback, jitter harness |

## Open risks

1. **vDU fronthaul timing tolerance** — even with a custom format, a tight U-plane
   window inherits a hard per-symbol deadline.
2. **Clock/PTP sync** between the vDU(s) and the GPU box (S-plane) — now across
   **multiple** time-aligned vDUs.
3. **Cross-channel memory footprint** — `H` grows as `O(C · U_total · 64 · R · Sc)`;
   storing only active links and using FP16/BF16 H are the mitigations (ADR 0002 §6).
4. **Compute-bound at high interference** — all-to-all on FR1 may exceed `T_proc` on
   one GPU; neighbor-limiting is the primary control, per-box-per-cell the escape hatch.
5. **Handover / batch-extent consistency** — dynamic interferer changes must stay
   within the captured graph's batch bound (size for configured `K`, not instantaneous
   count).

## Milestones

See [MILESTONES.md](MILESTONES.md).
