# ORCA (*O-RU Channel Applier*) — project memory

**ORCA** is a real-time GPU app that **stands in for the O-RU** between a real
5G NR **vDU** and **emulated UEs (vUE)**: per-OFDM-symbol precode → channel-apply →
combine, with a ray-traced channel, multi-cell interference, and grid-wise UE mobility.
It covers the **RU role** (fronthaul termination + precoding/combining on behalf of the RU).

**Three in-box processes (ADR 0007):** `ORU process ↔ ORCA ↔ vUE`. *North (vDU)* = a
**separate ORU process** owns the NIC + **ORU fronthaul packet format** (Spec B) over
Ethernet (**DOCA deferred**) → relays layer IQ to ORCA via **host shm + H2D** (Spec F).
*South (vUE, in-box)* = **DPDK shared memory** control; bulk per-antenna IQ in HBM via
CUDA IPC (ADR 0004 / Spec D). ORCA = the O-RU/RU role; it never sees Ethernet.

## Current state: design-first, NO code yet

The source tree was intentionally removed; the design is the deliverable. **The docs are
the source of truth** — read them before proposing anything:

- @README.md — entry point, key design points, intended build.
- @docs/architecture.md — settled requirements, topology, spatial dimensions, multi-cell.
- @docs/specs/timing-and-deadlines.md — Spec A: per-symbol timing & deadline budget.
- @docs/specs/fronthaul-packet-format.md — Spec B: ORU fronthaul wire format (north/vDU), eAxC, multi-cell addressing.
- @docs/specs/vue-interface-contract.md — Spec D: in-box vUE interface (south) — CUDA IPC bulk + DPDK shm control, handshake, per-symbol protocol.
- @docs/specs/gpu-kernel-design.md — Spec E: GPU kernels & memory — tensor layouts, allocation, K0–K5 grid/block/thread maps, coalescing, occupancy.
- @docs/decisions/0001-hot-path-synchronization.md — ADR 0001: CUDA Graph + CPU-controlled DOCA + indirection cell.
- @docs/decisions/0002-multi-cell-interference-mobility.md — ADR 0002: multi-cell, interference, mobility.
- @docs/decisions/0003-throughput-latency-pipeline.md — ADR 0003: throughput/latency decoupling, symbol pipeline, vUE in-box.
- @docs/decisions/0004-vue-interface-ipc.md — ADR 0004: vUE IPC — CUDA IPC (HBM) + DPDK shm control now; host/NVLink later.
- @docs/decisions/0005-su-mimo-phase1.md — ADR 0005: SU-MIMO for Phase 1; MU-MIMO deferred.
- @docs/decisions/0006-beam-indexed-precoding.md — ADR 0006: beam_id codebook precoding; SRS deferred.
- @docs/decisions/0007-process-topology-doca-deferral.md — ADR 0007: 3-process topology (ORU/ORCA/vUE); DOCA deferred; host-staged north.
- @docs/specs/oru-interface-contract.md — Spec F: ORU↔ORCA interface (north) — host shm bulk + H2D/D2H, DPDK control, alloc/beam map.
- @docs/deferred-goals.md — register of deferred goals + the compromises to re-enable each.
- @docs/MILESTONES.md — stage plan + hot-path invariants.

## Locked decisions (don't relitigate without an ADR)

- **Cadence:** per OFDM symbol, **µ=1** (35.7 µs). µ≥2 is out of scope.
- **Throughput vs latency (ADR 0003):** two clocks. *Throughput* = one symbol per
  `T_sym`=35.7 µs (bottleneck stage ≤ T_sym; this is where the ADR 0002 §6 bandwidth wall
  binds — pipelining does NOT relax it). *Latency* = deliver each symbol within `L_max`
  (working ~70 µs ≈ 2·T_sym, < real fronthaul tolerance TBD). **`T_proc ≤ 3 µs` is
  retired** — compute gets up to a full period for throughput, contributes ~6 µs to the
  latency sum.
- **vUE is in-box, GPU-resident (ADR 0003 §5):** only the vDU side crosses the NIC
  fronthaul; vUE side is an in-HBM handoff. NIC budget = vDU side only (layer-domain IQ).
- **Process topology + DOCA deferral (ADR 0007):** **three in-box processes** — ORU
  process (NIC + Spec B framing, kernel/DPDK, **DOCA deferred**), ORCA (GPU compute), vUE.
  North bulk = **host shm + H2D/D2H** (Spec F), justified by small vDU-side volume (~3 GB/s
  SU 2-cell). ORCA hot path (graph K0–K5) unchanged; only K0 source / K5 sink differ. Behind
  an `OruTransport` so DOCA/GPUDirect can swap in later (removes the copy). DOCA → deferred-goals #11.
- **vUE IPC (ADR 0004):** vUE is a **separate in-box process**. **Phase 1 (now):**
  GPU PHY on PCIe H100 — bulk per-antenna IQ stays in HBM, shared via **CUDA IPC**
  (+ IPC events); **DPDK shared memory** carries the control plane only. **Phase 2
  (future):** CPU PHY on **Grace-Hopper** — bulk moves to host over NVLink-C2C via DPDK
  shm. The two migrations are coupled. Behind a `VueTransport` interface; buffer layout
  stable across phases.
- **Scaling (ADR 0003 §6):** cells-per-box bounded by the throughput clock (cost lever —
  minimize boxes); more boxes = replication (later phase); inter-box comms only if
  interfering cells split across boxes — partition so each UE's interferer set stays on
  one box.
- **Hot-path sync:** CUDA Graph, CPU-controlled DOCA GPUNetIO, indirection-cell double
  buffering. **No** persistent kernels / doorbells / system-scope flags at µ=1 (ADR 0001).
- **Spatial:** 64 TRX. DL precodes 16 layers → 64 Tx; UL combines 64 Rx → 16 layers.
- **Multi-cell:** one GPU box hosts all cells; cross-link channel `H[cell][ue][rx][tx][sc]`;
  per-UE contributor set (serving + interferers) is configurable, neighbor-limited top-K
  by default → all-to-all (ADR 0002).
- **Spatial multiplexing (ADR 0005):** **Phase 1 = SU-MIMO only** (one UE per
  time-frequency resource; OFDMA scheduling separates UEs; per-resource layers = UE rank
  ≤4). **MU-MIMO is deferred to a very later phase** — it re-introduces a 16× `H`-read
  blow-up (→ needs Spec C). SU needs a **per-symbol scheduling/allocation map** from the
  vDU. All-to-all interference retained either way.
- **Phase-1 target:** SU-MIMO, **2 cells**, all-to-all interference, grid mobility, all
  `H` resident as **per-subcarrier FP16** (~0.38 TB/s, ~11% HBM → fits; Spec C deferred),
  µ=1, single PCIe H100.
- **Precoding (ADR 0006):** **beam-indexed codebook** resident in GPU memory; vDU supplies
  `beam_id` per resource at runtime (C-plane, alongside the SU scheduling map). Hot path
  gathers `precodeBook[beam_id]`→`W` (`64×rank`); UL combine symmetric. **SRS /
  GPU-computed ZF/MMSE/SVD deferred** (`estim/` dormant; no cuSOLVER in Phase 1).
- **Mobility:** UE positions on a discrete grid; offline ray-traced per-(cell, grid-point)
  CIR table; slow-plane lookup/interp on move; per-link per-symbol Doppler; handover.
- **Slow plane never touches the hot path**; publishes via atomic write to the indirection cell.
- **Never stall on a late symbol** — zero-fill missing PRBs and advance (Spec A §A.4).

## OPEN THREADS (carried over from the design discussion — not yet in the docs)

1. **Compute intensity / roofline — channel-apply is MEMORY-BOUND, not compute-bound.**
   ✅ Now recorded in **ADR 0002 §6** (corrected from the earlier "compute-bound GEMM"
   framing). Key results, for reference (7 cells × 16 UE = 112 UEs, T=64, R=4, Sc=3276,
   neighbor K=3, FP16 complex H, 8 FLOP/cMAC, 28,011 sym/s):
   - Raw compute ≈ 3.0 GFLOP/symbol DL → ~168 TFLOP/s both dirs — fine for H100 tensor.
   - H read once/symbol, ~no reuse → **AI ≈ 2 FLOP/byte** (batched GEMV); per-SC H ≈
     1.5 GB/symbol → **~42 TB/s vs 3.35 TB/s HBM** → bandwidth-bound, infeasible as-is.
   - Resolution: store H **per PRB-group** (~31 MB, fits L2) OR apply **from CIR taps**.
   - **Still open (→ Spec C):** channel coherence granularity (per-SC vs per-PRB-group vs
     tap-domain) and the variables that pin the layout — **H precision** (FP16/BF16/INT8),
     **PRB-group size**, **tap count P** — to be set against a target delay spread.
     **Deferred out of Phase 1 by ADR 0005** (SU-MIMO + per-SC FP16 fits at ~11% HBM);
     becomes critical when MU-MIMO / more cells return. See deferred-goals.md.

2. **Offline CIR-table toolchain (proposed Spec C)** — grid resolution, interpolation
   method, storage format, and how the OptiX tracer populates per-(cell, grid-point) CIR.
   Named in ADR 0002 §5 but unspecified.

3. **UL combiner** — ✅ resolved by **ADR 0006**: combine = `beam_id` codebook gather
   (`64 → rank`), symmetric with DL precode. SRS-derived MMSE/MRC weights are **deferred**
   (deferred-goals #7).

4. **Deferred at implementation time:** BFP exponent/scaling for bit-exactness (Spec B.4),
   telemetry wire format (Spec B.7), YAML config loader.

5. **`L_max` from real fronthaul tolerance** (ADR 0003) — working placeholder ~70 µs;
   must be measured/negotiated; caps pipeline depth.

## Working agreement

- It's early — favor design/decisions over code. New significant decisions get an ADR in
  `docs/decisions/` (sequential numbering; ADR 0008 is next). Deferred scope lives in
  @docs/deferred-goals.md — update it whenever something is pushed to a later phase.
- Keep tensor layouts carrying the `cell` dimension from Stage 1 so multi-cell is an
  extension, not a refactor (MILESTONES note).
- Intended layout when code resumes: `common/ fh/ orchestr/ dsp/ channel/ estim/
  scenario/ oru/ vue/ app/ tests/` (see architecture.md § module structure).
