# ORCA (*O-RU Channel Applier*) — project memory

**ORCA** is a real-time GPU application that stands in for the O-RU between a real
5G NR vDU and emulated UEs (vUE): per-OFDM-symbol precode → channel-apply → combine,
with a ray-traced channel, multi-cell interference, and grid-wise UE mobility. The
system covers the RU role, including fronthaul-facing termination in the ORU sidecar and
precoding/combining on behalf of the RU.

## Current state: design-first, no code yet

The source tree was intentionally removed; the design is the deliverable. The docs are the
source of truth. Read the project entry points before proposing implementation work:

- `README.md` — entry point, key design points, intended build.
- `docs/architecture.md` — settled requirements, topology, spatial dimensions, multi-cell.
- `docs/specs/timing-and-deadlines.md` — Spec A: per-symbol timing and deadline budget.
- `docs/specs/fronthaul-packet-format.md` — Spec B: ORU fronthaul wire format, eAxC,
  multi-cell addressing.
- `docs/specs/vue-interface-contract.md` — Spec D: in-box vUE interface: CUDA IPC bulk,
  DPDK shared-memory control, handshake, per-symbol protocol.
- `docs/specs/gpu-kernel-design.md` — Spec E: GPU kernels and memory: layouts,
  allocation, K0–K5 maps, coalescing, occupancy, `H`/`P`/`x`/`y` lifecycle, L2 cache
  behavior, dynamic-`H` update.
- `docs/specs/oru-interface-contract.md` — Spec F: ORU↔ORCA north interface: host shared
  memory bulk plus H2D/D2H, DPDK control, allocation and beam map.
- `docs/specs/cir-table-toolchain.md` — Spec G: offline OptiX→CIR-table toolchain,
  on-disk format, and slow-plane ray→`H` expansion feeding Spec E `H_dl`.
- `docs/decisions/0001-hot-path-synchronization.md` — ADR 0001: CUDA Graph,
  CPU-controlled DOCA, indirection-cell double buffering.
- `docs/decisions/0002-multi-cell-interference-mobility.md` — ADR 0002: multi-cell,
  interference, mobility, `H`-bandwidth roofline.
- `docs/decisions/0003-throughput-latency-pipeline.md` — ADR 0003: throughput/latency
  decoupling, symbol pipeline, vUE in-box.
- `docs/decisions/0004-vue-interface-ipc.md` — ADR 0004: vUE IPC: CUDA IPC (HBM) and
  DPDK shared-memory control now; host/NVLink later.
- `docs/decisions/0005-su-mimo-phase1.md` — ADR 0005: SU-MIMO for Phase 1; MU-MIMO
  deferred.
- `docs/decisions/0006-beam-indexed-precoding.md` — ADR 0006: `beam_id` codebook
  precoding; SRS deferred.
- `docs/decisions/0007-process-topology-doca-deferral.md` — ADR 0007: three-process
  topology (ORU/ORCA/vUE), DOCA deferred, host-staged north ingress.
- `docs/decisions/0008-geometric-path-channel-storage.md` — ADR 0008: store geometric
  paths (rays), not antenna-CIR or per-subcarrier `H`; slow-plane ray→`H` expansion;
  host-resident table.
- `docs/deferred-goals.md` — deferred goals and the compromises/enabling work to re-enable
  each.
- `docs/MILESTONES.md` — stage plan and hot-path invariants.

## Locked decisions

Do not relitigate these without a new ADR:

- **Cadence:** per OFDM symbol, µ=1 (`T_sym = 35.7 µs`). µ≥2 is out of scope.
- **Throughput vs latency:** throughput is one symbol per `T_sym` with each bottleneck
  stage ≤ `T_sym`; latency is delivery within `L_max` (working placeholder ~70 µs, less
  than real fronthaul tolerance TBD). `T_proc ≤ 3 µs` is retired.
- **Three in-box processes:** `ORU process ↔ ORCA ↔ vUE`. The ORU process owns NIC plus
  Spec B framing and relays layer IQ to ORCA over host shared memory plus H2D/D2H (Spec F).
  ORCA never sees Ethernet.
- **DOCA deferred:** north bulk is host shared memory plus H2D/D2H for Phase 1, justified
  by small vDU-side volume (~3 GB/s for SU 2-cell). Keep this behind `OruTransport` so
  DOCA GPUNetIO / GPUDirect can replace the copy later.
- **vUE is in-box and GPU-resident:** Phase 1 uses GPU PHY on PCIe H100, CUDA IPC for HBM
  bulk buffers and IPC events, and DPDK shared memory for control only. Keep this behind
  `VueTransport`; buffer layout should remain stable for future Grace-Hopper/NVLink-C2C
  migration.
- **Hot-path sync:** captured CUDA Graph launch per symbol, CPU-controlled orchestration,
  and indirection-cell double buffering. No persistent kernels, doorbells, or system-scope
  flags at µ=1.
- **Spatial:** 64 TRX. DL precodes `rank → 64 Tx`; UL combines `64 Rx → rank`. Phase-1 UE
  rank is ≤4.
- **Phase-1 target:** SU-MIMO, 2 cells, all-to-all inter-cell interference, grid mobility,
  all channel coefficients resident as per-subcarrier FP16 `H`, µ=1, vUE in-box, and a
  single PCIe H100. The per-symbol `H` read is ~0.38 TB/s (~11% of HBM), so Spec C channel
  compression is deferred.
- **Spatial multiplexing:** Phase 1 is SU-MIMO only: one UE per time-frequency resource,
  OFDMA scheduling separates UEs, and the vDU provides a per-symbol scheduling/allocation
  map. MU-MIMO is deferred because it reintroduces a 16× `H`-read blow-up.
- **Precoding:** use a beam-indexed codebook resident in GPU memory. The vDU supplies
  `beam_id` per resource at runtime. SRS-derived weights and GPU-computed ZF/MMSE/SVD are
  deferred; `estim/` remains dormant in Phase 1.
- **Mobility/channel storage:** UE positions are on a discrete grid. The offline table
  stores host-resident geometric paths per `(cell, grid-point)`, not antenna-CIR or
  per-subcarrier `H`; the slow plane expands rays→`H` into GPU-resident `H_dl` on moves.
- **Slow plane isolation:** slow-plane updates never touch the hot path directly; publish
  through the atomic indirection cell.
- **Late symbols:** never stall on a late symbol; zero-fill missing PRBs and advance.

## Open threads

- **Channel-apply roofline:** channel apply is memory-bandwidth bound, not compute-bound.
  Phase 1 fits because SU-MIMO plus per-subcarrier FP16 `H` is about 11% HBM. When MU-MIMO,
  more cells, or larger interferer sets return, Spec C must decide per-subcarrier vs
  per-PRB-group vs tap-domain application, `H` precision, PRB-group size, and tap count.
- **Offline CIR-table toolchain:** Spec G is Phase-1 work and generates the offline ray
  table plus slow-plane ray→`H` expansion. It is distinct from deferred Spec C, which
  decides how `H` is applied on the hot path.
- **UL combiner:** resolved by ADR 0006 as codebook gather (`64 → rank`), symmetric with
  DL precoding. SRS-derived MMSE/MRC weights are deferred.
- **Implementation-time deferrals:** BFP exponent/scaling for bit-exactness, telemetry wire
  format, and YAML config loader.
- **`L_max`:** working placeholder is ~70 µs; final value must be measured or negotiated
  from real fronthaul tolerance and caps pipeline depth.

## Working agreement

- Favor design and decisions over code until implementation resumes.
- Add a new ADR for significant decisions; ADR 0009 is next.
- Update `docs/deferred-goals.md` whenever a capability is pushed to a later phase.
- Keep tensor layouts carrying the `cell` dimension from Stage 1 so multi-cell remains an
  extension rather than a refactor.
- Intended layout when code resumes: `common/ fh/ orchestr/ dsp/ channel/ estim/ scenario/
  oru/ vue/ app/ tests/`.
