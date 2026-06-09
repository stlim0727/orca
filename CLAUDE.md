# oru-vue-emulator — project memory

Real-time GPU 5G channel emulator between an **ORU module** (custom fronthaul to a real
vDU) and a **vUE module** (many emulated UEs): per-OFDM-symbol precode → channel-apply →
combine, with a ray-traced channel, multi-cell interference, and grid-wise UE mobility.

## Current state: design-first, NO code yet

The source tree was intentionally removed; the design is the deliverable. **The docs are
the source of truth** — read them before proposing anything:

- @README.md — entry point, key design points, intended build.
- @docs/architecture.md — settled requirements, topology, spatial dimensions, multi-cell.
- @docs/specs/timing-and-deadlines.md — Spec A: per-symbol timing & deadline budget.
- @docs/specs/fronthaul-packet-format.md — Spec B: custom wire format, eAxC, multi-cell addressing.
- @docs/decisions/0001-hot-path-synchronization.md — ADR 0001: CUDA Graph + CPU-controlled DOCA + indirection cell.
- @docs/decisions/0002-multi-cell-interference-mobility.md — ADR 0002: multi-cell, interference, mobility.
- @docs/decisions/0003-throughput-latency-pipeline.md — ADR 0003: throughput/latency decoupling, symbol pipeline, vUE in-box.
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
  fronthaul; vUE side is an in-HBM handoff. Host-CPU vUE is rejected (per-antenna IQ would
  hit the PCIe wall). NIC budget = vDU side only (layer-domain IQ).
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

2. **Offline CIR-table toolchain (proposed Spec C)** — grid resolution, interpolation
   method, storage format, and how the OptiX tracer populates per-(cell, grid-point) CIR.
   Named in ADR 0002 §5 but unspecified.

3. **UL combiner detail** — `C` is `numLayers × numTx` (16×64) per PRB-group, conjugate-
   transpose shape of `W`; MMSE vs MRC selection and SRS→C path (Stage 5).

4. **Deferred at implementation time:** BFP exponent/scaling for bit-exactness (Spec B.4),
   telemetry wire format (Spec B.7), YAML config loader.

5. **`L_max` from real fronthaul tolerance** (ADR 0003) — working placeholder ~70 µs;
   must be measured/negotiated; caps pipeline depth. Plus the vUE-side interface form
   (in-memory handoff vs uniform packet API) — minor.

## Working agreement

- It's early — favor design/decisions over code. New significant decisions get an ADR in
  `docs/decisions/` (sequential numbering; ADR 0004 is next).
- Keep tensor layouts carrying the `cell` dimension from Stage 1 so multi-cell is an
  extension, not a refactor (MILESTONES note).
- Intended layout when code resumes: `common/ fh/ orchestr/ dsp/ channel/ estim/
  scenario/ oru/ vue/ app/ tests/` (see architecture.md § module structure).
