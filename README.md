# oru-vue-emulator

Real-time GPU channel emulator sitting between an **ORU module** (facing a real 5G NR
vDU over a custom fronthaul) and a **vUE module** (many emulated 5G UEs). It transforms
frequency-domain IQ between the two, applies a ray-traced channel in real time, and
performs precoding / receive-combining on behalf of the radio unit.

```
                custom fronthaul              GPU box              custom
  ┌──────────┐  (custom framing,    ┌──────────────────────┐     ┌──────────┐
  │ Real vDU │◄─ vDU symbol cadence)►│   CUDA emulator      │◄───►│   vUE    │
  │ (3rd pty)│   DL: PDSCH IQ        │  precode → channel    │ IQ  │ (N UEs)  │
  └──────────┘   UL: PUSCH/SRS IQ    │  → combine            │     └──────────┘
                                     └──────────────────────┘
```

- **DL:** vDU(s) → precode (16 layers → 64 Tx) → channel-apply summed over each UE's
  serving cell + interferers (cross-link H) → +noise → vUE
- **UL:** vUE → channel-apply → per-cell RU sums served + interfering UEs → combine
  (64 Rx → 16 layers) → vDU(s)
- **Multi-cell:** one GPU box hosts all cells with inter-cell interference; the
  contributor set per UE is configurable (neighbor-limited → all-to-all).
- **Slow plane:** grid-wise UE mobility via a precomputed per-`(cell, grid-point)` CIR
  table (lookup on move); SRS estimation updates precoding/combining weights per
  PRB-group.
- **Hot path:** per-OFDM-symbol apply within the symbol deadline (µ=1, 35.7 µs), all
  cells in one captured-graph launch.

## Status

**Design-first. No implementation yet** — the design is settled and the code is to be
rebuilt from clean against it. The documents under [`docs/`](docs/) are the source of
truth.

## Start here

| Document | What it covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | **Source of truth.** Locked requirements, topology, slow/fast split, spatial dimensions, module structure, open risks. |
| [`docs/specs/timing-and-deadlines.md`](docs/specs/timing-and-deadlines.md) | **Spec A** — per-symbol air-time, deadline budget, drop policy, reassembly ring, phased reassembly. |
| [`docs/specs/fronthaul-packet-format.md`](docs/specs/fronthaul-packet-format.md) | **Spec B** — custom fronthaul wire format: 20-byte header, eAxC, U/C/S-plane payloads, compression. |
| [`docs/decisions/0001-hot-path-synchronization.md`](docs/decisions/0001-hot-path-synchronization.md) | **ADR 0001** — why the hot path is a CUDA Graph + CPU-controlled DOCA + indirection-cell double buffering (not persistent kernels). |
| [`docs/decisions/0002-multi-cell-interference-mobility.md`](docs/decisions/0002-multi-cell-interference-mobility.md) | **ADR 0002** — multi-cell on one box, cross-link interference model, configurable contributor set, grid mobility, and the compute-bound reconsideration. |
| [`docs/MILESTONES.md`](docs/MILESTONES.md) | Stage-by-stage implementation plan and hot-path invariants. |

## Key design points

- **Per-symbol cadence at µ=1** — deadline = `T_air − T_egress − T_proc − T_margin` (Spec A).
- **CUDA Graph hot path** — capture `gather → precode → channel → combine → pack` once,
  replay per symbol with one `cudaGraphLaunch`; no persistent kernels at µ=1 (ADR 0001).
- **DOCA GPUNetIO + GPUDirect RDMA** — zero-copy packet → GPU memory.
- **Slow/fast channel split** — heavy ray tracing decoupled from the per-symbol apply,
  published to the hot path via an atomic write into a device **indirection cell**.
- **Both precoding modes** — vDU-supplied weights (C-plane) and GPU-computed
  (ZF/MMSE/SVD from SRS).
- **64 TRX, spatially symmetric** — DL precodes 16 layers → 64 Tx; UL combines
  64 Rx → 16 layers (see [architecture § Spatial dimensions](docs/architecture.md#spatial-dimensions)).
- **Multi-cell with interference** — one box hosts all cells; cross-link channel
  `H[cell][ue]`; per-UE contributor set (serving + interferers) is **configurable**
  (neighbor-limited top-K → all-to-all). Channel-apply is **memory-bandwidth bound**
  (AI ≈ 2 FLOP/byte): per-subcarrier `H` needs ~42 TB/s vs ~3.35 TB/s HBM, so `H` is
  stored **per PRB-group** (fits L2) or applied **from CIR taps**; neighbor-limiting
  bounds the contributor sum (ADR 0002 §6).
- **Grid mobility** — UEs move on a discrete grid; an offline ray-traced per-`(cell,
  grid-point)` CIR table is looked up on a move; per-symbol Doppler per link; handover
  supported.
- **Phased reassembly** — v1 whole-symbol, v2 first-arrived-section streaming, no wire
  change.

## Intended build (target: Linux, CUDA 12.x, DOCA 2.x)

> Not yet present — recorded here so the rebuild targets a known configuration.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
```

`-DEMU_WITH_DOCA=OFF` will build the DSP/orchestration logic against a loopback
software I/O backend (no NIC/DOCA) for development on non-target hosts.
