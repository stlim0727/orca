# ORCA — *O-RU Channel Applier*

**ORCA** is a real-time GPU application that stands in for the **O-RU (radio
unit)** between a real 5G NR **vDU** and a set of **emulated UEs (vUE)**. It terminates the
vDU's fronthaul as if it were the O-RU, applies a ray-traced radio channel to the
frequency-domain IQ in real time, and performs precoding / receive-combining **on behalf of
the RU**.

```
            Ethernet (Spec B)        host shm + DPDK         ORCA (GPU)        DPDK + CUDA IPC
  ┌──────────┐              ┌─────────────┐  H2D/D2H  ┌──────────────────┐  (HBM bulk)  ┌──────────┐
  │ Real vDU │◄────────────►│ ORU process │◄─(Spec F)►│ precode → channel │◄──(Spec D)──►│   vUE    │
  │ (3rd pty)│  DL/UL IQ    │ NIC+framing │           │ → combine         │             │ (N UEs)  │
  └──────────┘              └─────────────┘           │ ⇒ O-RU/RU role    │             └──────────┘
                          (DOCA deferred)             └──────────────────┘
```

**Role & interfaces** — three in-box processes ([ADR 0007](docs/decisions/0007-process-topology-doca-deferral.md)):

- **North (vDU side):** a **separate ORU process** terminates the **ORU fronthaul packet
  format** (Spec B) over Ethernet (**DOCA deferred**; NIC via kernel/DPDK) and relays
  de-framed layer IQ to ORCA over **host shared memory + H2D** (Spec F). ORCA covers the
  **RU role** (precoding/combining on behalf of the RU) and never sees Ethernet.
- **South (vUE side):** ORCA exchanges per-antenna IQ with the in-box **vUE** over
  **DPDK shared memory** (control + handles; bulk stays in HBM, shared via CUDA IPC —
  ADR 0004 / Spec D).

- **DL:** vDU → precode (`64 × rank` per resource) → channel-apply, summed over each UE's
  serving cell **+ interferers** (cross-link `H`) → +noise → vUE.
- **UL:** vUE → channel-apply → per-cell RU sums served **+ interfering** UEs → combine
  (`64 → rank`) → vDU.
- **Slow plane:** UE mobility on a grid; per-`(cell, grid-point)` channel **resident in
  GPU memory in advance**; SRS-based weight updates per PRB-group.
- **Hot path:** per-OFDM-symbol apply within the symbol deadline (µ=1, 35.7 µs), all cells
  in one captured **CUDA Graph** launch.

## Status

**Design phase — no implementation yet.** The design is settled and the code will be
rebuilt clean against it. The documents under [`docs/`](docs/) are the **source of truth**
(this README only orients). An early code scaffold was intentionally removed once the
synchronization model was locked.

## Phase-1 target (what we're building first)

**SU-MIMO**, **2 cells**, **all-to-all** inter-cell interference, **grid-based UE
mobility**, all channel coefficients **resident in GPU memory in advance** as
**per-subcarrier FP16 `H`**, **µ=1**, **vUE in-box** (GPU PHY, shared via CUDA IPC),
**CUDA-Graph** hot path on a **single PCIe H100**.

This scope is chosen so the per-symbol `H`-read stays at ~0.38 TB/s (~11% of HBM) — it
fits one GPU with full frequency selectivity and **no** channel-compression work. The
deliberately-deferred capabilities (MU-MIMO, more cells, FR2, multi-box, …) and what each
costs to bring back are tracked in [`docs/deferred-goals.md`](docs/deferred-goals.md).

## Start here

| Document | What it covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | **Source of truth.** Locked requirements, topology, spatial dimensions, multi-cell/interference/mobility, module structure, open risks. |
| [`docs/specs/timing-and-deadlines.md`](docs/specs/timing-and-deadlines.md) | **Spec A** — per-symbol air-time, the two budgets (throughput vs latency), drop policy, reassembly ring. |
| [`docs/specs/fronthaul-packet-format.md`](docs/specs/fronthaul-packet-format.md) | **Spec B** — ORU fronthaul wire format (vDU↔ORU): 20-byte header, eAxC, multi-cell addressing, U/C/S-plane. |
| [`docs/specs/oru-interface-contract.md`](docs/specs/oru-interface-contract.md) | **Spec F** — ORU↔ORCA interface (north): host shm bulk + H2D/D2H, DPDK control, allocation/beam map (DOCA deferred). |
| [`docs/specs/vue-interface-contract.md`](docs/specs/vue-interface-contract.md) | **Spec D** — in-box vUE interface (south): shared HBM via CUDA IPC, DPDK shm control ring, bring-up handshake, per-symbol protocol. |
| [`docs/specs/gpu-kernel-design.md`](docs/specs/gpu-kernel-design.md) | **Spec E** — GPU kernel & memory design: tensor layouts/allocation, the 6 hot-path kernels (grid/block/warp/thread, coalescing, occupancy), `H`/`P`/`x`/`y` lifecycle + cache behavior, dynamic-`H` update. |
| [`docs/specs/cir-table-toolchain.md`](docs/specs/cir-table-toolchain.md) | **Spec G** — offline OptiX→CIR-table toolchain: geometric-path storage, UE grid model, on-disk format, and the slow-plane ray→`H` expansion that feeds the resident `H_dl` (Spec E). Realizes ADR 0002 §5; distinct from the deferred Spec C. |
| [`docs/decisions/0001-hot-path-synchronization.md`](docs/decisions/0001-hot-path-synchronization.md) | **ADR 0001** — CUDA Graph + CPU-controlled DOCA + indirection-cell double buffering (not persistent kernels). |
| [`docs/decisions/0002-multi-cell-interference-mobility.md`](docs/decisions/0002-multi-cell-interference-mobility.md) | **ADR 0002** — multi-cell on one box, cross-link interference, grid mobility, the `H`-bandwidth roofline. |
| [`docs/decisions/0003-throughput-latency-pipeline.md`](docs/decisions/0003-throughput-latency-pipeline.md) | **ADR 0003** — throughput vs latency decoupling, the symbol pipeline, vUE in-box. |
| [`docs/decisions/0004-vue-interface-ipc.md`](docs/decisions/0004-vue-interface-ipc.md) | **ADR 0004** — vUE IPC: CUDA IPC (HBM) + DPDK shm control now; CPU PHY over NVLink later. |
| [`docs/decisions/0005-su-mimo-phase1.md`](docs/decisions/0005-su-mimo-phase1.md) | **ADR 0005** — SU-MIMO for Phase 1; MU-MIMO deferred (the 16× bandwidth reason). |
| [`docs/decisions/0006-beam-indexed-precoding.md`](docs/decisions/0006-beam-indexed-precoding.md) | **ADR 0006** — beam-indexed precoding (resident codebook, `beam_id` from vDU); SRS deferred. |
| [`docs/decisions/0007-process-topology-doca-deferral.md`](docs/decisions/0007-process-topology-doca-deferral.md) | **ADR 0007** — three-process topology (ORU/ORCA/vUE); DOCA deferred; host-staged north ingress. |
| [`docs/decisions/0008-geometric-path-channel-storage.md`](docs/decisions/0008-geometric-path-channel-storage.md) | **ADR 0008** — store the offline channel as geometric paths (rays), not antenna-CIR or per-SC `H`; slow-plane ray→`H` expansion; host-resident table. Realized by Spec G. |
| [`docs/decisions/0009-cell-count-scaling.md`](docs/decisions/0009-cell-count-scaling.md) | **ADR 0009** — cell-count scaling: single-box ceiling (`C²` `H`-bandwidth wall ≈ 6 cells, L2 cliff ≈ 4; neighbor-limit → Spec C) and the multi-box strategy (interferer-aware partitioning; NVLink-halo vs InfiniBand-clean-partition tiers). |
| [`docs/deferred-goals.md`](docs/deferred-goals.md) | Register of deferred capabilities + the compromises/enabling work to re-enable each. |
| [`docs/MILESTONES.md`](docs/MILESTONES.md) | Stage-by-stage implementation plan and hot-path invariants. |

## Key design points

- **Per-symbol cadence, µ=1** — `T_sym = 35.7 µs`. Two separate clocks (ADR 0003):
  *throughput* = one symbol per `T_sym` (bottleneck stage ≤ `T_sym`); *latency* = deliver
  within `L_max` (~70 µs working, < real fronthaul tolerance). `T_proc ≤ 3 µs` is retired.
- **CUDA Graph hot path** — capture `gather → precode → channel → combine → pack` once,
  replay per symbol with one `cudaGraphLaunch`; no persistent kernels at µ=1 (ADR 0001).
- **Three in-box processes, DOCA deferred** (ADR 0007) — a separate **ORU process** owns
  the NIC + Spec B framing and relays layer IQ to ORCA over **host shm + H2D** (Spec F);
  small vDU-side volume (~3 GB/s) makes zero-copy GPUDirect unnecessary for now. Re-enabling
  **DOCA GPUNetIO + GPUDirect** is a later-phase `OruTransport` backend swap.
- **vUE is in-box, GPU-resident** (ADR 0004) — bulk per-antenna IQ stays in HBM, shared
  across processes via **CUDA IPC**; **DPDK shared memory** carries the control plane.
  Phase 2 (later): CPU PHY on Grace-Hopper over NVLink-C2C.
- **SU-MIMO in Phase 1** (ADR 0005) — one UE per time-frequency resource; per-resource
  layers = UE rank (≤4); needs a per-symbol scheduling map. **MU-MIMO deferred** (16×
  `H`-read blow-up → would require Spec C).
- **Slow/fast channel split** — heavy ray tracing is the **offline** generator of a
  per-`(cell, grid-point)` table; the per-symbol kernel applies the resident `H` + a
  per-link Doppler rotor, published via an atomic **indirection cell** (ADR 0001 §4).
- **Multi-cell with interference** — one box hosts all cells; cross-link
  `H[cell][ue][rx][tx][sc]`; **all-to-all** in Phase 1, neighbor-limited top-K available
  for scale (ADR 0002).
- **Beam-indexed precoding** (ADR 0006) — a resident codebook indexed by `beam_id`; the
  vDU supplies `beam_id` per resource at runtime (C-plane). SRS / GPU-computed
  (ZF/MMSE/SVD) precoding is **deferred**.
- **64 TRX, spatially symmetric** — DL precodes `rank → 64 Tx`; UL combines `64 Rx → rank`
  ("precombining on behalf of the RU").
- **Channel-apply is memory-bandwidth bound** — the design lever is `H` footprint/precision
  and interference scale, not FLOPs (ADR 0002 §6).

## Intended build (target: Linux, CUDA 12.x, DOCA 2.x)

> Not present yet — recorded so the rebuild targets a known configuration.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build build -j
```

`-DEMU_WITH_DOCA=OFF` will build the DSP/orchestration logic against a loopback software
I/O backend (no NIC/DOCA) for development on non-target hosts.
