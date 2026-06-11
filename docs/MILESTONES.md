# Milestones

Implementation plan for the (to-be-rebuilt) code. Each stage names the module it
lives in per the [intended structure](architecture.md#intended-module-structure-to-rebuild-from-clean).

| Stage | Goal | Module(s) |
|---|---|---|
| **1** | **Loopback, no channel (identity transform).** *North:* vDU ↔ **ORU process** (Ethernet, Spec B; **DOCA deferred** — kernel/DPDK) ↔ ORCA over **host shm + H2D/D2H** (**Spec F**). *South:* ORCA ↔ vUE over **DPDK shared memory** + bulk in HBM via **CUDA IPC** (ADR 0004 / **Spec D**). ORCA covers the O-RU/RU role; per-symbol jitter measured. | ORU process (sep. program), `oru/` (`OruTransport`), `vue/`, `orchestr/`, `app/` |
| **2** | DL precode via **resident beam codebook**, `beam_id` from the vDU C-plane (ADR 0006) + AWGN; validate vs CPU golden. | `scenario/` (codebook), `dsp/` (precode) |
| **3** | Static multipath channel apply (single CIR) + Doppler rotor. | `channel/`, `dsp/` (channel-apply) |
| **4** | Slow ray tracer / trace replay feeding the indirection-cell double buffer; offline CIR-table generator + loader + ray→`H` expansion (**Spec G**). | `channel/` |
| **5** | **UL combine** via resident combining codebook, `beam_id` from the vDU (ADR 0006). *(SRS / GPU-computed ZF/MMSE/SVD deferred — see deferred-goals.)* | `scenario/`, `dsp/` (combine) |
| **6** | Scale-out: many vUEs, deadline scheduler, sustained real-time soak. | `orchestr/`, `app/` |
| **7** | **Multi-cell + inter-cell interference** (Phase-1 target: **2 cells, all-to-all, SU-MIMO**): cell dimension end-to-end; cross-link `H[cell][ue]`; per-symbol scheduling/allocation map (ADR 0005); channel-apply as a batched **GEMV** — per-SC FP16 `H` fits HBM under SU (~11%, ADR 0005), no `H`-footprint reduction needed. | `scenario/`, `dsp/`, `channel/`, `fh/` |
| **8** | **Dynamic grid mobility**: **host-resident** offline per-`(cell, grid-point)` CIR table (**Spec G**), slow-plane lookup on UE move → ray→`H` expansion into the GPU-resident `H_dl` back buffer; per-link per-symbol Doppler; serving-cell handover. | `scenario/`, `channel/` |
| **9** *(deferred — very later)* | **MU-MIMO**: stack UEs per resource. **Requires Spec C** (`H` per-PRB-group/tap-domain) for the 16× `H`-read blow-up (ADR 0002 §6), plus MU precoding/pairing. See [deferred-goals → MU-MIMO](deferred-goals.md#mu-mimo). | `dsp/`, `channel/`, `estim/` |

> **Phase 1 = SU-MIMO, 2 cells, all-to-all, per-SC FP16 `H` resident** (ADR 0005). MU-MIMO
> (Stage 9) and Spec C are deferred — see [deferred-goals.md](deferred-goals.md).
> **Design in the cell dimension from Stage 1.** Even while Stages 1–6 run single-cell,
> tensor layouts, the reassembly key, and eAxC addressing should carry `cell` from the
> start (ADR 0002) so Stage 7 is an extension, not a refactor.

## Hot-path invariants (do not regress)

Per [ADR 0001](decisions/0001-hot-path-synchronization.md):

- **CUDA Graph hot path, CPU-controlled DOCA.** Capture
  `gather → precode → channel → combine → pack` once; replay per symbol with one
  `cudaGraphLaunch`. Inter-stage ordering = graph node dependencies; compute↔egress
  = one event per symbol. **No** persistent kernels, doorbells, or system-scope
  flags at µ=1.
- **Slow plane never touches the hot path.** Ray tracing and weight computation
  publish via an **atomic write into a device indirection cell**; the apply kernels
  dereference the cell at runtime and only ever read the active buffer.
- **Never stall on a late symbol.** Zero-fill missing PRBs and advance. Stalling
  desyncs the real vDU. (Spec A §A.4)
- **Interference set is data, not topology.** Changing the contributor set (K, or
  all-to-all) updates a link list, not the kernels or graph. The captured graph is
  sized for the configured **max** contributor count; instantaneous handover changes
  stay within that bound. (ADR 0002 §3, §6)
- **Static kernel grids (graph-replayable).** A captured graph bakes `gridDim`, but
  scheduling (`numAllocs`, `scLen`, `numSections`) varies per symbol → every hot kernel
  uses a **static, worst-case grid** with runtime guards + per-`sc` lookup tables
  (`d_victim`, `d_allocs`). Per symbol the host updates device tables only, never grid
  dims. (Spec E §E.8)

## Phased reassembly

- **v1 (first):** whole-symbol — launch when coverage complete or deadline hit.
- **v2 (later):** first-arrived-section streaming. **No wire change** — the 20-byte
  header already carries `sectionId/startPrb/numPrb`. (Spec A §A.6)

## Per-symbol budget — two clocks (ADR 0003)

Throughput and latency are **separate** constraints; see
[Spec A §A.3](specs/timing-and-deadlines.md) for the full two-table budget.

- **Throughput (rate):** one symbol per `T_sym = 35.7 µs`; bottleneck stage ≤ `T_sym`.
  The compute stage's HBM bandwidth for `H` (ADR 0002 §6) is the real limit; pipelining
  does **not** relax it.
- **Latency (deadline):** Σ stage latencies ≤ `L_max` (working ~70 µs ≈ 2·`T_sym`, <
  real fronthaul tolerance TBD). Representative Σ ≈ ~60–65 µs cold / ~50 µs steady →
  closes (Spec A §A.3 / Spec E §E.13).

`T_proc ≤ 3 µs` is **retired** (it was a residual-of-one-symbol category error, ADR 0003
§3). All segment numbers are provisional pending H100/NIC measurement and Spec C. µ=2/µ=3
are out of scope; revisiting them reopens the persistent-kernel option (ADR 0001).
