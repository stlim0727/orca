# Deferred goals & future-phase considerations

**Purpose.** A single register of capabilities intentionally **out of the current phase**,
each with *why it's deferred* and — most importantly — **what it will cost to bring back**
(the compromises, the enabling work, and the constraints to respect). Read this before
proposing to pull any item forward: the "Considerations / compromises" column is the
trap-list.

**Current phase (Phase 1) in one line:** SU-MIMO, **2 cells**, all-to-all inter-cell
interference, grid-based UE mobility, all `H` resident in GPU memory as **per-subcarrier
FP16**, µ=1, vUE in-box (GPU PHY, CUDA-IPC), CUDA-Graph hot path on a single PCIe H100.

## Index

| # | Deferred goal | Target phase | Enabling work | Primary compromise to accept |
|---|---|---|---|---|
| 1 | [MU-MIMO](#mu-mimo) | very later | Spec C | sub-PRB-group selectivity **or** precision **or** scale |
| 2 | [Spec C — `H` coherence granularity](#spec-c) | when #1 or #3 | — (is the enabling work) | lose per-SC selectivity (per-PRB-group) or add compute (tap-domain) |
| 3 | [More cells (>2)](#more-cells) | later | Spec C and/or replication | interference pruning, or more GPUs |
| 4 | [µ≥2 / FR2](#fr2) | out of scope | persistent kernels, v2 reassembly | tighter timing, GPU-controlled DOCA |
| 5 | [Multi-box scaling + inter-box comms](#multi-box) | very later | cell partitioning, state exchange | partition so interferers co-reside, or pay cross-box latency |
| 6 | [vUE Phase 2 — CPU PHY / Grace-Hopper](#vue-phase2) | later | NVLink-C2C platform, DPDK-over-NVLink | coupled HW + PHY migration |
| 7 | [GPU-computed precoding (SRS)](#gpu-precoding) | deferred (ADR 0006) | SRS est. + cuSOLVER weight path | beam-indexed codebook (grid of beams) only |
| 8 | [v2 streaming reassembly](#v2-reassembly) | later | scheduler change only | none on the wire |
| 9 | [Continuous (non-grid) mobility](#continuous-mobility) | later | live ray tracing / fine interp | grid quantization now |
| 10 | [Implementation deferrals](#impl-deferrals) | at impl time | per item | bit-exactness / config polish |
| 11 | [DOCA / GPUDirect (zero-copy NIC→GPU)](#doca) | when north volume grows | DOCA GPUNetIO backend | host-staged H2D copy until then |

---

## MU-MIMO
<a id="mu-mimo"></a>

**Status:** deferred to a **very later phase** ([ADR 0005](decisions/0005-su-mimo-phase1.md)).
Phase 1 is SU-MIMO only.

**Why deferred.** MU-MIMO stacks all `U_c` UEs on every subcarrier, multiplying the
per-symbol `H` read by `U_c` (16×): 2-cell all-to-all jumps from **~0.38 TB/s (SU) to
~6.0 TB/s (MU)**, i.e. **1.8× over HBM** on one H100 (ADR 0002 §6).

**Considerations / compromises when re-enabling — pick at least one:**
- **Spec C is mandatory.** Per-subcarrier FP16 `H` cannot serve MU at 2 cells. Options:
  - *Per-PRB-group `H`* (~48× less, fits L2): **compromise = lose sub-PRB-group frequency
    selectivity.**
  - *Tap-domain apply*: full selectivity, **compromise = extra compute + a tap-apply
    kernel**, plus committing a tap count `P`.
  - *INT8/BF16 `H`*: ~2× only — **not enough alone** for MU at 2 cells; **compromise =
    amplitude precision** and still needs another lever.
- **Compute also grows ~16×** (precode/combine on the full layer aggregate, not UE rank).
- **Scheduling map no longer needed** for occupancy (everyone is everywhere), but the
  precoder must separate co-scheduled UEs (MU precoding / pairing) — new weight logic.
- **Per-cell UE count** becomes a direct bandwidth multiplier again → may need to cap
  `U_c`, reduce cells, or replicate across GPUs (#5).

**Refs:** ADR 0005, ADR 0002 §6.

## Spec C — `H` coherence granularity
<a id="spec-c"></a>

**Status:** deferred; **not on the Phase-1 critical path** because SU-MIMO + per-SC FP16
fits (ADR 0005). Becomes **critical the moment MU-MIMO (#1) or more cells (#3) return.**

**What it decides.** Whether `H` is applied **per-subcarrier** (Phase 1), **per-PRB-group**,
or **from CIR taps** — plus `H` **precision** (FP16/BF16/INT8) and **tap count `P`** — set
against a target delay spread / coherence bandwidth.

**Considerations / compromises:**
- *Per-PRB-group*: cheapest bandwidth, **compromise = sub-group selectivity** (acceptable
  only if delay spread keeps the coherence bandwidth ≥ group width).
- *Tap-domain*: keeps selectivity, **compromise = more compute and an explicit `P`**; mis-set
  `P` either over-costs or under-models the delay spread.
- Decision needs a **target delay spread** — without it, the granularity is a guess.

**Refs:** ADR 0002 §6 (open question), ADR 0005.

## More cells (>2)
<a id="more-cells"></a>

**Status:** Phase 1 fixed at **2 cells**. Cell count is the cost lever (ADR 0003 §6):
maximize cells/box, minimize boxes. Full scaling curve + roadmap in
[ADR 0009](decisions/0009-cell-count-scaling.md).

**Considerations / compromises:**
- Even under SU, per-SC `H` BW scales with `C²` (victims × contributors, all-to-all); the
  HBM wall lands at **≈ 6 cells**, with an earlier **L2-residency cliff at ≈ 4 cells**
  (steady state degrades to cold) — ADR 0009 §A.1/§A.2. Apply **Spec C** (per-PRB-group /
  tap-domain) or **neighbor-limit** the interferer set (`C²` → `C·(K+1)`, linear),
  **compromise = interference-model fidelity / sub-group selectivity**.
- Beyond one GPU's memory/compute → **replicate across boxes (#5)** with interferer-aware
  partitioning (ADR 0009 Part B).

**Refs:** [ADR 0009](decisions/0009-cell-count-scaling.md), ADR 0003 §6, ADR 0002 §3/§6.

## µ≥2 / FR2
<a id="fr2"></a>

**Status:** out of scope (ADR 0001, ADR 0003). Phase 1 is µ=1 (35.7 µs symbol).

**Considerations / compromises when re-enabling:**
- Symbol shrinks to ~8.9 µs (µ=3) → a ~5–10 µs kernel/graph launch is **fatal** →
  **reopen persistent kernels + GPU-controlled DOCA** (the model ADR 0001 rejected for µ=1)
  with all its synchronization cost (system-scope flags, cooperative launch).
- Arrival window must shrink → **v2 streaming reassembly (#8) becomes mandatory.**
- The latency budget (`L_max`) and pipeline depth must be re-derived.

**Refs:** ADR 0001 "Revisit if…", ADR 0003.

## Multi-box scaling + inter-box comms
<a id="multi-box"></a>

**Status:** very later phase (ADR 0003 §6; full strategy in
[ADR 0009 Part B](decisions/0009-cell-count-scaling.md)). Phase 1 is single-box.

**Considerations / compromises:**
- **Partition cells across boxes so each UE's interferer set stays on one box** → avoids
  per-symbol cross-box exchange. **Compromise = partitioning constraint** on cell
  placement. (Neighbor-limiting makes interference short-range → geographic clusters cut
  cleanly — ADR 0009 §B.1.)
- If interferers must span boxes → **exchange per-symbol antenna-domain `y` within the
  deadline** (~47 GB/s/ghost-cell) — hard synchronization; tractable only over NVLink
  (single-node, Tier 1); multi-node IB must partition cleanly (Tier 2). **Compromise =
  added latency / a tighter `L_max`**, or a fidelity cut for dense cross-node interference
  (Tier 3). See ADR 0009 §B.2–§B.4.

**Refs:** [ADR 0009](decisions/0009-cell-count-scaling.md) (Part B), ADR 0003 §6, ADR 0002 §1.

## vUE Phase 2 — CPU PHY / Grace-Hopper
<a id="vue-phase2"></a>

**Status:** later (ADR 0004). Phase 1 = GPU PHY on PCIe H100, bulk IQ in HBM via CUDA IPC.

**Considerations / compromises:**
- vUE-on-CPU and **Grace-Hopper (NVLink-C2C)** are a **coupled** migration — a CPU PHY
  needs ~160 GB/s of bulk IQ to host, viable only over NVLink, **not PCIe**. **Compromise
  = hardware dependency** (can't do CPU PHY on a PCIe box at full volume).
- Bulk transport swaps CUDA IPC → DPDK-over-NVLink; the `VueTransport` interface and
  buffer layout are designed to keep this a backend swap.

**Refs:** ADR 0004.

## GPU-computed precoding (from SRS)
<a id="gpu-precoding"></a>

**Status:** **deferred** ([ADR 0006](decisions/0006-beam-indexed-precoding.md)). Phase 1
uses a **resident beam codebook** with `beam_id` supplied by the vDU at runtime — no SRS,
no cuSOLVER, no slow-plane weight estimation (`estim/` dormant).

**Why deferred.** Beam-indexed precoding is sufficient for Phase 1 and far lighter (a
small index on the wire + a gather, vs estimation + matrix solve).

**Considerations / compromises when re-enabling:**
- Adds **SRS channel estimation + ZF/MMSE/SVD** weight computation on the slow plane
  (cuSOLVER batched). **Compromise = slow-plane compute + estimation accuracy.**
- Needed for **channel-matched / reciprocity-based** precoding and **MU-MIMO pairing**
  (it pairs with [MU-MIMO](#mu-mimo)).
- May add an **explicit-`W` C-plane** variant (Spec B §B.5) if weights are vDU-derived
  rather than codebook-indexed.
- **Compromise vs codebook:** beam-indexed precoding is limited to the predefined beam
  set (grid of beams); SRS enables arbitrary per-UE precoding at the above cost.

**Refs:** ADR 0006, architecture (precoding/combining), Spec B §B.5.

## v2 streaming reassembly
<a id="v2-reassembly"></a>

**Status:** later optimization (Spec A §A.6). Phase 1 = v1 whole-symbol.

**Considerations / compromises:**
- **No wire change** — the 20-byte header already carries `sectionId/startPrb/numPrb`.
- Pure scheduler change; needed for tighter arrival windows (e.g. µ≥2, #4).

**Refs:** Spec A §A.6.

## Continuous (non-grid) mobility
<a id="continuous-mobility"></a>

**Status:** later (ADR 0002 "Revisit if…"). Phase 1 = grid-quantized positions.

**Considerations / compromises:**
- Arbitrary continuous motion → **live ray tracing** or **finer-grid interpolation** on
  the runtime path. **Compromise = runtime compute** (live RT) or **table size /
  interpolation error** (finer grid).

**Refs:** ADR 0002 §5.

## Implementation deferrals
<a id="impl-deferrals"></a>

**Status:** deferred to implementation time (no design blocker).

- **BFP compression** exponent/scaling for bit-exactness (Spec B.4) — start int16.
- **Telemetry wire format** (Spec B.7) — format TBD.
- **YAML config loader** — defaults usable without a file until then.
- **Per-symbol scheduling/allocation map wire format** (needed by SU-MIMO, ADR 0005) —
  refine in Spec B C-plane.

**Refs:** Spec B, ADR 0005.

## DOCA / GPUDirect (zero-copy NIC→GPU)
<a id="doca"></a>

**Status:** deferred ([ADR 0007](decisions/0007-process-topology-doca-deferral.md)). Phase-1
north ingress is **host-staged**: ORU process → host shm → ORCA H2D ([Spec F](specs/oru-interface-contract.md)).

**Why deferred.** The vDU-side volume is **layer-domain** IQ (~3 GB/s SU 2-cell), so a
host-staged H2D/D2H copy (~1 µs/symbol) is fine; GPUDirect's zero-copy is only needed when
the NIC↔GPU volume is large.

**Considerations / compromises when re-enabling:**
- Adds the **DOCA GPUNetIO + GPUDirect RDMA** dependency; the ORU process either does
  GPUDirect RX/TX or merges into ORCA.
- Implemented as the **`OruTransport` DOCA backend** (Spec F §F.8) — NIC DMAs Spec B
  payloads straight to GPU; the H2D/D2H copies vanish.
- **Needed when north volume grows:** MU-MIMO (16 layers/cell), many cells, more layers, or
  full-band → H2D approaches PCIe / the budget.
- This restores ADR 0001's original "CPU-controlled DOCA" ingress.

**Refs:** ADR 0007, ADR 0001, Spec F.

---

## How to use this register

- An item moves out of here only via an **ADR** that accepts its compromises and names the
  enabling work.
- When you pull an item forward, **re-check every other item it touches** (e.g. MU-MIMO →
  Spec C → maybe more cells → maybe multi-box). The dependencies are real; this register's
  cross-refs are the map.
