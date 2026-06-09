# ADR 0003 — Throughput/latency decoupling and the symbol pipeline

- **Status:** Accepted
- **Date:** 2026-06-09
- **Context tags:** real-time, pipeline, throughput, latency, scaling, in-box vUE, µ=1
- **Refines:** the single budget table in [Spec A](../specs/timing-and-deadlines.md)
  (the `20/3/5/7` µs split) and the "compute is light / timing is hard" framing in
  [architecture.md](../architecture.md) and [ADR 0001](0001-hot-path-synchronization.md).
  ADR 0001's synchronization model and ADR 0002's bandwidth analysis are **unchanged**.

## Context

The original per-symbol budget squeezed ingest + compute + egress into **one** 35.7 µs
symbol window, which produced a bogus `T_proc ≤ 3 µs` (compute treated as the *residual*
of a single symbol). Two facts force a cleaner model:

1. **Symbol processing pipelines across the timeline** — while symbol N egresses, N+1
   computes, N+2 ingests. Processing latency for one symbol may therefore span **more
   than one symbol period**.
2. **The delivery deadline is not the symbol period.** Each symbol must be *delivered*
   within a tolerance set by the real vDU/vUE fronthaul, on the order of **~70 µs (≈ 2
   symbol periods, TBD)** — not 35.7 µs.

These are two different clocks. Conflating them is what broke the budget.

## Decisions

### 1. Two independent constraints

**Throughput (a rate).** The pipeline must complete one symbol every
`T_sym = 35.7 µs` (µ=1, fixed by numerology). The binding constraint is the **slowest
stage**:
```
max(stage service time) ≤ T_sym
```

**Latency (a deadline).** Each symbol must be delivered within `L_max`, the end-to-end
budget:
```
Σ stage latencies ≤ L_max
```
`L_max` is set by the vDU/vUE fronthaul timing tolerance. **Working placeholder
`L_max ≈ 70 µs ≈ 2·T_sym`**, and it **must be strictly smaller than the real
tolerance** (which is TBD). `L_max` caps pipeline depth.

### 2. The symbol pipeline

Realize the constraints with the N ≥ 3 symbol ring already in
[Spec A §A.5](../specs/timing-and-deadlines.md): `ingest → compute → egress` stages run
concurrently on consecutive symbols (N egress ∥ N+1 compute ∥ N+2 ingest). Then
**latency = Σ stage times** and **throughput = 1 / max(stage time)**. Pipelining
*spends latency to buy throughput*.

### 3. `T_proc ≤ 3 µs` is retired

Compute is **not** a 3 µs residual. On the **throughput** clock it may take up to a
**full `T_sym`** of service time (the next symbol's compute waits for the engine). On
the **latency** clock it contributes its *actual* time (~6 µs multi-cell, ADR 0002 §6)
to the `Σ ≤ L_max` sum. A 6–20 µs compute stage satisfies both clocks. The new budget
lives in Spec A §A.3 as **two tables** (throughput vs latency).

### 4. Pipelining does NOT relax the rate constraint

Overlapping *different* stages does not give any single stage more time per symbol.
Every symbol needs the compute stage, and symbols arrive every `T_sym`, so the compute
stage must **sustain** `H` streaming at the symbol rate. The **42 TB/s bandwidth wall of
ADR 0002 §6 is a throughput constraint** and is untouched by pipelining. It is relaxed
**only** by:
- reducing `H`'s footprint — per-PRB-group / tap-domain (Spec C), or
- **replicating across GPUs** (§6 below).

Both this and the §3 relaxation hold simultaneously: pipelining fixes the latency-driven
3 µs error; it does nothing for the rate-driven bandwidth wall.

### 5. vUE is in-box and GPU-resident

The vUE module runs **inside the GPU box**. Therefore:

- **Only the vDU side crosses the NIC** (custom fronthaul over DOCA GPUNetIO). The
  **vUE side is an in-GPU-memory handoff** — no NIC, and no fronthaul packetization is
  required on that side (Spec B is scoped to the vDU/fronthaul side). The vUE-side
  interface form is decided in **[ADR 0004](0004-vue-interface-ipc.md)**: a shared HBM
  buffer via **CUDA IPC** with a **DPDK shared-memory** control plane (Phase 1).
- **vUE compute must be GPU-resident, not host-CPU.** The DL output to the vUE is
  per-UE **per-rx-antenna** IQ — for the 7-cell reference, ~5.7 MB/symbol (~160 GB/s).
  Kept in HBM that is trivial (≪ 3.35 TB/s); pushed to a host-CPU vUE it would cross
  **PCIe (~64 GB/s) and become the bottleneck**. A host-resident vUE is therefore
  **rejected**.
- **The NIC/fronthaul bandwidth budget covers only the vDU side** (layer-domain IQ,
  `numLayers` not `numTx`): ~40 GB/s per direction for the 7-cell reference → a
  400G-class NIC. The large per-antenna traffic never leaves the GPU.

### 6. Scaling: cells-per-box vs box count

- **Cells-per-box** is bounded by the **throughput clock** (compute service ≤ `T_sym`,
  HBM bandwidth for `H`, GPU memory) — **not** by latency. This is the **cost lever**:
  maximize cells/box, minimize box count for the product.
- **More boxes = throughput scaling by replication** (parallel pipelines over disjoint
  symbol or cell subsets). This is a **later phase**.
- **Inter-box communication is required only if *interfering* cells are split across
  boxes** (then per-symbol IQ/channel contributions must be exchanged within the
  deadline — hard). **Partitioning rule:** assign cells to boxes so that each UE's
  neighbor-limited interferer set (ADR 0002 §3) stays **on one box**, eliminating
  cross-box exchange in the symbol loop. Cross-box channel-state exchange is deferred to
  the multi-box phase and only if partitioning cannot avoid it.

## Consequences

### Positive
- The budget is now correct and auditable: a rate constraint and a deadline constraint,
  each with its own table. The design **closes** — latency Σ ≈ 51–58 µs < ~70 µs, and
  throughput is met iff the compute stage sustains ≤ `T_sym` (which needs Spec C's `H`
  reduction).
- vUE-in-box removes the per-antenna PCIe wall and shrinks the NIC budget to the vDU
  side only.
- Cells-per-box becomes an explicit, measurable cost lever.

### Negative / costs
- Feasibility now rests squarely on the **throughput** clock: the compute stage must
  sustain ≤ `T_sym`, which is **not** achievable with per-subcarrier `H` (ADR 0002 §6).
  Spec C is on the critical path.
- A real `L_max` must be obtained from the fronthaul tolerance; ~70 µs is a placeholder
  and an upper bound on pipeline depth.
- Multi-box replication adds a partitioning constraint and, in the worst case, an
  inter-box channel-state exchange (later phase).

### Doc impact
- **Spec A §A.3** → split the single budget into **throughput** and **latency** tables;
  retire `T_proc ≤ 3 µs`.
- **architecture.md** → requirements/insight: add vUE-in-box, the two clocks, NIC budget
  scoped to vDU side.
- **CLAUDE.md** → locked decisions + open threads; next ADR is 0004.

## Revisit if…
- The real fronthaul tolerance forces `L_max` below the Σ of stage latencies (~50–58 µs)
  → shrink stages or reduce pipeline depth.
- Throughput cannot be met on one GPU even with `H` reduced (Spec C) → replication +
  partitioning (§6) becomes mandatory, reopening inter-box channel-state exchange.
- A reason emerges to run vUE off-GPU (e.g. external UE stack) → re-examine the PCIe
  egress wall in §5.

## Open
- **`L_max`** actual value from the vDU/vUE fronthaul timing tolerance (TBD; ≤ ~70 µs).
- ~~vUE-side interface form~~ → resolved in [ADR 0004](0004-vue-interface-ipc.md).
