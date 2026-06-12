# ADR 0004 — vUE interface & IPC transport (GPU-resident now → host/NVLink later)

- **Status:** Accepted
- **Date:** 2026-06-09
- **Context tags:** vUE, IPC, CUDA IPC, DPDK, shared memory, PCIe, Grace-Hopper, in-box
- **Resolves:** the "vUE-side interface form" open item in
  [ADR 0003 §5](0003-throughput-latency-pipeline.md) and its `## Open` list.
- **Builds on:** ADR 0003 (vUE in-box; only the vDU side crosses the NIC).

## Context

The vUE is a **separate in-box process** from the emulator. The two must exchange, per
symbol: **bulk per-antenna IQ** (DL `r[ue][rx][sc]` to the vUE, UL transmit IQ back —
~160 GB/s for the 7-cell reference) plus a small **control plane** (symbol-ready
doorbells, ring indices, descriptors, UE/association config — kilobytes/symbol).

The transport (DPDK shared memory) and the **data placement** (HBM vs host) are separate
choices; placement is what binds, because ~160 GB/s does **not** fit PCIe Gen5
(~55–64 GB/s) but is trivial in HBM (~3.35 TB/s) or over NVLink-C2C (~450 GB/s/dir).

## Decisions

### 1. Two phases, coupled migrations

| | **Phase 1 — now** | **Phase 2 — future** |
|---|---|---|
| vUE PHY compute | **GPU-accelerated** | **host CPU stack** |
| Platform CPU↔GPU | **PCIe-attached H100** | **Grace-Hopper (NVLink-C2C)** |
| Bulk IQ placement | **HBM** (never crosses PCIe) | **host / coherent memory** (over NVLink) |
| Bulk IQ sharing | **CUDA IPC** (GPU buffer shared across processes) | **DPDK shared memory** over NVLink-C2C / unified memory |
| Control plane | **DPDK shared memory** (`rte_ring`, hugepages) | DPDK shared memory (unchanged) |

The two migrations are **one step, not two**: a CPU PHY requires the bulk IQ in host
memory, which is only viable at ~160 GB/s over NVLink-C2C — so vUE-on-CPU and
Grace-Hopper are adopted together.

### 2. Phase 1 mechanism (current target)

- **Bulk IQ stays in HBM.** The emulator allocates the per-antenna symbol-slot ring in
  GPU memory and exports it with **CUDA IPC** (`cudaIpcGetMemHandle`); the vUE process
  imports it (`cudaIpcOpenMemHandle`) once at startup. Per symbol, only **indices**
  move — zero copy, no PCIe crossing for payload.
- **Control plane over DPDK shared memory.** A hugepage memzone + lockless SPSC
  `rte_ring` carries symbol-ready doorbells, slot indices, and descriptors (incl. the
  CUDA IPC mem/event handles at bring-up). This honors the chosen DPDK transport while
  keeping it off the bulk-bandwidth path. (A POSIX shm + eventfd ring is a lighter
  fallback if DPDK isn't otherwise needed.)
- **Cross-process GPU sync** via **CUDA IPC events** (`cudaIpcGetEventHandle`): the
  emulator records a per-slot "produced" event; the vUE waits on it before reading, and
  records a "consumed" event the emulator waits on before reusing the slot. Ordering
  follows the symbol pipeline (ADR 0003 §2).
- **No Spec B framing on the vUE side.** The vUE interface is a shared GPU buffer + the
  control ring — **not** the custom fronthaul packet format (Spec B stays scoped to the
  vDU/NIC side). This closes the ADR 0003 §5 open item.

### 3. Phase 2 mechanism (future)

- Bulk IQ crosses to host over **NVLink-C2C** (~450 GB/s/dir ≫ 160 GB/s). On GH200's
  **coherent** memory the GPU can write where the CPU vUE reads, minimizing explicit
  copies. DPDK shared memory (hugepages / unified memory) carries the bulk at full
  volume; the control plane is unchanged.

### 4. Keep the migration cheap

- Put the vUE link behind a **`VueTransport`** interface (e.g. `publishDl(slotIdx)` /
  `pollUl(slotIdx)` + lifecycle), with a **Phase-1 CUDA-IPC backend** and a **Phase-2
  DPDK-over-NVLink backend**. The emulator hot path and captured graph do not change;
  only the backend swaps.
- Keep the **buffer layout identical** across phases (the per-antenna symbol-slot ring
  and tensor shapes). Only *placement* (HBM → host/coherent) and *sharing mechanism*
  (CUDA IPC → DPDK shm) change. The control-plane ring stays put, de-risking the swap.

## Consequences

### Positive
- **Phase 1 has zero PCIe bulk crossing** → the per-antenna-IQ bandwidth wall is avoided
  entirely; single-GPU feasibility still rests only on the throughput clock / Spec C
  (ADR 0003 §4), unchanged.
- vUE is a **separate process from day 1**, so IPC and process isolation are designed in
  from the start — good for modularity and for the Phase-2 backend swap.
- The chosen DPDK shared-memory transport is retained throughout (control plane in both
  phases; bulk in Phase 2), so it is not throwaway work.

### Negative / costs
- Phase 1 adds **cross-process GPU synchronization** (CUDA IPC mem + event handles) and a
  bring-up handshake to exchange handles over the control ring.
- Phase 2 is **gated on hardware (GH200) and a host PHY stack** — explicitly future, and
  it inherits the coupled-migration constraint (can't do CPU PHY on PCIe at full volume).
- CUDA IPC requires both processes on the **same host and GPU** — already guaranteed by
  "in-box," but it forecloses moving the vUE to another node without revisiting (that
  would be the multi-box phase, ADR 0003 §6).

## Open (implementation-time)

> **Now fully specified in [Spec D — vUE interface contract](../specs/vue-interface-contract.md)**
> (buffer layout, control-ring schema, handshake, per-symbol protocol, sync rules).

- ~~Control-ring schema and the exact CUDA IPC mem/event-handle bring-up handshake.~~ → Spec D.
- Whether to use DPDK shm or POSIX shm + eventfd for the **Phase-1 control plane**
  (lean DPDK if the vUE is already a DPDK app, since it also pre-stages the Phase-2 path;
  otherwise POSIX shm is lighter). Default: **DPDK shm**, per the project's stated choice.
- UL bulk volume (vUE → emulator) to confirm it is ≤ the DL figure (fewer UE Tx antennas).
