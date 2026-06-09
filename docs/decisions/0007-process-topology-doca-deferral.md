# ADR 0007 — Process topology & DOCA deferral

- **Status:** Accepted
- **Date:** 2026-06-09
- **Context tags:** process architecture, IPC, DOCA, GPUDirect, deferral, host-staged, ORU
- **Refines:** the north-side ingress in [ADR 0001](0001-hot-path-synchronization.md)
  (which assumed CPU-controlled **DOCA GPUNetIO**). ADR 0001's CUDA-graph / sync model is
  **unchanged**; only the ingress/egress *mechanism* changes.
- **Mirrors:** [ADR 0004](0004-vue-interface-ipc.md) / [Spec D](../specs/vue-interface-contract.md)
  (the vUE/south interface). **Feeds:** [deferred-goals.md](../deferred-goals.md) (DOCA).

## Context

Two refinements to the north (vDU) side:
1. **DOCA / GPUDirect is deferred** to a later phase.
2. A **separate ORU process** terminates Ethernet to/from the vDU and relays to ORCA.

This makes the box a **three-process** system and changes north ingress from a zero-copy
NIC→GPU DMA to a **host-staged** path. It works for Phase 1 because the vDU-side data is
small (see §4).

## Decisions

### 1. Three in-box processes

```
  vDU ──Ethernet (Spec B)──► ┌─────────────┐  host shm + DPDK ctrl  ┌──────────┐  DPDK shm + CUDA IPC  ┌──────┐
                             │ ORU process │ ◄───────(Spec F)──────► │  ORCA    │ ◄──────(Spec D)──────► │ vUE  │
  vDU ◄─Ethernet (Spec B)─── │ NIC + framing│      H2D / D2H          │ GPU comp │   HBM (per-antenna)   │ (GPU)│
                             └─────────────┘                         └──────────┘                       └──────┘
```

- **ORU process** — owns the NIC and the **Spec B** ORU-fronthaul framing (RX/TX,
  reassembly, C-plane parse). NIC via kernel sockets / AF_XDP / DPDK (vendor's choice) —
  **not** DOCA in this phase.
- **ORCA** — GPU compute (precode → channel-apply → combine), the CUDA-graph hot path.
- **vUE** — emulated UEs (GPU PHY).
- All same host; ORCA holds the GPU context.

### 2. DOCA / GPUDirect deferred

Zero-copy NIC→GPU (DOCA GPUNetIO + GPUDirect RDMA) moves to
[deferred-goals → DOCA](../deferred-goals.md#doca). North ingress is **host-staged**: the
ORU process receives packets into **host memory**, de-frames Spec B, and places
frequency-domain **layer IQ** + the **allocation/beam_id map** into host shared memory.

### 3. ORU ↔ ORCA interface = host-bulk + DPDK control + H2D/D2H

Detailed in **[Spec F](../specs/oru-interface-contract.md)**. Bulk layer IQ lives in
**host shared memory** (pinned by ORCA via `cudaHostRegister`); ORCA `cudaMemcpyAsync` H2D
into GPU staging, kernel **K0** converts `ci16→cf32` into `x_dl`; UL reverses (D2H `z` →
host shm → ORU packetizes). Control over a **DPDK shared-memory** ring — the **same
mechanism as the vUE side** (uniform IPC).

### 4. Justified by the small vDU-side volume

The north side carries **layer-domain** IQ (`rank ≤ 4` layers/cell), not 64 antennas:

```
Phase-1 SU 2-cell, ci16:  ~ C · rank · numSc · 4 B  ≈ 2·4·3276·4 ≈ 105 KB/symbol
                          ≈ 2.9 GB/s  ≪ PCIe Gen5 (~64 GB/s)
```

A H2D copy of ~105 KB is ~1 µs/symbol — negligible against the budget (Spec A). This is
why DOCA's zero-copy is **not needed** at Phase-1 scale. (Contrast the vUE side's
~160 GB/s per-antenna IQ, which **must** stay in HBM — ADR 0004.)

### 5. ORCA hot path unchanged; north abstracted for the DOCA swap

The CUDA graph (kernels K0–K5, [Spec E](../specs/gpu-kernel-design.md)) is unchanged —
only K0's **source** (host-staged H2D) and K5's **sink** (D2H to host shm) differ. Wrap
the north link behind an **`OruTransport`** interface (like `VueTransport`): the Phase-1
backend is host-shm+H2D; the **deferred DOCA backend** DMAs straight to GPU and the H2D
copy disappears.

## Consequences

### Positive
- **Separation of concerns:** Ethernet / fronthaul stack lives in the ORU process; ORCA
  is pure GPU compute. The vDU vendor's FH library can live in the ORU process.
- **Drops a heavy dependency** (DOCA) from Phase 1; host-staged is simple and portable.
- Symmetric IPC: both ORCA sides use DPDK shm for control; bulk placement differs by
  volume (north = host, south = HBM).
- Re-enabling DOCA later is an `OruTransport` backend swap that **removes** the H2D copy.

### Negative / costs
- Adds a **per-symbol H2D/D2H copy** on the north side (~1 µs at SU 2-cell). It **grows**
  with vDU-side volume (MU-MIMO, more cells, more layers, full-band) and would eventually
  justify reinstating DOCA — see Revisit.
- A **third process** and its IPC (Spec F) to build and maintain.
- Two IPC contracts (Spec D south, Spec F north) instead of one NIC path.

### Doc impact
- **New:** Spec F (ORU↔ORCA), this ADR.
- **Spec B** → framed as the **vDU↔ORU-process** wire format (ORU terminates it; ORCA
  never sees Ethernet).
- **Spec E** → K0 ingress source = host-staged H2D; K5 egress sink = D2H to host shm.
- **ADR 0001** → note DOCA deferred; graph/sync model stands.
- **architecture.md / README** → three-process topology; ingress row = host-staged.
- **deferred-goals.md** → add DOCA / GPUDirect.
- **CLAUDE.md** → locked decision; next ADR 0008.

## Revisit if…
- **vDU-side volume grows** past the host-staged comfort zone — MU-MIMO (16 layers/cell),
  many cells, or full-band → H2D approaches PCIe / the budget → **re-enable DOCA /
  GPUDirect** (the `OruTransport` DOCA backend; removes the copy).
- The ORU and ORCA processes would be simpler **merged** (e.g. one process owning both NIC
  and GPU) — re-evaluate the split.
