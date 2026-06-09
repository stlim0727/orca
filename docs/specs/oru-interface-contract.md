# Spec F — ORU ↔ ORCA interface contract (in-box, north, Phase 1)

**Status:** Settled design (source of truth). No implementation yet.

Defines the **north-side interface** between the **ORU process** (NIC + Spec B framing) and
**ORCA** (GPU compute). Phase 1 is **host-staged** — DOCA/GPUDirect deferred
([ADR 0007](../decisions/0007-process-topology-doca-deferral.md)). Bulk **layer IQ** lives in
**host shared memory**; ORCA copies it H2D/D2H. Mirrors [Spec D](vue-interface-contract.md)
(the south/vUE side) and uses the **same DPDK shm control mechanism**.

Related: [ADR 0007](../decisions/0007-process-topology-doca-deferral.md),
[Spec B](fronthaul-packet-format.md) (the vDU↔ORU wire format the ORU process terminates),
[Spec E §E.7](gpu-kernel-design.md) (K0/K5), [Spec A](timing-and-deadlines.md) (deadlines).

## F.1 Model & invariants

- **ORU process** terminates Ethernet to/from the vDU (Spec B), reassembles per symbol, and
  parses the C-plane → an **allocation/beam_id map**. It does **not** touch the GPU.
- **ORCA** owns the GPU. It maps the host shm, **pins it** (`cudaHostRegister`) for fast
  async copies, and runs the hot path.
- **Bulk = host shared memory** (`ci16` layer IQ); **control = DPDK shm** (`rte_ring`); the
  GPU boundary is crossed by `cudaMemcpyAsync` **H2D (DL in)** / **D2H (UL out)**.
- Small volume (~3 GB/s, ADR 0007 §4) → the copy is ~1 µs/symbol; **no GPUDirect needed**.

## F.2 Host shared-memory buffers (bulk)

Hugepage memzone, ring of **N ≥ 3** symbol slots per direction (pipeline, Spec A §A.5):

| Ring | Dir | Slot tensor (`sc` innermost) | Elem | Size/slot (Phase 1) |
|---|---|---|---|---|
| `dlInRing` | ORU → ORCA | `x_dl_host[C][rank][numSc]` | **`ci16`** | 2·4·3276·4 B ≈ 105 KB |
| `ulOutRing` | ORCA → ORU | `z_host[C][rank][numSc]` | **`ci16`** | ≈ 105 KB |

- **`ci16` on the wire-adjacent host side** (matches Spec B U-plane): halves host shm and
  lets ORCA's **K0** do the `ci16→cf32` convert on the GPU (and **K5** the `cf32→ci16`).
- ORCA `cudaHostRegister`s both rings once at attach → `cudaMemcpyAsync` is true async
  pinned-DMA. (Alternatively K0 can read the pinned host buffer directly; default: explicit
  H2D into a GPU `x_dl_raw` staging slot, then K0 convert → `x_dl`.)
- Layout mirrors the GPU `x_dl`/`z` tensors (Spec E §E.2) so the copy is a flat memcpy.

## F.3 Allocation / scheduling map (per symbol)

The ORU process parses the Spec B C-plane (§B.5: allocation + `beam_id`, [ADR 0006](../decisions/0006-beam-indexed-precoding.md))
into the `Alloc[]` array ([Spec E §E.2](gpu-kernel-design.md)) and writes it into a host shm
region `allocBlock[slot]`:

```
struct Alloc { uint16 cell, ueId; uint16 scStart, scLen; uint8 dir, rank; uint16 beamId[4]; };
allocBlock[slot] = { uint16 numAllocs; Alloc allocs[MAX_ALLOCS]; }
```

ORCA H2D-copies `allocBlock[slot]` to device `d_allocs` and builds `d_victim`/`d_ulContrib`
(tiny kernel) before launching the graph. This is the SU-MIMO scheduling map (ADR 0005) +
beam_id (ADR 0006), delivered north-to-ORCA.

## F.4 Control region (DPDK shared memory)

Same shape as Spec D §D.3 — memzone `"orca.oru.ctrl.v1"`: header/config, four SPSC
`rte_ring`s (`dlDoorbell` ORU→ORCA, `dlReturn` ORCA→ORU, `ulDoorbell` ORCA→ORU, `ulReturn`
ORU→ORCA), stats. **Doorbell descriptor:**

| Field | Size | Meaning |
|---|---|---|
| `slotIdx` | u16 | host-ring slot |
| `sfn`/`slot`/`sym` | u16/u8/u8 | symbol key |
| `numAllocs` | u16 | entries in `allocBlock[slot]` |
| `flags` | u8 | bit0 = partial/late |
| `seq` | u32 | per-direction sequence |

## F.5 Bring-up handshake

Mirrors Spec D §D.5: ORU creates the memzone + host bulk rings, writes magic/version/config
(`C, rank, numSc, iqElem, N, MAX_ALLOCS`), sets `ORU_READY`; ORCA validates, maps, **pins**
the bulk rings (`cudaHostRegister`), sets `ORCA_ATTACHED`; steady state begins. Version/config
mismatch is a hard stop.

## F.6 Per-symbol protocol

**DL (vDU → ORU → ORCA):**
1. ORU receives/reassembles the symbol (Spec B), de-frames into `dlInRing[i]` (`ci16`), fills
   `allocBlock[i]`.
2. ORU `rte_ring_enqueue(dlDoorbell, {i, sfn,slot,sym, numAllocs, seq})`.
3. ORCA dequeues → `cudaMemcpyAsync(x_dl_raw, dlInRing[i], H2D, copyStream)` +
   `cudaMemcpyAsync(d_allocs, allocBlock[i], H2D)`; build `d_victim`.
4. The graph (K0 convert → … → K5) runs after the H2D completes (stream order / event).
5. ORCA `rte_ring_enqueue(dlReturn, {i, seq})` once the H2D has consumed `dlInRing[i]` (ORU
   may refill).

**UL (ORCA → ORU → vDU):**
1. After K4/K5 produce `z` (cf32) and pack to `ci16`, ORCA `cudaMemcpyAsync(ulOutRing[j],
   z_host_dev, D2H, copyStream)`.
2. On D2H completion (event/callback), ORCA `rte_ring_enqueue(ulDoorbell, {j, …})`.
3. ORU dequeues → packetizes `ulOutRing[j]` per Spec B → Ethernet to the vDU → `ulReturn`.

**Deadlines / drop:** the H2D must complete within the symbol's compute budget; D2H + ORU
packetization must finish before the vDU UL window (Spec A §A.3, `D_ul`). Late → Spec A §A.4
drop policy (zero-fill / skip, `flags.late`, never stall).

## F.7 Synchronization

- The **H2D copy gates K0** via stream ordering (copy and graph on ordered streams) or a
  recorded event the graph waits on — the graph must not read `x_dl_raw` before the copy lands.
- The **D2H completion gates the `ulDoorbell`** (the ORU must not read `ulOutRing[j]` before
  the copy lands) — use a copy-completion event or `cudaLaunchHostFunc` to enqueue the doorbell.
- The DPDK control ring is host-side signaling only; it never orders GPU memory.

## F.8 OruTransport API & the DOCA swap

```
attach()/detach()
pollDl() -> [(slotIdx, sfn,slot,sym, allocBlock*)]   // F.6 DL-3 (+ H2D)
returnDl(slotIdx)
publishUl(slotIdx, meta)                              // F.6 UL (+ D2H)
reclaimUl() -> [slotIdx]
```

ORCA calls `pollDl`/`publishUl`; it is unaware of the backend. **Phase-1 backend** =
host-shm + DPDK + H2D/D2H (this spec). **Deferred DOCA backend** (ADR 0007 §5,
[deferred-goals → DOCA](../deferred-goals.md#doca)) = NIC DMAs Spec B payloads **straight
into GPU memory** (GPUDirect); the H2D/D2H copies vanish and the ORU process role shrinks to
C-plane/control. The host tensor layout and the control ring are unchanged across the swap.

## F.9 Open / implementation-time

- `ci16` vs `cf32` on the host bulk (default `ci16` — smaller, GPU-side convert).
- K0 read-from-pinned-host vs explicit H2D-into-staging (default: explicit H2D).
- `MAX_ALLOCS` (shared with Spec E) and host-ring depth `N`.
- D2H-completion signaling: copy-event poll vs `cudaLaunchHostFunc` (host-latency trade).
- Whether `dlReturn`/`ulReturn` credits are needed given the small slot count (vs seq-only).
