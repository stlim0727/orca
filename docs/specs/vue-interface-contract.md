# Spec D — vUE interface contract (in-box, Phase 1)

**Status:** Settled design (source of truth). No implementation yet.

Defines the **south-side interface** between **ORCA** and the **in-box vUE process**:
bulk per-antenna IQ shared in **GPU HBM via CUDA IPC**, coordination over **DPDK shared
memory**, cross-process GPU ordering via **CUDA IPC events**. This is the concrete
realization of [ADR 0004](../decisions/0004-vue-interface-ipc.md) Phase 1 and resolves its
"implementation-time" open items. It is **not** the ORU fronthaul format (Spec B) — that
is the north/vDU side only.

Related: [ADR 0004](../decisions/0004-vue-interface-ipc.md) (vUE IPC),
[ADR 0003 §5](../decisions/0003-throughput-latency-pipeline.md) (vUE in-box),
[Spec A §A.5](timing-and-deadlines.md) (symbol ring / drop policy).

## D.1 Model & invariants

- **Two processes, same host, same GPU** (CUDA IPC requirement). vUE PHY runs on the GPU.
- **Bulk IQ → shared HBM** (CUDA IPC). **Control → DPDK shm** (hugepage + `rte_ring`).
  **GPU↔GPU ordering → CUDA IPC events.**
- **ORCA owns and allocates** all shared regions (HBM bulk rings + the control memzone)
  and exports handles; **vUE attaches** (maps them). This keeps a single allocator/owner.
- The bulk payload **never traverses the DPDK shm / host memory** in Phase 1 — the control
  ring carries only indices, metadata, and (once) the IPC handles.

## D.2 Shared HBM buffers (bulk)

ORCA `cudaMalloc`s two rings, each a contiguous array of **N symbol slots** (N ≥ 3 to match
the pipeline, [Spec A §A.5](timing-and-deadlines.md)):

| Ring | Direction | Slot tensor | Element |
|---|---|---|---|
| `dlRing` | ORCA → vUE | `r_dl[ue][rx][sc]` (UE received per-rx-antenna IQ) | `iqElem` |
| `ulRing` | vUE → ORCA | `x_ul[ue][ueTx][sc]` (UE transmit per-antenna IQ) | `iqElem` |

- **Index order (row-major, `sc` innermost):**
  `dl[(ue*numRx + rx)*numSc + sc]`, `ul[(ue*numUeTx + ueTx)*numSc + sc]`.
- **`iqElem`** is a config field — default **`cf32`** (8 B, no repack for GPU receiver);
  **`ci16`** (4 B) optional for footprint. Both processes must agree (checked at handshake).
- **Phase-1 sizes** (2 cells × 16 UE = 32 UE, `numRx=4`, `numUeTx=2`, `numSc=3276`,
  `cf32`): DL slot ≈ **3.35 MB**, UL slot ≈ **1.68 MB**; `N=4` → ≈ 20 MB total — trivial
  in HBM. SU-MIMO leaves a UE's unscheduled subcarriers **zero-filled** in its full-band
  slot (Phase-1 simplicity; a packed/scheduled-SC-only layout is a later optimization).

## D.3 Control region (DPDK shared memory)

One hugepage memzone, name `"orca.vue.ctrl.v1"`, fixed layout:

```
┌────────────────────────────────────────────────────────────┐
│ Header   : magic, version, state, config (D.4)              │
│ Handshake: dlRing/ulRing mem handles, event handles (D.5)   │
│ dlDoorbell: rte_ring SPSC, ORCA → vUE  (DL slot ready)      │
│ dlReturn : rte_ring SPSC, vUE → ORCA  (DL slot consumed)    │
│ ulDoorbell: rte_ring SPSC, vUE → ORCA  (UL slot ready)      │
│ ulReturn : rte_ring SPSC, ORCA → vUE  (UL slot consumed)    │
│ Stats    : counters (produced/consumed/late per dir)        │
└────────────────────────────────────────────────────────────┘
```

**Doorbell descriptor** (compact, fits a cache line):

| Field | Size | Meaning |
|---|---|---|
| `slotIdx` | u16 | ring slot index (0..N-1) |
| `sfn` | u16 | system frame number |
| `slot` | u8 | slot within frame |
| `sym` | u8 | OFDM symbol |
| `flags` | u8 | bit0 = partial/zero-filled (late), … |
| `seq` | u32 | monotonic per-direction sequence (gap detection) |

Return rings carry just `{slotIdx, seq}` (credit). Rings are **lockless SPSC** (one
producer, one consumer per ring) — no locks on the hot path.

## D.4 Config block (agreed at handshake)

`{ numUe, numRx, numUeTx, numSc, iqElem, N_dl, N_ul, layoutId, cellCount }`. The vUE
**must** validate these against its own build and refuse to attach on mismatch (version or
config) rather than silently misinterpret memory.

## D.5 Bring-up handshake (ordered, once)

```
ORCA                                              vUE
────                                              ───
1. create memzone; write magic+version+config
   set state = INIT
2. cudaMalloc dlRing/ulRing
   create IPC events (D.6)
   cudaIpcGetMemHandle(dlRing/ulRing)
   cudaIpcGetEventHandle(events)
   write handles into Handshake block
   set state = ORCA_READY  ───────────────────►   3. poll magic/version/config
                                                     (refuse on mismatch)
                                                     cudaIpcOpenMemHandle(dlRing/ulRing)
                                                     cudaIpcOpenEventHandle(events)
                                                     set state = VUE_ATTACHED
4. observe VUE_ATTACHED  ◄──────────────────────
   begin steady state (D.7)
```

Timeout + retry on each side; a version/config mismatch is a hard stop (logged), never a
best-effort attach.

## D.6 Cross-process GPU synchronization

- **Events, not host flags, order GPU memory.** A doorbell on the host ring tells the
  consumer *a slot index is ready*, but the consumer **must** `cudaStreamWaitEvent` on the
  producer's IPC event before its kernels read that slot — a host flag alone does not
  guarantee the producer's GPU writes are visible.
- **Event set:** either one IPC event **per ring slot** (`dlProduced[N]`, `dlConsumed[N]`,
  UL likewise) — simplest, recommended — or a smaller pool with seq tagging. Created with
  `cudaEventInterprocess | cudaEventDisableTiming`.
- Control-ring messages are **host-side signaling/metadata only**.

## D.7 Per-symbol protocol (steady state)

**DL (ORCA → vUE):**
1. ORCA's CUDA graph writes `r_dl` into `dlRing[i]`.
2. ORCA `cudaEventRecord(dlProduced[i], orcaStream)`.
3. ORCA `rte_ring_enqueue(dlDoorbell, {i, sfn,slot,sym, flags, seq})`.
4. vUE `rte_ring_dequeue(dlDoorbell)` → `i`; `cudaStreamWaitEvent(vueStream, dlProduced[i])`;
   launch receiver kernels on `dlRing[i]`.
5. vUE `cudaEventRecord(dlConsumed[i], vueStream)`; `rte_ring_enqueue(dlReturn, {i, seq})`.
6. ORCA may reuse `dlRing[i]` only after it sees the `dlReturn` credit **and**
   `cudaStreamWaitEvent(orcaStream, dlConsumed[i])` (so the next write waits for the read).

**UL (vUE → ORCA):** symmetric, with `ulDoorbell` / `ulReturn` and `ulProduced/ulConsumed`.

**Backpressure / recycling.** A producer must not advance into a slot whose consumer credit
hasn't returned (N-deep credit window). If the consumer falls behind past the symbol
deadline, the producer applies the [Spec A §A.4](timing-and-deadlines.md) **drop policy**:
overwrite the oldest slot, set `flags.late`, bump the late counter — never block. The vUE
side is best-effort within the pipeline; ORCA never stalls the vDU-facing hot path on the
vUE.

## D.8 VueTransport API (both phases implement)

```
attach()/detach()                  // D.5 handshake
publishDl(slotIdx, meta)           // steps DL-2,3
reclaimDl() -> [slotIdx]           // step DL-6 (drain dlReturn + wait dlConsumed)
submitUl(slotIdx, meta)            // UL produce
pollUl() -> [(slotIdx, meta)]      // UL consume (+ wait ulProduced)
returnUl(slotIdx)                  // UL credit
```

The ORCA hot path / captured graph is unaware of the backend — it calls `publishDl` /
`pollUl`. **Phase-1 backend = CUDA-IPC + DPDK-shm** (this spec). **Phase-2 backend =
DPDK-over-NVLink** (ADR 0004 §3) swaps only buffer *placement* (HBM → host/coherent) and
*sharing* (CUDA IPC → DPDK/unified); the descriptors, rings, slot layout, and protocol
above are **unchanged**.

## D.9 Open / implementation-time

- `rte_ring` depth and hugepage size; busy-poll vs `eventfd` wake for ring consumers.
- Per-slot vs pooled IPC events (default: per-slot).
- Multi-cell slot indexing if DL/UL rings are split per cell vs one combined ring.
- ~~Confirm `numUeTx`~~ → **`numUeTx = 2`, `numRx = 4`** (UL slot `x_ul[U][2][numSc]`).
- Packed scheduled-SC-only slot layout (drops the SU-MIMO zero-fill) — a later optimization.
