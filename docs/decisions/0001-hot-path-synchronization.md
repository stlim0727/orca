# ADR 0001 — Hot-path synchronization model

- **Status:** Accepted
- **Date:** 2026-06-09
- **Context tags:** real-time, CUDA, DOCA GPUNetIO, fronthaul, µ=1

## Context

The emulator processes frequency-domain IQ on a **per-OFDM-symbol** cadence between
the ORU module (custom fronthaul to a real vDU) and the vUE module. For each symbol
the GPU must run `precode → channel-apply → combine` and move data in/out, **within
the symbol deadline** (`T_air − T_egress − T_proc − T_margin`; see
`orchestr/timing.hpp`).

The open question: with the hot path on the GPU, how do we synchronize
**(1) input arrival into GPU memory, (2) kernel execution, (3) output departure**?

An earlier scaffold sketched a **persistent kernel** (launched once, `while(true)`
spinning on a doorbell). Review found that model:

- conflated two incompatible synchronization schemes (it referenced stream
  *events*, which only order *launched* kernels, inside a kernel that is never
  relaunched);
- required **system-scope** atomics / `__threadfence_system()` for any host- or
  NIC-driven flag handshake — the scaffold used default **device-scope**
  `atomicExch`, which is incorrect for a host/NIC producer;
- could not "rebind kernel arguments per slot" as written (a persistent kernel's
  args are fixed at launch);
- permanently occupies SMs and generally wants a cooperative launch.

That complexity is only justified when launch overhead is fatal.

## Decisive fact: we target µ=1

| | µ=1 (chosen) | µ=3 (out of scope) |
|---|---|---|
| Symbol period | **35.7 µs** | 8.9 µs |
| Tolerates a ~2 µs graph launch? | Yes, with huge margin | No |
| Tolerates a ~5–10 µs kernel launch? | Yes, but jittery | No |
| Persistent kernel justified? | **No** | Yes |

At µ=1 the actual compute is sub-microsecond; launch overhead is not existential.
The persistent-kernel model buys us nothing we need and costs us correctness
hazards we don't want.

## Decision

Adopt a **CUDA Graph hot path with CPU-controlled DOCA GPUNetIO** and
**indirection-cell double buffering** for the slow-plane stores.

1. **CUDA Graph per-symbol pipeline.** Capture
   `gather/reassemble → precode → channel-apply → combine → pack` **once** as a
   graph; replay with a single `cudaGraphLaunch` (~1–2 µs) per symbol. Inter-stage
   ordering is expressed as **graph node dependencies** — no flags, no fences.

2. **CPU-controlled DOCA GPUNetIO.** The NIC DMAs packets directly into GPU memory
   (GPUDirect RDMA — zero copy preserved). The **host** observes receive
   completions, tracks reassembly via the per-symbol coverage bitmap
   (`orchestr/symbol_ring.hpp`), and at "coverage complete OR deadline" issues the
   `cudaGraphLaunch`, then the DOCA TX. GPU-*controlled* (kernel-initiated) receive
   is **not** used.

3. **Synchronization of the three steps** uses the standard CUDA model:
   - **input ↔ compute:** stream ordering; host launches the graph only once the
     symbol's data is resident in GPU memory.
   - **compute stages:** graph node dependencies (`channel` after `precode`,
     `combine` after `channel`).
   - **compute ↔ output:** **one** `cudaEvent` per symbol that the egress path
     waits on before issuing DOCA TX (or DOCA send as the graph's tail node).

4. **Indirection-cell double buffering.** A captured graph bakes in kernel pointer
   arguments, which collides with swapping the active CIR/weight buffer. Resolve by
   keeping a **stable device cell** (`ChannelBuffer** d_activeCir`,
   `WeightBuffer** d_activeW`). Kernels dereference the cell at runtime
   (`const ChannelBuffer* H = *d_activeCir;`); the slow plane publishes by an
   **atomic single-pointer write into the cell**. The graph stays immutable; the
   kernel always reads the current (possibly one-generation-stale, which is the
   intended slow/fast decoupling) buffer.

5. **Stepping stone.** First implement plain **stream-ordered launches** (simplest
   route to a working loopback), measure jitter, then wrap the identical kernels in
   a captured graph. Kernels do not change between the two.

## Consequences

### Positive
- Synchronization reduces to the textbook stream/graph model — auditable and
  well-supported.
- Eliminates: persistent `while(true)` kernels, the doorbell/token protocol,
  system-scope flag accessors, `__threadfence_system`, cooperative launch, and
  SM-occupancy juggling.
- GPUDirect zero-copy ingress/egress is retained.
- Slow/fast channel decoupling is preserved (now via the indirection cell).

### Negative / costs
- One `cudaGraphLaunch` + one DOCA TX issue **per symbol** on a host thread. At
  35.7 µs this is comfortable; it would **not** scale to µ=3 — see "Revisit."
- Graph capture must be re-done if pipeline **topology** changes (not for per-symbol
  data, which flows through fixed buffers + the indirection cell).

### Scaffold changes implied
- Remove `dsp::Doorbell`, `launchPersistent*`, and the persistent kernels in
  `dsp/precode.cu`, `dsp/channel_apply.cu`, `dsp/combine.cu`.
- **Keep** the `*Once` kernels — they become graph nodes unchanged.
- `channel/cir_store.hpp`, `estim/weights.hpp`: replace "swap a host-held pointer"
  with "write a device indirection cell"; kernels dereference the cell.
- `app/main.cpp`: add graph capture/launch + per-symbol event-gated egress;
  drop persistent-kernel launch and doorbell wiring.
- Update `docs/MILESTONES.md` hot-path invariants accordingly.

## Rejected alternatives

- **Persistent kernel + GPU-controlled DOCA.** Correct for µ=3 but unjustified
  complexity and correctness hazards at µ=1.
- **Plain per-symbol stream launches (as the end state).** Works at µ=1 but higher,
  jitterier host overhead than a captured graph; kept only as the stepping stone.
- **`cudaGraphExecKernelNodeSetParams` per launch** to update buffer pointers.
  Works, but per-symbol host param-patching; the indirection cell is cleaner and
  keeps the hot path host-light.

## Revisit if…

- The target moves to **µ=2/µ=3** (FR2) — reopen persistent kernel +
  GPU-controlled DOCA, where eliminating launch overhead becomes mandatory.
- Per-symbol host issue cost or jitter is measured to be a material fraction of the
  budget even at µ=1.
