# AGENT.md — development handoff

**Purpose.** This file hands the ORCA implementation off to the next agent/developer working
in a **proper build environment**. Read [CLAUDE.md](CLAUDE.md) and the `docs/` first (the
design is the source of truth), then this file for *where we are and what to do next*.

---

## Where things stand (2026-06-11)

- **Design is complete; there is no code yet.** The source tree was intentionally removed —
  `docs/` is the deliverable so far (architecture, Spec A/B/D/E/F/G, ADR 0001–0008).
- **We are starting implementation at MILESTONES Stage 1** (loopback / identity transform).
- **A Stage-1 plan was reviewed and approved** (embedded in full below).
- **Toolchain choice locked:** **C++17**, CUDA/DOCA behind OFF-able CMake flags
  (`EMU_WITH_CUDA`, `EMU_WITH_DOCA`).

## Why this handoff exists — environment blocker

The machine where planning happened (Windows 11, `c:\Users\stk.lim\Downloads\claude-prj1\orca`)
has **no C++ build toolchain**: no `cmake`, no compiler (MSVC / MinGW / clang), no package
manager (winget/choco/scoop), no Python. WSL is installed but has **no Linux distro**. So code
can be *written* there but **not built or tested**.

**The next agent should work in an environment that can build C++17 (+ optionally CUDA 12.x).**
Recommended: the Linux target box (or WSL Ubuntu), which also matches the real deployment
target. The Stage-1 host config needs only a C++17 compiler + CMake — no GPU, no DOCA, no DPDK.

## Immediate next action

Implement **sub-stage 1a** and get a **green host-config test run**, then proceed 1b→1f:

1. Top-level `CMakeLists.txt` (C++17; `EMU_WITH_CUDA`/`EMU_WITH_DOCA` options default OFF;
   CTest enabled).
2. `common/` headers: `dims.hpp`, `complex.hpp`, `layout.hpp`, `symbol_id.hpp`.
3. `tests/test_layout.cpp`.
4. Build + run:
   ```bash
   cmake -B build -DEMU_WITH_CUDA=OFF -DEMU_WITH_DOCA=OFF
   cmake --build build -j
   ctest --test-dir build --output-on-failure
   ```
   Green = layout strides/alignment + `ci16↔cf32` round-trip pass. Then move to 1b.

Intended top-level module layout (from architecture.md): `common/ fh/ orchestr/ dsp/ channel/
estim/ scenario/ oru/ vue/ app/ tests/` — plus a separate `oru_process/` program (ADR 0007).

---

## Approved Stage-1 plan (embedded, self-contained)

### Strategy
Two build configs from day one:
- **Host config** (`EMU_WITH_CUDA=OFF`, `EMU_WITH_DOCA=OFF`) — everything in 1a–1e + a
  host-memcpy identity path; builds and unit-tests with no GPU/NIC.
- **Target config** (CUDA on, DOCA off-deferred) — GPU identity path + real DPDK/CUDA-IPC
  backends; built/run on the Linux H100.

Identity hot path = **K0 (`ci16→cf32`) → [no K1–K4] → K5 (`cf32→ci16`)**. Host config: CPU
converts. Target config: Spec E §E.7 kernels under a **stream-ordered launch** (CUDA-graph
capture deferred per ADR 0001 §5).

The real Phase-1 backends (DPDK shm control + CUDA IPC HBM bulk) are isolated behind two
interfaces — `OruTransport` (Spec F §F.8) and `VueTransport` (Spec D §D.8). Build a
**host-only, single-process loopback backend first**, then swap in Linux/CUDA backends with
**no change to ORCA hot-path code**.

### Sub-stages (each builds + tests before the next)

**1a — `common/` foundations** *(host-only)*
- `common/dims.hpp` — Phase-1 dims, **cell dimension present from the start**: `C=2, U=32,
  numTx=64, numRx=4, numUeTx=2, numSc=3276, numScP=3296, rankMax=4, N_ring=4, MAX_ALLOCS=512`.
- `common/complex.hpp` — `cf32{float re,im}`, `ci16{int16 re,im}`, host `half2` shim,
  `ci16↔cf32` converts (saturating round for K5) per Spec E §E.7.
- `common/layout.hpp` — flat row-major `idx(...)` macros for `H_dl, x_dl, y, r_dl, x_ul,
  r_ul, z` **exactly** per Spec E §E.11; `numScP = roundUp(numSc,32)`; `sc` innermost on every
  tensor.
- `common/symbol_id.hpp` — `(cell, dir, sfn, slot, sym)` key + continuous symbol counter.
- `tests/test_layout.cpp` — assert row strides (H_dl 13184 B, cf32 tensors 26368 B), 128-B row
  alignment, `roundUp(3276,32)==3296`, `ci16↔cf32` round-trip.

**1b — `orchestr/` symbol model + jitter** *(host-only)*
- `orchestr/timing.hpp` — `T_air(sfn,slot,sym)` (Spec A §A.1), `D_r`/`D_ul` deadlines (§A.3).
- `orchestr/symbol_ring.hpp` — N≥3 slot ring, lifecycle
  `Free→Filling→Ready→Computing→Egressing→Free`, PRB coverage bitmap, key
  `(cell,dir,sfn,slot,sym)` (Spec A §A.5); **drop policy** (zero-fill missing PRBs, advance,
  `late_drop` — Spec A §A.4).
- `orchestr/jitter.hpp` — per-symbol timestamp + p50/p99/p99.9 histogram (Stage-1 deliverable).
- `tests/test_ring.cpp`, `tests/test_timing.cpp`.

**1c — Transport interfaces + host loopback backends** *(host-only)*
- `oru/oru_transport.hpp` — `OruTransport`: `attach/detach, pollDl, returnDl, publishUl,
  reclaimUl` (Spec F §F.8) + doorbell descriptor (Spec F §F.4) + `Alloc`/`allocBlock`
  (Spec E §E.2 / Spec F §F.3).
- `vue/vue_transport.hpp` — `VueTransport`: `attach/detach, publishDl, reclaimDl, submitUl,
  pollUl, returnUl` (Spec D §D.8) + doorbell descriptor (Spec D §D.3).
- `oru/oru_loopback.*`, `vue/vue_loopback.*` — in-process backends using the **exact**
  descriptor schemas (Linux backend = drop-in swap). Shared SPSC ring helper in `common/`.
- `tests/test_transport_loopback.cpp` — produce/consume each direction, credit recycling,
  backpressure → drop (never block).

**1d — `fh/` Spec B framing** *(host-only — high value, no hardware)*
- `fh/fh_header.hpp` — 20-byte fixed header pack/unpack (Spec B §B.2), network byte order;
  enums `msgtyp/dir/cmp/ver`.
- `fh/eaxc.hpp` — `eAxC={DU_Port,BandSector,CC_ID,RU/UE_Port}` (default 4/2/2/8); `cell` from
  `{BandSector,CC_ID}` (Spec B §B.3).
- `fh/uplane.hpp` — U-plane ci16 payload (de)serialize per `(startPrb,numPrb)`.
- `fh/cplane.hpp` — C-plane alloc+`beam_id` parse → `Alloc` (Spec B §B.5).
- `tests/test_fh.cpp` — bit-exact round-trips.

**1e — ORU process (separate program) + test vDU stub**
- `oru_process/` — standalone program (not part of ORCA): terminates Spec B over a software
  socket / in-memory transport (kernel/DPDK on Linux; **DOCA deferred**, ADR 0007),
  reassembles per symbol, parses C-plane → `allocBlock`, relays via `OruTransport` producer
  side (Spec F §F.6).
- `tests/vdu_stub.*` — synthetic vDU: emits DL symbols (known IQ pattern), verifies UL
  loopback returns them bit-exact.

**1f — ORCA `app/` identity hot path + vUE stub** *(host-config end-to-end)*
- `app/main.cpp` — wire `OruTransport` (north) + `VueTransport` (south); per symbol:
  `pollDl → K0 convert → [identity] → publish to vUE → pollUl → K5 convert → publishUl`; drive
  jitter harness. Stream-ordered on target; host memcpy on host config.
- `dsp/convert.{hpp,cu/cpp}` — K0/K5 converts (CUDA kernel on target; CPU on host) behind one
  signature.
- `vue/vue_stub.*` — minimal vUE: consume DL slot, copy to UL slot (identity), signal per
  Spec D §D.7.
- `tests/test_e2e_identity.cpp` — full host-config pipeline; assert **vDU-in == vDU-out**
  bit-exact; print jitter percentiles.

**After Stage 1 (target box):** real backends as swaps only (interfaces unchanged) —
`oru/oru_dpdk_hostshm.*` (DPDK shm + `cudaHostRegister` + H2D/D2H, Spec F),
`vue/vue_cudaipc.*` (CUDA IPC mem/event handles, Spec D §D.5/§D.6), and CUDA-graph capture of
the identity path (ADR 0001).

### Invariants to honor (do not regress)
- **Cell dimension in every tensor layout from the start** (MILESTONES) — `C` carried in
  `dims` and all `idx()` macros even while Stage 1 runs effectively single-flow.
- **`sc` innermost** on every hot tensor (Spec E §E.1) — fixed by `layout.hpp`.
- **Never stall on a late symbol** — zero-fill + advance (Spec A §A.4) lives in the ring.
- **Transport seams are real** — ORCA hot path calls only `OruTransport`/`VueTransport` APIs,
  never a concrete backend, so the Linux swap touches no ORCA code.
- **Bulk never crosses the control ring** — descriptors/indices only (Spec D §D.1).

### Decisions intentionally deferred within Stage 1
- **DPDK vs POSIX-shm+eventfd** control plane (Spec D §D.9) — loopback backend is neutral;
  pick when building the Linux backend.
- **CUDA graph vs stream-ordered** — start stream-ordered (ADR 0001 §5); capture later.
- **Real vDU↔ORU socket transport** (AF_XDP/DPDK/kernel) — software/in-memory until Linux
  bring-up.
- **DOCA/GPUDirect** — deferred (ADR 0007); host-staged H2D path only.

### Verification
- **Host config (any C++17 box / WSL Ubuntu):**
  `cmake -B build -DEMU_WITH_CUDA=OFF -DEMU_WITH_DOCA=OFF && cmake --build build -j &&
  ctest --test-dir build --output-on-failure`. Green = layout strides/alignment, FH
  round-trips, ring lifecycle/drop, transport loopback, and **end-to-end identity
  (vDU-in == vDU-out)** all pass, with jitter percentiles printed.
- **Target config (Linux H100):** add `-DEMU_WITH_CUDA=ON`; same e2e test with CUDA K0/K5
  converts and stream-ordered launch; measure real per-symbol jitter on the GPU.

---

## Notes for the next agent
- Nothing has been written to the source tree yet — you start from a clean `common/`.
- This `AGENT.md` and the approved plan are the contract; if you deviate, note why (and add an
  ADR for any *new significant decision* — ADR 0009 is next).
- Update [docs/MILESTONES.md](docs/MILESTONES.md) Stage-1 status as sub-stages land.
