# Spec A — Per-symbol timing & deadline model

**Status:** Settled design (source of truth). No implementation yet.
**Cadence:** per **OFDM symbol** (not per slot). **Target numerology: µ=1.**

Related: [ADR 0001 — hot-path synchronization](../decisions/0001-hot-path-synchronization.md).

## A.1 Time reference

All times are nanoseconds on a **PTP/SyncE clock disciplined to the vDU** (the
S-plane). From it derive a continuous symbol counter and an **air-time** reference
for every symbol — the instant the symbol is notionally "on air": when the vUE
should consider it received (DL) and when the vDU expects it (UL).

```
T_air(SFN, slot, sym) = T0 + slotIndex·T_slot + symStart[sym]
    slotIndex   = SFN · slotsPerFrame + slot
    slotsPerFrame = 10 · 2^µ        (10 ms frame)
    symStart[sym] = CP-adjusted cumulative symbol offset within the slot
```

`T0` = air-time of (SFN=0, slot=0, sym=0), bootstrapped from an S-plane sync message.

## A.2 Symbol period by numerology

| µ | SCS | Symbol period (incl. CP) | Notes |
|---|---|---|---|
| 0 | 15 kHz | ~71.4 µs | |
| **1** | **30 kHz** | **~35.7 µs** | **target** — typical 5G NR TDD |
| 2 | 60 kHz | ~17.8 µs | |
| 3 | 120 kHz | ~8.9 µs | FR2; out of scope |

The first symbol of every 0.5 ms boundary has a longer CP. Because the fronthaul
carries **frequency-domain IQ** (CP insertion stays on the O-RU side), CP variation
only shifts the per-symbol **packet arrival/departure deadline** — it does not change
the payload.

## A.3 Deadline budget

All offsets are relative to `T_air(s)`, in nanoseconds.

| Symbol | Meaning |
|---|---|
| `Tdl_arrival_max` | earliest a DL section may arrive before `T_air` |
| `Tdl_arrival_min` | latest "normal" arrival |
| `T_proc` | compute budget (precode + channel-apply + combine + noise) |
| `T_egress` | GPU → peer serialization + NIC |
| `T_margin` | jitter guard |
| `T_ul_offset` | vDU UL receive-window offset from `T_air` |

**DL reassembly deadline** — process whatever has arrived by:
```
D_r(s) = T_air(s) − T_egress − T_proc − T_margin
```

**UL deadline** — combine and deliver to the vDU by:
```
D_ul(s) = T_air(s) + T_ul_offset − T_proc − T_egress − T_margin
```

### Two budgets: throughput (rate) vs latency (deadline)

Per [ADR 0003](../decisions/0003-throughput-latency-pipeline.md) these are **separate
constraints** — the symbol period is a *rate*, the delivery deadline is a *latency*, and
the symbol pipeline (§A.5) decouples them. The single `20/3/5/7 µs` table is retired.

**Throughput budget — a rate.** The pipeline must complete one symbol every
`T_sym = 35.7 µs` (µ=1). The binding constraint is the **slowest pipeline stage**:

```
max(stage service time) ≤ T_sym = 35.7 µs
```

| Stage | Service-time ceiling | What binds it |
|---|---|---|
| ingest / reassembly | ≤ T_sym | vDU arrival rate; NIC/PCIe (vDU side only) |
| compute | ≤ T_sym | **HBM bandwidth for `H`** (ADR 0002 §6) — the real limit, not FLOPs |
| egress | ≤ T_sym | vDU side: NIC line rate; vUE side: HBM (in-box, ADR 0003 §5) |

Pipelining does **not** relax this: every symbol needs each stage, so the bandwidth wall
(ADR 0002 §6) is a *throughput* constraint, relaxed only by reducing `H`'s footprint
(Spec C) or replicating across GPUs (ADR 0003 §6).

**Latency budget — a deadline.** Each symbol must be delivered within `L_max`:

```
Σ stage latencies ≤ L_max
```

`L_max` is set by the vDU/vUE fronthaul timing tolerance (TBD); **working placeholder
`~70 µs ≈ 2·T_sym`, and strictly smaller than the real tolerance**. Representative
latencies (µ=1, multi-cell, `H` footprint resolved):

| Segment | Latency | Note |
|---|---|---|
| arrival / reassembly wait | ~35 µs | mostly *waiting* for the vDU's sections (pipelined) |
| compute | ~6 µs kernels (**~13–16 µs** incl. H2D + graph-launch, Spec E §E.13) | cold; ~5–10 µs steady (L2) |
| egress | ~5–10 µs | vDU side: NIC; vUE side: in-HBM |
| margin | ~7 µs | should become measured p99.9 jitter |
| **Σ** | **~60–65 µs cold / ~50 µs steady** | **< ~70 µs `L_max` → closes** (tighter cold; Spec E §E.13) |

> **`T_proc ≤ 3 µs` is retired.** It came from treating compute as the *residual* of one
> symbol period alongside ingest + egress — a category error. Compute gets up to a
> **full period** of service time (throughput) and contributes its *actual* ~6 µs to the
> **latency** sum (ADR 0003 §3). Every number above is **provisional** pending
> measurement on the target H100 + NIC and the Spec C `H`-layout decision. (At µ=3 the
> arrival wait must shrink — out of scope; see ADR 0001 "Revisit if…".)

### Multi-cell note

All cells are time-aligned to the same S-plane clock (ADR 0002 §1) → one symbol cadence
and one deadline across every cell. The **compute stage** covers all cells'
precode/channel/combine + interference summation in one captured-graph launch; with
interference it is the stage most at risk on the **throughput** clock, which is why the
interferer set is neighbor-limited by default (ADR 0002 §3, §6). Per-flow ingress/egress
parallelize across cells/NIC queues. Only the **vDU side** uses the NIC fronthaul; the
**vUE side is in-box** (in-HBM handoff, ADR 0003 §5).

## A.4 Drop policy (never stall)

A stalled symbol desyncs the real vDU; an errored symbol is just an error vector.
Therefore:

- **DL:** if the symbol is not fully covered (all PRBs present) by `D_r(s)` →
  **zero-fill the missing PRBs**, process what arrived, emit a `late_drop`
  telemetry event, advance.
- **UL:** missing vUE input by `D_ul(s)` → null / noise-only for that stream.

Never block the pipeline waiting for a late symbol.

## A.5 Reassembly buffering

- A ring of **N ≥ 3 symbol slots** so input(n+2) / compute(n+1) / output(n) overlap.
- Each slot holds one symbol's frequency-domain IQ plus a **PRB coverage bitmap**
  (one bit per PRB) and lifecycle state.
- **Reassembly key:** `(cell, direction, SFN, slot, sym)` — `cell` from the eAxC
  (Spec B). Sections accumulate by `(startPrb, numPrb)` into the bitmap; "complete" =
  full PRB coverage. One ring per `(cell, direction)`.
- Per-slot lifecycle: `Free → Filling → Ready → Computing → Egressing → Free`.

> **Synchronization note.** Per ADR 0001, the hot path is a **CUDA Graph**
> (host-staged ingress in Phase 1 — DOCA deferred, ADR 0007), so slot hand-off uses
> **stream ordering + one event per symbol**, not persistent-kernel doorbells or
> system-scope flags. The ring and
> coverage bitmap are orchestration state owned by the host ingest path.

## A.6 Phased reassembly

| | **v1 — whole symbol (first)** | **v2 — first-arrived-first (later)** |
|---|---|---|
| Trigger compute | coverage complete **or** `D_r(s)` reached | as each section lands |
| Granularity | one graph launch over the symbol | per-PRB-region partial passes |
| Latency | needs full arrival margin | minimal |
| Wire change | none | **none** — header already carries `sectionId/startPrb/numPrb` |

v2 is purely a scheduler change; the [packet format](fronthaul-packet-format.md)
supports section granularity from day one.
