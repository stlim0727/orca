# ADR 0009 — Cell-count scaling: single-box ceiling and multi-box strategy

- **Status:** Accepted
- **Date:** 2026-06-12
- **Context tags:** scaling, multi-cell, bandwidth, memory, multi-box, partitioning, interconnect, µ=1
- **Builds on:** [ADR 0002 §6](0002-multi-cell-interference-mobility.md) (the `H`-bandwidth
  roofline), [ADR 0003 §6](0003-throughput-latency-pipeline.md) (cells-per-box as the cost
  lever; multi-box replication + partitioning), [ADR 0005](0005-su-mimo-phase1.md) (SU-MIMO).
- **Feeds:** [deferred-goals #3 (more cells)](../deferred-goals.md#more-cells),
  [#5 (multi-box)](../deferred-goals.md#multi-box), [Spec C (#2)](../deferred-goals.md#spec-c).

## Context

Phase 1 is fixed at **2 cells** (ADR 0005). Growing the cell count `C` is the primary cost
lever (ADR 0003 §6: maximize cells/box, minimize boxes), but the engineering changes are
sharply different **within one GPU box** versus **across boxes**, and the two regimes are
often conflated. This ADR separates them: **Part A** establishes the single-box ceiling and
the order in which limits bite; **Part B** plans the multi-box regime once that ceiling is
hit. It consolidates and extends the scaling notes scattered across ADR 0002 §6 and ADR 0003
§6 into one staged roadmap, and names the single decision that gates everything past a few
cells (Spec C).

All figures below are **µ=1, SU-MIMO, per-subcarrier FP16 `H`** (the Phase-1 model), so they
are the *optimistic* baseline — MU-MIMO (ADR 0005) multiplies the `H` read by `U_c` and is a
separate, harder axis, out of scope here.

---

# Part A — Single GPU box

The whole analysis in this part assumes **one GPU's HBM, L2, and compute**, with the full
cross-channel tensor resident locally (ADR 0002 §1). No inter-box anything.

## A.1 The binding constraint — `H` bandwidth scales as `C²` (all-to-all)

Channel-apply is **memory-bandwidth bound** (AI ≈ 2 FLOP/byte, ADR 0002 §6), so the question
is purely "when does the per-symbol `H` read hit HBM." Under **all-to-all** interference each
of `C` victim cells hears all `C` contributor cells, so the read is

```
H bytes/symbol = C² · numRx · numTx · numSc · b
              = C² · 4 · 64 · 3276 · 4 B  =  C² · 3.35 MB
BW(C) = C² · 3.35 MB / 35.7 µs
```

| `C` | `H` read/symbol | HBM % (3.35 TB/s) | per-symbol working set vs L2 (50 MB) |
|---|---|---|---|
| 2 | 13.4 MB | 11 % | 13 MB — **fits** |
| 3 | 30 MB | 25 % | 30 MB — **fits** |
| 4 | 54 MB | 45 % | 54 MB — **overflows** |
| 5 | 84 MB | 70 % | 84 MB — overflows |
| 6 | 121 MB | **101 % — wall** | 121 MB — overflows |

So **all-to-all SU per-SC FP16 saturates one H100's HBM at ≈ 6 cells**, and there is a
*second, earlier* cliff hidden in the right column (A.2). Compute is never the wall here
(~168 TFLOP/s at 7 cells is comfortable for the tensor/CUDA cores); only bandwidth is.

## A.2 The L2-residency cliff (≈ C=4) — the sneaky one

At C=2 the 13 MB per-symbol `H` working set **fits L2 (50 MB)**, and because scheduling is
stable over many symbols, the same lines are re-read → **steady state is L2-bound (~2 µs),
HBM nearly idle** (Spec E §E.9). The "0.38 TB/s" headline is the *cold* worst case, not the
steady state.

At **C=4 the working set (54 MB) exceeds nominal L2 capacity**, so full-symbol residency and
simple cross-symbol reuse **can no longer be relied on** (the exact behavior depends on
scheduling/tile order, set associativity, and other L2 users) → the design must assume close
to full HBM **every** symbol. The degradation is therefore **not smooth**: throughput is
comfortable to C=3, then **steps down around C=4** even though the cold bandwidth still looks
fine. Anything that shrinks the working set back under 50 MB (per-PRB-group `H`, A.3) restores
L2 residency — so that lever helps *twice* (peak BW and steady-state reuse).

## A.3 Mitigation stack (apply in order; the effects multiply)

1. **Neighbor-limit the contributor set — `C²` → `C·(K+1)`.** Real interference is
   neighbor-dominated; cap each victim to serving + `K` strongest interferers (the
   `scenario/association.hpp` machinery — serving + top-K by `pathlossDb`). The read becomes
   `C·(K+1)·3.35 MB`, **linear in `C`**:

   | scenario | `H` read/symbol | HBM % |
   |---|---|---|
   | C=7, K=3 (4 contributors) | 93.9 MB | 79 % |
   | C=8, K=3 | 107 MB | 90 % |

   This is the single biggest lever and the reason all-to-all is Phase-1-only. **Cost:** a
   fidelity compromise (weak interferers unmodeled).

2. **Spec C — shrink the per-link footprint** (independent of the `C` count, so it
   *multiplies* with (1)):
   - **Per-PRB-group `H`** (÷4 to ÷48 in the `sc` dimension): pushes the working set back into
     L2 (restores A.2) and cuts peak BW. **Cost:** loses sub-group frequency selectivity —
     acceptable only if the delay spread keeps the coherence bandwidth ≥ group width.
   - **Tap-domain apply** (`H[sc] = Σ_p g_p e^{-j2πf_sc τ_p}` from a few taps): tiny per-link
     tap gains reused across all `numSc` → AI rockets past the roofline ridge → genuinely
     **compute-bound** with *full* selectivity. **Cost:** a tap-apply kernel + committing a
     tap count `P`.

3. **Lower `H` precision** (INT8/BF16): ~2× only — never enough alone, but stacks.

**Spec C is the gating decision for any `C` past a few**, and it needs a **target
delay-spread / coherence-bandwidth** to pin per-PRB-group size or `P` — that input is the
real blocker, not the kernels.

## A.4 Secondary single-box limits (none bind before A.1/A.2)

- **Resident `H_dl` footprint** is also `C²` in cross-links: `C` cells × `U_total = C·16` UEs
  → `16·C²·3.37 MB`. C=2 → 215 MB (×2 double-buffer); C=7 → 2.6 GB; C=16 → 14 GB — fits 80 GB,
  so **not the first wall**, but storing **only active (neighbor-limited) links** collapses it
  back toward linear `16·C·(K+1)·3.37 MB`. The slow-plane ray→`H` expansion cost (Spec G §G.8)
  scales with it.
- **Captured-graph batch extent** (ADR 0002 §6): the CUDA graph bakes grid dims for the
  *configured max* `K`, not the instantaneous count, so a handover that transiently needs more
  contributors must stay within the captured bound — `K` becomes a capture-time constant sized
  conservatively, not a runtime variable.
- **North fronthaul ingress** is linear in `C·layers`; the host-staged H2D (~3 GB/s at 2
  cells, ADR 0007) eventually approaches PCIe → the documented trigger to reinstate **DOCA
  GPUNetIO + GPUDirect** (an `OruTransport` backend swap). Mechanical, not a wall.
- **Offline table size** (Spec G §G.11) scales as `cells·gridpoints·P`, host-resident
  (7-cell 1 km² ≈ 4.5 GB) — comfortable far past the bandwidth ceiling.

## A.5 Single-box ceiling and the order limits bite

| `C` | regime |
|---|---|
| ≤ 3 | all-to-all, per-SC FP16, L2-resident — *Phase-1-shaped, no new work* |
| 4 | **L2 cliff** → need per-PRB-group (Spec C) to keep steady state |
| 5–6 | all-to-all hits the **HBM wall** → neighbor-limiting becomes mandatory |
| 8–12 | neighbor-limited + Spec C; footprint + contributor management dominate effort |
| > ~12 | even neighbor-limited + Spec C exhausts one GPU → **go multi-box (Part B)** |

The exact "> ~12" boundary depends on the chosen `K`, PRB-group size / `P`, and precision —
i.e. on the Spec C decision. Everything in Part A keeps the **hot path on one GPU**, where the
synchronization model (ADR 0001) is unchanged.

---

# Part B — Multiple GPU boxes

Reached only when Part A's mitigations are exhausted: per-symbol `H` BW, resident memory, or
compute can no longer fit one GPU even neighbor-limited with Spec C. This regime is
qualitatively harder because it reintroduces **cross-box coordination inside the symbol
deadline** — exactly what the single-box design (ADR 0001) was built to avoid.

## B.1 The governing principle — partition so each UE's interferer set co-resides

Assign cells to boxes so that **every UE's neighbor-limited contributor set lives entirely on
one box** (ADR 0003 §6). Then each box runs an **independent ORCA instance** over its cell
subset with **zero per-symbol cross-box traffic** — pure replication, linear in box count.

This is feasible *because* of neighbor-limiting (A.3): interference is short-range, so a
**geographic partition** (cluster physically adjacent cells onto the same box) naturally
co-locates each UE's dominant interferers. The interference graph, once top-K-pruned, has
short edges → clean min-cut partitions exist for most real layouts.

A UE is **owned by its serving cell's box**; that box terminates the UE's vUE handoff and the
serving cell's fronthaul flow. The partition is a placement decision computed **offline** from
geometry + the association layer's `pathlossDb` ranking — never on the hot path.

## B.2 What must cross a box boundary, and why it is expensive

Partitioning is rarely perfectly clean — some **boundary cells** have an interferer set that
straddles the cut. Two ways to handle a boundary UE on box A whose contributor cell `c'` lives
on box B:

- **Halo / ghost cells (preferred where the interconnect allows):** box A stores the
  *cross-link* `H[c'][u]` for the few ghost contributors (small under neighbor-limiting) and
  **receives `c'`'s precoded antenna-domain `y_{c'}` from box B every symbol**, then folds it
  into u's channel-apply locally. Per ghost cell the exchange is
  `numTx·numSc·cf32 = 64·3276·8 B ≈ 1.68 MB/symbol ≈ 47 GB/s` — *antenna-domain* volume, the
  expensive kind.
- **Full cross-box exchange (the worst case):** if cells cannot be partitioned at all, every
  box needs every interfering cell's `y` → all-to-all `y` exchange across boxes. Infeasible at
  scale.

Exchanging the already-reduced contribution `H[c'][u]·y_{c'}` instead of `y` does **not** help:
the producing box B would need A's UE cross-link `H[c'][u]`, doubling `H` storage and coupling
the boxes — so the halo model keeps `H` on the consumer (A) and ships `y`.

## B.3 Interconnect tiers — the partition strategy is set by the link

The per-symbol halo volume (~47 GB/s per ghost cell, B.2) against a **35.7 µs** deadline
divides cleanly by interconnect:

| Tier | Topology | Per-symbol cross-box exchange? | Partition requirement |
|---|---|---|---|
| **1** | Multi-GPU, **single node, NVLink/NVSwitch** (~900 GB/s, low latency) | **Tolerable** — halo `y` exchange fits with margin | partition for memory/BW balance; boundary cells may ghost |
| **2** | Multi-node, **InfiniBand/RoCE** (~50 GB/s/dir, higher latency) | **Not at antenna-domain volume** — one ghost cell's `y` (~47 GB/s) consumes essentially the whole link before protocol/sync/return overhead | **must partition cleanly** (interferer sets fully co-resident); no per-symbol exchange |
| **3** | Un-partitionable interference across nodes | required, all-to-all `y` over IB | **infeasible within the deadline** — the boundary of the approach |

So the design rule is: **NVLink single-node can carry halo exchange (Tier 1); multi-node IB
must avoid per-symbol exchange entirely (Tier 2)**; Tier 3 (dense long-range interference
spanning nodes) is outside what this architecture supports in real time and must be handled by
*accepting* a fidelity cut (drop the cross-node interferers) rather than exchanging.

## B.4 Replication vs interference-coupled scaling

- **Disjoint cell clusters** (no shared interferers across the cut): independent pipelines,
  **linear** throughput scaling by box count, no coupling. This is the target and the cheapest
  multi-box mode (ADR 0003 §6 replication).
- **Interference-coupled** (boundary cells): pay the halo exchange (Tier 1) or forbid the cut
  (Tier 2). The latency budget `L_max` tightens by the exchange round-trip, and the symbol-loop
  synchronization gains an inter-box barrier — the hardest sync problem in the project.

## B.5 Cross-cutting multi-box concerns

- **S-plane clock across boxes.** All boxes must share one PTP/SyncE reference (ADR 0002 risk
  #2, now across boxes) so every box's per-symbol deadline is aligned. Time skew between boxes
  directly eats the halo-exchange margin.
- **Determinism.** Cross-box exchange reopens inter-box ordering; the golden-model
  bit-reproducibility (Spec G §G.10) holds per-box but a cross-box halo sum introduces an
  arrival-order dependence unless the reduction order is fixed — design the halo fold as a
  fixed-order accumulate, like K2's split-K reduce.
- **Fronthaul fan-out.** Each box owns its cells' vDU flows; the vDU PTP domain spans all
  boxes. North NIC budget is per-box (its cells' layer IQ).

---

## Decision (the staged strategy)

1. **Stay single-box as far as possible** (Part A) — it is the cost lever and keeps the ADR
   0001 hot path intact. Order of work as `C` grows: per-PRB-group `H` at the L2 cliff (C≈4) →
   neighbor-limiting at the HBM wall (C≈5–6) → both + active-link storage to ~tens of cells.
   **Spec C is the prerequisite** and is gated on a target delay spread.
2. **Go multi-box only when one GPU is exhausted** (Part B), and then **partition to make each
   UE's interferer set co-resident** (B.1) so the common case is zero-exchange replication.
3. **Match the partition to the interconnect** (B.3): NVLink single-node may ghost boundary
   cells with per-symbol `y` halo exchange; multi-node IB must partition cleanly with no
   per-symbol exchange. Never default to cross-node antenna-domain exchange.
4. **Treat un-partitionable dense interference as a fidelity decision, not a transport
   problem** — drop or coarsen the cross-node interferers rather than exchange them in real
   time.

## Consequences

### Positive
- One curve and one roadmap for "more cells," with the single-box and multi-box regimes and
  their *different* engineering clearly separated.
- The biggest lever (Spec C) and the biggest risk (cross-box exchange) are named explicitly,
  with quantitative thresholds (C≈4 L2 cliff, C≈6 HBM wall, the interconnect tiers).
- The partitioning principle turns multi-box into mostly-replication for realistic
  neighbor-limited scenarios.

### Negative / costs
- Past C≈4 the design **requires Spec C**, which is itself gated on a delay-spread target not
  yet chosen — so single-box scaling is blocked on that input, not on code.
- Multi-box with boundary interference reintroduces an **inter-box per-symbol barrier** and
  tightens `L_max` — the hardest synchronization in the project, and only NVLink-class
  interconnect makes it tractable.
- Tier 3 (dense cross-node interference) is explicitly *not* supported in real time; some
  large dense scenarios are only emulable with reduced interference fidelity.

## Rejected alternatives
- **One-box-per-cell from the start** (already rejected in ADR 0002 §1): more horizontally
  scalable but requires cross-box IQ/state exchange even at small `C` — far more complex sync
  for no benefit while everything fits one GPU.
- **Default cross-node antenna-domain `y` exchange** to "just scale": the ~47 GB/s/ghost-cell
  volume does not fit a 35.7 µs deadline over IB; partitioning-to-avoid is mandatory, not
  optional.
- **Scaling MU-MIMO and cell count together:** compounds the `H`-read blow-up (`C²·U_c`); MU is
  a separate deferral (ADR 0005) and must not be entangled with cell-count scaling.

## Revisit if…
- A **target delay spread / coherence bandwidth** is chosen → pin Spec C (per-PRB-group size /
  tap count `P`), which sets the real single-box ceiling and unblocks C > 4.
- A concrete **multi-GPU platform** is selected (NVLink node vs IB cluster) → fix the partition
  strategy to its tier (B.3) and design the halo-exchange path or the clean-partition
  constraint accordingly.
- The **interference model** changes (e.g. long-range/large-scale interferers become required)
  → the short-range-partition assumption (B.1) weakens and more scenarios fall to Tier 2/3.
