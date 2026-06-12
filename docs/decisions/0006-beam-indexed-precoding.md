# ADR 0006 — Beam-indexed precoding/combining; SRS deferred

- **Status:** Accepted
- **Date:** 2026-06-09
- **Context tags:** precoding, combining, beamforming, codebook, beam_id, SRS, phasing
- **Refines:** the "both precoding modes" framing in
  [architecture.md](../architecture.md) and MILESTONES Stage 2/5.
- **Feeds:** [deferred-goals.md](../deferred-goals.md) (SRS / GPU-computed precoding).

## Context

Precoding "on behalf of the RU" needs weights. The earlier design carried **two modes**:
vDU-supplied **explicit `W` matrices** (C-plane) and **GPU-computed** weights
(ZF/MMSE/SVD estimated from uplink **SRS**, on the slow plane via cuSOLVER).

Phase 1 adopts a simpler, codebook-based model and **defers SRS entirely**.

## Decision

### 1. Precoding via a resident beam codebook

A **codebook of precoding vectors indexed by `beam_id`** is **resident in GPU memory**,
loaded in advance (like the `H` table):

```
precodeBook[beam_id] → complex vector over numTx (64)         # one beamforming vector
```

It is small — e.g. `numBeams · numTx · b` (1024 beams · 64 · 4 B ≈ 256 KB).

### 2. `beam_id` supplied by the vDU at runtime

The vDU provides `beam_id` per resource (per section / PRB-group, per layer) at runtime
through the **C-plane control**. The emulator **gathers** the precoding vector(s) by
`beam_id` to assemble `W` (`64 × rank`) for that resource, then applies the existing
precode `y = W·x`. No weights are transmitted as matrices and none are computed on the
fly.

### 3. UL combining is symmetric

Receive combining uses a **resident combining codebook** indexed by `beam_id` (from the
vDU's UL grant); combine reduces `64 Rx → rank` ("precombining on behalf of the RU").

### 4. SRS / GPU-computed weights are deferred

SRS-based channel estimation and ZF/MMSE/SVD weight computation move to a later phase
(→ [deferred-goals.md → GPU-computed precoding](../deferred-goals.md#gpu-precoding)).
This removes **cuSOLVER** and the slow-plane weight-estimation path from Phase 1; the
`estim/` module is essentially dormant in Phase 1.

### 5. Consolidate the vDU per-symbol control

The SU-MIMO **scheduling/allocation map** (ADR 0005) and `beam_id` are the **same C-plane
payload**: per section, the vDU provides `{PRB range, UE id, beam_id(s)}`. **Spec B
C-plane** therefore carries **`beam_id` (a small index)** rather than explicit `W`
matrices. (An explicit-`W` variant remains possible but is not the Phase-1 path.)

## Consequences

### Positive
- Phase-1 precoding is a **tiny gather + GEMV** — no SRS, no cuSOLVER, no slow-plane weight
  computation. Lighter hot path and a much smaller C-plane (index vs matrix).
- **Realistic** — mirrors O-RAN grid-of-beams / `beamId` in C-plane section types.
- The beam codebook is resident and static, consistent with the "all coefficients resident
  in GPU memory in advance" Phase-1 model.

### Negative / costs
- **Codebook fidelity:** precoding is limited to the predefined beam set (grid of beams),
  not arbitrary per-UE/reciprocity-based precoding. A modeling compromise.
- The beam codebook must be **defined and loaded** — a new Phase-1 input/artifact.
- Adaptive precoding (channel-matched, reciprocity) is unavailable until SRS returns.

### Doc impact
- **architecture.md** — precoding/combining table → beam_id codebook (Phase 1), SRS
  deferred; `estim/` module note.
- **Spec B §B.5** — C-plane carries `beam_id` (+ UE id, PRB range); explicit-`W` optional.
- **MILESTONES** — Stage 2 = beam_id codebook precode; Stage 5 = UL combine via codebook
  (SRS removed).
- **deferred-goals.md** — update the GPU-computed-precoding entry (SRS deferred here).
- **CLAUDE.md** — locked decision + next ADR 0007.

## Revisit if…
- **Reciprocity / channel-matched precoding** is required (e.g. MU-MIMO pairing, ADR 0005
  / deferred-goals) → re-enable **SRS estimation + GPU-computed ZF/MMSE/SVD** and the
  cuSOLVER weight path.
- A **non-codebook** (continuous) precoder is needed → revisit the explicit-`W` C-plane
  variant or on-GPU weight synthesis.
