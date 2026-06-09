# Spec B — ORU fronthaul packet format

**Status:** Settled design (source of truth). No implementation yet.

This is the **ORU fronthaul packet format** spoken on the **north (vDU) interface** of
**ORCA**. ORCA terminates it **as the O-RU** — i.e. it covers the
RU role (fronthaul termination + precoding/combining on behalf of the RU). It is the only
NIC-crossing interface; the **south (vUE) interface uses DPDK shared memory**, not this
format (ADR 0004).

The format carries **frequency-domain IQ** (7.2x-style, custom framing) between the vDU
and ORCA. Carrying frequency-domain IQ keeps FFT/iFFT off the hot path; the
CIR→frequency-response transform happens only at the slow CIR-update rate.

Related: [Spec A — timing](timing-and-deadlines.md),
[ADR 0001 — synchronization](../decisions/0001-hot-path-synchronization.md).

## B.1 Transport

- **L2/L3:** Ethernet + **UDP/IPv4** (RoCEv2-friendly) so **DOCA GPUNetIO +
  GPUDirect RDMA** can DMA payloads straight into GPU memory. A raw-Ethernet
  variant is an option to shave the IP/UDP header.
- **One UDP flow per eAxC** (encoded antenna-carrier / spatial stream) → NIC RSS
  fans flows across receive queues.
- Multi-byte header fields are **network byte order** on the wire.

## B.2 Header — 20 bytes, fixed

Every field is **byte-aligned** (no cross-byte bitfields) so GPU/host parsing is
trivial. (This refines an earlier 16-byte sketch that needed cross-byte packing.)

| Offset | Field | Size | Purpose |
|---|---|---|---|
| 0 | `ver_msgtyp` | u8 | `ver:4` (hi) \| `msgtyp:4` (lo) |
| 1 | `dir_cmp` | u8 | `dir:4` (hi) \| `cmp:4` (lo) |
| 2 | `iqWidth` | u8 | bits per I or Q (16, 12, BFP mantissa…) |
| 3 | `reserved0` | u8 | |
| 4 | `eAxC` | u16 | encoded antenna-carrier / spatial-stream id |
| 6 | `sectionId` | u8 | which section within this symbol |
| 7 | `numSections` | u8 | total sections expected for the symbol (0 = unknown) |
| 8 | `sfn` | u16 | system frame number |
| 10 | `slot` | u8 | slot within frame |
| 11 | `sym` | u8 | OFDM symbol within slot (0..13) |
| 12 | `startPrb` | u16 | first PRB in this section |
| 14 | `numPrb` | u16 | PRB count in this section |
| 16 | `seqNum` | u16 | per-eAxC sequence number (loss detection) |
| 18 | `udCompParam` | u16 | BFP shared exponent / scale |

### Enumerations

- `msgtyp`: `0 = U-plane`, `1 = C-plane (weights)`, `2 = S-plane (sync)`, `3 = Telemetry`
- `dir`: `0 = DL`, `1 = UL`
- `cmp`: `0 = int16`, `1 = BFP`, `2 = int12`, `3 = reserved`
- `ver`: current = `1`

### Reassembly / scheduling key

`(cell, dir, sfn, slot, sym)` keys a symbol — `cell` derived from the eAxC
`{BandSector, CC_ID}` (multi-cell, above); `(startPrb, numPrb)` marks PRB coverage
(273 PRB fits in 10 bits, u16 has headroom). See [Spec A §A.5](timing-and-deadlines.md).

## B.3 eAxC encoding

16-bit `eAxC = {DU_Port, BandSector, CC_ID, RU/UE_Port}`, bit widths configurable
at deployment. **Default 4 / 2 / 2 / 8 = 16 bits**, packed MSB→LSB in that order.
Addresses Tx ports (precode) and UE Rx ports (channel-apply) alike.

### Multi-cell addressing

The single GPU box hosts **all cells** (one fronthaul flow set per vDU; see
[ADR 0002 §1](../decisions/0002-multi-cell-interference-mobility.md)). A **cell** is
identified by the `{BandSector, CC_ID}` portion of the eAxC (the `DU_Port`/`RU_Port`
sub-fields then address antennas/UE-ports within that cell). If more cells are needed
than `BandSector·CC_ID` can encode, widen those sub-fields at deployment — no header
change, since `eAxC` is a fixed u16. The receive path therefore demultiplexes
`(cell, antenna/port)` directly from the eAxC, and the reassembly key (below) is scoped
per cell.

## B.4 U-plane payload (`msgtyp = 0`)

Frequency-domain IQ for `numPrb` PRBs starting at `startPrb` → `numPrb · 12`
subcarriers, complex:

```
[ I0 Q0 | I1 Q1 | ... ]   each I/Q = iqWidth bits, layout per `cmp`
```

- **int16** (`cmp=0`, default): 4 bytes/SC — simplest, bit-exact validation.
- **BFP** (`cmp=1`): block floating point, shared exponent in `udCompParam` per
  section — ~half the bytes; matters at high line rate.

## B.5 C-plane payload (`msgtyp = 1`) — allocation + beam_id

**Phase 1 ([ADR 0006](../decisions/0006-beam-indexed-precoding.md)):** the C-plane carries
the per-resource **allocation + `beam_id`**, not weight matrices. Precoding/combining
vectors live in a **resident beam codebook**; the hot path gathers them by `beam_id`. This
also carries the SU-MIMO scheduling map (ADR 0005). Per section:

| Field | Size | |
|---|---|---|
| `prbStart` | u16 | first PRB of this allocation |
| `numPrb` | u16 | PRB count |
| `ueId` | u16 | scheduled UE (for `H`/grid lookup, SU-MIMO) |
| `numLayers` | u8 | rank for this resource (≤ 4) |
| `dir` | u8 | DL (precode) / UL (combine) |
| `beamId[numLayers]` | u16[] | one beam index per layer → codebook lookup |

A C-plane message for symbol `s` must arrive before `D_r(s)`. `beam_id` → vector gathering
happens on the hot path; the codebook itself is loaded once at startup.

> **Deferred:** an explicit-`W` variant (full `numTx · numLayers` `ci16` matrix per
> PRB-group) remains possible for non-codebook / SRS-derived precoding, but is not the
> Phase-1 path. See [deferred-goals](../deferred-goals.md#gpu-precoding).

## B.6 S-plane (`msgtyp = 2`)

Lightweight sync/heartbeat carrying the sender's PTP timestamp + `sfn/slot/sym`.
Used to (a) verify clock lock to the vDU and (b) bootstrap `T0` (Spec A §A.1).

## B.7 Telemetry (`msgtyp = 3`)

Out-of-band counters / late-drop events (see Spec A §A.4). Format TBD at
implementation time.

## B.8 Why this format

- **Section granularity is native** → v2 streaming reassembly needs no wire change.
- **`seqNum` + coverage bitmap** → loss/late detection feeds the drop policy.
- **`eAxC` per flow** → clean NIC RSS fan-out and per-UE/per-antenna addressing.
- **`cmp` / `iqWidth`** → start bit-exact in int16, switch to BFP later without
  touching the pipeline.
