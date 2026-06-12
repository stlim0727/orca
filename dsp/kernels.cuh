#pragma once
// Spec E §E.3-§E.7 — CUDA kernel launchers (K0–K5).
// Guards behind EMU_WITH_CUDA so this header is safe to include from any
// translation unit; callers that do not define the macro get nothing.

#ifdef EMU_WITH_CUDA

#include <cuda_runtime_api.h>
#include <cstdint>

#include "common/complex.hpp"
#include "common/dims.hpp"
#include "oru/oru_transport.hpp"  // Alloc struct

namespace orca {

// K2 split-K factor over tx (Spec E §E.12): 416 blocks saturate HBM on the cold
// symbol.  numTx must be divisible by kK2SplitK.  The caller allocates a scratch
// buffer of kK2ScratchElems cf32 for the partial sums (one buffer, reused — K2's
// two passes are stream-ordered, never concurrent across symbols).
constexpr uint32_t kK2SplitK      = 16;
constexpr uint32_t kK2ScratchElems =
    dims::C * dims::numRx * kK2SplitK * dims::numScP;
static_assert(dims::numTx % kK2SplitK == 0, "numTx divisible by kK2SplitK");

// K0 — ci16 → cf32 ingress (Spec E §E.7).  count = flat element count.
void launchK0(const ci16* d_in, cf32* d_out, uint32_t count,
              cudaStream_t stream = nullptr);

// K1 — DL precode (Spec E §E.4).
// d_precode = precodeBook[numBeams][numTx] cf32 (Spec E §E.2).
void launchK1(const cf32* d_x_dl, cf32* d_y, const cf32* d_precode,
              const Alloc* d_allocs, uint32_t numAllocs,
              cudaStream_t stream = nullptr);

// K2 — channel-apply DL, rx-in-thread + split-K over tx (Spec E §E.5/§E.12).
// Two stream-ordered passes: (1) k2Partial sums each tx-slice into d_scratch,
// (2) k2Reduce sums the kK2SplitK partials in fixed order, adds AWGN, writes
// r_dl.  The fixed reduction order keeps the output deterministic (Spec G §G.10),
// unlike an atomicAdd accumulation.  launchK2 also zero-fills r_dl first so
// unscheduled (u,sc) entries are 0, matching the CPU golden.
// d_H       = H_dl[C][U][numRx][numTx][numScP] half2c, active buffer.
// d_y       = y[C][numTx][numScP] cf32.
// d_r_dl    = r_dl[U][numRx][numScP] cf32 (output; zero-filled then written).
// d_victim  = victimMap[C][numScP] uint16_t (0xffff = unscheduled).
// d_doppler = doppler[C][U] cf32.
// d_scratch = kK2ScratchElems cf32 partial-sum buffer (caller-owned).
// symbolCtr keys the AWGN sampleIdx (= symbolCtr·numScP + sc) per the Philox
// noise contract (common/philox.hpp); must match the CPU golden's symbol index.
void launchK2(const half2c* d_H, const cf32* d_y, cf32* d_r_dl,
              const uint16_t* d_victim, const cf32* d_doppler, cf32* d_scratch,
              uint64_t symbolCtr, uint64_t noiseSeed, float noiseStd,
              cudaStream_t stream = nullptr);

// K3 — channel-apply UL, reciprocity from H_dl, RXT=8 rx tile (Spec E §E.6).
// d_x_ul      = x_ul[U][numUeTx][numScP] cf32.
// d_ulContrib = ulContrib[C][numScP] uint16_t (0xffff = none).
// d_ueTxToRx  = ueTxToRx[numUeTx] uint8_t (which H_dl rx index each UE-tx maps to).
void launchK3(const half2c* d_H, const cf32* d_x_ul, cf32* d_r_ul,
              const uint16_t* d_ulContrib, const cf32* d_doppler,
              const uint8_t* d_ueTxToRx,
              uint64_t symbolCtr, uint64_t noiseSeed, float noiseStd,
              cudaStream_t stream = nullptr);

// K4 — UL combine (Spec E §E.6).
// d_combine = combineBook[numBeams][numTx] cf32 (Spec E §E.2).
void launchK4(const cf32* d_r_ul, cf32* d_z, const cf32* d_combine,
              const Alloc* d_allocs, uint32_t numAllocs,
              cudaStream_t stream = nullptr);

// K5 — cf32 → ci16 egress pack (Spec E §E.7).  count = flat element count.
void launchK5(const cf32* d_in, ci16* d_out, uint32_t count,
              cudaStream_t stream = nullptr);

}  // namespace orca

#endif  // EMU_WITH_CUDA
