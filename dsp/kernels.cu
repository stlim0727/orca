// Spec E §E.3-§E.12 — K0–K5 CUDA hot-path kernels (Phase 1, SU-MIMO, 2 cells).
// All kernels use static worst-case grids (§E.8) for CUDA-graph compatibility:
//   K0/K5: flat over element count    K2/K3: (C, sc-tiles [, rx-tiles])
//   K1/K4: (MAX_ALLOCS, sc-tiles)     per-thread guards handle idle blocks.
//
// TODO (performance): K2 split-K over tx (SPLITK=16, §E.12) for cold-symbol
// HBM saturation.  Functional correctness does not depend on it.

#ifdef EMU_WITH_CUDA

#include <cuda_fp16.h>
#include <curand_kernel.h>

#include "common/complex.hpp"
#include "common/dims.hpp"
#include "oru/oru_transport.hpp"
#include "dsp/kernels.cuh"

namespace orca {

// ─────────────────────────────────────────────────────────────────────────────
// Internal constants
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int      kTile      = 256;   // threads per block / sc per tile
static constexpr int      kRxt       = 8;     // K3 rx_ru registers per thread
static constexpr uint16_t kNoVictim  = 0xffffu;

// ─────────────────────────────────────────────────────────────────────────────
// Device helpers
// ─────────────────────────────────────────────────────────────────────────────

// Load a half2c coefficient and convert to float re/im.
__device__ __forceinline__ void loadH(const half2c* __restrict__ H,
                                       unsigned long long idx,
                                       float& re, float& im) {
    const half2c h = H[idx];
    re = __half2float(__ushort_as_half(h.re.bits));
    im = __half2float(__ushort_as_half(h.im.bits));
}

// Saturating round to int16 with IEEE round-to-nearest-even (§E.7 / complex.hpp).
__device__ __forceinline__ int16_t satRound16(float x) {
    if (::isnan(x)) return 0;
    float r = rintf(x);
    r = fminf(fmaxf(r, -32768.f), 32767.f);
    return static_cast<int16_t>(r);
}

// Philox per-sample AWGN seeded exactly as the CPU golden model (awgn.hpp).
// subseq = (dir << 32) | (ue << 16) | rx;  offset = sc.
__device__ __forceinline__ void awgnSample(uint64_t seed, uint64_t subseq,
                                            uint32_t sc, float noiseStd,
                                            float& re, float& im) {
    curandStatePhilox4_32_10_t st;
    curand_init(seed,
                static_cast<unsigned long long>(subseq),
                static_cast<unsigned long long>(sc), &st);
    float2 n = curand_normal2(&st);
    re = n.x * noiseStd;
    im = n.y * noiseStd;
}

// ─────────────────────────────────────────────────────────────────────────────
// K0 — ci16 → cf32 ingress convert (Spec E §E.7)
// gridDim = (ceil(count/kTile), 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k0Kernel(const ci16* __restrict__ in,
                          cf32*        __restrict__ out,
                          uint32_t count) {
    uint32_t i = blockIdx.x * static_cast<uint32_t>(kTile) + threadIdx.x;
    if (i >= count) return;
    out[i] = cf32{static_cast<float>(in[i].re), static_cast<float>(in[i].im)};
}

void launchK0(const ci16* d_in, cf32* d_out, uint32_t count, cudaStream_t s) {
    if (!count) return;
    k0Kernel<<<(count + kTile - 1) / kTile, kTile, 0, s>>>(d_in, d_out, count);
}

// ─────────────────────────────────────────────────────────────────────────────
// K1 — DL precode (Spec E §E.4)
// gridDim = (MAX_ALLOCS, ceil(numScP/kTile))   — static for CUDA graph (§E.8)
// W gathered once per block into shared memory; x_dl pre-loaded into registers.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k1Kernel(const cf32* __restrict__ x_dl,
                          cf32*        __restrict__ y,
                          const cf32* __restrict__ precode,
                          const Alloc* __restrict__ allocs,
                          uint32_t numAllocs) {
    using namespace dims;
    if (blockIdx.x >= numAllocs) return;
    const Alloc a   = allocs[blockIdx.x];
    const uint32_t rank = a.rank;
    if (!rank || rank > rankMax || a.dir != 0) return;

    // Shared W[tx][l] = precode[beamId[l]][tx]  (2 KB, §E.4)
    __shared__ float W_re[numTx][rankMax];
    __shared__ float W_im[numTx][rankMax];
    const uint32_t tid = threadIdx.x;
    if (tid < numTx * rank) {
        const uint32_t tx  = tid / rank;
        const uint32_t l   = tid % rank;
        const cf32 src = precode[static_cast<uint32_t>(a.beamId[l]) * numTx + tx];
        W_re[tx][l] = src.re;
        W_im[tx][l] = src.im;
    }
    __syncthreads();

    const uint32_t scBase = static_cast<uint32_t>(a.scStart);
    const uint32_t sc     = scBase + blockIdx.y * static_cast<uint32_t>(kTile) + tid;
    if (sc >= scBase + a.scLen || sc >= numSc) return;
    const uint32_t c = a.cell;

    // Pre-load x_dl for this sc (all layers → registers; reused across numTx loop)
    float x_re[rankMax], x_im[rankMax];
    for (uint32_t l = 0; l < rank; ++l) {
        const cf32 x = x_dl[(c * rankMax + l) * numScP + sc];
        x_re[l] = x.re;
        x_im[l] = x.im;
    }

    for (uint32_t tx = 0; tx < numTx; ++tx) {
        float acc_re = 0.f, acc_im = 0.f;
        for (uint32_t l = 0; l < rank; ++l) {
            acc_re += W_re[tx][l] * x_re[l] - W_im[tx][l] * x_im[l];
            acc_im += W_re[tx][l] * x_im[l] + W_im[tx][l] * x_re[l];
        }
        y[(c * numTx + tx) * numScP + sc] = cf32{acc_re, acc_im};
    }
}

void launchK1(const cf32* d_x_dl, cf32* d_y, const cf32* d_precode,
              const Alloc* d_allocs, uint32_t numAllocs, cudaStream_t s) {
    if (!numAllocs) return;
    dim3 grid(dims::MAX_ALLOCS, (dims::numScP + kTile - 1) / kTile);
    k1Kernel<<<grid, kTile, 0, s>>>(d_x_dl, d_y, d_precode, d_allocs, numAllocs);
}

// ─────────────────────────────────────────────────────────────────────────────
// K2 — channel-apply DL (Spec E §E.5, variant (a): rx in-thread)
// gridDim = (C, ceil(numScP/kTile))   — static for CUDA graph (§E.8)
//
// Each block owns one (cell c, sc-tile).  Per thread (one sc), loop over all
// contributor cells c' and all tx to build acc[numRx].
//
// NOTE: If two cells schedule the same UE at the same sc (unusual in SU-MIMO),
// both blocks write r_dl[u][...][sc] concurrently.  Both compute the same sum,
// so the result is correct; the race is benign under this precondition.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k2Kernel(const half2c*   __restrict__ H,
                          const cf32*     __restrict__ y,
                          cf32*            __restrict__ r_dl,
                          const uint16_t* __restrict__ d_victim,
                          const cf32*     __restrict__ d_doppler,
                          uint64_t noiseSeed, float noiseStd) {
    using namespace dims;
    const uint32_t c  = blockIdx.x;
    const uint32_t sc = blockIdx.y * static_cast<uint32_t>(kTile) + threadIdx.x;
    if (sc >= numSc) return;
    const uint16_t v = d_victim[c * numScP + sc];
    if (v == kNoVictim) return;
    const uint32_t u = v;

    // Initialise accumulators with AWGN (dir=0, subseq=(u<<16)|rx)
    float acc_re[numRx], acc_im[numRx];
    for (uint32_t rx = 0; rx < numRx; ++rx) {
        const uint64_t subseq = (static_cast<uint64_t>(u) << 16) | rx;
        awgnSample(noiseSeed, subseq, sc, noiseStd, acc_re[rx], acc_im[rx]);
    }

    // Accumulate all contributor cells c' (Spec E §E.5 outer Σ)
    for (uint32_t c2 = 0; c2 < C; ++c2) {
        const cf32 dopp = d_doppler[c2 * U + u];
        for (uint32_t tx = 0; tx < numTx; ++tx) {
            // Load y once per (c2, tx); reuse for numRx rx values (variant a)
            const cf32 yv = y[(c2 * numTx + tx) * numScP + sc];
            for (uint32_t rx = 0; rx < numRx; ++rx) {
                float H_re, H_im;
                loadH(H,
                      (((static_cast<unsigned long long>(c2) * U + u) * numRx + rx)
                       * numTx + tx) * numScP + sc,
                      H_re, H_im);
                // partial = H * y
                const float p_re = H_re * yv.re - H_im * yv.im;
                const float p_im = H_re * yv.im + H_im * yv.re;
                // acc += dopp * partial
                acc_re[rx] += dopp.re * p_re - dopp.im * p_im;
                acc_im[rx] += dopp.re * p_im + dopp.im * p_re;
            }
        }
    }

    for (uint32_t rx = 0; rx < numRx; ++rx)
        r_dl[(u * numRx + rx) * numScP + sc] = cf32{acc_re[rx], acc_im[rx]};
}

void launchK2(const half2c* d_H, const cf32* d_y, cf32* d_r_dl,
              const uint16_t* d_victim, const cf32* d_doppler,
              uint64_t noiseSeed, float noiseStd, cudaStream_t s) {
    dim3 grid(dims::C, (dims::numScP + kTile - 1) / kTile);
    k2Kernel<<<grid, kTile, 0, s>>>(d_H, d_y, d_r_dl, d_victim, d_doppler,
                                     noiseSeed, noiseStd);
}

// ─────────────────────────────────────────────────────────────────────────────
// K3 — channel-apply UL, reciprocity from H_dl (Spec E §E.6)
// gridDim = (C, numTx/kRxt, ceil(numScP/kTile))   — static (§E.8)
//
// Each block owns (cell c, rx_ru tile of kRxt, sc-tile).  kRxt=8 limits the
// per-thread register file to 8 cf32 accumulators.
//
// Reciprocity (§E.6):
//   H_ul[u][c][rx_ru][ueTx][sc] = H_dl[c][u][ ueTxToRx[ueTx] ][rx_ru][sc]
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k3Kernel(const half2c*   __restrict__ H,
                          const cf32*     __restrict__ x_ul,
                          cf32*            __restrict__ r_ul,
                          const uint16_t* __restrict__ d_ulContrib,
                          const cf32*     __restrict__ d_doppler,
                          const uint8_t*  __restrict__ ueTxToRx,
                          uint64_t noiseSeed, float noiseStd) {
    using namespace dims;
    const uint32_t c         = blockIdx.x;
    const uint32_t rxru_base = blockIdx.y * static_cast<uint32_t>(kRxt);
    const uint32_t sc        = blockIdx.z * static_cast<uint32_t>(kTile) + threadIdx.x;
    if (sc >= numSc) return;

    // Initialise kRxt accumulators with AWGN
    // For UL: dir=1, "ue"=rx_ru, "rx"=c  →  subseq=(1<<32)|(rx_ru<<16)|c
    float acc_re[kRxt], acc_im[kRxt];
    for (int r = 0; r < kRxt; ++r) {
        const uint32_t rx_ru = rxru_base + r;
        const uint64_t subseq = (static_cast<uint64_t>(1) << 32) |
                                  (static_cast<uint64_t>(rx_ru) << 16) | c;
        awgnSample(noiseSeed, subseq, sc, noiseStd, acc_re[r], acc_im[r]);
    }

    const uint16_t v = d_ulContrib[c * numScP + sc];
    if (v != kNoVictim) {
        const uint32_t u    = v;
        const cf32     dopp = d_doppler[c * U + u];
        for (uint32_t ueTx = 0; ueTx < numUeTx; ++ueTx) {
            const uint32_t rxIdx = ueTxToRx[ueTx];          // H_dl rx dimension
            const cf32 xv = x_ul[(u * numUeTx + ueTx) * numScP + sc];
            for (int r = 0; r < kRxt; ++r) {
                const uint32_t rx_ru = rxru_base + r;
                float H_re, H_im;
                // H_dl[c][u][rxIdx][rx_ru][sc]  (tx dim ↔ rx_ru, Spec E §E.6)
                loadH(H,
                      (((static_cast<unsigned long long>(c) * U + u) * numRx + rxIdx)
                       * numTx + rx_ru) * numScP + sc,
                      H_re, H_im);
                const float p_re = H_re * xv.re - H_im * xv.im;
                const float p_im = H_re * xv.im + H_im * xv.re;
                acc_re[r] += dopp.re * p_re - dopp.im * p_im;
                acc_im[r] += dopp.re * p_im + dopp.im * p_re;
            }
        }
    }

    for (int r = 0; r < kRxt; ++r)
        r_ul[(c * numTx + rxru_base + r) * numScP + sc] = cf32{acc_re[r], acc_im[r]};
}

void launchK3(const half2c* d_H, const cf32* d_x_ul, cf32* d_r_ul,
              const uint16_t* d_ulContrib, const cf32* d_doppler,
              const uint8_t* d_ueTxToRx,
              uint64_t noiseSeed, float noiseStd, cudaStream_t s) {
    dim3 grid(dims::C,
              dims::numTx / kRxt,
              (dims::numScP + kTile - 1) / kTile);
    k3Kernel<<<grid, kTile, 0, s>>>(d_H, d_x_ul, d_r_ul, d_ulContrib,
                                     d_doppler, d_ueTxToRx, noiseSeed, noiseStd);
}

// ─────────────────────────────────────────────────────────────────────────────
// K4 — UL combine (Spec E §E.6)
// gridDim = (MAX_ALLOCS, ceil(numScP/kTile))   — static (§E.8)
// Comb gathered once per block into shared memory.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k4Kernel(const cf32*  __restrict__ r_ul,
                          cf32*         __restrict__ z,
                          const cf32*  __restrict__ combine,
                          const Alloc* __restrict__ allocs,
                          uint32_t numAllocs) {
    using namespace dims;
    if (blockIdx.x >= numAllocs) return;
    const Alloc a    = allocs[blockIdx.x];
    const uint32_t rank = a.rank;
    if (!rank || rank > rankMax || a.dir != 1) return;

    // Shared Comb[l][rx]  (2 KB, §E.6)
    __shared__ float Comb_re[rankMax][numTx];
    __shared__ float Comb_im[rankMax][numTx];
    const uint32_t tid = threadIdx.x;
    if (tid < rank * numTx) {
        const uint32_t l  = tid / numTx;
        const uint32_t rx = tid % numTx;
        const cf32 src = combine[static_cast<uint32_t>(a.beamId[l]) * numTx + rx];
        Comb_re[l][rx] = src.re;
        Comb_im[l][rx] = src.im;
    }
    __syncthreads();

    const uint32_t scBase = static_cast<uint32_t>(a.scStart);
    const uint32_t sc     = scBase + blockIdx.y * static_cast<uint32_t>(kTile) + tid;
    if (sc >= scBase + a.scLen || sc >= numSc) return;
    const uint32_t c = a.cell;

    for (uint32_t l = 0; l < rank; ++l) {
        float acc_re = 0.f, acc_im = 0.f;
        for (uint32_t rx = 0; rx < numTx; ++rx) {
            const cf32 rv = r_ul[(c * numTx + rx) * numScP + sc];
            acc_re += Comb_re[l][rx] * rv.re - Comb_im[l][rx] * rv.im;
            acc_im += Comb_re[l][rx] * rv.im + Comb_im[l][rx] * rv.re;
        }
        z[(c * rankMax + l) * numScP + sc] = cf32{acc_re, acc_im};
    }
}

void launchK4(const cf32* d_r_ul, cf32* d_z, const cf32* d_combine,
              const Alloc* d_allocs, uint32_t numAllocs, cudaStream_t s) {
    if (!numAllocs) return;
    dim3 grid(dims::MAX_ALLOCS, (dims::numScP + kTile - 1) / kTile);
    k4Kernel<<<grid, kTile, 0, s>>>(d_r_ul, d_z, d_combine, d_allocs, numAllocs);
}

// ─────────────────────────────────────────────────────────────────────────────
// K5 — cf32 → ci16 egress pack (Spec E §E.7)
// gridDim = (ceil(count/kTile), 1, 1)
// ─────────────────────────────────────────────────────────────────────────────

__global__ void k5Kernel(const cf32* __restrict__ in,
                          ci16*        __restrict__ out,
                          uint32_t count) {
    uint32_t i = blockIdx.x * static_cast<uint32_t>(kTile) + threadIdx.x;
    if (i >= count) return;
    out[i] = ci16{satRound16(in[i].re), satRound16(in[i].im)};
}

void launchK5(const cf32* d_in, ci16* d_out, uint32_t count, cudaStream_t s) {
    if (!count) return;
    k5Kernel<<<(count + kTile - 1) / kTile, kTile, 0, s>>>(d_in, d_out, count);
}

}  // namespace orca

#endif  // EMU_WITH_CUDA
