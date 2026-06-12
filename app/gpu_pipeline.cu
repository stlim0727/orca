#ifdef EMU_WITH_CUDA

#include "app/gpu_pipeline.hpp"
#include "dsp/kernels.cuh"
#include "scenario/victim_map.hpp"

#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

namespace orca {

// ── helpers ──────────────────────────────────────────────────────────────────

static bool chk(cudaError_t e, const char* tag) {
    if (e == cudaSuccess) return true;
    fprintf(stderr, "[GpuPipeline] %s: %s\n", tag, cudaGetErrorString(e));
    return false;
}

bool GpuPipeline::allocDev(void** p, size_t bytes) {
    return chk(cudaMalloc(p, bytes), "cudaMalloc");
}

void GpuPipeline::freeAll() {
    auto fdev = [](void* p) { if (p) cudaFree(p); };

    fdev(d_xDlRaw_);   d_xDlRaw_   = nullptr;
    fdev(d_xDl_);      d_xDl_      = nullptr;
    fdev(d_y_);        d_y_        = nullptr;
    fdev(d_rDl_);      d_rDl_      = nullptr;
    fdev(d_xUl_);      d_xUl_      = nullptr;
    fdev(d_rUl_);      d_rUl_      = nullptr;
    fdev(d_z_);        d_z_        = nullptr;
    fdev(d_zPacked_);  d_zPacked_  = nullptr;
    fdev(d_k2Scratch_); d_k2Scratch_ = nullptr;

    fdev(d_victim_);    d_victim_    = nullptr;
    fdev(d_ulContrib_); d_ulContrib_ = nullptr;
    fdev(d_allocs_);    d_allocs_    = nullptr;
    fdev(d_doppler_);   d_doppler_   = nullptr;
    fdev(d_ueTxToRx_);  d_ueTxToRx_  = nullptr;

    fdev(d_precodeBook_); d_precodeBook_ = nullptr;
    fdev(d_combineBook_); d_combineBook_ = nullptr;

    fdev(d_Hdl_[0]);  d_Hdl_[0]       = nullptr;
    fdev(d_Hdl_[1]);  d_Hdl_[1]       = nullptr;
    fdev(d_HdlActiveCell_); d_HdlActiveCell_ = nullptr;

    if (dlStream_)   { cudaStreamDestroy(dlStream_);   dlStream_   = nullptr; }
    if (ulStream_)   { cudaStreamDestroy(ulStream_);   ulStream_   = nullptr; }
    if (slowStream_) { cudaStreamDestroy(slowStream_); slowStream_ = nullptr; }
}

// ── create / destroy ─────────────────────────────────────────────────────────

bool GpuPipeline::create(uint32_t numBeams) {
    if (created_) return true;

    // Streams
    if (!chk(cudaStreamCreateWithFlags(&dlStream_,   cudaStreamNonBlocking), "dlStream"))   goto fail;
    if (!chk(cudaStreamCreateWithFlags(&ulStream_,   cudaStreamNonBlocking), "ulStream"))   goto fail;
    if (!chk(cudaStreamCreateWithFlags(&slowStream_, cudaStreamNonBlocking), "slowStream")) goto fail;

    // Per-slot ring buffers
    if (!allocDev((void**)&d_xDlRaw_,  sizeof(ci16)   * N * kXdlElems))  goto fail;
    if (!allocDev((void**)&d_xDl_,     sizeof(cf32)   * N * kXdlElems))  goto fail;
    if (!allocDev((void**)&d_y_,       sizeof(cf32)   * N * kYElems))    goto fail;
    if (!allocDev((void**)&d_rDl_,     sizeof(cf32)   * N * kRdlElems))  goto fail;
    if (!allocDev((void**)&d_xUl_,     sizeof(cf32)   * N * kXulElems))  goto fail;
    if (!allocDev((void**)&d_rUl_,     sizeof(cf32)   * N * kRulElems))  goto fail;
    if (!allocDev((void**)&d_z_,       sizeof(cf32)   * N * kZElems))    goto fail;
    if (!allocDev((void**)&d_zPacked_, sizeof(ci16)   * N * kZElems))    goto fail;

    // K2 split-K scratch (single buffer, reused across symbols — Spec E §E.12)
    if (!allocDev((void**)&d_k2Scratch_, sizeof(cf32) * kK2ScratchElems)) goto fail;

    // Per-symbol control
    if (!allocDev((void**)&d_victim_,    sizeof(uint16_t) * dims::C * dims::numScP)) goto fail;
    if (!allocDev((void**)&d_ulContrib_, sizeof(uint16_t) * dims::C * dims::numScP)) goto fail;
    if (!allocDev((void**)&d_allocs_,    sizeof(Alloc)    * dims::MAX_ALLOCS))        goto fail;
    if (!allocDev((void**)&d_doppler_,   sizeof(cf32)     * dims::C * dims::U))       goto fail;
    if (!allocDev((void**)&d_ueTxToRx_,  sizeof(uint8_t)  * dims::numUeTx))           goto fail;

    // Upload default ueTxToRx = {0, 1} (Spec E §E.6, ADR 0008)
    {
        const uint8_t def[dims::numUeTx] = {0, 1};
        chk(cudaMemcpy(d_ueTxToRx_, def, sizeof(def), cudaMemcpyHostToDevice), "ueTxToRx init");
    }

    // Resident beam codebook
    numBeams_ = numBeams;
    if (!allocDev((void**)&d_precodeBook_, sizeof(cf32) * numBeams * dims::numTx)) goto fail;
    if (!allocDev((void**)&d_combineBook_, sizeof(cf32) * numBeams * dims::numTx)) goto fail;

    // H_dl double buffer (Spec E §E.2)
    if (!allocDev((void**)&d_Hdl_[0], sizeof(half2c) * kHdlElems)) goto fail;
    if (!allocDev((void**)&d_Hdl_[1], sizeof(half2c) * kHdlElems)) goto fail;
    if (!allocDev((void**)&d_HdlActiveCell_, sizeof(half2c*)))      goto fail;

    // Initialise indirection cell to point at d_Hdl_[0]
    activeHdlIdx_    = 0;
    d_HdlActiveCopy_ = d_Hdl_[0];
    chk(cudaMemcpy(d_HdlActiveCell_, &d_HdlActiveCopy_, sizeof(half2c*),
                   cudaMemcpyHostToDevice), "indirection cell init");

    created_ = true;
    return true;

fail:
    freeAll();
    return false;
}

void GpuPipeline::destroy() {
    if (!created_) return;
    cudaDeviceSynchronize();
    freeAll();
    created_  = false;
    numBeams_ = 0;
}

// ── codebook & H_dl uploads ──────────────────────────────────────────────────

void GpuPipeline::uploadCodebook(const cf32* precode, const cf32* combine,
                                 uint32_t numBeams) {
    const size_t bytes = sizeof(cf32) * (size_t)numBeams * dims::numTx;
    chk(cudaMemcpy(d_precodeBook_, precode, bytes, cudaMemcpyHostToDevice), "precodeBook");
    chk(cudaMemcpy(d_combineBook_, combine, bytes, cudaMemcpyHostToDevice), "combineBook");
    numBeams_ = numBeams;
}

void GpuPipeline::uploadHdl(const half2c* H_host) {
    const size_t bytes = sizeof(half2c) * kHdlElems;
    // Load both copies so neither is stale (Spec E §E.10 note)
    chk(cudaMemcpy(d_Hdl_[0], H_host, bytes, cudaMemcpyHostToDevice), "uploadHdl[0]");
    chk(cudaMemcpy(d_Hdl_[1], H_host, bytes, cudaMemcpyHostToDevice), "uploadHdl[1]");
    // Active stays at d_Hdl_[0]; indirection cell already set in create()
    activeHdlIdx_    = 0;
    d_HdlActiveCopy_ = d_Hdl_[0];
    chk(cudaMemcpy(d_HdlActiveCell_, &d_HdlActiveCopy_, sizeof(half2c*),
                   cudaMemcpyHostToDevice), "uploadHdl cell");
}

void GpuPipeline::updateHdl(const half2c* H_host) {
    // Write into the back buffer on the slow stream, wait, then swap
    const int backIdx  = 1 - activeHdlIdx_;
    const size_t bytes = sizeof(half2c) * kHdlElems;

    chk(cudaMemcpyAsync(d_Hdl_[backIdx], H_host, bytes,
                        cudaMemcpyHostToDevice, slowStream_), "updateHdl async");
    chk(cudaStreamSynchronize(slowStream_), "updateHdl sync");

    // Publish: update device indirection cell and host mirror
    activeHdlIdx_    = backIdx;
    d_HdlActiveCopy_ = d_Hdl_[backIdx];
    chk(cudaMemcpy(d_HdlActiveCell_, &d_HdlActiveCopy_, sizeof(half2c*),
                   cudaMemcpyHostToDevice), "updateHdl cell");
}

// ── per-symbol uploads ────────────────────────────────────────────────────────

void GpuPipeline::uploadSymbol(uint32_t slotIdx,
                               const ci16* x_dl_host,
                               const Alloc* allocs, uint32_t numAllocs,
                               const cf32* doppler_host) {
    numAllocs_ = numAllocs;

    // H2D: x_dl_raw (ci16)
    const size_t xdlBytes = sizeof(ci16) * kXdlElems;
    chk(cudaMemcpyAsync(d_xDlRaw_ + slotIdx * kXdlElems, x_dl_host, xdlBytes,
                        cudaMemcpyHostToDevice, dlStream_), "x_dl_raw H2D");

    // Build DL victim map on host, then H2D
    buildVictimMap(allocs, numAllocs, 0, victimScratch_);
    chk(cudaMemcpyAsync(d_victim_, victimScratch_,
                        sizeof(uint16_t) * dims::C * dims::numScP,
                        cudaMemcpyHostToDevice, dlStream_), "victim H2D");

    // Build UL contributor map on host, then H2D
    buildVictimMap(allocs, numAllocs, 1, ulContribScratch_);
    chk(cudaMemcpyAsync(d_ulContrib_, ulContribScratch_,
                        sizeof(uint16_t) * dims::C * dims::numScP,
                        cudaMemcpyHostToDevice, dlStream_), "ulContrib H2D");

    // H2D: allocs
    if (numAllocs > 0)
        chk(cudaMemcpyAsync(d_allocs_, allocs, sizeof(Alloc) * numAllocs,
                            cudaMemcpyHostToDevice, dlStream_), "allocs H2D");

    // H2D: doppler rotors
    chk(cudaMemcpyAsync(d_doppler_, doppler_host,
                        sizeof(cf32) * dims::C * dims::U,
                        cudaMemcpyHostToDevice, dlStream_), "doppler H2D");
}

// ── stream-ordered kernel launches ───────────────────────────────────────────

void GpuPipeline::launchDl(uint32_t slotIdx, uint64_t symbolCtr,
                           float noiseStd, uint64_t noiseSeed) {
    // K0: ci16 → cf32  (reads d_xDlRaw_[slot], writes d_xDl_[slot])
    launchK0(d_xDlRaw_ + slotIdx * kXdlElems,
             d_xDl_    + slotIdx * kXdlElems,
             kXdlElems, dlStream_);

    // K1: DL precode  (reads d_xDl_[slot], writes d_y_[slot])
    launchK1(d_xDl_        + slotIdx * kXdlElems,
             d_y_          + slotIdx * kYElems,
             d_precodeBook_,
             d_allocs_, numAllocs_, dlStream_);

    // K2: channel-apply DL  (reads d_y_[slot], writes d_rDl_[slot])
    // Uses host-side active pointer for the stepping-stone path.
    launchK2(d_HdlActiveCopy_,
             d_y_       + slotIdx * kYElems,
             d_rDl_     + slotIdx * kRdlElems,
             d_victim_, d_doppler_, d_k2Scratch_,
             symbolCtr, noiseSeed, noiseStd, dlStream_);
}

void GpuPipeline::launchUl(uint32_t slotIdx, uint64_t symbolCtr,
                           float noiseStd, uint64_t noiseSeed) {
    // K3: channel-apply UL  (reads d_xUl_[slot], writes d_rUl_[slot])
    launchK3(d_HdlActiveCopy_,
             d_xUl_     + slotIdx * kXulElems,
             d_rUl_     + slotIdx * kRulElems,
             d_ulContrib_, d_doppler_, d_ueTxToRx_,
             symbolCtr, noiseSeed, noiseStd, ulStream_);

    // K4: UL combine  (reads d_rUl_[slot], writes d_z_[slot])
    launchK4(d_rUl_    + slotIdx * kRulElems,
             d_z_      + slotIdx * kZElems,
             d_combineBook_,
             d_allocs_, numAllocs_, ulStream_);

    // K5: cf32 → ci16  (reads d_z_[slot], writes d_zPacked_[slot])
    launchK5(d_z_       + slotIdx * kZElems,
             d_zPacked_ + slotIdx * kZElems,
             kZElems, ulStream_);
}

// ── read-back and accessors ───────────────────────────────────────────────────

void GpuPipeline::readUl(uint32_t slotIdx, ci16* z_host) {
    chk(cudaStreamSynchronize(ulStream_), "readUl sync");
    chk(cudaMemcpy(z_host,
                   d_zPacked_ + slotIdx * kZElems,
                   sizeof(ci16) * kZElems,
                   cudaMemcpyDeviceToHost), "readUl D2H");
}

cf32* GpuPipeline::d_rDlSlot(uint32_t slotIdx) const {
    return d_rDl_ + slotIdx * kRdlElems;
}

cf32* GpuPipeline::d_xUlSlot(uint32_t slotIdx) const {
    return d_xUl_ + slotIdx * kXulElems;
}

}  // namespace orca

#endif  // EMU_WITH_CUDA
