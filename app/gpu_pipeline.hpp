#pragma once
// ORCA GPU pipeline (ADR 0001 §5 stepping stone): stream-ordered K0–K5 launches
// over all Spec E §E.2 device buffers, double-buffered H_dl with an indirection
// cell (ADR 0001 §4), and upload helpers for per-symbol control data.
//
// Usage outline:
//   GpuPipeline pipe;
//   pipe.create(numBeams);
//   pipe.uploadCodebook(precode, combine, numBeams);
//   pipe.uploadHdl(H_initial);   // loads both double-buffer copies
//   // per symbol:
//   pipe.uploadSymbol(slot, x_dl_host, allocs, numAllocs, doppler);
//   pipe.launchDl(slot);         // K0 → K1 → K2 on dlStream_
//   // ... vUE reads d_rDlSlot(slot) via CUDA IPC ...
//   pipe.launchUl(slot);         // K3 → K4 → K5 on ulStream_
//   pipe.readUl(slot, z_host);   // sync + D2H z_packed
//   // slow plane:
//   pipe.updateHdl(H_new);       // writes back buffer, swaps indirection cell
//
// TODO: captureGraph() — wrap the stream-ordered launches into a cudaGraph for
// the ADR 0001 §1 final hot path (no topology change, only the launch mechanism).

#ifdef EMU_WITH_CUDA

#include <cuda_runtime_api.h>
#include <cstdint>

#include "common/complex.hpp"
#include "common/dims.hpp"
#include "oru/oru_transport.hpp"

namespace orca {

class GpuPipeline {
  public:
    GpuPipeline()  = default;
    ~GpuPipeline() { destroy(); }

    // Allocate all device buffers and CUDA streams.
    // numBeams: size of the precode/combine codebooks.  Returns false on error.
    bool create(uint32_t numBeams = 1024);
    void destroy();
    bool created() const { return created_; }

    // Upload resident beam codebook from host (once at startup).
    // precode, combine: [numBeams][numTx] cf32
    void uploadCodebook(const cf32* precode, const cf32* combine, uint32_t numBeams);

    // Upload H_dl into both double-buffer copies so neither generation is stale.
    // H_host: H_dl[C][U][numRx][numTx][numScP] half2c  (kHdlElems half2c)
    void uploadHdl(const half2c* H_host);

    // Slow-plane H_dl update: cudaMemcpy H_host → back buffer (on slowStream_),
    // then write the back-buffer pointer into the device indirection cell.
    // The hot path sees the new buffer on the next symbol launch.
    void updateHdl(const half2c* H_host);

    // Per-symbol upload: H2D x_dl_raw, build and H2D victim/ulContrib maps,
    // H2D allocs and doppler rotors.  All copies go on dlStream_ so the kernel
    // launches that follow (launchDl/launchUl) see the data by stream ordering.
    // doppler: C*U cf32  (d_doppler[c][u] rotor)
    void uploadSymbol(uint32_t slotIdx,
                      const ci16* x_dl_host,          // C*rankMax*numScP ci16
                      const Alloc* allocs, uint32_t numAllocs,
                      const cf32* doppler_host);

    // Stream-ordered DL: K0 (ci16→cf32) → K1 (precode) → K2 (channel apply).
    // Reads d_xDlRaw_[slotIdx], d_precodeBook_, d_Hdl_active_ (host copy).
    // Writes d_rDlSlot(slotIdx).
    // symbolCtr keys the per-symbol AWGN draw (Philox contract, common/philox.hpp).
    void launchDl(uint32_t slotIdx, uint64_t symbolCtr,
                  float noiseStd = 1.f, uint64_t noiseSeed = 12345);

    // Stream-ordered UL: K3 (channel apply) → K4 (combine) → K5 (cf32→ci16).
    // Reads d_xUlSlot(slotIdx), d_Hdl_active_ (host copy).
    // Writes d_zPackedSlot(slotIdx); call readUl() to D2H.
    void launchUl(uint32_t slotIdx, uint64_t symbolCtr,
                  float noiseStd = 1.f, uint64_t noiseSeed = 12345);

    // Synchronise ulStream_ and D2H z_packed → z_host.
    // z_host: C*rankMax*numScP ci16
    void readUl(uint32_t slotIdx, ci16* z_host);

    // Device pointers for CUDA IPC export to the vUE (Spec D §D.2).
    cf32* d_rDlSlot(uint32_t slotIdx) const;
    cf32* d_xUlSlot(uint32_t slotIdx) const;

    // CUDA streams (exposed for external event recording, Spec D §D.6).
    cudaStream_t dlStream()   const { return dlStream_;   }
    cudaStream_t ulStream()   const { return ulStream_;   }
    cudaStream_t slowStream() const { return slowStream_; }

  private:
    bool allocDev(void** p, size_t bytes);
    void freeAll();

    // ── Buffer sizes ──────────────────────────────────────────────────────────
    static constexpr uint32_t N  = dims::N_ring;

    // Per-slot counts (element count, not byte count)
    static constexpr uint32_t kXdlElems  = dims::C * dims::rankMax  * dims::numScP;
    static constexpr uint32_t kYElems    = dims::C * dims::numTx    * dims::numScP;
    static constexpr uint32_t kRdlElems  = dims::U * dims::numRx    * dims::numScP;
    static constexpr uint32_t kXulElems  = dims::U * dims::numUeTx  * dims::numScP;
    static constexpr uint32_t kRulElems  = dims::C * dims::numTx    * dims::numScP;
    static constexpr uint32_t kZElems    = dims::C * dims::rankMax  * dims::numScP;

    // Full H_dl table: [C][U][numRx][numTx][numScP] half2c
    static constexpr uint32_t kHdlElems =
        dims::C * dims::U * dims::numRx * dims::numTx * dims::numScP;

    // Victim map host scratch: [C][numScP]
    static constexpr uint32_t kVictimElems = dims::C * dims::numScP;

    // ── CUDA streams ─────────────────────────────────────────────────────────
    cudaStream_t dlStream_    = nullptr;
    cudaStream_t ulStream_    = nullptr;
    cudaStream_t slowStream_  = nullptr;  // H_dl update (off the hot path)

    // ── Per-symbol ring slots ─────────────────────────────────────────────────
    ci16*    d_xDlRaw_   = nullptr;  // [N][kXdlElems]  ci16   H2D staging (K0 in)
    cf32*    d_xDl_      = nullptr;  // [N][kXdlElems]  cf32   K0 out / K1 in
    cf32*    d_y_        = nullptr;  // [N][kYElems]    cf32   K1 out / K2 in
    cf32*    d_rDl_      = nullptr;  // [N][kRdlElems]  cf32   K2 out (vUE DL, Spec D)
    cf32*    d_xUl_      = nullptr;  // [N][kXulElems]  cf32   vUE UL in (Spec D)
    cf32*    d_rUl_      = nullptr;  // [N][kRulElems]  cf32   K3 out / K4 in
    cf32*    d_z_        = nullptr;  // [N][kZElems]    cf32   K4 out / K5 in
    ci16*    d_zPacked_  = nullptr;  // [N][kZElems]    ci16   K5 out  (D2H staging)

    // K2 split-K partial-sum scratch (Spec E §E.12); one buffer, reused (K2's two
    // passes are stream-ordered, never concurrent across symbols on dlStream_).
    cf32*    d_k2Scratch_ = nullptr;  // [kK2ScratchElems] cf32

    // ── Per-symbol control (updated each symbol) ──────────────────────────────
    uint16_t* d_victim_    = nullptr;   // [C * numScP]  DL victim map
    uint16_t* d_ulContrib_ = nullptr;   // [C * numScP]  UL contributor map
    Alloc*    d_allocs_    = nullptr;   // [MAX_ALLOCS]
    cf32*     d_doppler_   = nullptr;   // [C * U]
    uint8_t*  d_ueTxToRx_  = nullptr;  // [numUeTx]  default {0, 1}

    uint32_t  numAllocs_   = 0;         // saved per uploadSymbol() for launchDl/Ul

    // Host scratch for victim map construction before H2D
    uint16_t  victimScratch_[kVictimElems];
    uint16_t  ulContribScratch_[kVictimElems];

    // ── Resident codebook (loaded once) ──────────────────────────────────────
    cf32*    d_precodeBook_ = nullptr;  // [numBeams][numTx]
    cf32*    d_combineBook_ = nullptr;  // [numBeams][numTx]
    uint32_t numBeams_      = 0;

    // ── H_dl double buffer + device indirection cell (ADR 0001 §4) ───────────
    // Kernels will use d_HdlActive_ (host-side copy) for the stepping-stone;
    // the TODO graph path would pass d_HdlActiveCell_ and dereference on device.
    half2c*   d_Hdl_[2]        = {nullptr, nullptr};  // [0]=active, [1]=back
    half2c**  d_HdlActiveCell_  = nullptr;   // device ptr-to-ptr (indirection cell)
    half2c*   d_HdlActiveCopy_  = nullptr;   // host-side mirror of active pointer
    int       activeHdlIdx_     = 0;

    bool created_ = false;
};

}  // namespace orca

#endif  // EMU_WITH_CUDA
