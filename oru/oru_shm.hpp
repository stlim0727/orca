#pragma once
// Linux-only POSIX-shm backend for the ORU ↔ ORCA north interface (Spec F §F.8).
// Replaces OruLoopback for cross-process use: one shm region holds all four
// control rings and both bulk IQ slot-arrays.
//
// Usage model (mirrors Spec F §F.5 handshake):
//   ORU process: OruShm shm("..."); shm.create();   // allocates shm, zero-init by OS
//   ORCA:        OruShm shm("..."); shm.attach();    // maps existing region
//
// After fork() the child inherits the mmap; call disownCreator() in the child
// so only the parent calls shm_unlink at cleanup.

#ifdef __linux__

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <string>

#include "common/spsc_ring.hpp"
#include "oru/oru_transport.hpp"

// Require lock-free uint32 so atomics are safe in shared memory (no per-address
// futex/mutex table that would break across process boundaries).
static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "lock-free atomic<uint32_t> required for shm ring use");

namespace orca {

class OruShm final : public OruTransport {
  public:
    static constexpr uint32_t kRingDepth   = 8;
    static constexpr const char* kDefaultName = "/orca.oru.v1";

    explicit OruShm(const char* name = kDefaultName) : name_(name) {}
    ~OruShm() override { detach(); }

    // ORU-process side: allocate the named shm object and mmap it.
    // Newly-created shm pages are OS-zero-filled (POSIX guarantee); that is a
    // valid empty-ring state for every SpscRing (head=0, tail=0).
    bool create() {
        if (attached_) return true;
        fd_ = ::shm_open(name_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0600);
        if (fd_ < 0) return false;
        if (::ftruncate(fd_, static_cast<off_t>(sizeof(ShmLayout))) < 0) {
            ::close(fd_); fd_ = -1; return false;
        }
        void* p = ::mmap(nullptr, sizeof(ShmLayout),
                         PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) { ::close(fd_); fd_ = -1; return false; }
        region_   = static_cast<ShmLayout*>(p);
        creator_  = true;
        attached_ = true;
        return true;
    }

    // After fork(): tell the child not to call shm_unlink (parent owns cleanup).
    void disownCreator() { creator_ = false; }

    // ORCA side: map an existing region (name must match what the ORU created).
    bool attach() override {
        if (attached_) return true;
        fd_ = ::shm_open(name_.c_str(), O_RDWR, 0);
        if (fd_ < 0) return false;
        void* p = ::mmap(nullptr, sizeof(ShmLayout),
                         PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) { ::close(fd_); fd_ = -1; return false; }
        region_   = static_cast<ShmLayout*>(p);
        attached_ = true;
        return true;
    }

    void detach() override {
        if (!attached_) return;
        if (region_) { ::munmap(region_, sizeof(ShmLayout)); region_ = nullptr; }
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
        if (creator_) { ::shm_unlink(name_.c_str()); creator_ = false; }
        attached_ = false;
    }

    // ── ORCA side (OruTransport §F.8) ────────────────────────────────────────
    bool pollDl(OruDoorbell& out) override {
        return attached_ && region_->dlDoorbell.pop(out);
    }
    bool returnDl(uint16_t slotIdx, uint32_t seq) override {
        return attached_ && region_->dlReturn.push(OruCredit{slotIdx, seq});
    }
    bool publishUl(uint16_t slotIdx, OruDoorbell meta) override {
        if (!attached_) return false;
        meta.slotIdx = slotIdx;
        return region_->ulDoorbell.push(meta);
    }
    bool reclaimUl(OruCredit& out) override {
        return attached_ && region_->ulReturn.pop(out);
    }
    ci16* dlSlot(uint16_t slotIdx) override {
        return (attached_ && slotIdx < dims::N_ring)
            ? region_->dl[slotIdx] : nullptr;
    }
    ci16* ulSlot(uint16_t slotIdx) override {
        return (attached_ && slotIdx < dims::N_ring)
            ? region_->ul[slotIdx] : nullptr;
    }
    AllocBlock* allocBlock(uint16_t slotIdx) override {
        return (attached_ && slotIdx < dims::N_ring)
            ? &region_->allocs[slotIdx] : nullptr;
    }

    // ── ORU-process side (mirrors OruLoopback oru*() API, §F.6) ──────────────
    bool oruPublishDl(const OruDoorbell& d) {
        return attached_ && region_->dlDoorbell.push(d);
    }
    bool oruReclaimDl(OruCredit& out) {
        return attached_ && region_->dlReturn.pop(out);
    }
    bool oruPollUl(OruDoorbell& out) {
        return attached_ && region_->ulDoorbell.pop(out);
    }
    bool oruReturnUl(const OruCredit& c) {
        return attached_ && region_->ulReturn.push(c);
    }
    ci16* oruDlSlot(uint16_t slotIdx) {
        return (attached_ && slotIdx < dims::N_ring)
            ? region_->dl[slotIdx] : nullptr;
    }
    ci16* oruUlSlot(uint16_t slotIdx) {
        return (attached_ && slotIdx < dims::N_ring)
            ? region_->ul[slotIdx] : nullptr;
    }
    AllocBlock* oruAllocBlock(uint16_t slotIdx) {
        return (attached_ && slotIdx < dims::N_ring)
            ? &region_->allocs[slotIdx] : nullptr;
    }

  private:
    // All of this lives in the shm region. OS zero-fills new shm pages, which
    // is a valid initial state: every SpscRing is empty (head==tail==0),
    // every bulk slot is zero.
    struct ShmLayout {
        SpscRing<OruDoorbell, kRingDepth> dlDoorbell;
        SpscRing<OruCredit,   kRingDepth> dlReturn;
        SpscRing<OruDoorbell, kRingDepth> ulDoorbell;
        SpscRing<OruCredit,   kRingDepth> ulReturn;
        ci16       dl[dims::N_ring][kSlotElems];
        ci16       ul[dims::N_ring][kSlotElems];
        AllocBlock allocs[dims::N_ring];
    };

    std::string name_;
    ShmLayout*  region_   = nullptr;
    int         fd_       = -1;
    bool        creator_  = false;
    bool        attached_ = false;
};

}  // namespace orca

#endif  // __linux__
