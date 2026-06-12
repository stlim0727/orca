#pragma once
// Linux-only POSIX-shm backend for the ORCA ↔ vUE south interface (Spec D §D.8).
// One shm region holds all four control rings and both bulk IQ slot-arrays
// (cf32; HBM on the target GPU, host memory here for CPU-only testing).
//
// Usage model (mirrors Spec D §D.5 handshake):
//   ORCA:  VueShm vue("..."); vue.create();   // ORCA owns allocation (§D.1)
//   vUE:   VueShm vue("..."); vue.attach();   // maps existing region
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
#include "vue/vue_transport.hpp"

namespace orca {

class VueShm final : public VueTransport {
  public:
    static constexpr uint32_t kRingDepth   = 8;
    static constexpr const char* kDefaultName = "/orca.vue.v1";

    explicit VueShm(const char* name = kDefaultName) : name_(name) {}
    ~VueShm() override { detach(); }

    // ORCA side: allocate the named shm object and mmap it.
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

    // vUE side: map an existing region.
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

    // ── ORCA side (VueTransport §D.8) ────────────────────────────────────────
    bool publishDl(uint16_t slotIdx, VueDoorbell meta) override {
        if (!attached_) return false;
        meta.slotIdx = slotIdx;
        return region_->dlDoorbell.push(meta);
    }
    bool reclaimDl(VueCredit& out) override {
        return attached_ && region_->dlReturn.pop(out);
    }
    bool pollUl(VueDoorbell& out) override {
        return attached_ && region_->ulDoorbell.pop(out);
    }
    bool returnUl(uint16_t slotIdx, uint32_t seq) override {
        return attached_ && region_->ulReturn.push(VueCredit{slotIdx, seq});
    }

    // ── vUE side (§D.8) ──────────────────────────────────────────────────────
    bool pollDl(VueDoorbell& out) override {
        return attached_ && region_->dlDoorbell.pop(out);
    }
    bool returnDl(uint16_t slotIdx, uint32_t seq) override {
        return attached_ && region_->dlReturn.push(VueCredit{slotIdx, seq});
    }
    bool submitUl(uint16_t slotIdx, VueDoorbell meta) override {
        if (!attached_) return false;
        meta.slotIdx = slotIdx;
        return region_->ulDoorbell.push(meta);
    }
    bool reclaimUl(VueCredit& out) override {
        return attached_ && region_->ulReturn.pop(out);
    }

    // ── bulk slot access ──────────────────────────────────────────────────────
    cf32* dlSlot(uint16_t slotIdx) override {
        return (attached_ && slotIdx < dims::N_ring)
            ? region_->dl[slotIdx] : nullptr;
    }
    cf32* ulSlot(uint16_t slotIdx) override {
        return (attached_ && slotIdx < dims::N_ring)
            ? region_->ul[slotIdx] : nullptr;
    }

  private:
    struct ShmLayout {
        SpscRing<VueDoorbell, kRingDepth> dlDoorbell;
        SpscRing<VueCredit,   kRingDepth> dlReturn;
        SpscRing<VueDoorbell, kRingDepth> ulDoorbell;
        SpscRing<VueCredit,   kRingDepth> ulReturn;
        cf32 dl[dims::N_ring][kDlSlotElems];
        cf32 ul[dims::N_ring][kUlSlotElems];
    };

    std::string name_;
    ShmLayout*  region_   = nullptr;
    int         fd_       = -1;
    bool        creator_  = false;
    bool        attached_ = false;
};

}  // namespace orca

#endif  // __linux__
