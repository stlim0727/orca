#pragma once
// Spec A §A.4/§A.5 — reassembly ring: N ≥ 3 symbol slots, per-slot lifecycle
// Free → Filling → Ready → Computing → Egressing → Free, PRB coverage bitmap,
// key (cell, dir, sfn, slot, sym), and the never-stall drop policy (zero-fill
// missing PRBs, mark late_drop, advance).
//
// This is host-owned orchestration state (Spec A §A.5 note); the bulk IQ lives
// in external buffers addressed by the slot index (Spec D/F slot model). The
// ring records *which* PRBs were missing; zero-filling the IQ is the buffer
// owner's job.

#include <array>
#include <cassert>
#include <cstdint>

#include "common/dims.hpp"
#include "common/symbol_id.hpp"

namespace orca {

enum class SlotState : uint8_t { Free, Filling, Ready, Computing, Egressing };

// PRB coverage bitmap — one bit per PRB (273 → 5 × u64 words).
class PrbCoverage {
  public:
    static constexpr uint32_t kWords = (dims::numPrb + 63u) / 64u;

    void clear() {
        for (auto& w : bits_) w = 0;
        covered_ = 0;
    }

    // Marks [startPrb, startPrb+numPrb); clamps to numPrb (overflow-safe).
    // Returns the count of *newly* covered PRBs (re-marking adds 0).
    uint32_t mark(uint32_t startPrb, uint32_t numPrb) {
        if (startPrb >= dims::numPrb) return 0;
        const uint32_t maxLen = dims::numPrb - startPrb;
        const uint32_t end = startPrb + (numPrb < maxLen ? numPrb : maxLen);
        uint32_t newly = 0;
        for (uint32_t p = startPrb; p < end; ++p) {
            const uint64_t bit = 1ull << (p & 63u);
            if (!(bits_[p >> 6] & bit)) {
                bits_[p >> 6] |= bit;
                ++newly;
            }
        }
        covered_ += newly;
        return newly;
    }

    bool test(uint32_t prb) const {
        return prb < dims::numPrb && (bits_[prb >> 6] & (1ull << (prb & 63u)));
    }

    bool complete() const { return covered_ == dims::numPrb; }
    uint32_t covered() const { return covered_; }

  private:
    uint64_t bits_[kWords] = {};
    uint32_t covered_ = 0;
};

struct RingStats {
    uint64_t claimed = 0;     // slots claimed for a new symbol
    uint64_t completed = 0;   // reached Ready with full coverage
    uint64_t lateDrops = 0;   // §A.4 events: forced partial or evicted unfinished
    uint64_t overwrites = 0;  // oldest-Filling slot reclaimed by a newer symbol
    uint64_t rejected = 0;    // claim found no Free/Filling slot (input dropped)
};

struct SymbolSlot {
    SymbolId id{};
    SlotState state = SlotState::Free;
    PrbCoverage coverage;
    bool partial = false;  // Ready was forced with missing PRBs (§A.4)
    uint64_t seq = 0;      // claim order — oldest-victim selection
};

template <uint32_t N = dims::N_ring>
class SymbolRing {
    static_assert(N >= 3, "Spec A §A.5: input/compute/output overlap needs N >= 3");

  public:
    static constexpr uint32_t kNone = N;  // "no slot" sentinel

    // Claims a slot for `id`. If `id` is already Filling, returns that slot
    // (sections accumulate). Otherwise takes a Free slot; if none, evicts the
    // oldest Filling slot (drop policy — never stall; the evicted symbol is a
    // late_drop). If every slot is Ready/Computing/Egressing, returns kNone:
    // the caller drops the input (never blocks the pipeline).
    uint32_t claim(SymbolId id) {
        uint32_t freeIdx = kNone, oldestFilling = kNone;
        uint64_t oldestSeq = UINT64_MAX;
        for (uint32_t i = 0; i < N; ++i) {
            SymbolSlot& s = slots_[i];
            if (s.state == SlotState::Filling) {
                if (s.id == id) return i;
                if (s.seq < oldestSeq) {
                    oldestSeq = s.seq;
                    oldestFilling = i;
                }
            } else if (s.state == SlotState::Free && freeIdx == kNone) {
                freeIdx = i;
            }
        }
        uint32_t idx = freeIdx;
        if (idx == kNone) {
            if (oldestFilling == kNone) {
                ++stats_.rejected;
                return kNone;
            }
            idx = oldestFilling;  // evict: the unfinished symbol is dropped late
            ++stats_.overwrites;
            ++stats_.lateDrops;
        }
        SymbolSlot& s = slots_[idx];
        s.id = id;
        s.state = SlotState::Filling;
        s.coverage.clear();
        s.partial = false;
        s.seq = nextSeq_++;
        ++stats_.claimed;
        return idx;
    }

    // Accumulates a section into a Filling slot. Returns true when coverage
    // is complete (caller may then markReady). Out-of-range idx (incl. kNone)
    // is a no-op returning false.
    bool addSection(uint32_t idx, uint32_t startPrb, uint32_t numPrb) {
        if (idx >= N) return false;
        SymbolSlot& s = slots_[idx];
        if (s.state != SlotState::Filling) return false;
        s.coverage.mark(startPrb, numPrb);
        return s.coverage.complete();
    }

    // Coverage complete → Ready.
    void markReady(uint32_t idx) {
        if (idx >= N) return;
        SymbolSlot& s = slots_[idx];
        if (s.state != SlotState::Filling || !s.coverage.complete()) return;
        s.state = SlotState::Ready;
        s.partial = false;
        ++stats_.completed;
    }

    // §A.4 drop policy: the deadline D_r(s) arrived with PRBs missing —
    // process what's there. Marks the slot Ready+partial and counts late_drop.
    // (Zero-filling the missing PRBs' IQ is the buffer owner's job; the
    // missing set is queryable via slot.coverage.test().)
    void forceReadyAtDeadline(uint32_t idx) {
        if (idx >= N) return;
        SymbolSlot& s = slots_[idx];
        if (s.state != SlotState::Filling) return;
        s.state = SlotState::Ready;
        if (!s.coverage.complete()) {
            s.partial = true;
            ++stats_.lateDrops;
        } else {
            ++stats_.completed;
        }
    }

    void markComputing(uint32_t idx) {
        if (idx < N && slots_[idx].state == SlotState::Ready)
            slots_[idx].state = SlotState::Computing;
    }
    void markEgressing(uint32_t idx) {
        if (idx < N && slots_[idx].state == SlotState::Computing)
            slots_[idx].state = SlotState::Egressing;
    }
    // Returns the slot to Free and clears its state (no stale id/coverage
    // observable on a Free slot).
    void release(uint32_t idx) {
        if (idx < N && slots_[idx].state == SlotState::Egressing)
            slots_[idx] = SymbolSlot{};
    }

    // Lowest-seq Ready slot (FIFO order), or kNone.
    uint32_t nextReady() const {
        uint32_t best = kNone;
        uint64_t bestSeq = UINT64_MAX;
        for (uint32_t i = 0; i < N; ++i) {
            if (slots_[i].state == SlotState::Ready && slots_[i].seq < bestSeq) {
                bestSeq = slots_[i].seq;
                best = i;
            }
        }
        return best;
    }

    // Slot currently holding `id` in any non-Free state, or kNone.
    uint32_t find(SymbolId id) const {
        for (uint32_t i = 0; i < N; ++i)
            if (slots_[i].state != SlotState::Free && slots_[i].id == id) return i;
        return kNone;
    }

    // Precondition: idx < N (checked by assert; never pass kNone).
    SymbolSlot& at(uint32_t idx) {
        assert(idx < N);
        return slots_[idx];
    }
    const SymbolSlot& at(uint32_t idx) const {
        assert(idx < N);
        return slots_[idx];
    }
    const RingStats& stats() const { return stats_; }

  private:
    std::array<SymbolSlot, N> slots_{};
    RingStats stats_{};
    uint64_t nextSeq_ = 1;
};

}  // namespace orca
