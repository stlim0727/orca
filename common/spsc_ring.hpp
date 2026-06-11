#pragma once
// Lockless single-producer/single-consumer descriptor ring — the shared
// helper behind the Spec D §D.3 / Spec F §F.4 control rings. The Linux
// backends replace this with rte_ring; the schema (one descriptor per entry,
// push fails when full — never blocks) is identical, which is what makes the
// loopback backends drop-in compatible.
//
// Carries control descriptors only — bulk IQ never crosses a control ring
// (Spec D §D.1 invariant).

#include <atomic>
#include <cstdint>

namespace orca {

template <typename T, uint32_t N>
class SpscRing {
    static_assert(N >= 2 && (N & (N - 1)) == 0, "capacity must be a power of 2");

  public:
    // Producer side. Returns false when full (caller drops — never blocks).
    // Counters are free-running u32; their wrap is benign (unsigned
    // subtraction yields the true distance while the SPSC invariant
    // 0 ≤ head−tail ≤ N holds). `>= N` rather than `== N` so a violated
    // invariant degrades to "full", never to overwriting unread entries.
    bool push(const T& v) {
        const uint32_t head = head_.load(std::memory_order_relaxed);
        const uint32_t tail = tail_.load(std::memory_order_acquire);
        if (head - tail >= N) return false;
        buf_[head & (N - 1)] = v;
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // Consumer side. Returns false when empty.
    bool pop(T& out) {
        const uint32_t tail = tail_.load(std::memory_order_relaxed);
        const uint32_t head = head_.load(std::memory_order_acquire);
        if (head == tail) return false;
        out = buf_[tail & (N - 1)];
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    uint32_t size() const {
        return head_.load(std::memory_order_acquire) -
               tail_.load(std::memory_order_acquire);
    }
    bool empty() const { return size() == 0; }
    static constexpr uint32_t capacity() { return N; }

  private:
    // Producer- and consumer-owned counters live on their own cache lines so
    // the two sides don't false-share; the buffer sits apart from both.
    alignas(64) std::atomic<uint32_t> head_{0};
    alignas(64) std::atomic<uint32_t> tail_{0};
    alignas(64) T buf_[N] = {};
};

}  // namespace orca
