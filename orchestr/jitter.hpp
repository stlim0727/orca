#pragma once
// Stage-1 jitter harness (AGENT.md 1b): per-symbol latency samples into a
// fixed-bin histogram; p50/p99/p99.9 extraction. Host-side instrumentation,
// not hot-path code.

#include <cmath>
#include <cstdint>
#include <vector>

namespace orca {

class JitterHistogram {
  public:
    static constexpr uint64_t kBinNs = 100;     // 100 ns resolution
    static constexpr uint32_t kBins = 10000;    // covers 0 .. 1 ms

    JitterHistogram() : bins_(kBins, 0) {}

    void record(uint64_t ns) {
        const uint64_t bin = ns / kBinNs;
        if (bin < kBins)
            ++bins_[static_cast<uint32_t>(bin)];
        else
            ++overflow_;
        ++count_;
        if (ns < min_) min_ = ns;
        if (ns > max_) max_ = ns;
        sum_ += ns;
    }

    // Nearest-rank percentile (rank = ceil(p·count), 1-based): returns the
    // upper edge of the bin holding that sample; samples past the histogram
    // range report max(). p in [0,1]; 0 samples → 0.
    uint64_t percentileNs(double p) const {
        if (count_ == 0) return 0;
        uint64_t rank =
            static_cast<uint64_t>(std::ceil(p * static_cast<double>(count_)));
        if (rank == 0) rank = 1;
        if (rank > count_) rank = count_;
        uint64_t seen = 0;
        for (uint32_t b = 0; b < kBins; ++b) {
            seen += bins_[b];
            if (seen >= rank) return (uint64_t{b} + 1) * kBinNs;
        }
        return max_;
    }

    uint64_t count() const { return count_; }
    uint64_t minNs() const { return count_ ? min_ : 0; }
    uint64_t maxNs() const { return max_; }
    uint64_t meanNs() const { return count_ ? sum_ / count_ : 0; }
    uint64_t overflow() const { return overflow_; }

    void reset() {
        for (auto& b : bins_) b = 0;
        overflow_ = count_ = sum_ = max_ = 0;
        min_ = UINT64_MAX;
    }

  private:
    std::vector<uint64_t> bins_;
    uint64_t overflow_ = 0;
    uint64_t count_ = 0;
    uint64_t sum_ = 0;
    uint64_t min_ = UINT64_MAX;
    uint64_t max_ = 0;
};

}  // namespace orca
