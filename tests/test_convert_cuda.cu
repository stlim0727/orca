#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "common/complex.hpp"
#include "dsp/convert.hpp"

namespace {

void requireCuda(cudaError_t error) {
  assert(error == cudaSuccess);
}

void testCudaRoundTripAndSaturation() {
  using orca::common::cf32;
  using orca::common::ci16;

  const std::vector<ci16> wire{{0, 1}, {-2, 3}, {32767, -32768}, {-1234, 2345}};
  const std::vector<cf32> packCases{{40000.0F, -40000.0F}, {1.4F, -1.6F}, {0.0F, 0.0F}};

  ci16* dWire = nullptr;
  cf32* dConverted = nullptr;
  ci16* dRoundTrip = nullptr;
  cf32* dPackCases = nullptr;
  ci16* dPacked = nullptr;

  requireCuda(cudaMalloc(reinterpret_cast<void**>(&dWire), wire.size() * sizeof(ci16)));
  requireCuda(cudaMalloc(reinterpret_cast<void**>(&dConverted), wire.size() * sizeof(cf32)));
  requireCuda(cudaMalloc(reinterpret_cast<void**>(&dRoundTrip), wire.size() * sizeof(ci16)));
  requireCuda(cudaMalloc(reinterpret_cast<void**>(&dPackCases), packCases.size() * sizeof(cf32)));
  requireCuda(cudaMalloc(reinterpret_cast<void**>(&dPacked), packCases.size() * sizeof(ci16)));

  requireCuda(cudaMemcpy(dWire, wire.data(), wire.size() * sizeof(ci16), cudaMemcpyHostToDevice));
  requireCuda(cudaMemcpy(dPackCases, packCases.data(), packCases.size() * sizeof(cf32),
                         cudaMemcpyHostToDevice));

  assert(orca::dsp::k0IngressConvertCuda(dWire, dConverted, wire.size()) ==
         orca::dsp::CudaLaunchStatus::success);
  assert(orca::dsp::k5EgressPackCuda(dConverted, dRoundTrip, wire.size()) ==
         orca::dsp::CudaLaunchStatus::success);
  assert(orca::dsp::k5EgressPackCuda(dPackCases, dPacked, packCases.size()) ==
         orca::dsp::CudaLaunchStatus::success);
  requireCuda(cudaDeviceSynchronize());

  std::vector<ci16> roundTrip(wire.size());
  std::vector<ci16> packed(packCases.size());
  requireCuda(cudaMemcpy(roundTrip.data(), dRoundTrip, roundTrip.size() * sizeof(ci16),
                         cudaMemcpyDeviceToHost));
  requireCuda(cudaMemcpy(packed.data(), dPacked, packed.size() * sizeof(ci16),
                         cudaMemcpyDeviceToHost));

  for (std::size_t i = 0; i < wire.size(); ++i) {
    assert(roundTrip[i].re == wire[i].re);
    assert(roundTrip[i].im == wire[i].im);
  }

  assert(packed[0].re == 32767);
  assert(packed[0].im == -32768);
  assert(packed[1].re == 1);
  assert(packed[1].im == -2);
  assert(packed[2].re == 0);
  assert(packed[2].im == 0);

  requireCuda(cudaFree(dWire));
  requireCuda(cudaFree(dConverted));
  requireCuda(cudaFree(dRoundTrip));
  requireCuda(cudaFree(dPackCases));
  requireCuda(cudaFree(dPacked));
}

void testCudaLaunchStatus() {
  using orca::common::cf32;
  using orca::common::ci16;

  ci16 input{};
  cf32 output{};
  assert(orca::dsp::k0IngressConvertCuda(nullptr, nullptr, 0) ==
         orca::dsp::CudaLaunchStatus::success);
  assert(orca::dsp::k0IngressConvertCuda(nullptr, &output, 1) ==
         orca::dsp::CudaLaunchStatus::invalidArgument);
  assert(orca::dsp::k5EgressPackCuda(nullptr, &input, 1) ==
         orca::dsp::CudaLaunchStatus::invalidArgument);
}

}  // namespace

int main() {
  testCudaRoundTripAndSaturation();
  testCudaLaunchStatus();
  return 0;
}
