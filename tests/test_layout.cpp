#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

#include "common/complex.hpp"
#include "common/dims.hpp"
#include "common/layout.hpp"
#include "common/symbol_id.hpp"
#include "dsp/convert.hpp"

namespace {

void testDimsAndRows() {
  using namespace orca::common;
  static_assert(roundUp(3276, 32) == 3296);
  assert(numScP == 3296);
  assert(layout::hDlRowBytes() == 13184);
  assert(layout::cf32RowBytes() == 26368);
  assert(layout::hDlRowBytes() % 128 == 0);
  assert(layout::cf32RowBytes() % 128 == 0);
}

void testFlatIndexes() {
  using namespace orca::common;
  assert(layout::idxHdl(0, 0, 0, 0, 0) == 0);
  assert(layout::idxHdl(0, 0, 0, 1, 0) == numScP);
  assert(layout::idxHdl(0, 0, 1, 0, 0) == numTx * numScP);
  assert(layout::idxHdl(0, 1, 0, 0, 0) == numRx * numTx * numScP);
  assert(layout::idxHdl(1, 0, 0, 0, 0) == U * numRx * numTx * numScP);

  assert(layout::idxXdl(1, 2, 3) == ((1 * rankMax + 2) * numScP + 3));
  assert(layout::idxY(1, 63, 7) == ((1 * numTx + 63) * numScP + 7));
  assert(layout::idxRdl(31, 3, 11) == ((31 * numRx + 3) * numScP + 11));
  assert(layout::idxXul(31, 1, 13) == ((31 * numUeTx + 1) * numScP + 13));
  assert(layout::idxRul(1, 63, 17) == ((1 * numTx + 63) * numScP + 17));
  assert(layout::idxZ(1, 3, 19) == ((1 * rankMax + 3) * numScP + 19));
}

void testConversions() {
  using orca::common::cf32;
  using orca::common::ci16;

  const std::vector<ci16> wire{{0, 1}, {-2, 3}, {32767, -32768}, {-1234, 2345}};
  std::vector<cf32> unpacked(wire.size());
  std::vector<ci16> repacked(wire.size());

  orca::dsp::k0IngressConvertCpu(wire.data(), unpacked.data(), wire.size());
  orca::dsp::k5EgressPackCpu(unpacked.data(), repacked.data(), repacked.size());

  for (std::size_t i = 0; i < wire.size(); ++i) {
    assert(repacked[i].re == wire[i].re);
    assert(repacked[i].im == wire[i].im);
  }

  const cf32 saturateCases[]{{40000.0F, -40000.0F}, {1.4F, -1.6F}, {0.0F, 0.0F}};
  ci16 saturated[3]{};
  orca::dsp::k5EgressPackCpu(saturateCases, saturated, 3);
  assert(saturated[0].re == std::numeric_limits<std::int16_t>::max());
  assert(saturated[0].im == std::numeric_limits<std::int16_t>::min());
  assert(saturated[1].re == 1);
  assert(saturated[1].im == -2);
}

void testSymbolCounter() {
  using namespace orca::common;
  assert(continuousSymbolCounter(0, 0, 0) == 0);
  assert(continuousSymbolCounter(0, 0, 13) == 13);
  assert(continuousSymbolCounter(0, 1, 0) == 14);
  assert(continuousSymbolCounter(1, 0, 0) == 20 * 14);
}

}  // namespace

int main() {
  testDimsAndRows();
  testFlatIndexes();
  testConversions();
  testSymbolCounter();
  return 0;
}
