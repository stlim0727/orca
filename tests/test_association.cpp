// ADR 0002 §3/§4 golden test: serving-cell + interferer association
// (scenario/association.hpp). A 2-cell 2×2-grid table with per-(cell,gp) path
// loss; verifies serving selection (min pathlossDb), contributor ordering,
// top-K limiting, noCoverage exclusion, full outage, and handover detection.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "channel/cir_table.hpp"
#include "channel/cir_writer.hpp"
#include "channel/cir_loader.hpp"
#include "common/dims.hpp"
#include "scenario/association.hpp"

using namespace orca;
using namespace orca::scenario;
using namespace orca::channel;

static int failures = 0;

#define CHECK(cond) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++failures; \
    } } while (0)

static CirHeader makeHdr() {
    CirHeader h{};
    std::memcpy(h.magic, "ORCACIR1", 8);
    h.version  = kCirVersion;
    h.numCells = dims::C;
    h.gridDims = 2;
    h.nx = 2; h.ny = 2; h.nz = 1;
    h.spacingX = h.spacingY = 1.0f;
    h.ueHeight = 1.5f;
    h.carrierHz = 3.5e9;
    h.pMax = 4;
    h.pathRecBytes   = sizeof(PathRecord);
    h.linkBlockBytes = linkBlockStride(4);
    return h;
}
static CellDesc makeCell() {
    CellDesc c{};
    c.numTx = dims::numTx; c.arrayType = 0; c.elemSpacing = 0.0428f;
    return c;
}
static UeArrayDesc makeUe() {
    UeArrayDesc u{};
    u.numRx = dims::numRx; u.numUeTx = dims::numUeTx;
    u.ueTxToRx[0] = 0; u.ueTxToRx[1] = 1; u.arrayType = 0; u.elemSpacing = 0.0428f;
    return u;
}

int main() {
    const std::string fname = "assoc_test.cir";

    // Lower pathlossDb = stronger link = preferred serving cell.
    CirHeader hdr = makeHdr();
    CellDesc cells[dims::C] = {makeCell(), makeCell()};
    CirWriter w(hdr, cells, makeUe());
    PathRecord p{};
    p.gainRe = 1.0f; p.flags = kPathLoS;
    w.setLink(0, 0, 1, 0, 80.f, &p);  // gp0: cell0 80 dB
    w.setLink(1, 0, 1, 0, 70.f, &p);  //      cell1 70 dB (stronger → serving)
    w.setLink(0, 1, 1, 0, 65.f, &p);  // gp1: cell0 65 dB (stronger → serving)
    w.setLink(1, 1, 1, 0, 90.f, &p);  //      cell1 90 dB
    w.setLink(0, 2, 1, 0, 85.f, &p);  // gp2: cell0 covered
    w.setLink(1, 2, 0, kLinkNoCoverage, 0.f, nullptr);  //   cell1 outage
    w.setLink(0, 3, 0, kLinkNoCoverage, 0.f, nullptr);  // gp3: both outage
    w.setLink(1, 3, 0, kLinkNoCoverage, 0.f, nullptr);
    CHECK(w.write(fname));

    CirTable tbl;
    std::string err;
    CHECK(tbl.load(fname, err));
    if (!tbl.valid()) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }

    // --- gp0: cell1 serves; contributors strongest-first [1,0] ---------------
    UeAssoc a0 = computeAssoc(tbl, 0);
    CHECK(a0.servingCell == 1);
    CHECK(a0.numContrib == 2);
    CHECK(a0.contrib[0] == 1 && a0.contrib[1] == 0);

    // --- gp1: cell0 serves (handover vs gp0); contributors [0,1] -------------
    UeAssoc a1 = computeAssoc(tbl, 1);
    CHECK(a1.servingCell == 0);
    CHECK(a1.numContrib == 2);
    CHECK(a1.contrib[0] == 0 && a1.contrib[1] == 1);
    CHECK(isHandover(a0, a1));
    CHECK(!isHandover(a0, a0));

    // --- gp2: cell1 outage → cell0 serves, single contributor ----------------
    UeAssoc a2 = computeAssoc(tbl, 2);
    CHECK(a2.servingCell == 0);
    CHECK(a2.numContrib == 1);
    CHECK(a2.contrib[0] == 0);

    // --- gp3: full outage → no serving cell ----------------------------------
    UeAssoc a3 = computeAssoc(tbl, 3);
    CHECK(a3.servingCell == kNoCell);
    CHECK(a3.numContrib == 0);
    CHECK(isHandover(a2, a3));  // 0 → kNoCell

    // --- top-K (maxK=1): only the serving cell is a contributor --------------
    UeAssoc a0k = computeAssoc(tbl, 0, /*maxK=*/1);
    CHECK(a0k.servingCell == 1);
    CHECK(a0k.numContrib == 1);
    CHECK(a0k.contrib[0] == 1);

    std::remove(fname.c_str());

    if (failures == 0) {
        std::printf("test_association: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_association: %d failure(s)\n", failures);
    return 1;
}
