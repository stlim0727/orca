// Cross-process IPC test (Linux only, step C).
//
// Verifies the POSIX-shm backends (OruShm, VueShm) by exchanging bulk slots
// and control-ring messages between a fork()-parent and fork()-child, proving
// that:
//   • bulk IQ written by one process is readable by the other via shared memory;
//   • SpscRing push/pop in shared memory is correctly visible cross-process;
//   • the doorbell+credit round-trip (both directions) completes correctly.
//
// Each sub-test forks once; the child exits with 0 on success and the parent
// collects the exit code.  spinWait loops bound at 10M iterations (~50–100 ms)
// so the test never hangs even on bugs.

#ifdef __linux__

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "oru/oru_shm.hpp"
#include "vue/vue_shm.hpp"

using namespace orca;

static int failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            ++failures;                                                          \
        }                                                                        \
    } while (0)

// Bounded spin-wait.  Returns true if fn() returned true within the limit.
template <typename Fn>
static bool spinWait(Fn&& fn) {
    for (int i = 0; i < 10'000'000; ++i) if (fn()) return true;
    return false;
}

// Collect child exit and flag failure if non-zero.
static void collect(pid_t pid, const char* label) {
    int status = 0;
    ::waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::fprintf(stderr, "FAIL child %s: status=%d exit=%d\n", label,
                     status, WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        ++failures;
    }
}

// ── OruShm test: DL (ORU→ORCA) then UL (ORCA→ORU) ────────────────────────────
//
// Parent = ORU side.  Child = ORCA side.
static void testOruShm() {
    OruShm shm("/orca.test.oru.v1");
    if (!shm.create()) {
        std::fprintf(stderr, "FAIL OruShm::create\n");
        ++failures;
        return;
    }

    const pid_t pid = ::fork();
    if (pid < 0) { ++failures; return; }

    if (pid == 0) {
        // ── Child (ORCA side) ──────────────────────────────────────────────
        shm.disownCreator();
        int cf = 0;

        // Wait for DL doorbell from ORU.
        OruDoorbell db{};
        if (!spinWait([&] { return shm.pollDl(db); })) {
            std::fprintf(stderr, "FAIL child: DL doorbell timeout\n"); ++cf;
            std::exit(cf);
        }
        // Verify bulk DL data written by parent.
        const ci16* slot = shm.dlSlot(db.slotIdx);
        if (!slot || slot[0].re != 42 || slot[0].im != -7)  ++cf;
        if (!slot || slot[1].re != 100 || slot[1].im != -99) ++cf;
        // Verify alloc block.
        const AllocBlock* ab = shm.allocBlock(db.slotIdx);
        if (!ab || ab->numAllocs != 1)                       ++cf;
        if (ab && (ab->allocs[0].cell != 0 || ab->allocs[0].ueId != 5)) ++cf;
        // Return DL credit.
        shm.returnDl(db.slotIdx, db.seq);

        // Now write UL data into a slot and publish doorbell.
        const uint16_t uslotIdx = 1;
        ci16* uslot = shm.ulSlot(uslotIdx);
        if (uslot) { uslot[0] = ci16{77, 88}; uslot[1] = ci16{-11, 22}; }
        AllocBlock* uab = shm.allocBlock(uslotIdx);
        if (uab) {
            uab->numAllocs = 1;
            uab->allocs[0] = Alloc{1, 3, 0, 12, 1, 1, {2, 0, 0, 0}};
        }
        OruDoorbell udb{uslotIdx, 20, 2, 5, 1, 1, 100};
        shm.publishUl(uslotIdx, udb);

        // Wait for UL return credit from ORU.
        OruCredit cr{};
        if (!spinWait([&] { return shm.reclaimUl(cr); })) {
            std::fprintf(stderr, "FAIL child: UL credit timeout\n"); ++cf;
        }
        if (cr.slotIdx != uslotIdx) ++cf;

        std::exit(cf > 0 ? 1 : 0);
    }

    // ── Parent (ORU side) ──────────────────────────────────────────────────────

    // Write DL bulk data + alloc, then publish DL doorbell.
    const uint16_t dslotIdx = 0;
    ci16* slot = shm.oruDlSlot(dslotIdx);
    if (slot) { slot[0] = ci16{42, -7}; slot[1] = ci16{100, -99}; }
    AllocBlock* ab = shm.oruAllocBlock(dslotIdx);
    if (ab) {
        ab->numAllocs = 1;
        ab->allocs[0] = Alloc{0, 5, 0, 12, 0, 1, {3, 0, 0, 0}};
    }
    OruDoorbell db{dslotIdx, 10, 1, 3, 1, 0, 42};
    shm.oruPublishDl(db);

    // Wait for DL return credit from ORCA.
    OruCredit cr{};
    CHECK(spinWait([&] { return shm.oruReclaimDl(cr); }));
    CHECK(cr.slotIdx == dslotIdx);

    // Wait for UL doorbell from ORCA.
    OruDoorbell udb{};
    CHECK(spinWait([&] { return shm.oruPollUl(udb); }));
    CHECK(udb.slotIdx == 1);
    CHECK(udb.sfn == 20);

    // Verify UL bulk data written by child.
    const ci16* uslot = shm.oruUlSlot(udb.slotIdx);
    CHECK(uslot != nullptr);
    CHECK(uslot && uslot[0].re == 77  && uslot[0].im == 88);
    CHECK(uslot && uslot[1].re == -11 && uslot[1].im == 22);

    // Verify UL alloc block written by child.
    const AllocBlock* uab = shm.oruAllocBlock(udb.slotIdx);
    CHECK(uab != nullptr);
    CHECK(uab && uab->numAllocs == 1);
    CHECK(uab && uab->allocs[0].cell == 1 && uab->allocs[0].ueId == 3);

    // Return UL credit to ORCA.
    shm.oruReturnUl(OruCredit{udb.slotIdx, udb.seq});

    collect(pid, "OruShm");
}

// ── VueShm test: DL (ORCA→vUE) then UL (vUE→ORCA) ───────────────────────────
//
// Parent = ORCA side.  Child = vUE side.
static void testVueShm() {
    VueShm vue("/orca.test.vue.v1");
    if (!vue.create()) {
        std::fprintf(stderr, "FAIL VueShm::create\n");
        ++failures;
        return;
    }

    const pid_t pid = ::fork();
    if (pid < 0) { ++failures; return; }

    if (pid == 0) {
        // ── Child (vUE side) ───────────────────────────────────────────────
        vue.disownCreator();
        int cf = 0;

        // Wait for DL doorbell from ORCA.
        VueDoorbell db{};
        if (!spinWait([&] { return vue.pollDl(db); })) {
            std::fprintf(stderr, "FAIL child: VUE DL doorbell timeout\n"); ++cf;
            std::exit(cf);
        }
        // Verify DL bulk data.
        const cf32* dslot = vue.dlSlot(db.slotIdx);
        if (!dslot || dslot[0].re != 1.25f || dslot[0].im != -3.5f)  ++cf;
        if (!dslot || dslot[1].re != 0.5f  || dslot[1].im != 2.0f)   ++cf;
        // Return DL credit.
        vue.returnDl(db.slotIdx, db.seq);

        // Write UL data and submit doorbell.
        const uint16_t uslotIdx = 2;
        cf32* uslot = vue.ulSlot(uslotIdx);
        if (uslot) { uslot[0] = cf32{-1.0f, 4.0f}; uslot[1] = cf32{0.125f, -0.5f}; }
        VueDoorbell udb{uslotIdx, 30, 3, 7, 0, 200};
        vue.submitUl(uslotIdx, udb);

        // Wait for UL return credit from ORCA.
        VueCredit cr{};
        if (!spinWait([&] { return vue.reclaimUl(cr); })) {
            std::fprintf(stderr, "FAIL child: VUE UL credit timeout\n"); ++cf;
        }
        if (cr.slotIdx != uslotIdx) ++cf;

        std::exit(cf > 0 ? 1 : 0);
    }

    // ── Parent (ORCA side) ─────────────────────────────────────────────────────

    // Write DL bulk data and publish doorbell.
    const uint16_t dslotIdx = 0;
    cf32* dslot = vue.dlSlot(dslotIdx);
    if (dslot) { dslot[0] = cf32{1.25f, -3.5f}; dslot[1] = cf32{0.5f, 2.0f}; }
    VueDoorbell db{dslotIdx, 5, 0, 1, 0, 77};
    vue.publishDl(dslotIdx, db);

    // Wait for DL return credit from vUE.
    VueCredit cr{};
    CHECK(spinWait([&] { return vue.reclaimDl(cr); }));
    CHECK(cr.slotIdx == dslotIdx);

    // Wait for UL doorbell from vUE.
    VueDoorbell udb{};
    CHECK(spinWait([&] { return vue.pollUl(udb); }));
    CHECK(udb.slotIdx == 2);
    CHECK(udb.sfn == 30);

    // Verify UL bulk data written by child.
    const cf32* uslot = vue.ulSlot(udb.slotIdx);
    CHECK(uslot != nullptr);
    CHECK(uslot && uslot[0].re == -1.0f  && uslot[0].im == 4.0f);
    CHECK(uslot && uslot[1].re == 0.125f && uslot[1].im == -0.5f);

    // Return UL credit to vUE.
    vue.returnUl(udb.slotIdx, udb.seq);

    collect(pid, "VueShm");
}

int main() {
    testOruShm();
    testVueShm();

    if (failures) {
        std::fprintf(stderr, "%d check(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    std::puts("test_ipc: all checks passed");
    return EXIT_SUCCESS;
}

#else

#include <cstdio>
int main() {
    std::puts("test_ipc: skipped (Linux only)");
    return 0;
}

#endif  // __linux__
