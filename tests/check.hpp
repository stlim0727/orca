// Minimal, dependency-free test-assertion helpers shared by every test in
// tests/. Replaces the per-file `static int failures` counter, the duplicated
// `CHECK` macro, and the hand-written main() epilogue that each test used to
// carry. This keeps ORCA's testing model intact — zero third-party deps, one
// executable per test, CTest-driven (no GoogleTest/Catch2/doctest) — while
// removing the boilerplate and adding value-printing CHECK_EQ / CHECK_NEAR.
//
// Usage:
//   #include "tests/check.hpp"
//   ...
//   CHECK(cond);                 // prints file:line + expression on failure
//   CHECK_EQ(got, want);         // + prints both arithmetic values
//   CHECK_NEAR(got, want, eps);  // floating-point tolerance compare
//   ...
//   int main() { runTests(); return orca::test::report("test_name"); }

#ifndef ORCA_TESTS_CHECK_HPP
#define ORCA_TESTS_CHECK_HPP

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>

namespace orca::test {

// One shared failure counter per test executable (inline → safe even if the
// header is pulled into multiple translation units of the same test).
inline int g_failures = 0;

// Stringify arithmetic operands for the *_EQ / *_NEAR diagnostics. Non-arithmetic
// operands fall back to "?" — use plain CHECK(a == b) for those.
template <typename T>
inline std::string toStr(const T& v) {
    if constexpr (std::is_floating_point_v<T>) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(v));
        return buf;
    } else if constexpr (std::is_integral_v<T>) {
        return std::to_string(static_cast<long long>(v));
    } else {
        return "?";
    }
}

inline void fail(const char* file, int line, const char* what) {
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, what);
    ++g_failures;
}

// main() epilogue: prints the summary and returns a process exit code.
inline int report(const char* name) {
    if (g_failures) {
        std::fprintf(stderr, "%s: %d check(s) failed\n", name, g_failures);
        return EXIT_FAILURE;
    }
    std::printf("%s: all checks passed\n", name);
    return EXIT_SUCCESS;
}

}  // namespace orca::test

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) ::orca::test::fail(__FILE__, __LINE__, #cond);        \
    } while (0)

#define CHECK_EQ(a, b)                                                          \
    do {                                                                       \
        const auto _ca = (a);                                                  \
        const auto _cb = (b);                                                  \
        if (!(_ca == _cb)) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s == %s  (%s vs %s)\n",         \
                         __FILE__, __LINE__, #a, #b,                           \
                         ::orca::test::toStr(_ca).c_str(),                     \
                         ::orca::test::toStr(_cb).c_str());                    \
            ++::orca::test::g_failures;                                        \
        }                                                                      \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                  \
    do {                                                                       \
        const double _na = static_cast<double>(a);                            \
        const double _nb = static_cast<double>(b);                            \
        if (!(std::fabs(_na - _nb) <= static_cast<double>(eps))) {            \
            std::fprintf(stderr, "FAIL %s:%d: |%s - %s| <= %s  (%.9g vs %.9g)\n", \
                         __FILE__, __LINE__, #a, #b, #eps, _na, _nb);          \
            ++::orca::test::g_failures;                                        \
        }                                                                      \
    } while (0)

#endif  // ORCA_TESTS_CHECK_HPP
