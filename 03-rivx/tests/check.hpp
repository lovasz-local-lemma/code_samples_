// backend/tests/check.hpp
// Minimal dependency-free test harness for the geometry core. 
#pragma once
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace rive_test
{
struct Case { std::string name; std::function<void()> fn; };
inline std::vector<Case>& registry() { static std::vector<Case> r; return r; }
inline int& failures() { static int f = 0; return f; }
struct Reg { Reg(const char* n, std::function<void()> fn) { registry().push_back({n, std::move(fn)}); } };

inline void fail(const char* expr, const char* file, int line)
{
    ++failures();
    std::printf("  FAIL: %s  (%s:%d)\n", expr, file, line);
}

#define RIVE_TEST(name) \
    static void name(); \
    static ::rive_test::Reg reg_##name(#name, name); \
    static void name()

#define RIVE_CHECK(cond) \
    do { if (!(cond)) ::rive_test::fail(#cond, __FILE__, __LINE__); } while (0)

#define RIVE_CHECK_NEAR(a, b, eps) \
    do { if (std::fabs(double((a)) - double((b))) > (eps)) \
        ::rive_test::fail(#a " ~= " #b, __FILE__, __LINE__); } while (0)

inline int run_all()
{
    for (auto& c : registry())
    {
        std::printf("[ RUN  ] %s\n", c.name.c_str());
        const int before = failures();
        c.fn();
        std::printf(before == failures() ? "[  OK  ] %s\n" : "[ FAIL ] %s\n",
                    c.name.c_str());
    }
    std::printf("\n%s (%d failure(s))\n",
                failures() == 0 ? "PASSED" : "FAILED", failures());
    return failures() == 0 ? 0 : 1;
}
} // namespace rive_test
