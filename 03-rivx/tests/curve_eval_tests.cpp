// backend/tests/curve_eval_tests.cpp
// GL-free unit tests for the .rivx curve evaluation core (curve_eval.hpp).
// No rive headers, no GL -- pure math checks, same harness style as
// splat_field_tests.cpp.
#include "curve/curve_eval.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace rive_backend;

static int g_failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++g_failures; } } while (0)

static float dist(CurvePt a, CurvePt b)
{
    const float dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

static bool allFinite(const std::vector<CurvePt>& v)
{
    for (const CurvePt& p : v)
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) return false;
    return true;
}

// A deliberately asymmetric pentagon: no collinear triplets, no symmetry that
// could mask a continuity bug.
static std::vector<CurvePt> pentagon()
{
    return {{0.0f, 0.0f}, {2.0f, 0.5f}, {3.0f, 2.0f}, {1.5f, 3.0f}, {-0.5f, 1.5f}};
}

// One-sided derivative estimates at sample index q from 4 consecutive samples
// (2nd-order accurate). dir = +1 looks forward (right limit), -1 backward
// (left limit). h is the parameter step between adjacent samples. Indices wrap
// when total > 0 (closed polylines). Computed in double to keep the h^2
// division from amplifying float noise.
static void sideDerivs(const std::vector<CurvePt>& poly, int q, int dir,
                       double h, int wrapTotal, double d1[2], double d2[2])
{
    auto at = [&](int i) -> CurvePt {
        int idx = q + dir * i;
        if (wrapTotal > 0) idx = ((idx % wrapTotal) + wrapTotal) % wrapTotal;
        return poly[(size_t)idx];
    };
    const CurvePt f0 = at(0), f1 = at(1), f2 = at(2), f3 = at(3);
    d1[0] = dir * (-3.0 * f0.x + 4.0 * f1.x - f2.x) / (2.0 * h);
    d1[1] = dir * (-3.0 * f0.y + 4.0 * f1.y - f2.y) / (2.0 * h);
    d2[0] = (2.0 * f0.x - 5.0 * f1.x + 4.0 * f2.x - f3.x) / (h * h);
    d2[1] = (2.0 * f0.y - 5.0 * f1.y + 4.0 * f2.y - f3.y) / (h * h);
}

// ---- (1) Bezier ------------------------------------------------------------

static void testBezierCubic()
{
    CurveSpec s;
    s.kind = CurveKind::Bezier;
    s.closed = false;
    s.pts = {{0.0f, 0.0f}, {1.0f, 2.0f}, {3.0f, 2.0f}, {4.0f, 0.0f}};
    const int m = 5; // S = npts-1 = 3 spans -> 13 samples, index 6 is t = 1/2
    auto poly = sampleCurve(s, m);
    CHECK(poly.size() == 13u, "bezier open: S*(m-1)+1 samples");
    CHECK(dist(poly.front(), s.pts.front()) < 1e-5f, "bezier interpolates p0 at t=0");
    CHECK(dist(poly.back(), s.pts.back()) < 1e-5f, "bezier interpolates p3 at t=1");
    // Closed form for a cubic at t = 1/2: (p0 + 3 p1 + 3 p2 + p3) / 8.
    const CurvePt mid = {(0.0f + 3.0f * 1.0f + 3.0f * 3.0f + 4.0f) / 8.0f,
                         (0.0f + 3.0f * 2.0f + 3.0f * 2.0f + 0.0f) / 8.0f};
    CHECK(dist(poly[6], mid) < 1e-5f, "de Casteljau matches cubic closed form at t=1/2");
    CHECK(allFinite(poly), "bezier open: finite samples");
}

static void testBezierClosedWrap()
{
    CurveSpec s;
    s.kind = CurveKind::Bezier;
    s.closed = true;
    s.pts = {{0.0f, 0.0f}, {2.0f, 0.0f}, {2.0f, 2.0f}, {0.0f, 2.0f}};
    const int m = 4; // S = npts = 4 (pts[0] appended) -> 4*3 = 12 samples
    auto poly = sampleCurve(s, m);
    CHECK(poly.size() == 12u, "bezier closed: S*(m-1) samples, wrap not duplicated");
    CHECK(dist(poly.front(), s.pts.front()) < 1e-5f, "bezier closed starts at p0");
    CHECK(dist(poly.back(), poly.front()) > 1e-4f,
          "bezier closed: last sample is NOT the duplicated wrap point");
    CHECK(allFinite(poly), "bezier closed: finite samples");
}

// ---- (2) BSpline -----------------------------------------------------------

static void testBSplineClampedEndpoints()
{
    CurveSpec s;
    s.kind = CurveKind::BSpline;
    s.closed = false;
    s.degree = 3;
    s.pts = {{0.0f, 0.0f}, {1.0f, 2.0f}, {3.0f, 3.0f}, {5.0f, 2.0f}, {6.0f, 0.0f}, {4.0f, -2.0f}};
    const int m = 8; // S = n - d = 3 -> 22 samples
    auto poly = sampleCurve(s, m);
    CHECK(poly.size() == 22u, "bspline open: (n-d)*(m-1)+1 samples");
    CHECK(dist(poly.front(), s.pts.front()) < 1e-4f, "clamped bspline interpolates first ctrl pt");
    CHECK(dist(poly.back(), s.pts.back()) < 1e-4f, "clamped bspline interpolates last ctrl pt");
    CHECK(allFinite(poly), "bspline open: finite samples");

    // Degree is clamped to npts-1: degree 9 with 4 points behaves as a cubic
    // Bezier (single span), still endpoint-interpolating.
    CurveSpec hi = s;
    hi.degree = 9;
    hi.pts = {{0.0f, 0.0f}, {1.0f, 2.0f}, {3.0f, 2.0f}, {4.0f, 0.0f}};
    CHECK(curveSpanCount(hi) == 1, "bspline degree clamps to npts-1 (one span)");
    auto hp = sampleCurve(hi, 5);
    CHECK(hp.size() == 5u, "bspline clamped-degree sample count");
    CHECK(dist(hp.front(), hi.pts.front()) < 1e-4f && dist(hp.back(), hi.pts.back()) < 1e-4f,
          "bspline clamped-degree still endpoint-interpolating");
}

static void testBSplineClosedSeam()
{
    CurveSpec s;
    s.kind = CurveKind::BSpline;
    s.closed = true;
    s.degree = 3;
    s.pts = {{0.0f, 0.0f}, {1.0f, 2.0f}, {3.0f, 3.0f}, {5.0f, 2.0f}, {6.0f, 0.0f}, {3.0f, -2.0f}};
    const int m = 9; // S = n = 6 -> 48 samples
    auto poly = sampleCurve(s, m);
    CHECK(poly.size() == 48u, "bspline closed: n*(m-1) samples");
    CHECK(allFinite(poly), "bspline closed: finite samples");

    // No seam, part 1 (C0 at the wrap): the implicit closing edge last->first
    // must be a normal sample step, not a jump.
    float maxStep = 0.0f;
    for (size_t i = 1; i < poly.size(); ++i) maxStep = std::max(maxStep, dist(poly[i - 1], poly[i]));
    CHECK(dist(poly.back(), poly.front()) < 3.0f * maxStep,
          "bspline closed: wrap edge is a normal step (C0 seam)");

    // No seam, part 2 (periodicity): rotating the control points by one is the
    // SAME loop reparameterized by one span, so rotated sample 0 must equal the
    // original curve evaluated at the wrap of span 0 -> sample m-1.
    CurveSpec rot = s;
    for (size_t i = 0; i < s.pts.size(); ++i) rot.pts[i] = s.pts[(i + 1) % s.pts.size()];
    auto rp = sampleCurve(rot, m);
    CHECK(dist(rp.front(), poly[(size_t)(m - 1)]) < 1e-4f,
          "bspline closed: control-point rotation == one-span parameter shift (no seam)");
}

// ---- (3) Yuksel interpolation property -------------------------------------

static void checkYukselInterpolates(CurveKind kind, bool closed, const char* name)
{
    CurveSpec s;
    s.kind = kind;
    s.closed = closed;
    s.pts = pentagon();
    const int m = 8;
    auto poly = sampleCurve(s, m);
    const size_t expect = closed ? 5u * 7u : 4u * 7u + 1u;
    char msg[96];
    std::snprintf(msg, sizeof(msg), "%s %s: sample count", name, closed ? "closed" : "open");
    CHECK(poly.size() == expect, msg);
    for (size_t i = 0; i < s.pts.size(); ++i)
    {
        const size_t idx = i * (size_t)(m - 1); // span boundaries hit the ctrl pts
        std::snprintf(msg, sizeof(msg), "%s %s: passes through ctrl pt %zu",
                      name, closed ? "closed" : "open", i);
        CHECK(idx < poly.size() && dist(poly[idx], s.pts[i]) < 1e-4f, msg);
    }
}

static void testYukselInterpolation()
{
    const CurveKind kinds[4] = {CurveKind::YukselCircular, CurveKind::YukselElliptical,
                                CurveKind::YukselHybrid, CurveKind::YukselParabolic};
    const char* names[4] = {"circular", "elliptical", "hybrid", "parabolic"};
    for (int k = 0; k < 4; ++k)
    {
        checkYukselInterpolates(kinds[k], false, names[k]);
        checkYukselInterpolates(kinds[k], true, names[k]);
    }
}

// ---- (4) Yuksel join continuity (GEOMETRIC G1 / G2) ------------------------
//
// We check GEOMETRIC continuity (tangent DIRECTION for G1, signed CURVATURE for
// G2), NOT raw parametric C1/C2. With the chord-length node the parabola/circle
// halves are reparameterized to each span's local t by a per-side affine map of
// DIFFERENT slope (1-u vs u), so the parameter SPEED jumps at a join even though
// the curve image is the paper's C2 curve: tangent direction and curvature stay
// continuous (G2), only |dC/dt| changes. Since we sample the image (for fills
// and strokes), geometric continuity is the property that matters; a real corner
// or curvature break still shows up as an O(1) jump. Both curvature and unit
// tangent are reparameterization-invariant, so they are computed straight from
// the one-sided derivatives. m = 65 gives h = 1/64 (2nd-order one-sided stencils).
//   G1 (unit tangent):  tol 2e-2  -- truncation ~1e-3; a corner is O(1).
//   G2 (curvature):     tol 0.05 + 15% -- 2nd-difference float noise / h^2 at
//                       unit scale; a genuine C2 break is O(100%).
static void checkJoins(CurveKind kind, bool closed, bool wantG2, const char* name)
{
    CurveSpec s;
    s.kind = kind;
    s.closed = closed;
    s.pts = pentagon();
    const int m = 129;
    const double h = 1.0 / 128.0;
    auto poly = sampleCurve(s, m);
    const int total = (int)poly.size();
    const int n = (int)s.pts.size();
    const int jFirst = closed ? 0 : 1;       // open ends are not joins
    const int jLast = closed ? n - 1 : n - 2;
    for (int j = jFirst; j <= jLast; ++j)
    {
        const int q = j * (m - 1);
        const int wrap = closed ? total : 0;
        double l1[2], l2[2], r1[2], r2[2];
        sideDerivs(poly, q, -1, h, wrap, l1, l2);
        sideDerivs(poly, q, +1, h, wrap, r1, r2);
        char msg[120];
        const double lmag = std::sqrt(l1[0] * l1[0] + l1[1] * l1[1]);
        const double rmag = std::sqrt(r1[0] * r1[0] + r1[1] * r1[1]);
        if (lmag < 1e-9 || rmag < 1e-9) continue; // tangent ill-defined (cusp test elsewhere)
        // G1: unit tangent DIRECTION continuous (speed may differ).
        const double dt = std::sqrt((l1[0] / lmag - r1[0] / rmag) * (l1[0] / lmag - r1[0] / rmag) +
                                    (l1[1] / lmag - r1[1] / rmag) * (l1[1] / lmag - r1[1] / rmag));
        std::snprintf(msg, sizeof(msg), "%s %s: G1 tangent dir at join %d (|dt|=%.2e)",
                      name, closed ? "closed" : "open", j, dt);
        CHECK(dt <= 2e-2, msg);
        if (wantG2)
        {
            // Signed curvature kappa = (x'y'' - y'x'') / |d1|^3 (invariant).
            const double kL = (l1[0] * l2[1] - l1[1] * l2[0]) / (lmag * lmag * lmag);
            const double kR = (r1[0] * r2[1] - r1[1] * r2[0]) / (rmag * rmag * rmag);
            const double ek = std::fabs(kL - kR);
            std::snprintf(msg, sizeof(msg), "%s %s: G2 curvature at join %d (|dk|=%.2e)",
                          name, closed ? "closed" : "open", j, ek);
            CHECK(ek <= 0.05 + 0.15 * std::max(std::fabs(kL), std::fabs(kR)), msg);
        }
    }
}

static void testYukselJoinContinuity()
{
    // G1 for every variant; G2 (curvature continuity) for the circular /
    // elliptical / hybrid bases per the paper. Parabolic is only advertised G1
    // by the .rivx spec.
    checkJoins(CurveKind::YukselCircular, true, true, "circular");
    checkJoins(CurveKind::YukselCircular, false, true, "circular");
    checkJoins(CurveKind::YukselElliptical, true, true, "elliptical");
    checkJoins(CurveKind::YukselElliptical, false, true, "elliptical");
    checkJoins(CurveKind::YukselHybrid, true, true, "hybrid");
    checkJoins(CurveKind::YukselHybrid, false, true, "hybrid");
    checkJoins(CurveKind::YukselParabolic, true, false, "parabolic");
    checkJoins(CurveKind::YukselParabolic, false, false, "parabolic");
}

// ---- (5) degenerate inputs --------------------------------------------------

static void testDegenerateInputs()
{
    const CurveKind kinds[6] = {CurveKind::Bezier, CurveKind::BSpline,
                                CurveKind::YukselCircular, CurveKind::YukselElliptical,
                                CurveKind::YukselHybrid, CurveKind::YukselParabolic};
    // Collinear run WITH a duplicated point (radius -> inf for the circular
    // base, zero-length chords for everything else).
    const std::vector<CurvePt> line = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f},
                                       {2.0f, 0.0f}, {3.0f, 0.0f}};
    // Every point identical.
    const std::vector<CurvePt> same(4, CurvePt{2.0f, 3.0f});
    for (int k = 0; k < 6; ++k)
        for (int c = 0; c < 2; ++c)
        {
            CurveSpec s;
            s.kind = kinds[k];
            s.closed = (c == 1);
            s.pts = line;
            char msg[96];
            std::snprintf(msg, sizeof(msg), "kind %d %s: collinear+duplicate stays finite",
                          k, c ? "closed" : "open");
            CHECK(allFinite(sampleCurve(s, 6)), msg);
            s.pts = same;
            std::snprintf(msg, sizeof(msg), "kind %d %s: all-identical pts stay finite",
                          k, c ? "closed" : "open");
            CHECK(allFinite(sampleCurve(s, 6)), msg);
        }

    // Collinear input must yield a collinear curve for every Yuksel base (the
    // circular fallback IS the straight line; ellipse/parabola degenerate to
    // the chord).
    for (int k = 2; k < 6; ++k)
        for (int c = 0; c < 2; ++c)
        {
            CurveSpec s;
            s.kind = kinds[k];
            s.closed = (c == 1);
            s.pts = {{0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}};
            auto poly = sampleCurve(s, 9);
            float maxY = 0.0f;
            for (const CurvePt& p : poly) maxY = std::max(maxY, std::fabs(p.y));
            char msg[96];
            std::snprintf(msg, sizeof(msg), "kind %d %s: collinear input stays on the line",
                          k, c ? "closed" : "open");
            CHECK(allFinite(poly) && maxY < 1e-4f, msg);
        }
}

// ---- (6) sample counts ------------------------------------------------------

static void testSampleCounts()
{
    CurveSpec y;
    y.kind = CurveKind::YukselHybrid;
    y.pts = pentagon();
    y.closed = true;
    CHECK(curveSpanCount(y) == 5, "yuksel closed: span count n");
    CHECK(sampleCurve(y, 4).size() == 5u * 3u, "yuksel closed: S*(m-1) samples");
    // m = 2 closed -> exactly the control points (t = 0 of every span).
    auto coarse = sampleCurve(y, 2);
    CHECK(coarse.size() == 5u, "yuksel closed m=2: one sample per span");
    bool onPts = coarse.size() == 5u;
    for (size_t i = 0; i < coarse.size() && onPts; ++i)
        onPts = dist(coarse[i], y.pts[i]) < 1e-5f;
    CHECK(onPts, "yuksel closed m=2: samples are the control points");

    y.closed = false;
    CHECK(curveSpanCount(y) == 4, "yuksel open: span count n-1");
    CHECK(sampleCurve(y, 4).size() == 4u * 3u + 1u, "yuksel open: S*(m-1)+1 samples");

    // samplesPerSpan clamps to >= 2 (0, 1 and negatives behave like 2).
    CHECK(sampleCurve(y, 1).size() == sampleCurve(y, 2).size(), "samplesPerSpan=1 clamps to 2");
    CHECK(sampleCurve(y, 0).size() == sampleCurve(y, 2).size(), "samplesPerSpan=0 clamps to 2");
    CHECK(sampleCurve(y, -7).size() == sampleCurve(y, 2).size(), "negative samplesPerSpan clamps");

    // Bezier / BSpline documented counts.
    CurveSpec b;
    b.kind = CurveKind::Bezier;
    b.pts = {{0.0f, 0.0f}, {1.0f, 2.0f}, {3.0f, 2.0f}, {4.0f, 0.0f}};
    b.closed = false;
    CHECK(sampleCurve(b, 4).size() == 10u, "bezier open count (n-1)*(m-1)+1");
    b.closed = true;
    CHECK(sampleCurve(b, 4).size() == 12u, "bezier closed count n*(m-1)");
    CurveSpec bs;
    bs.kind = CurveKind::BSpline;
    bs.degree = 3;
    bs.pts = {{0.0f, 0.0f}, {1.0f, 2.0f}, {3.0f, 3.0f}, {5.0f, 2.0f}, {6.0f, 0.0f}, {4.0f, -2.0f}};
    bs.closed = false;
    CHECK(sampleCurve(bs, 4).size() == 10u, "bspline open count (n-d)*(m-1)+1");
    bs.closed = true;
    CHECK(sampleCurve(bs, 4).size() == 18u, "bspline closed count n*(m-1)");

    // Degenerates: too few points -> the points themselves.
    CurveSpec d;
    d.kind = CurveKind::YukselCircular;
    d.closed = true;
    d.pts = {{1.0f, 1.0f}, {4.0f, 2.0f}};
    auto two = sampleCurve(d, 16);
    CHECK(two.size() == 2u && dist(two[0], d.pts[0]) < 1e-6f && dist(two[1], d.pts[1]) < 1e-6f,
          "yuksel npts<3: polyline of the points themselves");
    d.pts = {{1.0f, 1.0f}};
    CHECK(sampleCurve(d, 16).size() == 1u, "single point: single sample");
    d.pts.clear();
    CHECK(sampleCurve(d, 16).empty(), "no points: empty polyline");
    d.kind = CurveKind::Bezier;
    d.pts = {{1.0f, 1.0f}};
    CHECK(sampleCurve(d, 16).size() == 1u, "bezier single point: single sample");
}

// ---- (9) Visualization helpers (base weights + local interpolants) ---------

static void testBaseWeights() // 
{
    // Hybrid pentagon: weights parallel to the polyline, each sample sums to 1.
    CurveSpec s;
    s.kind = CurveKind::YukselHybrid;
    s.closed = true;
    s.pts = pentagon();
    const std::vector<CurvePt> poly = sampleCurve(s, 8);
    const std::vector<CurveBaseWeights> w = sampleCurveBaseWeights(s, 8);
    CHECK(w.size() == poly.size(), "viz: weights parallel to sampleCurve");
    bool sum1 = true, inRange = true;
    for (const CurveBaseWeights& bw : w)
    {
        if (std::fabs(bw.ellipse + bw.circle + bw.parabola - 1.0f) > 1e-4f)
            sum1 = false;
        if (bw.ellipse < -1e-6f || bw.circle < -1e-6f || bw.parabola < -1e-6f)
            inRange = false;
    }
    CHECK(sum1, "viz: hybrid base weights sum to 1 at every sample");
    CHECK(inRange, "viz: base weights non-negative");
    // Pure kinds attribute 100% to their own base.
    s.kind = CurveKind::YukselElliptical;
    for (const CurveBaseWeights& bw : sampleCurveBaseWeights(s, 4))
        if (std::fabs(bw.ellipse - 1.0f) > 1e-6f) { CHECK(false, "viz: elliptical kind -> all ellipse"); break; }
    s.kind = CurveKind::YukselParabolic;
    for (const CurveBaseWeights& bw : sampleCurveBaseWeights(s, 4))
        if (std::fabs(bw.parabola - 1.0f) > 1e-6f) { CHECK(false, "viz: parabolic kind -> all parabola"); break; }
    // Non-Yuksel kinds: empty.
    s.kind = CurveKind::BSpline;
    CHECK(sampleCurveBaseWeights(s, 4).empty(), "viz: BSpline -> no base weights");
}

static void testLocalInterpolant() 
{
    CurveSpec s;
    s.kind = CurveKind::YukselHybrid;
    s.closed = true;
    s.pts = pentagon();
    const int n = (int)s.pts.size();
    for (int i = 0; i < n; ++i)
    {
        LocalInterpolantInfo info;
        const std::vector<CurvePt> f = sampleLocalInterpolant(s, i, 33, &info);
        CHECK(f.size() == 33, "viz: local interpolant sample count");
        if (f.size() != 33) continue;
        CHECK(allFinite(f), "viz: local interpolant finite");
        // F_i passes through its triplet at s = 0, 1/2, 1.
        CHECK(dist(f.front(), s.pts[(size_t)((i - 1 + n) % n)]) < 1e-3f,
              "viz: F_i(0) = p_{i-1}");
        CHECK(dist(f[16], s.pts[(size_t)i]) < 1e-3f, "viz: F_i(1/2) = p_i");
        CHECK(dist(f.back(), s.pts[(size_t)((i + 1) % n)]) < 1e-3f,
              "viz: F_i(1) = p_{i+1}");
        CHECK(std::fabs(info.ellipse + info.circle + info.parabola - 1.0f) < 1e-4f,
              "viz: local mix sums to 1");
    }
    // Open curve: end indices have no interpolant.
    s.closed = false;
    CHECK(sampleLocalInterpolant(s, 0, 9).empty(), "viz: open curve has no F_0");
    CHECK(!sampleLocalInterpolant(s, 1, 9).empty(), "viz: open curve has F_1");
    CHECK(sampleLocalInterpolant(s, n - 1, 9).empty(), "viz: open curve has no F_{n-1}");
}

// ---- (Yuksel) per-SEGMENT self-intersection ---------------------------------
// Yuksel 2020 guarantees each curve SEGMENT (the piece between two consecutive
// control points) of the CIRCULAR variant is self-intersection-free for ANY
// control points. Global self-intersection between different segments (e.g. a
// figure-8 waist) is allowed and NOT checked here -- we test each span in
// isolation. The break used to be a fixed 1/2 interpolation node, which let a
// lopsided triplet's local interpolant turn back on itself; the chord-length
// node (solveYukselTi) keeps every parabola/circle/hybrid span clean. The pure
// elliptical base is the one exception (see below). 
static bool segProperCross(CurvePt a, CurvePt b, CurvePt c, CurvePt d)
{
    auto cr = [](CurvePt o, CurvePt p, CurvePt q) {
        return (double)(p.x - o.x) * (q.y - o.y) - (double)(p.y - o.y) * (q.x - o.x);
    };
    const double d1 = cr(c, d, a), d2 = cr(c, d, b);
    const double d3 = cr(a, b, c), d4 = cr(a, b, d);
    return ((d1 > 0) != (d2 > 0)) && ((d3 > 0) != (d4 > 0)); // proper crossing only
}
static bool spanLoops(const std::vector<CurvePt>& seg)
{
    const int e = (int)seg.size() - 1; // edge count
    for (int i = 0; i < e; ++i)
        for (int j = i + 2; j < e; ++j) // non-adjacent edges
            if (segProperCross(seg[i], seg[i + 1], seg[j], seg[j + 1]))
                return true;
    return false;
}
static void testYukselSegmentNoSelfIntersect()
{
    const int m = 32;             // samples per span
    const int per = m - 1;
    const CurveKind kinds[] = {CurveKind::YukselHybrid, CurveKind::YukselCircular,
                               CurveKind::YukselElliptical, CurveKind::YukselParabolic};
    auto anyLoop = [&](CurveKind kind, const std::vector<CurvePt>& pts) {
        CurveSpec s; s.kind = kind; s.closed = true; s.pts = pts;
        const std::vector<CurvePt> poly = sampleCurve(s, m);
        const int S = curveSpanCount(s);
        const int total = (int)poly.size();
        for (int sp = 0; sp < S; ++sp)
        {
            std::vector<CurvePt> seg;
            seg.reserve(per + 1);
            for (int k = 0; k <= per; ++k)
                seg.push_back(poly[(size_t)((sp * per + k) % total)]);
            if (spanLoops(seg)) return true;
        }
        return false;
    };
    // The pure ELLIPTICAL base is NOT guaranteed loop-free (conjugate-diameter
    // shape + quadratic phi on extreme triplets); it is reported, never asserted.
    // The Circular/Parabolic/Hybrid bases ARE hard-asserted. 
    auto checkOrReport = [&](CurveKind kind, const std::vector<CurvePt>& pts,
                             const char* tag) {
        if (kind == CurveKind::YukselElliptical)
        {
            if (anyLoop(kind, pts))
                std::printf("  KNOWN: elliptical bulge (%s)\n", tag);
            return;
        }
        CHECK(!anyLoop(kind, pts),
              "yuksel SEGMENT self-intersection-free (circular/parabolic/hybrid)");
    };
    // HARD assertion: moderately lopsided triplets (a vertex clearly closer to one
    // neighbour than the other). The chord-length node keeps every Circular /
    // Parabolic / Hybrid span loop-free per Yuksel.
    const std::vector<std::vector<CurvePt>> ok = {
        {{-6, 0}, {-1, 0.1f}, {6, 0}, {5.5f, 0.2f}, {0, -7}},      // two lopsided in a row
        {{0, 0}, {0.5f, 5.0f}, {1.0f, 0.1f}, {8, 0.2f}, {4, -6}},  // tall + lopsided
        {{0, 0}, {3, 0.3f}, {10, 0}, {5, -7}},                     // mild lopsided
    };
    for (int ki = 0; ki < 4; ++ki)
        for (const std::vector<CurvePt>& pts : ok)
            checkOrReport(kinds[ki], pts, "lopsided");
    // NEAR-DEGENERATE triplets: a vertex almost coincident with a neighbour. The
    // chord-length node makes CIRCULAR, PARABOLIC and HYBRID loop-free even here,
    // so those are now HARD asserted. (kinds[] = {hybrid, circular, elliptical,
    // parabolic}.) 
    const std::vector<std::vector<CurvePt>> degenerate = {
        {{0, 0}, {1.0f, 0.2f}, {10, 0}, {5, -8}},   // p1 nearly on p0
        {{0, 0}, {9.0f, -0.3f}, {10, 0}, {5, 9}},   // p1 nearly on p2
    };
    for (int ki = 0; ki < 4; ++ki)
        for (const std::vector<CurvePt>& pts : degenerate)
            checkOrReport(kinds[ki], pts, "near-degenerate");
}

int main()
{
    testBezierCubic();
    testBezierClosedWrap();
    testBSplineClampedEndpoints();
    testBSplineClosedSeam();
    testYukselInterpolation();
    testYukselJoinContinuity();
    testDegenerateInputs();
    testSampleCounts();
    testBaseWeights();        // 
    testLocalInterpolant();   // 
    testYukselSegmentNoSelfIntersect(); // 
    if (g_failures == 0) std::printf("all curve_eval tests passed\n");
    return g_failures == 0 ? 0 : 1;
}
