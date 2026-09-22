// backend/src/curve_eval.cpp
//
// Math notes
// ----------
// Bezier: de Casteljau recursion (repeated lerp), stable for any degree because
// every intermediate value is a convex combination of control points.
//
// BSpline: de Boor recursion. Open curves use a CLAMPED uniform knot vector
//   U[i] = 0 (i <= d), i-d (d < i < n), n-d (i >= n)
// so the curve starts at pts[0] and ends at pts[n-1]. Closed curves use plain
// uniform knots U[i] = i with control indices taken mod n (periodic spline):
// every span looks the same, so the seam is as smooth as the interior
// (C^{d-1}).
//
// Yuksel class (Cem Yuksel, "A Class of C2 Interpolating Splines", ACM TOG
// 39(5), 2020): each control point p_i owns a local interpolant F_i(s) fitted
// to the triplet (p_{i-1}, p_i, p_{i+1}) at s = 0, u_i, 1. The node u_i is NOT a
// fixed 1/2 -- it is the chord-length parameter the paper prescribes (the root
// in (0,1) of a cubic in the inter-point distances, see solveYukselTi). Placing
// p_i at its chord-length parameter keeps each F_i monotonic (no mid-segment
// turn-back), the property that makes the CIRCULAR variant free of per-segment
// self-intersections regardless of point placement; a fixed 1/2 node made
// lopsided triplets bulge back on themselves. ALL bases share the one node u
// (the ellipse via a quadratic phi(s)) so a join's two sides have collinear
// tangents for every kind including Hybrid (a per-base node would kink it). The
// only cost: for an EXTREME triplet (u near 0 or 1) the ellipse's quadratic phi
// can be non-monotonic, so the PURE elliptical arc may swing back -- a known
// limitation of that variant. The segment from p_j to p_{j+1} blends the two
// interpolants that overlap it:
//   C(t) = cos^2(pi/2 t) * F_j(u_j + (1-u_j) t) + sin^2(pi/2 t) * F_{j+1}(u_{j+1} t)
// Why this is smooth at p_i: the blend weight of the "other" interpolant and
// its FIRST derivative vanish at each segment end, and the interpolant
// DIFFERENCE (which the second derivative of the weight multiplies) vanishes
// there too because ALL THREE interpolants meeting at a join pass through p_i.
// So from both sides the curve agrees with F_i up to the second derivative.
// With the chord-length node u_i, each segment is reparameterized to its own
// t in [0,1] by a DIFFERENT affine map per side (slopes 1-u_j and u_{j+1}),
// so the global parameter speed is not continuous at a join -- but the IMAGE is
// exactly the paper's C2 curve (the s(t) maps are the same affine functions of
// the paper's global knot parameter), hence the tangent DIRECTION and the
// curvature are continuous: the rendered geometry is G2 (visually seamless).
// We sample the image, so this is the property we need.
//
// Bases (each fits one triplet, s in [0,1], F(0)=p_{i-1}, F(u)=p_i,
// F(1)=p_{i+1}):
//  * Parabolic: quadratic Lagrange in power basis, always defined.
//  * Circular: circumcircle, angle alpha(s) = quadratic through the three
//    unwrapped point angles (a single LINEAR alpha cannot hit three unevenly
//    spaced angles; the quadratic is the paper's formulation). The traversal
//    direction is the triangle orientation: circumcircle vertices appear in
//    triangle order around the circle. Collinear guard: |2*area| <=
//    1e-4 * maxChord^2 -> fall back to the parabolic base. Rationale: beyond
//    that the circumradius exceeds ~5000x the local point scale and the
//    centre+radius float representation starts to eat visible precision; the
//    parabolic image for exactly collinear points IS the straight line, so the
//    fallback is what the limit radius->inf looks like and never NaNs.
//  * Elliptical: conjugate-diameter (affine-mapped circle) ellipse, see the
//    header. Defined for every input (collinear -> degenerates to the chord).
//  * Hybrid: pointwise mix w*elliptical + (1-w)*circular with w constant per
//    triplet (see header for the asymmetry criterion). A constant mix of two
//    C2 interpolants through the same triplet is itself a C2 interpolant
//    through that triplet, so the blending theorem above still applies.
//
// Continuity summary: Bezier closed seam C0; open clamped BSpline C^{d-1}
// interior; closed BSpline C^{d-1} everywhere; YukselCircular / Elliptical /
// Hybrid C2 at joins; YukselParabolic guaranteed C1 (in fact C2 with this
// uniform parameterization -- it is the paper's quadratic-Bezier base -- but
// we only advertise/test C1 for it per the .rivx spec).
#include "curve/curve_eval.hpp"
#include <algorithm>
#include <cmath>

namespace rive_backend
{
namespace
{
constexpr float kPi = 3.14159265358979323846f;

inline CurvePt lerpPt(CurvePt a, CurvePt b, float t)
{
    return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t};
}

// ---- Bezier ---------------------------------------------------------------

CurvePt deCasteljau(const CurvePt* ctrl, int count, float t,
                    std::vector<CurvePt>& scratch)
{
    scratch.assign(ctrl, ctrl + count);
    for (int r = count - 1; r >= 1; --r)
        for (int j = 0; j < r; ++j)
            scratch[(size_t)j] = lerpPt(scratch[(size_t)j], scratch[(size_t)j + 1], t);
    return scratch[0];
}

// ---- BSpline (de Boor) ----------------------------------------------------

inline float clampedKnot(int i, int n, int d)
{
    if (i <= d) return 0.0f;
    if (i >= n) return (float)(n - d);
    return (float)(i - d);
}

// Open clamped uniform B-spline, u in [0, n-d]. Degree d <= 7.
CurvePt deBoorClamped(const std::vector<CurvePt>& p, int d, float u)
{
    const int n = (int)p.size();
    u = std::min(std::max(u, 0.0f), (float)(n - d));
    int span = d + (int)u;                       // U[span] <= u < U[span+1]
    span = std::min(span, n - 1);                // u == n-d lands in the last span
    CurvePt c[8];                                // d+1 <= 8 working points
    for (int j = 0; j <= d; ++j) c[j] = p[(size_t)(span - d + j)];
    for (int r = 1; r <= d; ++r)
        for (int j = d; j >= r; --j)
        {
            const int i = j + span - d;
            const float lo = clampedKnot(i, n, d);
            const float den = clampedKnot(i + d - r + 1, n, d) - lo;
            const float a = den > 0.0f ? (u - lo) / den : 0.0f;
            c[j] = lerpPt(c[j - 1], c[j], a);
        }
    return c[d];
}

// Closed periodic uniform B-spline: knots U[i] = i, control points mod n,
// u in [0, n) covers the whole loop once. Degree d <= 7, any n >= 2.
CurvePt deBoorPeriodic(const std::vector<CurvePt>& p, int d, float u)
{
    const int n = (int)p.size();
    const float period = (float)n;
    u -= period * std::floor(u / period);        // wrap into [0, n)
    const float x = u + (float)d;                // shift into the valid window
    int span = (int)x;
    span = std::min(span, n + d - 1);
    CurvePt c[8];
    for (int j = 0; j <= d; ++j) c[j] = p[(size_t)((span - d + j) % n)];
    for (int r = 1; r <= d; ++r)
        for (int j = d; j >= r; --j)
        {
            const int i = j + span - d;
            // Uniform knots: denominator is the constant knot distance d-r+1.
            const float a = (x - (float)i) / (float)(d - r + 1);
            c[j] = lerpPt(c[j - 1], c[j], a);
        }
    return c[d];
}

// ---- Yuksel-class local interpolants ---------------------------------------

// One 3-point interpolant F_i, fitted once per control point and evaluated
// many times per frame. All four bases are precomputed where cheap so Hybrid
// can mix without refitting.
struct LocalFit
{
    // Node u in (0,1): the parameter at which this interpolant passes through
    // its MIDDLE point p1 (p0 at s=0, p1 at s=u, p2 at s=1). Chosen per triplet
    // from the chord lengths (Yuksel 2020 parameterization, NOT a fixed 1/2) --
    // this is what keeps a lopsided triplet's interpolant free of a mid-segment
    // turn-back, the deviation a fixed 1/2 node could not avoid.
    float u = 0.5f;
    // Quadratic Lagrange through (0, p0), (u, p1), (1, p2) in power basis:
    // Q(s) = qa + qb*s + qc*s^2. Parabolic base AND the circular fallback.
    CurvePt qa{}, qb{}, qc{};
    // Circumcircle base: centre, radius, angle alpha(s) = aa + ab*s + ac*s^2
    // (quadratic through the three point angles at s = 0, u, 1).
    bool hasCircle = false;
    float ccx = 0.0f, ccy = 0.0f, crad = 0.0f;
    float aa = 0.0f, ab = 0.0f, ac = 0.0f;
    // Conjugate-diameter ellipse: E(s) = em + e1*sin(phi(s)) + e2*cos(phi(s)),
    // phi(s) quadratic through (0,-pi/2), (u,0), (1,pi/2) so E hits p0,p1,p2 at
    // the SAME node u as the other bases (reduces to pi*(s-1/2) when u = 1/2). A
    // single shared node per fit is what keeps Hybrid's tangent continuous at a
    // join (both sides' tangent are scalar multiples of H'(u)); giving the
    // ellipse a different node would put a small kink in Hybrid. The price is
    // that for an EXTREME triplet (u near 0 or 1) the quadratic phi can be non-
    // monotonic, so the PURE elliptical arc may swing back -- a known limitation
    // of the elliptical variant (Circular/Hybrid stay loop-free).
    CurvePt em{}, e1{}, e2{};
    float epa = 0.0f, epb = 0.0f, epc = 0.0f;
    // Hybrid mix weight of the elliptical base (constant per triplet).
    float wEll = 1.0f;
};

// Power-basis coefficients (a + b*s + c*s^2) of the quadratic taking the three
// values v0, v1, v2 at the nodes s = 0, u, 1. The single shared parameterization
// helper for every Yuksel base (points componentwise, angles, ellipse phi).
struct Quad3 { float a, b, c; };
inline Quad3 lagrange3(float u, float v0, float v1, float v2)
{
    const float iu = 1.0f / u, i1u = 1.0f / (1.0f - u);
    const float iuu = 1.0f / (u * (u - 1.0f));
    const float a = v0;
    const float b = v0 * (-(u + 1.0f) * iu) + v1 * (-iuu) + v2 * (-u * i1u);
    const float c = v0 * iu + v1 * iuu + v2 * i1u;
    return {a, b, c};
}

// Wrap an angle delta into the traversal direction: ccw -> [0, 2pi),
// cw -> (-2pi, 0].
inline float wrapDelta(float delta, bool ccw)
{
    const float twoPi = 2.0f * kPi;
    delta -= twoPi * std::floor(delta / twoPi);  // [0, 2pi)
    if (!ccw && delta > 0.0f) delta -= twoPi;
    return delta;
}

// Yuksel's per-triplet interpolation parameter t_i: the root in (0,1) of
//   |p2-p0|^2 t^3 + 3(p2-p0).(p0-p1) t^2 + (3p0-2p1-p2).(p0-p1) t - |p0-p1|^2
// (Yuksel 2020 / reference impl). f(0) = -|p0-p1|^2 <= 0 and f(1) = |p2-p1|^2
// >= 0, so a sign-change root always exists in [0,1]; bisection is robust and
// branch-free. This places p1 at the chord-length-appropriate parameter, which
// is precisely what keeps the local interpolant monotonic (no mid-segment turn-
// back) even when one edge of the triplet is far longer than the other -- the
// failure a fixed 1/2 node could not avoid.
inline float solveYukselTi(CurvePt p0, CurvePt p1, CurvePt p2)
{
    const float ax = p2.x - p0.x, ay = p2.y - p0.y;       // a = p2 - p0
    const float dx = p1.x - p0.x, dy = p1.y - p0.y;       // d = p1 - p0
    const float a2 = ax * ax + ay * ay;
    if (a2 < 1e-20f) return 0.5f;                         // p0 == p2: stay symmetric
    const float c3 = a2;
    const float c2 = -3.0f * (ax * dx + ay * dy);         // 3 a.(p0-p1)
    const float c1 = (2.0f * dx + ax) * dx + (2.0f * dy + ay) * dy; // (2d+a).d
    const float c0 = -(dx * dx + dy * dy);
    float lo = 0.0f, hi = 1.0f;
    for (int it = 0; it < 40; ++it)
    {
        const float m = 0.5f * (lo + hi);
        const float fm = ((c3 * m + c2) * m + c1) * m + c0;
        if (fm <= 0.0f) lo = m; else hi = m;
    }
    const float ti = 0.5f * (lo + hi);
    return std::min(std::max(ti, 1e-3f), 1.0f - 1e-3f);
}

LocalFit buildFit(CurvePt p0, CurvePt p1, CurvePt p2)
{
    LocalFit f;
    const float u = solveYukselTi(p0, p1, p2);
    f.u = u;
    // Quadratic Lagrange power basis (nodes 0, u, 1).
    const Quad3 qx = lagrange3(u, p0.x, p1.x, p2.x);
    const Quad3 qy = lagrange3(u, p0.y, p1.y, p2.y);
    f.qa = {qx.a, qy.a};
    f.qb = {qx.b, qy.b};
    f.qc = {qx.c, qy.c};
    // Ellipse (always defined; collinear/duplicate degenerate gracefully).
    f.em = {0.5f * (p0.x + p2.x), 0.5f * (p0.y + p2.y)};
    f.e1 = {0.5f * (p2.x - p0.x), 0.5f * (p2.y - p0.y)};
    f.e2 = {p1.x - f.em.x, p1.y - f.em.y};
    // phi(s) quadratic through (0,-pi/2), (u,0), (1,pi/2): keeps E(0)=p0,
    // E(u)=p1, E(1)=p2 at the shared node u (reduces to pi*(s-1/2) at u=1/2).
    const Quad3 ph = lagrange3(u, -0.5f * kPi, 0.0f, 0.5f * kPi);
    f.epa = ph.a; f.epb = ph.b; f.epc = ph.c;
    // Circumcircle, guarded against (near-)collinear triplets.
    const float bx = p1.x - p0.x, by = p1.y - p0.y;
    const float cx = p2.x - p0.x, cy = p2.y - p0.y;
    const float dx = p2.x - p1.x, dy = p2.y - p1.y;
    const float cross = bx * cy - by * cx;       // 2 * signed triangle area
    const float scale2 = std::max(bx * bx + by * by,
                         std::max(cx * cx + cy * cy, dx * dx + dy * dy));
    if (std::fabs(cross) > 1e-4f * scale2 && scale2 > 0.0f)
    {
        const float b2 = bx * bx + by * by;
        const float c2 = cx * cx + cy * cy;
        const float inv = 1.0f / (2.0f * cross);
        const float ux = (cy * b2 - by * c2) * inv;   // centre relative to p0
        const float uy = (bx * c2 - cx * b2) * inv;
        f.ccx = p0.x + ux;
        f.ccy = p0.y + uy;
        f.crad = std::sqrt(ux * ux + uy * uy);
        // Point angles, unwrapped in triangle orientation (circumcircle
        // vertices appear in triangle order around the circle).
        const bool ccw = cross > 0.0f;
        const float t0 = std::atan2(p0.y - f.ccy, p0.x - f.ccx);
        float t1 = std::atan2(p1.y - f.ccy, p1.x - f.ccx);
        float t2 = std::atan2(p2.y - f.ccy, p2.x - f.ccx);
        t1 = t0 + wrapDelta(t1 - t0, ccw);
        t2 = t1 + wrapDelta(t2 - t1, ccw);
        // alpha(s): quadratic through (0,t0), (u,t1), (1,t2).
        const Quad3 al = lagrange3(u, t0, t1, t2);
        f.aa = al.a; f.ab = al.b; f.ac = al.c;
        f.hasCircle = true;
        // MONOTONIC-ANGLE GUARD (Yuksel 2020 per-segment loop-freedom). The
        // quadratic angle alpha(s) must not swing back on [0,1], i.e. alpha'(0)=ab
        // and alpha'(1)=ab+2ac must share sign. With the chord-length node this
        // rarely trips; when it does we drop to the parabola (now itself turn-
        // back-free) rather than risk a mid-arc reversal.
        const float dA0 = f.ab, dA1 = f.ab + 2.0f * f.ac;
        if (dA0 * dA1 < 0.0f)
            f.hasCircle = false;
    }
    // Hybrid weight: how far the apex projects from the chord midpoint, in
    // half-chord units (0 -> pure ellipse, >= 1 -> pure "other"). Computed even
    // when the circle was dropped, so Hybrid then mixes the ellipse with the
    // turn-back-free PARABOLA instead of collapsing to a possibly-bulging pure
    // ellipse on a lopsided triplet.
    const float e1len2 = f.e1.x * f.e1.x + f.e1.y * f.e1.y;
    if (e1len2 > 1e-12f)
    {
        const float asym = std::fabs(f.e2.x * f.e1.x + f.e2.y * f.e1.y) / e1len2;
        f.wEll = std::min(std::max(1.0f - asym, 0.0f), 1.0f);
    }
    return f; // e1 ~ 0 (p0 == p2): keep wEll = 1, the ellipse handles it.
}

inline CurvePt evalQuad(const LocalFit& f, float s)
{
    return {f.qa.x + (f.qb.x + f.qc.x * s) * s,
            f.qa.y + (f.qb.y + f.qc.y * s) * s};
}

inline CurvePt evalCircle(const LocalFit& f, float s)
{
    const float a = f.aa + (f.ab + f.ac * s) * s;
    return {f.ccx + f.crad * std::cos(a), f.ccy + f.crad * std::sin(a)};
}

inline CurvePt evalEllipse(const LocalFit& f, float s)
{
    const float phi = f.epa + (f.epb + f.epc * s) * s;
    const float sn = std::sin(phi), cs = std::cos(phi);
    return {f.em.x + f.e1.x * sn + f.e2.x * cs,
            f.em.y + f.e1.y * sn + f.e2.y * cs};
}

// Evaluate the local interpolant for one HALF of a blend, at segment parameter
// t in [0,1]. 'secondHalf' = true traverses [u,1] (p_i -> p_{i+1}); false
// traverses [0,u] (p_{i-1} -> p_i). ALL bases share the chord-length node u
// (the ellipse via its quadratic phi), so at t=0 every base lands on p_i and at
// t=1 on its endpoint -- a SINGLE node per fit, which is what keeps Hybrid's
// tangent continuous at joins.
CurvePt evalSeg(const LocalFit& f, CurveKind kind, bool secondHalf, float t)
{
    const float s = secondHalf ? f.u + (1.0f - f.u) * t : f.u * t;
    switch (kind)
    {
        case CurveKind::YukselParabolic:
            return evalQuad(f, s);
        case CurveKind::YukselCircular:
            return f.hasCircle ? evalCircle(f, s) : evalQuad(f, s);
        case CurveKind::YukselElliptical:
            return evalEllipse(f, s);
        default: // YukselHybrid
        {
            const CurvePt e = evalEllipse(f, s);
            const CurvePt o = f.hasCircle ? evalCircle(f, s) : evalQuad(f, s);
            const float w = f.wEll;
            return {e.x * w + o.x * (1.0f - w), e.y * w + o.y * (1.0f - w)};
        }
    }
}

// The constant base mix one fit contributes under 'kind' (the circle's
// collinear fallback to the parabola already applied).
LocalInterpolantInfo fitBaseMix(const LocalFit& f, CurveKind kind)
{
    LocalInterpolantInfo m;
    switch (kind)
    {
        case CurveKind::YukselParabolic:
            m.parabola = 1.0f;
            break;
        case CurveKind::YukselCircular:
            (f.hasCircle ? m.circle : m.parabola) = 1.0f;
            break;
        case CurveKind::YukselElliptical:
            m.ellipse = 1.0f;
            break;
        default: // YukselHybrid
            m.ellipse = f.wEll;
            (f.hasCircle ? m.circle : m.parabola) = 1.0f - f.wEll;
            break;
    }
    return m;
}

bool isYuksel(CurveKind k)
{
    return k != CurveKind::Bezier && k != CurveKind::BSpline;
}
} // namespace

int curveSpanCount(const CurveSpec& spec)
{
    const int n = (int)spec.pts.size();
    switch (spec.kind)
    {
        case CurveKind::Bezier:
        case CurveKind::CubicPen: // chain of cubic segments (one per edge)
            return n < 2 ? 0 : (spec.closed ? n : n - 1);
        case CurveKind::BSpline:
        {
            if (n < 2) return 0;
            if (spec.closed) return n;
            const int d = std::min(std::max(spec.degree, 1), std::min(7, n - 1));
            return n - d;
        }
        default: // Yuksel family
            return n < 3 ? 0 : (spec.closed ? n : n - 1);
    }
}

std::vector<CurvePt> sampleCurve(const CurveSpec& spec, int samplesPerSpan)
{
    std::vector<CurvePt> out;
    const int n = (int)spec.pts.size();
    if (n == 0) return out;
    const int m = std::max(2, samplesPerSpan);
    const int S = curveSpanCount(spec);
    if (S <= 0) { out = spec.pts; return out; } // degenerate: points verbatim
    const int total = spec.closed ? S * (m - 1) : S * (m - 1) + 1;
    out.reserve((size_t)total);

    switch (spec.kind)
    {
        case CurveKind::Bezier:
        {
            // closed: append pts[0] as a final control point so the segment
            // returns to its start (C0 closure -- see header).
            std::vector<CurvePt> ring;
            const CurvePt* ctrl = spec.pts.data();
            int count = n;
            if (spec.closed)
            {
                ring = spec.pts;
                ring.push_back(spec.pts[0]);
                ctrl = ring.data();
                count = n + 1;
            }
            std::vector<CurvePt> scratch;
            scratch.reserve((size_t)count);
            const float inv = 1.0f / (float)(S * (m - 1)); // open total-1 == S*(m-1)
            for (int k = 0; k < total; ++k)
                out.push_back(deCasteljau(ctrl, count, (float)k * inv, scratch));
            break;
        }
        case CurveKind::CubicPen:
        {
            // Rive's model: one cubic Bezier per edge, using the vertices' in/
            // out handles as the segment control points.
            auto handleAt = [&](int idx) -> CubicHandle {
                return idx >= 0 && idx < (int)spec.handles.size()
                           ? spec.handles[(size_t)idx]
                           : CubicHandle{};
            };
            const float dt = 1.0f / (float)(m - 1);
            std::vector<CurvePt> scratch;
            for (int j = 0; j < S; ++j)
            {
                const int j2 = spec.closed ? (j + 1) % n : j + 1;
                const CubicHandle hj = handleAt(j), hj2 = handleAt(j2);
                const CurvePt p0 = spec.pts[(size_t)j];
                const CurvePt p1 = spec.pts[(size_t)j2];
                const CurvePt seg[4] = {
                    p0,
                    {p0.x + hj.outDx, p0.y + hj.outDy},
                    {p1.x + hj2.inDx, p1.y + hj2.inDy},
                    p1};
                // closed / non-final spans emit m-1 points; the final open span
                // emits the last endpoint too (m points).
                const int cnt = (!spec.closed && j == S - 1) ? m : m - 1;
                for (int k = 0; k < cnt; ++k)
                    out.push_back(deCasteljau(seg, 4, (float)k * dt, scratch));
            }
            break;
        }
        case CurveKind::BSpline:
        {
            if (spec.closed)
            {
                const int d = std::min(std::max(spec.degree, 1), 7);
                const float du = 1.0f / (float)(m - 1); // u in [0, n)
                for (int k = 0; k < total; ++k)
                    out.push_back(deBoorPeriodic(spec.pts, d, (float)k * du));
            }
            else
            {
                const int d = std::min(std::max(spec.degree, 1), std::min(7, n - 1));
                const float uMax = (float)(n - d);
                const float inv = 1.0f / (float)(total - 1);
                for (int k = 0; k < total; ++k)
                    out.push_back(deBoorClamped(spec.pts, d, uMax * ((float)k * inv)));
            }
            break;
        }
        default: // Yuksel family
        {
            // Fit one local interpolant per (available) control point.
            std::vector<LocalFit> fits((size_t)n);
            const int iFirst = spec.closed ? 0 : 1;
            const int iLast = spec.closed ? n - 1 : n - 2;
            for (int i = iFirst; i <= iLast; ++i)
                fits[(size_t)i] = buildFit(spec.pts[(size_t)((i - 1 + n) % n)],
                                           spec.pts[(size_t)i],
                                           spec.pts[(size_t)((i + 1) % n)]);
            const float dt = 1.0f / (float)(m - 1);
            for (int j = 0; j < S; ++j)
            {
                // Closed: every segment blends two interpolants. Open: the
                // first/last segment has only one (the paper's end rule).
                const LocalFit* L = (spec.closed || j >= 1) ? &fits[(size_t)j] : nullptr;
                const int rIdx = spec.closed ? (j + 1) % n : j + 1;
                const LocalFit* R =
                    (spec.closed || rIdx <= n - 2) ? &fits[(size_t)rIdx] : nullptr;
                const bool lastOpenSeg = !spec.closed && j == S - 1;
                const int kEnd = lastOpenSeg ? m - 1 : m - 2; // skip shared t=1
                for (int k = 0; k <= kEnd; ++k)
                {
                    const float t = (float)k * dt;
                    CurvePt p;
                    if (L != nullptr && R != nullptr)
                    {
                        const float c = std::cos(0.5f * kPi * t);
                        const float w = c * c; // cos^2; the sin^2 weight is 1-w
                        // L owns p_j: traverse its SECOND half (p_j -> p_{j+1}).
                        // R owns p_{j+1}: traverse its FIRST half. evalSeg picks
                        // each base's own node, so lopsided triplets stay loop-
                        // free.
                        const CurvePt a = evalSeg(*L, spec.kind, true, t);
                        const CurvePt b = evalSeg(*R, spec.kind, false, t);
                        p = {a.x * w + b.x * (1.0f - w), a.y * w + b.y * (1.0f - w)};
                    }
                    else if (R != nullptr)
                    {
                        p = evalSeg(*R, spec.kind, false, t);
                    }
                    else
                    {
                        p = evalSeg(*L, spec.kind, true, t);
                    }
                    out.push_back(p);
                }
            }
            break;
        }
    }
    return out;
}

std::vector<CurveBaseWeights> sampleCurveBaseWeights(const CurveSpec& spec,
                                                     int samplesPerSpan)
{
    // Mirrors the Yuksel branch of sampleCurve EXACTLY (same span/sample
    // structure), accumulating each sample's unfolded base weights instead of
    // its position -- the two outputs stay index-parallel.
    std::vector<CurveBaseWeights> out;
    const int n = (int)spec.pts.size();
    if (!isYuksel(spec.kind) || n == 0)
        return out;
    const int m = std::max(2, samplesPerSpan);
    const int S = curveSpanCount(spec);
    if (S <= 0)
    {
        out.assign((size_t)n, CurveBaseWeights{}); // degenerate polyline
        return out;
    }
    std::vector<LocalFit> fits((size_t)n);
    std::vector<LocalInterpolantInfo> mixes((size_t)n);
    const int iFirst = spec.closed ? 0 : 1;
    const int iLast = spec.closed ? n - 1 : n - 2;
    for (int i = iFirst; i <= iLast; ++i)
    {
        fits[(size_t)i] = buildFit(spec.pts[(size_t)((i - 1 + n) % n)],
                                   spec.pts[(size_t)i],
                                   spec.pts[(size_t)((i + 1) % n)]);
        mixes[(size_t)i] = fitBaseMix(fits[(size_t)i], spec.kind);
    }
    const int total = spec.closed ? S * (m - 1) : S * (m - 1) + 1;
    out.reserve((size_t)total);
    const float dt = 1.0f / (float)(m - 1);
    for (int j = 0; j < S; ++j)
    {
        const bool hasL = spec.closed || j >= 1;
        const int rIdx = spec.closed ? (j + 1) % n : j + 1;
        const bool hasR = spec.closed || rIdx <= n - 2;
        const bool lastOpenSeg = !spec.closed && j == S - 1;
        const int kEnd = lastOpenSeg ? m - 1 : m - 2;
        for (int k = 0; k <= kEnd; ++k)
        {
            const float t = (float)k * dt;
            CurveBaseWeights bw;
            const LocalInterpolantInfo* L = hasL ? &mixes[(size_t)j] : nullptr;
            const LocalInterpolantInfo* R = hasR ? &mixes[(size_t)rIdx] : nullptr;
            if (L != nullptr && R != nullptr)
            {
                const float c = std::cos(0.5f * kPi * t);
                const float w = c * c;
                bw.ellipse = L->ellipse * w + R->ellipse * (1.0f - w);
                bw.circle = L->circle * w + R->circle * (1.0f - w);
                bw.parabola = L->parabola * w + R->parabola * (1.0f - w);
            }
            else
            {
                const LocalInterpolantInfo* one = (R != nullptr) ? R : L;
                bw.ellipse = one->ellipse;
                bw.circle = one->circle;
                bw.parabola = one->parabola;
            }
            out.push_back(bw);
        }
    }
    return out;
}

std::vector<CurvePt> sampleLocalInterpolant(const CurveSpec& spec, int i, int m,
                                            LocalInterpolantInfo* info)
{
    std::vector<CurvePt> out;
    const int n = (int)spec.pts.size();
    if (!isYuksel(spec.kind) || n < 3)
        return out;
    const int iFirst = spec.closed ? 0 : 1;
    const int iLast = spec.closed ? n - 1 : n - 2;
    if (i < iFirst || i > iLast)
        return out;
    const LocalFit f = buildFit(spec.pts[(size_t)((i - 1 + n) % n)],
                                spec.pts[(size_t)i],
                                spec.pts[(size_t)((i + 1) % n)]);
    if (info != nullptr)
        *info = fitBaseMix(f, spec.kind);
    m = std::max(2, m);
    out.reserve((size_t)m);
    // Walk the interpolant as its two halves so it passes through the triplet
    // (p_{i-1} at g=0, p_i at g=1/2, p_{i+1} at g=1) for EVERY base, including
    // Hybrid -- the two halves use each base's own node (see evalSeg).
    const float dg = 1.0f / (float)(m - 1);
    for (int k = 0; k < m; ++k)
    {
        const float g = (float)k * dg;
        out.push_back(g <= 0.5f ? evalSeg(f, spec.kind, false, 2.0f * g)
                                : evalSeg(f, spec.kind, true, 2.0f * g - 1.0f));
    }
    return out;
}
} // namespace rive_backend
