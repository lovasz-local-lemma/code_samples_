// backend/src/shape_geometry.cpp
// GL-free geometry/math core. Implementations land progressively across Tasks
// 3-6 of the stencil-shape-renderer plan.
#include "render/shape_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace
{
using rive_backend::Pt;

Pt xf(const rive::Mat2D& m, rive::Vec2D v)
{
    const rive::Vec2D r = m * v;
    return Pt{r.x, r.y};
}

// Subdivision count for a cubic so each chord is <= maxSegmentPx in device px.
int cubicSteps(Pt p0, Pt p1, Pt p2, Pt p3, float maxSegmentPx)
{
    auto d = [](Pt a, Pt b) { return std::hypot(a.x - b.x, a.y - b.y); };
    const float approxLen = d(p0, p1) + d(p1, p2) + d(p2, p3);
    const int steps =
        static_cast<int>(std::ceil(approxLen / std::max(maxSegmentPx, 1.0f)));
    return std::clamp(steps, 1, 256);
}

void appendCubic(std::vector<Pt>& out, Pt a, Pt b, Pt c, Pt d, float maxSegmentPx)
{
    const int n = cubicSteps(a, b, c, d, maxSegmentPx);
    for (int i = 1; i <= n; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(n);
        const float it = 1.0f - t;
        out.push_back(Pt{
            it * it * it * a.x + 3 * it * it * t * b.x + 3 * it * t * t * c.x +
                t * t * t * d.x,
            it * it * it * a.y + 3 * it * it * t * b.y + 3 * it * t * t * c.y +
                t * t * t * d.y});
    }
}
} // namespace

namespace rive_backend
{
std::vector<Contour> flattenPath(const rive::RawPath& path,
                                 const rive::Mat2D& transform,
                                 float maxSegmentPx)
{
    std::vector<Contour> out;
    Contour cur;
    auto finish = [&]() {
        if (cur.points.empty()) return;
        // Drop consecutive near-duplicate points (and a closing duplicate) so
        // strokes don't produce zero-length edges / degenerate normals (barbs).
        std::vector<Pt> dd;
        dd.reserve(cur.points.size());
        for (const Pt& p : cur.points)
            if (dd.empty() ||
                std::hypot(p.x - dd.back().x, p.y - dd.back().y) > 0.05f)
                dd.push_back(p);
        if (dd.size() > 1 &&
            std::hypot(dd.front().x - dd.back().x,
                       dd.front().y - dd.back().y) <= 0.05f)
            dd.pop_back();
        cur.points = std::move(dd);
        if (!cur.points.empty())
            out.push_back(std::move(cur));
        cur = Contour{};
    };
    for (auto e : path)
    {
        const rive::PathVerb verb = std::get<0>(e);
        const rive::Vec2D* p = std::get<1>(e);
        switch (verb)
        {
            case rive::PathVerb::move:
                finish();
                cur.points.push_back(xf(transform, p[0]));
                break;
            case rive::PathVerb::line:
                cur.points.push_back(xf(transform, p[1]));
                break;
            case rive::PathVerb::quad:
            {
                // Elevate quad to cubic then subdivide.
                const rive::Vec2D c1{p[0].x + (p[1].x - p[0].x) * 2.0f / 3.0f,
                                     p[0].y + (p[1].y - p[0].y) * 2.0f / 3.0f};
                const rive::Vec2D c2{p[2].x + (p[1].x - p[2].x) * 2.0f / 3.0f,
                                     p[2].y + (p[1].y - p[2].y) * 2.0f / 3.0f};
                appendCubic(cur.points, xf(transform, p[0]), xf(transform, c1),
                            xf(transform, c2), xf(transform, p[2]), maxSegmentPx);
                break;
            }
            case rive::PathVerb::cubic:
                appendCubic(cur.points, xf(transform, p[0]), xf(transform, p[1]),
                            xf(transform, p[2]), xf(transform, p[3]), maxSegmentPx);
                break;
            case rive::PathVerb::close:
                cur.closed = true;
                finish();
                break;
        }
    }
    finish();
    return out;
}

bool contoursBounds(const std::vector<Contour>& contours, Pt& mn, Pt& mx)
{
    bool any = false;
    mn = Pt{std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    mx = Pt{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
    for (const Contour& c : contours)
        for (const Pt& p : c.points)
        {
            any = true;
            mn.x = std::min(mn.x, p.x);
            mn.y = std::min(mn.y, p.y);
            mx.x = std::max(mx.x, p.x);
            mx.y = std::max(mx.y, p.y);
        }
    return any;
}

bool classifyFill(const std::vector<Contour>& contours, rive::FillRule rule, Pt p)
{
    int winding = 0; // signed crossings for nonZero
    int crosses = 0; // raw crossings for evenOdd
    for (const Contour& c : contours)
    {
        const size_t n = c.points.size();
        if (n < 2) continue;
        for (size_t i = 0; i < n; ++i)
        {
            const Pt a = c.points[i];
            const Pt b = c.points[(i + 1) % n];
            // Cast a +x ray from p: does edge (a,b) cross the scanline y==p.y
            // to the right of p?
            const bool aAbove = a.y > p.y;
            const bool bAbove = b.y > p.y;
            if (aAbove != bAbove)
            {
                const float t = (p.y - a.y) / (b.y - a.y);
                const float xCross = a.x + t * (b.x - a.x);
                if (xCross > p.x)
                {
                    ++crosses;
                    winding += bAbove ? 1 : -1; // upward edge +1, downward -1
                }
            }
        }
    }
    return rule == rive::FillRule::evenOdd ? (crosses & 1) != 0 : winding != 0;
}

static void lerpStops(const std::vector<GradientStop>& s, float t, float out[4])
{
    if (s.empty()) { out[0]=out[1]=out[2]=out[3]=0.0f; return; }
    if (t <= s.front().offset)
    { out[0]=s.front().r; out[1]=s.front().g; out[2]=s.front().b; out[3]=s.front().a; return; }
    if (t >= s.back().offset)
    { out[0]=s.back().r; out[1]=s.back().g; out[2]=s.back().b; out[3]=s.back().a; return; }
    for (size_t i = 1; i < s.size(); ++i)
        if (t <= s[i].offset)
        {
            const float span = std::max(s[i].offset - s[i-1].offset, 1e-6f);
            const float u = (t - s[i-1].offset) / span;
            out[0]=s[i-1].r+(s[i].r-s[i-1].r)*u;
            out[1]=s[i-1].g+(s[i].g-s[i-1].g)*u;
            out[2]=s[i-1].b+(s[i].b-s[i-1].b)*u;
            out[3]=s[i-1].a+(s[i].a-s[i-1].a)*u;
            return;
        }
}

void sampleGradient(bool radial, Pt start, Pt end, float radius,
                    const std::vector<GradientStop>& stops, Pt p, float out[4])
{
    float t;
    if (radial)
    {
        const float d = std::hypot(p.x - start.x, p.y - start.y);
        t = radius > 1e-6f ? d / radius : 0.0f;
    }
    else
    {
        const float dx = end.x - start.x, dy = end.y - start.y;
        const float len2 = dx*dx + dy*dy;
        t = len2 > 1e-6f ? ((p.x-start.x)*dx + (p.y-start.y)*dy) / len2 : 0.0f;
    }
    lerpStops(stops, t, out);
}

float featherSigma(float featherDevicePx)
{
    // Rive's feather value behaves like a blur diameter; sigma ~ value/2.
    // Constant refined visually against the official renderer.
    return featherDevicePx > 0.0f ? featherDevicePx * 0.5f : 0.0f;
}

std::vector<Pt> tessellateStroke(const Contour& c, float thicknessPx,
                                 rive::StrokeJoin /*join*/, rive::StrokeCap /*cap*/)
{
    // Round stroke = Minkowski sum of the path with a disc of radius h: one band
    // quad per segment + a disc at every vertex (round joins/caps). Nothing ever
    // reaches past distance h from the path, so it can't cross into the shape
    // interior (no internal seams), and the per-vertex discs avoid miter spikes
    // (barbs). Rendered via stencil-union so overlaps don't double-blend.
    std::vector<Pt> tris;
    const float h = thicknessPx * 0.5f;
    const std::vector<Pt>& pts = c.points;
    const int n = static_cast<int>(pts.size());
    if (n < 2) return tris;

    const int fan = 8; // wedges in the per-point disc (declared here: the reserve needs it)

    // TESS-RESERVE: the output size is known EXACTLY before a single point is
    // written -- 6 vertices per segment band, plus a 8-wedge disc (24 vertices) at
    // every point. Without this the repeated insert() grew the vector by doubling,
    // so a 512-point sweep paid ~11 reallocations and copied its own output about
    // twice over. The bracket (--tess-bench) is what made this visible: a 2-point
    // leg producing 54 vertices measured 2.47 us, which is not a number any amount
    // of arithmetic can explain -- it was the allocator, and the trig was hiding it.
    // BYTE-IDENTICAL: reserve changes capacity, never a value or an order.
    const int segs = c.closed ? n : n - 1;
    tris.reserve((size_t)segs * 6 + (size_t)n * (size_t)fan * 3);
    for (int i = 0; i < segs; ++i)
    {
        const Pt a = pts[i], b = pts[(i + 1) % n];
        const float dx = b.x - a.x, dy = b.y - a.y;
        const float len = std::max(std::hypot(dx, dy), 1e-6f);
        const float nx = -dy / len * h, ny = dx / len * h;
        const Pt a0{a.x + nx, a.y + ny}, a1{a.x - nx, a.y - ny};
        const Pt b0{b.x + nx, b.y + ny}, b1{b.x - nx, b.y - ny};
        tris.insert(tris.end(), {a0, b0, b1, a0, b1, a1});
    }

    // TRIG-LUT: the disc fan's angles are the SAME nine constants on every call --
    // 6.2831853f*k/8 for k = 0..8 -- so the old inline std::cos/std::sin cost 4 trig
    // calls per wedge, 32 per VERTEX, on values that never vary. A two-point stroked
    // leg (the shape most of the estimator arms emit) paid 64 of them to redraw the
    // same octagon it drew last frame.
    // BYTE-IDENTICAL BY CONSTRUCTION, and the construction is the point:
    //   * The table is filled at static-init by the SAME std::cos/std::sin on the SAME
    //     expression, so every entry is the identical libm result the inline code
    //     produced -- not a re-derivation, a memo. (Filling it from a literal table of
    //     decimal constants, or from double-precision values, would NOT be safe: the
    //     float rounding would have to match libm's exactly.)
    //   * NINE entries, not eight wrapped to zero. k = 7 asks for a1 = 6.2831853f,
    //     which is NOT 2*pi and NOT 0 -- std::cos of it need not return exactly the
    //     std::cos(0.0f) == 1.0f the wrap would substitute. The extra entry is what
    //     keeps the last wedge's vertices bit-for-bit what they were.
    static const struct FanTable
    {
        float cs[9], sn[9];
        FanTable()
        {
            for (int k = 0; k <= 8; ++k)
            {
                const float a = 6.2831853f * k / 8;
                cs[k] = std::cos(a);
                sn[k] = std::sin(a);
            }
        }
    } kFan;
    for (int i = 0; i < n; ++i)
    {
        const Pt ctr = pts[i];
        for (int k = 0; k < fan; ++k)
        {
            tris.insert(tris.end(),
                        {ctr,
                         Pt{ctr.x + kFan.cs[k] * h, ctr.y + kFan.sn[k] * h},
                         Pt{ctr.x + kFan.cs[k + 1] * h, ctr.y + kFan.sn[k + 1] * h}});
        }
    }
    return tris;
}
} // namespace rive_backend
