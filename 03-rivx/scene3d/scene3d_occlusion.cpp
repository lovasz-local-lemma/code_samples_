// backend/src/scene3d/scene3d_occlusion.cpp
// The scene3d viewer's pluggable, GL-free occlusion stage. See scene3d_occlusion.hpp for the
// design + the build order (None -> ClipPath -> ZBuffer -> Split -> Analytic).
//
// OC-T2 implements the ClipPath method: the exportable, rivx-faithful core. It makes occlusion
// ORDER-INDEPENDENT -- for every OPAQUE drawable D it subtracts the UNION of the screen
// silhouettes of every OPAQUE drawable N that is actually IN FRONT of D over their overlap, so
// each opaque surface paints only where it is the frontmost surface, regardless of the painter
// paint order. The UNION (not the raw overlapping silhouettes) is stored as boundary loops in
// D.clip so the subtraction is k-INDEPENDENT: overlapping raw holes double-count (and diverge
// between the even-odd and nonZero realisations at odd k>=3), while a non-overlapping union
// subtracts exactly once under both rules. The three render consumers -- Ours (even-odd stencil
// clip), RIVE (nonZero opposite-wound holes on D's path), .riv export (nonZero holes in the
// writer) -- realise the SAME "D minus union" region from that one list. TRANSLUCENT drawables
// (!opaque, e.g. the glass ball) neither clip nor are clipped -- they composite through.
#include "scene3d/scene3d_occlusion.hpp"
#include "scene3d/scene3d_ellipse.hpp" // ellipseNGon, kEllipseSegs (silhouette of an AnalyticEllipse)
#include "scene3d/scene3d_unproject.hpp" // GS1-T1: invert4 + rayDirThroughPixel (THE one copy)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>

namespace scene3d
{
using rive_backend::Mat4f;
using rive_backend::Vec3f;
using rive_backend::dot;
using rive_backend::length;
using rive_backend::normalize;

namespace
{
constexpr float kInf = std::numeric_limits<float>::infinity();

// GS1-T1: this TU's private V4/mulVec4, invert4() and rayDirThroughPixel() (the last of which
// already carried an "identical to ..." comment) are GONE -- they now live once, out-of-line, in
// scene3d_unproject.cpp, shared with scene3d_project.cpp and scene3d_render_ours.cpp. mulVec4's
// only caller was rayDirThroughPixel, and its (v,1) form is exactly math_3d's transform(), which
// the promoted body uses.

// A drawable's screen silhouette as a closed convex px loop: the polygon points for a
// FilledPoly / AnalyticQuad, or the tessellated N-gon of an AnalyticEllipse (the sphere keeps
// its exact projected ellipse outline). Wireframe polylines never reach here (they are not
// opaque occluders).
std::vector<P2> silhouetteLoop(const RivxDrawable& d)
{
    if (d.kind == RivxDrawable::AnalyticEllipse)
    {
        std::vector<P2> out;
        ellipseNGon(d.ellipse, kEllipseSegs, out);
        return out;
    }
    return d.pts;
}

struct AABB { float x0 = 0, y0 = 0, x1 = 0, y1 = 0; bool valid = false; };
AABB aabbOf(const std::vector<P2>& p)
{
    AABB b;
    if (p.empty()) return b;
    b.x0 = b.x1 = p[0].x;
    b.y0 = b.y1 = p[0].y;
    for (const P2& q : p)
    {
        b.x0 = std::min(b.x0, q.x); b.y0 = std::min(b.y0, q.y);
        b.x1 = std::max(b.x1, q.x); b.y1 = std::max(b.y1, q.y);
    }
    b.valid = true;
    return b;
}
bool aabbOverlap(const AABB& a, const AABB& b)
{
    if (!a.valid || !b.valid) return false;
    return !(a.x1 < b.x0 || b.x1 < a.x0 || a.y1 < b.y0 || b.y1 < a.y0);
}
P2 centroid(const std::vector<P2>& p)
{
    double sx = 0, sy = 0;
    for (const P2& q : p) { sx += q.x; sy += q.y; }
    const double n = p.empty() ? 1.0 : (double)p.size();
    return P2{(float)(sx / n), (float)(sy / n)};
}

// Signed twice-area of a px polygon (y-down): positive/negative sets the interior half-plane.
double signedArea2(const std::vector<P2>& p)
{
    double s = 0;
    for (size_t i = 0; i < p.size(); ++i)
    {
        const P2& a = p[i];
        const P2& b = p[(i + 1) % p.size()];
        s += (double)a.x * b.y - (double)b.x * a.y;
    }
    return s;
}
// Signed side of point P relative to directed edge A->B (>0 left in a y-up frame).
double edgeSide(const P2& A, const P2& B, const P2& P)
{
    return (double)(B.x - A.x) * (P.y - A.y) - (double)(B.y - A.y) * (P.x - A.x);
}

// Sutherland-Hodgman: clip the (possibly-non-convex) subject polygon to the CONVEX window
// polygon; returns their intersection (a convex loop, or empty on no overlap). `window` must be
// convex (box faces project to convex quads; a sphere's silhouette N-gon is convex). Clipping
// each near silhouette to D's own outline makes the stored clip loop a HOLE strictly inside D,
// so the even-odd realisation in all three consumers ("D minus holes") is identical.
std::vector<P2> clipToConvex(const std::vector<P2>& subject, const std::vector<P2>& window)
{
    if (subject.size() < 3 || window.size() < 3) return {};
    const double wsgn = signedArea2(window) >= 0.0 ? 1.0 : -1.0;
    std::vector<P2> out = subject;
    for (size_t i = 0; i < window.size(); ++i)
    {
        if (out.empty()) break;
        const P2 A = window[i];
        const P2 B = window[(i + 1) % window.size()];
        const std::vector<P2> in = out;
        out.clear();
        const size_t m = in.size();
        for (size_t j = 0; j < m; ++j)
        {
            const P2 P = in[j];
            const P2 Q = in[(j + 1) % m];
            const double sp = edgeSide(A, B, P) * wsgn;
            const double sq = edgeSide(A, B, Q) * wsgn;
            const bool inP = sp >= -1e-6;
            const bool inQ = sq >= -1e-6;
            if (inP) out.push_back(P);
            if (inP != inQ)
            {
                const double denom = sp - sq;
                const double t = std::fabs(denom) > 1e-12 ? sp / denom : 0.0;
                out.push_back(P2{(float)(P.x + t * (Q.x - P.x)),
                                 (float)(P.y + t * (Q.y - P.y))});
            }
        }
    }
    return out;
}

// Point STRICTLY inside a convex polygon `poly` (wound signedArea2 > 0 == CCW in this y-down
// shoelace convention)? On or outside any edge -> false. Used to classify union boundary edges.
bool insideConvexPos(const std::vector<P2>& poly, P2 p)
{
    const size_t n = poly.size();
    for (size_t i = 0; i < n; ++i)
    {
        const P2& a = poly[i];
        const P2& b = poly[(i + 1) % n];
        if (edgeSide(a, b, p) <= 1e-6) return false;
    }
    return true;
}

// UNION of convex polygons -> boundary loops. Edge-classification: normalize each polygon to
// signedArea2 > 0, split every edge at its crossings with the other polygons, keep each sub-edge
// whose midpoint is NOT strictly inside any other polygon, then chain the kept (directed) sub-
// edges into closed loops. A simple union comes out as one loop (signedArea2 > 0); a ring union
// adds inner hole loops (signedArea2 < 0). Correct for ANY count/arrangement of occluders, which
// is exactly what makes "D MINUS union" k-independent (no double-count) and identical across the
// even-odd (Ours) and nonZero (RIVE / export) realisations. Non-overlapping inputs come back as
// separate loops. Degenerate/near-zero polys are dropped.
std::vector<std::vector<P2>> unionConvexPolys(std::vector<std::vector<P2>> polys)
{
    std::vector<std::vector<P2>> P;
    for (std::vector<P2>& q : polys)
        if (q.size() >= 3 && std::fabs(signedArea2(q)) > 1e-3)
        {
            if (signedArea2(q) < 0.0) std::reverse(q.begin(), q.end()); // normalize to area2 > 0
            P.push_back(std::move(q));
        }
    if (P.size() <= 1) return P; // 0 or 1 poly: the union is the poly itself

    struct Seg { P2 a, b; };
    std::vector<Seg> kept;
    for (size_t i = 0; i < P.size(); ++i)
    {
        const std::vector<P2>& pi = P[i];
        const size_t ni = pi.size();
        for (size_t e = 0; e < ni; ++e)
        {
            const P2 A = pi[e], B = pi[(e + 1) % ni];
            const double dx = B.x - A.x, dy = B.y - A.y;
            std::vector<double> ts = {0.0, 1.0};
            for (size_t j = 0; j < P.size(); ++j)
            {
                if (j == i) continue;
                const std::vector<P2>& pj = P[j];
                const size_t nj = pj.size();
                for (size_t f = 0; f < nj; ++f)
                {
                    const P2 C = pj[f], Dp = pj[(f + 1) % nj];
                    const double sx = Dp.x - C.x, sy = Dp.y - C.y;
                    const double denom = dx * sy - dy * sx;
                    if (std::fabs(denom) < 1e-12) continue;
                    const double t = ((C.x - A.x) * sy - (C.y - A.y) * sx) / denom;
                    const double u = ((C.x - A.x) * dy - (C.y - A.y) * dx) / denom;
                    if (t > 1e-9 && t < 1.0 - 1e-9 && u > -1e-9 && u < 1.0 + 1e-9)
                        ts.push_back(t);
                }
            }
            std::sort(ts.begin(), ts.end());
            for (size_t k = 0; k + 1 < ts.size(); ++k)
            {
                const double t0 = ts[k], t1 = ts[k + 1];
                if (t1 - t0 < 1e-6) continue;
                const double tm = 0.5 * (t0 + t1);
                const P2 mid{(float)(A.x + tm * dx), (float)(A.y + tm * dy)};
                bool inside = false;
                for (size_t j = 0; j < P.size(); ++j)
                    if (j != i && insideConvexPos(P[j], mid)) { inside = true; break; }
                if (!inside)
                    kept.push_back(Seg{P2{(float)(A.x + t0 * dx), (float)(A.y + t0 * dy)},
                                       P2{(float)(A.x + t1 * dx), (float)(A.y + t1 * dy)}});
            }
        }
    }
    if (kept.empty()) return {};

    // Chain the directed boundary sub-edges into closed loops (endpoint match within ~1/8 px).
    auto key = [](P2 p) {
        return std::make_pair((int64_t)std::llround((double)p.x * 8.0),
                              (int64_t)std::llround((double)p.y * 8.0));
    };
    std::multimap<std::pair<int64_t, int64_t>, size_t> startMap;
    for (size_t i = 0; i < kept.size(); ++i) startMap.emplace(key(kept[i].a), i);
    std::vector<char> used(kept.size(), 0);
    std::vector<std::vector<P2>> loops;
    for (size_t s = 0; s < kept.size(); ++s)
    {
        if (used[s]) continue;
        std::vector<P2> loop;
        size_t cur = s;
        for (size_t guard = 0; guard <= kept.size(); ++guard)
        {
            if (used[cur]) break;
            used[cur] = 1;
            loop.push_back(kept[cur].a);
            const auto range = startMap.equal_range(key(kept[cur].b));
            size_t next = SIZE_MAX;
            for (auto it = range.first; it != range.second; ++it)
                if (!used[it->second]) { next = it->second; break; }
            if (next == SIZE_MAX) break;
            cur = next;
        }
        if (loop.size() >= 3 && std::fabs(signedArea2(loop)) > 1e-3)
            loops.push_back(std::move(loop));
    }
    return loops;
}

// Ray parameter t (distance along a unit-direction ray from the eye) where it meets a surface;
// +inf on a miss. Plane: ray-plane. Sphere: nearest positive root of the quadratic (a==1 for a
// normalized direction). These give the true 3D depth at a screen pixel so front/back does NOT
// depend on the painter-by-centre order.
float planeT(const Surface3D& s, Vec3f ro, Vec3f rd)
{
    const float denom = dot(rd, s.n);
    if (std::fabs(denom) < 1e-9f) return kInf;
    const float t = dot(s.p - ro, s.n) / denom;
    return t > 1e-6f ? t : kInf;
}
float sphereT(const Surface3D& s, Vec3f ro, Vec3f rd)
{
    const Vec3f oc = ro - s.c;
    const float b = dot(oc, rd);
    const float c = dot(oc, oc) - s.r * s.r;
    const float disc = b * b - c;
    if (disc < 0.0f) return kInf;
    const float sq = std::sqrt(disc);
    const float t0 = -b - sq;
    if (t0 > 1e-6f) return t0;
    const float t1 = -b + sq;
    return t1 > 1e-6f ? t1 : kInf;
}
// VEL3D: PROMOTED out of the anonymous namespace -- the header now declares it, because the
// velocity buffer unprojects each drawable centroid onto its own source surface (the EXACT-1
// precedent: a second consumer is the moment to promote rather than copy). Definition
// unchanged; only the linkage moved, so every existing caller here is byte-identical.
} // namespace (the TU-local helpers above)

float surfaceT(const Surface3D& s, Vec3f ro, Vec3f rd)
{
    if (s.kind == Surface3D::Sphere) return sphereT(s, ro, rd);
    if (s.kind == Surface3D::Plane) return planeT(s, ro, rd);
    return kInf;
}

namespace
{
Vec3f surfaceRep(const Surface3D& s) { return s.kind == Surface3D::Sphere ? s.c : s.p; }

// (GS1-T1: rayDirThroughPixel now comes from scene3d_unproject.hpp -- see the note above.)

// Decide whether N's surface is IN FRONT of D's surface over their screen overlap. Samples a few
// candidate pixels inside the overlap (its AABB centre, then the two silhouette centroids); at
// the first candidate where the ray hits BOTH surfaces, front == (tN < tD). Falls back to the
// representative-point distance (painter-by-centre) if no candidate hits both.
bool nInFrontOfD(const Surface3D& sN, const Surface3D& sD, const Mat4f& invVP, Vec3f eye,
                 float W, float H, const std::vector<P2>& dPoly, const std::vector<P2>& nPoly,
                 const AABB& dB, const AABB& nB)
{
    const float ox0 = std::max(dB.x0, nB.x0), oy0 = std::max(dB.y0, nB.y0);
    const float ox1 = std::min(dB.x1, nB.x1), oy1 = std::min(dB.y1, nB.y1);
    const P2 cand[3] = {P2{0.5f * (ox0 + ox1), 0.5f * (oy0 + oy1)},
                        centroid(nPoly), centroid(dPoly)};
    for (const P2& c : cand)
    {
        const Vec3f rd = rayDirThroughPixel(invVP, eye, c.x, c.y, W, H);
        const float tN = surfaceT(sN, eye, rd);
        const float tD = surfaceT(sD, eye, rd);
        if (tN < kInf && tD < kInf) return tN < tD;
    }
    return length(surfaceRep(sN) - eye) < length(surfaceRep(sD) - eye);
}

// The ClipPath method: subtract every nearer opaque silhouette from each opaque drawable.
void applyClipPath(const Mat4f& viewProj, Vec3f eye, int w, int h, std::vector<RivxDrawable>& io)
{
    if (w <= 0 || h <= 0 || io.empty()) return;
    const Mat4f invVP = invert4(viewProj);
    const float W = (float)w, H = (float)h;

    // Precompute each drawable's silhouette + AABB once (opaque, real-surface drawables only).
    const size_t n = io.size();
    std::vector<std::vector<P2>> loops(n);
    std::vector<AABB> boxes(n);
    std::vector<char> occ(n, 0); // participates as an occluder / occludee
    for (size_t i = 0; i < n; ++i)
    {
        const RivxDrawable& d = io[i];
        if (!d.opaque || d.surf.kind == Surface3D::None) continue;
        loops[i] = silhouetteLoop(d);
        if (loops[i].size() < 3) continue;
        boxes[i] = aabbOf(loops[i]);
        occ[i] = 1;
    }

    for (size_t di = 0; di < n; ++di)
    {
        if (!occ[di]) continue;               // D must be opaque (translucent never gets clipped)
        RivxDrawable& D = io[di];
        // Collect every nearer opaque silhouette (clipped to D's outline) that occludes D.
        std::vector<std::vector<P2>> occluders;
        for (size_t ni = 0; ni < n; ++ni)
        {
            if (ni == di || !occ[ni]) continue; // N must be opaque (translucent never clips)
            // Same source object never self-occludes: the Cornell room's interior faces (all
            // objectId 1) tile the view without occluding one another; without this the shared
            // trapezoid edges would spuriously clip. Distinct objects (walls vs the opaque ball,
            // the light quad) still interact.
            if (io[ni].objectId == D.objectId) continue;
            if (!aabbOverlap(boxes[di], boxes[ni])) continue;
            if (!nInFrontOfD(io[ni].surf, D.surf, invVP, eye, W, H, loops[di], loops[ni],
                             boxes[di], boxes[ni]))
                continue;
            // N is in front of D over the overlap -> clip N's silhouette to D's own outline so
            // the piece is strictly inside D. Empty clip (no true polygon overlap) adds nothing.
            std::vector<P2> hole = clipToConvex(loops[ni], loops[di]);
            if (hole.size() >= 3) occluders.push_back(std::move(hole));
        }
        if (occluders.empty()) continue;
        // D is hidden wherever it is behind >= 1 opaque occluder = D MINUS the UNION of the near
        // silhouettes. Store the UNION boundary loops (NOT the raw overlapping silhouettes) so the
        // subtraction is k-INDEPENDENT and identical in every consumer: overlapping raw holes
        // double-count (even-odd hides on odd k, nonZero on k==1) and diverge at odd k>=3, but a
        // non-overlapping union subtracts exactly once under both rules. Store each union loop
        // REVERSED (union region negatively wound relative to D's clockwise-normalized outline) so
        // a nonZero fill of D-outer + these loops = D minus union (Ours reads them even-odd, which
        // is orientation-independent, so both realise the same region).
        std::vector<std::vector<P2>> uni = unionConvexPolys(std::move(occluders));
        for (std::vector<P2>& loop : uni)
        {
            if (loop.size() < 3) continue;
            std::reverse(loop.begin(), loop.end());
            D.clip.push_back(std::move(loop));
        }
    }
}

// ===== OC-T4: Split -- Newell/BSP screen-space split of INTERPENETRATING opaque pairs ==========
// Split makes the painter order an EXACT total order. Where two OPAQUE surfaces overlap in screen
// AND their depth order FLIPS across the overlap (interpenetration -- a plane crossing another
// surface), it cuts BOTH screen polygons along the projected surface-surface crossing line into
// pieces that each have an unambiguous depth, then re-sorts every piece far->near. The output is
// MORE plain FilledPoly drawables (no clips), so Ours / RIVE / .riv export all just paint the
// sorted pieces and occlude IDENTICALLY. This is the case ClipPath's silhouette-subtraction cannot
// resolve (neither surface is wholly in front). SCOPE + LIMITS (the plan explicitly allows Split to
// be less-than-fully-general): (1) plane-plane interpenetration is EXACT (the crossing line is the
// projected 3D plane-plane intersection = the depth-equal locus); (2) a SPHERE that participates is
// tessellated to an N-gon FilledPoly (so Split LOSES the analytic ellipse, unlike ClipPath) and
// split along the crossing of the other plane with the sphere's eye-facing TANGENT plane -- an
// approximation of the true curved equal-depth locus (piece DEPTHS stay exact via the sphere ray-
// cast); (3) non-interpenetrating / cleanly-ordered overlaps pass through unchanged (already correct
// under painter order); (4) the per-piece representative depth is the true distance at the piece
// centroid -- a scalar painter key, exact for the pairwise crossing case, not a full Newell
// topological order for adversarial multi-way cyclic overlaps.

// Solve the 3x3 system whose ROWS are r0,r1,r2 for M x = rhs (Cramer). false if (near-)singular.
bool solve3(Vec3f r0, Vec3f r1, Vec3f r2, Vec3f rhs, Vec3f& x)
{
    const double a = r0.x, b = r0.y, c = r0.z;
    const double d = r1.x, e = r1.y, f = r1.z;
    const double g = r2.x, h = r2.y, i = r2.z;
    const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (std::fabs(det) < 1e-12) return false;
    const double A = rhs.x, B = rhs.y, C = rhs.z;
    const double dx = A * (e * i - f * h) - b * (B * i - f * C) + c * (B * h - e * C);
    const double dy = a * (B * i - f * C) - A * (d * i - f * g) + c * (d * C - B * g);
    const double dz = a * (e * C - B * h) - b * (d * C - B * g) + A * (d * h - e * g);
    x = Vec3f{(float)(dx / det), (float)(dy / det), (float)(dz / det)};
    return true;
}

// The 3D line (point p0 + unit dir) where planes A,B meet. false if parallel (no crossing).
bool planePlaneLine(Vec3f pA, Vec3f nA, Vec3f pB, Vec3f nB, Vec3f& p0, Vec3f& dir)
{
    dir = cross(nA, nB);
    if (length(dir) < 1e-6f) return false;
    // Point on both planes closest to the origin along dir (the 3rd row pins dir.x0 = 0).
    if (!solve3(nA, nB, dir, Vec3f{dot(nA, pA), dot(nB, pB), 0.0f}, p0)) return false;
    dir = normalize(dir);
    return true;
}

// The plane a surface is split ALONG: exact for a Plane; the eye-facing tangent plane for a Sphere
// (the documented sphere approximation -- see SCOPE above).
void splitPlaneOf(const Surface3D& s, Vec3f eye, Vec3f& p, Vec3f& n)
{
    if (s.kind == Surface3D::Sphere)
    {
        n = normalize(eye - s.c);
        p = s.c + n * s.r;
    }
    else
    {
        p = s.p;
        n = s.n;
    }
}

// Two DISTINCT in-front screen points on the projected 3D line p0 + t*dir, or false. Samples t over
// a few world units and keeps the two most-separated in-front projections (robust to one endpoint
// falling behind the camera).
bool projectedLine(Vec3f p0, Vec3f dir, const Mat4f& vp, float W, float H, P2& a, P2& b)
{
    constexpr int K = 7;
    const float ts[K] = {-3.f, -2.f, -1.f, 0.f, 1.f, 2.f, 3.f};
    P2 sp[K];
    bool ok[K];
    int cnt = 0;
    for (int i = 0; i < K; ++i)
    {
        const rive_backend::Projected pr =
            rive_backend::projectPoint(vp, p0 + dir * ts[i], W, H);
        ok[i] = pr.inFront;
        if (pr.inFront) { sp[i] = P2{pr.x, pr.y}; ++cnt; }
    }
    if (cnt < 2) return false;
    float best = -1.0f;
    int bi = -1, bj = -1;
    for (int i = 0; i < K; ++i)
        if (ok[i])
            for (int j = i + 1; j < K; ++j)
                if (ok[j])
                {
                    const float dx = sp[i].x - sp[j].x, dy = sp[i].y - sp[j].y;
                    const float d2 = dx * dx + dy * dy;
                    if (d2 > best) { best = d2; bi = i; bj = j; }
                }
    if (bi < 0 || best < 1.0f) return false; // < 1px apart -> unusable as a split line
    a = sp[bi];
    b = sp[bj];
    return true;
}

// Do surfaces A,B interpenetrate over their screen overlap? True iff the true 3D depth order FLIPS
// -- at some overlap pixel A is nearer, at another B is nearer. Uses the EXACT surfaces (plane /
// sphere ray-cast), so a sphere merely TOUCHING a plane (no crossing) is correctly not a flip.
bool depthFlip(const Surface3D& sA, const Surface3D& sB, const std::vector<P2>& overlap,
               const Mat4f& invVP, Vec3f eye, float W, float H)
{
    if (overlap.size() < 3) return false;
    const P2 c = centroid(overlap);
    bool aFront = false, bFront = false;
    auto probe = [&](P2 p) {
        const Vec3f rd = rayDirThroughPixel(invVP, eye, p.x, p.y, W, H);
        const float tA = surfaceT(sA, eye, rd), tB = surfaceT(sB, eye, rd);
        if (tA < kInf && tB < kInf)
        {
            if (tA < tB - 1e-4f) aFront = true;
            else if (tB < tA - 1e-4f) bFront = true;
        }
    };
    probe(c);
    for (const P2& v : overlap)
        probe(P2{c.x + 0.8f * (v.x - c.x), c.y + 0.8f * (v.y - c.y)});
    return aFront && bFront;
}

// Split a convex polygon by the infinite line through A,B into the pos-side and neg-side pieces
// (a vertex ON the line joins BOTH, so the shared boundary is preserved). Convex in -> two convex
// out; the box-face quads and sphere N-gons that reach here are convex.
void splitPolyByLine(const std::vector<P2>& poly, P2 A, P2 B, std::vector<P2>& pos,
                     std::vector<P2>& neg)
{
    pos.clear();
    neg.clear();
    const size_t n = poly.size();
    if (n < 3) return;
    constexpr double eps = 1e-6;
    for (size_t i = 0; i < n; ++i)
    {
        const P2 P = poly[i], Q = poly[(i + 1) % n];
        const double sp = edgeSide(A, B, P), sq = edgeSide(A, B, Q);
        if (sp >= -eps) pos.push_back(P);
        if (sp <= eps) neg.push_back(P);
        if ((sp > eps && sq < -eps) || (sp < -eps && sq > eps))
        {
            const double t = sp / (sp - sq);
            const P2 X{(float)(P.x + t * (Q.x - P.x)), (float)(P.y + t * (Q.y - P.y))};
            pos.push_back(X);
            neg.push_back(X);
        }
    }
}

// The Split method: cut every INTERPENETRATING opaque pair along its projected crossing line and
// re-sort all pieces far->near. Non-interpenetrating scenes return unchanged (identity == None).
void applySplit(const Mat4f& viewProj, Vec3f eye, int w, int h, std::vector<RivxDrawable>& io)
{
    if (w <= 0 || h <= 0 || io.size() < 2) return;
    const Mat4f invVP = invert4(viewProj);
    const float W = (float)w, H = (float)h;
    const size_t n = io.size();

    // Participants = opaque, real-surface drawables (planes AND spheres). Precompute silhouette +
    // AABB. Spheres carry their tessellated N-gon here (silhouetteLoop), so a participating sphere
    // becomes an N-gon FilledPoly (loses the analytic ellipse -- the documented Split trade).
    std::vector<std::vector<P2>> sil(n);
    std::vector<AABB> box(n);
    std::vector<char> part(n, 0);
    for (size_t i = 0; i < n; ++i)
    {
        const RivxDrawable& d = io[i];
        if (!d.opaque || d.surf.kind == Surface3D::None) continue;
        sil[i] = silhouetteLoop(d);
        if (sil[i].size() < 3) continue;
        box[i] = aabbOf(sil[i]);
        part[i] = 1;
    }

    // Gather, per participant, the projected crossing line of every partner it interpenetrates.
    std::vector<std::vector<std::pair<P2, P2>>> cuts(n);
    for (size_t i = 0; i < n; ++i)
    {
        if (!part[i]) continue;
        Vec3f pi, ni;
        splitPlaneOf(io[i].surf, eye, pi, ni);
        for (size_t j = i + 1; j < n; ++j)
        {
            if (!part[j]) continue;
            if (io[i].objectId == io[j].objectId) continue; // a solid's own faces never self-split
            if (!aabbOverlap(box[i], box[j])) continue;
            const std::vector<P2> ov = clipToConvex(sil[i], sil[j]);
            if (ov.size() < 3) continue;
            // Only a genuine depth-order FLIP interpenetrates; a clean front/back passes through.
            if (!depthFlip(io[i].surf, io[j].surf, ov, invVP, eye, W, H)) continue;
            Vec3f pj, nj;
            splitPlaneOf(io[j].surf, eye, pj, nj);
            Vec3f p0, dir;
            if (!planePlaneLine(pi, ni, pj, nj, p0, dir)) continue;
            P2 a, b;
            if (!projectedLine(p0, dir, viewProj, W, H, a, b)) continue;
            cuts[i].push_back({a, b});
            cuts[j].push_back({a, b});
        }
    }

    bool any = false;
    for (size_t i = 0; i < n && !any; ++i) any = !cuts[i].empty();
    if (!any) return; // no interpenetration -> leave the painter list untouched (Split == None)

    // Rebuild the list: uncut drawables pass through; each cut participant is replaced by its
    // sub-pieces (FilledPoly, clip-free, ellipse cleared) with a representative centroid depth.
    std::vector<RivxDrawable> out;
    out.reserve(n + 8);
    for (size_t i = 0; i < n; ++i)
    {
        // BK2 review (FIX 3c) -- defensive: ImageMesh routes through renderOurs today; guard the
        // silent-default trap. A cut copies the drawable then forces f.kind=FilledPoly WITHOUT
        // clearing imageId/uv, so a cut textured quad would keep a stale image ref + uv and lose its
        // texture. Never cut an ImageMesh: pass textured quads through unchanged (like an uncut one).
        if (cuts[i].empty() || io[i].kind == RivxDrawable::ImageMesh) { out.push_back(io[i]); continue; }
        std::vector<std::vector<P2>> pieces;
        pieces.push_back(sil[i]);
        for (const std::pair<P2, P2>& ln : cuts[i])
        {
            std::vector<std::vector<P2>> next;
            for (const std::vector<P2>& pc : pieces)
            {
                std::vector<P2> pa, nb;
                splitPolyByLine(pc, ln.first, ln.second, pa, nb);
                if (pa.size() >= 3 && std::fabs(signedArea2(pa)) > 1.0) next.push_back(std::move(pa));
                if (nb.size() >= 3 && std::fabs(signedArea2(nb)) > 1.0) next.push_back(std::move(nb));
            }
            if (!next.empty()) pieces.swap(next);
        }
        for (std::vector<P2>& pc : pieces)
        {
            if (pc.size() < 3) continue;
            RivxDrawable f = io[i];
            f.kind = RivxDrawable::FilledPoly; // tessellated: a sphere loses its analytic ellipse
            f.pts = pc;
            f.closed = true;
            f.clip.clear();
            f.ellipse = Ellipse2D{};
            const P2 cc = centroid(pc);
            const Vec3f rd = rayDirThroughPixel(invVP, eye, cc.x, cc.y, W, H);
            const float t = surfaceT(io[i].surf, eye, rd);
            f.depth = (t < kInf) ? -t : io[i].depth; // -distance = the painter key convention
            out.push_back(std::move(f));
        }
    }

    // Painter re-sort: ascending depth (= -distance), stable, so the farthest piece is first ->
    // back-to-front. Splitting at the exact plane-plane depth-equal locus makes each cut pair
    // cleanly one-in-front over their overlap. NOTE: this scalar per-own-centroid key is NOT a
    // full total order -- it can still mis-order a piece vs a THIRD surface it was never split
    // against, and even a direct split pair when the two pieces' centroids sample different pixels
    // at grazing/asymmetric angles (a correct order exists; this key doesn't always find it). Split
    // strictly improves on None for detected interpenetrations; true generality needs a
    // common-overlap-point depth comparison (or full Newell). The shipped occlusionSplit scene is
    // symmetric, so neither residual case fires.
    std::stable_sort(out.begin(), out.end(),
                     [](const RivxDrawable& a, const RivxDrawable& b) { return a.depth < b.depth; });
    io.swap(out);
}

// ===== OC-T5: Analytic -- closed-form per-primitive-pair visibility (vector, exportable) ========
// The most PRECISE method for the sphere+plane primitive set: each opaque surface's VISIBLE
// sub-region is emitted in closed form. Two pair cases:
//   * SPHERE in front of a PLANE: the plane's visible region = the plane polygon MINUS the sphere's
//     EXACT projected ELLIPSE (reconstructed closed-form from the sphere's Surface3D, kept a true
//     ellipse -- the precision edge: Split tessellates the sphere to an N-gon, ClipPath subtracts
//     the drawable's silhouette polygon, Analytic subtracts a genuine 4-arc ellipse that every
//     consumer traces identically via the shared ellipse helpers). Stored in D.clipEllipses.
//   * PLANE in front of a PLANE: the far plane's visible region = its polygon MINUS the nearer
//     plane's projected quad (polygon subtraction, unioned) -- reuses the ClipPath machinery,
//     stored in D.clip.
// The SPHERE itself is always left unchanged (its own visible silhouette). front/back is the SAME
// 3D ray-depth decision the other methods use (nInFrontOfD). SCOPE / non-generalizing: only
// Plane/Sphere surfaces participate; any other kind is left as-is (best-effort bail). An exact
// ellipse hole is used ONLY when it is fully CONTAINED in the far plane's outline AND isolated from
// every other occluder on that plane (so the lone even-odd/nonZero subtraction is identical across
// Ours / RIVE / export); a sphere that straddles the plane edge or overlaps another occluder is
// demoted to the polygon-union path (correct, tessellated) -- the documented fidelity fallback.

// Reconstruct a sphere's EXACT projected silhouette ellipse from its Surface3D (world centre +
// radius) + the camera, so the Analytic hole matches the sphere drawable's own outline exactly.
// false if the eye is inside/at the sphere or a rim point falls behind the camera (then the pair
// is left to the polygon path / skipped).
// SH-4 DEDUPE: this (and the fitEllipse2Dpx copy of project()'s fitEllipse2D that it used) was a
// verbatim copy of project()'s sphereSilhouetteEllipse. It now FORWARDS to
// that one implementation (exported as sphereSilhouetteEllipsePx), so the Analytic occlusion hole,
// the sphere drawable's own outline and the SH-4 blur-clip window are the same math by
// construction rather than by three copies staying in lockstep by hand.
bool sphereEllipseFromSurf(const Surface3D& s, const Mat4f& vp, Vec3f eye, float W, float H,
                           Ellipse2D& out)
{
    if (s.kind != Surface3D::Sphere) return false;
    return sphereSilhouetteEllipsePx(s.c, s.r, vp, eye, W, H, out);
}

// Even-odd point-in-(simple)-polygon in px. Used to test ellipse-in-plane containment.
bool pointInPoly(const std::vector<P2>& poly, float x, float y)
{
    bool in = false;
    const size_t n = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++)
    {
        const P2& a = poly[i];
        const P2& b = poly[j];
        if (((a.y > y) != (b.y > y)) &&
            (x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x))
            in = !in;
    }
    return in;
}

// Is the tessellated ellipse fully INSIDE polygon D? Required before an EXACT (un-clipped) ellipse
// hole is safe for the nonZero consumers (RIVE / export): a hole loop that pokes outside D's outer
// boundary would paint a crescent there (winding non-zero outside the outer). Ours (rect-nonzero
// clip) tolerates a poking hole, but we gate uniformly so all three subtract the identical region.
bool ellipseInsidePoly(const Ellipse2D& e, const std::vector<P2>& D)
{
    if (D.size() < 3) return false;
    std::vector<P2> ring;
    ellipseNGon(e, kEllipseSegs, ring);
    for (const P2& p : ring)
        if (!pointInPoly(D, p.x, p.y)) return false;
    return true;
}

// px AABB of an ellipse (via its tessellation) -- for the isolation (non-overlap) test.
AABB ellipseAABBpx(const Ellipse2D& e)
{
    std::vector<P2> ring;
    ellipseNGon(e, kEllipseSegs, ring);
    return aabbOf(ring);
}

// The Analytic method: emit each opaque plane's visible sub-region (sphere ellipse holes exact,
// plane holes polygon-subtracted). Spheres + non-plane/sphere kinds are left unchanged.
void applyAnalytic(const Mat4f& viewProj, Vec3f eye, int w, int h, std::vector<RivxDrawable>& io)
{
    if (w <= 0 || h <= 0 || io.empty()) return;
    const Mat4f invVP = invert4(viewProj);
    const float W = (float)w, H = (float)h;

    const size_t n = io.size();
    std::vector<std::vector<P2>> loops(n);
    std::vector<AABB> boxes(n);
    std::vector<char> occ(n, 0); // opaque, real-surface (Plane/Sphere) participant
    for (size_t i = 0; i < n; ++i)
    {
        const RivxDrawable& d = io[i];
        if (!d.opaque) continue;
        if (d.surf.kind != Surface3D::Plane && d.surf.kind != Surface3D::Sphere)
            continue; // outside the primitive set -> never occludes / is occluded (bail)
        loops[i] = silhouetteLoop(d);
        if (loops[i].size() < 3) continue;
        boxes[i] = aabbOf(loops[i]);
        occ[i] = 1;
    }

    // One in-front occluder of a plane D: either an EXACT sphere ellipse (candidate) or a polygon.
    struct Occ
    {
        bool ellipse = false;    // sphere, contained -> exact-ellipse candidate
        Ellipse2D e;             // the exact projected ellipse (ellipse candidates)
        std::vector<P2> poly;    // fallback / plane polygon (clipped to D)
        AABB box;                // px extent (for the isolation test)
    };
    // A tessellated sphere (SphereMode Mesh/Rings/Phi) is emitted as M>1 FilledPoly patches that
    // ALL carry the identical whole-sphere Surface3D, so each reconstructs the SAME full-sphere
    // ellipse occluder. Dedup those by (centre, radius) so one sphere contributes ONE ellipse hole.
    struct SphereKey
    {
        Vec3f c;
        float r;
    };

    for (size_t di = 0; di < n; ++di)
    {
        if (!occ[di]) continue;
        RivxDrawable& D = io[di];
        if (D.surf.kind != Surface3D::Plane) continue; // spheres stay unchanged (own silhouette)

        std::vector<Occ> occs;
        std::vector<SphereKey> seenSpheres; // exact-ellipse sphere occluders already collected for D
        for (size_t ni = 0; ni < n; ++ni)
        {
            if (ni == di || !occ[ni]) continue;
            if (io[ni].objectId == D.objectId) continue; // a solid's own faces never self-occlude
            if (!aabbOverlap(boxes[di], boxes[ni])) continue;
            if (!nInFrontOfD(io[ni].surf, D.surf, invVP, eye, W, H, loops[di], loops[ni],
                             boxes[di], boxes[ni]))
                continue; // N behind D over the overlap -> does not occlude
            Occ o;
            if (io[ni].surf.kind == Surface3D::Sphere)
            {
                Ellipse2D e;
                if (sphereEllipseFromSurf(io[ni].surf, viewProj, eye, W, H, e) &&
                    ellipseInsidePoly(e, loops[di]))
                {
                    // DEDUP a tessellated sphere: every patch of one sphere rebuilds the identical
                    // whole-sphere ellipse here. Keeping all M would give M identical-AABB ellipse
                    // occluders that defeat the isolation test below -> all demoted to the polygon-
                    // union path -> M coincident boundaries -> nonZero hole winding 1-M != 0 (the
                    // sphere silhouette renders FILLED, the export corrupted). Keep only the FIRST
                    // per distinct sphere -> one exact ellipse = the correct single hole. (Only the
                    // exact-ellipse branch dedups; the straddling polygon branch below still needs
                    // every patch's distinct clipped silhouette to reconstruct the union.)
                    bool dupSphere = false;
                    for (const SphereKey& k : seenSpheres)
                        if (std::fabs(k.c.x - io[ni].surf.c.x) < 1e-5f &&
                            std::fabs(k.c.y - io[ni].surf.c.y) < 1e-5f &&
                            std::fabs(k.c.z - io[ni].surf.c.z) < 1e-5f &&
                            std::fabs(k.r - io[ni].surf.r) < 1e-5f)
                        {
                            dupSphere = true;
                            break;
                        }
                    if (dupSphere) continue; // this sphere already contributes its exact ellipse
                    seenSpheres.push_back(SphereKey{io[ni].surf.c, io[ni].surf.r});
                    o.ellipse = true;   // exact-ellipse candidate (fully inside D)
                    o.e = e;
                    o.box = ellipseAABBpx(e);
                }
                else
                {
                    o.poly = clipToConvex(loops[ni], loops[di]); // straddles the edge -> polygon
                    if (o.poly.size() < 3) continue;
                    o.box = aabbOf(o.poly);
                }
            }
            else // Plane in front of plane -> polygon subtraction (clipped to D)
            {
                o.poly = clipToConvex(loops[ni], loops[di]);
                if (o.poly.size() < 3) continue;
                o.box = aabbOf(o.poly);
            }
            occs.push_back(std::move(o));
        }
        if (occs.empty()) continue;

        // An exact-ellipse candidate stays exact only if ISOLATED (its px box disjoint from every
        // other occluder's) -- then its lone subtraction is identical under even-odd (Ours) and
        // nonZero (RIVE/export). Overlapping candidates are demoted to the polygon-union path so a
        // k>=2 overlap can never double-count differently between the consumers.
        std::vector<std::vector<P2>> polyHoles;
        for (size_t i = 0; i < occs.size(); ++i)
        {
            bool isolated = occs[i].ellipse;
            if (isolated)
                for (size_t j = 0; j < occs.size(); ++j)
                    if (j != i && aabbOverlap(occs[i].box, occs[j].box)) { isolated = false; break; }
            if (occs[i].ellipse && isolated)
            {
                D.clipEllipses.push_back(occs[i].e); // EXACT ellipse hole (no tessellation)
            }
            else
            {
                std::vector<P2> poly = occs[i].poly;
                if (poly.size() < 3 && occs[i].ellipse) // demoted candidate: tessellate its ellipse
                {
                    ellipseNGon(occs[i].e, kEllipseSegs, poly);
                    poly = clipToConvex(poly, loops[di]);
                }
                if (poly.size() >= 3) polyHoles.push_back(std::move(poly));
            }
        }
        // Plane / demoted occluders -> UNION boundary loops (k-independent), stored reversed so a
        // nonZero fill of D-outer + these loops = D minus union (identical to the ClipPath path).
        if (!polyHoles.empty())
        {
            std::vector<std::vector<P2>> uni = unionConvexPolys(std::move(polyHoles));
            for (std::vector<P2>& loop : uni)
            {
                if (loop.size() < 3) continue;
                std::reverse(loop.begin(), loop.end());
                D.clip.push_back(std::move(loop));
            }
        }
    }
}

} // namespace

void applyOcclusion(Occlusion m, const Scene3D& sc, const rive_backend::Mat4f& viewProj,
                    rive_backend::Vec3f eye, int w, int h, std::vector<RivxDrawable>& io)
{
    (void)sc; // the surf descriptors on each drawable carry the geometry the vector methods need

    switch (m)
    {
        case Occlusion::None:
        case Occlusion::ZBuffer:
            // None keeps the raw painter order; ZBuffer is an Ours-live per-fragment depth test
            // applied render-side (OC-T3), not a list rewrite. Both are no-ops here.
            return;
        case Occlusion::ClipPath:
            applyClipPath(viewProj, eye, w, h, io);
            return;
        case Occlusion::Split: // OC-T4: Newell/BSP screen-space split of interpenetrating pairs
            applySplit(viewProj, eye, w, h, io);
            return;
        case Occlusion::Analytic: // OC-T5: closed-form per-primitive-pair visible sub-regions
            applyAnalytic(viewProj, eye, w, h, io);
            return;
    }
}

// ---- EXACT-1: the promoted boolean wrappers (see the header for the units warning) ----------
// Anonymous-namespace names stay visible for the remainder of the TU, so these forward without
// moving a single line of the implementations the occlusion stage is built on.
std::vector<P2> occlusionClipToConvex(const std::vector<P2>& subject,
                                      const std::vector<P2>& window)
{
    return clipToConvex(subject, window);
}

std::vector<std::vector<P2>> occlusionUnionConvex(std::vector<std::vector<P2>> polys)
{
    return unionConvexPolys(std::move(polys));
}
} // namespace scene3d
