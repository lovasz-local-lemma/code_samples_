// backend/src/lab3d/caustic_solver.hpp -- GL-free EXACT ray-fan analytics for the
// {planar walls, ONE glass sphere, point light} scene class (c = 1). Extends
// wavefront_solver.hpp (NOT modified; its consumers are byte-locked) by KEEPING
// per-ray records so the RIM caustic envelope (adjacent-ray crossings), the
// AXIAL caustic segment (separate on purpose -- consumers draw them as distinct
// primitives), fold-localized wavefront BRANCHES, intersection circles, and
// revolution samples all become small closed-form functions of one structure.
// Double internally, float at the API edge, scale-free relative thresholds only
// (the wavefront_solver convention).
#pragma once
#include "lab3d/math_3d.hpp"          // Vec3f (+ normalize/cross/dot)
#include "lab3d/wavefront_solver.hpp" // imageSource, refractDir2, MeridianPt
#include <algorithm>
#include <cmath>
#include <vector>

namespace rive_backend
{

enum class FanRayStatus : int { Alive = 0, Miss = 1, TIR = 2, Graze = 3 };

// One meridian ray: pts[0]=light(0,0), pts[1]=entry, pts[2]=exit; dir = the final
// unit direction after the exit refraction; tCum = cumulative OPTICAL time.
struct FanRay
{
    double theta = 0.0;
    int n = 0;                       // valid points (3 when Alive)
    double pts[3][2] = {{0,0},{0,0},{0,0}};
    double tCum[3] = {0,0,0};
    double dir[2] = {1,0};
    FanRayStatus status = FanRayStatus::Miss;
};

struct RayFan
{
    std::vector<FanRay> rays;        // ascending theta
    double D = 0.0, R = 0.0, ior = 1.5;
    float light[3] = {0,0,0};
    float axis[3] = {0,0,1};         // unit, light -> sphere center (world)
    float coneFrac = 1.0f;
};

struct MeridianCurve { std::vector<MeridianPt> pts; };

struct Circle3D
{
    float center[3] = {0,0,0};
    float radius = 0.0f;
    float normal[3] = {0,1,0};
    bool valid = false;
};

// Trace ONE meridian ray at launch angle theta against the fan's sphere. The
// Snell math replicates wavefront_solver.hpp:144-187 exactly (entry root,
// refract in eta=1/n, exit chord s2 = -2 g.(e1-C), refract out eta=n).
inline FanRay traceMeridianRay(const RayFan& fan, double theta)
{
    FanRay ray;
    ray.theta = theta;
    const double D = fan.D, Rd = fan.R, nd = fan.ior;
    const double dA = std::cos(theta), dR = std::sin(theta);
    const double b = dA * D;
    const double disc = b * b - (D * D - Rd * Rd);
    if (disc <= 0.0) { ray.status = FanRayStatus::Miss; return ray; }
    const double s1 = b - std::sqrt(disc);
    if (s1 <= 0.0) { ray.status = FanRayStatus::Miss; return ray; }
    const double e1A = dA * s1, e1R = dR * s1;
    const double m1A = (e1A - D) / Rd, m1R = e1R / Rd;
    double g1A, g1R;
    if (!refractDir2(dA, dR, m1A, m1R, 1.0 / nd, g1A, g1R))
    { ray.status = FanRayStatus::TIR; return ray; }
    const double w1A = e1A - D, w1R = e1R;
    const double s2 = -2.0 * (g1A * w1A + g1R * w1R);
    if (s2 <= 0.0) { ray.status = FanRayStatus::Graze; return ray; }
    const double e2A = e1A + g1A * s2, e2R = e1R + g1R * s2;
    const double m2A = (D - e2A) / Rd, m2R = -e2R / Rd;
    double hA, hR;
    if (!refractDir2(g1A, g1R, m2A, m2R, nd, hA, hR))
    { ray.status = FanRayStatus::TIR; return ray; }
    ray.status = FanRayStatus::Alive;
    ray.n = 3;
    ray.pts[0][0] = 0.0;  ray.pts[0][1] = 0.0;
    ray.pts[1][0] = e1A;  ray.pts[1][1] = e1R;
    ray.pts[2][0] = e2A;  ray.pts[2][1] = e2R;
    ray.tCum[0] = 0.0;
    ray.tCum[1] = s1;
    ray.tCum[2] = s1 + nd * s2;      // optical: n * geometric inside the glass
    ray.dir[0] = hA; ray.dir[1] = hR;
    return ray;
}

// Trace the fan: N rays over theta = thetaMax*(k+0.5)/N, thetaMax =
// coneFrac*asin(R/D) (the wavefront_solver launch grid). Degenerate configs
// (light inside/on the sphere, bad params) return an empty fan.
inline RayFan traceMeridianFan(const float light[3], const float center[3], float R,
                               float n, int N, float coneFrac = 1.0f)
{
    RayFan fan;
    for (int i = 0; i < 3; ++i) fan.light[i] = light[i];
    const double cA0 = (double)center[0] - light[0], cA1 = (double)center[1] - light[1],
                 cA2 = (double)center[2] - light[2];
    const double D = std::sqrt(cA0 * cA0 + cA1 * cA1 + cA2 * cA2);
    fan.D = D; fan.R = R; fan.ior = n;
    if (N <= 0 || R <= 0.0f || n <= 0.0f || D <= (double)R + 1e-9) return fan;
    if (coneFrac > 1.0f) coneFrac = 1.0f;
    if (coneFrac <= 0.0f) return fan;
    fan.coneFrac = coneFrac;
    fan.axis[0] = (float)(cA0 / D);
    fan.axis[1] = (float)(cA1 / D);
    fan.axis[2] = (float)(cA2 / D);
    const double sinMax = (double)R / D;
    const double thetaMax = (double)coneFrac * std::asin(sinMax < 1.0 ? sinMax : 1.0);
    fan.rays.reserve((size_t)N);
    for (int k = 0; k < N; ++k)
        fan.rays.push_back(traceMeridianRay(fan, thetaMax * ((double)k + 0.5) / (double)N));
    return fan;
}

// Position of an ALIVE ray at time t (t >= entry time; the direct front covers
// pre-lens times). ok=false when t < entry. TIME MODE (CL-T4 review F1): ior <= 0
// (the default) keeps the locked OPTICAL clock -- t advances by n * geometric
// length inside the glass (tCum[2] = s1 + n*s2, light slows to 1/n). ior > 0 means
// t is GEOMETRIC time (the estimators' default TOF clock, uTofOptical == 0): the
// air legs are identical (entry still at tCum[1]), but the in-glass leg takes only
// its geometric length s2 = (tCum[2]-tCum[1])/ior, so the front traverses the
// glass n times faster and everything after it leads the optical front by
// (n-1)*s2 along the exit direction.
inline bool rayPositionAt(const FanRay& r, double t, double& a, double& rr,
                          double ior = 0.0)
{
    if (r.status != FanRayStatus::Alive || t < r.tCum[1]) return false;
    const double optLen = r.tCum[2] - r.tCum[1];                   // n * s2
    const double segLen = ior > 0.0 ? optLen / ior : optLen;       // s2 when geometric
    if (t < r.tCum[1] + segLen)
    {   // inside the glass: linear along entry->exit in the selected clock
        const double f = (t - r.tCum[1]) / (segLen > 1e-15 ? segLen : 1e-15);
        a  = r.pts[1][0] + (r.pts[2][0] - r.pts[1][0]) * f;
        rr = r.pts[1][1] + (r.pts[2][1] - r.pts[1][1]) * f;
        return true;
    }
    const double tau = t - (r.tCum[1] + segLen);
    a  = r.pts[2][0] + r.dir[0] * tau;
    rr = r.pts[2][1] + r.dir[1] * tau;
    return true;
}

// The refracted wavefront at time t, split into BRANCHES at fold points (turning
// sign changes of the |r|-reflected polyline, the wavefront_solver convention:
// relative cutoff |cross| >= 1e-9*|v1||v2|). Point-for-point identical to
// lensMeridianWavefront's emitted points; only the branch structure is new.
// The hinge point is SHARED (last point of one branch == first of the next).
// Branches carry >= 1 point: only the FIRST branch can be a singleton (a lone
// ray in the entry window); fold branches start at 2 via the shared hinge.
// TIME MODE (CL-T4 review F1): opticalTime = true (the default) keeps the locked
// optical clock; false passes fan.ior through to rayPositionAt so t is GEOMETRIC
// time (matches the estimators' default TOF, uTofOptical == 0). Fold detection /
// branching is unchanged -- it runs on the emitted points whatever the clock.
// causticEnvelope / causticAxial / lensShadowSeam and the direct-front rings are
// air-only exit/tangent geometry, unaffected by the time mode.
inline std::vector<MeridianCurve> wavefrontAt(const RayFan& fan, float t,
                                              bool opticalTime = true)
{
    const double posIor = opticalTime ? 0.0 : fan.ior;
    std::vector<MeridianCurve> out;
    MeridianCurve cur;
    double p0a = 0, p0r = 0, p1a = 0, p1r = 0;
    int nPts = 0, prevSign = 0;
    auto flush = [&]() {
        if (!cur.pts.empty()) out.push_back(cur);
        cur.pts.clear();
    };
    for (const FanRay& r : fan.rays)
    {
        double A, Rr;
        if (!rayPositionAt(r, (double)t, A, Rr, posIor)) continue;
        if (Rr < 0.0) Rr = -Rr;
        if (nPts >= 2)
        {
            const double v1a = p1a - p0a, v1r = p1r - p0r;
            const double v2a = A - p1a, v2r = Rr - p1r;
            const double cr = v1a * v2r - v1r * v2a;
            const double mag =
                std::sqrt((v1a * v1a + v1r * v1r) * (v2a * v2a + v2r * v2r));
            if (mag > 0.0 && std::fabs(cr) >= 1e-9 * mag)
            {
                const int s = cr > 0.0 ? 1 : -1;
                if (prevSign != 0 && s != prevSign)
                {   // FOLD between the previous point and this one: split here.
                    flush();
                    MeridianPt back; back.a = (float)p1a; back.r = (float)p1r;
                    cur.pts.push_back(back);  // share the hinge point
                }
                prevSign = s;
            }
        }
        p0a = p1a; p0r = p1r; p1a = A; p1r = Rr; ++nPts;
        MeridianPt mp; mp.a = (float)A; mp.r = (float)Rr;
        cur.pts.push_back(mp);
    }
    flush();
    return out;
}

// Crossing of two ALIVE rays' post-exit segments (2D line intersection with both
// parameters >= 0 -- forward of both exits). The adjacent-ray crossing IS the
// envelope point between their thetas.
inline bool rayPairCrossing(const FanRay& a, const FanRay& b, double& cx, double& cr)
{
    if (a.status != FanRayStatus::Alive || b.status != FanRayStatus::Alive) return false;
    const double ax = a.pts[2][0], ar = a.pts[2][1];
    const double bx = b.pts[2][0], br = b.pts[2][1];
    const double det = a.dir[0] * (-b.dir[1]) - a.dir[1] * (-b.dir[0]);
    const double mag = std::sqrt((a.dir[0]*a.dir[0] + a.dir[1]*a.dir[1]) *
                                 (b.dir[0]*b.dir[0] + b.dir[1]*b.dir[1]));
    if (std::fabs(det) < 1e-12 * (mag > 0.0 ? mag : 1.0)) return false; // parallel
    const double rx = bx - ax, rr = br - ar;
    const double sA = (rx * (-b.dir[1]) - rr * (-b.dir[0])) / det;
    const double sB = (a.dir[0] * rr - a.dir[1] * rx) / det;
    if (sA < 0.0 || sB < 0.0) return false;
    cx = ax + a.dir[0] * sA;
    cr = ar + a.dir[1] * sA;
    return true;
}

// The meridian RIM caustic: adjacent-ray crossing loci, theta-ordered. RIM ONLY:
// the axial focus segment lives in causticAxial below -- appending it here as a
// tail made the two parts inseparable, so every polyline consumer drew a bogus
// rim-to-axis chord (and revolved/exported it).
inline MeridianCurve causticEnvelope(const RayFan& fan)
{
    MeridianCurve env;
    for (size_t i = 0; i + 1 < fan.rays.size(); ++i)
    {
        double cx, cr;
        if (!rayPairCrossing(fan.rays[i], fan.rays[i + 1], cx, cr)) continue;
        MeridianPt p; p.a = (float)cx; p.r = (float)(cr < 0.0 ? -cr : cr);
        env.pts.push_back(p);
    }
    return env;
}

// The AXIAL caustic segment: each ALIVE exit ray with dir.r < 0 AND exit.r > 0
// crosses r = 0 at a = exit.a - exit.r * dir.a/dir.r; the [min,max] span of those
// crossings is returned as a 2-point curve (empty when no ray crosses).
// FORWARD-INTERCEPT REQUIREMENT: with dir.r < 0 the ray parameter of the axis
// crossing is s = -exit.r/dir.r, which is >= 0 iff exit.r >= 0. A ray that
// already crossed the axis INSIDE the glass exits below it (exit.r < 0) and its
// line intercept lies BEHIND the exit point -- fictitious, so it is skipped.
// NEAR-COLLIMATION CAVEAT: with the light near the front focal point the exit
// rays approach parallel and the span legitimately extends toward infinity; the
// solver does NOT cap it -- consumers must clamp to their draw volume.
inline MeridianCurve causticAxial(const RayFan& fan)
{
    MeridianCurve seg;
    double aMin = 1e30, aMax = -1e30;
    for (const FanRay& r : fan.rays)
    {
        if (r.status != FanRayStatus::Alive || r.dir[1] >= -1e-15) continue;
        if (r.pts[2][1] <= 0.0) continue;   // forward intercepts only (see above)
        const double aAxis = r.pts[2][0] - r.pts[2][1] * (r.dir[0] / r.dir[1]);
        if (aAxis < aMin) aMin = aAxis;
        if (aAxis > aMax) aMax = aAxis;
    }
    if (aMax >= aMin)
    {
        MeridianPt p0; p0.a = (float)aMin; p0.r = 0.0f;
        MeridianPt p1; p1.a = (float)aMax; p1.r = 0.0f;
        seg.pts.push_back(p0);
        seg.pts.push_back(p1);
    }
    return seg;
}

// --- intersection circles (all closed-form) --------------------------------
inline Circle3D sphereSphereCircle(Vec3f c0, float r0, Vec3f c1, float r1)
{
    Circle3D c;
    const Vec3f dv{c1.x - c0.x, c1.y - c0.y, c1.z - c0.z};
    const float d = std::sqrt(dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
    // Concentric guard is RELATIVE to the radius scale (scale-free convention).
    if (d < 1e-9f * (r0 + r1) || d >= r0 + r1 || d <= std::fabs(r0 - r1)) return c;
    const float a = (d * d + r0 * r0 - r1 * r1) / (2.0f * d);
    const float rr2 = r0 * r0 - a * a;
    if (rr2 <= 0.0f) return c;
    const float inv = 1.0f / d;
    c.normal[0] = dv.x * inv; c.normal[1] = dv.y * inv; c.normal[2] = dv.z * inv;
    c.center[0] = c0.x + c.normal[0] * a;
    c.center[1] = c0.y + c.normal[1] * a;
    c.center[2] = c0.z + c.normal[2] * a;
    c.radius = std::sqrt(rr2);
    c.valid = true;
    return c;
}
inline Circle3D spherePlaneCircle(Vec3f c0, float r, Vec3f p, Vec3f nrm)
{
    Circle3D c;
    const float nl = std::sqrt(nrm.x*nrm.x + nrm.y*nrm.y + nrm.z*nrm.z);
    if (nl < 1e-9f) return c;
    const float nx = nrm.x / nl, ny = nrm.y / nl, nz = nrm.z / nl;
    const float h = (c0.x - p.x) * nx + (c0.y - p.y) * ny + (c0.z - p.z) * nz;
    if (std::fabs(h) >= r) return c;
    c.center[0] = c0.x - h * nx; c.center[1] = c0.y - h * ny; c.center[2] = c0.z - h * nz;
    c.normal[0] = nx; c.normal[1] = ny; c.normal[2] = nz;
    c.radius = std::sqrt(r * r - h * h);
    c.valid = true;
    return c;
}
// The seam where the DIRECT spherical front meets the lens shadow cone: a circle
// about the axis at angle thetaGeo = asin(R/D) from the light, radius t*sin.
// REACH GATE: the seam only exists once the front reaches the TANGENT circle at
// distance sqrt(D^2 - R^2); between first sphere contact (t = D - R) and that
// reach the true shadow boundary is the front-sphere intersection circle, which
// this function deliberately does NOT model (returns invalid there -- use
// spherePlaneCircle/sphereSphereCircle style constructions if that regime is
// ever needed). A parameter-REJECTED fan (rays empty: N<=0, R<=0, n<=0,
// coneFrac<=0, light inside the sphere) is invalid too: its D/R may be
// populated but its axis was never set.
inline Circle3D lensShadowSeam(const RayFan& fan, float t)
{
    Circle3D c;
    if (fan.rays.empty() || fan.D <= fan.R + 1e-9 || t <= 0.0f) return c;
    if ((double)t < std::sqrt(fan.D * fan.D - fan.R * fan.R)) return c;
    const double sinT = fan.R / fan.D;
    const double cosT = std::sqrt(1.0 - sinT * sinT);
    for (int i = 0; i < 3; ++i) c.normal[i] = fan.axis[i];
    for (int i = 0; i < 3; ++i)
        c.center[i] = fan.light[i] + fan.axis[i] * (float)((double)t * cosT);
    c.radius = (float)((double)t * sinT);
    c.valid = true;
    return c;
}

// --- revolution ------------------------------------------------------------
// Orthonormal frame perpendicular to the fan axis.
inline void fanBasis(const RayFan& fan, Vec3f& u1, Vec3f& u2)
{
    const Vec3f ax{fan.axis[0], fan.axis[1], fan.axis[2]};
    const Vec3f ref = std::fabs(ax.x) < 0.9f ? Vec3f{1, 0, 0} : Vec3f{0, 1, 0};
    u1 = normalize(cross(ax, ref));
    u2 = cross(ax, u1);
}
// The meridian curve revolved into `azimuthN` 3D profile polylines.
inline std::vector<std::vector<Vec3f>> revolveCurve(const MeridianCurve& c,
                                                    const RayFan& fan, int azimuthN)
{
    std::vector<std::vector<Vec3f>> out;
    if (c.pts.size() < 2 || azimuthN < 1) return out;
    Vec3f u1, u2; fanBasis(fan, u1, u2);
    const Vec3f L{fan.light[0], fan.light[1], fan.light[2]};
    const Vec3f ax{fan.axis[0], fan.axis[1], fan.axis[2]};
    out.reserve((size_t)azimuthN);
    for (int j = 0; j < azimuthN; ++j)
    {
        const float phi = 6.2831853f * (float)j / (float)azimuthN;
        const float cp = std::cos(phi), spn = std::sin(phi);
        std::vector<Vec3f> line;
        line.reserve(c.pts.size());
        for (const MeridianPt& p : c.pts)
            line.push_back(Vec3f{L.x + ax.x * p.a + (u1.x * cp + u2.x * spn) * p.r,
                                 L.y + ax.y * p.a + (u1.y * cp + u2.y * spn) * p.r,
                                 L.z + ax.z * p.a + (u1.z * cp + u2.z * spn) * p.r});
        out.push_back(std::move(line));
    }
    return out;
}
// A single revolved ring (for circles / per-point rings).
inline std::vector<Vec3f> circlePolyline(const Circle3D& c, int segs)
{
    std::vector<Vec3f> out;
    if (!c.valid || segs < 3) return out;
    const Vec3f n{c.normal[0], c.normal[1], c.normal[2]};
    const Vec3f ref = std::fabs(n.x) < 0.9f ? Vec3f{1, 0, 0} : Vec3f{0, 1, 0};
    const Vec3f u1 = normalize(cross(n, ref));
    const Vec3f u2 = cross(n, u1);
    out.reserve((size_t)segs + 1);
    for (int j = 0; j <= segs; ++j)
    {
        const float phi = 6.2831853f * (float)j / (float)segs;
        const float cp = std::cos(phi), spn = std::sin(phi);
        out.push_back(Vec3f{c.center[0] + (u1.x * cp + u2.x * spn) * c.radius,
                            c.center[1] + (u1.y * cp + u2.y * spn) * c.radius,
                            c.center[2] + (u1.z * cp + u2.z * spn) * c.radius});
    }
    return out;
}
// Surface samples of the revolved meridian curve, each with a COMPRESSION weight
// = 1 / max(parametric speed, eps): where consecutive curve points bunch (the
// focus/tip) the weight peaks -- the splat-shading brightness heuristic. The
// spacing floor eps is RELATIVE to the curve's total arclength (1e-6 * arc) so
// the weights are scale-free; an absolute floor saturated every weight to 1.0
// at world scales <= ~1e-3.
struct RevolvedSample { Vec3f pos; float compression = 0.0f; };
inline std::vector<RevolvedSample> revolveSurfaceSamples(const MeridianCurve& c,
                                                         const RayFan& fan, int azimuthN)
{
    std::vector<RevolvedSample> out;
    if (c.pts.size() < 2 || azimuthN < 3) return out;
    float arc = 0.0f;
    for (size_t i = 1; i < c.pts.size(); ++i)
    {
        const float da = c.pts[i].a - c.pts[i - 1].a, dr = c.pts[i].r - c.pts[i - 1].r;
        arc += std::sqrt(da * da + dr * dr);
    }
    const float eps = (arc > 0.0f ? arc : 1.0f) * 1e-6f;
    std::vector<float> w(c.pts.size(), 0.0f);
    for (size_t i = 0; i < c.pts.size(); ++i)
    {
        const size_t j = i + 1 < c.pts.size() ? i + 1 : i - 1;
        const float da = c.pts[i].a - c.pts[j].a, dr = c.pts[i].r - c.pts[j].r;
        const float sp = std::sqrt(da * da + dr * dr);
        w[i] = 1.0f / (sp > eps ? sp : eps);
    }
    float wMax = 0.0f;
    for (float v : w) wMax = std::max(wMax, v);
    if (wMax <= 0.0f) wMax = 1.0f;
    Vec3f u1, u2; fanBasis(fan, u1, u2);
    const Vec3f L{fan.light[0], fan.light[1], fan.light[2]};
    const Vec3f ax{fan.axis[0], fan.axis[1], fan.axis[2]};
    out.reserve(c.pts.size() * (size_t)azimuthN);
    for (size_t i = 0; i < c.pts.size(); ++i)
        for (int j = 0; j < azimuthN; ++j)
        {
            const float phi = 6.2831853f * (float)j / (float)azimuthN;
            const float cp = std::cos(phi), spn = std::sin(phi);
            RevolvedSample s;
            s.pos = Vec3f{L.x + ax.x * c.pts[i].a + (u1.x * cp + u2.x * spn) * c.pts[i].r,
                          L.y + ax.y * c.pts[i].a + (u1.y * cp + u2.y * spn) * c.pts[i].r,
                          L.z + ax.z * c.pts[i].a + (u1.z * cp + u2.z * spn) * c.pts[i].r};
            s.compression = w[i] / wMax; // normalized 0..1
            out.push_back(s);
        }
    return out;
}

// --- overlay clipping (CL-UI3 item 1) --------------------------------------
// The analytic curves are unbounded geometry (whole spheres, whole wall-plane
// circles); the scene is a finite box with one solid glass sphere. These helpers
// trim the world-space polylines to the actual scene so boundaries end at the
// walls and the lens shadow reads as a hole. GL-free, unit-tested, scale-free.

// Liang-Barsky clip of the segment (a, a+d) to the axis-aligned box centered at
// the origin with half-extents half[3] (grown by eps so points ON a face -- the
// wall-plane intersection rings -- survive). Returns false if the segment is
// wholly outside; else t0 <= t1 in [0,1] bound the inside portion.
inline bool clipSegmentToBox(const Vec3f& a, const Vec3f& d, const float half[3],
                             float eps, float& t0, float& t1)
{
    t0 = 0.0f; t1 = 1.0f;
    const float ac[3] = {a.x, a.y, a.z};
    const float dc[3] = {d.x, d.y, d.z};
    for (int i = 0; i < 3; ++i)
    {
        const float lo = -half[i] - eps, hi = half[i] + eps;
        if (std::fabs(dc[i]) < 1e-12f)
        {
            if (ac[i] < lo || ac[i] > hi) return false; // parallel to slab, outside it
        }
        else
        {
            float tA = (lo - ac[i]) / dc[i];
            float tB = (hi - ac[i]) / dc[i];
            if (tA > tB) std::swap(tA, tB);
            t0 = std::max(t0, tA);
            t1 = std::min(t1, tB);
            if (t0 > t1) return false;
        }
    }
    return true;
}

// Clip a world-space polyline to the box, returning the inside runs (a curve that
// leaves and re-enters the box yields multiple runs). Runs with < 2 points drop.
inline std::vector<std::vector<Vec3f>> clipPolylineToBox(const std::vector<Vec3f>& pl,
                                                         const float half[3],
                                                         float eps = 1e-4f)
{
    std::vector<std::vector<Vec3f>> out;
    std::vector<Vec3f> run;
    auto flush = [&]() { if (run.size() >= 2) out.push_back(std::move(run)); run.clear(); };
    for (size_t i = 0; i + 1 < pl.size(); ++i)
    {
        const Vec3f& a = pl[i];
        const Vec3f& b = pl[i + 1];
        const Vec3f d{b.x - a.x, b.y - a.y, b.z - a.z};
        float t0, t1;
        if (!clipSegmentToBox(a, d, half, eps, t0, t1)) { flush(); continue; }
        const Vec3f ca{a.x + d.x * t0, a.y + d.y * t0, a.z + d.z * t0};
        const Vec3f cb{a.x + d.x * t1, a.y + d.y * t1, a.z + d.z * t1};
        if (t0 > 0.0f) flush();          // segment entered the box mid-way: start a new run
        if (run.empty()) run.push_back(ca);
        run.push_back(cb);
        if (t1 < 1.0f) flush();          // segment left the box mid-way: end this run
    }
    flush();
    return out;
}

// True when the sphere (center c, radius R) blocks the straight line of sight from
// L to P -- i.e. a ray-sphere boundary crossing along [L,P] falls strictly between
// them. This is the DIRECT-light visibility test: P is in the sphere's shadow (or
// behind glass from an origin inside it) exactly when this returns true.
// TF-T3 fix 3: EITHER quadratic root counts. The old near-root-only test had s0 < 0
// for an origin inside/on the sphere (cc <= 0) and silently reported UNBLOCKED for
// every P -- an inside origin is blocked toward any OUTSIDE P (the segment must exit
// through the glass at the far root), and an on-surface origin heading inward crosses
// the whole sphere with s0 ~ 0 < eps. An inside-to-inside segment (no root within
// (eps, 1-eps)) stays unblocked. sHit returns the first blocking root when blocked
// (the near crossing normally; the EXIT root for the inside/on-surface origin), else
// the near root for callers that refine the boundary.
inline bool segmentBlockedBySphere(const Vec3f& L, const Vec3f& P, const Vec3f& c,
                                   float R, float eps, float& sHit)
{
    const Vec3f f{P.x - L.x, P.y - L.y, P.z - L.z};
    const float aa = f.x * f.x + f.y * f.y + f.z * f.z;
    sHit = 2.0f;
    if (aa < 1e-20f) return false;
    const Vec3f oc{L.x - c.x, L.y - c.y, L.z - c.z};
    const float bb = 2.0f * (oc.x * f.x + oc.y * f.y + oc.z * f.z);
    const float cc = oc.x * oc.x + oc.y * oc.y + oc.z * oc.z - R * R;
    const float disc = bb * bb - 4.0f * aa * cc;
    if (disc <= 0.0f) return false;
    const float sq = std::sqrt(disc);
    const float s0 = (-bb - sq) / (2.0f * aa);
    const float s1 = (-bb + sq) / (2.0f * aa);
    if (s0 > eps && s0 < 1.0f - eps) { sHit = s0; return true; }
    if (s1 > eps && s1 < 1.0f - eps) { sHit = s1; return true; }
    sHit = s0;
    return false;
}

// Keep only the parts of a DIRECT-light polyline with clear line of sight from L
// (not blocked by the sphere). Lit<->shadow transitions are refined to the shadow
// boundary by bisection so the arcs terminate exactly on the lens-shadow seam.
inline std::vector<std::vector<Vec3f>> clipPolylineBySphereShadow(
    const std::vector<Vec3f>& pl, const Vec3f& L, const Vec3f& c, float R,
    float eps = 1e-4f)
{
    std::vector<std::vector<Vec3f>> out;
    if (pl.size() < 2 || R <= 0.0f) { if (pl.size() >= 2) out.push_back(pl); return out; }
    auto lit = [&](const Vec3f& P) { float s; return !segmentBlockedBySphere(L, P, c, R, eps, s); };
    auto refine = [&](Vec3f litP, Vec3f darkP) {
        for (int it = 0; it < 20; ++it)
        {
            const Vec3f m{0.5f * (litP.x + darkP.x), 0.5f * (litP.y + darkP.y),
                          0.5f * (litP.z + darkP.z)};
            if (lit(m)) litP = m; else darkP = m;
        }
        return litP;
    };
    std::vector<Vec3f> run;
    auto flush = [&]() { if (run.size() >= 2) out.push_back(std::move(run)); run.clear(); };
    bool prevLit = lit(pl[0]);
    if (prevLit) run.push_back(pl[0]);
    for (size_t i = 1; i < pl.size(); ++i)
    {
        const bool curLit = lit(pl[i]);
        if (curLit && prevLit)
            run.push_back(pl[i]);
        else if (curLit && !prevLit)     // shadow -> lit: enter at the boundary
        {
            run.push_back(refine(pl[i], pl[i - 1]));
            run.push_back(pl[i]);
        }
        else if (!curLit && prevLit)     // lit -> shadow: exit at the boundary
        {
            run.push_back(refine(pl[i - 1], pl[i]));
            flush();
        }
        prevLit = curLit;
    }
    flush();
    return out;
}

// Image-source composition: one mirror fold, then the sphere -- still
// axisymmetric about (image -> center).
inline RayFan foldedFan(const float light[3], const float panelP[3],
                        const float panelN[3], const float center[3], float R, float n,
                        int N, float coneFrac = 1.0f)
{
    float img[3];
    imageSource(light, panelP, panelN, img);
    return traceMeridianFan(img, center, R, n, N, coneFrac);
}

} // namespace rive_backend
