#pragma once
// =========================================================================
// Photon-Primitive analytical SHAPE CLASSIFICATION.
//
// A single swept vertex turns its outgoing ray into a parametric family of
// endpoints. Which dims are swept (theta / phi / dist) — together with the
// original polar angle theta0 and a couple of degeneracy checks — fully
// determine WHICH analytical surface (or curve) the family traces. The
// "correlated photon primitive" tracer needs this so it can intersect the
// EXACT surface (sphere, cone, disk, plane sector, ...) instead of a
// discrete splat grid.
//
// Geometry recap (single swept vertex V, incoming polar axis A, original
// outgoing direction oOut at polar angle theta0 from A, original distance
// d0; an azimuth phi=0 reference in the plane spanned by A and oOut):
//
//   endpoint(theta, phi, r) = V + r * dir(theta, phi)
//   dir(theta, phi) = cos(theta) * A + sin(theta) * (cos(phi)*T + sin(phi)*B)
//   where {A, T, B} is an orthonormal frame, T = component of oOut perp A.
//
//   sweep set        | dim | analytical shape
//   -----------------+-----+-------------------------------------------------
//   (none)           |  0  | Point            (the original endpoint)
//   phi              |  1  | Circle / CircleArc (radius d0*sin theta0, on the
//                    |     |   plane perp A at height d0*cos theta0); Point if
//                    |     |   theta0 ~ 0
//   theta            |  1  | Arc              (great-circle arc, radius d0)
//   dist             |  1  | LineSegment      (along oOut)
//   theta + phi      |  2  | Sphere / SphericalCap (radius d0, centre V)
//   phi + dist       |  2  | Cone (half-angle theta0); Disk if theta0 ~ pi/2;
//                    |     |   LineSegment if theta0 ~ 0
//   theta + dist     |  2  | PlanarSector     (flat annular sector in the
//                    |     |   A-T plane)
//   theta+phi+dist   |  3  | BallSector       (solid spherical cone)
//
// This module is header-only and depends ONLY on glm so it can be unit
// tested without the GL engine.
// =========================================================================
#include <glm/glm.hpp>
#include <cmath>
#include <vector>

namespace Prism {

enum class PPShape {
    None = 0,      // no swept dims
    Point,         // degenerate (e.g. phi sweep at theta0 == 0)
    LineSegment,   // 1-D along a direction (dist sweep, or degenerate cone)
    Arc,           // 1-D great-circle arc (theta sweep)
    CircleArc,     // 1-D partial circle (phi sweep, range < 2pi)
    Circle,        // 1-D full circle (phi sweep, range ~ 2pi)
    SphericalCap,  // 2-D spherical patch (theta+phi, not full sphere)
    Sphere,        // 2-D full sphere (theta+phi, full ranges)
    Cone,          // 2-D cone lateral surface (phi+dist)
    Disk,          // 2-D flat annular sector (phi+dist at theta0 ~ pi/2)
    PlanarSector,  // 2-D flat sector in the A-T plane (theta+dist)
    BallSector,    // 3-D solid spherical cone (theta+phi+dist)
    // ---- multi-vertex (path-wide) shapes ----
    Parallelogram, // 2-D flat parallelogram (TWO dist sweeps -> two translations)
    Parallelepiped,// 3-D box (THREE dist sweeps -> three translations)
    Parallelotope, // N>3 translation sweeps (higher-dim; not directly drawable)
    Compound       // mixed angular+translation across >1 vertex (general swept
                   // manifold; report dimensionality, no simple analytic form)
};

inline const char* ppShapeName(PPShape s) {
    switch (s) {
        case PPShape::None:         return "None";
        case PPShape::Point:        return "Point";
        case PPShape::LineSegment:  return "LineSegment";
        case PPShape::Arc:          return "Arc";
        case PPShape::CircleArc:    return "CircleArc";
        case PPShape::Circle:       return "Circle";
        case PPShape::SphericalCap: return "SphericalCap";
        case PPShape::Sphere:       return "Sphere";
        case PPShape::Cone:         return "Cone";
        case PPShape::Disk:         return "Disk";
        case PPShape::PlanarSector: return "PlanarSector";
        case PPShape::BallSector:   return "BallSector";
        case PPShape::Parallelogram:  return "Parallelogram";
        case PPShape::Parallelepiped: return "Parallelepiped";
        case PPShape::Parallelotope:  return "Parallelotope";
        case PPShape::Compound:       return "Compound";
    }
    return "?";
}

// Pure inputs — no engine types, so the classifier is unit-testable.
struct PPSweptShapeInput {
    bool  sweepTheta = false;
    bool  sweepPhi   = false;
    bool  sweepDist  = false;
    float rangeTheta = 0.0f;   // radians (polar half-angle extent)
    float rangePhi   = 0.0f;   // radians (azimuth extent; 2pi = full revolution)
    float rangeDist  = 0.0f;   // fraction of d0 (dNew = d0 * u * rangeDist)
    float theta0     = 0.0f;   // original polar angle of oOut from the axis (rad)
    float dist0      = 0.0f;   // original outgoing distance
};

// Analytical parameters the tracer needs. Interpretation depends on `type`:
//   Sphere/SphericalCap : radius = d0; thetaExtent/phiExtent = swept ranges.
//   Cone                : halfAngle = theta0; slant = d0*rangeDist (max radial
//                         distance along the cone ray); phiExtent.
//   Disk                : outerRadius = d0*rangeDist (in the plane perp A).
//   Circle/CircleArc    : radius = d0*sin(theta0); axisOffset = d0*cos(theta0).
//   Arc                 : radius = d0.
//   LineSegment         : length = d0*rangeDist (dist sweep) or d0 (deg. cone).
//   PlanarSector        : outerRadius = d0*rangeDist; thetaExtent.
//   BallSector          : radius = d0*rangeDist; thetaExtent/phiExtent.
struct PPShapeResult {
    PPShape type        = PPShape::None;
    int     dim         = 0;     // 0/1/2/3 swept dims
    float   radius      = 0.0f;
    float   halfAngle   = 0.0f;  // cone half-angle (= theta0) when applicable
    float   slant       = 0.0f;  // cone max radial distance
    float   outerRadius = 0.0f;  // disk / planar-sector / ball-sector extent
    float   axisOffset  = 0.0f;  // circle plane offset along the axis
    float   length      = 0.0f;  // line-segment length
    float   thetaExtent = 0.0f;  // swept theta range echoed for convenience
    float   phiExtent   = 0.0f;  // swept phi range
    bool    full        = false; // full sphere / full circle (else partial)
    // Translation edge vectors for the parallelotope family (Parallelogram /
    // Parallelepiped). Each edge = outDir_v * dist0_v * rangeDist_v for a dist
    // sweep at vertex v. edgeCount = number of dist sweeps.
    glm::vec3 edge0 = glm::vec3(0.0f);
    glm::vec3 edge1 = glm::vec3(0.0f);
    glm::vec3 edge2 = glm::vec3(0.0f);
    int       edgeCount = 0;
};

// Degeneracy / fullness thresholds. Public so tests can reason about them.
inline constexpr float kPPShapeAngleEps = 1e-3f;            // theta0 ~ 0 / pi/2
inline constexpr float kPPShapeTwoPi    = 6.28318530718f;
inline constexpr float kPPShapePi       = 3.14159265359f;
inline constexpr float kPPShapeRangeEps = 1e-2f;            // "full" range slack

inline PPShapeResult classifySweptShape(const PPSweptShapeInput& in) {
    PPShapeResult r;
    int dim = (in.sweepTheta ? 1 : 0) + (in.sweepPhi ? 1 : 0)
            + (in.sweepDist ? 1 : 0);
    r.dim         = dim;
    r.radius      = in.dist0;
    r.halfAngle   = in.theta0;
    r.thetaExtent = in.sweepTheta ? in.rangeTheta : 0.0f;
    r.phiExtent   = in.sweepPhi   ? in.rangePhi   : 0.0f;

    const float th0   = in.theta0;
    const float aeps  = kPPShapeAngleEps;
    const bool  polar = (th0 < aeps);                       // theta0 ~ 0
    const bool  equat = (std::fabs(th0 - kPPShapePi * 0.5f) < aeps); // ~ pi/2

    if (dim == 0) { r.type = PPShape::Point; return r; }

    // ---- 1-D sweeps ----
    if (dim == 1) {
        if (in.sweepDist) {
            r.type   = PPShape::LineSegment;
            r.length = in.dist0 * in.rangeDist;
            return r;
        }
        if (in.sweepTheta) {
            r.type   = PPShape::Arc;          // great-circle arc, radius d0
            r.radius = in.dist0;
            return r;
        }
        // phi-only.
        if (polar) {                          // sin(theta0) ~ 0 -> a point
            r.type = PPShape::Point;
            return r;
        }
        r.radius     = in.dist0 * std::sin(th0);
        r.axisOffset = in.dist0 * std::cos(th0);
        bool fullRev = (in.rangePhi >= kPPShapeTwoPi - kPPShapeRangeEps);
        r.full       = fullRev;
        r.type       = fullRev ? PPShape::Circle : PPShape::CircleArc;
        return r;
    }

    // ---- 3-D sweep ----
    if (dim == 3) {
        r.type        = PPShape::BallSector;
        r.outerRadius = in.dist0 * in.rangeDist;
        return r;
    }

    // ---- 2-D sweeps ----
    if (in.sweepTheta && in.sweepPhi) {       // sphere family
        r.radius   = in.dist0;
        bool fullT = (in.rangeTheta >= kPPShapePi    - kPPShapeRangeEps);
        bool fullP = (in.rangePhi   >= kPPShapeTwoPi - kPPShapeRangeEps);
        r.full     = fullT && fullP;
        r.type     = r.full ? PPShape::Sphere : PPShape::SphericalCap;
        return r;
    }
    if (in.sweepPhi && in.sweepDist) {        // cone family
        if (polar) {                          // dir ~ along axis -> a line
            r.type   = PPShape::LineSegment;
            r.length = in.dist0 * in.rangeDist;
            return r;
        }
        if (equat) {                          // flat -> disk (perp the axis)
            r.type        = PPShape::Disk;
            r.outerRadius = in.dist0 * in.rangeDist;
            return r;
        }
        r.type      = PPShape::Cone;
        r.halfAngle = th0;
        r.slant     = in.dist0 * in.rangeDist;
        return r;
    }
    // theta + dist -> flat annular sector in the A-T plane.
    r.type        = PPShape::PlanarSector;
    r.outerRadius = in.dist0 * in.rangeDist;
    return r;
}

// =========================================================================
// PATH-WIDE (multi-vertex) classification.
//
// A path may sweep dims on MORE THAN ONE vertex. The endpoint manifold's
// shape depends on the FULL multiset of swept (vertex, dim) pairs, not a
// single vertex:
//   - all swept dims are DIST (pure translations): parallelotope by count
//       1 dist  -> LineSegment
//       2 dist  -> Parallelogram (a flat "plane" patch)
//       3 dist  -> Parallelepiped (a box)
//       N>3     -> Parallelotope (higher-dim)
//   - all swept dims on ONE vertex: the single-vertex shapes above
//       (Sphere / Cone / Disk / Circle / Arc / ...).
//   - otherwise (angular dims mixed across >1 vertex): Compound — a general
//     swept manifold with no simple closed form; we report its dimensionality.
//
// A dist sweep at vertex v translates the endpoint along v's outgoing
// direction by dist0_v * rangeDist_v (u in [0,1]); pure translations commute,
// so the all-dist manifold is exactly the parallelotope spanned by those edge
// vectors. Those edges are returned for the (future) analytic tracer.
struct PPVertexSweep {
    int   vertexIdx  = -1;
    bool  sweepTheta = false, sweepPhi = false, sweepDist = false;
    float rangeTheta = 0.0f, rangePhi = 0.0f, rangeDist = 0.0f;
    float theta0     = 0.0f, dist0    = 0.0f;
    glm::vec3 outDir = glm::vec3(0.0f);  // outgoing dir (for translation edges)
};

inline PPShapeResult classifyPathShape(const std::vector<PPVertexSweep>& sweeps) {
    PPShapeResult r;

    int nTheta = 0, nPhi = 0, nDist = 0, nActive = 0, firstActive = -1;
    for (size_t i = 0; i < sweeps.size(); ++i) {
        const PPVertexSweep& s = sweeps[i];
        if (!(s.sweepTheta || s.sweepPhi || s.sweepDist)) continue;
        if (firstActive < 0) firstActive = (int)i;
        ++nActive;
        if (s.sweepTheta) ++nTheta;
        if (s.sweepPhi)   ++nPhi;
        if (s.sweepDist)  ++nDist;
    }
    int total = nTheta + nPhi + nDist;
    r.dim = total;
    if (total == 0) { r.type = PPShape::Point; return r; }

    // ---- all-translation (only dist sweeps) -> parallelotope ----
    if (nTheta == 0 && nPhi == 0) {
        glm::vec3 e[3]; int ne = 0;
        for (const PPVertexSweep& s : sweeps) {
            if (!s.sweepDist) continue;
            if (ne < 3) e[ne] = s.outDir * (s.dist0 * s.rangeDist);
            ++ne;
        }
        r.dim       = nDist;
        r.edgeCount = nDist;
        if (nDist >= 1) r.edge0 = e[0];
        if (nDist >= 2) r.edge1 = e[1];
        if (nDist >= 3) r.edge2 = e[2];
        if (nDist == 1) { r.type = PPShape::LineSegment;
                          r.length = glm::length(e[0]); return r; }
        if (nDist == 2) { r.type = PPShape::Parallelogram;  return r; }
        if (nDist == 3) { r.type = PPShape::Parallelepiped; return r; }
        r.type = PPShape::Parallelotope;                    return r;  // N>3
    }

    // ---- all swept dims on ONE vertex -> single-vertex shapes ----
    if (nActive == 1) {
        const PPVertexSweep& s = sweeps[(size_t)firstActive];
        PPSweptShapeInput in;
        in.sweepTheta = s.sweepTheta; in.sweepPhi = s.sweepPhi; in.sweepDist = s.sweepDist;
        in.rangeTheta = s.rangeTheta; in.rangePhi = s.rangePhi; in.rangeDist = s.rangeDist;
        in.theta0     = s.theta0;     in.dist0    = s.dist0;
        return classifySweptShape(in);
    }

    // ---- mixed angular across >1 vertex -> compound manifold ----
    r.type = PPShape::Compound;
    r.dim  = total;
    return r;
}

} // namespace Prism
