// backend/src/lab3d/primitive_project.hpp
// GL-free analytic projection of 3D primitives to 2D screen shapes for the portable
// transform-keyed .riv export, in two camera families:
//   ORTHOGRAPHIC (projectSphere/Disk/PlaneQuad/ConeTriangle): affine -- a sphere is a
//     constant-radius circle, a disk a foreshortened ellipse (minor = major *
//     |normal . viewFwd|, oriented along the screen projection of the tilt).
//   PERSPECTIVE (CS-T5, the *Persp variants): the live-lab pinhole camera -- a sphere's
//     silhouette is the EXACT tangent-cone ellipse, a disk's rim a moment-fit ellipse,
//     planes/cones project their corners through projectPoint.
#pragma once
#include "lab3d/math_3d.hpp"
#include <cmath>
#include <vector>
#include <utility>
namespace rive_backend
{
// cx,cy = screen centre; a = semi-MAJOR (px); b = semi-MINOR (px); rot = radians of the
// major axis on screen. Maps to a Rive Shape(x=cx,y=cy,rotation=rot) + Ellipse(W=2a,H=2b).
struct ScreenEllipse { float cx = 0, cy = 0, a = 0, b = 0, rot = 0; };

inline ScreenEllipse projectSphere(const CameraFrame& c, Vec3f target, Vec3f center, float r,
                                   float vpW, float vpH, float halfExtent)
{
    const float s = orthoScale(vpW, vpH, halfExtent);
    const Projected p = projectPointOrtho(c, target, center, vpW, vpH, halfExtent);
    ScreenEllipse e; e.cx = p.x; e.cy = p.y; e.a = r * s; e.b = r * s; e.rot = 0.0f;
    return e;
}

inline ScreenEllipse projectDisk(const CameraFrame& c, Vec3f target, Vec3f center, float r,
                                 Vec3f normal, float vpW, float vpH, float halfExtent)
{
    const float s = orthoScale(vpW, vpH, halfExtent);
    const Projected p = projectPointOrtho(c, target, center, vpW, vpH, halfExtent);
    const Vec3f n = normalize(normal);
    const float cosv = std::fabs(dot(n, c.fwd));
    ScreenEllipse e; e.cx = p.x; e.cy = p.y;
    e.a = r * s;
    e.b = r * s * cosv;
    const float nrx = dot(n, c.right), nry = -dot(n, c.up);
    if (cosv > 0.999f || (nrx * nrx + nry * nry) < 1e-8f)
        e.rot = 0.0f;
    else
        e.rot = std::atan2(nry, nrx) + 1.57079632679f;
    return e;
}

inline std::vector<std::pair<float,float>> projectPlaneQuad(const CameraFrame& c, Vec3f target,
                                                            const Vec3f corners[4],
                                                            float vpW, float vpH, float halfExtent)
{
    std::vector<std::pair<float,float>> out; out.reserve(4);
    for (int i = 0; i < 4; ++i) {
        const Projected p = projectPointOrtho(c, target, corners[i], vpW, vpH, halfExtent);
        out.emplace_back(p.x, p.y);
    }
    return out;
}

inline std::vector<std::pair<float,float>> projectConeTriangle(const CameraFrame& c, Vec3f target,
                                                               Vec3f apex, Vec3f baseCenter,
                                                               float baseR, Vec3f axis,
                                                               float vpW, float vpH, float halfExtent)
{
    const ScreenEllipse be = projectDisk(c, target, baseCenter, baseR, axis, vpW, vpH, halfExtent);
    const float ca = std::cos(be.rot), sa = std::sin(be.rot);
    const Projected ap = projectPointOrtho(c, target, apex, vpW, vpH, halfExtent);
    std::vector<std::pair<float,float>> out;
    out.emplace_back(ap.x, ap.y);
    out.emplace_back(be.cx + be.a * ca, be.cy + be.a * sa);
    out.emplace_back(be.cx - be.a * ca, be.cy - be.a * sa);
    return out;
}

// ---------------------------------------------------------------------------
// CS-T5 perspective solvers. All take the SAME CameraFrame the live lab renders
// with (orbitCamera / lookAtCamera, fovY 1.0 rad default) and REQUIRE the frame
// to have been built with aspect = vpW/vpH -- then the pixel focal length
// focalPx = (vpH/2) / tan(fovY/2) is shared by both axes (a similarity, so
// conics stay similar conics) and is read back from the camera's own projection
// matrix (proj.m[5] = 1/tan(fovY/2)), never re-derived from a fovY parameter
// that could drift from the caller's camera.
// ---------------------------------------------------------------------------

// EXACT perspective sphere silhouette. A sphere seen by a pinhole camera hides
// behind its tangent cone (apex = the eye, half-angle alpha, sin(alpha) = R/d
// with d = |eye -> center|); the silhouette on screen is that cone's
// intersection with the image plane -- an ELLIPSE whenever the whole silhouette
// lies in front of the camera.
//
// DERIVATION, in camera coordinates (x = right, y = up, z = depth along fwd;
// the image plane z = 1 carries u = x/z, v = y/z; pixels are
// px = vpW/2 + focalPx*u, py = vpH/2 - focalPx*v with the ONE shared pixel
// focal focalPx = (vpH/2)/tan(fovY/2) -- see the family note above):
//   * eye->center direction cosines (ax, ay, az); write sin(beta) =
//     sqrt(ax^2 + ay^2) for the axis' tilt off the optical axis (cos(beta) = az).
//   * A point p lies on the cone iff (p . axis)^2 = |p|^2 cos^2(alpha). Rotate
//     the image plane so the axis falls in the xi-z plane (xi = the RADIAL
//     direction from the principal point toward the projected center -- the
//     symmetry plane of the cone-image geometry, hence the ellipse's major
//     axis), substitute p = (xi, eta, 1) and expand:
//       xi^2 (sin^2 b - cos^2 a) + 2 xi sin b cos b + (cos^2 b - cos^2 a)
//         = eta^2 cos^2 a.
//   * Complete the square with A = cos^2(alpha) - sin^2(beta):
//       A (xi - xi0)^2 + eta^2 cos^2(alpha) = sin^2(alpha) cos^2(alpha) / A,
//     where xi0 = sin(beta)cos(beta)/A (the constant term collapses via
//     (cos^2 b - cos^2 a) A + sin^2 b cos^2 b = cos^2 a sin^2 a).
//   * Ergo an ellipse centred xi0 along the radial direction (NOT at the
//     projected center tan(beta) -- the classic perspective offset), with
//       semi-major a = sin(alpha)cos(alpha) / A       (radial)
//       semi-minor b = sin(alpha) / sqrt(A)           (perpendicular)
//     a >= b because A <= cos^2(alpha); on axis (beta = 0) both collapse to
//     the circle tan(alpha). Cross-checks: the radial extremes work out to
//     (tan(b+a) +- tan(b-a))/2 via cos(b+a)cos(b-a) = A, matching the two
//     tangent RAYS at angles beta +- alpha.
// VALIDITY: requires d > R (eye outside the sphere) and A > 0 (A <= 0 means the
// silhouette meets the camera plane: a parabola/hyperbola, not representable as
// a Rive Ellipse). Outside that regime a DEGENERATE zero ellipse returns --
// finite, never NaN (the orbit bake keeps the camera outside the scene, so the
// Cornell family never hits it).
inline ScreenEllipse projectSpherePersp(const CameraFrame& c, Vec3f center, float r,
                                        float vpW, float vpH)
{
    ScreenEllipse e;
    const Vec3f dv = center - c.eye;
    const float d2 = dot(dv, dv);
    const float d = std::sqrt(d2);
    if (!(d > r) || d < 1e-9f)
        return e;                              // eye inside/on the sphere: degenerate
    // Direction cosines of the cone axis in camera coordinates.
    const float ax = dot(dv, c.right) / d;
    const float ay = dot(dv, c.up) / d;
    const float az = dot(dv, c.fwd) / d;
    const float sinA2 = (r * r) / d2;          // sin^2(alpha)
    const float cosA2 = 1.0f - sinA2;
    const float sinB2 = ax * ax + ay * ay;     // sin^2(beta)
    const float A = cosA2 - sinB2;
    if (!(A > 1e-9f) || az <= 0.0f)
        return e;                              // grazing/behind the camera plane: degenerate
    const float focalPx = 0.5f * vpH * c.proj.m[5]; // m[5] = 1/tan(fovY/2)
    // Centre: xi0 * radial = (sin b cos b / A) * (ax, -ay)/sin b -- the sin b
    // cancels, so the on-axis limit is regular.
    e.cx = 0.5f * vpW + focalPx * ax * az / A;
    e.cy = 0.5f * vpH - focalPx * ay * az / A;
    const float sinA = std::sqrt(sinA2), cosA = std::sqrt(cosA2);
    e.a = focalPx * sinA * cosA / A;
    e.b = focalPx * sinA / std::sqrt(A);
    // Major axis = the radial direction from the principal point, in pixel
    // space (y flips): (ax, -ay). A circle (on-axis) has no direction -- rot 0.
    e.rot = sinB2 > 1e-12f ? std::atan2(-ay, ax) : 0.0f;
    return e;
}

// Perspective disk -> MOMENT-FIT ellipse. The image of a 3D circle under
// perspective is exactly a conic, but unlike the sphere there is no pleasant
// closed form for its axes in this parameterization -- so FIT: project K = 16
// rim samples through the one true projection (projectPoint), then take their
// centroid + 2x2 covariance and eigen-decompose. Axes scale as sqrt(2*lambda):
// for an affine (ortho) image the uniform-theta samples of (a cos t, b sin t)
// have second moments exactly (a^2/2, b^2/2) (discrete sums of cos^2 over
// K > 2 equal K/2), so the fit is EXACT in the affine limit and matches the
// samples' RMS radius by construction under perspective (E[rho^2] =
// lambda1 + lambda2). The unit test pins RMS < 0.75 px against 64 dense rim
// projections at a typical lab pose. Degenerate inputs stay FINITE: eigenvalues
// clamp to >= 0 (edge-on disks collapse to b ~ 0), a rim sample behind the
// camera returns the degenerate zero ellipse.
inline ScreenEllipse projectDiskPersp(const CameraFrame& c, Vec3f center, float r,
                                      Vec3f normal, float vpW, float vpH)
{
    ScreenEllipse e;
    const Vec3f n = normalize(normal);
    Vec3f t1 = cross(n, std::fabs(n.y) < 0.99f ? Vec3f{0, 1, 0} : Vec3f{1, 0, 0});
    const float t1len = length(t1);
    if (t1len < 1e-9f) return e;               // zero normal: degenerate
    t1 = t1 * (1.0f / t1len);
    const Vec3f t2 = cross(n, t1);
    constexpr int K = 16;
    float xs[K], ys[K];
    for (int k = 0; k < K; ++k) {
        const float th = 6.28318530718f * (float)k / (float)K;
        const Vec3f p = center + (t1 * std::cos(th) + t2 * std::sin(th)) * r;
        const Projected pr = projectPoint(c.viewProj, p, vpW, vpH);
        if (!pr.inFront) return ScreenEllipse{}; // rim crosses the camera plane
        xs[k] = pr.x; ys[k] = pr.y;
    }
    double mx = 0.0, my = 0.0;
    for (int k = 0; k < K; ++k) { mx += xs[k]; my += ys[k]; }
    mx /= K; my /= K;
    double cxx = 0.0, cxy = 0.0, cyy = 0.0;
    for (int k = 0; k < K; ++k) {
        const double dx = xs[k] - mx, dy = ys[k] - my;
        cxx += dx * dx; cxy += dx * dy; cyy += dy * dy;
    }
    cxx /= K; cxy /= K; cyy /= K;
    // 2x2 symmetric eigen-decomposition: lambda = tr/2 +- sqrt((hd)^2 + cxy^2).
    const double hd = 0.5 * (cxx - cyy);
    const double disc = std::sqrt(hd * hd + cxy * cxy);
    const double l1 = 0.5 * (cxx + cyy) + disc; // major
    const double l2 = 0.5 * (cxx + cyy) - disc; // minor (>= 0 up to rounding)
    e.cx = (float)mx;
    e.cy = (float)my;
    e.a = (float)std::sqrt(2.0 * (l1 > 0.0 ? l1 : 0.0));
    e.b = (float)std::sqrt(2.0 * (l2 > 0.0 ? l2 : 0.0));
    // Major-axis angle = the lambda1 eigenvector; a circle (disc ~ 0) has no
    // direction (atan2(0,0) is ill-defined) -- rot 0.
    e.rot = disc > 1e-12 ? 0.5f * (float)std::atan2(2.0 * cxy, cxx - cyy) : 0.0f;
    return e;
}

// Plane quad under perspective: the 4 corners through projectPoint (a plane's
// image is exactly the polygon of its projected corners). A behind-camera
// corner projects to projectPoint's zero point -- the orbit camera aims at the
// scene centre from outside, so the bake never hits it.
inline std::vector<std::pair<float,float>> projectPlaneQuadPersp(const CameraFrame& c,
                                                                 const Vec3f corners[4],
                                                                 float vpW, float vpH)
{
    std::vector<std::pair<float,float>> out; out.reserve(4);
    for (int i = 0; i < 4; ++i) {
        const Projected p = projectPoint(c.viewProj, corners[i], vpW, vpH);
        out.emplace_back(p.x, p.y);
    }
    return out;
}

// Cone silhouette triangle under perspective: apex through projectPoint + the
// base disk's moment-fit ellipse extremes (the same construction the ortho
// projectConeTriangle uses).
inline std::vector<std::pair<float,float>> projectConeTrianglePersp(const CameraFrame& c,
                                                                    Vec3f apex, Vec3f baseCenter,
                                                                    float baseR, Vec3f axis,
                                                                    float vpW, float vpH)
{
    const ScreenEllipse be = projectDiskPersp(c, baseCenter, baseR, axis, vpW, vpH);
    const float ca = std::cos(be.rot), sa = std::sin(be.rot);
    const Projected ap = projectPoint(c.viewProj, apex, vpW, vpH);
    std::vector<std::pair<float,float>> out;
    out.emplace_back(ap.x, ap.y);
    out.emplace_back(be.cx + be.a * ca, be.cy + be.a * sa);
    out.emplace_back(be.cx - be.a * ca, be.cy - be.a * sa);
    return out;
}
} // namespace rive_backend
