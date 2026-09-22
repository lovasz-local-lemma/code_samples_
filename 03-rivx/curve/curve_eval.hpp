// backend/src/curve_eval.hpp
// Curve evaluation core for the experimental .rivx vector format. GL-free,
// deterministic, no exceptions in hot paths: turns a list of control points
// into a sampled polyline for the interactive editor (drag handles) and the
// polygon-fill renderer. Unit-tested (rive_curve_tests).
#pragma once
#include <vector>

namespace rive_backend
{
struct CurvePt { float x, y; };

// Which curve family the control points describe.
//  Bezier:           a SINGLE Bezier segment of degree pts.size()-1 evaluated by
//                    de Casteljau (numerically stable for any degree). closed =
//                    pts[0] is appended as one extra control point so the curve
//                    returns to its start; that closure is C0 ONLY (the seam
//                    tangents do not match in general).
//  BSpline:          uniform B-spline of 'degree' (clamped, see sampleCurve).
//                    open  -> CLAMPED end knots, so the curve interpolates the
//                             first and last control points; C^{degree-1} inside.
//                    closed-> periodic control-point wrapping; a true loop that
//                             is C^{degree-1} everywhere INCLUDING the seam.
//  Yuksel*:          interpolating splines THROUGH the control points after
//                    Cem Yuksel, "A Class of C2 Interpolating Splines",
//                    ACM TOG 39(5), 2020: consecutive segments blend two local
//                    3-point interpolants F_i with cos^2/sin^2 trigonometric
//                    weights (theta = pi/2 * local t).
//    YukselCircular:   F_i = circumcircle through (p_{i-1}, p_i, p_{i+1}) with a
//                      quadratic-in-s angle parameterization; collinear or
//                      duplicate triplets fall back to the quadratic Lagrange
//                      interpolant (whose image is the straight line) -- never
//                      NaN. C2 at joins.
//    YukselElliptical: affine-mapped circle (conjugate-diameter ellipse): centre
//                      m = (p_{i-1}+p_{i+1})/2, semi-diameters e1 = (p_{i+1}-
//                      p_{i-1})/2 and e2 = p_i - m, F_i(s) = m + e1*sin(phi) +
//                      e2*cos(phi), phi = pi*(s-1/2). The tangent at p_i is
//                      parallel to the chord p_{i-1}p_{i+1} (the paper's
//                      "middle point is an extremum" property). Defined for
//                      EVERY input, including collinear/duplicate. C2 at joins.
//    YukselHybrid:     per-triplet constant mix w*elliptical + (1-w)*circular
//                      (circular falling back to parabolic when degenerate);
//                      w = clamp(1 - |proj of (p_i - m) onto the chord| /
//                      |half-chord|, 0, 1): symmetric apex -> pure ellipse,
//                      apex projecting at/past a chord end -> pure circle. C2.
//    YukselParabolic:  F_i = quadratic Lagrange (parabola) through the triplet;
//                      always well-defined. Guaranteed C1 at joins (and with
//                      this module's uniform parameterization it is in fact the
//                      paper's quadratic-Bezier base, which is C2 -- we only
//                      ADVERTISE C1 for this kind; see curve_eval.cpp notes).
enum class CurveKind
{
    Bezier,
    BSpline,
    YukselCircular,
    YukselElliptical,
    YukselHybrid,
    YukselParabolic,
    // CubicPen: the OFFICIAL Rive model -- a chain of cubic Bezier segments,
    // each vertex carrying an in- and out-handle (see CubicHandle, parallel to
    // pts). Segment i uses controls pts[i]+out[i] and pts[i+1]+in[i+1]. This is
    // the only kind that exports to NATIVE .riv as true smooth cubics
    // (CubicMirrored/Asymmetric/Detached vertices).
    CubicPen,
};

// Per-vertex tangent handles for CurveKind::CubicPen. inDx/inDy and outDx/outDy
// are the handle CONTROL-POINT OFFSETS from the vertex (canvas px). type
// constrains how the two relate when one is dragged (the editor enforces it;
// evaluation/export just read the stored offsets):
//   Mirrored   -- colinear + equal length (smooth, symmetric)
//   Asymmetric -- colinear, independent lengths (smooth direction)
//   Detached   -- fully independent (a corner)
// Maps 1:1 onto Rive's CubicMirroredVertex / CubicAsymmetricVertex /
// CubicDetachedVertex.
enum class HandleType { Mirrored, Asymmetric, Detached };
struct CubicHandle
{
    float inDx = 0.0f, inDy = 0.0f;   // in-handle offset from the vertex
    float outDx = 0.0f, outDy = 0.0f; // out-handle offset from the vertex
    HandleType type = HandleType::Mirrored;
};

struct CurveSpec
{
    CurveKind kind = CurveKind::YukselHybrid;
    int degree = 3;   // BSpline only; clamped to [1, min(7, npts-1)] open,
                      // [1, 7] closed (periodic wrap supplies the points).
    bool closed = true;
    std::vector<CurvePt> pts;
    // CubicPen only: per-vertex handles, parallel to pts (resized to match on
    // demand; missing entries default to zero handles = sharp corners).
    std::vector<CubicHandle> handles;
};

// How many parameter spans sampleCurve subdivides 'spec' into. n = pts.size():
//   Bezier : open n-1, closed n            (0 when n < 2)
//   BSpline: open n-degreeClamped, closed n (0 when n < 2)
//   Yuksel*: open n-1, closed n            (0 when n < 3)
// 0 means "degenerate": sampleCurve returns the control points verbatim.
int curveSpanCount(const CurveSpec& spec);

// Sample the curve as a polyline. samplesPerSpan is clamped to >= 2. With
// S = curveSpanCount(spec) and m = max(2, samplesPerSpan):
//   open   -> S*(m-1) + 1 points, first/last exactly on the curve endpoints;
//   closed -> S*(m-1) points; the last point connects back to the first
//             (the wrap point is NOT duplicated);
//   S == 0 -> the control points themselves (polyline fallback; for a closed
//             spec the renderer's implicit last->first edge closes it).
// Deterministic, allocation-light (one result vector + O(degree) scratch),
// cheap enough to run every frame for editor-sized inputs (< 100 points).
std::vector<CurvePt> sampleCurve(const CurveSpec& spec, int samplesPerSpan);

// ---- Visualization / introspection (Yuksel family only) --------------------

// Effective base-function weights at one sampled point: how much of the final
// position came from the elliptical, circumcircle, and parabolic (quadratic
// Bezier) bases. Sums to 1 for Yuksel-kind curves. A blended sample mixes TWO
// local interpolants (cos^2/sin^2), each of which is itself a constant
// ellipse/circle mix for Hybrid (circle falling back to parabola when the
// triplet is collinear) -- these are the fully-unfolded weights.
struct CurveBaseWeights
{
    float ellipse = 0.0f;
    float circle = 0.0f;
    float parabola = 0.0f;
};

// Per-sample base weights, PARALLEL to sampleCurve(spec, samplesPerSpan)
// (same count, same order). Non-Yuksel kinds return an empty vector.
std::vector<CurveBaseWeights> sampleCurveBaseWeights(const CurveSpec& spec,
                                                     int samplesPerSpan);

// The constant base mix of ONE local interpolant F_i (Hybrid's per-triplet
// ellipse weight; degenerate-circle fallback already applied).
struct LocalInterpolantInfo
{
    float ellipse = 0.0f;
    float circle = 0.0f;
    float parabola = 0.0f;
};

// Sample the local interpolant F_i (fitted to the triplet p_{i-1}, p_i,
// p_{i+1}) over its full parameter range s in [0,1] with m >= 2 points --
// the curve's intermediate construction pieces. Valid i: closed -> [0, n-1];
// open -> [1, n-2]. Returns empty for non-Yuksel kinds, invalid i, or n < 3.
// info (optional) receives F_i's constant base mix.
std::vector<CurvePt> sampleLocalInterpolant(const CurveSpec& spec, int i, int m,
                                            LocalInterpolantInfo* info = nullptr);
} // namespace rive_backend
