// backend/src/shape_geometry.hpp
// GL-free geometry/math core for the custom shape renderer: flattening,
// fill-rule classification, stroke tessellation, gradient sampling, feather
// sigma. Kept free of GL so it is unit-testable.
#pragma once

#include "rive/math/raw_path.hpp"
#include "rive/math/mat2d.hpp"
#include "rive/math/vec2d.hpp"
#include "rive/math/path_types.hpp"          // FillRule, PathVerb
#include "rive/shapes/paint/stroke_cap.hpp"  // StrokeCap
#include "rive/shapes/paint/stroke_join.hpp" // StrokeJoin

#include <cstdint>
#include <vector>

namespace rive_backend
{
struct Pt { float x = 0.0f; float y = 0.0f; };

struct Contour
{
    std::vector<Pt> points; // device/pixel space after transform
    bool closed = false;
};

struct GradientStop { float offset; float r, g, b, a; }; // straight alpha 0..1

// Flatten a RawPath to device-space polyline contours. maxSegmentPx caps the
// chord length so large on-screen curves stay smooth (scale-aware).
std::vector<Contour> flattenPath(const rive::RawPath& path,
                                 const rive::Mat2D& transform,
                                 float maxSegmentPx);

// Axis-aligned bounds of flattened contours; returns false if empty.
bool contoursBounds(const std::vector<Contour>& contours,
                    Pt& outMin, Pt& outMax);

// True if `p` is inside the contour set under `rule`. Mirrors the stencil
// semantics: nonZero uses winding != 0, evenOdd uses odd crossing count.
bool classifyFill(const std::vector<Contour>& contours,
                  rive::FillRule rule,
                  Pt p);

// Sample a linear (radius==0) or radial gradient at device-space point p.
// Returns straight-alpha RGBA in 0..1. start/end/radius are device-space.
void sampleGradient(bool radial,
                    Pt start, Pt end, float radius,
                    const std::vector<GradientStop>& stops,
                    Pt p, float outRgba[4]);

// Map a Rive feather value (already scaled to device px) to a Gaussian sigma.
// Monotonic; featherSigma(0)==0. Constant tuned against the official renderer.
float featherSigma(float featherDevicePx);

// Tessellate a stroked contour into device-space triangles (CCW). Emits
// pairs of triangles per segment plus join/cap fills. thicknessPx is the full
// stroke width.
std::vector<Pt> tessellateStroke(const Contour& contour,
                                 float thicknessPx,
                                 rive::StrokeJoin join,
                                 rive::StrokeCap cap);
} // namespace rive_backend
