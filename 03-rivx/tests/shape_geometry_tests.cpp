// backend/tests/shape_geometry_tests.cpp
// Unit tests for the GL-free shape geometry core. 
#include "check.hpp"
#include "render/shape_geometry.hpp"

#include <algorithm>

using namespace rive_backend;

RIVE_TEST(harness_sanity) { RIVE_CHECK(1 + 1 == 2); }

// --- flattening ----------------------------------------------------------
RIVE_TEST(flatten_line_keeps_endpoints)
{
    rive::RawPath p;
    p.moveTo(0, 0);
    p.lineTo(100, 0);
    p.close();
    auto c = flattenPath(p, rive::Mat2D(), 8.0f);
    RIVE_CHECK(c.size() == 1);
    RIVE_CHECK(c[0].closed);
    RIVE_CHECK(c[0].points.size() >= 2);
    RIVE_CHECK_NEAR(c[0].points.front().x, 0.0f, 0.01);
    RIVE_CHECK_NEAR(c[0].points.back().x, 100.0f, 0.01);
}

RIVE_TEST(flatten_cubic_scales_with_size)
{
    rive::RawPath small, big;
    small.moveTo(0, 0); small.cubicTo(0, 10, 10, 10, 10, 0);
    big.moveTo(0, 0);   big.cubicTo(0, 1000, 1000, 1000, 1000, 0);
    auto cs = flattenPath(small, rive::Mat2D(), 8.0f);
    auto cb = flattenPath(big, rive::Mat2D(), 8.0f);
    RIVE_CHECK(cb[0].points.size() > cs[0].points.size());
}

RIVE_TEST(bounds_of_square)
{
    rive::RawPath p; p.moveTo(10, 20); p.lineTo(40, 20);
    p.lineTo(40, 60); p.lineTo(10, 60); p.close();
    auto c = flattenPath(p, rive::Mat2D(), 8.0f);
    Pt mn, mx;
    RIVE_CHECK(contoursBounds(c, mn, mx));
    RIVE_CHECK_NEAR(mn.x, 10, 0.01); RIVE_CHECK_NEAR(mn.y, 20, 0.01);
    RIVE_CHECK_NEAR(mx.x, 40, 0.01); RIVE_CHECK_NEAR(mx.y, 60, 0.01);
}

// --- fill-rule classification --------------------------------------------
static Contour rect(float x0, float y0, float x1, float y1, bool ccw)
{
    Contour c; c.closed = true;
    if (ccw) c.points = {{x0,y0},{x1,y0},{x1,y1},{x0,y1}};
    else     c.points = {{x0,y0},{x0,y1},{x1,y1},{x1,y0}};
    return c;
}

RIVE_TEST(donut_hole_both_rules)
{
    // Outer CCW, inner CW (opposite winding) => hole for nonZero AND evenOdd.
    std::vector<Contour> d = {rect(0,0,100,100,true), rect(30,30,70,70,false)};
    using rive::FillRule;
    RIVE_CHECK(classifyFill(d, FillRule::nonZero, {10,10}));   // ring
    RIVE_CHECK(!classifyFill(d, FillRule::nonZero, {50,50}));  // hole
    RIVE_CHECK(classifyFill(d, FillRule::evenOdd, {10,10}));
    RIVE_CHECK(!classifyFill(d, FillRule::evenOdd, {50,50}));
}

RIVE_TEST(overlap_differs_by_rule)
{
    // Two same-wound overlapping squares; center is covered twice.
    std::vector<Contour> s = {rect(0,0,60,60,true), rect(40,0,100,60,true)};
    using rive::FillRule;
    // Overlap x in [40,60]: nonZero winding 2 => inside; evenOdd even => outside.
    RIVE_CHECK(classifyFill(s, FillRule::nonZero, {50,30}));
    RIVE_CHECK(!classifyFill(s, FillRule::evenOdd, {50,30}));
    // Non-overlap region inside one square: inside under both.
    RIVE_CHECK(classifyFill(s, FillRule::nonZero, {10,30}));
    RIVE_CHECK(classifyFill(s, FillRule::evenOdd, {10,30}));
}

// --- gradient sampling + feather sigma -----------------------------------
RIVE_TEST(linear_gradient_midpoint)
{
    std::vector<GradientStop> stops = {{0.f,0,0,0,1}, {1.f,1,1,1,1}}; // black->white
    float c[4];
    sampleGradient(false, {0,0}, {100,0}, 0.f, stops, {50,0}, c);
    RIVE_CHECK_NEAR(c[0], 0.5f, 0.02); RIVE_CHECK_NEAR(c[3], 1.0f, 0.01);
}

RIVE_TEST(linear_gradient_clamps)
{
    std::vector<GradientStop> stops = {{0.f,0,0,0,1},{1.f,1,1,1,1}};
    float c[4];
    sampleGradient(false, {0,0}, {100,0}, 0.f, stops, {-20,0}, c);
    RIVE_CHECK_NEAR(c[0], 0.0f, 0.01); // before first stop -> first color
    sampleGradient(false, {0,0}, {100,0}, 0.f, stops, {200,0}, c);
    RIVE_CHECK_NEAR(c[0], 1.0f, 0.01); // after last -> last color
}

RIVE_TEST(radial_gradient_center_vs_edge)
{
    std::vector<GradientStop> stops = {{0.f,1,0,0,1},{1.f,0,0,1,1}};
    float c[4];
    sampleGradient(true, {50,50}, {0,0}, 50.f, stops, {50,50}, c);
    RIVE_CHECK_NEAR(c[0], 1.0f, 0.02); // center -> first stop (red)
    sampleGradient(true, {50,50}, {0,0}, 50.f, stops, {100,50}, c);
    RIVE_CHECK_NEAR(c[2], 1.0f, 0.02); // edge -> last stop (blue)
}

RIVE_TEST(feather_sigma_monotonic_zero)
{
    RIVE_CHECK_NEAR(featherSigma(0.f), 0.0f, 1e-6);
    RIVE_CHECK(featherSigma(20.f) > featherSigma(5.f));
}

// --- stroke tessellation -------------------------------------------------
RIVE_TEST(stroke_segment_width)
{
    Contour c; c.closed = false; c.points = {{0,50},{100,50}};
    auto tris = tessellateStroke(c, 10.f, rive::StrokeJoin::miter,
                                 rive::StrokeCap::butt);
    RIVE_CHECK(tris.size() >= 6); // >= 2 triangles
    float minY = 1e9f, maxY = -1e9f;
    for (auto& p : tris) { minY = std::min(minY, p.y); maxY = std::max(maxY, p.y); }
    RIVE_CHECK_NEAR(maxY - minY, 10.0f, 0.5); // full width
}

RIVE_TEST(stroke_corner_emits_join)
{
    Contour c; c.closed = false;
    c.points = {{0,0},{100,0},{100,100}};
    auto tris = tessellateStroke(c, 8.f, rive::StrokeJoin::round,
                                 rive::StrokeCap::butt);
    RIVE_CHECK(tris.size() >= 12); // two segments + join fan
}

int main() { return rive_test::run_all(); }
