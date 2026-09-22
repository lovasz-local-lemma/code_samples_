// Half-edge / DCEL topology layer on top of the Apollonius diagram.
//
// Lifts the flat (vertex / edge / polyline) Apollonius output into a
// half-edge data structure with face / cell identification. "Open"
// segments (Apollonius edges that run out to the diagram bounds without
// hitting a triple-tangent vertex) would leave cells touching the bbox
// unable to form closed boundary cycles, so the topology is closed with
// a bbox ring: each Apollonius edge whose polyline reaches the bbox
// (0 or 1 anchor) terminates at a "boundary vertex" on the bbox
// perimeter. The four bbox corners are added as additional vertices,
// and the boundary + corner vertices are sorted around the perimeter
// and connected by straight half-edge pairs along each bbox edge. The
// result is a single outside-face cycle walking the perimeter CW
// (outside on the LEFT of each half-edge in that cycle) and closed CCW
// boundary cycles for every cell -- including cells touching the bbox.
//
// Header-only by design so both the demo (src/app/app.cpp) and the
// standalone tests (tests/fortune_core_tests.cpp) can include it
// without dragging in the full Apollonius solver chain. The builder
// takes plain "translator" inputs (ApolloniusVertexInput /
// ApolloniusEdgeInput) so callers can either feed it a real
// ApolloniusDiagram via a small adapter or hand-roll synthetic inputs.

#pragma once

#include "fortune/common.h"
#include "fortune/pair_curve.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fortune {

// ---------------------------------------------------------------------------
// Translator inputs (mirror app.cpp's ApolloniusVertex / ApolloniusEdge
// without forcing the test exe to link the full Apollonius solver).
// ---------------------------------------------------------------------------

struct ApolloniusVertexInput {
    Vec2 point;
    std::array<int, 3> sites{-1, -1, -1};
};

struct ApolloniusEdgeInput {
    int left_site = -1;
    int right_site = -1;
    std::vector<Vec2> polyline;
};

// ---------------------------------------------------------------------------
// DCEL records.
// ---------------------------------------------------------------------------

struct DCELVertex {
    Vec2 point;
    std::array<int, 3> sites{-1, -1, -1};
    int incident_halfedge = -1;
    // True for vertices created on the bbox perimeter (corners and
    // "boundary" endpoints where an unbounded Apollonius edge meets
    // the bbox). The sites array holds {a, b, -1} for an edge-endpoint
    // boundary vertex (a/b = sites flanking the unbounded edge) and
    // {-1, -1, -1} for the four bbox corner vertices.
    bool is_boundary = false;
};

struct DCELHalfEdge {
    int origin = -1;
    int twin = -1;
    int next = -1;
    int prev = -1;
    int face = -1;
    int left_site = -1;
    int right_site = -1;
    std::vector<Vec2> polyline;  // oriented from origin to next->origin
};

struct DCELFace {
    int site = -1;
    int boundary_halfedge = -1;
    bool is_outside = false;
};

struct ApolloniusDCEL {
    std::vector<DCELVertex> vertices;
    std::vector<DCELHalfEdge> halfedges;
    std::vector<DCELFace> faces;
    int outside_face = -1;

    int unmatched_polyline_endpoints = 0;
    int orphan_halfedges = 0;
    // Number of new vertices created at the bbox perimeter
    // (boundary-edge endpoints + the four corners). Informational.
    int boundary_vertices_added = 0;
    // Number of outside-face cycles found by face identification
    // that deviate from the expected "exactly one" invariant. > 0 means
    // the bbox-ring stitch did not produce a single outside cycle (and
    // build_note carries a diagnostic).
    int outside_face_anomalies = 0;
    // Number of degenerate "lens" tips repaired by
    // repair_degenerate_lens (a deg-4 {p,q,r} triple point whose partner
    // tip was left degree-2 / missing its p|r branch). 0 on well-formed
    // diagrams. Informational; lets the UI show the toggle's effect.
    int degenerate_lenses_repaired = 0;
    std::string build_note;
};

// ---------------------------------------------------------------------------
// Builder.
// ---------------------------------------------------------------------------

namespace detail {

inline double dcel_additive_metric(const Vec2& point, const Vec2& site, const double radius) {
    return length(point - site) - radius;
}

// Cumulative arclength of a polyline.
inline std::vector<double> dcel_arclengths(const std::vector<Vec2>& poly) {
    std::vector<double> s(poly.size(), 0.0);
    for (std::size_t i = 1; i < poly.size(); ++i) {
        s[i] = s[i - 1] + length(poly[i] - poly[i - 1]);
    }
    return s;
}

// Arclength parameter on `poly` of the point nearest to `target`. Returns
// the cumulative arclength s in [0, total]. We pick the polyline segment
// whose closest-point distance to `target` is minimal, then take that
// segment's projected parameter.
inline double dcel_param_at_point(const std::vector<Vec2>& poly,
                                  const std::vector<double>& cum_s,
                                  const Vec2& target) {
    if (poly.size() < 2) {
        return 0.0;
    }
    double best_s = 0.0;
    double best_dist_sq = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i + 1 < poly.size(); ++i) {
        const Vec2 a = poly[i];
        const Vec2 b = poly[i + 1];
        const Vec2 ab = b - a;
        const double denom = length_sq(ab);
        double t = 0.0;
        Vec2 proj = a;
        if (denom > kEpsilon) {
            t = clamp(dot(target - a, ab) / denom, 0.0, 1.0);
            proj = a + ab * t;
        }
        const double d2 = length_sq(target - proj);
        if (d2 < best_dist_sq) {
            best_dist_sq = d2;
            const double seg_len = length(ab);
            best_s = cum_s[i] + t * seg_len;
        }
    }
    return best_s;
}

// Slice a polyline into a sub-polyline whose arclength runs s_lo .. s_hi
// (with s_lo < s_hi). Returns a polyline that starts on the curve at
// s_lo and ends at s_hi, including any intermediate sample points.
inline std::vector<Vec2> dcel_subpolyline(const std::vector<Vec2>& poly,
                                          const std::vector<double>& cum_s,
                                          const double s_lo,
                                          const double s_hi) {
    std::vector<Vec2> out;
    if (poly.size() < 2 || s_hi <= s_lo) {
        return out;
    }
    // Find start sample on the segment containing s_lo.
    auto sample_at = [&](double s) -> Vec2 {
        if (poly.size() < 2) return poly.empty() ? Vec2{} : poly.front();
        if (s <= cum_s.front()) return poly.front();
        if (s >= cum_s.back()) return poly.back();
        for (std::size_t i = 0; i + 1 < poly.size(); ++i) {
            if (s <= cum_s[i + 1]) {
                const double seg = cum_s[i + 1] - cum_s[i];
                if (seg <= kEpsilon) return poly[i];
                const double t = (s - cum_s[i]) / seg;
                return poly[i] + (poly[i + 1] - poly[i]) * t;
            }
        }
        return poly.back();
    };
    out.push_back(sample_at(s_lo));
    for (std::size_t i = 0; i < poly.size(); ++i) {
        if (cum_s[i] > s_lo + 1.0e-12 && cum_s[i] < s_hi - 1.0e-12) {
            if (length_sq(out.back() - poly[i]) > 1.0e-20) {
                out.push_back(poly[i]);
            }
        }
    }
    const Vec2 end_pt = sample_at(s_hi);
    if (length_sq(out.back() - end_pt) > 1.0e-20) {
        out.push_back(end_pt);
    }
    return out;
}

// Decide which of {a_site, b_site} bounds the LEFT side of the
// directed polyline. We sample a tiny CCW-normal offset from the
// origin of the polyline and pick whichever site has the smaller
// additive metric there. If the metrics tie even after doubling the
// offset a few times, fall back to (a_site is left). Symmetry of the
// twin pair makes this fallback harmless.
inline std::pair<int, int> dcel_left_right_assignment(const std::vector<Vec2>& poly,
                                                      const int a_site,
                                                      const int b_site,
                                                      const std::vector<Vec2>& sites,
                                                      const std::vector<double>& radii) {
    if (poly.size() < 2) {
        return {a_site, b_site};
    }
    const Vec2 origin = poly.front();
    Vec2 tangent = poly[1] - origin;
    const double tan_len = length(tangent);
    if (tan_len <= kEpsilon) {
        // Try a later sample.
        for (std::size_t i = 2; i < poly.size(); ++i) {
            tangent = poly[i] - origin;
            if (length(tangent) > kEpsilon) break;
        }
    }
    tangent = normalize(tangent);
    if (length(tangent) <= kEpsilon) {
        return {a_site, b_site};
    }
    const Vec2 ccw_normal = perp(tangent);

    double eps = 1.0e-4;
    for (int attempt = 0; attempt < 5; ++attempt) {
        const Vec2 probe = origin + ccw_normal * eps;
        if (a_site < 0 || b_site < 0 ||
            a_site >= static_cast<int>(sites.size()) ||
            b_site >= static_cast<int>(sites.size())) {
            break;
        }
        const double ma = dcel_additive_metric(probe, sites[a_site], radii[a_site]);
        const double mb = dcel_additive_metric(probe, sites[b_site], radii[b_site]);
        if (std::abs(ma - mb) > 1.0e-9) {
            return ma < mb ? std::pair<int, int>{a_site, b_site}
                           : std::pair<int, int>{b_site, a_site};
        }
        eps *= 4.0;
    }
    return {a_site, b_site};
}

inline double dcel_outgoing_angle(const DCELHalfEdge& h) {
    if (h.polyline.size() < 2) return 0.0;
    const Vec2 d = h.polyline[1] - h.polyline[0];
    return std::atan2(d.y, d.x);
}

// Project a (potentially slightly-inside or slightly-outside)
// polyline endpoint onto the bbox by clamping both coordinates, then
// snapping to whichever bbox edge is closest. Returns the snapped point.
inline Vec2 dcel_snap_to_bbox(const Vec2& p, const Rect& b) {
    Vec2 q;
    q.x = clamp(p.x, b.min_x, b.max_x);
    q.y = clamp(p.y, b.min_y, b.max_y);
    // If the clamped point is interior, push it to the nearest edge so
    // it actually sits on the perimeter (so the perimeter parameter is
    // well defined). This only fires for points the tracer left a hair
    // inside the bbox -- the projection distance is small.
    const double dl = q.x - b.min_x;
    const double dr = b.max_x - q.x;
    const double db = q.y - b.min_y;
    const double dt = b.max_y - q.y;
    const double m = std::min(std::min(dl, dr), std::min(db, dt));
    if (m > 0.0) {
        if (m == dl) q.x = b.min_x;
        else if (m == dr) q.x = b.max_x;
        else if (m == db) q.y = b.min_y;
        else q.y = b.max_y;
    }
    return q;
}

// Extend a polyline's free end to the bbox boundary by shooting a
// ray from `p_inner` through `p_end` (the current last point) and finding
// the first intersection with the bbox perimeter. Returns the intersection
// point on the bbox. If the direction is degenerate or the ray does not exit
// the bbox in the forward direction, falls back to dcel_snap_to_bbox.
inline Vec2 dcel_extend_end_to_bbox(const Vec2& p_inner, const Vec2& p_end, const Rect& b) {
    const Vec2 dir = p_end - p_inner;
    const double dlen = length(dir);
    if (dlen <= kEpsilon) {
        return dcel_snap_to_bbox(p_end, b);
    }
    // Parametric ray: P(t) = p_end + dir * t, find smallest t > 0 that
    // hits a bbox edge.
    double t_best = std::numeric_limits<double>::infinity();
    if (dir.x > kEpsilon) {
        const double t = (b.max_x - p_end.x) / dir.x;
        if (t > -1.0e-9) t_best = std::min(t_best, std::max(0.0, t));
    } else if (dir.x < -kEpsilon) {
        const double t = (b.min_x - p_end.x) / dir.x;
        if (t > -1.0e-9) t_best = std::min(t_best, std::max(0.0, t));
    }
    if (dir.y > kEpsilon) {
        const double t = (b.max_y - p_end.y) / dir.y;
        if (t > -1.0e-9) t_best = std::min(t_best, std::max(0.0, t));
    } else if (dir.y < -kEpsilon) {
        const double t = (b.min_y - p_end.y) / dir.y;
        if (t > -1.0e-9) t_best = std::min(t_best, std::max(0.0, t));
    }
    if (std::isinf(t_best)) {
        return dcel_snap_to_bbox(p_end, b);
    }
    Vec2 hit = p_end + dir * t_best;
    // Clamp to bbox to fix any floating-point overshoot.
    hit.x = clamp(hit.x, b.min_x, b.max_x);
    hit.y = clamp(hit.y, b.min_y, b.max_y);
    return dcel_snap_to_bbox(hit, b);
}

// Perimeter parameter s in [0, 4) for a point already on the
// bbox edge. Bottom edge (BL -> BR): s in [0, 1). Right edge (BR -> TR):
// s in [1, 2). Top edge (TR -> TL): s in [2, 3). Left edge (TL -> BL):
// s in [3, 4). For points exactly at a corner the param resolves to the
// "starting" value of the next edge in CCW order.
inline double dcel_perimeter_s(const Vec2& p, const Rect& b) {
    const double sx = b.max_x - b.min_x;
    const double sy = b.max_y - b.min_y;
    const double tol = 1.0e-9 * std::max({sx, sy, 1.0});
    // Order matters: we want corners to map to a consistent CCW value
    // (BL -> 0, BR -> 1, TR -> 2, TL -> 3). Test bottom edge first
    // (covers BL and BR), then right edge (TR), then top edge (TL),
    // then fall through to left edge.
    if (std::abs(p.y - b.min_y) <= tol) {  // bottom edge BL -> BR
        return sx > 0.0 ? (p.x - b.min_x) / sx : 0.0;
    }
    if (std::abs(p.x - b.max_x) <= tol) {  // right edge BR -> TR
        return 1.0 + (sy > 0.0 ? (p.y - b.min_y) / sy : 0.0);
    }
    if (std::abs(p.y - b.max_y) <= tol) {  // top edge TR -> TL
        return 2.0 + (sx > 0.0 ? (b.max_x - p.x) / sx : 0.0);
    }
    // left edge TL -> BL
    return 3.0 + (sy > 0.0 ? (b.max_y - p.y) / sy : 0.0);
}

// For the perimeter segment between v_lo and v_hi (consecutive
// CCW sorted), evaluate the bbox-edge midpoint and return the site that
// owns it (minimum additive_distance), or -1 if no site qualifies.
inline int dcel_inner_site_at_midpoint(const Vec2& v_lo,
                                       const Vec2& v_hi,
                                       const std::vector<Vec2>& sites,
                                       const std::vector<double>& radii) {
    if (sites.empty()) return -1;
    const Vec2 mid = (v_lo + v_hi) * 0.5;
    int best = -1;
    double best_metric = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < sites.size(); ++i) {
        const double r = i < radii.size() ? radii[i] : 0.0;
        const double m = length(mid - sites[i]) - r;
        if (m < best_metric) {
            best_metric = m;
            best = static_cast<int>(i);
        }
    }
    return best;
}

// Stitch the bbox perimeter ring (4 corners + boundary vertices) into the
// DCEL. Adds 4 corner vertices if not already present, collects all
// boundary vertices (is_boundary=true), sorts by perimeter parameter s,
// and stitches consecutive pairs with inner+outer halfedge twin pairs.
//
// Preconditions:
//   - dcel.vertices contains all interior vertices and (optionally) any
//     pre-existing boundary vertices added by edge-tail extension. The
//     four corner vertices may or may not be present.
//   - dcel.halfedges contains interior halfedges but no perimeter-ring
//     halfedges (or any pre-existing perimeter halfedges have already
//     been stripped).
//
// Updates:
//   - dcel.vertices: adds 4 corner vertices if missing (with
//     sites={-1,-1,-1}, is_boundary=true).
//   - dcel.halfedges: appends inner+outer halfedge pairs for each
//     consecutive boundary-vertex pair on the perimeter.
//   - dcel.boundary_vertices_added: updated to reflect added corners.
//   - DCELVertex::incident_halfedge: set for any vertex whose value was -1.
//
// inner_site is determined via dcel_inner_site_at_midpoint(p_lo, p_hi,
// sites, radii). The polyline for each ring halfedge is the 2-point
// straight segment between the two boundary vertices.
inline void build_perimeter_ring(ApolloniusDCEL& dcel,
                                 const std::vector<Vec2>& sites,
                                 const std::vector<double>& radii,
                                 const Rect& bounds) {
    // Add 4 corner vertices -- but only if they are not already present
    // (to avoid duplicate corners when called on a DCEL that already
    // carries corners from a previous ring build).
    // A corner vertex is identified by is_boundary==true and sites=={-1,-1,-1}.
    // We check point proximity (< 1e-10) against the four expected positions.
    auto ensure_corner = [&](const Vec2& p) -> int {
        // Check if any existing corner vertex is close to this position.
        constexpr double corner_tol_sq = 1.0e-20;
        for (int vi = 0; vi < static_cast<int>(dcel.vertices.size()); ++vi) {
            const auto& v = dcel.vertices[vi];
            if (v.is_boundary && v.sites[0] < 0 && v.sites[1] < 0 && v.sites[2] < 0) {
                if (length_sq(v.point - p) <= corner_tol_sq) {
                    return vi;  // already present
                }
            }
        }
        // Not found: add a new corner vertex.
        DCELVertex cv;
        cv.point = p;
        cv.sites = {-1, -1, -1};
        cv.incident_halfedge = -1;
        cv.is_boundary = true;
        const int idx = static_cast<int>(dcel.vertices.size());
        dcel.vertices.push_back(cv);
        dcel.boundary_vertices_added += 1;
        return idx;
    };
    (void)ensure_corner({bounds.min_x, bounds.min_y});  // BL
    (void)ensure_corner({bounds.max_x, bounds.min_y});  // BR
    (void)ensure_corner({bounds.max_x, bounds.max_y});  // TR
    (void)ensure_corner({bounds.min_x, bounds.max_y});  // TL

    // Collect all boundary + corner vertices with their perimeter
    // parameter and sort CCW.
    struct RingEntry {
        int vertex_index = -1;
        double s = 0.0;
    };
    std::vector<RingEntry> ring;
    ring.reserve(dcel.boundary_vertices_added);
    for (int vi = 0; vi < static_cast<int>(dcel.vertices.size()); ++vi) {
        if (!dcel.vertices[vi].is_boundary) continue;
        RingEntry e;
        e.vertex_index = vi;
        e.s = dcel_perimeter_s(dcel.vertices[vi].point, bounds);
        ring.push_back(e);
    }
    std::sort(ring.begin(), ring.end(),
              [](const RingEntry& l, const RingEntry& r) { return l.s < r.s; });

    // Stitch consecutive pairs along the perimeter with a half-edge
    // pair. The inner half-edge goes CCW (v_lo -> v_hi) with the
    // inner cell on the LEFT; the outer half-edge goes CW (v_hi ->
    // v_lo) with the outside face on the LEFT.
    const std::size_t n = ring.size();
    for (std::size_t i = 0; i < n; ++i) {
        const int v_lo = ring[i].vertex_index;
        const int v_hi = ring[(i + 1) % n].vertex_index;
        // Skip zero-length segments (a boundary vertex landing
        // exactly on a corner, etc.). Two distinct vertex indices
        // are required.
        if (v_lo == v_hi) continue;
        const Vec2& p_lo = dcel.vertices[v_lo].point;
        const Vec2& p_hi = dcel.vertices[v_hi].point;
        if (length_sq(p_hi - p_lo) <= 1.0e-20) continue;

        // Inner-site test: which cell owns the perimeter midpoint?
        const int inner_site = dcel_inner_site_at_midpoint(p_lo, p_hi, sites, radii);

        // Inner half-edge (CCW, on inner cell's CCW boundary).
        DCELHalfEdge inner_he;
        inner_he.origin = v_lo;
        inner_he.left_site = inner_site;
        inner_he.right_site = -1;
        inner_he.polyline = {p_lo, p_hi};

        // Outer half-edge (CW, on outside face's CCW boundary).
        DCELHalfEdge outer_he;
        outer_he.origin = v_hi;
        outer_he.left_site = -1;
        outer_he.right_site = inner_site;
        outer_he.polyline = {p_hi, p_lo};

        const int inner_idx = static_cast<int>(dcel.halfedges.size());
        const int outer_idx = inner_idx + 1;
        inner_he.twin = outer_idx;
        outer_he.twin = inner_idx;
        dcel.halfedges.push_back(std::move(inner_he));
        dcel.halfedges.push_back(std::move(outer_he));

        if (dcel.vertices[v_lo].incident_halfedge < 0) {
            dcel.vertices[v_lo].incident_halfedge = inner_idx;
        }
        if (dcel.vertices[v_hi].incident_halfedge < 0) {
            dcel.vertices[v_hi].incident_halfedge = outer_idx;
        }
    }
}

// Repair the degenerate-lens topology defect.
//
// When the additively-weighted diagram has a thin "lens" cell (site q
// squeezed between sites p and r), the two triple points of {p,q,r} that cap
// the lens should each be a clean degree-3 vertex: the two lens arcs (p|q and
// q|r) plus one branch of the p|r bisector leaving away from the lens. The
// analytic tracer sometimes routes BOTH p|r branches through one tip V (making
// it degree 4) and leaves the other tip W degree 2 (missing its p|r branch).
// That vertex is locally inconsistent -- cell p enters twice but leaves once --
// so the angular relink cannot close it and the perimeter splits into a
// spurious second "outside" cycle.
//
// This pass detects exactly that signature and re-origins one p|r branch from V
// to W (the branch whose far endpoint is nearer W), relocating its lens-side
// endpoint, so both tips become balanced degree-3 triple points and the relink
// closes. It fires ONLY on the imbalanced-lens signature and is inert on
// well-formed diagrams. Returns the number of lens tips repaired.
inline int repair_degenerate_lens(ApolloniusDCEL& dcel) {
    std::vector<std::vector<int>> out_he(dcel.vertices.size());
    for (std::size_t hi = 0; hi < dcel.halfedges.size(); ++hi) {
        const int v = dcel.halfedges[hi].origin;
        if (v >= 0 && v < static_cast<int>(out_he.size())) out_he[v].push_back(static_cast<int>(hi));
    }
    auto sites_equal = [](std::array<int, 3> a, std::array<int, 3> b) {
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());
        return a == b && a[0] >= 0;
    };
    auto target_of = [&](int he) {
        const int tw = dcel.halfedges[he].twin;
        return tw >= 0 ? dcel.halfedges[tw].origin : -1;
    };
    auto pair_of = [&](int he) {
        const int l = dcel.halfedges[he].left_site, r = dcel.halfedges[he].right_site;
        return std::pair<int, int>(std::min(l, r), std::max(l, r));
    };
    auto dist2 = [](const Vec2& a, const Vec2& b) {
        const double dx = a.x - b.x, dy = a.y - b.y;
        return dx * dx + dy * dy;
    };

    int repaired = 0;
    for (std::size_t vi = 0; vi < dcel.vertices.size(); ++vi) {
        const DCELVertex& V = dcel.vertices[vi];
        if (V.is_boundary || V.sites[0] < 0 || V.sites[1] < 0 || V.sites[2] < 0) continue;
        if (out_he[vi].size() != 4) continue;  // lens tip V is degree 4

        // Candidate partner W: a target reached by exactly two of V's edges.
        std::array<int, 4> tgt{};
        for (int k = 0; k < 4; ++k) tgt[k] = target_of(out_he[vi][k]);
        int W = -1;
        for (int k = 0; k < 4 && W < 0; ++k) {
            int c = 0;
            for (int j = 0; j < 4; ++j) c += (tgt[j] == tgt[k]);
            if (c == 2) W = tgt[k];
        }
        if (W < 0 || W >= static_cast<int>(dcel.vertices.size()) || dcel.vertices[W].is_boundary) continue;
        if (!sites_equal(V.sites, dcel.vertices[W].sites)) continue;
        if (out_he[W].size() != 2) continue;  // partner tip W is degree 2
        if (target_of(out_he[W][0]) != static_cast<int>(vi) || target_of(out_he[W][1]) != static_cast<int>(vi)) continue;

        // Split V's edges into the two lens arcs (to W) and the doubled p|r bisector.
        std::vector<int> to_w, others;
        for (int k = 0; k < 4; ++k) (tgt[k] == W ? to_w : others).push_back(out_he[vi][k]);
        if (to_w.size() != 2 || others.size() != 2) continue;
        if (pair_of(to_w[0]) == pair_of(to_w[1])) continue;     // lens arcs are distinct pairs (p|q, q|r)
        if (pair_of(others[0]) != pair_of(others[1])) continue;  // doubled bisector pair matches (p|r, p|r)
        if (pair_of(others[0]) == pair_of(to_w[0]) || pair_of(others[0]) == pair_of(to_w[1])) continue;

        // Move the doubled-bisector branch whose far endpoint is nearer W.
        const Vec2 pW = dcel.vertices[W].point;
        const int t0 = target_of(others[0]), t1 = target_of(others[1]);
        const int hmov = (dist2(dcel.vertices[t0].point, pW) <= dist2(dcel.vertices[t1].point, pW)) ? others[0] : others[1];
        const int htwin = dcel.halfedges[hmov].twin;

        dcel.halfedges[hmov].origin = W;
        if (!dcel.halfedges[hmov].polyline.empty()) dcel.halfedges[hmov].polyline.front() = pW;
        if (htwin >= 0 && !dcel.halfedges[htwin].polyline.empty()) dcel.halfedges[htwin].polyline.back() = pW;

        dcel.vertices[W].incident_halfedge = hmov;
        dcel.vertices[vi].incident_halfedge = to_w[0];
        ++repaired;
    }
    return repaired;
}

// Run steps (d), (e), (f) of the DCEL build over the current vertices
// and halfedges arrays. Assumes:
//   - Every halfedge has its twin, origin, polyline, left_site,
//     right_site set correctly.
//   - prev/next/face on halfedges may be in any state; this function
//     overwrites them.
//
// Updates:
//   - DCELHalfEdge::next, prev, face — set per the angular sort +
//     ring-walk logic.
//   - dcel.faces — cleared and rebuilt from cycle walks.
//   - dcel.outside_face — set to the index of the (canonical) outside
//     face, or -1 if none found.
//   - dcel.orphan_halfedges — recomputed.
//   - dcel.outside_face_anomalies — recomputed (> 0 if more than one
//     outside cycle was identified).
//   - dcel.build_note — set to a short status string ("dcel built",
//     "dcel built (warning: ...)", etc.).
inline void relink_and_label_dcel(ApolloniusDCEL& dcel) {
    // (d) Per-vertex angular sort of outgoing half-edges.
    std::vector<std::vector<int>> outgoing_per_vertex(dcel.vertices.size());
    for (std::size_t hi = 0; hi < dcel.halfedges.size(); ++hi) {
        const int v = dcel.halfedges[hi].origin;
        if (v >= 0 && v < static_cast<int>(outgoing_per_vertex.size())) {
            outgoing_per_vertex[v].push_back(static_cast<int>(hi));
        }
    }
    for (auto& list : outgoing_per_vertex) {
        std::sort(list.begin(), list.end(),
                  [&dcel](int lhs, int rhs) {
                      return dcel_outgoing_angle(dcel.halfedges[lhs]) <
                             dcel_outgoing_angle(dcel.halfedges[rhs]);
                  });
    }

    // (e) Link next / prev.
    //
    // For each half-edge h ending at vertex v_end (origin of h.twin):
    //   find h.twin in v_end's sorted outgoing list, and take the
    //   PREVIOUS entry in CCW (ascending-angle) order, wrapping. That
    //   entry is h.next. This is the standard half-edge convention for
    //   CCW face traversal: the next half-edge in the boundary of the
    //   face on h's LEFT is the outgoing half-edge from v_end that
    //   makes the SMALLEST clockwise turn from h.twin's direction (so
    //   the same face stays on the left). In a CCW-sorted outgoing
    //   list that is the entry immediately CW (one step backward),
    //   not CCW, from h.twin.
    //
    // NOTE: taking the entry immediately CCW instead (the smallest angle
    // greater than h.twin's angle, wrapping around) would produce a CW
    // face traversal and yield inconsistent left_site cycles at vertices
    // of degree >= 3. The rule below keeps the standard CCW invariant.
    for (std::size_t hi = 0; hi < dcel.halfedges.size(); ++hi) {
        DCELHalfEdge& h = dcel.halfedges[hi];
        const int twin_idx = h.twin;
        if (twin_idx < 0) continue;
        const int v_end = dcel.halfedges[twin_idx].origin;
        if (v_end < 0 || v_end >= static_cast<int>(outgoing_per_vertex.size())) continue;
        const auto& list = outgoing_per_vertex[v_end];
        if (list.size() < 2) continue;
        // Locate twin in `list`.
        std::size_t pos = list.size();
        for (std::size_t k = 0; k < list.size(); ++k) {
            if (list[k] == twin_idx) { pos = k; break; }
        }
        if (pos == list.size()) continue;
        const std::size_t next_pos = (pos + list.size() - 1) % list.size();
        const int next_idx = list[next_pos];
        h.next = next_idx;
        dcel.halfedges[next_idx].prev = static_cast<int>(hi);
    }

    // (f) Face identification by cycle walking.
    std::vector<int> halfedge_face(dcel.halfedges.size(), -1);
    auto walk_cycle = [&](const int start) {
        std::vector<int> cycle;
        int cur = start;
        const std::size_t cap = dcel.halfedges.size() + 4;
        while (cur >= 0 && cycle.size() < cap) {
            cycle.push_back(cur);
            const int nxt = dcel.halfedges[cur].next;
            if (nxt == start) {
                return cycle;  // closed cycle.
            }
            if (nxt < 0) {
                cycle.clear();  // open chain: cannot form a face.
                return cycle;
            }
            cur = nxt;
        }
        cycle.clear();  // exceeded cap (defensive).
        return cycle;
    };

    int outside_cycle_count = 0;
    for (std::size_t hi = 0; hi < dcel.halfedges.size(); ++hi) {
        if (halfedge_face[hi] >= 0) continue;
        if (dcel.halfedges[hi].next < 0) continue;
        const std::vector<int> cycle = walk_cycle(static_cast<int>(hi));
        if (cycle.empty()) continue;

        // Determine the face's owning site: it's the common left_site
        // of every half-edge in the cycle. A cycle with
        // left_site == -1 throughout IS the outside face (its boundary
        // walks the bbox perimeter CW). If the left_sites disagree
        // we still treat the cycle as "outside" defensively, but log
        // it as an anomaly (the bbox-ring stitch should produce a
        // single uniform -1 cycle and any deviation needs attention).
        int site = dcel.halfedges[cycle.front()].left_site;
        bool consistent = true;
        for (int idx : cycle) {
            if (dcel.halfedges[idx].left_site != site) {
                consistent = false;
                break;
            }
        }
        DCELFace face;
        face.boundary_halfedge = cycle.front();
        if (consistent && site >= 0) {
            face.site = site;
            face.is_outside = false;
        } else {
            face.site = -1;
            face.is_outside = true;
            outside_cycle_count += 1;
        }
        const int face_idx = static_cast<int>(dcel.faces.size());
        dcel.faces.push_back(face);
        if (face.is_outside) {
            dcel.outside_face = face_idx;
        }
        for (int idx : cycle) {
            halfedge_face[idx] = face_idx;
            dcel.halfedges[idx].face = face_idx;
        }
    }

    // Recount orphans after face linking (a half-edge with no face
    // either had next == -1, or sat in an over-cap chain).
    dcel.orphan_halfedges = 0;
    for (const auto& h : dcel.halfedges) {
        if (h.face < 0) {
            dcel.orphan_halfedges += 1;
        }
    }

    // Expect exactly one outside-face cycle after bbox-ring
    // stitching. Anything else is reported in build_note.
    if (outside_cycle_count != 1) {
        dcel.outside_face_anomalies = outside_cycle_count;
    }

    if (dcel.halfedges.empty()) {
        dcel.build_note = "no segments produced (no edges and no bbox ring)";
    } else if (outside_cycle_count == 0) {
        dcel.build_note = "dcel built (warning: no outside-face cycle found)";
    } else if (outside_cycle_count > 1) {
        dcel.build_note = "dcel built (warning: multiple outside-face cycles: " +
                          std::to_string(outside_cycle_count) + ")";
    } else {
        dcel.build_note = "dcel built";
    }
}

}  // namespace detail

inline ApolloniusDCEL build_apollonius_dcel(
    const std::vector<ApolloniusVertexInput>& in_vertices,
    const std::vector<ApolloniusEdgeInput>& in_edges,
    const std::vector<Vec2>& sites,
    const std::vector<double>& radii,
    const Rect& bounds,
    bool repair_lens = true) {
    ApolloniusDCEL dcel;

    // (a) Vertex copy with bbox filter.
    //
    // The analytic engine's solve_apollonius_vertices accepts triple-tangent
    // points up to 40% outside the bbox -- that padding keeps the engine's
    // pair-curve parameter sampling well-resolved near the boundary. But
    // those phantom out-of-bbox vertices, if let into the DCEL, participate
    // in boundary walks as halfedge origins. At an out-of-bbox origin the
    // angular sort places the halfedge in an unpredictable slot, the
    // next-link logic jumps to the wrong sibling, and the resulting cycle
    // crosses cell boundaries (mixing left_sites). That trips
    // outside_face_anomalies.
    //
    // Keep in-bounds vertices only. Edges whose polylines anchored to a
    // dropped vertex will find no matching anchor at that endpoint and
    // get routed through the free-end bbox-extension code path -- which
    // is exactly the correct behaviour for an edge that ends near (but
    // inside) the bbox after the visibility filter clipped it.
    constexpr double kInBoundsTol = 1.0e-9;
    auto in_bbox = [&bounds](const Vec2& p) {
        return p.x >= bounds.min_x - kInBoundsTol &&
               p.x <= bounds.max_x + kInBoundsTol &&
               p.y >= bounds.min_y - kInBoundsTol &&
               p.y <= bounds.max_y + kInBoundsTol;
    };
    dcel.vertices.reserve(in_vertices.size() + 4);
    for (const auto& v : in_vertices) {
        if (!in_bbox(v.point)) continue;
        DCELVertex dv;
        dv.point = v.point;
        dv.sites = v.sites;
        dv.incident_halfedge = -1;
        dv.is_boundary = false;
        dcel.vertices.push_back(dv);
    }
    const int interior_vertex_count = static_cast<int>(dcel.vertices.size());


    // (b) + (c) Edge segmentation + half-edge pair creation.
    auto vertex_has_site = [](const std::array<int, 3>& vs, const int s) {
        return vs[0] == s || vs[1] == s || vs[2] == s;
    };

    for (const auto& edge : in_edges) {
        if (edge.polyline.size() < 2) {
            continue;
        }
        const int a = edge.left_site;
        const int b = edge.right_site;
        if (a < 0 || b < 0) {
            continue;
        }

        const std::vector<double> cum_s = detail::dcel_arclengths(edge.polyline);
        const double total_s = cum_s.back();

        // Find anchor vertex indices for this edge (those whose sites
        // include BOTH a and b).
        struct Anchor {
            int vertex_index = -1;
            double s = 0.0;
        };
        std::vector<Anchor> anchors;
        for (int vi = 0; vi < interior_vertex_count; ++vi) {
            const auto& vs = dcel.vertices[vi].sites;
            if (vertex_has_site(vs, a) && vertex_has_site(vs, b)) {
                Anchor anc;
                anc.vertex_index = vi;
                anc.s = detail::dcel_param_at_point(edge.polyline, cum_s, dcel.vertices[vi].point);
                anchors.push_back(anc);
            }
        }
        std::sort(anchors.begin(), anchors.end(),
                  [](const Anchor& l, const Anchor& r) { return l.s < r.s; });

        // Deduplicate anchors with near-identical parameters (a single
        // vertex sometimes maps to two close param values when the
        // pair-curve dips). Keep the first occurrence.
        anchors.erase(std::unique(anchors.begin(), anchors.end(),
                                  [](const Anchor& l, const Anchor& r) {
                                      return std::abs(l.s - r.s) <= 1.0e-7;
                                  }),
                      anchors.end());

        const std::size_t anchor_count = anchors.size();

        // Build the analytic pair-curve for sites a / b once per
        // input edge. When valid we resample each output segment along
        // the true curve via adaptive subdivision; line-type curves
        // (equal radii) short-circuit and the polyline collapses to
        // just the two endpoints. When the pair-curve is degenerate the
        // polyline-slice fallback below keeps the original sampling.
        const Vec2& a_site_pos =
            (a >= 0 && a < static_cast<int>(sites.size())) ? sites[a] : Vec2{};
        const Vec2& b_site_pos =
            (b >= 0 && b < static_cast<int>(sites.size())) ? sites[b] : Vec2{};
        const double a_radius =
            (a >= 0 && a < static_cast<int>(radii.size())) ? radii[a] : 0.0;
        const double b_radius =
            (b >= 0 && b < static_cast<int>(radii.size())) ? radii[b] : 0.0;
        const std::optional<PairCurve> pair_curve =
            make_pair_curve(a_site_pos, a_radius, b_site_pos, b_radius);

        // Build interior segments (between consecutive anchors). Then,
        // for the FRONT tail (anchor 0's polyline tail back to s=0 with
        // no anchor) and BACK tail (anchor N-1's tail forward to
        // s=total_s with no anchor), produce a boundary vertex if the
        // corresponding polyline end actually reaches the bbox.
        //
        // Also handle 0 anchors: both ends are boundary (full polyline
        // becomes a single boundary-to-boundary segment).

        // Helper: create a boundary vertex at point `p` (snapped to the
        // bbox), with sites {a, b, -1}. Returns the new vertex index.
        auto add_boundary_vertex = [&](const Vec2& p) -> int {
            const Vec2 snapped = detail::dcel_snap_to_bbox(p, bounds);
            DCELVertex bv;
            bv.point = snapped;
            bv.sites = {a, b, -1};
            bv.incident_halfedge = -1;
            bv.is_boundary = true;
            const int idx = static_cast<int>(dcel.vertices.size());
            dcel.vertices.push_back(bv);
            return idx;
        };

        // Helper: build the polyline slice from s_lo to s_hi, snapping
        // start/end to provided override points (so seg endpoints
        // exactly equal the vertex coordinates). When a valid
        // analytic pair-curve exists, replace the polyline slice with
        // an adaptive analytic resampling between the override points
        // -- smooth hyperbolic arcs instead of straight chord segments.
        auto build_seg = [&](double s_lo, double s_hi, const Vec2& start_override,
                             const Vec2& end_override) -> std::vector<Vec2> {
            std::vector<Vec2> seg = detail::dcel_subpolyline(edge.polyline, cum_s, s_lo, s_hi);
            if (seg.size() >= 2) {
                seg.front() = start_override;
                seg.back() = end_override;
            }
            if (seg.size() >= 2 && pair_curve && pair_curve->valid) {
                seg = sample_pair_curve_arc(*pair_curve, start_override, end_override);
            }
            return seg;
        };

        // Helper: emit a half-edge pair with given polyline and origin
        // / target vertex indices. Uses dcel_left_right_assignment to
        // pick which of {a, b} sits on the left of the forward direction.
        auto emit_pair = [&](int origin_v, int target_v, std::vector<Vec2> seg) {
            if (seg.size() < 2) return;
            const auto [left_a, right_a] =
                detail::dcel_left_right_assignment(seg, a, b, sites, radii);
            std::vector<Vec2> seg_rev(seg.rbegin(), seg.rend());
            DCELHalfEdge fwd;
            fwd.origin = origin_v;
            fwd.left_site = left_a;
            fwd.right_site = right_a;
            fwd.polyline = std::move(seg);
            DCELHalfEdge bwd;
            bwd.origin = target_v;
            bwd.left_site = right_a;
            bwd.right_site = left_a;
            bwd.polyline = std::move(seg_rev);
            const int fwd_idx = static_cast<int>(dcel.halfedges.size());
            const int bwd_idx = fwd_idx + 1;
            fwd.twin = bwd_idx;
            bwd.twin = fwd_idx;
            dcel.halfedges.push_back(std::move(fwd));
            dcel.halfedges.push_back(std::move(bwd));
            if (dcel.vertices[origin_v].incident_halfedge < 0) {
                dcel.vertices[origin_v].incident_halfedge = fwd_idx;
            }
            if (dcel.vertices[target_v].incident_halfedge < 0) {
                dcel.vertices[target_v].incident_halfedge = bwd_idx;
            }
        };

        // Helper: compute the bbox-boundary endpoint for a free end of the
        // polyline. We shoot a ray from the second-to-last point through the
        // last point and find the first intersection with the bbox perimeter.
        // This correctly handles the common case where the Apollonius tracer
        // stops inside the bbox (because a third site takes over) rather than
        // stopping exactly on the bbox edge.
        auto front_bbox_point = [&]() -> Vec2 {
            // Front end: ray goes from poly[1] through poly[0].
            if (edge.polyline.size() >= 2) {
                return detail::dcel_extend_end_to_bbox(
                    edge.polyline[1], edge.polyline[0], bounds);
            }
            return detail::dcel_snap_to_bbox(edge.polyline.front(), bounds);
        };
        auto back_bbox_point = [&]() -> Vec2 {
            // Back end: ray goes from poly[n-2] through poly[n-1].
            const std::size_t n = edge.polyline.size();
            if (n >= 2) {
                return detail::dcel_extend_end_to_bbox(
                    edge.polyline[n - 2], edge.polyline[n - 1], bounds);
            }
            return detail::dcel_snap_to_bbox(edge.polyline.back(), bounds);
        };

        // A free polyline end genuinely reaches the bbox only when
        // one of the edge's two sites {a, b} owns the boundary at the exit
        // point. The analytic tracer over-traces a bisector past the
        // triple-tangent vertex where a THIRD site takes over (a
        // "briefly-visible" tail), leaving an occluded stub that runs out to
        // the bbox. Emitting a boundary vertex for such a stub injects a
        // spurious perimeter vertex whose flanking sites do not own the
        // adjacent boundary -- corrupting the perimeter-ring stitch and the
        // angular relink, which splits the outside-face cycle and leaves the
        // genuinely-bbox-touching cells unclosed.
        // Suppress the tail when a third site is strictly closer than BOTH a
        // and b at the exit point (additive metric, matching the perimeter
        // ring's own inner-site test).
        auto tail_reaches_bbox = [&](const Vec2& ext) -> bool {
            if (a < 0 || b < 0 ||
                a >= static_cast<int>(sites.size()) ||
                b >= static_cast<int>(sites.size())) {
                return true;
            }
            const double ra = a < static_cast<int>(radii.size()) ? radii[a] : 0.0;
            const double rb = b < static_cast<int>(radii.size()) ? radii[b] : 0.0;
            const double m_ab = std::min(detail::dcel_additive_metric(ext, sites[a], ra),
                                         detail::dcel_additive_metric(ext, sites[b], rb));
            double best = m_ab;
            for (std::size_t i = 0; i < sites.size(); ++i) {
                const double ri = i < radii.size() ? radii[i] : 0.0;
                best = std::min(best, detail::dcel_additive_metric(ext, sites[i], ri));
            }
            constexpr double kOcclusionTol = 1.0e-7;
            return m_ab <= best + kOcclusionTol;  // a or b is (tied for) closest
        };

        // Every free (non-anchor) polyline end is unconditionally
        // extended to the bbox boundary by shooting a ray along the
        // polyline's terminal segment. The Apollonius tracer stops inside
        // the bbox (when a third site takes over the region), so we cannot
        // use an endpoint-proximity-to-bbox check; we always extend.
        if (anchor_count == 0) {
            // Full polyline from free-end to free-end: extend both ends to
            // the bbox boundary.
            const Vec2 ext_front = front_bbox_point();
            const Vec2 ext_back  = back_bbox_point();
            // Skip if both extensions land on the same bbox point, or if
            // either end is occluded (a third site owns the boundary there).
            if (length_sq(ext_front - ext_back) > 1.0e-20 &&
                tail_reaches_bbox(ext_front) && tail_reaches_bbox(ext_back)) {
                const int v_front = add_boundary_vertex(ext_front);
                const int v_back  = add_boundary_vertex(ext_back);
                std::vector<Vec2> seg =
                    build_seg(0.0, total_s, dcel.vertices[v_front].point,
                              dcel.vertices[v_back].point);
                if (seg.size() >= 2) {
                    emit_pair(v_front, v_back, std::move(seg));
                }
            } else {
                dcel.unmatched_polyline_endpoints += 1;
            }
        } else {
            // anchor_count >= 1.
            //
            // A "free front end" exists when the polyline's front point is
            // geometrically distinct from the first anchor vertex, i.e., the
            // polyline extends beyond the anchor back toward the front. The
            // threshold is relative rather than a raw arclength epsilon: the
            // tail must be at least 0.1% of the total polyline arclength.
            // This filters out the sub-1e-3 gaps the Apollonius tracer
            // leaves between a polyline endpoint and the nearest
            // triple-tangent vertex (the tracer lands within ~1e-4 units of
            // the vertex because of finite sampling; a raw 1e-9 arclength
            // threshold would treat these near-zero stubs as real tails and
            // extend them to the bbox, creating spurious boundary vertices).
            const double tail_threshold = total_s * 1.0e-3;
            const bool has_front_tail = anchors.front().s > tail_threshold;
            const bool has_back_tail =
                (total_s - anchors.back().s) > tail_threshold;

            // FRONT tail: from polyline start to first anchor. Extend the
            // free front end to the bbox boundary -- unless it is occluded
            // (a third site owns the boundary there: a briefly-visible tail).
            if (has_front_tail) {
                const Vec2 ext_front = front_bbox_point();
                if (tail_reaches_bbox(ext_front)) {
                    const int v_front = add_boundary_vertex(ext_front);
                    std::vector<Vec2> seg = build_seg(
                        0.0, anchors.front().s,
                        dcel.vertices[v_front].point,
                        dcel.vertices[anchors.front().vertex_index].point);
                    if (seg.size() >= 2) {
                        emit_pair(v_front, anchors.front().vertex_index, std::move(seg));
                    }
                }
            }

            // Interior segments between consecutive anchors.
            for (std::size_t i = 0; i + 1 < anchor_count; ++i) {
                const Anchor& a0 = anchors[i];
                const Anchor& a1 = anchors[i + 1];
                std::vector<Vec2> seg = build_seg(
                    a0.s, a1.s,
                    dcel.vertices[a0.vertex_index].point,
                    dcel.vertices[a1.vertex_index].point);
                if (seg.size() >= 2) {
                    emit_pair(a0.vertex_index, a1.vertex_index, std::move(seg));
                }
            }

            // BACK tail: from last anchor to polyline end. Extend the free
            // back end to the bbox boundary -- unless it is occluded (a third
            // site owns the boundary there: a briefly-visible tail).
            if (has_back_tail) {
                const Vec2 ext_back = back_bbox_point();
                if (tail_reaches_bbox(ext_back)) {
                    const int v_back = add_boundary_vertex(ext_back);
                    std::vector<Vec2> seg = build_seg(
                        anchors.back().s, total_s,
                        dcel.vertices[anchors.back().vertex_index].point,
                        dcel.vertices[v_back].point);
                    if (seg.size() >= 2) {
                        emit_pair(anchors.back().vertex_index, v_back, std::move(seg));
                    }
                }
            }
        }
    }

    dcel.boundary_vertices_added =
        static_cast<int>(dcel.vertices.size()) - interior_vertex_count;

    detail::build_perimeter_ring(dcel, sites, radii, bounds);
    if (repair_lens) {
        dcel.degenerate_lenses_repaired = detail::repair_degenerate_lens(dcel);
    }
    detail::relink_and_label_dcel(dcel);

    return dcel;
}

}  // namespace fortune
