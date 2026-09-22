#include "fortune/apollonius_engine.h"

#include "fortune/common.h"
#include "fortune/pair_curve.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fortune {

// ---------------------------------------------------------------------------
// Geometric helpers
// ---------------------------------------------------------------------------

bool inside_bounds_local(const Vec2& point, const Rect& bounds, const double pad) {
    return point.x >= bounds.min_x - pad && point.x <= bounds.max_x + pad &&
           point.y >= bounds.min_y - pad && point.y <= bounds.max_y + pad;
}

std::vector<double> solve_real_quadratic(const double a, const double b, const double c) {
    std::vector<double> roots;
    if (std::abs(a) <= kEpsilon) {
        if (std::abs(b) <= kEpsilon) {
            return roots;
        }
        roots.push_back(-c / b);
        return roots;
    }
    const double disc = b * b - 4.0 * a * c;
    if (disc < -1.0e-10) {
        return roots;
    }
    if (std::abs(disc) <= 1.0e-10) {
        roots.push_back(-b / (2.0 * a));
        return roots;
    }
    const double s = std::sqrt(std::max(0.0, disc));
    const double q = -0.5 * (b + std::copysign(s, b));
    roots.push_back(q / a);
    roots.push_back(c / q);
    return roots;
}

// ---------------------------------------------------------------------------
// Apollonius vertex solver
// ---------------------------------------------------------------------------

std::vector<ApolloniusVertex> solve_apollonius_vertices(
    const std::vector<Vec2>& points,
    const std::vector<double>& radii,
    const std::vector<int>& active_sites,
    const int ia,
    const int ib,
    const int ic,
    const Rect& bounds) {
    const Vec2& a = points[ia];
    const Vec2& b = points[ib];
    const Vec2& c = points[ic];
    const double ra = radii[ia];
    const double rb = radii[ib];
    const double rc = radii[ic];

    const Vec2 row0 = (b - a) * 2.0;
    const Vec2 row1 = (c - a) * 2.0;
    const double det = row0.x * row1.y - row0.y * row1.x;
    if (std::abs(det) <= 1.0e-10) {
        return {};
    }

    const auto solve_linear = [&](const Vec2& rhs) {
        return Vec2{(rhs.x * row1.y - row0.y * rhs.y) / det,
                    (row0.x * rhs.y - rhs.x * row1.x) / det};
    };

    const Vec2 rhs_const{
        ra * ra - rb * rb - length_sq(a) + length_sq(b),
        ra * ra - rc * rc - length_sq(a) + length_sq(c),
    };
    const Vec2 rhs_d{
        2.0 * (ra - rb),
        2.0 * (ra - rc),
    };

    const Vec2 x_const = solve_linear(rhs_const);
    const Vec2 x_d = solve_linear(rhs_d);
    const Vec2 base = x_const - a;
    const double qa = length_sq(x_d) - 1.0;
    const double qb = 2.0 * (dot(base, x_d) - ra);
    const double qc = length_sq(base) - ra * ra;

    std::vector<ApolloniusVertex> out;
    for (double value : solve_real_quadratic(qa, qb, qc)) {
        if (!std::isfinite(value)) {
            continue;
        }
        const Vec2 point = x_const + x_d * value;
        if (!inside_bounds_local(point, bounds,
                                  std::max(bounds.max_x - bounds.min_x,
                                           bounds.max_y - bounds.min_y) * 0.4)) {
            continue;
        }

        const double metric_a = additive_distance(point, a, ra);
        const double metric_b = additive_distance(point, b, rb);
        const double metric_c = additive_distance(point, c, rc);
        const double best = std::min({metric_a, metric_b, metric_c});
        if (std::abs(metric_a - metric_b) > 1.0e-5 || std::abs(metric_a - metric_c) > 1.0e-5) {
            continue;
        }

        bool valid = true;
        for (int site_id : active_sites) {
            const double metric = additive_distance(point, points[site_id], radii[site_id]);
            if (metric < best - 1.0e-5) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            continue;
        }

        bool duplicate = false;
        for (const auto& existing : out) {
            if (length_sq(existing.point - point) <= 1.0e-12) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            out.push_back(ApolloniusVertex{point, metric_a, {ia, ib, ic}});
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Pair visibility
// ---------------------------------------------------------------------------

bool apollonius_pair_visible(const Vec2& point,
                             const std::vector<Vec2>& points,
                             const std::vector<double>& radii,
                             const std::vector<int>& active_sites,
                             const int left_site,
                             const int right_site,
                             const Rect& bounds,
                             ApolloniusPredicateKernel* kernel) {
    if (!inside_bounds_local(point, bounds)) {
        return false;
    }
    const double left_metric = additive_distance(point, points[left_site], radii[left_site]);
    const double right_metric = additive_distance(point, points[right_site], radii[right_site]);
    const bool same_pair_metric =
        kernel ? kernel->metrics_equal(point,
                                       points[left_site],
                                       radii[left_site],
                                       points[right_site],
                                       radii[right_site],
                                       2.5e-4,
                                       "pair visibility equality S" + std::to_string(left_site) +
                                           "/S" + std::to_string(right_site))
               : std::abs(left_metric - right_metric) <= 2.5e-4;
    if (!same_pair_metric) {
        return false;
    }
    const double best = std::min(left_metric, right_metric);
    for (int site_id : active_sites) {
        if (site_id == left_site || site_id == right_site) {
            continue;
        }
        const double metric = additive_distance(point, points[site_id], radii[site_id]);
        const bool beats_pair =
            kernel ? kernel->metric_leq(point, points[site_id], radii[site_id], best - 2.5e-4,
                                        "pair visibility blocker S" + std::to_string(site_id))
                   : metric <= best - 2.5e-4;
        if (beats_pair) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Main diagram builder
// ---------------------------------------------------------------------------

ApolloniusDiagram build_apollonius_diagram(const std::vector<Vec2>& points,
                                           const std::vector<double>& radii,
                                           const Rect& bounds,
                                           const int trace_resolution,
                                           const ApolloniusArithmeticMode arithmetic_mode,
                                           ApolloniusPredicateAudit* audit) {
    return build_apollonius_diagram(points, radii, bounds, trace_resolution, arithmetic_mode,
                                    audit, nullptr);
}

ApolloniusDiagram build_apollonius_diagram(const std::vector<Vec2>& points,
                                           const std::vector<double>& radii,
                                           const Rect& bounds,
                                           const int trace_resolution,
                                           const ApolloniusArithmeticMode mode,
                                           ApolloniusPredicateAudit* audit,
                                           ApolloniusEngineState* out_state) {
    ApolloniusPredicateKernel kernel(mode, audit);
    ApolloniusDiagram diagram;
    if (points.empty()) {
        diagram.note = "no sites";
        if (out_state) {
            out_state->site_count = 0;
            out_state->dominated.clear();
            out_state->active_sites.clear();
            out_state->dominated_sites.clear();
            out_state->vertices.clear();
            out_state->edges.clear();
            out_state->raw_pair_polylines.clear();
            out_state->note = diagram.note;
        }
        return diagram;
    }

    std::vector<bool> dominated(points.size(), false);
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        for (int j = 0; j < static_cast<int>(points.size()); ++j) {
            if (i == j || dominated[j]) {
                continue;
            }
            const double dist = length(points[i] - points[j]);
            if (radii[i] > radii[j] + dist + 1.0e-8 ||
                (i < j && std::abs(radii[i] - (radii[j] + dist)) <= 1.0e-8)) {
                dominated[j] = true;
            }
        }
    }
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        if (dominated[i]) {
            diagram.dominated_sites.push_back(i);
        } else {
            diagram.active_sites.push_back(i);
        }
    }

    if (diagram.active_sites.size() < 2) {
        diagram.note = "one site dominates the whole view";
        if (out_state) {
            out_state->site_count = static_cast<int>(points.size());
            out_state->dominated = dominated;
            out_state->active_sites = diagram.active_sites;
            out_state->dominated_sites = diagram.dominated_sites;
            out_state->vertices.clear();
            out_state->edges.clear();
            out_state->raw_pair_polylines.clear();
            out_state->note = diagram.note;
        }
        return diagram;
    }

    for (std::size_t a = 0; a < diagram.active_sites.size(); ++a) {
        for (std::size_t b = a + 1; b < diagram.active_sites.size(); ++b) {
            for (std::size_t c = b + 1; c < diagram.active_sites.size(); ++c) {
                for (const auto& vertex : solve_apollonius_vertices(
                         points, radii, diagram.active_sites, diagram.active_sites[a],
                         diagram.active_sites[b], diagram.active_sites[c], bounds)) {
                    bool duplicate = false;
                    for (const auto& existing : diagram.vertices) {
                        if (length_sq(existing.point - vertex.point) <= 1.0e-12) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        diagram.vertices.push_back(vertex);
                    }
                }
            }
        }
    }

    const double world_w = bounds.max_x - bounds.min_x;
    const double world_h = bounds.max_y - bounds.min_y;
    const double line_extent = std::hypot(world_w, world_h) * 1.5;
    const int samples = std::max(48, trace_resolution * 2);

    if (out_state) {
        out_state->raw_pair_polylines.clear();
    }

    for (std::size_t ai = 0; ai < diagram.active_sites.size(); ++ai) {
        for (std::size_t bi = ai + 1; bi < diagram.active_sites.size(); ++bi) {
            const int left_site = diagram.active_sites[ai];
            const int right_site = diagram.active_sites[bi];
            const auto curve = make_pair_curve(points[left_site], radii[left_site],
                                               points[right_site], radii[right_site]);
            if (!curve || !curve->valid) {
                continue;
            }

            std::vector<double> parameters;
            parameters.reserve(samples + diagram.vertices.size());
            if (curve->line) {
                for (int step = 0; step < samples; ++step) {
                    const double t =
                        static_cast<double>(step) / static_cast<double>(samples - 1);
                    parameters.push_back(-line_extent + 2.0 * line_extent * t);
                }
            } else {
                const double t_max = 4.8;
                for (int step = 0; step < samples; ++step) {
                    const double t =
                        static_cast<double>(step) / static_cast<double>(samples - 1);
                    parameters.push_back(-t_max + 2.0 * t_max * t);
                }
            }
            for (const auto& vertex : diagram.vertices) {
                const bool uses_pair =
                    (vertex.sites[0] == left_site || vertex.sites[1] == left_site ||
                     vertex.sites[2] == left_site) &&
                    (vertex.sites[0] == right_site || vertex.sites[1] == right_site ||
                     vertex.sites[2] == right_site);
                if (uses_pair) {
                    parameters.push_back(parameter_on_pair_curve(*curve, vertex.point));
                }
            }
            std::sort(parameters.begin(), parameters.end());
            parameters.erase(
                std::unique(parameters.begin(), parameters.end(),
                            [](const double lhs, const double rhs) {
                                return std::abs(lhs - rhs) <= 1.0e-6;
                            }),
                parameters.end());

            // Build the raw pre-filter polyline (all curve samples, no visibility check).
            std::vector<Vec2> raw_polyline;
            raw_polyline.reserve(parameters.size());
            for (double parameter : parameters) {
                raw_polyline.push_back(point_on_pair_curve(*curve, parameter));
            }
            if (out_state) {
                const int lo = std::min(left_site, right_site);
                const int hi = std::max(left_site, right_site);
                out_state->raw_pair_polylines[{lo, hi}] = raw_polyline;
            }

            auto smooth_polyline = [&](const std::vector<Vec2>& chord_poly) {
                std::vector<Vec2> out;
                if (chord_poly.size() < 2) {
                    return chord_poly;
                }
                out.reserve(chord_poly.size() * 4);
                out.push_back(chord_poly.front());
                for (std::size_t i = 0; i + 1 < chord_poly.size(); ++i) {
                    const std::vector<Vec2> arc =
                        sample_pair_curve_arc(*curve, chord_poly[i], chord_poly[i + 1]);
                    for (std::size_t k = 1; k < arc.size(); ++k) {
                        out.push_back(arc[k]);
                    }
                }
                return out;
            };

            ApolloniusEdge current;
            current.left_site = left_site;
            current.right_site = right_site;
            bool in_run = false;
            for (std::size_t i = 0; i < parameters.size(); ++i) {
                const Vec2& point = raw_polyline[i];
                const bool visible = apollonius_pair_visible(
                    point, points, radii, diagram.active_sites, left_site, right_site, bounds,
                    &kernel);
                if (!visible) {
                    if (in_run && current.polyline.size() >= 2) {
                        current.polyline = smooth_polyline(current.polyline);
                        diagram.edges.push_back(current);
                    }
                    current = ApolloniusEdge{left_site, right_site, {}};
                    in_run = false;
                    continue;
                }

                if (!in_run) {
                    current = ApolloniusEdge{left_site, right_site, {}};
                    in_run = true;
                }
                if (current.polyline.empty() ||
                    length_sq(current.polyline.back() - point) > 1.0e-12) {
                    current.polyline.push_back(point);
                }
            }
            if (in_run && current.polyline.size() >= 2) {
                current.polyline = smooth_polyline(current.polyline);
                diagram.edges.push_back(current);
            }
        }
    }

    diagram.note = diagram.vertices.empty()
                       ? "analytic Apollonius trace; no visible triple tangencies in bounds"
                       : "analytic Apollonius trace with exact triple-tangent vertices";

    if (out_state) {
        out_state->site_count = static_cast<int>(points.size());
        out_state->dominated = dominated;
        out_state->active_sites = diagram.active_sites;
        out_state->dominated_sites = diagram.dominated_sites;
        out_state->vertices = diagram.vertices;
        out_state->edges = diagram.edges;
        out_state->note = diagram.note;
    }

    return diagram;
}

// ---------------------------------------------------------------------------
// Power diagram (Laguerre / weighted Voronoi)
// ---------------------------------------------------------------------------

// Sutherland-Hodgman half-plane clip.
// Keeps the side of the hyperplane { p : dot(p, normal) <= offset }.
static std::vector<Vec2> clip_polygon_power(const std::vector<Vec2>& polygon,
                                            const Vec2& normal,
                                            const double offset) {
    std::vector<Vec2> result;
    if (polygon.empty()) return result;

    const auto inside = [&](const Vec2& p) {
        return dot(p, normal) <= offset + kRangeEpsilon;
    };
    const auto edge_intersect = [&](const Vec2& a, const Vec2& b) {
        const double sa = dot(a, normal) - offset;
        const double sb = dot(b, normal) - offset;
        const double denom = sa - sb;
        if (std::abs(denom) <= kEpsilon) return a;
        const double t = sa / denom;
        return a + (b - a) * t;
    };

    const std::size_t n = polygon.size();
    for (std::size_t i = 0; i < n; ++i) {
        const Vec2& cur  = polygon[i];
        const Vec2& prev = polygon[(i + n - 1) % n];
        const bool cur_in  = inside(cur);
        const bool prev_in = inside(prev);
        if (cur_in) {
            if (!prev_in) result.push_back(edge_intersect(prev, cur));
            result.push_back(cur);
        } else if (prev_in) {
            result.push_back(edge_intersect(prev, cur));
        }
    }
    return result;
}

PowerDiagram build_power_diagram(const std::vector<Vec2>& points,
                                 const std::vector<double>& radii,
                                 const Rect& bounds) {
    PowerDiagram diagram;
    const int n = static_cast<int>(points.size());
    if (n == 0) {
        diagram.note = "no sites";
        return diagram;
    }

    diagram.cells.resize(n);

    // Bbox corner polygon (CW from min_x,min_y so that winding is
    // consistent with the Sutherland-Hodgman sense used here).
    const std::vector<Vec2> bbox_poly = {
        {bounds.min_x, bounds.min_y},
        {bounds.max_x, bounds.min_y},
        {bounds.max_x, bounds.max_y},
        {bounds.min_x, bounds.max_y},
    };

    for (int s = 0; s < n; ++s) {
        std::vector<Vec2> poly = bbox_poly;
        bool dominated = false;
        for (int t = 0; t < n; ++t) {
            if (t == s) continue;
            // Power bisector: keep the side where pow(p,s) <= pow(p,t)
            //   |p-s|^2 - rs^2 <= |p-t|^2 - rt^2
            //   p·(t-s)*2 <= |t|^2 - |s|^2 - rt^2 + rs^2
            const Vec2 normal{points[t].x - points[s].x,
                              points[t].y - points[s].y};
            const double rhs = 0.5 * (
                (points[t].x * points[t].x + points[t].y * points[t].y) -
                (points[s].x * points[s].x + points[s].y * points[s].y) -
                radii[t] * radii[t] + radii[s] * radii[s]);
            poly = clip_polygon_power(poly, normal, rhs);
            if (poly.size() < 3) { dominated = true; break; }
        }
        if (dominated || poly.size() < 3) {
            diagram.dominated_sites.push_back(s);
        } else {
            diagram.active_sites.push_back(s);
            diagram.cells[s] = std::move(poly);
        }
    }

    // Build edge list: for each active site pair (s,t), find the shared
    // boundary segment between their cells.  The power bisector is the
    // straight line; clamp to the actual shared portion by intersecting
    // successive vertex pairs on the cell boundary.
    const int active_count = static_cast<int>(diagram.active_sites.size());
    for (int ai = 0; ai < active_count; ++ai) {
        for (int bi = ai + 1; bi < active_count; ++bi) {
            const int s = diagram.active_sites[ai];
            const int t = diagram.active_sites[bi];
            const auto& cell_s = diagram.cells[s];
            const auto& cell_t = diagram.cells[t];
            if (cell_s.empty() || cell_t.empty()) continue;

            // The bisector normal and offset.
            const Vec2 normal{points[t].x - points[s].x,
                              points[t].y - points[s].y};
            const double rhs = 0.5 * (
                (points[t].x * points[t].x + points[t].y * points[t].y) -
                (points[s].x * points[s].x + points[s].y * points[s].y) -
                radii[t] * radii[t] + radii[s] * radii[s]);

            // Collect vertices of cell_s that lie exactly on the bisector
            // (i.e., both cells share them).  A vertex is "on the bisector"
            // if |dot(v, normal) - rhs| <= small tolerance.
            // Also collect edge-edge intersections of cell_s edges with the
            // bisector line that also lie inside cell_t.
            std::vector<Vec2> shared;
            const double tol = std::max(1.0e-7,
                                        std::max(bounds.max_x - bounds.min_x,
                                                 bounds.max_y - bounds.min_y) * 1.0e-6);

            const auto on_bisector = [&](const Vec2& p) {
                return std::abs(dot(p, normal) - rhs) <= tol * (length(normal) + 1.0);
            };
            const auto in_cell_t = [&](const Vec2& p) {
                // p is inside cell_t's convex polygon (point-in-convex test via
                // all half-planes used to build cell_t is expensive; use a simple
                // winding test instead).
                const std::size_t sz = cell_t.size();
                int crossings = 0;
                for (std::size_t i = 0; i < sz; ++i) {
                    const Vec2& va = cell_t[i];
                    const Vec2& vb = cell_t[(i + 1) % sz];
                    if ((va.y > p.y) != (vb.y > p.y)) {
                        const double x_cross =
                            vb.x + (p.y - vb.y) / (va.y - vb.y) * (va.x - vb.x);
                        if (p.x < x_cross) ++crossings;
                    }
                }
                return (crossings & 1) != 0;
            };

            // Walk edges of cell_s and collect bisector-crossing segments.
            const std::size_t ns = cell_s.size();
            for (std::size_t i = 0; i < ns; ++i) {
                const Vec2& va = cell_s[i];
                const Vec2& vb = cell_s[(i + 1) % ns];
                // Check each endpoint.
                for (const Vec2* vp : {&va, &vb}) {
                    if (on_bisector(*vp)) {
                        // Dedup
                        bool dup = false;
                        for (const auto& existing : shared) {
                            if (length_sq(existing - *vp) <= tol * tol) { dup = true; break; }
                        }
                        if (!dup) shared.push_back(*vp);
                    }
                }
                // Edge-bisector intersection.
                const double da = dot(va, normal) - rhs;
                const double db = dot(vb, normal) - rhs;
                if (da * db < -tol * tol) {
                    const double denom = da - db;
                    if (std::abs(denom) > kEpsilon) {
                        const Vec2 pt = va + (vb - va) * (da / denom);
                        bool dup = false;
                        for (const auto& existing : shared) {
                            if (length_sq(existing - pt) <= tol * tol) { dup = true; break; }
                        }
                        if (!dup) shared.push_back(pt);
                    }
                }
            }

            // Filter to points that lie inside (or on the boundary of) cell_t.
            std::vector<Vec2> edge_pts;
            for (const auto& p : shared) {
                if (in_cell_t(p) || on_bisector(p)) {
                    edge_pts.push_back(p);
                }
            }

            if (edge_pts.size() < 2) continue;

            // Sort along the bisector direction (tangent to normal).
            const Vec2 tangent = perp(normal);
            std::sort(edge_pts.begin(), edge_pts.end(), [&](const Vec2& x, const Vec2& y) {
                return dot(x, tangent) < dot(y, tangent);
            });

            PowerEdge e;
            e.left_site  = s;
            e.right_site = t;
            e.a = edge_pts.front();
            e.b = edge_pts.back();
            diagram.edges.push_back(e);
        }
    }

    // Build vertices: collect all unique corners of all cells that appear
    // at least in 3 distinct cell polygons (i.e., Power diagram vertices
    // where 3 or more sites' cells meet).
    // Also collect boundary corners that border two cells.
    std::vector<Vec2> all_verts;
    std::vector<std::vector<int>> vert_sites;  // which cells contributed each vertex

    for (int ai = 0; ai < active_count; ++ai) {
        const int s = diagram.active_sites[ai];
        for (const auto& p : diagram.cells[s]) {
            // Find or insert in all_verts.
            int vidx = -1;
            for (int vi = 0; vi < static_cast<int>(all_verts.size()); ++vi) {
                if (length_sq(all_verts[vi] - p) <= 1.0e-10) {
                    vidx = vi;
                    break;
                }
            }
            if (vidx < 0) {
                vidx = static_cast<int>(all_verts.size());
                all_verts.push_back(p);
                vert_sites.emplace_back();
            }
            // Record this site for the vertex (dedup).
            auto& sv = vert_sites[vidx];
            if (std::find(sv.begin(), sv.end(), s) == sv.end()) {
                sv.push_back(s);
            }
        }
    }

    for (int vi = 0; vi < static_cast<int>(all_verts.size()); ++vi) {
        const auto& sv = vert_sites[vi];
        if (sv.size() >= 3) {
            // Inside the bbox (not a corner of the bounding box itself).
            const Vec2& p = all_verts[vi];
            const bool on_bbox =
                (std::abs(p.x - bounds.min_x) < 1.0e-9 ||
                 std::abs(p.x - bounds.max_x) < 1.0e-9 ||
                 std::abs(p.y - bounds.min_y) < 1.0e-9 ||
                 std::abs(p.y - bounds.max_y) < 1.0e-9);
            if (!on_bbox) {
                PowerVertex pv;
                pv.point = p;
                pv.sites[0] = sv[0];
                pv.sites[1] = sv.size() >= 2 ? sv[1] : -1;
                pv.sites[2] = sv.size() >= 3 ? sv[2] : -1;
                diagram.vertices.push_back(pv);
            }
        }
    }

    if (n == 1) {
        diagram.note = "power diagram: single site (whole bbox)";
    } else if (diagram.active_sites.empty()) {
        diagram.note = "power diagram: all sites dominated";
    } else {
        diagram.note = "power diagram: Laguerre/weighted Voronoi (straight-line bisectors)";
    }

    return diagram;
}

// ---------------------------------------------------------------------------
// k-th order Voronoi helper
// ---------------------------------------------------------------------------

HigherOrderCellHit k_nearest_subset(const Vec2& p,
                                    const std::vector<Vec2>& points,
                                    int k) {
    HigherOrderCellHit hit;
    const int n = static_cast<int>(points.size());
    if (n == 0 || k <= 0) {
        return hit;
    }
    k = std::min(k, n);

    // Build index array and squared-distance array.
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::vector<double> dists(n);
    for (int s = 0; s < n; ++s) {
        const double dx = p.x - points[s].x;
        const double dy = p.y - points[s].y;
        dists[s] = dx * dx + dy * dy;
    }

    // Partial sort: move the k smallest indices to the front.
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return dists[a] < dists[b]; });
    idx.resize(k);

    // Canonical subset key: ascending site index order.
    std::sort(idx.begin(), idx.end());
    hit.subset = std::move(idx);
    return hit;
}

// ---------------------------------------------------------------------------
// Multiplicatively-weighted Voronoi (MW Voronoi)
// ---------------------------------------------------------------------------

MWDiagram build_mw_voronoi(const std::vector<Vec2>& points,
                           const std::vector<double>& weights,
                           const Rect& /*bounds*/) {
    MWDiagram diagram;
    const int n = static_cast<int>(points.size());
    if (n == 0) {
        diagram.note = "mw voronoi: no sites";
        return diagram;
    }

    // Collect active sites (positive weight).
    for (int i = 0; i < n; ++i) {
        const double w = (i < static_cast<int>(weights.size())) ? weights[i] : 0.0;
        if (w > 0.0) {
            diagram.active_sites.push_back(i);
        }
    }

    if (diagram.active_sites.empty()) {
        diagram.note = "mw voronoi: all weights zero or negative";
        return diagram;
    }

    // Compute bisectors for all active pairs (a, b) with a < b.
    const double weight_eq_tol = 1.0e-9;

    const int active_count = static_cast<int>(diagram.active_sites.size());
    for (int ai = 0; ai < active_count; ++ai) {
        for (int bi = ai + 1; bi < active_count; ++bi) {
            const int a = diagram.active_sites[ai];
            const int b = diagram.active_sites[bi];
            const double wa = (a < static_cast<int>(weights.size())) ? std::max(weights[a], 1.0e-12) : 1.0e-12;
            const double wb = (b < static_cast<int>(weights.size())) ? std::max(weights[b], 1.0e-12) : 1.0e-12;
            const Vec2& sa = points[a];
            const Vec2& sb = points[b];

            MWBisector bis;
            bis.site_a = a;
            bis.site_b = b;

            if (std::abs(wa - wb) < weight_eq_tol * (wa + wb + 1.0)) {
                // Equal weights: perpendicular bisector line.
                bis.is_line = true;
                bis.line_mid = (sa + sb) * 0.5;
                bis.line_normal = Vec2{sb.x - sa.x, sb.y - sa.y};
            } else {
                // Unequal weights: Apollonius circle.
                bis.is_line = false;
                const double wa2 = wa * wa;
                const double wb2 = wb * wb;
                const double denom = wb2 - wa2;  // non-zero since wa != wb
                // center = (wb^2 * sa - wa^2 * sb) / (wb^2 - wa^2)
                bis.circle_center.x = (wb2 * sa.x - wa2 * sb.x) / denom;
                bis.circle_center.y = (wb2 * sa.y - wa2 * sb.y) / denom;
                // radius = (wa * wb * |sa - sb|) / |wb^2 - wa^2|
                const double dist_ab = length(sa - sb);
                bis.circle_radius = (wa * wb * dist_ab) / std::abs(denom);
                // Larger weight wins outside the circle.
                if (wa < wb) {
                    bis.inner_site = a;  // smaller weight -> inside
                    bis.outer_site = b;  // larger weight  -> outside
                } else {
                    bis.inner_site = b;
                    bis.outer_site = a;
                }
            }

            diagram.bisectors.push_back(bis);
        }
    }

    if (diagram.active_sites.size() == 1) {
        diagram.note = "mw voronoi: single active site";
    } else {
        diagram.note = "mw voronoi: MW Voronoi / Apollonius circles; bisectors = circles (|p-s|/w_s metric)";
    }

    return diagram;
}

// ---------------------------------------------------------------------------
// Incremental analytic insert
// ---------------------------------------------------------------------------

void apollonius_diagram_incremental_insert(
    const ApolloniusEngineState& prev_state,
    const int new_site_idx,
    const std::vector<Vec2>& after_points,
    const std::vector<double>& after_radii,
    const Rect& bounds,
    const int trace_resolution,
    const ApolloniusArithmeticMode mode,
    ApolloniusEngineState* out_state,
    ApolloniusDiagram* out_diagram,
    ApolloniusPredicateAudit* audit) {

    // Precondition: only append-at-end is supported.
    if (new_site_idx != prev_state.site_count) {
        throw std::invalid_argument(
            "apollonius_diagram_incremental_insert: new_site_idx (" +
            std::to_string(new_site_idx) +
            ") must equal prev_state.site_count (" +
            std::to_string(prev_state.site_count) +
            "); generic mid-insertion is not supported");
    }

    const int n = new_site_idx + 1;  // total sites after insertion

    // -----------------------------------------------------------------------
    // Step 1: Dominance update.
    // Start from prev_state.dominated, extend by one element, then check
    // interactions between the new site and all previous active sites.
    // -----------------------------------------------------------------------
    std::vector<bool> dominated(n, false);
    // Copy previous dominated flags.
    for (int i = 0; i < prev_state.site_count; ++i) {
        dominated[i] = prev_state.dominated[i];
    }
    // New site starts un-dominated (will be updated below).
    dominated[new_site_idx] = false;

    // Check new site against all existing sites (using the same predicate as
    // build_apollonius_diagram's double-loop).
    for (int j = 0; j < prev_state.site_count; ++j) {
        // Can existing site j dominate the new site?
        if (!dominated[j]) {
            const double dist = length(after_points[j] - after_points[new_site_idx]);
            if (after_radii[j] > after_radii[new_site_idx] + dist + 1.0e-8 ||
                (j < new_site_idx &&
                 std::abs(after_radii[j] - (after_radii[new_site_idx] + dist)) <= 1.0e-8)) {
                dominated[new_site_idx] = true;
                break;
            }
        }
    }
    // Can the new site dominate any existing (currently active) sites?
    if (!dominated[new_site_idx]) {
        for (int j = 0; j < prev_state.site_count; ++j) {
            if (dominated[j]) {
                continue;
            }
            const double dist = length(after_points[new_site_idx] - after_points[j]);
            if (after_radii[new_site_idx] > after_radii[j] + dist + 1.0e-8 ||
                (new_site_idx < j &&
                 std::abs(after_radii[new_site_idx] - (after_radii[j] + dist)) <= 1.0e-8)) {
                dominated[j] = true;
            }
        }
    }

    // Build active/dominated site lists in index order.
    out_state->dominated = dominated;
    out_state->active_sites.clear();
    out_state->dominated_sites.clear();
    for (int i = 0; i < n; ++i) {
        if (dominated[i]) {
            out_state->dominated_sites.push_back(i);
        } else {
            out_state->active_sites.push_back(i);
        }
    }
    out_state->site_count = n;

    // Build a quick lookup: is site i active?
    auto is_active = [&](const int site_id) -> bool {
        for (int s : out_state->active_sites) {
            if (s == site_id) return true;
        }
        return false;
    };

    // -----------------------------------------------------------------------
    // Step 2: Carry over surviving triple-tangent vertices from prev_state.
    // A vertex survives iff all three of its sites are still active AND no
    // currently-active site beats the vertex's metric value. The new site
    // may have a better metric at a previously-valid triple point, making
    // that point no longer a valid Apollonius vertex.
    // -----------------------------------------------------------------------
    out_state->vertices.clear();
    for (const auto& v : prev_state.vertices) {
        if (!is_active(v.sites[0]) || !is_active(v.sites[1]) || !is_active(v.sites[2])) {
            continue;
        }
        // Re-validate: no active site may have a metric strictly better than
        // the triple vertex's metric value (using the same 1e-5 threshold as
        // solve_apollonius_vertices).
        bool still_valid = true;
        for (int site_id : out_state->active_sites) {
            const double metric =
                additive_distance(v.point, after_points[site_id], after_radii[site_id]);
            if (metric < v.value - 1.0e-5) {
                still_valid = false;
                break;
            }
        }
        if (still_valid) {
            out_state->vertices.push_back(v);
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Add new triple-tangent vertices involving the new site.
    // Only iterate pairs (a, b) where a < b, both active, neither == new site.
    // -----------------------------------------------------------------------
    if (!dominated[new_site_idx]) {
        // Build a list of existing active sites (those that were already in
        // prev_state and are still active in the new active set).
        std::vector<int> existing_active;
        existing_active.reserve(out_state->active_sites.size());
        for (int s : out_state->active_sites) {
            if (s != new_site_idx) {
                existing_active.push_back(s);
            }
        }

        for (std::size_t ai = 0; ai < existing_active.size(); ++ai) {
            for (std::size_t bi = ai + 1; bi < existing_active.size(); ++bi) {
                const int a = existing_active[ai];
                const int b = existing_active[bi];
                for (const auto& vertex : solve_apollonius_vertices(
                         after_points, after_radii, out_state->active_sites,
                         a, b, new_site_idx, bounds)) {
                    // Dedup against all vertices accumulated so far.
                    bool duplicate = false;
                    for (const auto& existing : out_state->vertices) {
                        if (length_sq(existing.point - vertex.point) <= 1.0e-12) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) {
                        out_state->vertices.push_back(vertex);
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 4: Build edges (per-pair loop matching build_apollonius_diagram).
    //
    // Optimisation: scoped visibility-filter pass.
    //
    // Most pairs (a, b) are UNAFFECTED by the insertion of the new site.
    // A pair is "affected" if the new site's metric beats the pair edge's
    // metric at any cached raw-polyline sample — i.e., if m_new < m_edge - tol
    // for some sample point.  If no sample triggers this, the new site cannot
    // clip this pair's edges (it is never closer than a and b), and the cached
    // filtered edges from prev_state can be forwarded verbatim.
    //
    // Correctness justification: for an unaffected pair (a, b), the new site
    // never beats the pair metric at any sample.  A valid triple-tangent point
    // (a, b, new_site) would require the three sites to share the same metric
    // at some point, which contradicts the new site never beating the (a, b)
    // metric.  Therefore no new anchors exist for this pair, and the cached
    // edges from prev_state are exactly correct.
    //
    // Pairs involving the new site are always "affected" (new geometry).
    // Pairs with no cached raw polyline (brand new pairs that were not in
    // prev_state) are also treated as affected.
    // -----------------------------------------------------------------------
    out_state->edges.clear();
    out_state->raw_pair_polylines.clear();

    if (out_state->active_sites.size() >= 2) {
        ApolloniusPredicateKernel kernel(mode, audit);
        const double world_w = bounds.max_x - bounds.min_x;
        const double world_h = bounds.max_y - bounds.min_y;
        const double line_extent = std::hypot(world_w, world_h) * 1.5;
        const int samples = std::max(48, trace_resolution * 2);

        // Cached values for the new site (used in the affect-check).
        const Vec2 new_site_center = after_points[new_site_idx];
        const double new_site_radius = after_radii[new_site_idx];
        // Tolerance for the affect-check: the new site beats the pair edge
        // if m_new < m_edge - kAffectTol.  Using 1e-5 matches the existing
        // solve_apollonius_vertices threshold.
        constexpr double kAffectTol = 1.0e-5;

        // Pre-build flat lookups indexed by encoded pair key (lo * prev_n + hi)
        // to replace O(log n) std::map lookups with O(1) vector accesses.
        // Using raw pointers into prev_state (const-ref, lives for the duration
        // of this function) is safe.
        const int prev_n = prev_state.site_count;
        const std::size_t flat_size =
            static_cast<std::size_t>(prev_n) * static_cast<std::size_t>(prev_n);

        // raw_poly_ptr[lo * prev_n + hi] -> pointer into prev_state.raw_pair_polylines,
        // or nullptr if not present.
        std::vector<const std::vector<Vec2>*> raw_poly_ptr(flat_size, nullptr);
        for (const auto& kv : prev_state.raw_pair_polylines) {
            const int lo = kv.first.first;
            const int hi = kv.first.second;
            if (lo >= 0 && hi < prev_n) {
                raw_poly_ptr[static_cast<std::size_t>(lo) * prev_n + hi] = &kv.second;
            }
        }

        // For unaffected pairs, we copy their edges from prev_state.edges verbatim.
        // Build a flat sorted list of (encoded_key, edge_index) pairs so we can
        // copy edges with a single linear pass per pair — avoiding vector-of-vectors
        // heap allocations.
        // Each element: pair<encoded_key, edge_index>.
        std::vector<std::pair<std::size_t, std::size_t>> edge_by_key;
        edge_by_key.reserve(prev_state.edges.size());
        for (std::size_t ei = 0; ei < prev_state.edges.size(); ++ei) {
            const auto& e = prev_state.edges[ei];
            const int lo = std::min(e.left_site, e.right_site);
            const int hi = std::max(e.left_site, e.right_site);
            if (lo >= 0 && hi < prev_n) {
                edge_by_key.emplace_back(
                    static_cast<std::size_t>(lo) * prev_n + hi, ei);
            }
        }
        std::sort(edge_by_key.begin(), edge_by_key.end());  // sort by encoded key

        auto smooth_polyline = [&](const PairCurve& curve,
                                   const std::vector<Vec2>& chord_poly) {
            std::vector<Vec2> out_v;
            if (chord_poly.size() < 2) {
                return chord_poly;
            }
            out_v.reserve(chord_poly.size() * 4);
            out_v.push_back(chord_poly.front());
            for (std::size_t i = 0; i + 1 < chord_poly.size(); ++i) {
                const std::vector<Vec2> arc =
                    sample_pair_curve_arc(curve, chord_poly[i], chord_poly[i + 1]);
                for (std::size_t k = 1; k < arc.size(); ++k) {
                    out_v.push_back(arc[k]);
                }
            }
            return out_v;
        };

        for (std::size_t ai = 0; ai < out_state->active_sites.size(); ++ai) {
            for (std::size_t bi = ai + 1; bi < out_state->active_sites.size(); ++bi) {
                const int left_site = out_state->active_sites[ai];
                const int right_site = out_state->active_sites[bi];

                const std::pair<int,int> key{std::min(left_site, right_site),
                                             std::max(left_site, right_site)};
                const bool involves_new =
                    (left_site == new_site_idx || right_site == new_site_idx);

                // -------------------------------------------------------
                // Affect-check for pairs that don't touch the new site.
                // -------------------------------------------------------
                // Flat O(1) lookup for the cached raw polyline (valid for any
                // pair whose both sites were present in prev_state).
                const int lo_k = key.first;
                const int hi_k = key.second;
                const std::vector<Vec2>* const raw_ptr =
                    (!involves_new && lo_k >= 0 && hi_k < prev_n)
                        ? raw_poly_ptr[static_cast<std::size_t>(lo_k) * prev_n + hi_k]
                        : nullptr;

                if (!involves_new) {
                    if (raw_ptr != nullptr) {
                        const auto& raw = *raw_ptr;
                        // Two-phase affect-check for speed:
                        //   Phase 1 — sparse scan (every kStride-th sample).
                        //     If any sample is affected, we know immediately.
                        //     If none are, proceed to phase 2 for confirmation.
                        //   Phase 2 — dense scan (all samples).
                        //     Only runs when the sparse scan found no hits.
                        //     Guarantees we don't falsely skip a briefly-affected pair.
                        //
                        // For AFFECTED pairs: phase 1 usually fires in the first few
                        // samples (the new site broadly beats the pair), so cost ≈ O(1).
                        // For UNAFFECTED pairs: phase 1 scans kStride samples cheaply,
                        // then phase 2 confirms with the full set.  Without this split,
                        // unaffected pairs each pay the full O(|raw|) scan.
                        constexpr std::size_t kStride = 8;
                        bool affected = false;

                        // Phase 1: sparse.
                        for (std::size_t si = 0; si < raw.size(); si += kStride) {
                            const Vec2& p = raw[si];
                            const double m_edge =
                                additive_distance(p, after_points[left_site],
                                                  after_radii[left_site]);
                            const double m_new =
                                additive_distance(p, new_site_center, new_site_radius);
                            if (m_new < m_edge - kAffectTol) {
                                affected = true;
                                break;
                            }
                        }
                        // Phase 2: dense confirmation (only when sparse found nothing).
                        if (!affected) {
                            for (const Vec2& p : raw) {
                                const double m_edge =
                                    additive_distance(p, after_points[left_site],
                                                      after_radii[left_site]);
                                const double m_new =
                                    additive_distance(p, new_site_center, new_site_radius);
                                if (m_new < m_edge - kAffectTol) {
                                    affected = true;
                                    break;
                                }
                            }
                        }

                        if (!affected) {
                            // Unaffected pair — reuse prev_state edges verbatim.
                            // Forward the cached raw polyline unchanged (curve
                            // geometry is unaltered, no new anchors exist — see
                            // correctness justification above).
                            const std::size_t ek =
                                static_cast<std::size_t>(lo_k) * prev_n + hi_k;
                            // Binary search into edge_by_key for this pair's entries.
                            auto range_lo = std::lower_bound(
                                edge_by_key.begin(), edge_by_key.end(),
                                std::make_pair(ek, std::size_t{0}));
                            while (range_lo != edge_by_key.end() && range_lo->first == ek) {
                                out_state->edges.push_back(prev_state.edges[range_lo->second]);
                                ++range_lo;
                            }
                            out_state->raw_pair_polylines[key] = *raw_ptr;
                            continue;
                        }
                    }
                    // No cached polyline or affected — fall through to full pipeline.
                }

                // -------------------------------------------------------
                // Full visibility-filter pipeline.
                //
                // For non-new-site pairs that ARE affected: the pair curve
                // geometry (and thus the raw polyline points) is UNCHANGED —
                // only the visibility filter output changes.  Reuse the cached
                // raw polyline from prev_state directly, injecting only any
                // new anchor points from (left, right, new_site) triples.
                // This avoids make_pair_curve + full parameter resampling.
                //
                // For pairs involving the new site: no cached polyline exists,
                // so go through the full make_pair_curve path.
                // -------------------------------------------------------
                std::vector<Vec2> raw_polyline;
                std::optional<PairCurve> curve_opt;

                // Helper to get (or lazy-construct) the pair curve.
                auto get_curve = [&]() -> const PairCurve* {
                    if (!curve_opt) {
                        curve_opt = make_pair_curve(after_points[left_site],
                                                    after_radii[left_site],
                                                    after_points[right_site],
                                                    after_radii[right_site]);
                    }
                    return (curve_opt && curve_opt->valid) ? &*curve_opt : nullptr;
                };

                // Determine whether we can reuse the cached raw polyline:
                // - Pair must not involve the new site (curve is unchanged)
                // - Cached raw polyline must exist
                // - No (a, b, new_site) triple-vertex anchor must exist for this pair
                //   (if one does, it shifts the visibility-filter anchor point, which
                //    could change the filtered edge count).
                bool use_cached_raw = false;
                if (!involves_new && raw_ptr != nullptr) {
                    bool has_new_triple_anchor = false;
                    for (const auto& vertex : out_state->vertices) {
                        const bool uses_pair =
                            (vertex.sites[0] == left_site || vertex.sites[1] == left_site ||
                             vertex.sites[2] == left_site) &&
                            (vertex.sites[0] == right_site || vertex.sites[1] == right_site ||
                             vertex.sites[2] == right_site);
                        const bool uses_new =
                            (vertex.sites[0] == new_site_idx ||
                             vertex.sites[1] == new_site_idx ||
                             vertex.sites[2] == new_site_idx);
                        if (uses_pair && uses_new) {
                            has_new_triple_anchor = true;
                            break;
                        }
                    }
                    use_cached_raw = !has_new_triple_anchor;
                }

                if (use_cached_raw) {
                    // No new anchors and curve unchanged: reuse cached raw polyline.
                    raw_polyline = *raw_ptr;
                    out_state->raw_pair_polylines[key] = raw_polyline;
                } else {
                    // Pairs involving the new site (or no cached polyline available):
                    // full parameter-list build and resample.
                    const PairCurve* curve = get_curve();
                    if (!curve) continue;

                    std::vector<double> parameters;
                    parameters.reserve(samples + out_state->vertices.size());
                    if (curve->line) {
                        for (int step = 0; step < samples; ++step) {
                            const double t =
                                static_cast<double>(step) / static_cast<double>(samples - 1);
                            parameters.push_back(-line_extent + 2.0 * line_extent * t);
                        }
                    } else {
                        const double t_max = 4.8;
                        for (int step = 0; step < samples; ++step) {
                            const double t =
                                static_cast<double>(step) / static_cast<double>(samples - 1);
                            parameters.push_back(-t_max + 2.0 * t_max * t);
                        }
                    }
                    for (const auto& vertex : out_state->vertices) {
                        const bool uses_pair =
                            (vertex.sites[0] == left_site || vertex.sites[1] == left_site ||
                             vertex.sites[2] == left_site) &&
                            (vertex.sites[0] == right_site || vertex.sites[1] == right_site ||
                             vertex.sites[2] == right_site);
                        if (uses_pair) {
                            parameters.push_back(parameter_on_pair_curve(*curve, vertex.point));
                        }
                    }
                    std::sort(parameters.begin(), parameters.end());
                    parameters.erase(
                        std::unique(parameters.begin(), parameters.end(),
                                    [](const double lhs, const double rhs) {
                                        return std::abs(lhs - rhs) <= 1.0e-6;
                                    }),
                        parameters.end());

                    raw_polyline.reserve(parameters.size());
                    for (double parameter : parameters) {
                        raw_polyline.push_back(point_on_pair_curve(*curve, parameter));
                    }
                    out_state->raw_pair_polylines[key] = raw_polyline;
                }

                // Visibility filter — same logic as build_apollonius_diagram.
                // Requires the PairCurve for smooth_polyline.  For affected old pairs
                // that had no new anchor points, curve_opt may not have been constructed
                // yet (lazy); call get_curve() here to ensure it's available.
                const PairCurve* filter_curve = get_curve();
                if (!filter_curve) continue;

                ApolloniusEdge current;
                current.left_site = left_site;
                current.right_site = right_site;
                bool in_run = false;
                for (std::size_t i = 0; i < raw_polyline.size(); ++i) {
                    const Vec2& point = raw_polyline[i];
                    const bool visible = apollonius_pair_visible(
                        point, after_points, after_radii, out_state->active_sites,
                        left_site, right_site, bounds, &kernel);
                    if (!visible) {
                        if (in_run && current.polyline.size() >= 2) {
                            current.polyline = smooth_polyline(*filter_curve, current.polyline);
                            out_state->edges.push_back(current);
                        }
                        current = ApolloniusEdge{left_site, right_site, {}};
                        in_run = false;
                        continue;
                    }
                    if (!in_run) {
                        current = ApolloniusEdge{left_site, right_site, {}};
                        in_run = true;
                    }
                    if (current.polyline.empty() ||
                        length_sq(current.polyline.back() - point) > 1.0e-12) {
                        current.polyline.push_back(point);
                    }
                }
                if (in_run && current.polyline.size() >= 2) {
                    current.polyline = smooth_polyline(*filter_curve, current.polyline);
                    out_state->edges.push_back(current);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 5: Populate note.
    // -----------------------------------------------------------------------
    if (out_state->active_sites.size() < 2) {
        if (out_state->active_sites.empty()) {
            // All sites dominated — degenerate.
            out_state->note = "no sites";
        } else {
            out_state->note = "one site dominates the whole view";
        }
    } else {
        out_state->note = out_state->vertices.empty()
                              ? "analytic Apollonius trace; no visible triple tangencies in bounds"
                              : "analytic Apollonius trace with exact triple-tangent vertices";
    }

    // -----------------------------------------------------------------------
    // Step 6: Copy to optional flat diagram output.
    // -----------------------------------------------------------------------
    if (out_diagram) {
        out_diagram->active_sites = out_state->active_sites;
        out_diagram->dominated_sites = out_state->dominated_sites;
        out_diagram->vertices = out_state->vertices;
        out_diagram->edges = out_state->edges;
        out_diagram->note = out_state->note;
    }
}

}  // namespace fortune
