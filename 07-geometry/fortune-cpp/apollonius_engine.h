// Apollonius diagram analytic engine.
//
// Kept separate from the app layer so both the demo binary and the test
// binary can link against the engine without dragging in the full UI layer.
//
// All symbols live in namespace fortune.  There are no anonymous namespaces
// here (that would cause ODR issues when included from multiple TUs); the
// heavy implementations live in apollonius_engine.cpp.

#pragma once

#include "fortune/common.h"
#include "fortune/pair_curve.h"

#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fortune {

// ---------------------------------------------------------------------------
// Core diagram types
// ---------------------------------------------------------------------------

struct ApolloniusVertex {
    Vec2 point;
    double value = 0.0;
    std::array<int, 3> sites{-1, -1, -1};
};

struct ApolloniusEdge {
    int left_site = -1;
    int right_site = -1;
    std::vector<Vec2> polyline;
};

struct ApolloniusDiagram {
    bool exact_sites_only = true;
    std::string note;
    std::vector<int> active_sites;
    std::vector<int> dominated_sites;
    std::vector<ApolloniusVertex> vertices;
    std::vector<ApolloniusEdge> edges;
};

enum class ApolloniusArithmeticMode {
    analytic_double,
    lazy_filtered,
};

struct ApolloniusPredicateAudit {
    int fast_accepts = 0;
    int lazy_evaluations = 0;
    int unresolved_fallbacks = 0;
    // Counts input predicates where |fast| < tolerance * 0.1 in
    // lazy_filtered mode. Includes nonzero near-ties; not an exactness test.
    int near_zero_predicates = 0;
    std::vector<std::string> notes;

    void record(std::string note) {
        if (notes.size() < 96) {
            notes.push_back(std::move(note));
        }
    }
};

// ---------------------------------------------------------------------------
// Standalone arithmetic helpers
// ---------------------------------------------------------------------------

inline double additive_distance(const Vec2& point, const Vec2& site, const double radius) {
    return length(point - site) - radius;
}

struct MetricInterval {
    double lo = 0.0;
    double hi = 0.0;
};

inline MetricInterval sqrt_interval(const double value) {
    const double root = std::sqrt(std::max(0.0, value));
    const double hi = std::nextafter(root, std::numeric_limits<double>::infinity());
    const double lo = std::nextafter(root, -std::numeric_limits<double>::infinity());
    const double pad = (std::abs(root) + 1.0) * 8.0 * std::numeric_limits<double>::epsilon();
    return {lo - pad, hi + pad};
}

// ---------------------------------------------------------------------------
// Predicate kernel
// ---------------------------------------------------------------------------

class ApolloniusPredicateKernel {
public:
    ApolloniusPredicateKernel(const ApolloniusArithmeticMode mode, ApolloniusPredicateAudit* audit)
        : mode_(mode), audit_(audit) {}

    [[nodiscard]] double metric(const Vec2& point, const Vec2& site, const double radius) const {
        return additive_distance(point, site, radius);
    }

    [[nodiscard]] bool metric_less(const Vec2& query,
                                   const Vec2& lhs_site,
                                   const double lhs_radius,
                                   const Vec2& rhs_site,
                                   const double rhs_radius,
                                   const std::string& label = {}) {
        return compare_metrics(query, lhs_site, lhs_radius, rhs_site, rhs_radius, label) < 0;
    }

    [[nodiscard]] bool contains_site(const Vec2& outer_center,
                                     const double outer_radius,
                                     const Vec2& inner_center,
                                     const double inner_radius,
                                     const std::string& label = {}) {
        const double fast = length(outer_center - inner_center) + inner_radius - outer_radius;
        const double scale = std::max({1.0, std::abs(outer_radius), std::abs(inner_radius),
                                       length(outer_center - inner_center)});
        const double tol = scale * 64.0 * std::numeric_limits<double>::epsilon();
        const int sgn = sign_or_filter(
            label.empty() ? "IsHidden/contains" : label,
            fast,
            tol,
            [&]() {
                const MetricInterval dist = sqrt_interval(length_sq(outer_center - inner_center));
                return MetricInterval{dist.lo + inner_radius - outer_radius,
                                      dist.hi + inner_radius - outer_radius};
            });
        return sgn <= 0;
    }

    [[nodiscard]] bool metric_leq(const Vec2& point,
                                  const Vec2& site,
                                  const double radius,
                                  const double value,
                                  const std::string& label = {}) {
        const double fast = additive_distance(point, site, radius) - value;
        const double scale = std::max({1.0, std::abs(fast), std::abs(value),
                                       length(point - site), std::abs(radius)});
        const double tol = scale * 96.0 * std::numeric_limits<double>::epsilon();
        const int sgn = sign_or_filter(
            label.empty() ? "metric <= value" : label,
            fast,
            tol,
            [&]() {
                const MetricInterval dist = sqrt_interval(length_sq(point - site));
                return MetricInterval{dist.lo - radius - value, dist.hi - radius - value};
            });
        return sgn <= 0;
    }

    [[nodiscard]] bool metrics_equal(const Vec2& point,
                                     const Vec2& lhs_site,
                                     const double lhs_radius,
                                     const Vec2& rhs_site,
                                     const double rhs_radius,
                                     const double tolerance,
                                     const std::string& label = {}) {
        const int cmp = compare_metrics(point, lhs_site, lhs_radius, rhs_site, rhs_radius, label);
        if (cmp == 0) {
            return true;
        }
        const double diff = additive_distance(point, lhs_site, lhs_radius) -
                            additive_distance(point, rhs_site, rhs_radius);
        return std::abs(diff) <= tolerance;
    }

private:
    [[nodiscard]] int compare_metrics(const Vec2& query,
                                      const Vec2& lhs_site,
                                      const double lhs_radius,
                                      const Vec2& rhs_site,
                                      const double rhs_radius,
                                      const std::string& label) {
        const double lhs = additive_distance(query, lhs_site, lhs_radius);
        const double rhs = additive_distance(query, rhs_site, rhs_radius);
        const double fast = lhs - rhs;
        const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs),
                                       length(query - lhs_site), length(query - rhs_site)});
        const double tol = scale * 128.0 * std::numeric_limits<double>::epsilon();
        return sign_or_filter(
            label.empty() ? "SideOfBisector(metric compare)" : label,
            fast,
            tol,
            [&]() {
                const MetricInterval lhs_i = metric_interval(query, lhs_site, lhs_radius);
                const MetricInterval rhs_i = metric_interval(query, rhs_site, rhs_radius);
                return MetricInterval{lhs_i.lo - rhs_i.hi, lhs_i.hi - rhs_i.lo};
            });
    }

    [[nodiscard]] MetricInterval metric_interval(const Vec2& point,
                                                 const Vec2& site,
                                                 const double radius) const {
        const MetricInterval dist = sqrt_interval(length_sq(point - site));
        return {dist.lo - radius, dist.hi - radius};
    }

    template <typename IntervalFn>
    [[nodiscard]] int sign_or_filter(const std::string& label,
                                     const double fast,
                                     const double tolerance,
                                     IntervalFn interval_fn);

    ApolloniusArithmeticMode mode_ = ApolloniusArithmeticMode::analytic_double;
    ApolloniusPredicateAudit* audit_ = nullptr;
};

// ---------------------------------------------------------------------------
// Geometric predicates and main builder (declarations; bodies in .cpp)
// ---------------------------------------------------------------------------

bool inside_bounds_local(const Vec2& point, const Rect& bounds, double pad = 0.0);

std::vector<double> solve_real_quadratic(double a, double b, double c);

std::vector<ApolloniusVertex> solve_apollonius_vertices(const std::vector<Vec2>& points,
                                                        const std::vector<double>& radii,
                                                        const std::vector<int>& active_sites,
                                                        int ia,
                                                        int ib,
                                                        int ic,
                                                        const Rect& bounds);

bool apollonius_pair_visible(const Vec2& point,
                             const std::vector<Vec2>& points,
                             const std::vector<double>& radii,
                             const std::vector<int>& active_sites,
                             int left_site,
                             int right_site,
                             const Rect& bounds,
                             ApolloniusPredicateKernel* kernel = nullptr);

ApolloniusDiagram build_apollonius_diagram(
    const std::vector<Vec2>& points,
    const std::vector<double>& radii,
    const Rect& bounds,
    int trace_resolution,
    ApolloniusArithmeticMode arithmetic_mode = ApolloniusArithmeticMode::analytic_double,
    ApolloniusPredicateAudit* audit = nullptr);

// ---------------------------------------------------------------------------
// Power diagram (Laguerre / weighted Voronoi with squared-distance metric)
// ---------------------------------------------------------------------------
//
// Power distance: pow(p, s) = |p - s|^2 - r_s^2
// Power bisector between sites s and t is a STRAIGHT LINE (unlike the
// hyperbolic arc of the additive Apollonius metric).
//
// Implementation: per-site half-plane clipping (O(n^3) total, fine for
// small n). Each cell is initialised to the bounding box and clipped by
// the n-1 half-planes defined by the power bisectors.

struct PowerVertex {
    Vec2 point;
    std::array<int, 3> sites{-1, -1, -1};  // 3 sites whose power is equal here
};

struct PowerEdge {
    int left_site  = -1;
    int right_site = -1;
    Vec2 a;   // segment endpoint A (world space)
    Vec2 b;   // segment endpoint B (world space)
};

struct PowerDiagram {
    std::string note;
    std::vector<int> active_sites;    // non-dominated sites
    std::vector<int> dominated_sites; // empty-cell sites
    std::vector<PowerVertex> vertices;
    std::vector<PowerEdge>   edges;
    std::vector<std::vector<Vec2>> cells; // size == points.size(); empty for dominated
};

PowerDiagram build_power_diagram(const std::vector<Vec2>& points,
                                 const std::vector<double>& radii,
                                 const Rect& bounds);

// ---------------------------------------------------------------------------
// Multiplicatively-weighted Voronoi (MW Voronoi / Apollonius circles)
// ---------------------------------------------------------------------------
//
// Distance:  d(p, s) = |p - s| / w_s,  where w_s = radii[s] > 0.
//
// Bisector between sites s (weight w_s) and t (weight w_t):
//   - w_s == w_t  => perpendicular bisector LINE through the midpoint.
//   - w_s != w_t  => CIRCLE with
//       center = (w_t^2 * s - w_s^2 * t) / (w_t^2 - w_s^2)
//       radius = (w_s * w_t * |s - t|) / |w_t^2 - w_s^2|
//     The site with the LARGER weight wins OUTSIDE the circle;
//     the site with the SMALLER weight wins INSIDE the circle.
//
// Cells can be non-convex and multiply-connected.  This implementation uses
// a grid-scan (rasterised per-pixel winner) renderer in the app layer;
// the engine here computes the analytical bisectors only.

struct MWBisector {
    int site_a = -1;
    int site_b = -1;
    bool is_line = false;      // true when weights are equal -> perp-bisector line
    // Line case:
    Vec2 line_mid{};           // midpoint of segment ab
    Vec2 line_normal{};        // direction (b - a), NOT normalised
    // Circle case (is_line == false):
    Vec2 circle_center{};
    double circle_radius = 0.0;
    int   inner_site = -1;     // smaller-weight site — wins INSIDE the circle
    int   outer_site = -1;     // larger-weight site  — wins OUTSIDE the circle
};

struct MWDiagram {
    std::string note;
    std::vector<int> active_sites;   // sites with w > 0
    std::vector<MWBisector> bisectors; // one per pair (a, b) with a < b
};

MWDiagram build_mw_voronoi(const std::vector<Vec2>& points,
                           const std::vector<double>& weights,
                           const Rect& bounds);

// ---------------------------------------------------------------------------
// k-th order Voronoi helper
// ---------------------------------------------------------------------------
//
// For a query point p and a set of sites, return the UNORDERED k-subset of
// nearest sites as a sorted ascending index vector.  For k=1 this reduces to
// the standard nearest-site lookup; for k>1 each unique subset defines one
// cell of the k-th order Voronoi diagram.
//
// Algorithm: compute squared distances to all sites, partial-sort to find the
// k smallest, sort those k indices ascending to form the canonical subset key.

struct HigherOrderCellHit {
    std::vector<int> subset;  // sorted ascending, size == k
};

HigherOrderCellHit k_nearest_subset(const Vec2& p,
                                    const std::vector<Vec2>& points,
                                    int k);

// ---------------------------------------------------------------------------
// Engine state (for incremental rebuilds)
// ---------------------------------------------------------------------------

struct ApolloniusEngineState {
    int site_count = 0;
    std::vector<bool> dominated;       // [site_count]: per-site domination flag
    std::vector<int> active_sites;     // indices of non-dominated sites (matches diagram.active_sites)
    std::vector<int> dominated_sites;  // matches diagram.dominated_sites
    std::vector<ApolloniusVertex> vertices;  // triple-tangent vertices (matches diagram.vertices)
    std::vector<ApolloniusEdge> edges;       // pair-edges post-visibility-filter (matches diagram.edges)
    std::string note;                        // matches diagram.note

    // For incremental rebuilds: cache of pair-curve polylines BEFORE the
    // visibility filter ran. Keyed by sorted (lo, hi) site pair, where lo
    // = min(left_site, right_site) and hi = max(...). The polyline is the
    // sequence of point_on_pair_curve(parameter[i]) for i in 0..samples-1,
    // INCLUDING points that the visibility filter ultimately rejected.
    //
    // The incremental insert path uses these cached raw polylines to skip
    // re-sampling the analytic curve when a pair survives between
    // consecutive insertions.
    std::map<std::pair<int,int>, std::vector<Vec2>> raw_pair_polylines;
};

// State-populating overload. When out_state is non-null, populates it with
// the engine's intermediate state (dominance, active sites, vertices, edges,
// AND the raw pair polylines before the visibility filter). When out_state
// is null, behaves identically to the no-state overload. Otherwise, the
// returned ApolloniusDiagram is byte-for-byte identical to what the no-state
// call would return.
ApolloniusDiagram build_apollonius_diagram(
    const std::vector<Vec2>& points,
    const std::vector<double>& radii,
    const Rect& bounds,
    int trace_resolution,
    ApolloniusArithmeticMode mode,
    ApolloniusPredicateAudit* audit,
    ApolloniusEngineState* out_state);

// ---------------------------------------------------------------------------
// Incremental analytic insert
// ---------------------------------------------------------------------------
//
// Given a prev_state describing a diagram for prev_state.site_count sites,
// produces the diagram for prev_state.site_count+1 sites by inserting the
// new site at index new_site_idx == prev_state.site_count.
//
// out_state is REQUIRED and receives the full next engine state (dominance,
// active/dominated sites, triple-tangent vertices, pair-edge polylines, and
// raw pair polyline cache).
//
// out_diagram is OPTIONAL — when non-null it receives the same vertices,
// edges, active_sites, dominated_sites, and note as *out_state.
//
// after_points and after_radii must have size == prev_state.site_count + 1,
// with the new site at index new_site_idx.
void apollonius_diagram_incremental_insert(
    const ApolloniusEngineState& prev_state,
    int new_site_idx,
    const std::vector<Vec2>& after_points,
    const std::vector<double>& after_radii,
    const Rect& bounds,
    int trace_resolution,
    ApolloniusArithmeticMode mode,
    ApolloniusEngineState* out_state,
    ApolloniusDiagram* out_diagram,
    ApolloniusPredicateAudit* audit);

}  // namespace fortune

// ---------------------------------------------------------------------------
// Template member body (must be in the header so each TU can instantiate it)
// ---------------------------------------------------------------------------

namespace fortune {

// Helper declared here to avoid pulling <sstream> into every TU.
// It's only called from sign_or_filter when audit_ is non-null.
inline std::string apollonius_engine_short_float(const double value, const int precision) {
    std::ostringstream out;
    out.setf(std::ios::fixed, std::ios::floatfield);
    out.precision(precision);
    out << value;
    return out.str();
}

template <typename IntervalFn>
int ApolloniusPredicateKernel::sign_or_filter(const std::string& label,
                                              const double fast,
                                              const double tolerance,
                                              IntervalFn interval_fn) {
    const auto sign_of = [](const double value) {
        return (value > 0.0) - (value < 0.0);
    };

    // Count close input predicates before choosing the evaluation tier. In
    // lazy-filtered mode they cannot enter the fast-accept branch below.
    // This is a diagnostic tolerance-band count, not an exact-zero verdict.
    if (audit_ && mode_ == ApolloniusArithmeticMode::lazy_filtered &&
        std::abs(fast) < tolerance * 0.1) {
        ++audit_->near_zero_predicates;
    }

    if (mode_ == ApolloniusArithmeticMode::analytic_double || std::abs(fast) > tolerance) {
        if (audit_) {
            ++audit_->fast_accepts;
        }
        return sign_of(fast);
    }

    if (audit_) {
        ++audit_->lazy_evaluations;
    }
    const MetricInterval interval = interval_fn();
    if (interval.lo > 0.0 || interval.hi < 0.0) {
        if (audit_) {
            audit_->record(label + " lazy interval resolved [" +
                           apollonius_engine_short_float(interval.lo, 3) + ", " +
                           apollonius_engine_short_float(interval.hi, 3) + "]");
        }
        return interval.lo > 0.0 ? 1 : -1;
    }

    if (audit_) {
        ++audit_->unresolved_fallbacks;
        audit_->record(label + " needs future exact fallback; using double sign " +
                       apollonius_engine_short_float(fast, 10));
    }
    return sign_of(fast);
}

}  // namespace fortune
