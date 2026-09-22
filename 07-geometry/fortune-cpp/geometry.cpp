#include "fortune/geometry.h"

#include <limits>
#include <sstream>

namespace fortune {

namespace {

std::string short_scalar(const double value, const int precision = 8) {
    std::ostringstream out;
    out.setf(std::ios::fixed, std::ios::floatfield);
    out.precision(precision);
    out << value;
    return out.str();
}

int sign_of(const double value) {
    return (value > 0.0) - (value < 0.0);
}

int sign_of(const long double value) {
    return (value > 0.0L) - (value < 0.0L);
}

long double orient_wide(const Vec2& a, const Vec2& b, const Vec2& c) {
    const long double bax = static_cast<long double>(b.x) - static_cast<long double>(a.x);
    const long double bay = static_cast<long double>(b.y) - static_cast<long double>(a.y);
    const long double cax = static_cast<long double>(c.x) - static_cast<long double>(a.x);
    const long double cay = static_cast<long double>(c.y) - static_cast<long double>(a.y);
    return bax * cay - bay * cax;
}

}  // namespace

FortunePredicateKernel::FortunePredicateKernel(const FortuneArithmeticMode mode, FortunePredicateAudit* audit)
    : mode_(mode), audit_(audit) {}

const char* fortune_arithmetic_mode_label(const FortuneArithmeticMode mode) {
    switch (mode) {
        case FortuneArithmeticMode::analytic_double:
            return "Analytic double";
        case FortuneArithmeticMode::lazy_exact:
            return "Lazy exact";
        case FortuneArithmeticMode::low_precision:
            return "Low precision";
    }
    return "Analytic double";
}

int FortunePredicateKernel::resolve_sign(const std::string& label,
                                         const double fast,
                                         const double tolerance,
                                         const long double wide) {
    // Only lazy_exact runs the tie policy; analytic_double and low_precision
    // both take the fast double sign straight away (low_precision wants its
    // coarse-double inaccuracy to actually surface, not be smoothed over).
    // Note what the policy is: `wide` is a long double re-derivation, and on
    // this compiler long double IS double, so it cannot overturn the fast
    // sign. The band is what matters -- it funnels every too-close-to-call
    // comparison through one decision, which buys consistency between call
    // sites, not precision. Only a genuine 0 is reported as a tie.
    if (mode_ != FortuneArithmeticMode::lazy_exact || std::abs(fast) > tolerance) {
        if (audit_) {
            ++audit_->fast_accepts;
        }
        return sign_of(fast);
    }

    if (audit_) {
        ++audit_->lazy_evaluations;
    }

    const int wide_sign = sign_of(wide);
    if (wide_sign != 0) {
        if (audit_) {
            ++audit_->wide_fallbacks;
            audit_->record(label + " inside the tolerance band; sign fixed by the tie policy (value " +
                           short_scalar(fast, 10) + ")");
        }
        return wide_sign;
    }

    if (audit_) {
        audit_->record(label + " is exactly 0 -- a genuine tie, not a rounding artefact");
    }
    return 0;
}

int FortunePredicateKernel::orientation_sign(const Vec2& a,
                                            const Vec2& b,
                                            const Vec2& c,
                                            const std::string& label) {
    const Vec2 ab = b - a;
    const Vec2 ac = c - a;
    const double fast = cross(ab, ac);
    const double scale = std::max({1.0, std::abs(ab.x), std::abs(ab.y), std::abs(ac.x), std::abs(ac.y)});
    const double tolerance = scale * scale * 128.0 * std::numeric_limits<double>::epsilon();
    return resolve_sign(label.empty() ? "orient2d" : label, fast, tolerance, orient_wide(a, b, c));
}

bool FortunePredicateKernel::scalar_less(const double lhs, const double rhs, const std::string& label) {
    const double fast = lhs - rhs;
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    const double tolerance = scale * 64.0 * std::numeric_limits<double>::epsilon();
    const long double wide = static_cast<long double>(lhs) - static_cast<long double>(rhs);
    return resolve_sign(label.empty() ? "scalar <" : label, fast, tolerance, wide) < 0;
}

bool FortunePredicateKernel::scalar_less_equal(const double lhs, const double rhs, const std::string& label) {
    const double fast = lhs - rhs;
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    const double tolerance = scale * 64.0 * std::numeric_limits<double>::epsilon();
    const long double wide = static_cast<long double>(lhs) - static_cast<long double>(rhs);
    return resolve_sign(label.empty() ? "scalar <=" : label, fast, tolerance, wide) <= 0;
}

double parabola_y(const Vec2& focus, const double directrix, const double x) {
    const double denom = 2.0 * (focus.y - directrix);
    if (std::abs(denom) <= kEpsilon) {
        return directrix;
    }
    const double dx = x - focus.x;
    return (dx * dx) / denom + (focus.y + directrix) * 0.5;
}

std::vector<double> parabola_intersections(const Vec2& left_focus, const Vec2& right_focus, const double directrix) {
    if (std::abs(left_focus.y - directrix) <= kEpsilon && std::abs(right_focus.y - directrix) <= kEpsilon) {
        return {(left_focus.x + right_focus.x) * 0.5};
    }
    if (std::abs(left_focus.y - directrix) <= kEpsilon) {
        return {left_focus.x};
    }
    if (std::abs(right_focus.y - directrix) <= kEpsilon) {
        return {right_focus.x};
    }
    // Equal-height foci: the two congruent parabolas meet exactly on the
    // vertical bisector. Use a RELATIVE tolerance, not the absolute
    // kEpsilon -- grids, cocircular rings and post-rounding rows leave a
    // near-equal-y band where the quadratic's leading coefficient
    // (d0 - d1) is ~0 and the roots blow up, sending the beachline search
    // down the wrong subtree.
    const double yscale = 1.0 + std::abs(left_focus.y) + std::abs(right_focus.y);
    if (std::abs(left_focus.y - right_focus.y) <= 1.0e-7 * yscale) {
        return {(left_focus.x + right_focus.x) * 0.5};
    }

    const double d0 = 1.0 / (2.0 * (left_focus.y - directrix));
    const double d1 = 1.0 / (2.0 * (right_focus.y - directrix));

    const double a = d0 - d1;
    const double b = 2.0 * (right_focus.x * d1 - left_focus.x * d0);
    const double c =
        (left_focus.x * left_focus.x + left_focus.y * left_focus.y - directrix * directrix) * d0 -
        (right_focus.x * right_focus.x + right_focus.y * right_focus.y - directrix * directrix) * d1;

    // Near-zero leading coefficient: the quadratic is effectively linear.
    // Test RELATIVE to b,c -- an absolute kEpsilon leaves an ill-
    // conditioned band (|a| tiny but > 1e-9) where -b/2a explodes; that
    // band is exactly the grid/ring near-equal-y rows. Fall back to the
    // stable linear root there instead.
    if (std::abs(a) <= 1.0e-12 * (1.0 + std::abs(b) + std::abs(c))) {
        if (std::abs(b) <= kEpsilon) {
            return {};
        }
        return {-c / b};
    }

    const double discriminant = b * b - 4.0 * a * c;
    if (discriminant < -kEpsilon) {
        return {};
    }
    if (std::abs(discriminant) <= kEpsilon) {
        return {-b / (2.0 * a)};
    }

    const double root = std::sqrt(std::max(0.0, discriminant));
    double x0 = (-b - root) / (2.0 * a);
    double x1 = (-b + root) / (2.0 * a);
    if (x1 < x0) {
        std::swap(x0, x1);
    }
    return {x0, x1};
}

double choose_breakpoint_x(const Vec2& left_focus, const Vec2& right_focus, const double directrix) {
    const auto roots = parabola_intersections(left_focus, right_focus, directrix);
    if (roots.empty()) {
        return (left_focus.x + right_focus.x) * 0.5;
    }
    if (roots.size() == 1) {
        return roots.front();
    }

    const double separation = std::max(1.0e-5, std::abs(roots[1] - roots[0]) * 0.25);
    for (const double candidate : roots) {
        const double lx = candidate - separation;
        const double rx = candidate + separation;
        const double left_before = parabola_y(left_focus, directrix, lx);
        const double right_before = parabola_y(right_focus, directrix, lx);
        const double left_after = parabola_y(left_focus, directrix, rx);
        const double right_after = parabola_y(right_focus, directrix, rx);
        if (left_before <= right_before + kRangeEpsilon && right_after <= left_after + kRangeEpsilon) {
            return candidate;
        }
    }

    return roots.front();
}

Vec2 breakpoint_point(const Vec2& left_focus, const Vec2& right_focus, const double directrix) {
    const double x = choose_breakpoint_x(left_focus, right_focus, directrix);
    return {x, parabola_y(left_focus, directrix, x)};
}

std::optional<Circumcircle> circumcircle_from_sites(const Vec2& a,
                                                    const Vec2& b,
                                                    const Vec2& c,
                                                    FortunePredicateKernel* predicates) {
    if (predicates && predicates->orientation_sign(a, b, c, "circumcircle collinearity") == 0) {
        return std::nullopt;
    }

    const long double ax = static_cast<long double>(a.x);
    const long double ay = static_cast<long double>(a.y);
    const long double bx = static_cast<long double>(b.x);
    const long double by = static_cast<long double>(b.y);
    const long double cx = static_cast<long double>(c.x);
    const long double cy = static_cast<long double>(c.y);
    const long double d = 2.0L * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (!predicates && std::abs(static_cast<double>(d)) <= kEpsilon) {
        return std::nullopt;
    }
    if (d == 0.0L) {
        return std::nullopt;
    }

    const long double a_sq = ax * ax + ay * ay;
    const long double b_sq = bx * bx + by * by;
    const long double c_sq = cx * cx + cy * cy;
    const long double ux = (a_sq * (by - cy) + b_sq * (cy - ay) + c_sq * (ay - by)) / d;
    const long double uy = (a_sq * (cx - bx) + b_sq * (ax - cx) + c_sq * (bx - ax)) / d;

    if (!std::isfinite(static_cast<double>(ux)) || !std::isfinite(static_cast<double>(uy))) {
        return std::nullopt;
    }

    const Vec2 center{static_cast<double>(ux), static_cast<double>(uy)};
    const double radius = length(center - a);
    if (!std::isfinite(radius)) {
        return std::nullopt;
    }
    return Circumcircle{center, radius, center.y - radius};
}

bool breakpoints_converge(const Vec2& left, const Vec2& mid, const Vec2& right, const double directrix) {
    const auto circle = circumcircle_from_sites(left, mid, right);
    if (!circle || circle->event_y >= directrix - kRangeEpsilon) {
        return false;
    }

    const double gap_to_event = directrix - circle->event_y;
    const double delta = std::max(1.0e-6, std::min(1.0e-4, gap_to_event * 0.25));
    const double d0 = directrix - delta;
    const double d1 = directrix - 2.0 * delta;
    if (d1 <= circle->event_y + kEpsilon) {
        return false;
    }

    const double left0 = choose_breakpoint_x(left, mid, d0);
    const double right0 = choose_breakpoint_x(mid, right, d0);
    const double left1 = choose_breakpoint_x(left, mid, d1);
    const double right1 = choose_breakpoint_x(mid, right, d1);

    const double gap0 = std::abs(right0 - left0);
    const double gap1 = std::abs(right1 - left1);
    if (gap1 + 1.0e-6 < gap0) {
        return true;
    }

    return cross(mid - left, right - left) < 0.0;
}

std::vector<EdgeTrace> weld_edge_vertices(std::vector<EdgeTrace> edges, const double tol) {
    if (tol <= 0.0) {
        return edges;
    }
    const double tol_sq = tol * tol;

    // Cluster every recorded endpoint. Edge count is O(sites) here, so a
    // plain O(P^2) nearest-cluster pass is cheap and fully deterministic.
    struct Cluster {
        Vec2 sum;
        int count = 0;
        Vec2 rep() const { return sum / static_cast<double>(count); }
    };
    std::vector<Cluster> clusters;
    const auto assign = [&](const Vec2& p) -> int {
        int best = -1;
        double best_d = tol_sq;
        for (std::size_t i = 0; i < clusters.size(); ++i) {
            const double d = length_sq(clusters[i].rep() - p);
            if (d <= best_d) {
                best_d = d;
                best = static_cast<int>(i);
            }
        }
        if (best < 0) {
            clusters.push_back(Cluster{p, 1});
            return static_cast<int>(clusters.size()) - 1;
        }
        clusters[best].sum += p;
        clusters[best].count += 1;
        return best;
    };

    // Two passes: first form the clusters, then rewrite endpoints to the
    // (now stable) representative so every edge touching one cocircular
    // vertex lands on the exact same point.
    std::vector<int> start_cluster(edges.size(), -1);
    std::vector<int> end_cluster(edges.size(), -1);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        if (edges[i].start) start_cluster[i] = assign(*edges[i].start);
        if (edges[i].end) end_cluster[i] = assign(*edges[i].end);
    }

    std::vector<EdgeTrace> result;
    result.reserve(edges.size());
    for (std::size_t i = 0; i < edges.size(); ++i) {
        EdgeTrace e = edges[i];
        if (start_cluster[i] >= 0) e.start = clusters[start_cluster[i]].rep();
        if (end_cluster[i] >= 0) e.end = clusters[end_cluster[i]].rep();
        // Drop micro-edges: both ends welded into the same cluster (the
        // spurious zero-length segments a sequential sweep emits around a
        // cocircular vertex). Half-open edges (one endpoint) are kept --
        // they radiate out to the box.
        if (e.start && e.end && length_sq(*e.start - *e.end) <= tol_sq) {
            continue;
        }
        result.push_back(e);
    }
    return result;
}

std::optional<Vec2> ray_box_intersection(const Vec2& origin, const Vec2& direction, const Rect& bounds) {
    if (length_sq(direction) <= kEpsilon) {
        return std::nullopt;
    }

    double t_enter = 0.0;
    double t_exit = std::numeric_limits<double>::infinity();

    const auto clip_axis = [&](const double o, const double d, const double min_v, const double max_v) -> bool {
        if (std::abs(d) <= kEpsilon) {
            return o >= min_v - kEpsilon && o <= max_v + kEpsilon;
        }

        double a = (min_v - o) / d;
        double b = (max_v - o) / d;
        if (a > b) {
            std::swap(a, b);
        }
        t_enter = std::max(t_enter, a);
        t_exit = std::min(t_exit, b);
        return t_enter <= t_exit + kEpsilon;
    };

    if (!clip_axis(origin.x, direction.x, bounds.min_x, bounds.max_x)) {
        return std::nullopt;
    }
    if (!clip_axis(origin.y, direction.y, bounds.min_y, bounds.max_y)) {
        return std::nullopt;
    }

    const double t = t_enter > kEpsilon ? t_enter : t_exit;
    if (t < kEpsilon || !std::isfinite(t)) {
        return std::nullopt;
    }

    return origin + direction * t;
}

std::optional<std::pair<Vec2, Vec2>> clip_segment_to_box(const Vec2& a, const Vec2& b, const Rect& bounds) {
    double t0 = 0.0;
    double t1 = 1.0;
    const Vec2 d = b - a;

    const auto clip = [&](const double p, const double q) -> bool {
        if (std::abs(p) <= kEpsilon) {
            return q >= 0.0;
        }
        const double r = q / p;
        if (p < 0.0) {
            t0 = std::max(t0, r);
        } else {
            t1 = std::min(t1, r);
        }
        return t0 <= t1 + kEpsilon;
    };

    if (!clip(-d.x, a.x - bounds.min_x)) return std::nullopt;
    if (!clip(d.x, bounds.max_x - a.x)) return std::nullopt;
    if (!clip(-d.y, a.y - bounds.min_y)) return std::nullopt;
    if (!clip(d.y, bounds.max_y - a.y)) return std::nullopt;

    return std::pair<Vec2, Vec2>{a + d * t0, a + d * t1};
}

std::optional<std::pair<double, double>> ray_box_clip(const Vec2& origin,
                                                      const Vec2& direction,
                                                      const Rect& bounds) {
    double t_enter = -std::numeric_limits<double>::infinity();
    double t_exit = std::numeric_limits<double>::infinity();

    if (std::abs(direction.x) > kEpsilon) {
        double t1 = (bounds.min_x - origin.x) / direction.x;
        double t2 = (bounds.max_x - origin.x) / direction.x;
        if (t1 > t2) std::swap(t1, t2);
        t_enter = std::max(t_enter, t1);
        t_exit = std::min(t_exit, t2);
    } else if (origin.x < bounds.min_x - kEpsilon || origin.x > bounds.max_x + kEpsilon) {
        return std::nullopt;
    }

    if (std::abs(direction.y) > kEpsilon) {
        double t1 = (bounds.min_y - origin.y) / direction.y;
        double t2 = (bounds.max_y - origin.y) / direction.y;
        if (t1 > t2) std::swap(t1, t2);
        t_enter = std::max(t_enter, t1);
        t_exit = std::min(t_exit, t2);
    } else if (origin.y < bounds.min_y - kEpsilon || origin.y > bounds.max_y + kEpsilon) {
        return std::nullopt;
    }

    if (t_enter >= t_exit) {
        return std::nullopt;
    }
    return std::pair<double, double>{t_enter, t_exit};
}

EdgeTrace finalize_edge_to_box(EdgeTrace edge,
                               const Vec2& left_site,
                               const Vec2& right_site,
                               const std::optional<Vec2>& third_site,
                               const Rect& bounds) {
    if (edge.start && edge.end) {
        // Genuine swept segment -- both endpoints came from circle events.
        // Only clip to the box; do NOT fabricate.
        const auto clipped = clip_segment_to_box(*edge.start, *edge.end, bounds);
        if (clipped) {
            edge.start = clipped->first;
            edge.end = clipped->second;
        } else {
            edge.start.reset();
            edge.end.reset();
        }
        edge.reconstructed = false;
        return edge;
    }

    // From here down the sweep never finished this edge, so anything we
    // draw is reconstructed from the ideal site bisector, not the sweep.
    edge.reconstructed = true;

    const Vec2 midpoint = (left_site + right_site) * 0.5;
    Vec2 dir = normalize(perp(right_site - left_site));
    if (length_sq(dir) <= kEpsilon) {
        edge.start.reset();
        edge.end.reset();
        return edge;
    }

    if (!edge.start && !edge.end) {
        // Doubly unbounded -- clip the full bisector line.
        const auto clip = ray_box_clip(midpoint, dir, bounds);
        if (!clip) {
            edge.start.reset();
            edge.end.reset();
            return edge;
        }
        edge.start = midpoint + dir * clip->first;
        edge.end = midpoint + dir * clip->second;
        return edge;
    }

    // Half-open. CRFortune outward_dir: shoot away from the third
    // circumcircle site; fall back to the midpoint side if unknown.
    const Vec2 vv = edge.start ? *edge.start : *edge.end;
    if (third_site) {
        if (dot(dir, *third_site - vv) > 0.0) {
            dir = dir * -1.0;
        }
    } else if (dot(dir, midpoint - vv) < 0.0) {
        dir = dir * -1.0;
    }

    const auto clip = ray_box_clip(vv, dir, bounds);
    if (!clip) {
        edge.start.reset();
        edge.end.reset();
        return edge;
    }
    const double t_exit = clip->second;
    if (t_exit <= 0.0) {
        edge.start.reset();
        edge.end.reset();
        return edge;
    }
    const double t_start = clip->first > 0.0 ? clip->first : 0.0;
    if (t_start >= t_exit) {
        edge.start.reset();
        edge.end.reset();
        return edge;
    }

    const Vec2 far = vv + dir * t_exit;
    if (t_start <= 1.0e-9) {
        // vv is inside the box -- just fill in the missing endpoint.
        if (!edge.start) {
            edge.start = far;
        } else {
            edge.end = far;
        }
    } else {
        // vv is outside the box -- replace both endpoints with the
        // visible (box-clipped) segment.
        const Vec2 near = vv + dir * t_start;
        if (!edge.start) {
            edge.start = far;
            edge.end = near;
        } else {
            edge.start = near;
            edge.end = far;
        }
    }
    return edge;
}

}  // namespace fortune
