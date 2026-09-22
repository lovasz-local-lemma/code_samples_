#include "fortune/fortune_engine.h"
#include "fortune/lazy_scalar.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace fortune {

namespace {

std::vector<Vec2> clip_polygon_half_plane(const std::vector<Vec2>& polygon, const Vec2& mid, const Vec2& normal) {
    std::vector<Vec2> result;
    if (polygon.empty()) {
        return result;
    }

    const auto inside = [&](const Vec2& p) { return dot(p - mid, normal) <= kRangeEpsilon; };
    const auto intersection = [&](const Vec2& a, const Vec2& b) {
        const Vec2 ab = b - a;
        const double denom = dot(ab, normal);
        if (std::abs(denom) <= kEpsilon) {
            return a;
        }
        const double t = dot(mid - a, normal) / denom;
        return a + ab * t;
    };

    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const Vec2 current = polygon[i];
        const Vec2 previous = polygon[(i + polygon.size() - 1) % polygon.size()];
        const bool current_inside = inside(current);
        const bool previous_inside = inside(previous);

        if (current_inside) {
            if (!previous_inside) {
                result.push_back(intersection(previous, current));
            }
            result.push_back(current);
        } else if (previous_inside) {
            result.push_back(intersection(previous, current));
        }
    }

    return result;
}

std::string format_vec2(const Vec2& point) {
    std::ostringstream out;
    out.setf(std::ios::fixed, std::ios::floatfield);
    out.precision(4);
    out << "(" << point.x << ", " << point.y << ")";
    return out.str();
}

double segment_distance_sq(const Vec2& a, const Vec2& b, const Vec2& p) {
    const Vec2 ab = b - a;
    const double denom = length_sq(ab);
    if (denom <= kEpsilon) {
        return length_sq(p - a);
    }
    const double t = clamp(dot(p - a, ab) / denom, 0.0, 1.0);
    const Vec2 closest = a + ab * t;
    return length_sq(p - closest);
}

void append_unique_point(std::vector<Vec2>& points, const Vec2& point, const double eps = 1.0e-8) {
    for (const auto& existing : points) {
        if (length_sq(existing - point) <= eps * eps) {
            return;
        }
    }
    points.push_back(point);
}

bool point_on_polygon_boundary(const std::vector<Vec2>& polygon, const Vec2& point) {
    if (polygon.size() < 2) {
        return false;
    }
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const Vec2 a = polygon[i];
        const Vec2 b = polygon[(i + 1) % polygon.size()];
        if (segment_distance_sq(a, b, point) <= 1.0e-12) {
            return true;
        }
    }
    return false;
}

bool point_in_polygon(const std::vector<Vec2>& polygon, const Vec2& point) {
    if (polygon.empty()) {
        return false;
    }
    if (point_on_polygon_boundary(polygon, point)) {
        return true;
    }

    bool inside = false;
    for (std::size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
        const Vec2& a = polygon[i];
        const Vec2& b = polygon[j];
        const bool intersects =
            ((a.y > point.y) != (b.y > point.y)) &&
            (point.x < (b.x - a.x) * (point.y - a.y) / std::max(std::abs(b.y - a.y), kEpsilon) + a.x);
        if (intersects) {
            inside = !inside;
        }
    }
    return inside;
}

std::vector<Vec2> polygon_line_contacts(const std::vector<Vec2>& polygon, const Vec2& mid, const Vec2& normal) {
    std::vector<Vec2> contacts;
    if (polygon.empty()) {
        return contacts;
    }

    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const Vec2 a = polygon[i];
        const Vec2 b = polygon[(i + 1) % polygon.size()];
        const double sa = dot(a - mid, normal);
        const double sb = dot(b - mid, normal);

        if (std::abs(sa) <= 1.0e-8) {
            append_unique_point(contacts, a);
        }
        if (std::abs(sb) <= 1.0e-8) {
            append_unique_point(contacts, b);
        }

        if (sa * sb < -1.0e-12) {
            const double t = sa / (sa - sb);
            append_unique_point(contacts, a + (b - a) * t);
        } else if (std::abs(sa) <= 1.0e-8 && std::abs(sb) <= 1.0e-8) {
            append_unique_point(contacts, a);
            append_unique_point(contacts, b);
        }
    }

    return contacts;
}

std::optional<std::pair<Vec2, Vec2>> shared_segment_from_cells(const CellPolygon& left_cell,
                                                               const CellPolygon& right_cell,
                                                               const Vec2& left_site,
                                                               const Vec2& right_site) {
    if (left_cell.polygon.empty() || right_cell.polygon.empty()) {
        return std::nullopt;
    }

    const Vec2 normal = right_site - left_site;
    const Vec2 tangent = normalize(perp(normal));
    if (length_sq(tangent) <= kEpsilon) {
        return std::nullopt;
    }
    const Vec2 mid = (left_site + right_site) * 0.5;

    std::vector<Vec2> candidates = polygon_line_contacts(left_cell.polygon, mid, normal);
    for (const auto& point : polygon_line_contacts(right_cell.polygon, mid, normal)) {
        append_unique_point(candidates, point);
    }

    std::vector<Vec2> shared;
    for (const auto& point : candidates) {
        if (!point_in_polygon(left_cell.polygon, point) || !point_in_polygon(right_cell.polygon, point)) {
            continue;
        }
        append_unique_point(shared, point);
    }

    if (shared.size() < 2) {
        return std::nullopt;
    }

    std::sort(shared.begin(), shared.end(), [&](const Vec2& a, const Vec2& b) {
        return dot(a, tangent) < dot(b, tangent);
    });

    if (length_sq(shared.back() - shared.front()) <= 1.0e-12) {
        return std::nullopt;
    }
    return std::pair<Vec2, Vec2>{shared.front(), shared.back()};
}

std::vector<double> monotone_beachline_boundaries(const std::vector<Beachline::LeafNode*>& leaves,
                                                  const std::vector<Site>& sites,
                                                  const Rect& bounds,
                                                  const double directrix) {
    std::vector<double> boundaries;
    boundaries.reserve(leaves.size() + 1);
    boundaries.push_back(bounds.min_x);
    for (std::size_t i = 0; i + 1 < leaves.size(); ++i) {
        const auto& left = sites[leaves[i]->site_id].point;
        const auto& right = sites[leaves[i + 1]->site_id].point;
        double x = choose_breakpoint_x(left, right, directrix);
        x = clamp(x, bounds.min_x, bounds.max_x);
        x = std::max(x, boundaries.back());
        boundaries.push_back(x);
    }
    boundaries.push_back(std::max(bounds.max_x, boundaries.back()));
    return boundaries;
}

std::vector<int> active_edge_ids_from_breakpoints(const std::vector<SnapshotBreakpoint>& breakpoints) {
    std::vector<int> active;
    for (const auto& breakpoint : breakpoints) {
        if (breakpoint.edge_id < 0) {
            continue;
        }
        if (std::find(active.begin(), active.end(), breakpoint.edge_id) == active.end()) {
            active.push_back(breakpoint.edge_id);
        }
    }
    return active;
}

std::string ordered_leaf_summary(const std::vector<Beachline::LeafNode*>& leaves) {
    std::ostringstream out;
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        if (i > 0) {
            out << " -> ";
        }
        out << "N" << leaves[i]->id << ":S" << leaves[i]->site_id;
    }
    return out.str();
}

std::string active_edge_summary(const std::vector<int>& edge_ids) {
    if (edge_ids.empty()) {
        return "none";
    }
    std::ostringstream out;
    for (std::size_t i = 0; i < edge_ids.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << "#" << edge_ids[i];
    }
    return out.str();
}

}  // namespace

bool FortuneEngine::QueueLess::operator()(const QueueEntry& a,
                                          const QueueEntry& b) const {
    if (cfg && !cfg->transitive_ordering) {
        // NAIVE (pre-331201a): epsilon near() equality -> NOT a strict
        // weak ordering (intransitive). Intentionally reproduced.
        if (!near(a.y, b.y)) return a.y < b.y;
        if (a.kind != b.kind) return a.kind == EventKind::circle;
        if (!near(a.x, b.x)) return a.x > b.x;
        return a.id > b.id;
    }
    // std::priority_queue is a max-heap: the element processed first must
    // compare "greatest". a has lower priority than b iff a is processed
    // AFTER b, i.e. b is processed before a. Delegating to the single
    // canonical total order keeps the heap, pending_events() and
    // queued_events() in lockstep and (unlike the old near()-based form)
    // makes this a valid strict-weak-ordering.
    return event_processed_before(b.y, b.kind, b.x, b.id, a.y, a.kind, a.x, a.id);
}

FortuneEngine::FortuneEngine(std::vector<Vec2> points) {
    reset(std::move(points));
}

const std::vector<EdgeTrace>& FortuneEngine::edges() const {
    return finalized_ && reference_final_enabled_ ? finalized_edges_ : edges_;
}

void FortuneEngine::reset(std::vector<Vec2> points) {
    beachline_.clear();
    beachline_.set_observer(this);
    queue_ = std::priority_queue<QueueEntry, std::vector<QueueEntry>, QueueLess>(
        QueueLess{&robustness_});
    events_.set_unsafe(!robustness_.safe_event_storage);
    events_.clear();
    processed_events_.clear();
    edges_.clear();
    finalized_edges_.clear();
    final_cells_.clear();
    snapshots_.clear();
    predicate_audit_.clear();
    finalized_ = false;
    current_event_id_ = -1;

    sites_.clear();
    for (std::size_t i = 0; i < points.size(); ++i) {
        sites_.push_back(Site{static_cast<int>(i), points[i]});
    }

    // Low Precision: snap the *working* sites onto a coarse decimal grid.
    // The app keeps its exact points (the O(n^2) reference is built from
    // those), so this divergence is exactly what the diagnostic viz show.
    if (arithmetic_mode_ == FortuneArithmeticMode::low_precision) {
        for (auto& site : sites_) {
            site.point.x = round_fixed_digits(site.point.x, low_precision_digits_);
            site.point.y = round_fixed_digits(site.point.y, low_precision_digits_);
        }
    }

    if (sites_.empty()) {
        bounds_ = Rect{-1.0, -1.0, 1.0, 1.0};
    } else {
        double min_x = sites_.front().point.x;
        double max_x = min_x;
        double min_y = sites_.front().point.y;
        double max_y = min_y;
        for (const auto& site : sites_) {
            min_x = std::min(min_x, site.point.x);
            max_x = std::max(max_x, site.point.x);
            min_y = std::min(min_y, site.point.y);
            max_y = std::max(max_y, site.point.y);
        }
        const double span = std::max({max_x - min_x, max_y - min_y, 0.5});
        const double pad = span * 0.3 + 0.1;
        bounds_ = Rect{min_x - pad, min_y - pad, max_x + pad, max_y + pad};
    }

    directrix_ = bounds_.max_y + 0.1;

    for (const auto& site : sites_) {
        push_site_event(site.id);
    }

    capture("initial state");
}

int FortuneEngine::push_event(EventRecord event) {
    event.id = static_cast<int>(events_.size());
    events_.push_back(event);
    queue_.push(QueueEntry{event.point.y, event.point.x, event.kind, event.id});
    return event.id;
}

int FortuneEngine::push_site_event(const int site_id) {
    EventRecord event;
    event.kind = EventKind::site;
    event.point = sites_[site_id].point;
    event.site_id = site_id;
    return push_event(event);
}

int FortuneEngine::push_circle_event(const int leaf_id, const Circumcircle& circle) {
    EventRecord event;
    event.kind = EventKind::circle;
    event.point = Vec2{circle.center.x, circle.event_y};
    event.leaf_id = leaf_id;
    event.center = circle.center;
    event.radius = circle.radius;
    return push_event(event);
}

int FortuneEngine::create_edge(const int left_site, const int right_site) {
    const int edge_id = static_cast<int>(edges_.size());
    edges_.push_back(EdgeTrace{edge_id, left_site, right_site, std::nullopt, std::nullopt});
    return edge_id;
}

void FortuneEngine::complete_edge(const int edge_id, const Vec2& point) {
    if (edge_id < 0 || edge_id >= static_cast<int>(edges_.size())) {
        return;
    }
    auto& edge = edges_[edge_id];
    if (!edge.start) {
        edge.start = point;
        return;
    }
    if (!edge.end) {
        edge.end = point;
    }
}

void FortuneEngine::invalidate_circle_event(Beachline::LeafNode* leaf) {
    if (!leaf || leaf->circle_event < 0) {
        return;
    }
    const int event_id = leaf->circle_event;
    if (leaf->circle_event < static_cast<int>(events_.size())) {
        events_[leaf->circle_event].valid = false;
    }
    leaf->circle_event = -1;
    if (verbose_) {
        capture("circle event invalidated",
                {leaf->id},
                {},
                {leaf->site_id},
                {"arc " + std::to_string(leaf->id) + " (site " + std::to_string(leaf->site_id) +
                 ") dropped pending circle event #" + std::to_string(event_id)});
    }
}

void FortuneEngine::check_circle_event(Beachline::LeafNode* left,
                                       Beachline::LeafNode* mid,
                                       Beachline::LeafNode* right) {
    if (!left || !mid || !right) {
        if (verbose_) {
            capture("circle check skipped", {}, {}, {}, {"triple incomplete"});
        }
        return;
    }
    if (left->site_id == mid->site_id || mid->site_id == right->site_id || left->site_id == right->site_id) {
        if (verbose_) {
            capture("circle check rejected",
                    {left->id, mid->id, right->id},
                    {},
                    {left->site_id, mid->site_id, right->site_id},
                    {"duplicate site ids in triple"});
        }
        return;
    }

    FortunePredicateKernel predicates(arithmetic_mode_, &predicate_audit_);
    const int audit_lazy_before = predicate_audit_.lazy_evaluations;
    const int audit_wide_before = predicate_audit_.wide_fallbacks;
    const std::size_t audit_notes_before = predicate_audit_.notes.size();
    const Vec2& left_point = sites_[left->site_id].point;
    const Vec2& mid_point = sites_[mid->site_id].point;
    const Vec2& right_point = sites_[right->site_id].point;
    const int orientation = predicates.orientation_sign(
        left_point, mid_point, right_point, "circle event orientation S" + std::to_string(left->site_id) + "/S" +
                                                std::to_string(mid->site_id) + "/S" + std::to_string(right->site_id));
    const double orientation_value = cross(mid_point - left_point, right_point - left_point);
    const auto circle = circumcircle_from_sites(left_point, mid_point, right_point, &predicates);
    if (!circle) {
        if (verbose_) {
            capture("circle check rejected",
                    {left->id, mid->id, right->id},
                    {},
                    {left->site_id, mid->site_id, right->site_id},
                    {"sites are collinear"});
        }
        return;
    }
    if (!predicates.scalar_less(circle->event_y, directrix_ - kRangeEpsilon, "circle bottom below directrix")) {
        if (verbose_) {
            capture("circle check rejected",
                    {left->id, mid->id, right->id},
                    {},
                    {left->site_id, mid->site_id, right->site_id},
                    {"circle bottom " + std::to_string(circle->event_y) +
                     " is not below directrix " + std::to_string(directrix_)});
        }
        return;
    }
    const bool removes_middle =
        arithmetic_mode_ != FortuneArithmeticMode::lazy_exact ? orientation_value < -kRangeEpsilon : orientation < 0;
    if (!removes_middle) {
        if (verbose_) {
            capture("circle check rejected",
                    {left->id, mid->id, right->id},
                    {},
                    {left->site_id, mid->site_id, right->site_id},
                    {"orientation sign " + std::to_string(orientation) + " does not produce a disappearing middle arc"});
        }
        return;
    }

    invalidate_circle_event(mid);
    mid->circle_event = push_circle_event(mid->id, *circle);
    {
        const int lazy_delta = predicate_audit_.lazy_evaluations - audit_lazy_before;
        const int wide_delta = predicate_audit_.wide_fallbacks - audit_wide_before;
        auto& ev = events_[mid->circle_event];
        ev.lazy_derived = (lazy_delta > 0 || wide_delta > 0);
        // Entered the lazy path more times than wide could resolve =>
        // at least one predicate was exactly tied even at wide precision.
        ev.lazy_unresolved = (lazy_delta - wide_delta) > 0;
        ev.lazy_scheduled_step = static_cast<int>(processed_events_.size());
        std::ostringstream detail;
        detail << "vertex = circumcenter(S" << left->site_id << ", S" << mid->site_id << ", S"
               << right->site_id << ")\n"
               << "S" << left->site_id << " " << format_vec2(left_point) << "\n"
               << "S" << mid->site_id << " " << format_vec2(mid_point) << "\n"
               << "S" << right->site_id << " " << format_vec2(right_point) << "\n"
               << "center " << format_vec2(circle->center) << "  r=" << circle->radius << "\n"
               << "event y=" << circle->event_y << "\n";
        if (ev.lazy_unresolved) {
            detail << "EXACTLY TIED (degenerate): the value came out exactly 0, so no tie policy "
                      "can give it a side ("
                   << lazy_delta << " lazy, " << wide_delta << " wide)";
            for (std::size_t i = audit_notes_before; i < predicate_audit_.notes.size(); ++i) {
                detail << "\n  - " << predicate_audit_.notes[i];
            }
        } else if (ev.lazy_derived) {
            detail << "LAZY (band-resolved): the value fell inside the tolerance band, so the tie "
                      "policy fixed its sign -- consistency, not extra precision (long double is "
                      "double here) ("
                   << lazy_delta << " lazy, " << wide_delta << " wide)";
            for (std::size_t i = audit_notes_before; i < predicate_audit_.notes.size(); ++i) {
                detail << "\n  - " << predicate_audit_.notes[i];
            }
        } else {
            detail << "collapsed: predicate resolved by a confident double sign";
        }

      if (arithmetic_mode_ != FortuneArithmeticMode::lazy_exact) {
        // Analytic Double / Low Precision: the value is collapsed to a
        // plain number the instant it is computed -- no formula is kept.
        const bool low_prec = arithmetic_mode_ == FortuneArithmeticMode::low_precision;
        const double grid_h = low_precision_step(low_precision_digits_);
        ev.lazy_analytic = false;
        ev.lazy_has_radical = false;
        ev.lazy_forced = 0;
        // Low Precision: the inputs are known only to +/- h/2. Rather than
        // a flat "everything is uncertain by h" (which made every halo
        // saturate while the diagram still looked right), propagate that
        // input box through the circumcircle one coordinate at a time.
        // A well-conditioned triple barely moves -> tiny width -> small
        // halo (matches "the result still looks about right"); a near-
        // degenerate triple (tiny circumcircle determinant) explodes ->
        // big halo, and that is exactly where the divergence and the
        // cocircular clusters are. Analytic Double stays exact -> 0.
        double low_prec_width = 0.0;
        if (low_prec) {
            const double half = 0.5 * grid_h;
            const Vec2 base[3] = {left_point, mid_point, right_point};
            for (int p = 0; p < 3 && low_prec_width < 1.0e6; ++p) {
                for (int axis = 0; axis < 2; ++axis) {
                    Vec2 lo[3] = {base[0], base[1], base[2]};
                    Vec2 hi[3] = {base[0], base[1], base[2]};
                    double& lo_c = axis == 0 ? lo[p].x : lo[p].y;
                    double& hi_c = axis == 0 ? hi[p].x : hi[p].y;
                    lo_c -= half;
                    hi_c += half;
                    const auto clo = circumcircle_from_sites(lo[0], lo[1], lo[2], nullptr);
                    const auto chi = circumcircle_from_sites(hi[0], hi[1], hi[2], nullptr);
                    if (clo && chi) {
                        low_prec_width += std::abs(chi->event_y - clo->event_y);
                    } else {
                        // Perturbation tipped the triple collinear: the
                        // circumcenter ran off to infinity -> unbounded.
                        low_prec_width = 1.0e6;
                    }
                }
            }
        }
        ev.lazy_interval_width = low_prec_width;
        ev.lazy_atoms.clear();
        ev.lazy_tree = LazyScalar::constant(circle->event_y).view();
        std::ostringstream nv;
        nv << circle->event_y;
        ev.lazy_formula = nv.str();
        if (low_prec) {
            const double k = low_prec_width / (grid_h + 1.0e-300);
            const char* verdict =
                low_prec_width >= 1.0e6 ? "DEGENERATE: a +/- h/2 nudge makes the triple collinear (unbounded)"
                : k > 1.0e3            ? "near-degenerate: rounding badly tips this circumcenter"
                : k > 30.0             ? "ill-conditioned: noticeably amplifies the grid error"
                                       : "well-conditioned: moves about a grid step (looks right)";
            std::ostringstream hs;
            hs << "Low Precision (" << low_precision_digits_ << " dp grid, h=" << grid_h
               << "): sites snapped to the grid; event_y = " << nv.str()
               << ", center = " << format_vec2(circle->center)
               << "; propagated uncertainty ~ " << low_prec_width << " (" << k << "x h)";
            ev.lazy_history = hs.str();
            detail << "\nLOW PRECISION: site coords quantized to " << low_precision_digits_
                   << " decimal places (h=" << grid_h << "). Propagated event_y"
                      " uncertainty ~ " << low_prec_width << " (" << k << "x h) -- " << verdict;
        } else {
            ev.lazy_history = "Analytic Double: collapsed immediately to a number"
                              " (no formula kept). event_y = " + nv.str() +
                              ", center = " + format_vec2(circle->center);
            detail << "\ncollapsed (Analytic Double): event_y = " << nv.str()
                   << "  -- switch Number Mode to Lazy Exact to keep it symbolic";
        }
        ev.lazy_detail = detail.str();
      } else {
        // Lazy Exact: build the vertex as a LazyScalar from the three
        // site coordinates. The circumcenter is rational in the inputs;
        // event_y = cy - sqrt(R2) keeps one radical. It stays a formula
        // for life -- the directrix comparison below only peeks a
        // transient instance (recorded as a forced tick), it does not
        // collapse it.
        const std::string ls = "S" + std::to_string(left->site_id);
        const std::string ms = "S" + std::to_string(mid->site_id);
        const std::string rs = "S" + std::to_string(right->site_id);
        const LazyScalar ax = LazyScalar::named(ls + "x", left_point.x);
        const LazyScalar ay = LazyScalar::named(ls + "y", left_point.y);
        const LazyScalar bx = LazyScalar::named(ms + "x", mid_point.x);
        const LazyScalar by = LazyScalar::named(ms + "y", mid_point.y);
        const LazyScalar cx = LazyScalar::named(rs + "x", right_point.x);
        const LazyScalar cy = LazyScalar::named(rs + "y", right_point.y);
        const LazyScalar two = LazyScalar::named("two", 2.0);
        const LazyScalar det =
            (two * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))).with_label("D");
        const LazyScalar asq = (ax * ax + ay * ay).with_label(ls + "|2");
        const LazyScalar bsq = (bx * bx + by * by).with_label(ms + "|2");
        const LazyScalar csq = (cx * cx + cy * cy).with_label(rs + "|2");
        const LazyScalar cx_ =
            ((asq * (by - cy) + bsq * (cy - ay) + csq * (ay - by)) / det).with_label("cx");
        const LazyScalar cy_ =
            ((asq * (cx - bx) + bsq * (ax - cx) + csq * (bx - ax)) / det).with_label("cy");
        const LazyScalar r2 =
            ((cx_ - ax) * (cx_ - ax) + (cy_ - ay) * (cy_ - ay)).with_label("R2");
        const LazyScalar event_y = (cy_ - r2.sqrt()).with_label("event_y");
        // Pre-peek double-interval width: how uncertain the value is
        // before the predicate forces any refinement.
        ev.lazy_interval_width = event_y.interval_width();
        // The actual predicate "is the circle bottom below the sweep?" --
        // one transient instantiation, formula preserved.
        (void)event_y.compare(LazyScalar::named("directrix", directrix_));
        ev.lazy_analytic = event_y.is_formula();
        ev.lazy_has_radical = event_y.has_radical();
        ev.lazy_forced = event_y.forced_count();
        ev.lazy_formula = event_y.formula(9);
        ev.lazy_atoms = event_y.atoms();
        ev.lazy_tree = event_y.view();
        ev.lazy_history = ls + " ^ " + ms + " ^ " + rs +
                          " -> circumcenter c (rational)"
                          " -> R2 = |c - " + ls + "|^2"
                          " -> event_y = c_y - sqrt(R2)"
                          " -> test  event_y < directrix";
        detail << "\nsymbolic event_y kept analytic (rational" << (ev.lazy_has_radical ? " + sqrt" : "")
               << "); comparison peeks so far: " << ev.lazy_forced;

        ev.lazy_detail = detail.str();
      }
    }
    capture("scheduled circle event",
            {left->id, mid->id, right->id},
            {},
            {left->site_id, mid->site_id, right->site_id},
            verbose_
                ? std::vector<std::string>{
                      "triple: " + std::to_string(left->site_id) + " / " + std::to_string(mid->site_id) + " / " +
                          std::to_string(right->site_id),
                      "center: " + format_vec2(circle->center),
                      "radius: " + std::to_string(circle->radius),
                      "event y: " + std::to_string(circle->event_y),
                  }
                : std::vector<std::string>{});
}

void FortuneEngine::handle_site_event(EventRecord& event) {
    const int site_id = event.site_id;
    const auto& site = sites_[site_id];

    if (beachline_.empty()) {
        last_site_branch_ = "first S" + std::to_string(site_id);
        auto* leaf = beachline_.insert_first(site_id);
        capture("site event: inserted first arc",
                {leaf->id},
                {},
                {site_id},
                verbose_ ? std::vector<std::string>{"site " + std::to_string(site_id) + " at " + format_vec2(site.point)} : std::vector<std::string>{});
        return;
    }

    const double search_directrix = site.point.y + 1.0e-9;
    std::vector<int> search_path;
    std::vector<std::string> search_details;
    FortunePredicateKernel predicates(arithmetic_mode_, &predicate_audit_);
    const int site_lazy_before = predicate_audit_.lazy_evaluations;
    const int site_wide_before = predicate_audit_.wide_fallbacks;
    const std::size_t site_notes_before = predicate_audit_.notes.size();
    auto* above =
        beachline_.find_arc_above(site.point.x, sites_, search_directrix, bounds_, &search_path, &search_details, &predicates);
    {
        const int ld = predicate_audit_.lazy_evaluations - site_lazy_before;
        const int wd = predicate_audit_.wide_fallbacks - site_wide_before;
        event.lazy_derived = (ld > 0 || wd > 0);
        event.lazy_unresolved = (ld - wd) > 0;
        event.lazy_scheduled_step = static_cast<int>(processed_events_.size());
        std::ostringstream sd;
        sd << "site event S" << site_id << " " << format_vec2(site.point) << "\n"
           << "arc-placement breakpoint search through the beachline\n";
        if (event.lazy_unresolved) {
            sd << "EXACTLY TIED (degenerate): a breakpoint comparison came out exactly 0, so no "
               << "tie policy can give it a side (" << ld << " lazy, " << wd << " wide)";
        } else if (event.lazy_derived) {
            sd << "LAZY (band-resolved): a breakpoint comparison fell inside the tolerance band, "
               << "so the tie policy fixed its sign -- consistency, not extra precision (long "
               << "double is double here) (" << ld << " lazy, " << wd << " wide)";
        } else {
            sd << "collapsed: every breakpoint comparison resolved by a confident double sign";
        }
        for (std::size_t i = site_notes_before; i < predicate_audit_.notes.size(); ++i) {
            sd << "\n  - " << predicate_audit_.notes[i];
        }
        event.lazy_detail = sd.str();
    }
    if (!above) {
        last_site_branch_ = "no-arc S" + std::to_string(site_id);
        capture("site event: no arc found",
                {},
                {},
                {site_id},
                verbose_ ? std::vector<std::string>{"site " + std::to_string(site_id) + " at " + format_vec2(site.point)} : std::vector<std::string>{});
        return;
    }

    if (verbose_) {
        search_details.push_back("search comparisons: " + std::to_string(search_path.size()) +
                                 ", tree height: " +
                                 std::to_string(beachline_.root() ? beachline_.root()->height : 0));
        search_details.push_back("balanced search walks breakpoint tests down the tree");
        capture("site event: found containing arc",
                search_path,
                {},
                {above->site_id, site_id},
                [&]() {
                    auto details = search_details;
                    details.push_back("site " + std::to_string(site_id) + " falls under leaf " + std::to_string(above->id) +
                                      " (site " + std::to_string(above->site_id) + ")");
                    return details;
                }());
    }

    if (std::abs(site.point.y - sites_[above->site_id].point.y) <= kRangeEpsilon) {
        const Side side = site.point.x < sites_[above->site_id].point.x ? Side::left : Side::right;
        last_site_branch_ = std::string("same-y ") + (side == Side::left ? "L" : "R") + " S" +
                            std::to_string(site_id) + " under S" + std::to_string(above->site_id);
        auto* old_neighbor = side == Side::left ? above->prev : above->next;
        const int inherited_edge = side == Side::left ? above->left_edge : above->right_edge;

        invalidate_circle_event(above);
        invalidate_circle_event(old_neighbor);
        const auto adjacent = beachline_.insert_adjacent(above, site_id, side);
        auto* inserted = adjacent.inserted;
        if (!inserted || !adjacent.breakpoint) {
            return;
        }

        const int edge_id =
            side == Side::left ? create_edge(site_id, above->site_id) : create_edge(above->site_id, site_id);
        adjacent.breakpoint->edge_id = edge_id;
        if (side == Side::left) {
            inserted->left_edge = inherited_edge;
            inserted->right_edge = edge_id;
            above->left_edge = edge_id;
            if (old_neighbor) {
                old_neighbor->right_edge = inherited_edge;
            }
        } else {
            above->right_edge = edge_id;
            inserted->left_edge = edge_id;
            inserted->right_edge = inherited_edge;
            if (old_neighbor) {
                old_neighbor->left_edge = inherited_edge;
            }
        }

        capture("site event: inserted same-y adjacent arc",
                {above->id, inserted->id, adjacent.breakpoint->id},
                {edge_id},
                {above->site_id, site_id},
                verbose_
                    ? std::vector<std::string>{
                          "site " + std::to_string(site_id) + " shares y with site " + std::to_string(above->site_id),
                          "inserted as an adjacent arc instead of splitting into duplicate old arcs",
                          "new same-y bisector edge #" + std::to_string(edge_id),
                      }
                    : std::vector<std::string>{});

        if (inserted->prev && inserted->prev->prev) {
            check_circle_event(inserted->prev->prev, inserted->prev, inserted);
        }
        if (inserted->prev && inserted->next) {
            check_circle_event(inserted->prev, inserted, inserted->next);
        }
        if (inserted->next && inserted->next->next) {
            check_circle_event(inserted, inserted->next, inserted->next->next);
        }
        return;
    }

    last_site_branch_ = "split S" + std::to_string(site_id) + " under S" + std::to_string(above->site_id);
    invalidate_circle_event(above);
    const auto split = beachline_.split_arc(above, site_id);

    const int edge_id = create_edge(split.left_copy->site_id, split.inserted->site_id);
    split.left_breakpoint->edge_id = edge_id;
    split.right_breakpoint->edge_id = edge_id;

    split.left_copy->right_edge = edge_id;
    split.inserted->left_edge = edge_id;
    split.inserted->right_edge = edge_id;
    split.right_copy->left_edge = edge_id;

    capture("site event: created paired breakpoint edge",
            {split.left_breakpoint->id, split.right_breakpoint->id, split.inserted->id},
            {edge_id},
            {site_id, split.left_copy->site_id},
            verbose_
                ? std::vector<std::string>{
                      "new site " + std::to_string(site_id) + " split arc site " + std::to_string(above->site_id),
                      "paired edge #" + std::to_string(edge_id) + " tracks both breakpoints",
                      "degenerate segment starts at " + format_vec2(site.point),
                  }
                : std::vector<std::string>{});

    if (split.left_copy->prev) {
        check_circle_event(split.left_copy->prev, split.left_copy, split.inserted);
    }
    if (split.right_copy->next) {
        check_circle_event(split.inserted, split.right_copy, split.right_copy->next);
    }
}

void FortuneEngine::handle_circle_event(EventRecord& event) {
    last_site_branch_ = "circle leaf-id=" + std::to_string(event.leaf_id) +
                        " @" + format_vec2(event.center);
    auto* leaf = beachline_.leaf_by_id(event.leaf_id);
    if (!leaf || leaf->circle_event != event.id) {
        last_site_branch_ += " [stale: id mismatch]";
        return;
    }

    auto* left = leaf->prev;
    auto* right = leaf->next;
    if (!left || !right) {
        return;
    }

    // Robustness against cocircular degeneracy (grids): several circle
    // events can share one center. Processing one changes adjacency, so a
    // later event may still pass the id check yet now describe a different
    // triple. Re-derive the circumcircle of the CURRENT left/mid/right
    // sites and require it to still match this event's center. If it does
    // not, the adjacency moved out from under the event -- treat it as
    // stale instead of letting remove_arc corrupt the beachline.
    if (left->site_id == leaf->site_id || right->site_id == leaf->site_id ||
        left->site_id == right->site_id) {
        return;
    }
    {
        FortunePredicateKernel predicates(arithmetic_mode_, &predicate_audit_);
        const auto live_circle = circumcircle_from_sites(sites_[left->site_id].point,
                                                         sites_[leaf->site_id].point,
                                                         sites_[right->site_id].point,
                                                         &predicates);
        if (!live_circle) {
            return;
        }
        const double drift = length(live_circle->center - event.center);
        const double tol = 1.0e-6 * (1.0 + length(event.center));
        if (drift > tol) {
            if (verbose_) {
                capture("circle event: stale (adjacency moved)",
                        {leaf->id},
                        {},
                        {left->site_id, leaf->site_id, right->site_id},
                        {"live circumcenter " + format_vec2(live_circle->center) +
                         " no longer matches event center " + format_vec2(event.center)});
            }
            return;
        }
    }

    if (!robustness_.grouped_collapse) {
        // NAIVE (pre-5088aff): one circle event removes exactly one arc,
        // using event.center per event and opening its own replacement
        // edge. Spurious micro-edges + wrong topology at cocircular
        // vertices -- intentionally reproduced here for the A/B.
        auto* left_break = beachline_.left_breakpoint(leaf);
        auto* right_break = beachline_.right_breakpoint(leaf);
        if (!left_break || !right_break) {
            last_site_branch_ += " [naive: missing breakpoint]";
            return;
        }
        complete_edge(left_break->edge_id, event.center);
        complete_edge(right_break->edge_id, event.center);
        capture("circle event: closed converging edges (naive)",
                {left_break->id, right_break->id, leaf->id},
                {left_break->edge_id, right_break->edge_id},
                {left->site_id, leaf->site_id, right->site_id});
        invalidate_circle_event(left);
        invalidate_circle_event(right);
        leaf->circle_event = -1;
        const auto removal = beachline_.remove_arc(leaf);
        auto* nl = removal.left_neighbor;
        auto* nr = removal.right_neighbor;
        if (removal.surviving_breakpoint && nl && nr) {
            const int edge_id = create_edge(nl->site_id, nr->site_id);
            removal.surviving_breakpoint->edge_id = edge_id;
            nl->right_edge = edge_id;
            nr->left_edge = edge_id;
            edges_[edge_id].start = event.center;
            capture("circle event: opened replacement edge (naive)",
                    {removal.surviving_breakpoint->id, nl->id, nr->id},
                    {edge_id}, {nl->site_id, nr->site_id});
        }
        if (nl && nl->prev && nr) {
            check_circle_event(nl->prev, nl, nr);
        }
        if (nl && nr && nr->next) {
            check_circle_event(nl, nr, nr->next);
        }
        return;
    }

    // --- Cocircular-aware collapse -------------------------------------
    // A degree-d Voronoi vertex (d > 3: grid cell corners, points on a
    // circle) is really d-2 circle events at the SAME point. Handling
    // them one at a time -- each recomputing a slightly different centre
    // and opening its own replacement edge -- produced spurious
    // micro-edges and outright wrong topology (the Divergence overlay
    // showed it). Instead: take event.center as the single exact vertex
    // V, find the maximal contiguous run of arcs that all collapse at V,
    // terminate every breakpoint among them at V, remove the whole run
    // at once, and open ONE replacement edge between the two outer
    // survivors. For a normal triple the run is a single arc and this
    // reduces exactly to the previous one-arc behaviour.
    const Vec2 vertex = event.center;
    const double group_tol = 1.0e-6 * (1.0 + length(vertex));
    const auto collapses_here = [&](Beachline::LeafNode* a) -> bool {
        if (!a) {
            return false;
        }
        auto* p = a->prev;
        auto* n = a->next;
        if (!p || !n || p->site_id == a->site_id || n->site_id == a->site_id ||
            p->site_id == n->site_id) {
            return false;
        }
        const auto cc = circumcircle_from_sites(sites_[p->site_id].point,
                                                sites_[a->site_id].point,
                                                sites_[n->site_id].point);
        return cc.has_value() && length(cc->center - vertex) <= group_tol;
    };

    std::vector<Beachline::LeafNode*> run;
    for (auto* a = leaf->prev; collapses_here(a); a = a->prev) {
        run.push_back(a);
    }
    std::reverse(run.begin(), run.end());
    run.push_back(leaf);
    for (auto* a = leaf->next; collapses_here(a); a = a->next) {
        run.push_back(a);
    }

    auto* survive_left = run.front()->prev;
    auto* survive_right = run.back()->next;
    if (!survive_left || !survive_right) {
        return;
    }

    // Terminate every breakpoint along [SL, run..., SR] at the shared
    // vertex -- that is d-1 incoming edges for a degree-d vertex.
    std::vector<int> closed;
    {
        Beachline::LeafNode* x = survive_left;
        for (std::size_t i = 0; i < run.size() + 1; ++i) {
            auto* bp = beachline_.right_breakpoint(x);
            if (bp && bp->edge_id >= 0) {
                complete_edge(bp->edge_id, vertex);
                closed.push_back(bp->edge_id);
            }
            x = (i < run.size()) ? run[i] : nullptr;
        }
    }

    capture("circle event: closed converging edges",
            {leaf->id},
            closed,
            {survive_left->site_id, leaf->site_id, survive_right->site_id},
            verbose_ ? std::vector<std::string>{
                           std::to_string(run.size()) + " arc(s) collapse at " +
                               format_vec2(vertex)}
                     : std::vector<std::string>{});

    // Drop the colliding circle events and remove the entire run; the
    // surviving breakpoint after the last removal is the SL|SR boundary.
    invalidate_circle_event(survive_left);
    invalidate_circle_event(survive_right);
    Beachline::BreakpointNode* surviving = nullptr;
    for (auto* arc : run) {
        invalidate_circle_event(arc);
        arc->circle_event = -1;
        const auto removal = beachline_.remove_arc(arc);
        surviving = removal.surviving_breakpoint;
    }

    if (surviving) {
        const int edge_id = create_edge(survive_left->site_id, survive_right->site_id);
        surviving->edge_id = edge_id;
        survive_left->right_edge = edge_id;
        survive_right->left_edge = edge_id;
        edges_[edge_id].start = vertex;
        capture("circle event: opened replacement edge",
                {surviving->id, survive_left->id, survive_right->id},
                {edge_id},
                {survive_left->site_id, survive_right->site_id},
                verbose_ ? std::vector<std::string>{
                               "degree-" + std::to_string(run.size() + 2) + " vertex; new edge #" +
                                   std::to_string(edge_id)}
                         : std::vector<std::string>{});
    }

    if (survive_left->prev && survive_right) {
        check_circle_event(survive_left->prev, survive_left, survive_right);
    }
    if (survive_right->next) {
        check_circle_event(survive_left, survive_right, survive_right->next);
    }
}

bool FortuneEngine::step() {
    while (!queue_.empty()) {
        const QueueEntry top = queue_.top();
        queue_.pop();

        auto& event = events_[top.id];
        if (event.processed) {
            continue;
        }
        if (!event.valid) {
            if (verbose_ && event.kind == EventKind::circle) {
                current_event_id_ = event.id;
                capture("discarded invalid circle event",
                        {},
                        {},
                        {},
                        {"lazy invalidation: event #" + std::to_string(event.id) + " removed only when popped from the heap"});
                current_event_id_ = -1;
            }
            event.processed = true;
            continue;
        }

        current_event_id_ = event.id;
        event.processed = true;
        directrix_ = event.point.y;

        capture(event.kind == EventKind::site ? "processing site event" : "processing circle event",
                {},
                {},
                event.kind == EventKind::site ? std::vector<int>{event.site_id} : std::vector<int>{});

        if (event.kind == EventKind::site) {
            handle_site_event(event);
        } else {
            handle_circle_event(event);
        }

        processed_events_.push_back(event.id);
        capture(event.kind == EventKind::site ? "site event complete" : "circle event complete");
        debug_log_event(event.kind == EventKind::site ? "site-event" : "circle-event");
        current_event_id_ = -1;

        if (pending_events().empty() && !finalized_) {
            finalize();
            debug_log_event("finalize");
        }
        return true;
    }

    if (!finalized_) {
        finalize();
        debug_log_event("finalize");
    }
    current_event_id_ = -1;
    return false;
}

void FortuneEngine::run_all() {
    while (step()) {
    }
}

int FortuneEngine::snapshot_index_at_directrix(double y) const {
    if (snapshots_.empty()) {
        return -1;
    }
    int lo = 0;
    int hi = static_cast<int>(snapshots_.size()) - 1;
    int ans = 0;  // if y is above all, clamp to the first (highest-directrix) snapshot
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        if (snapshots_[mid].directrix >= y - 1.0e-9) {
            ans = mid;
            lo = mid + 1;  // go later (smaller directrix) while still >= y
        } else {
            hi = mid - 1;
        }
    }
    return ans;
}

std::vector<EdgeTrace> FortuneEngine::raw_final_edges() const {
    // Collapse the cocircular vertex cluster (a ring's centre, a grid's
    // shared corners) BEFORE the per-edge box-resolve, so every spoke
    // ends on one shared point and the spurious micro-edges vanish.
    const double span =
        std::max({bounds_.max_x - bounds_.min_x, bounds_.max_y - bounds_.min_y, 1.0e-6});
    std::vector<EdgeTrace> sweep;
    sweep.reserve(edges_.size());
    for (const auto& edge : edges_) {
        if (edge.left_site < 0 || edge.left_site >= static_cast<int>(sites_.size()) ||
            edge.right_site < 0 || edge.right_site >= static_cast<int>(sites_.size())) {
            continue;
        }
        sweep.push_back(edge);
    }
    sweep = weld_edge_vertices(std::move(sweep), 1.0e-6 * span);

    std::vector<EdgeTrace> result;
    result.reserve(sweep.size());
    for (const auto& edge : sweep) {
        const Vec2 a = sites_[edge.left_site].point;
        const Vec2 b = sites_[edge.right_site].point;
        const auto third = third_circumcircle_site(sites_, edge, a,
                                                   robustness_.third_site_disambig);
        EdgeTrace resolved = finalize_edge_to_box(edge, a, b, third, bounds_);
        // Only emit drawable edges. A record finalize_edge_to_box could
        // not anchor (both endpoints reset -- e.g. a spurious cross-box
        // segment from a still-imperfect degenerate sweep) is not a
        // Voronoi edge; returning it would draw garbage.
        if (resolved.start && resolved.end) {
            result.push_back(resolved);
        }
    }
    return result;
}

void FortuneEngine::build_final_cells() {
    final_cells_.clear();
    std::vector<Vec2> box{
        {bounds_.min_x, bounds_.min_y},
        {bounds_.max_x, bounds_.min_y},
        {bounds_.max_x, bounds_.max_y},
        {bounds_.min_x, bounds_.max_y},
    };

    for (const auto& site : sites_) {
        std::vector<Vec2> polygon = box;
        for (const auto& other : sites_) {
            if (other.id == site.id) {
                continue;
            }
            const Vec2 normal = other.point - site.point;
            const Vec2 mid = (site.point + other.point) * 0.5;
            polygon = clip_polygon_half_plane(polygon, mid, normal);
            if (polygon.empty()) {
                break;
            }
        }
        final_cells_.push_back(CellPolygon{site.id, polygon});
    }
}

void FortuneEngine::build_final_edges() {
    finalized_edges_.clear();
    for (std::size_t left = 0; left < final_cells_.size(); ++left) {
        for (std::size_t right = left + 1; right < final_cells_.size(); ++right) {
            const auto segment = shared_segment_from_cells(final_cells_[left],
                                                           final_cells_[right],
                                                           sites_[left].point,
                                                           sites_[right].point);
            if (!segment) {
                continue;
            }
            finalized_edges_.push_back(EdgeTrace{
                static_cast<int>(finalized_edges_.size()),
                static_cast<int>(left),
                static_cast<int>(right),
                segment->first,
                segment->second,
            });
        }
    }
}

void FortuneEngine::finalize() {
    finalized_ = true;
    final_cells_.clear();
    finalized_edges_.clear();
    if (reference_final_enabled_) {
        build_final_cells();
        build_final_edges();
        capture("finalized: engine rebuilt reference",
                {},
                {},
                {},
                {"engine ran O(n^2) cell clip",
                 "see HUD legend for what's drawn"});
    } else {
        capture("finalized: engine kept raw sweep",
                {},
                {},
                {},
                {"engine: no O(n^2) rebuild",
                 "'Ref Overlay' may still draw a ref"});
    }
}

void FortuneEngine::debug_log_event(const std::string& phase) {
    const char* path = std::getenv("FORTUNE_BEACHLINE_LOG");
    if (!path || !*path) {
        return;
    }
    std::ofstream out(path, std::ios::app);
    if (!out) {
        return;
    }

    out << "#" << processed_events_.size() << " " << phase
        << " | dirx=" << directrix_;
    if (!last_site_branch_.empty()) {
        out << " | branch=" << last_site_branch_;
    }

    out << "\n  beach: ";
    const auto leaves = beachline_.ordered_leaves();
    if (leaves.empty()) {
        out << "(empty)";
    } else {
        for (std::size_t i = 0; i < leaves.size(); ++i) {
            if (i) {
                out << " | ";
            }
            const auto* leaf = leaves[i];
            const auto& p = sites_[leaf->site_id].point;
            out << "S" << leaf->site_id << "(" << p.x << "," << p.y << ")";
        }
    }

    const auto errors = beachline_.validate();
    out << "\n  validate: ";
    if (errors.empty()) {
        out << "OK";
    } else {
        out << "BROKEN";
        for (const auto& e : errors) {
            out << "\n    - " << e;
        }
    }
    out << "\n";
}

std::vector<EventRecord> FortuneEngine::pending_events() const {
    std::vector<EventRecord> pending;
    for (const auto& event : events_) {
        if (!event.processed && event.valid) {
            pending.push_back(event);
        }
    }
    std::sort(pending.begin(), pending.end(), [](const EventRecord& a, const EventRecord& b) {
        return event_processed_before(a.point.y, a.kind, a.point.x, a.id,
                                      b.point.y, b.kind, b.point.x, b.id);
    });
    return pending;
}

std::vector<EventRecord> FortuneEngine::queued_events() const {
    std::vector<EventRecord> queued;
    for (const auto& event : events_) {
        if (!event.processed) {
            queued.push_back(event);
        }
    }
    std::sort(queued.begin(), queued.end(), [](const EventRecord& a, const EventRecord& b) {
        return event_processed_before(a.point.y, a.kind, a.point.x, a.id,
                                      b.point.y, b.kind, b.point.x, b.id);
    });
    return queued;
}

double FortuneEngine::snapshot_display_directrix(const std::string& label) const {
    double sample = directrix_;
    if (current_event_id_ < 0 || current_event_id_ >= static_cast<int>(events_.size())) {
        return sample;
    }

    const double span = std::max({bounds_.max_x - bounds_.min_x, bounds_.max_y - bounds_.min_y, 1.0});
    const double delta = span * 1.0e-4;
    if (label.rfind("processing ", 0) == 0) {
        sample += delta;
    } else {
        sample -= delta;
    }
    return sample;
}

std::vector<SnapshotBreakpoint> FortuneEngine::active_breakpoints(const double directrix) const {
    std::vector<SnapshotBreakpoint> breakpoints;
    auto leaves = beachline_.ordered_leaves();
    const auto boundaries = monotone_beachline_boundaries(leaves, sites_, bounds_, directrix);
    for (std::size_t i = 0; i + 1 < leaves.size(); ++i) {
        auto* left = leaves[i];
        auto* right = leaves[i + 1];
        auto* breakpoint = beachline_.right_breakpoint(left);
        if (!breakpoint) {
            continue;
        }
        const double x = boundaries[i + 1];
        breakpoints.push_back(SnapshotBreakpoint{
            breakpoint->edge_id,
            breakpoint->id,
            left->id,
            right->id,
            Vec2{x, parabola_y(sites_[left->site_id].point, directrix, x)},
        });
    }
    return breakpoints;
}

void FortuneEngine::capture(const std::string& label,
                            std::vector<int> highlight_nodes,
                            std::vector<int> highlight_edges,
                            std::vector<int> highlight_sites,
                            std::vector<std::string> details) {
    Snapshot snapshot;
    snapshot.label = label;
    snapshot.directrix = directrix_;
    snapshot.display_directrix = snapshot_display_directrix(label);
    snapshot.arithmetic_mode = arithmetic_mode_;
    snapshot.predicate_audit = predicate_audit_;
    snapshot.finished = pending_events().empty();
    snapshot.finalized = finalized_;
    snapshot.reference_final = finalized_ && reference_final_enabled_;
    snapshot.current_event_id = current_event_id_;
    if (current_event_id_ >= 0 && current_event_id_ < static_cast<int>(events_.size())) {
        snapshot.current_event = events_[current_event_id_];
    }
    snapshot.edges = edges_;
    snapshot.breakpoints = active_breakpoints(snapshot.display_directrix);
    snapshot.active_edge_ids = active_edge_ids_from_breakpoints(snapshot.breakpoints);
    snapshot.queue_events = queued_events();
    snapshot.pending_events = pending_events();
    for (const auto& event : events_) {
        if (event.kind != EventKind::circle) {
            continue;
        }
        if (!event.valid) {
            snapshot.invalidated_events.push_back(event);
        } else if (event.processed) {
            snapshot.processed_events.push_back(event);
        }
    }
    snapshot.validation_errors = beachline_.validate();
    snapshot.highlight_nodes = std::move(highlight_nodes);
    snapshot.highlight_edges = std::move(highlight_edges);
    snapshot.highlight_sites = std::move(highlight_sites);
    if (finalized_) {
        snapshot.final_cells = final_cells_;
        snapshot.resolved_edges = finalized_edges_;
    }

    for (auto* node : beachline_.all_nodes()) {
        SnapshotTreeNode item;
        item.id = node->id;
        item.is_leaf = node->is_leaf;
        item.stores_arc = node->is_leaf;
        item.parent_id = node->parent ? node->parent->id : -1;
        item.height = node->height;
        item.subtree_size = 1;
        item.left_boundary_site = node->extreme[0] ? node->extreme[0]->site_id : -1;
        item.right_boundary_site = node->extreme[1] ? node->extreme[1]->site_id : -1;

        if (node->is_leaf) {
            auto* leaf = static_cast<Beachline::LeafNode*>(node);
            item.site_id = leaf->site_id;
            item.prev_leaf_id = leaf->prev ? leaf->prev->id : -1;
            item.next_leaf_id = leaf->next ? leaf->next->id : -1;
            item.left_child_id = -1;
            item.right_child_id = -1;
        } else {
            auto* breakpoint = static_cast<Beachline::BreakpointNode*>(node);
            item.edge_id = breakpoint->edge_id;
            item.left_child_id = breakpoint->child[0] ? breakpoint->child[0]->id : -1;
            item.right_child_id = breakpoint->child[1] ? breakpoint->child[1]->id : -1;
        }

        snapshot.tree_nodes.push_back(item);
    }

    const auto leaves = beachline_.ordered_leaves();
    const auto boundaries = monotone_beachline_boundaries(leaves, sites_, bounds_, snapshot.display_directrix);
    if (verbose_) {
        details.push_back("arc order: " + ordered_leaf_summary(leaves));
        details.push_back("active edges: " + active_edge_summary(snapshot.active_edge_ids));
    }
    snapshot.details = std::move(details);
    const double display_gap = std::max({bounds_.max_x - bounds_.min_x, bounds_.max_y - bounds_.min_y, 1.0}) * 2.0e-4;
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        auto* leaf = leaves[i];
        snapshot.leaf_order.push_back(leaf->id);
        const double left_x = boundaries[i];
        const double right_x = boundaries[i + 1];
        const bool degenerate = std::abs(sites_[leaf->site_id].point.y - snapshot.display_directrix) <= display_gap;
        const double anchor_x = degenerate ? sites_[leaf->site_id].point.x : (left_x + right_x) * 0.5;
        snapshot.arcs.push_back(SnapshotArc{
            leaf->id,
            leaf->site_id,
            leaf->prev ? leaf->prev->id : -1,
            leaf->next ? leaf->next->id : -1,
            leaf->left_edge,
            leaf->right_edge,
            leaf->circle_event,
            left_x,
            right_x,
            degenerate,
            sites_[leaf->site_id].point,
            Vec2{anchor_x, parabola_y(sites_[leaf->site_id].point, snapshot.display_directrix, anchor_x)},
        });
    }

    snapshots_.push_back(std::move(snapshot));
}

void FortuneEngine::on_beachline_mutation(const std::string& label, const int primary_id, const int secondary_id) {
    std::vector<int> nodes;
    if (primary_id >= 0) {
        nodes.push_back(primary_id);
    }
    if (secondary_id >= 0) {
        nodes.push_back(secondary_id);
    }
    capture("tree: " + label, std::move(nodes));
}

}  // namespace fortune
