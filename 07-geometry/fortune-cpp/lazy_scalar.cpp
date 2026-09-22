#include "fortune/lazy_scalar.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace fortune {

namespace {
enum class Op { kConst, kNeg, kAdd, kSub, kMul, kDiv, kSqrt };

const char* op_symbol(Op op) {
    switch (op) {
        case Op::kAdd: return " + ";
        case Op::kSub: return " - ";
        case Op::kMul: return " * ";
        case Op::kDiv: return " / ";
        default: return "?";
    }
}
}  // namespace

struct LazyScalar::Node {
    Op op = Op::kConst;
    double value = 0.0;  // exact leaf payload
    std::string name;    // non-empty => named atom
    std::string label;   // non-empty => semantic composite name
    std::shared_ptr<Node> a;
    std::shared_ptr<Node> b;

    // Lazy interval cache (double first, refined to long double on demand).
    mutable bool has_interval = false;
    mutable long double lo = 0.0L;
    mutable long double hi = 0.0L;
    mutable bool wide = false;        // refined past the double interval
    mutable int forced = 0;          // comparisons that materialised this

    [[nodiscard]] bool is_leaf() const { return op == Op::kConst; }
};

namespace {
using Node = LazyScalar::Node;
using NodePtr = std::shared_ptr<Node>;

NodePtr make_const(double v) {
    auto n = std::make_shared<Node>();
    n->op = Op::kConst;
    n->value = v;
    return n;
}

NodePtr make_named(const std::string& nm, double v) {
    auto n = make_const(v);
    n->name = nm;
    return n;
}

NodePtr make_unary(Op op, NodePtr a) {
    auto n = std::make_shared<Node>();
    n->op = op;
    n->a = std::move(a);
    return n;
}

NodePtr make_binary(Op op, NodePtr a, NodePtr b) {
    auto n = std::make_shared<Node>();
    n->op = op;
    n->a = std::move(a);
    n->b = std::move(b);
    return n;
}

// Evaluate the interval. `wide` selects long double refinement; without it
// the interval is just the value with a tiny rounding pad so disjointness
// is meaningful.
void evaluate(const Node& n, bool wide) {
    if (n.has_interval && (n.wide || !wide)) {
        return;
    }
    const auto pad = [](long double v) {
        const long double e = std::fabs(v) * 8.0L * std::numeric_limits<double>::epsilon() +
                              std::numeric_limits<double>::min();
        return std::pair<long double, long double>{v - e, v + e};
    };
    switch (n.op) {
        case Op::kConst: {
            n.lo = n.hi = static_cast<long double>(n.value);
            break;
        }
        case Op::kNeg: {
            evaluate(*n.a, wide);
            n.lo = -n.a->hi;
            n.hi = -n.a->lo;
            break;
        }
        case Op::kAdd: {
            evaluate(*n.a, wide);
            evaluate(*n.b, wide);
            n.lo = n.a->lo + n.b->lo;
            n.hi = n.a->hi + n.b->hi;
            break;
        }
        case Op::kSub: {
            evaluate(*n.a, wide);
            evaluate(*n.b, wide);
            n.lo = n.a->lo - n.b->hi;
            n.hi = n.a->hi - n.b->lo;
            break;
        }
        case Op::kMul: {
            evaluate(*n.a, wide);
            evaluate(*n.b, wide);
            const long double p1 = n.a->lo * n.b->lo;
            const long double p2 = n.a->lo * n.b->hi;
            const long double p3 = n.a->hi * n.b->lo;
            const long double p4 = n.a->hi * n.b->hi;
            n.lo = std::min({p1, p2, p3, p4});
            n.hi = std::max({p1, p2, p3, p4});
            break;
        }
        case Op::kDiv: {
            evaluate(*n.a, wide);
            evaluate(*n.b, wide);
            // Assumes the denominator interval does not span zero (true for
            // the Fortune constructions: circumcircle determinant != 0).
            const long double q1 = n.a->lo / n.b->lo;
            const long double q2 = n.a->lo / n.b->hi;
            const long double q3 = n.a->hi / n.b->lo;
            const long double q4 = n.a->hi / n.b->hi;
            n.lo = std::min({q1, q2, q3, q4});
            n.hi = std::max({q1, q2, q3, q4});
            break;
        }
        case Op::kSqrt: {
            evaluate(*n.a, wide);
            const long double lo = n.a->lo > 0.0L ? n.a->lo : 0.0L;
            const long double hi = n.a->hi > 0.0L ? n.a->hi : 0.0L;
            n.lo = std::sqrt(lo);
            n.hi = std::sqrt(hi);
            break;
        }
    }
    if (!wide && n.op != Op::kConst) {
        const auto p = pad((n.lo + n.hi) * 0.5L);
        n.lo = std::min(n.lo, p.first);
        n.hi = std::max(n.hi, p.second);
    }
    n.has_interval = true;
    n.wide = wide;
}

void mark_forced(const Node& n) {
    ++n.forced;
    if (n.a) mark_forced(*n.a);
    if (n.b) mark_forced(*n.b);
}

int interval_sign(const Node& n) {
    evaluate(n, false);
    if (n.lo > 0.0L) return 1;
    if (n.hi < 0.0L) return -1;
    // Ambiguous in double -- refine to long double (the "exact peek").
    n.has_interval = false;
    evaluate(n, true);
    if (n.lo > 0.0L) return 1;
    if (n.hi < 0.0L) return -1;
    return 0;
}

int count_forced(const Node& n) {
    int total = n.forced;
    if (n.a) total = std::max(total, count_forced(*n.a));
    if (n.b) total = std::max(total, count_forced(*n.b));
    return total;
}

bool has_radical(const Node& n) {
    if (n.op == Op::kSqrt) return true;
    if (n.a && has_radical(*n.a)) return true;
    if (n.b && has_radical(*n.b)) return true;
    return false;
}

std::string render(const Node& n, int depth, bool is_root) {
    if (!is_root && !n.label.empty()) {
        return n.label;  // concise: a labeled composite prints its name
    }
    if (n.op == Op::kConst) {
        if (!n.name.empty()) {
            return n.name;
        }
        std::ostringstream os;
        os << n.value;
        return os.str();
    }
    if (depth <= 0) {
        return n.label.empty() ? "(..)" : n.label;
    }
    if (n.op == Op::kNeg) {
        return "- ( " + render(*n.a, depth - 1, false) + " )";
    }
    if (n.op == Op::kSqrt) {
        return "sqrt ( " + render(*n.a, depth - 1, false) + " )";
    }
    return "( " + render(*n.a, depth - 1, false) + op_symbol(n.op) + render(*n.b, depth - 1, false) + " )";
}

const char* op_name(Op op) {
    switch (op) {
        case Op::kConst: return "";
        case Op::kNeg: return "neg";
        case Op::kAdd: return "+";
        case Op::kSub: return "-";
        case Op::kMul: return "*";
        case Op::kDiv: return "/";
        case Op::kSqrt: return "sqrt";
    }
    return "";
}

LazyNodeView build_view(const Node& n) {
    LazyNodeView v;
    v.label = n.label;
    if (n.op == Op::kConst) {
        v.is_leaf = true;
        v.name = n.name;
        v.value = n.value;
        return v;
    }
    v.op = op_name(n.op);
    if (n.a) v.children.push_back(build_view(*n.a));
    if (n.b) v.children.push_back(build_view(*n.b));
    return v;
}

void collect_atoms(const Node& n, std::vector<std::pair<std::string, double>>& out) {
    if (n.op == Op::kConst) {
        if (!n.name.empty()) {
            for (const auto& kv : out) {
                if (kv.first == n.name) {
                    return;
                }
            }
            out.emplace_back(n.name, n.value);
        }
        return;
    }
    if (n.a) collect_atoms(*n.a, out);
    if (n.b) collect_atoms(*n.b, out);
}
}  // namespace

LazyScalar::LazyScalar(double value) : node_(make_const(value)) {}

LazyScalar LazyScalar::constant(double value) { return LazyScalar(value); }

LazyScalar LazyScalar::named(const std::string& name, double value) {
    return LazyScalar(make_named(name, value));
}

LazyScalar LazyScalar::operator+(const LazyScalar& rhs) const {
    return LazyScalar(make_binary(Op::kAdd, node_, rhs.node_));
}
LazyScalar LazyScalar::operator-(const LazyScalar& rhs) const {
    return LazyScalar(make_binary(Op::kSub, node_, rhs.node_));
}
LazyScalar LazyScalar::operator*(const LazyScalar& rhs) const {
    return LazyScalar(make_binary(Op::kMul, node_, rhs.node_));
}
LazyScalar LazyScalar::operator/(const LazyScalar& rhs) const {
    return LazyScalar(make_binary(Op::kDiv, node_, rhs.node_));
}
LazyScalar LazyScalar::operator-() const { return LazyScalar(make_unary(Op::kNeg, node_)); }
LazyScalar LazyScalar::sqrt() const { return LazyScalar(make_unary(Op::kSqrt, node_)); }

int LazyScalar::sign() const {
    const int s = interval_sign(*node_);
    mark_forced(*node_);
    return s;
}

int LazyScalar::compare(const LazyScalar& rhs) const {
    return (*this - rhs).sign();
}

double LazyScalar::approx() const {
    evaluate(*node_, false);
    return static_cast<double>((node_->lo + node_->hi) * 0.5L);
}

double LazyScalar::interval_width() const {
    if (!node_) {
        return 0.0;
    }
    // Force a fresh double-precision interval (not the wide refinement)
    // so the width reflects what the fast path alone can bound.
    node_->has_interval = false;
    evaluate(*node_, false);
    return static_cast<double>(node_->hi - node_->lo);
}

bool LazyScalar::is_formula() const { return node_ && node_->op != Op::kConst; }
int LazyScalar::forced_count() const { return node_ ? count_forced(*node_) : 0; }
bool LazyScalar::has_radical() const { return node_ && fortune::has_radical(*node_); }
std::string LazyScalar::formula(int max_depth) const {
    return node_ ? render(*node_, max_depth, true) : "0";
}
std::vector<std::pair<std::string, double>> LazyScalar::atoms() const {
    std::vector<std::pair<std::string, double>> out;
    if (node_) collect_atoms(*node_, out);
    return out;
}
LazyScalar LazyScalar::with_label(const std::string& label) const {
    if (!node_) {
        return *this;
    }
    // Clone the top node (children stay shared) and tag it -- structure
    // and evaluation are unchanged; only display/concision is affected.
    auto n = std::make_shared<Node>(*node_);
    n->has_interval = false;  // independent cache for the labeled copy
    n->forced = 0;
    n->label = label;
    return LazyScalar(std::move(n));
}
LazyNodeView LazyScalar::view() const {
    if (!node_) {
        LazyNodeView v;
        v.is_leaf = true;
        v.value = 0.0;
        return v;
    }
    return build_view(*node_);
}

}  // namespace fortune
