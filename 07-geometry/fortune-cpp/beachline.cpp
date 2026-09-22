#include "fortune/beachline.h"

#include <sstream>
#include <unordered_set>

namespace fortune {

void Beachline::clear() {
    root_ = nullptr;
    head_ = nullptr;
    tail_ = nullptr;
    next_id_ = 0;
    nodes_.clear();
}

Beachline::LeafNode* Beachline::make_leaf(const int site_id) {
    auto leaf = std::make_unique<LeafNode>();
    leaf->id = next_id_++;
    leaf->is_leaf = true;
    leaf->site_id = site_id;
    leaf->extreme = {leaf.get(), leaf.get()};
    auto* raw = leaf.get();
    nodes_.emplace(raw->id, std::move(leaf));
    return raw;
}

Beachline::BreakpointNode* Beachline::make_breakpoint() {
    auto node = std::make_unique<BreakpointNode>();
    node->id = next_id_++;
    node->is_leaf = false;
    auto* raw = node.get();
    nodes_.emplace(raw->id, std::move(node));
    return raw;
}

int Beachline::height_of(const NodeBase* node) {
    return node ? node->height : 0;
}

Beachline::LeafNode* Beachline::as_leaf(NodeBase* node) {
    return node && node->is_leaf ? static_cast<LeafNode*>(node) : nullptr;
}

const Beachline::LeafNode* Beachline::as_leaf(const NodeBase* node) {
    return node && node->is_leaf ? static_cast<const LeafNode*>(node) : nullptr;
}

Beachline::BreakpointNode* Beachline::as_breakpoint(NodeBase* node) {
    return node && !node->is_leaf ? static_cast<BreakpointNode*>(node) : nullptr;
}

const Beachline::BreakpointNode* Beachline::as_breakpoint(const NodeBase* node) {
    return node && !node->is_leaf ? static_cast<const BreakpointNode*>(node) : nullptr;
}

Beachline::NodeBase* Beachline::child_on(BreakpointNode* node, const Side side) {
    return node ? node->child[index_of(side)] : nullptr;
}

const Beachline::NodeBase* Beachline::child_on(const BreakpointNode* node, const Side side) {
    return node ? node->child[index_of(side)] : nullptr;
}

void Beachline::set_child(BreakpointNode* node, const Side side, NodeBase* child) {
    node->child[index_of(side)] = child;
    if (child) {
        child->parent = node;
    }
}

int Beachline::balance_factor(const BreakpointNode* node) {
    return height_of(node->child[index_of(Side::left)]) - height_of(node->child[index_of(Side::right)]);
}

bool Beachline::is_left_child(const NodeBase* node) {
    return node->parent && node->parent->child[index_of(Side::left)] == node;
}

void Beachline::replace_node(NodeBase* old_node, NodeBase* new_node) {
    auto* parent = old_node->parent;
    new_node->parent = parent;
    if (!parent) {
        root_ = new_node;
        return;
    }
    parent->child[is_left_child(old_node) ? 0 : 1] = new_node;
}

void Beachline::erase_node(NodeBase* node) {
    if (node) {
        nodes_.erase(node->id);
    }
}

void Beachline::update_node(NodeBase* node) {
    if (!node) {
        return;
    }

    if (auto* leaf = as_leaf(node)) {
        leaf->height = 1;
        leaf->extreme = {leaf, leaf};
        return;
    }

    auto* breakpoint = as_breakpoint(node);
    breakpoint->height = 1 + std::max(height_of(breakpoint->child[0]), height_of(breakpoint->child[1]));
    breakpoint->extreme[index_of(Side::left)] = breakpoint->child[0] ? breakpoint->child[0]->extreme[index_of(Side::left)] : nullptr;
    breakpoint->extreme[index_of(Side::right)] = breakpoint->child[1] ? breakpoint->child[1]->extreme[index_of(Side::right)] : nullptr;
}

void Beachline::notify(const std::string& label, const int primary_id, const int secondary_id) const {
    if (observer_) {
        observer_->on_beachline_mutation(label, primary_id, secondary_id);
    }
}

void Beachline::refresh_to_root(NodeBase* start) {
    auto* current = start;
    while (current) {
        update_node(current);
        current = current->parent;
    }
}

Beachline::BreakpointNode* Beachline::rotate(BreakpointNode* node, const Side direction) {
    auto* promoted = as_breakpoint(child_on(node, opposite(direction)));
    if (!promoted) {
        return node;
    }

    auto* parent = node->parent;
    const bool was_left = parent && parent->child[0] == node;
    auto* transfer = child_on(promoted, direction);

    set_child(node, opposite(direction), transfer);
    set_child(promoted, direction, node);
    promoted->parent = parent;

    if (!parent) {
        root_ = promoted;
    } else {
        parent->child[was_left ? 0 : 1] = promoted;
    }

    update_node(node);
    update_node(promoted);
    refresh_to_root(parent);
    notify(direction == Side::left ? "rotate left" : "rotate right", node->id, promoted->id);
    return promoted;
}

void Beachline::rebalance_from(BreakpointNode* start) {
    auto* current = start;
    while (current) {
        update_node(current);
        const int bf = balance_factor(current);
        if (bf > 1) {
            auto* left = as_breakpoint(current->child[0]);
            if (left && balance_factor(left) < 0) {
                rotate(left, Side::left);
            }
            current = rotate(current, Side::right);
        } else if (bf < -1) {
            auto* right = as_breakpoint(current->child[1]);
            if (right && balance_factor(right) > 0) {
                rotate(right, Side::right);
            }
            current = rotate(current, Side::left);
        }
        current = current->parent;
    }
}

Beachline::LeafNode* Beachline::insert_first(const int site_id) {
    auto* leaf = make_leaf(site_id);
    root_ = leaf;
    head_ = leaf;
    tail_ = leaf;
    notify("insert first arc", leaf->id, -1);
    return leaf;
}

Beachline::NodeBase* Beachline::node_by_id(const int id) const {
    const auto it = nodes_.find(id);
    return it == nodes_.end() ? nullptr : it->second.get();
}

Beachline::LeafNode* Beachline::leaf_by_id(const int id) const {
    return as_leaf(node_by_id(id));
}

Beachline::BreakpointNode* Beachline::breakpoint_by_id(const int id) const {
    return as_breakpoint(node_by_id(id));
}

Beachline::LeafNode* Beachline::find_arc_above(const double x,
                                               const std::vector<Site>& sites,
                                               const double directrix,
                                               const Rect& bounds,
                                               std::vector<int>* path,
                                               std::vector<std::string>* details,
                                               FortunePredicateKernel* predicates) const {
    NodeBase* current = root_;
    while (current) {
        if (path) {
            path->push_back(current->id);
        }
        if (auto* leaf = as_leaf(current)) {
            if (details) {
                const auto [left_x, right_x] = leaf_range(leaf, sites, directrix, bounds, predicates);
                std::ostringstream out;
                out.setf(std::ios::fixed, std::ios::floatfield);
                out.precision(4);
                out << "leaf " << leaf->id << " site " << leaf->site_id << " owns [" << left_x << ", " << right_x << "]";
                details->push_back(out.str());
            }
            return leaf;
        }

        auto* breakpoint = as_breakpoint(current);
        // Defensive: a well-formed beachline always has both children and
        // their extreme[] pointers populated. If a mid-mutation glitch
        // leaves one null we would deref null (crash) or silently fall
        // through to tail_ and misplace the new arc with no signal. Make
        // it visible instead of corrupting the structure quietly.
        if (!breakpoint->child[0] || !breakpoint->child[1] ||
            !breakpoint->child[0]->extreme[1] || !breakpoint->child[1]->extreme[0]) {
            notify("find_arc_above: broken breakpoint subtree", breakpoint->id, -1);
            return tail_;
        }
        auto* left_boundary = breakpoint->child[0]->extreme[1];
        auto* right_boundary = breakpoint->child[1]->extreme[0];
        double breakpoint_x =
            choose_breakpoint_x(sites[left_boundary->site_id].point, sites[right_boundary->site_id].point, directrix);
        breakpoint_x = clamp(breakpoint_x, bounds.min_x, bounds.max_x);
        if (details) {
            std::ostringstream out;
            out.setf(std::ios::fixed, std::ios::floatfield);
            out.precision(4);
            out << "breakpoint " << breakpoint->id << " compares x=" << x << " to " << breakpoint_x << " between S"
                << left_boundary->site_id << " and S" << right_boundary->site_id;
            details->push_back(out.str());
        }
        const bool go_left =
            predicates ? predicates->scalar_less_equal(
                             x,
                             breakpoint_x,
                             "breakpoint search x <= B(S" + std::to_string(left_boundary->site_id) + ",S" +
                                 std::to_string(right_boundary->site_id) + ")")
                       : x <= breakpoint_x + kRangeEpsilon;
        current = go_left ? breakpoint->child[0] : breakpoint->child[1];
    }
    return tail_;
}

Beachline::AdjacentInsertResult Beachline::insert_adjacent(LeafNode* anchor, const int new_site_id, const Side side) {
    AdjacentInsertResult result;
    if (!anchor) {
        return result;
    }

    auto* inserted = make_leaf(new_site_id);
    auto* breakpoint = make_breakpoint();

    if (side == Side::left) {
        auto* old_prev = anchor->prev;
        inserted->prev = old_prev;
        inserted->next = anchor;
        anchor->prev = inserted;
        if (old_prev) {
            old_prev->next = inserted;
        } else {
            head_ = inserted;
        }

        replace_node(anchor, breakpoint);
        set_child(breakpoint, Side::left, inserted);
        set_child(breakpoint, Side::right, anchor);
    } else {
        auto* old_next = anchor->next;
        inserted->prev = anchor;
        inserted->next = old_next;
        anchor->next = inserted;
        if (old_next) {
            old_next->prev = inserted;
        } else {
            tail_ = inserted;
        }

        replace_node(anchor, breakpoint);
        set_child(breakpoint, Side::left, anchor);
        set_child(breakpoint, Side::right, inserted);
    }

    update_node(anchor);
    update_node(inserted);
    update_node(breakpoint);
    refresh_to_root(breakpoint);
    notify(side == Side::left ? "insert same-y arc before leaf" : "insert same-y arc after leaf", inserted->id, anchor->id);
    rebalance_from(breakpoint->parent);

    result.anchor = anchor;
    result.inserted = inserted;
    result.breakpoint = breakpoint;
    return result;
}

Beachline::SplitResult Beachline::split_arc(LeafNode* leaf, const int new_site_id) {
    SplitResult result;
    const int old_site_id = leaf->site_id;

    auto* left_copy = make_leaf(old_site_id);
    auto* inserted = make_leaf(new_site_id);
    auto* right_copy = make_leaf(old_site_id);
    left_copy->left_edge = leaf->left_edge;
    right_copy->right_edge = leaf->right_edge;

    auto* left_break = make_breakpoint();
    auto* right_break = make_breakpoint();

    set_child(left_break, Side::left, left_copy);
    set_child(left_break, Side::right, inserted);
    set_child(right_break, Side::left, left_break);
    set_child(right_break, Side::right, right_copy);

    update_node(left_copy);
    update_node(inserted);
    update_node(right_copy);
    update_node(left_break);
    update_node(right_break);

    auto* old_prev = leaf->prev;
    auto* old_next = leaf->next;
    left_copy->prev = old_prev;
    left_copy->next = inserted;
    inserted->prev = left_copy;
    inserted->next = right_copy;
    right_copy->prev = inserted;
    right_copy->next = old_next;

    if (old_prev) {
        old_prev->next = left_copy;
    } else {
        head_ = left_copy;
    }
    if (old_next) {
        old_next->prev = right_copy;
    } else {
        tail_ = right_copy;
    }

    replace_node(leaf, right_break);
    erase_node(leaf);

    refresh_to_root(right_break);
    notify("split leaf into three arcs", inserted->id, right_break->id);
    rebalance_from(right_break->parent);

    result.left_copy = left_copy;
    result.inserted = inserted;
    result.right_copy = right_copy;
    result.left_breakpoint = left_break;
    result.right_breakpoint = right_break;
    return result;
}

Beachline::BreakpointNode* Beachline::left_breakpoint(LeafNode* leaf) const {
    NodeBase* current = leaf;
    auto* parent = leaf->parent;
    while (parent) {
        if (parent->child[index_of(Side::right)] == current) {
            return parent;
        }
        current = parent;
        parent = parent->parent;
    }
    return nullptr;
}

Beachline::BreakpointNode* Beachline::right_breakpoint(LeafNode* leaf) const {
    NodeBase* current = leaf;
    auto* parent = leaf->parent;
    while (parent) {
        if (parent->child[index_of(Side::left)] == current) {
            return parent;
        }
        current = parent;
        parent = parent->parent;
    }
    return nullptr;
}

Beachline::BreakpointNode* Beachline::lowest_common_ancestor(LeafNode* left, LeafNode* right) const {
    std::unordered_set<int> ancestors;
    NodeBase* current = left;
    while (current) {
        ancestors.insert(current->id);
        current = current->parent;
    }

    current = right;
    while (current) {
        if (ancestors.contains(current->id)) {
            return as_breakpoint(current);
        }
        current = current->parent;
    }
    return nullptr;
}

Beachline::RemoveResult Beachline::remove_arc(LeafNode* leaf) {
    RemoveResult result;
    result.left_neighbor = leaf->prev;
    result.right_neighbor = leaf->next;
    result.left_breakpoint = left_breakpoint(leaf);
    result.right_breakpoint = right_breakpoint(leaf);

    auto* parent = leaf->parent;
    if (!parent) {
        root_ = nullptr;
        head_ = nullptr;
        tail_ = nullptr;
        erase_node(leaf);
        notify("remove last arc", -1, -1);
        return result;
    }

    auto* sibling = parent->child[parent->child[0] == leaf ? 1 : 0];
    auto* grandparent = parent->parent;
    result.surviving_breakpoint = result.left_breakpoint == parent ? result.right_breakpoint : result.left_breakpoint;

    if (result.left_neighbor) {
        result.left_neighbor->next = result.right_neighbor;
    } else {
        head_ = result.right_neighbor;
    }
    if (result.right_neighbor) {
        result.right_neighbor->prev = result.left_neighbor;
    } else {
        tail_ = result.left_neighbor;
    }

    if (!grandparent) {
        root_ = sibling;
        sibling->parent = nullptr;
    } else {
        grandparent->child[grandparent->child[0] == parent ? 0 : 1] = sibling;
        sibling->parent = grandparent;
    }

    erase_node(leaf);
    erase_node(parent);

    refresh_to_root(sibling);
    notify("remove disappearing arc", leaf->id, sibling->id);
    rebalance_from(grandparent);
    return result;
}

std::pair<double, double> Beachline::leaf_range(const LeafNode* leaf,
                                                const std::vector<Site>& sites,
                                                const double directrix,
                                                const Rect& bounds,
                                                FortunePredicateKernel* predicates) const {
    double left_x = bounds.min_x;
    double right_x = bounds.max_x;

    if (const auto* prev = leaf->prev) {
        left_x = choose_breakpoint_x(sites[prev->site_id].point, sites[leaf->site_id].point, directrix);
    }
    if (const auto* next = leaf->next) {
        right_x = choose_breakpoint_x(sites[leaf->site_id].point, sites[next->site_id].point, directrix);
    }

    if (predicates && !predicates->scalar_less_equal(left_x, right_x, "leaf range left <= right")) {
        std::swap(left_x, right_x);
    }
    return {clamp(left_x, bounds.min_x, bounds.max_x), clamp(right_x, bounds.min_x, bounds.max_x)};
}

std::vector<Beachline::LeafNode*> Beachline::ordered_leaves() const {
    std::vector<LeafNode*> leaves;
    auto* leaf = head_;
    while (leaf) {
        leaves.push_back(leaf);
        leaf = leaf->next;
    }
    return leaves;
}

void Beachline::collect_nodes(NodeBase* node, std::vector<NodeBase*>& out) const {
    if (!node) {
        return;
    }
    out.push_back(node);
    if (auto* breakpoint = as_breakpoint(node)) {
        collect_nodes(breakpoint->child[0], out);
        collect_nodes(breakpoint->child[1], out);
    }
}

std::vector<Beachline::NodeBase*> Beachline::all_nodes() const {
    std::vector<NodeBase*> nodes;
    collect_nodes(root_, nodes);
    return nodes;
}

void Beachline::collect_leaves_inorder(NodeBase* node, std::vector<LeafNode*>& out) const {
    if (!node) {
        return;
    }
    if (auto* leaf = as_leaf(node)) {
        out.push_back(leaf);
        return;
    }
    auto* breakpoint = as_breakpoint(node);
    collect_leaves_inorder(breakpoint->child[0], out);
    collect_leaves_inorder(breakpoint->child[1], out);
}

std::array<Beachline::LeafNode*, 2> Beachline::computed_extremes(const NodeBase* node) const {
    if (!node) {
        return {nullptr, nullptr};
    }
    if (const auto* leaf = as_leaf(node)) {
        return {const_cast<LeafNode*>(leaf), const_cast<LeafNode*>(leaf)};
    }
    const auto* breakpoint = as_breakpoint(node);
    auto left_extreme = computed_extremes(breakpoint->child[0]);
    auto right_extreme = computed_extremes(breakpoint->child[1]);
    return {left_extreme[0], right_extreme[1]};
}

int Beachline::computed_height(const NodeBase* node) const {
    if (!node) {
        return 0;
    }
    if (as_leaf(node)) {
        return 1;
    }
    const auto* breakpoint = as_breakpoint(node);
    return 1 + std::max(computed_height(breakpoint->child[0]), computed_height(breakpoint->child[1]));
}

void Beachline::validate_node(const NodeBase* node,
                              const BreakpointNode* expected_parent,
                              std::vector<std::string>& errors) const {
    if (!node) {
        return;
    }
    if (node->parent != expected_parent) {
        errors.push_back("parent pointer mismatch at node " + std::to_string(node->id));
    }

    if (const auto* leaf = as_leaf(node)) {
        if (leaf->height != 1) {
            errors.push_back("leaf height mismatch at node " + std::to_string(node->id));
        }
        if (leaf->extreme[0] != leaf || leaf->extreme[1] != leaf) {
            errors.push_back("leaf extremes mismatch at node " + std::to_string(node->id));
        }
        return;
    }

    const auto* breakpoint = as_breakpoint(node);
    if (!breakpoint->child[0] || !breakpoint->child[1]) {
        errors.push_back("breakpoint missing child at node " + std::to_string(node->id));
        return;
    }

    validate_node(breakpoint->child[0], breakpoint, errors);
    validate_node(breakpoint->child[1], breakpoint, errors);

    if (computed_height(node) != node->height) {
        errors.push_back("height mismatch at node " + std::to_string(node->id));
    }

    const auto extremes = computed_extremes(node);
    if (extremes[0] != node->extreme[0] || extremes[1] != node->extreme[1]) {
        errors.push_back("extreme leaf mismatch at node " + std::to_string(node->id));
    }

    // NOTE: AVL balance is deliberately NOT validated here. It is a
    // search-speed optimization, not a beachline-correctness invariant,
    // and split_arc()/insert_adjacent() intentionally snapshot the tree
    // *before* rebalance_from() runs -- so a transient |bf|>1 at those
    // mutation snapshots is expected and self-heals on the next line.
    // Flagging it made the validity ribbon paint correctly-ordered
    // degenerate-case beachlines as "broken". Genuine corruption (arc
    // order, threading, parent links, breakpoint adjacency) is still
    // fully checked below and in validate().

    const auto* left_boundary = breakpoint->child[0]->extreme[1];
    const auto* right_boundary = breakpoint->child[1]->extreme[0];
    if (!left_boundary || !right_boundary || left_boundary->next != right_boundary || right_boundary->prev != left_boundary) {
        errors.push_back("breakpoint adjacency violation at node " + std::to_string(node->id));
    }
}

std::vector<std::string> Beachline::validate() const {
    std::vector<std::string> errors;
    if (!root_) {
        if (head_ || tail_) {
            errors.push_back("empty root with non-empty threaded list");
        }
        return errors;
    }

    validate_node(root_, nullptr, errors);

    std::vector<LeafNode*> inorder;
    collect_leaves_inorder(root_, inorder);
    const auto threaded = ordered_leaves();
    if (inorder.size() != threaded.size()) {
        errors.push_back("inorder leaf count does not match threaded leaf count");
    } else {
        for (std::size_t i = 0; i < inorder.size(); ++i) {
            if (inorder[i] != threaded[i]) {
                errors.push_back("inorder and threaded order diverge at index " + std::to_string(i));
                break;
            }
        }
    }

    if (!threaded.empty()) {
        if (threaded.front() != head_) {
            errors.push_back("head pointer mismatch");
        }
        if (threaded.back() != tail_) {
            errors.push_back("tail pointer mismatch");
        }
    }

    for (std::size_t i = 0; i < threaded.size(); ++i) {
        auto* leaf = threaded[i];
        if (leaf->prev != (i == 0 ? nullptr : threaded[i - 1])) {
            errors.push_back("prev pointer mismatch at leaf " + std::to_string(leaf->id));
        }
        if (leaf->next != (i + 1 == threaded.size() ? nullptr : threaded[i + 1])) {
            errors.push_back("next pointer mismatch at leaf " + std::to_string(leaf->id));
        }
    }

    return errors;
}

}  // namespace fortune
