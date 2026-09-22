#include "fortune/arc_beachline.h"

#include <sstream>

namespace fortune {

namespace {

std::uint32_t splitmix32(std::uint32_t x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    return x ^ (x >> 16);
}

}  // namespace

void ArcBeachline::clear() {
    root_ = nullptr;
    head_ = nullptr;
    tail_ = nullptr;
    next_id_ = 0;
    nodes_.clear();
}

ArcBeachline::Node* ArcBeachline::make_node(const int site_id) {
    auto node = std::make_unique<Node>();
    node->id = next_id_++;
    node->site_id = site_id;
    node->red = balance_ == BalancePolicy::rbt;
    node->priority = splitmix32(static_cast<std::uint32_t>(node->id + 1));
    auto* raw = node.get();
    nodes_.emplace(raw->id, std::move(node));
    return raw;
}

void ArcBeachline::erase_node(Node* node) {
    if (node) {
        nodes_.erase(node->id);
    }
}

void ArcBeachline::notify(const std::string& label, const int primary_id, const int secondary_id) const {
    if (observer_) {
        observer_->on_beachline_mutation(label, primary_id, secondary_id);
    }
}

ArcBeachline::Node* ArcBeachline::child_on(Node* node, const Side side) {
    return node ? node->child[index_of(side)] : nullptr;
}

const ArcBeachline::Node* ArcBeachline::child_on(const Node* node, const Side side) {
    return node ? node->child[index_of(side)] : nullptr;
}

void ArcBeachline::set_child(Node* node, const Side side, Node* child) {
    node->child[index_of(side)] = child;
    if (child) {
        child->parent = node;
    }
}

bool ArcBeachline::is_left_child(const Node* node) {
    return node->parent && node->parent->child[index_of(Side::left)] == node;
}

int ArcBeachline::height_of(const Node* node) {
    return node ? node->height : 0;
}

int ArcBeachline::size_of(const Node* node) {
    return node ? node->subtree_size : 0;
}

bool ArcBeachline::is_red(const Node* node) {
    return node && node->red;
}

bool ArcBeachline::is_black(const Node* node) {
    return !node || !node->red;
}

void ArcBeachline::update_node(Node* node) {
    if (!node) {
        return;
    }
    node->height = 1 + std::max(height_of(node->child[0]), height_of(node->child[1]));
    node->subtree_size = 1 + size_of(node->child[0]) + size_of(node->child[1]);
}

void ArcBeachline::refresh_to_root(Node* start) {
    auto* current = start;
    while (current) {
        update_node(current);
        current = current->parent;
    }
}

int ArcBeachline::balance_factor(const Node* node) const {
    return node ? height_of(node->child[0]) - height_of(node->child[1]) : 0;
}

ArcBeachline::Node* ArcBeachline::rotate(Node* node, const Side direction) {
    auto* promoted = child_on(node, opposite(direction));
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

void ArcBeachline::avl_rebalance_from(Node* start) {
    auto* current = start;
    while (current) {
        update_node(current);
        const int bf = balance_factor(current);
        if (bf > 1) {
            auto* left = current->child[0];
            if (left && balance_factor(left) < 0) {
                rotate(left, Side::left);
            }
            current = rotate(current, Side::right);
        } else if (bf < -1) {
            auto* right = current->child[1];
            if (right && balance_factor(right) > 0) {
                rotate(right, Side::right);
            }
            current = rotate(current, Side::left);
        }
        current = current ? current->parent : nullptr;
    }
}

void ArcBeachline::recolor(Node* node, const bool red, const std::string& reason) {
    if (!node || node->red == red) {
        return;
    }
    node->red = red;
    notify(reason + (red ? " -> red" : " -> black"), node->id, -1);
}

void ArcBeachline::rb_insert_fixup(Node* node) {
    auto* current = node;
    while (current != root_ && is_red(current->parent)) {
        auto* parent = current->parent;
        auto* grand = parent ? parent->parent : nullptr;
        if (!grand) {
            break;
        }
        const Side parent_side = is_left_child(parent) ? Side::left : Side::right;
        auto* uncle = child_on(grand, opposite(parent_side));
        if (is_red(uncle)) {
            recolor(parent, false, "recolor parent");
            recolor(uncle, false, "recolor uncle");
            recolor(grand, true, "recolor grandparent");
            current = grand;
            continue;
        }
        if (parent_side == Side::left) {
            if (current == parent->child[1]) {
                current = parent;
                rotate(current, Side::left);
                parent = current->parent;
                grand = parent ? parent->parent : nullptr;
            }
            recolor(parent, false, "fixup parent");
            recolor(grand, true, "fixup grandparent");
            if (grand) {
                rotate(grand, Side::right);
            }
        } else {
            if (current == parent->child[0]) {
                current = parent;
                rotate(current, Side::right);
                parent = current->parent;
                grand = parent ? parent->parent : nullptr;
            }
            recolor(parent, false, "fixup parent");
            recolor(grand, true, "fixup grandparent");
            if (grand) {
                rotate(grand, Side::left);
            }
        }
    }
    recolor(root_, false, "root black");
}

void ArcBeachline::rb_delete_fixup(Node* node, Node* parent) {
    auto* current = node;
    auto* current_parent = parent;
    while (current != root_ && is_black(current)) {
        if (!current_parent) {
            break;
        }
        const bool is_left = current == current_parent->child[0];
        auto* sibling = current_parent->child[is_left ? 1 : 0];
        if (is_red(sibling)) {
            recolor(sibling, false, "delete fix sibling");
            recolor(current_parent, true, "delete fix parent");
            rotate(current_parent, is_left ? Side::left : Side::right);
            sibling = current_parent->child[is_left ? 1 : 0];
        }

        const bool outer_black = is_black(sibling ? sibling->child[is_left ? 1 : 0] : nullptr);
        const bool inner_black = is_black(sibling ? sibling->child[is_left ? 0 : 1] : nullptr);
        if (outer_black && inner_black) {
            recolor(sibling, true, "delete fix sibling black");
            current = current_parent;
            current_parent = current ? current->parent : nullptr;
            continue;
        }

        if (is_left) {
            if (is_black(sibling ? sibling->child[1] : nullptr)) {
                recolor(sibling ? sibling->child[0] : nullptr, false, "delete fix inner child");
                recolor(sibling, true, "delete fix sibling red");
                if (sibling) {
                    rotate(sibling, Side::right);
                }
                sibling = current_parent->child[1];
            }
            if (sibling) {
                recolor(sibling, current_parent->red, "delete fix sibling inherit");
            }
            recolor(current_parent, false, "delete fix parent black");
            recolor(sibling ? sibling->child[1] : nullptr, false, "delete fix outer child");
            rotate(current_parent, Side::left);
        } else {
            if (is_black(sibling ? sibling->child[0] : nullptr)) {
                recolor(sibling ? sibling->child[1] : nullptr, false, "delete fix inner child");
                recolor(sibling, true, "delete fix sibling red");
                if (sibling) {
                    rotate(sibling, Side::left);
                }
                sibling = current_parent->child[0];
            }
            if (sibling) {
                recolor(sibling, current_parent->red, "delete fix sibling inherit");
            }
            recolor(current_parent, false, "delete fix parent black");
            recolor(sibling ? sibling->child[0] : nullptr, false, "delete fix outer child");
            rotate(current_parent, Side::right);
        }
        current = root_;
        break;
    }
    recolor(current, false, "delete fix current black");
}

void ArcBeachline::treap_bubble_up(Node* node) {
    auto* current = node;
    while (current && current->parent && current->priority < current->parent->priority) {
        auto* parent = current->parent;
        rotate(parent, is_left_child(current) ? Side::right : Side::left);
    }
}

void ArcBeachline::splay(Node* node) {
    auto* current = node;
    while (current && current->parent) {
        auto* parent = current->parent;
        auto* grand = parent->parent;
        if (!grand) {
            rotate(parent, is_left_child(current) ? Side::right : Side::left);
            continue;
        }

        const bool current_left = is_left_child(current);
        const bool parent_left = is_left_child(parent);
        if (current_left == parent_left) {
            rotate(grand, parent_left ? Side::right : Side::left);
            rotate(parent, current_left ? Side::right : Side::left);
        } else {
            rotate(parent, current_left ? Side::right : Side::left);
            rotate(grand, parent_left ? Side::right : Side::left);
        }
    }
}

void ArcBeachline::sbt_maintain(Node* node, const Side side) {
    if (!node) {
        return;
    }
    if (side == Side::left) {
        auto* left = node->child[0];
        if (left && size_of(left->child[0]) > size_of(node->child[1])) {
            node = rotate(node, Side::right);
        } else if (left && size_of(left->child[1]) > size_of(node->child[1])) {
            rotate(left, Side::left);
            node = rotate(node, Side::right);
        } else {
            return;
        }
    } else {
        auto* right = node->child[1];
        if (right && size_of(right->child[1]) > size_of(node->child[0])) {
            node = rotate(node, Side::left);
        } else if (right && size_of(right->child[0]) > size_of(node->child[0])) {
            rotate(right, Side::right);
            node = rotate(node, Side::left);
        } else {
            return;
        }
    }

    sbt_maintain(node->child[0], Side::left);
    sbt_maintain(node->child[1], Side::right);
    sbt_maintain(node, Side::left);
    sbt_maintain(node, Side::right);
}

void ArcBeachline::sbt_rebalance_from(Node* start) {
    auto* current = start;
    while (current) {
        update_node(current);
        sbt_maintain(current, Side::left);
        sbt_maintain(current, Side::right);
        current = current->parent;
    }
}

void ArcBeachline::transplant(Node* old_node, Node* new_node) {
    auto* parent = old_node->parent;
    if (!parent) {
        root_ = new_node;
    } else if (parent->child[0] == old_node) {
        parent->child[0] = new_node;
    } else {
        parent->child[1] = new_node;
    }
    if (new_node) {
        new_node->parent = parent;
    }
}

ArcBeachline::Node* ArcBeachline::leftmost(Node* node) const {
    auto* current = node;
    while (current && current->child[0]) {
        current = current->child[0];
    }
    return current;
}

ArcBeachline::Node* ArcBeachline::rightmost(Node* node) const {
    auto* current = node;
    while (current && current->child[1]) {
        current = current->child[1];
    }
    return current;
}

void ArcBeachline::link_before(Node* anchor, Node* node) {
    node->prev = anchor->prev;
    node->next = anchor;
    if (anchor->prev) {
        anchor->prev->next = node;
    } else {
        head_ = node;
    }
    anchor->prev = node;
}

void ArcBeachline::link_after(Node* anchor, Node* node) {
    node->prev = anchor;
    node->next = anchor->next;
    if (anchor->next) {
        anchor->next->prev = node;
    } else {
        tail_ = node;
    }
    anchor->next = node;
}

void ArcBeachline::insert_before(Node* anchor, Node* node) {
    link_before(anchor, node);
    Node* rebalance_from = nullptr;
    if (anchor->child[0]) {
        rebalance_from = rightmost(anchor->child[0]);
        set_child(rebalance_from, Side::right, node);
    } else {
        set_child(anchor, Side::left, node);
        rebalance_from = anchor;
    }

    refresh_to_root(rebalance_from);
    switch (balance_) {
        case BalancePolicy::avl:
            avl_rebalance_from(rebalance_from);
            break;
        case BalancePolicy::rbt:
            rb_insert_fixup(node);
            break;
        case BalancePolicy::treap:
            treap_bubble_up(node);
            break;
        case BalancePolicy::splay:
            splay(node);
            break;
        case BalancePolicy::sbt:
            sbt_rebalance_from(rebalance_from);
            break;
    }
}

void ArcBeachline::insert_after(Node* anchor, Node* node) {
    link_after(anchor, node);
    Node* rebalance_from = nullptr;
    if (anchor->child[1]) {
        rebalance_from = leftmost(anchor->child[1]);
        set_child(rebalance_from, Side::left, node);
    } else {
        set_child(anchor, Side::right, node);
        rebalance_from = anchor;
    }

    refresh_to_root(rebalance_from);
    switch (balance_) {
        case BalancePolicy::avl:
            avl_rebalance_from(rebalance_from);
            break;
        case BalancePolicy::rbt:
            rb_insert_fixup(node);
            break;
        case BalancePolicy::treap:
            treap_bubble_up(node);
            break;
        case BalancePolicy::splay:
            splay(node);
            break;
        case BalancePolicy::sbt:
            sbt_rebalance_from(rebalance_from);
            break;
    }
}

ArcBeachline::Node* ArcBeachline::node_by_id(const int id) const {
    const auto it = nodes_.find(id);
    return it == nodes_.end() ? nullptr : it->second.get();
}

ArcBeachline::Node* ArcBeachline::insert_first(const int site_id) {
    auto* node = make_node(site_id);
    node->red = false;
    root_ = node;
    head_ = node;
    tail_ = node;
    notify("insert first arc node", node->id, -1);
    return node;
}

ArcBeachline::Node* ArcBeachline::find_arc_above(const double x,
                                                 const std::vector<Site>& sites,
                                                 const double directrix,
                                                 const Rect& bounds,
                                                 std::vector<int>* path,
                                                 std::vector<std::string>* details,
                                                 FortunePredicateKernel* predicates) {
    auto* node = root_;
    while (node) {
        if (path) {
            path->push_back(node->id);
        }
        auto [left_x, right_x] = node_range(node, sites, directrix, bounds, predicates);
        if (details) {
            std::ostringstream out;
            out.setf(std::ios::fixed, std::ios::floatfield);
            out.precision(4);
            out << "compare node " << node->id << " site " << node->site_id << " with x=" << x
                << " against [" << left_x << ", " << right_x << "]";
            details->push_back(out.str());
        }
        const bool left_of_range =
            predicates ? predicates->scalar_less(x, left_x, "arc search x < left range") : x < left_x - kRangeEpsilon;
        if (left_of_range && node->child[0]) {
            node = node->child[0];
            continue;
        }
        const bool right_of_range =
            predicates ? predicates->scalar_less(right_x, x, "arc search right range < x") : x > right_x + kRangeEpsilon;
        if (right_of_range && node->child[1]) {
            node = node->child[1];
            continue;
        }
        if (balance_ == BalancePolicy::splay) {
            splay(node);
        }
        return node;
    }
    if (balance_ == BalancePolicy::splay && tail_) {
        splay(tail_);
    }
    return tail_;
}

ArcBeachline::AdjacentInsertResult ArcBeachline::insert_adjacent(Node* anchor, const int new_site_id, const Side side) {
    AdjacentInsertResult result;
    if (!anchor) {
        return result;
    }

    auto* inserted = make_node(new_site_id);
    if (side == Side::left) {
        insert_before(anchor, inserted);
    } else {
        insert_after(anchor, inserted);
    }
    notify(side == Side::left ? "insert same-y arc before node" : "insert same-y arc after node", inserted->id, anchor->id);

    result.anchor = anchor;
    result.inserted = inserted;
    return result;
}

ArcBeachline::SplitResult ArcBeachline::split_arc(Node* node, const int new_site_id) {
    SplitResult result;
    const int old_site = node->site_id;

    auto* left_copy = make_node(old_site);
    auto* right_copy = make_node(old_site);
    left_copy->left_edge = node->left_edge;
    right_copy->right_edge = node->right_edge;

    insert_before(node, left_copy);
    insert_after(node, right_copy);

    node->site_id = new_site_id;
    node->left_edge = -1;
    node->right_edge = -1;
    node->circle_event = -1;
    refresh_to_root(node);
    if (balance_ == BalancePolicy::splay) {
        splay(node);
    }
    notify("split arc node into three arcs", node->id, left_copy->id);

    result.left_copy = left_copy;
    result.inserted = node;
    result.right_copy = right_copy;
    return result;
}

ArcBeachline::RemoveResult ArcBeachline::remove_arc(Node* node) {
    RemoveResult result;
    result.left_neighbor = node->prev;
    result.right_neighbor = node->next;

    // Unlink the threaded list PAIRED with the tree erase (not up-front): treap
    // and splay rotate `node` while it is still in the tree, and the per-rotate
    // validation snapshot must see the node in BOTH the tree and the threaded
    // list (otherwise inorder-count != threaded-count). avl/sbt/rbt erase before
    // their rebalance rotations, so detaching at erase time is consistent for
    // every policy.
    const auto detach_node = [&]() {
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
        erase_node(node);
    };

    if (balance_ == BalancePolicy::avl || balance_ == BalancePolicy::sbt) {
        Node* rebalance_from = nullptr;
        if (!node->child[0]) {
            rebalance_from = node->parent;
            transplant(node, node->child[1]);
        } else if (!node->child[1]) {
            rebalance_from = node->parent;
            transplant(node, node->child[0]);
        } else {
            auto* succ = leftmost(node->child[1]);
            auto* succ_parent = succ->parent;
            if (succ_parent != node) {
                rebalance_from = succ_parent;
                transplant(succ, succ->child[1]);
                succ->child[1] = node->child[1];
                succ->child[1]->parent = succ;
            } else {
                rebalance_from = succ;
            }
            transplant(node, succ);
            succ->child[0] = node->child[0];
            succ->child[0]->parent = succ;
            update_node(succ);
        }
        detach_node();
        refresh_to_root(rebalance_from ? rebalance_from : root_);
        if (balance_ == BalancePolicy::avl) {
            avl_rebalance_from(rebalance_from ? rebalance_from : root_);
        } else {
            sbt_rebalance_from(rebalance_from ? rebalance_from : root_);
        }
    } else if (balance_ == BalancePolicy::rbt) {
        Node* y = node;
        Node* x = nullptr;
        Node* x_parent = nullptr;
        const bool y_original_red = y->red;

        if (!node->child[0]) {
            x = node->child[1];
            x_parent = node->parent;
            transplant(node, node->child[1]);
        } else if (!node->child[1]) {
            x = node->child[0];
            x_parent = node->parent;
            transplant(node, node->child[0]);
        } else {
            y = leftmost(node->child[1]);
            const bool successor_red = y->red;
            x = y->child[1];
            if (y->parent == node) {
                x_parent = y;
            } else {
                x_parent = y->parent;
                transplant(y, y->child[1]);
                y->child[1] = node->child[1];
                y->child[1]->parent = y;
            }
            transplant(node, y);
            y->child[0] = node->child[0];
            y->child[0]->parent = y;
            y->red = node->red;
            update_node(y);
            detach_node();
            refresh_to_root(x_parent ? x_parent : y);
            if (!successor_red) {
                rb_delete_fixup(x, x_parent);
            }
            refresh_to_root(root_);
            notify("remove arc node", y->id, -1);
            return result;
        }
        detach_node();
        refresh_to_root(x_parent ? x_parent : root_);
        if (!y_original_red) {
            rb_delete_fixup(x, x_parent);
        }
    } else if (balance_ == BalancePolicy::treap) {
        while (node->child[0] && node->child[1]) {
            const bool promote_left = node->child[0]->priority < node->child[1]->priority;
            rotate(node, promote_left ? Side::right : Side::left);
        }
        auto* parent = node->parent;
        auto* child = node->child[0] ? node->child[0] : node->child[1];
        transplant(node, child);
        detach_node();
        refresh_to_root(parent ? parent : root_);
    } else {
        // Splay-tree delete. Bring `node` to the root, then -- crucially -- keep
        // it fully linked in BOTH the tree and the threaded list while splaying
        // its predecessor. splay(max_left) rotates `node` down to become
        // max_left's right child (max_left ends with no right child of its own);
        // only AFTER that do we unlink `node` via transplant + detach_node, with
        // no validating rotation following. The previous formulation detached the
        // right subtree from the tree (right->parent = nullptr) BEFORE
        // splay(max_left), so every rotation in that splay snapshotted a tree
        // missing the detached nodes -> inorder < threaded count mismatch. The
        // resulting tree shape is identical; only the unlink timing changed.
        // (See tests run_arc_degenerate_regression: grid5x5 arc_splay.)
        splay(node);
        if (!node->child[0]) {
            transplant(node, node->child[1]);
            detach_node();
            refresh_to_root(root_);
        } else {
            auto* max_left = rightmost(node->child[0]);
            splay(max_left);                    // node becomes max_left->child[1]; node->child[0] == nullptr
            transplant(node, node->child[1]);   // splice node out; its right subtree rises under max_left
            detach_node();
            refresh_to_root(max_left);
        }
    }

    notify("remove arc node", result.left_neighbor ? result.left_neighbor->id : -1, result.right_neighbor ? result.right_neighbor->id : -1);
    return result;
}

int ArcBeachline::edge_between(const Node* left, const Node* right) const {
    if (!left || !right || left->next != right || right->prev != left) {
        return -1;
    }
    return left->right_edge >= 0 ? left->right_edge : right->left_edge;
}

void ArcBeachline::set_edge_between(Node* left, Node* right, const int edge_id) {
    if (!left || !right) {
        return;
    }
    left->right_edge = edge_id;
    right->left_edge = edge_id;
}

std::pair<double, double> ArcBeachline::node_range(const Node* node,
                                                   const std::vector<Site>& sites,
                                                   const double directrix,
                                                   const Rect& bounds,
                                                   FortunePredicateKernel* predicates) const {
    double left_x = bounds.min_x;
    double right_x = bounds.max_x;
    if (node->prev) {
        left_x = choose_breakpoint_x(sites[node->prev->site_id].point, sites[node->site_id].point, directrix);
    }
    if (node->next) {
        right_x = choose_breakpoint_x(sites[node->site_id].point, sites[node->next->site_id].point, directrix);
    }
    left_x = clamp(left_x, bounds.min_x, bounds.max_x);
    right_x = clamp(right_x, bounds.min_x, bounds.max_x);
    if (predicates && !predicates->scalar_less_equal(left_x, right_x, "node range left <= right")) {
        std::swap(left_x, right_x);
    }
    right_x = std::max(right_x, left_x);
    return {left_x, right_x};
}

std::vector<ArcBeachline::Node*> ArcBeachline::ordered_nodes() const {
    std::vector<Node*> ordered;
    auto* node = head_;
    while (node) {
        ordered.push_back(node);
        node = node->next;
    }
    return ordered;
}

void ArcBeachline::collect_nodes(Node* node, std::vector<Node*>& out) const {
    if (!node) {
        return;
    }
    out.push_back(node);
    collect_nodes(node->child[0], out);
    collect_nodes(node->child[1], out);
}

void ArcBeachline::collect_nodes_inorder(Node* node, std::vector<Node*>& out) const {
    if (!node) {
        return;
    }
    collect_nodes_inorder(node->child[0], out);
    out.push_back(node);
    collect_nodes_inorder(node->child[1], out);
}

std::vector<ArcBeachline::Node*> ArcBeachline::all_nodes() const {
    std::vector<Node*> nodes;
    collect_nodes(root_, nodes);
    return nodes;
}

int ArcBeachline::validate_rb_height(const Node* node, std::vector<std::string>& errors) const {
    if (!node) {
        return 1;
    }
    if (is_red(node) && (is_red(node->child[0]) || is_red(node->child[1]))) {
        errors.push_back("red-red violation at node " + std::to_string(node->id));
    }
    const int left_height = validate_rb_height(node->child[0], errors);
    const int right_height = validate_rb_height(node->child[1], errors);
    if (left_height != right_height) {
        errors.push_back("black height mismatch at node " + std::to_string(node->id));
    }
    return left_height + (node->red ? 0 : 1);
}

std::vector<std::string> ArcBeachline::validate() const {
    std::vector<std::string> errors;
    if (!root_) {
        if (head_ || tail_) {
            errors.push_back("empty root with non-empty threaded list");
        }
        return errors;
    }

    std::vector<Node*> inorder;
    collect_nodes_inorder(root_, inorder);
    const auto threaded = ordered_nodes();
    if (inorder.size() != threaded.size()) {
        errors.push_back("inorder count mismatch (inorder=" + std::to_string(inorder.size()) +
                         " threaded=" + std::to_string(threaded.size()) + ")");
    } else {
        for (std::size_t i = 0; i < inorder.size(); ++i) {
            if (inorder[i] != threaded[i]) {
                errors.push_back("inorder/threaded mismatch at index " + std::to_string(i));
                break;
            }
        }
    }

    for (const auto* node : inorder) {
        if (node->child[0] && node->child[0]->parent != node) {
            errors.push_back("left parent mismatch at node " + std::to_string(node->id));
        }
        if (node->child[1] && node->child[1]->parent != node) {
            errors.push_back("right parent mismatch at node " + std::to_string(node->id));
        }
        const int computed_height = 1 + std::max(height_of(node->child[0]), height_of(node->child[1]));
        if (node->height != computed_height) {
            errors.push_back("height mismatch at node " + std::to_string(node->id));
        }
        const int computed_size = 1 + size_of(node->child[0]) + size_of(node->child[1]);
        if (node->subtree_size != computed_size) {
            errors.push_back("size mismatch at node " + std::to_string(node->id));
        }
        // Balance invariants (AVL bf / treap heap / RB colors below) are
        // search-speed optimizations, NOT beachline-correctness
        // invariants, and are intentionally violated transiently between
        // a mutation's snapshot and its rebalance. Validating them made
        // the validity ribbon flag correctly-ordered degenerate-case
        // beachlines as "broken". Only genuine structure is checked.
    }

    if (!threaded.empty()) {
        if (threaded.front() != head_) {
            errors.push_back("head mismatch");
        }
        if (threaded.back() != tail_) {
            errors.push_back("tail mismatch");
        }
        for (std::size_t i = 0; i < threaded.size(); ++i) {
            auto* node = threaded[i];
            if (node->prev != (i == 0 ? nullptr : threaded[i - 1])) {
                errors.push_back("prev mismatch at node " + std::to_string(node->id));
            }
            if (node->next != (i + 1 == threaded.size() ? nullptr : threaded[i + 1])) {
                errors.push_back("next mismatch at node " + std::to_string(node->id));
            }
        }
    }

    // RB color / black-height are likewise perf invariants, not
    // beachline correctness; not validated (see note above).

    return errors;
}

}  // namespace fortune
