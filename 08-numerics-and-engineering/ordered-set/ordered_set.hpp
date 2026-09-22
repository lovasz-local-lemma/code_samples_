#pragma once
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace samples::containers
{
// A new left-leaning red-black variant inspired by the old engine's tree sample.
// It is not presented as a formatting-only conversion of logtcn_rbt.h.
template <class Key, class Compare = std::less<>> class OrderedSet
{
    static_assert(std::is_nothrow_invocable_r_v<bool, Compare, const Key &, const Key &>,
                  "Balancing requires a nonthrowing comparator");
    static_assert(std::is_nothrow_copy_assignable_v<Key>,
                  "Erasure replaces a key without throwing");
    struct Node;
    using Link = std::unique_ptr<Node>;
    struct Node
    {
        explicit Node(const Key &value) : key(value) {}
        Key key;
        bool red = true;
        std::size_t count = 1;
        Link left, right;
    };

  public:
    OrderedSet() = default;
    explicit OrderedSet(Compare compare) : compare_(std::move(compare)) {}
    OrderedSet(const OrderedSet &) = delete;
    OrderedSet &operator=(const OrderedSet &) = delete;
    OrderedSet(OrderedSet &&) = default;
    OrderedSet &operator=(OrderedSet &&) = default;
    std::size_t size() const
    {
        return count(root_);
    }
    bool contains(const Key &key) const
    {
        auto *node = root_.get();
        while (node)
        {
            if (compare_(key, node->key))
                node = node->left.get();
            else if (compare_(node->key, key))
                node = node->right.get();
            else
                return true;
        }
        return false;
    }
    bool insert(const Key &key)
    {
        auto candidate = std::make_unique<Node>(key); // Allocate before changing the tree.
        const bool inserted = insert_at(root_, candidate);
        root_->red = false;
        return inserted;
    }
    bool erase(const Key &key)
    {
        if (!contains(key))
            return false;
        if (!red(root_->left) && !red(root_->right))
            root_->red = true;
        root_ = erase_at(std::move(root_), key);
        if (root_)
            root_->red = false;
        return true;
    }
    std::vector<Key> sorted() const
    {
        std::vector<Key> values;
        values.reserve(size());
        const auto visit = [&](const auto &self, const Node *node) -> void
        {
            if (!node)
                return;
            self(self, node->left.get());
            values.push_back(node->key);
            self(self, node->right.get());
        };
        visit(visit, root_.get());
        return values;
    }
    bool valid() const
    {
        if (red(root_))
            return false;
        return inspect(root_.get(), nullptr, nullptr) >= 0;
    }

  private:
    static bool red(const Link &p)
    {
        return p && p->red;
    }
    static std::size_t count(const Link &p)
    {
        return p ? p->count : 0;
    }
    static void update(Node &n)
    {
        n.count = 1 + count(n.left) + count(n.right);
    }
    static Link rotate_left(Link h)
    {
        auto x = std::move(h->right);
        h->right = std::move(x->left);
        x->left = std::move(h);
        x->red = x->left->red;
        x->left->red = true;
        update(*x->left);
        update(*x);
        return x;
    }
    static Link rotate_right(Link h)
    {
        auto x = std::move(h->left);
        h->left = std::move(x->right);
        x->right = std::move(h);
        x->red = x->right->red;
        x->right->red = true;
        update(*x->right);
        update(*x);
        return x;
    }
    static void flip(Node &h)
    {
        h.red = !h.red;
        if (h.left)
            h.left->red = !h.left->red;
        if (h.right)
            h.right->red = !h.right->red;
    }
    static Link balance(Link h)
    {
        if (red(h->right) && !red(h->left))
            h = rotate_left(std::move(h));
        if (red(h->left) && red(h->left->left))
            h = rotate_right(std::move(h));
        if (red(h->left) && red(h->right))
            flip(*h);
        update(*h);
        return h;
    }
    bool insert_at(Link &h, Link &candidate)
    {
        if (!h)
        {
            h = std::move(candidate);
            return true;
        }
        bool inserted;
        if (compare_(candidate->key, h->key))
            inserted = insert_at(h->left, candidate);
        else if (compare_(h->key, candidate->key))
            inserted = insert_at(h->right, candidate);
        else
            return false;
        h = balance(std::move(h));
        return inserted;
    }
    static Link move_red_left(Link h)
    {
        flip(*h);
        if (h->right && red(h->right->left))
        {
            h->right = rotate_right(std::move(h->right));
            h = rotate_left(std::move(h));
            flip(*h);
        }
        return h;
    }
    static Link move_red_right(Link h)
    {
        flip(*h);
        if (h->left && red(h->left->left))
        {
            h = rotate_right(std::move(h));
            flip(*h);
        }
        return h;
    }
    static Link erase_min(Link h)
    {
        if (!h->left)
            return {};
        if (!red(h->left) && !red(h->left->left))
            h = move_red_left(std::move(h));
        h->left = erase_min(std::move(h->left));
        return balance(std::move(h));
    }
    Link erase_at(Link h, const Key &key)
    {
        if (compare_(key, h->key))
        {
            if (!red(h->left) && !red(h->left->left))
                h = move_red_left(std::move(h));
            h->left = erase_at(std::move(h->left), key);
        }
        else
        {
            if (red(h->left))
                h = rotate_right(std::move(h));
            const bool equal = !compare_(key, h->key) && !compare_(h->key, key);
            if (equal && !h->right)
                return {};
            if (!red(h->right) && !red(h->right->left))
                h = move_red_right(std::move(h));
            if (!compare_(key, h->key) && !compare_(h->key, key))
            {
                auto *successor = h->right.get();
                while (successor->left)
                    successor = successor->left.get();
                h->key = successor->key;
                h->right = erase_min(std::move(h->right));
            }
            else
                h->right = erase_at(std::move(h->right), key);
        }
        return balance(std::move(h));
    }
    int inspect(const Node *n, const Key *lower, const Key *upper) const
    {
        if (!n)
            return 1;
        if ((lower && !compare_(*lower, n->key)) || (upper && !compare_(n->key, *upper)) ||
            red(n->right) || (n->red && red(n->left)) ||
            n->count != 1 + count(n->left) + count(n->right))
            return -1;
        const auto l = inspect(n->left.get(), lower, &n->key);
        const auto r = inspect(n->right.get(), &n->key, upper);
        return l < 0 || l != r ? -1 : l + (n->red ? 0 : 1);
    }
    Link root_;
    Compare compare_{};
};
} // namespace samples::containers
