#pragma once

namespace LegacyNodeGraphActivity {

struct DisjointSet
{
    M<int, int> parent;
    M<int, int> rank;

    void add(int node_id)
    {
        if (!parent.contains(node_id)) {
            parent[node_id] = node_id;
            rank[node_id] = 0;
        }
    }

    int find(int node_id)
    {
        add(node_id);
        int root = parent[node_id];
        if (root != node_id) {
            root = find(root);
            parent[node_id] = root;
        }
        return root;
    }

    void unite(int a, int b)
    {
        a = find(a);
        b = find(b);
        if (a == b) {
            return;
        }

        if (rank[a] < rank[b]) {
            std::swap(a, b);
        }

        parent[b] = a;
        if (rank[a] == rank[b]) {
            ++rank[a];
        }
    }
};

inline void collect_render_dependencies(
    NodeEditorElement::Node* node,
    const S<int>& active_node_ids,
    S<int>& visited_node_ids,
    V<NodeEditorElement::Node*>& render_order)
{
    if (node == nullptr) {
        return;
    }

    const int node_id = node->ID.Get();
    if (!active_node_ids.contains(node_id) || visited_node_ids.contains(node_id)) {
        return;
    }

    visited_node_ids.insert(node_id);

    for (const auto& input : node->Inputs) {
        auto connected_pin = node->FindConnectedPin(input.ID);
        if (!connected_pin) {
            continue;
        }

        auto* upstream = NodeEditorElement::Node::FindNodeByPinId(connected_pin);
        collect_render_dependencies(upstream, active_node_ids, visited_node_ids, render_order);
    }

    render_order.emplace_back(node);
}

inline void activate_special_component(
    Node::NODE_TYPE special_type,
    DisjointSet& components,
    S<int>& active_roots)
{
    if (!Node::special_node_spawned(special_type)) {
        return;
    }

    auto* special_node = NodeEditorElement::FindNode(Node::special_node_id(special_type));
    if (special_node == nullptr || special_node->Inputs.empty()) {
        return;
    }

    bool linked = false;
    for (const auto& input : special_node->Inputs) {
        if (special_node->FindConnectedPin(input.ID)) {
            linked = true;
            break;
        }
    }

    if (!linked) {
        return;
    }

    active_roots.insert(components.find(special_node->ID.Get()));
}

inline void build_active_render_plan(
    S<int>& active_node_ids,
    S<int>& active_link_ids,
    V<NodeEditorElement::Node*>& render_order)
{
    struct CachedRenderPlan
    {
        unsigned long long graph_revision = 0;
        int projected_node_id = 0;
        unsigned long long transient_preview_revision = 0;
        S<int> active_node_ids;
        S<int> active_link_ids;
        V<NodeEditorElement::Node*> render_order;
    };

    static CachedRenderPlan cache;
    auto* projected = NodeEditorElement::FindNode(G::projected_node);
    if (projected == nullptr) {
        projected = NodeEditorElement::FindNode(G::active_node);
    }
    const int projected_node_id = projected ? projected->ID.Get() : 0;
    const bool has_transient_previews = Node::has_transient_previews_pending();
    if (!has_transient_previews &&
        cache.graph_revision == Node::graph_revision &&
        cache.projected_node_id == projected_node_id &&
        cache.transient_preview_revision == Node::transient_preview_revision) {
        active_node_ids = cache.active_node_ids;
        active_link_ids = cache.active_link_ids;
        render_order = cache.render_order;
        return;
    }

    active_node_ids.clear();
    active_link_ids.clear();
    render_order.clear();

    DisjointSet components;
    for (const auto& node : NodeEditorElement::nodes) {
        components.add(node->ID.Get());
    }

    for (const auto& link : NodeEditorElement::links) {
        auto* start_node = NodeEditorElement::Node::FindNodeByPinId(link.StartPinID);
        auto* end_node = NodeEditorElement::Node::FindNodeByPinId(link.EndPinID);
        if (start_node != nullptr && end_node != nullptr) {
            components.unite(start_node->ID.Get(), end_node->ID.Get());
        }
    }
    if (projected == nullptr) {
        for (const auto& node : NodeEditorElement::nodes) {
            active_node_ids.insert(node->ID.Get());
        }
    }
    else {
        S<int> active_roots;
        active_roots.insert(components.find(projected->ID.Get()));
        activate_special_component(Node::MYSELF_TEXTEDITOR, components, active_roots);
        activate_special_component(Node::MYSELF_NODELAYER, components, active_roots);
        activate_special_component(Node::MYSELF_SCREEN, components, active_roots);

        for (const auto& node : NodeEditorElement::nodes) {
            if (active_roots.contains(components.find(node->ID.Get()))) {
                active_node_ids.insert(node->ID.Get());
            }
        }
    }

    const auto transient_nodes = Node::consume_transient_preview_node_ids();
    for (const int transient_id : transient_nodes) {
        active_node_ids.insert(transient_id);
    }

    for (const auto& link : NodeEditorElement::links) {
        auto* start_node = NodeEditorElement::Node::FindNodeByPinId(link.StartPinID);
        auto* end_node = NodeEditorElement::Node::FindNodeByPinId(link.EndPinID);
        if (start_node == nullptr || end_node == nullptr) {
            continue;
        }
        if (active_node_ids.contains(start_node->ID.Get()) && active_node_ids.contains(end_node->ID.Get())) {
            active_link_ids.insert(link.ID.Get());
        }
    }

    S<int> visited_node_ids;
    for (const auto& node : NodeEditorElement::nodes) {
        if (active_node_ids.contains(node->ID.Get())) {
            collect_render_dependencies(node.get(), active_node_ids, visited_node_ids, render_order);
        }
    }

    if (!has_transient_previews) {
        cache.graph_revision = Node::graph_revision;
        cache.projected_node_id = projected_node_id;
        cache.transient_preview_revision = Node::transient_preview_revision;
        cache.active_node_ids = active_node_ids;
        cache.active_link_ids = active_link_ids;
        cache.render_order = render_order;
    } else {
        cache.graph_revision = 0;
        cache.projected_node_id = 0;
        cache.transient_preview_revision = 0;
        cache.active_node_ids.clear();
        cache.active_link_ids.clear();
        cache.render_order.clear();
    }
}

inline void refresh_special_node_outputs_for_current_frame()
{
    for (const auto special_type : { Node::MYSELF_TEXTEDITOR, Node::MYSELF_NODELAYER, Node::MYSELF_SCREEN, Node::MYSELF_WINDOWCAPTURE, Node::MYSELF_SYSTEMSOUND }) {
        if (!Node::special_node_spawned(special_type)) {
            continue;
        }

        auto* special_node = NodeEditorElement::FindNode(Node::special_node_id(special_type));
        if (special_node != nullptr) {
            special_node->wire_pins();
        }
    }
}

} // namespace LegacyNodeGraphActivity
