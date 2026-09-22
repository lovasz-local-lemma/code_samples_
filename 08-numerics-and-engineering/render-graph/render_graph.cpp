#include "samples/render_graph.hpp"
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>

namespace samples::render
{
void Graph::changed()
{
    ++revision_;
    cache_.reset();
}
void Graph::add(NodeId id)
{
    if (!nodes_.emplace(id, std::vector<Input>{}).second)
        throw std::invalid_argument("Duplicate node");
    changed();
}
void Graph::set_inputs(NodeId id, std::vector<Input> inputs)
{
    if (!nodes_.count(id))
        throw std::invalid_argument("Unknown consumer");
    for (const auto &input : inputs)
        if (!nodes_.count(input.producer))
            throw std::invalid_argument("Unknown producer");
    nodes_.at(id) = std::move(inputs);
    changed();
}
void Graph::erase(NodeId id)
{
    if (!nodes_.count(id))
        throw std::invalid_argument("Unknown node");
    for (const auto &[consumer, inputs] : nodes_)
        for (const auto &input : inputs)
            if (consumer != id && input.producer == id)
                throw std::invalid_argument("Node is still referenced");
    nodes_.erase(id);
    changed();
}
std::shared_ptr<const Plan> Graph::plan(std::vector<NodeId> outputs) const
{
    std::sort(outputs.begin(), outputs.end());
    outputs.erase(std::unique(outputs.begin(), outputs.end()), outputs.end());
    if (cache_ && outputs == cached_outputs_)
        return cache_;
    auto result = std::make_shared<Plan>();
    result->revision = revision_;
    enum class Visit
    {
        unseen,
        active,
        complete
    };
    std::map<NodeId, Visit> state;
    std::vector<NodeId> stack;
    std::function<void(NodeId)> visit = [&](NodeId id)
    {
        if (!nodes_.count(id))
            throw std::invalid_argument("Unknown output");
        if (state[id] == Visit::complete)
            return;
        if (state[id] == Visit::active)
        {
            std::string message = "Current-frame cycle:";
            for (auto node : stack)
                message += " " + std::to_string(node);
            throw std::invalid_argument(message + " " + std::to_string(id));
        }
        state[id] = Visit::active;
        stack.push_back(id);
        for (const auto &input : nodes_.at(id))
        {
            if (input.when == Read::previous_frame)
                result->history.push_back({id, input.producer});
            else
                visit(input.producer);
        }
        stack.pop_back();
        state[id] = Visit::complete;
        result->order.push_back(id);
    };
    for (auto id : outputs)
        visit(id);
    cached_outputs_ = std::move(outputs);
    cache_ = result;
    return result;
}
} // namespace samples::render
