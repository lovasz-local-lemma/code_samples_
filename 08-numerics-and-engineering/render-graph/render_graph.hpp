#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace samples::render
{
using NodeId = std::uint32_t;
enum class Read
{
    current_frame,
    previous_frame
};
struct Input
{
    NodeId producer;
    Read when = Read::current_frame;
};
struct HistoryRead
{
    NodeId consumer, producer;
};
struct Plan
{
    std::vector<NodeId> order;
    std::vector<HistoryRead> history;
    std::uint64_t revision;
};

// UI- and graphics-API-independent extraction of dependency planning concerns.
// History edges must read separately initialized, previous-frame resources.
class Graph
{
  public:
    void add(NodeId id);
    void set_inputs(NodeId id, std::vector<Input> inputs);
    void erase(NodeId id);
    std::shared_ptr<const Plan> plan(std::vector<NodeId> outputs) const;

  private:
    void changed();
    std::map<NodeId, std::vector<Input>> nodes_;
    std::uint64_t revision_ = 0;
    mutable std::vector<NodeId> cached_outputs_;
    mutable std::shared_ptr<const Plan> cache_;
};
} // namespace samples::render
