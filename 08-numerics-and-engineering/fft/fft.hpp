#pragma once
#include <complex>
#include <cstddef>
#include <string_view>
#include <vector>

namespace samples::fft
{
using Complex = std::complex<double>;
enum class Backend
{
    serial,
    openmp,
    tasks
};
enum class Direction
{
    forward,
    inverse
};
struct Timing
{
    Backend backend;
    double median_microseconds;
};
std::string_view name(Backend backend);
std::vector<Backend> available_backends();

struct TaskPolicy
{
    unsigned workers = 4;
    std::size_t minimum_chunk = 1024;
};

// Roots and selected backend belong to a plan, not to global mutable storage.
// A const plan may execute on separate buffers concurrently; OpenMP team setup
// and task-worker policy remain the caller's resource-management choices.
class Plan
{
  public:
    Plan(std::size_t size, Backend backend, TaskPolicy tasks = {});
    static Plan measured(std::size_t size, TaskPolicy tasks = {});
    void execute(std::vector<Complex> &values, Direction direction = Direction::forward) const;
    Backend backend() const noexcept
    {
        return backend_;
    }
    const std::vector<Timing> &timings() const noexcept
    {
        return timings_;
    }
    std::size_t size() const noexcept
    {
        return size_;
    }

  private:
    template <Backend B> void kernel(std::vector<Complex> &values, Direction direction) const;
    std::size_t size_;
    unsigned bits_ = 0;
    Backend backend_;
    TaskPolicy tasks_;
    std::vector<Complex> roots_;
    std::vector<Timing> timings_;
};
} // namespace samples::fft
