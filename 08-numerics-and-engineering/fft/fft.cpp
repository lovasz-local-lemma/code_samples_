#include "samples/fft.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>

namespace samples::fft
{
namespace
{
constexpr auto reversal_table()
{
    std::array<std::uint8_t, 256> table{};
    for (unsigned i = 0; i < table.size(); ++i)
    {
        unsigned v = 0;
        for (unsigned b = 0; b < 8; ++b)
            v = (v << 1) | ((i >> b) & 1u);
        table[i] = static_cast<std::uint8_t>(v);
    }
    return table;
}
constexpr auto reversed_byte = reversal_table();
std::uint32_t reverse(std::uint32_t x, unsigned bits)
{
    const auto byte = [](std::uint32_t v) { return std::uint32_t(reversed_byte[v & 0xffu]); };
    return ((byte(x) << 24) | (byte(x >> 8) << 16) | (byte(x >> 16) << 8) | byte(x >> 24)) >>
           (32 - bits);
}

// Bounded fork/join policy. It is a standard-C++ task backend, not a claim to
// reproduce Cilk's work-stealing scheduler. Each split receives a worker budget.
template <class Function>
void task_range(std::size_t begin, std::size_t end, unsigned workers, std::size_t grain,
                const Function &function)
{
    if (workers <= 1 || (end - begin) / 2 < grain)
    {
        for (auto i = begin; i < end; ++i)
            function(i);
        return;
    }
    const auto middle = begin + (end - begin) / 2;
    const auto left_workers = workers / 2;
    auto left = std::async(std::launch::async, [&function, begin, middle, left_workers, grain]
                           { task_range(begin, middle, left_workers, grain, function); });
    task_range(middle, end, workers - left_workers, grain, function);
    left.get();
}
} // namespace

std::string_view name(Backend b)
{
    switch (b)
    {
    case Backend::serial:
        return "serial";
    case Backend::openmp:
        return "openmp";
    case Backend::tasks:
        return "tasks";
    }
    throw std::invalid_argument("Unknown FFT backend");
}
std::vector<Backend> available_backends()
{
    std::vector<Backend> result;
#ifdef SAMPLE_FFT_SERIAL
    result.push_back(Backend::serial);
#endif
#ifdef SAMPLE_FFT_OPENMP
    result.push_back(Backend::openmp);
#endif
#ifdef SAMPLE_FFT_TASKS
    result.push_back(Backend::tasks);
#endif
    return result;
}

Plan::Plan(std::size_t size, Backend backend, TaskPolicy tasks)
    : size_(size), backend_(backend), tasks_(tasks)
{
    if (!size || (size & (size - 1)))
        throw std::invalid_argument("FFT requires a positive power of two");
    if (size > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("FFT index range exceeded");
    if (!tasks.workers || tasks.workers > 64 || !tasks.minimum_chunk)
        throw std::invalid_argument("Task policy needs 1..64 workers and a positive chunk size");
    const auto available = available_backends();
    if (std::find(available.begin(), available.end(), backend) == available.end())
        throw std::invalid_argument("Requested FFT backend was not compiled in");
    for (auto n = size; n > 1; n >>= 1)
        ++bits_;
    roots_.resize(size / 2);
    const double turn = 2 * std::acos(-1.0);
    for (std::size_t k = 0; k < roots_.size(); ++k)
        roots_[k] = std::polar(1.0, -turn * double(k) / double(size));
}

template <Backend B> void Plan::kernel(std::vector<Complex> &values, Direction direction) const
{
    const bool inverse = direction == Direction::inverse;
    // if constexpr removes task scheduling from serial/OpenMP instantiations.
    const auto range = [&](std::size_t end, const auto &function)
    {
        if constexpr (B == Backend::tasks)
        {
            task_range(0, end, tasks_.workers, tasks_.minimum_chunk, function);
        }
        else
        {
            for (std::size_t i = 0; i < end; ++i)
                function(i);
        }
    };
    const auto permute = [&](std::size_t i)
    {
        const auto j = reverse(static_cast<std::uint32_t>(i), bits_);
        if (j > i)
            std::swap(values[i], values[j]);
    };
    const auto butterfly = [&](std::size_t i, std::size_t half)
    {
        const auto offset = i % half;
        const auto first = (i / half) * (2 * half) + offset;
        const auto root = roots_[offset * (size_ / (2 * half))];
        const auto a = values[first];
        const auto b = values[first + half] * (inverse ? std::conj(root) : root);
        values[first] = a + b;
        values[first + half] = a - b;
    };
    if constexpr (B == Backend::openmp)
    {
#ifdef SAMPLE_FFT_OPENMP
        // One team, with implicit barriers after permutation and every stage.
#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (std::int64_t i = 0; i < static_cast<std::int64_t>(size_); ++i)
                permute(std::size_t(i));
            for (unsigned stage = 0; stage < bits_; ++stage)
            {
                const auto half = std::size_t{1} << stage;
#pragma omp for schedule(static)
                for (std::int64_t i = 0; i < static_cast<std::int64_t>(size_ / 2); ++i)
                    butterfly(std::size_t(i), half);
            }
            if (inverse)
            {
#pragma omp for schedule(static)
                for (std::int64_t i = 0; i < static_cast<std::int64_t>(size_); ++i)
                    values[std::size_t(i)] /= double(size_);
            }
        }
#endif
    }
    else
    {
        range(size_, permute);
        for (unsigned stage = 0; stage < bits_; ++stage)
        {
            const auto half = std::size_t{1} << stage;
            range(size_ / 2, [&](std::size_t i) { butterfly(i, half); });
        }
        if (inverse)
            range(size_, [&](std::size_t i) { values[i] /= double(size_); });
    }
}

void Plan::execute(std::vector<Complex> &values, Direction direction) const
{
    if (values.size() != size_)
        throw std::invalid_argument("FFT buffer length differs from plan");
    if (size_ == 1)
        return;
    switch (backend_)
    {
#ifdef SAMPLE_FFT_SERIAL
    case Backend::serial:
        kernel<Backend::serial>(values, direction);
        return;
#endif
#ifdef SAMPLE_FFT_OPENMP
    case Backend::openmp:
        kernel<Backend::openmp>(values, direction);
        return;
#endif
#ifdef SAMPLE_FFT_TASKS
    case Backend::tasks:
        kernel<Backend::tasks>(values, direction);
        return;
#endif
    default:
        throw std::logic_error("Unavailable backend in FFT plan");
    }
}

Plan Plan::measured(std::size_t size, TaskPolicy tasks)
{
    const auto backends = available_backends();
    if (backends.empty())
        throw std::logic_error("No FFT backends enabled");
    Plan result(size, backends.front(), tasks);
    std::vector<Complex> input(size);
    for (std::size_t i = 0; i < size; ++i)
        input[i] = {std::sin(double(i)), std::cos(double(i) * 0.37)};
    auto reference = input;
    result.execute(reference);
    double best = std::numeric_limits<double>::infinity();
    for (auto backend : backends)
    {
        Plan candidate(size, backend, tasks);
        auto warmup = input;
        candidate.execute(warmup);
        for (std::size_t i = 0; i < size; ++i)
            if (std::abs(warmup[i] - reference[i]) > 1e-10 * (1 + std::abs(reference[i])))
                throw std::runtime_error("FFT backends disagree during planning");
        std::array<double, 3> elapsed{};
        for (auto &duration : elapsed)
        {
            auto scratch = input; // Copy and allocation are outside the timed interval.
            const auto start = std::chrono::steady_clock::now();
            candidate.execute(scratch);
            duration =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start)
                    .count();
        }
        std::sort(elapsed.begin(), elapsed.end());
        result.timings_.push_back({backend, elapsed[1]});
        if (elapsed[1] < best)
        {
            best = elapsed[1];
            result.backend_ = backend;
        }
    }
    return result;
}
} // namespace samples::fft
