#include "samples/bilateral_grid.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace samples::imaging
{
void Image::validate() const
{
    if (!width || !height || !channels || channels > 4 ||
        width > std::numeric_limits<std::size_t>::max() / height / channels ||
        pixels.size() != width * height * channels)
        throw std::invalid_argument("Invalid image dimensions or storage");
    for (auto v : pixels)
        if (!std::isfinite(v))
            throw std::invalid_argument("Nonfinite signal");
}
Image bilateral_grid(const Image &signal, const std::vector<double> &guide, GridParameters p)
{
    signal.validate();
    if (guide.size() != signal.width * signal.height)
        throw std::invalid_argument("Guide size mismatch");
    for (auto g : guide)
        if (!std::isfinite(g) || g < 0 || g > 1)
            throw std::invalid_argument("Guide must lie in [0,1]");
    for (auto value :
         {p.spatial_sigma, p.range_sigma, p.spatial_step, p.range_step, p.gaussian_cutoff})
        if (!std::isfinite(value) || value <= 0)
            throw std::invalid_argument("Grid parameters must be positive");
    if (!p.max_cells || p.max_cells > std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Invalid grid memory budget");
    const auto bounded_ceil = [&](double x) -> std::size_t
    {
        if (!std::isfinite(x) || x < 0 || x > double(p.max_cells))
            throw std::length_error("Bilateral grid exceeds memory budget");
        return static_cast<std::size_t>(std::ceil(x));
    };
    const std::array<double, 3> sigma{p.spatial_sigma / p.spatial_step,
                                      p.spatial_sigma / p.spatial_step,
                                      p.range_sigma / p.range_step};
    std::array<std::size_t, 3> radius{}, padding{}, dimensions{};
    const std::array<double, 3> extent{double(signal.width - 1) / p.spatial_step,
                                       double(signal.height - 1) / p.spatial_step,
                                       1.0 / p.range_step};
    std::size_t cells = 1;
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        radius[axis] = bounded_ceil(p.gaussian_cutoff * sigma[axis]);
        padding[axis] = radius[axis] + 1;
        dimensions[axis] = bounded_ceil(extent[axis]) + 2 * padding[axis] + 2;
        if (dimensions[axis] > p.max_cells / cells)
            throw std::length_error("Bilateral grid exceeds memory budget");
        cells *= dimensions[axis];
    }
    const std::array<std::size_t, 3> stride{1, dimensions[0], dimensions[0] * dimensions[1]};
    using Cell = std::array<double, 5>; // Up to four signal channels, then weight.
    std::vector<Cell> grid(cells), scratch(cells);
    const auto visit_corners = [&](std::size_t pixel, const auto &visitor)
    {
        const std::array<double, 3> position{
            double(pixel % signal.width) / p.spatial_step + double(padding[0]),
            double(pixel / signal.width) / p.spatial_step + double(padding[1]),
            guide[pixel] / p.range_step + double(padding[2])};
        std::array<std::size_t, 3> base{};
        std::array<double, 3> fraction{};
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            base[axis] = static_cast<std::size_t>(std::floor(position[axis]));
            fraction[axis] = position[axis] - double(base[axis]);
        }
        for (unsigned corner = 0; corner < 8; ++corner)
        {
            std::size_t index = 0;
            double weight = 1;
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                const auto bit = (corner >> axis) & 1;
                index += (base[axis] + bit) * stride[axis];
                weight *= bit ? fraction[axis] : 1 - fraction[axis];
            }
            visitor(index, weight);
        }
    };
    for (std::size_t pixel = 0; pixel < guide.size(); ++pixel)
    {
        visit_corners(pixel,
                      [&](std::size_t index, double weight)
                      {
                          for (std::size_t c = 0; c < signal.channels; ++c)
                              grid[index][c] += weight * signal.pixels[pixel * signal.channels + c];
                          grid[index][signal.channels] += weight;
                      });
    }
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        const auto r = static_cast<int>(radius[axis]);
        std::vector<double> kernel(std::size_t(2 * r + 1));
        double sum = 0;
        for (int offset = -r; offset <= r; ++offset)
        {
            const auto q = double(offset) / sigma[axis];
            sum += kernel[std::size_t(offset + r)] = std::exp(-0.5 * q * q);
        }
        for (auto &value : kernel)
            value /= sum;
        for (std::size_t index = 0; index < cells; ++index)
        {
            Cell filtered{};
            const auto coordinate = static_cast<int>((index / stride[axis]) % dimensions[axis]);
            for (int offset = -r; offset <= r; ++offset)
            {
                const auto neighbor = coordinate + offset;
                if (neighbor < 0 || neighbor >= static_cast<int>(dimensions[axis]))
                    continue;
                const auto source = index - std::size_t(coordinate) * stride[axis] +
                                    std::size_t(neighbor) * stride[axis];
                for (std::size_t c = 0; c <= signal.channels; ++c)
                    filtered[c] += kernel[std::size_t(offset + r)] * grid[source][c];
            }
            scratch[index] = filtered;
        }
        grid.swap(scratch);
    }
    Image output = signal;
    for (std::size_t pixel = 0; pixel < guide.size(); ++pixel)
    {
        Cell sliced{};
        visit_corners(pixel,
                      [&](std::size_t index, double weight)
                      {
                          for (std::size_t c = 0; c <= signal.channels; ++c)
                              sliced[c] += weight * grid[index][c];
                      });
        if (!(sliced[signal.channels] > 0))
            throw std::runtime_error("Empty bilateral-grid support");
        for (std::size_t c = 0; c < signal.channels; ++c)
            output.pixels[pixel * signal.channels + c] = sliced[c] / sliced[signal.channels];
    }
    return output;
}
} // namespace samples::imaging
