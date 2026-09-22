#pragma once
#include <cstddef>
#include <vector>

namespace samples::imaging
{
struct Image
{
    std::size_t width, height, channels;
    std::vector<double> pixels; // Interleaved, row major; 1..4 channels.
    void validate() const;
};
struct GridParameters
{
    double spatial_sigma = 2;
    double range_sigma = 0.1;
    double spatial_step = 1;
    double range_step = 0.05;
    double gaussian_cutoff = 3;
    std::size_t max_cells = 2000000;
};
// Joint bilateral filtering: guide is one finite [0,1] value per pixel.
// Signal channels are splatted with a weight channel, blurred, then normalized
// after trilinear slicing. The guide is explicit rather than silently treating
// independent RGB channels as unrelated range coordinates.
Image bilateral_grid(const Image &signal, const std::vector<double> &guide,
                     GridParameters parameters = {});
} // namespace samples::imaging
