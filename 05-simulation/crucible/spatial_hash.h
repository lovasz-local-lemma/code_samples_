#pragma once

#include "core/types.h"
#include "gpu/buffer.h"
#include "gpu/compute_shader.h"

namespace ng {

class ParticleBuffer;

// GPU spatial hash for neighbor queries.
// Uses counting sort: hash → count → prefix sum → scatter.
class SpatialHash {
public:
    struct Config {
        u32   table_size = 262144; // 2^18, must be power of 2
        f32   cell_size  = 0.05f;  // Should be ~2x smoothing radius
        vec2  world_min  = vec2(-10.0f);
        vec2  world_max  = vec2(10.0f);
    };

    void init(const Config& config);

    // Rebuild hash from particle positions. After this, cell_start/cell_end
    // and sorted_indices are valid for neighbor queries.
    void build(ParticleBuffer& particles, u32 particle_offset, u32 particle_count);

    // Bind cell_start and cell_end for neighbor queries
    void bind() const;

    f32 cell_size() const { return cell_size_; }
    u32 table_size() const { return table_size_; }
    vec2 world_min() const { return world_min_; }

    static constexpr u32 BIND_CELL_START    = 13;
    static constexpr u32 BIND_CELL_END      = 14;
    static constexpr u32 BIND_CELL_COUNT    = 15; // temp, also used as write offset
    static constexpr u32 BIND_BLOCK_SUMS    = 16;

private:
    u32 table_size_ = 0;
    f32 cell_size_ = 0.0f;
    vec2 world_min_{0.0f};

    GPUBuffer cell_count_buf_;    // Counts per cell (preserved for cell_end computation)
    GPUBuffer cell_start_buf_;    // Exclusive prefix sum of counts
    GPUBuffer cell_end_buf_;      // cell_start + count
    GPUBuffer block_sums_buf_;    // Temp for prefix sum
    GPUBuffer block_sums_scan_;   // Temp for scanning block sums

    ComputeShader assign_shader_;
    ComputeShader prefix_sum_shader_;
    ComputeShader end_shader_;
    ComputeShader scatter_shader_;

    void exclusive_prefix_sum(GPUBuffer& data, u32 n);
};

} // namespace ng
