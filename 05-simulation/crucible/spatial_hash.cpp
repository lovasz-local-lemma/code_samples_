#include "physics/common/spatial_hash.h"
#include "physics/common/particle_buffer.h"
#include "core/log.h"

#include <glad/gl.h>

namespace ng {

void SpatialHash::init(const Config& config) {
    table_size_ = config.table_size;
    cell_size_ = config.cell_size;
    world_min_ = config.world_min;

    cell_count_buf_.create(table_size_ * sizeof(u32));
    cell_start_buf_.create(table_size_ * sizeof(u32));
    cell_end_buf_.create(table_size_ * sizeof(u32));

    // For prefix sum: one block sum per 512 elements
    u32 num_blocks = (table_size_ + 511) / 512;
    block_sums_buf_.create(num_blocks * sizeof(u32));
    block_sums_scan_.create(num_blocks * sizeof(u32)); // for scanning block sums

    assign_shader_.load("shaders/physics/spatial_hash_assign.comp");
    prefix_sum_shader_.load("shaders/physics/prefix_sum.comp");
    end_shader_.load("shaders/physics/spatial_hash_count.comp");
    scatter_shader_.load("shaders/physics/spatial_hash_scatter.comp");

    LOG_INFO("SpatialHash: table_size=%u, cell_size=%.4f, blocks=%u", table_size_, cell_size_, num_blocks);
}

void SpatialHash::exclusive_prefix_sum(GPUBuffer& data, u32 n) {
    u32 num_blocks = (n + 511) / 512;

    // Bind data buffer and block sums
    data.bind_base(0);  // prefix_sum shader reads/writes binding 0
    block_sums_buf_.bind_base(1); // block sums at binding 1

    prefix_sum_shader_.bind();

    // Pass 1: Local scan per block, extract block totals
    prefix_sum_shader_.set_uint("u_n", n);
    prefix_sum_shader_.set_int("u_mode", 0);
    prefix_sum_shader_.dispatch(num_blocks, 1, 1);
    ComputeShader::barrier_ssbo();

    // Pass 2: Scan block totals
    if (num_blocks > 1) {
        // Copy block_sums to data binding for scanning
        block_sums_buf_.bind_base(0);
        block_sums_scan_.bind_base(1);
        prefix_sum_shader_.set_uint("u_n", num_blocks);
        prefix_sum_shader_.set_int("u_mode", 1);
        prefix_sum_shader_.dispatch(1, 1, 1); // Single workgroup for block sums
        ComputeShader::barrier_ssbo();

        // Pass 3: Add block offsets back
        data.bind_base(0);
        block_sums_buf_.bind_base(1);
        prefix_sum_shader_.set_uint("u_n", n);
        prefix_sum_shader_.set_int("u_mode", 2);
        prefix_sum_shader_.dispatch(num_blocks, 1, 1);
        ComputeShader::barrier_ssbo();
    }
}

void SpatialHash::build(ParticleBuffer& particles, u32 particle_offset, u32 particle_count) {
    if (particle_count == 0) return;

    particles.bind_all();

    // Step 1: Clear cell counts
    cell_count_buf_.bind_base(BIND_CELL_COUNT);
    cell_count_buf_.clear();
    ComputeShader::barrier_ssbo();

    // Step 2: Hash particles to cells, count per cell
    assign_shader_.bind();
    assign_shader_.set_uint("u_particle_offset", particle_offset);
    assign_shader_.set_uint("u_particle_count", particle_count);
    assign_shader_.set_uint("u_table_size", table_size_);
    assign_shader_.set_float("u_cell_size", cell_size_);
    assign_shader_.set_vec2("u_world_min", world_min_);
    assign_shader_.dispatch_1d(particle_count);
    ComputeShader::barrier_ssbo();

    // Step 3: Copy counts (we need originals for cell_end later)
    glCopyNamedBufferSubData(
        cell_count_buf_.handle(), cell_start_buf_.handle(),
        0, 0, static_cast<GLsizeiptr>(table_size_ * sizeof(u32)));
    ComputeShader::barrier_ssbo();

    // Step 4: Exclusive prefix sum on cell_start (converts counts to starts)
    exclusive_prefix_sum(cell_start_buf_, table_size_);

    // Step 5: Compute cell_end = cell_start + cell_count
    cell_start_buf_.bind_base(BIND_CELL_START);
    cell_count_buf_.bind_base(BIND_CELL_COUNT);
    cell_end_buf_.bind_base(BIND_CELL_END);
    end_shader_.bind();
    end_shader_.set_uint("u_table_size", table_size_);
    end_shader_.dispatch_1d(table_size_);
    ComputeShader::barrier_ssbo();

    // Step 6: Clear cell_count (reuse as per-cell write offset for scatter)
    cell_count_buf_.clear();
    ComputeShader::barrier_ssbo();

    // Step 7: Scatter particles into sorted order
    particles.bind_all(); // re-bind particle SSBOs
    cell_start_buf_.bind_base(BIND_CELL_START);
    cell_count_buf_.bind_base(BIND_CELL_COUNT); // write offsets
    scatter_shader_.bind();
    scatter_shader_.set_uint("u_particle_offset", particle_offset);
    scatter_shader_.set_uint("u_particle_count", particle_count);
    scatter_shader_.set_uint("u_table_size", table_size_);
    scatter_shader_.set_float("u_cell_size", cell_size_);
    scatter_shader_.set_vec2("u_world_min", world_min_);
    scatter_shader_.dispatch_1d(particle_count);
    ComputeShader::barrier_ssbo();
}

void SpatialHash::bind() const {
    cell_start_buf_.bind_base(BIND_CELL_START);
    cell_end_buf_.bind_base(BIND_CELL_END);
}

} // namespace ng
