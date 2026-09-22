#include "physics/eulerian/shock_front.h"
#include "physics/eulerian/euler_fluid.h"
#include "physics/sdf/sdf_field.h"
#include "physics/common/particle_buffer.h"
#include "core/log.h"

#include <glad/gl.h>
#include <algorithm>
#include <cmath>

namespace ng {

void ShockFrontSolver::init(u32 particle_capacity) {
    particle_capacity_ = particle_capacity;
    fronts_buf_.create(kMaxFronts * 3 * sizeof(vec4));
    guard_buf_.create(particle_capacity * sizeof(u32));
    guard_buf_.clear_u32(0xFFFFFFFFu); // no front has hit any particle yet

    if (!source_shader_.load("shaders/physics/shock_front_source.comp"))
        LOG_ERROR("FAILED to load shock_front_source.comp!");
    if (!wind_shader_.load("shaders/physics/shock_front_wind.comp"))
        LOG_ERROR("FAILED to load shock_front_wind.comp!");
    if (!particle_shader_.load("shaders/physics/shock_front_particles.comp"))
        LOG_ERROR("FAILED to load shock_front_particles.comp!");
}

u32 ShockFrontSolver::spawn(vec2 origin, f32 energy, f32 time,
                            vec2 dir, f32 cone_cos, f32 push_mul, f32 heat_mul) {
    energy = std::max(energy * params.energy_scale, 0.0f);
    if (energy <= 1e-4f) return 0u;
    if (fronts_.size() >= kMaxFronts) {
        // Drop the oldest (smallest birth_time) to make room.
        auto oldest = std::min_element(fronts_.begin(), fronts_.end(),
            [](const Front& a, const Front& b) { return a.birth_time < b.birth_time; });
        if (oldest != fronts_.end()) fronts_.erase(oldest);
    }
    f32 dl = std::sqrt(dir.x * dir.x + dir.y * dir.y);
    vec2 ndir = (dl > 1e-5f) ? vec2(dir.x / dl, dir.y / dl) : vec2(0.0f, 1.0f);
    Front f;
    f.origin = origin;
    f.energy = energy;
    f.birth_time = time;
    f.r_prev = 0.0f;
    f.r_now = 0.0f;
    f.dir = ndir;
    f.cone_cos = cone_cos;
    f.push_mul = push_mul;
    f.heat_mul = heat_mul;
    f.id = next_id_++;
    fronts_.push_back(f);
    return f.id;
}

void ShockFrontSolver::seed_guard(u32 offset, u32 count, u32 front_id) {
    if (count == 0 || offset >= particle_capacity_) return;
    u32 c = std::min(count, particle_capacity_ - offset);
    // Fill guard_buf_[offset .. offset+c] with front_id. glClearNamedBufferSubData
    // fills a sub-range with a single value (DSA, no binding needed) -- the sub-range
    // analog of GPUBuffer::clear_u32. GL orders this write before the particle pass
    // reads the guard later in the frame (same buffer, same context).
    glClearNamedBufferSubData(guard_buf_.handle(), GL_R32UI,
        static_cast<GLintptr>(offset) * sizeof(u32),
        static_cast<GLsizeiptr>(c) * sizeof(u32),
        GL_RED_INTEGER, GL_UNSIGNED_INT, &front_id);
}

void ShockFrontSolver::detect_vent_fronts(EulerianFluid& air, f32 time) {
    vent_circles_.clear();
    if (!params.vent_fronts_enabled) { vent_prev_valid_ = false; return; }

    const ivec2 res = air.airtight_resolution();
    const int W = res.x, H = res.y;
    const size_t N = static_cast<size_t>(W) * static_cast<size_t>(H);
    const std::vector<f32>& out = air.airtight_outside_cpu();
    const std::vector<f32>& prs = air.airtight_pressure_cpu();
    if (out.size() < N || prs.size() < N) return;

    // Need one frame of history before diffing.
    if (!vent_prev_valid_ || prev_outside_.size() != N) {
        prev_outside_.assign(out.begin(), out.begin() + N);
        prev_pressure_.assign(prs.begin(), prs.begin() + N);
        vent_prev_valid_ = true;
        return;
    }

    // 1) Flag cells that just went sealed->open while holding pressure (hysteresis
    //    against shell-wobble flicker: full transition + prior stored pressure).
    std::vector<uint8_t> flag(N, 0);
    for (size_t i = 0; i < N; ++i) {
        if (prev_outside_[i] < 0.3f && out[i] > 0.7f && prev_pressure_[i] > params.vent_p_min)
            flag[i] = 1;
    }

    // 2) 4-neighbour flood cluster over the flagged cells (small grid, cheap).
    struct Cluster { double sx = 0, sy = 0, sumP = 0; int n = 0; };
    std::vector<int> label(N, -1);
    std::vector<int> stack;
    std::vector<Cluster> cls;
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        size_t i = static_cast<size_t>(y) * W + x;
        if (!flag[i] || label[i] >= 0) continue;
        int L = static_cast<int>(cls.size());
        cls.push_back({});
        stack.clear(); stack.push_back(static_cast<int>(i)); label[i] = L;
        while (!stack.empty()) {
            int c = stack.back(); stack.pop_back();
            int cx = c % W, cy = c / W;
            Cluster& cl = cls[L];
            cl.sx += cx; cl.sy += cy; cl.sumP += prev_pressure_[c]; cl.n++;
            const int dx4[4] = {-1, 1, 0, 0}, dy4[4] = {0, 0, -1, 1};
            for (int k = 0; k < 4; ++k) {
                int nx = cx + dx4[k], ny = cy + dy4[k];
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                size_t j = static_cast<size_t>(ny) * W + nx;
                if (flag[j] && label[j] < 0) { label[j] = L; stack.push_back(static_cast<int>(j)); }
            }
        }
    }

    // 3) Rank clusters by stored pressure; spawn a cone vent front out of the top ones.
    std::vector<int> order(cls.size());
    for (size_t k = 0; k < cls.size(); ++k) order[k] = static_cast<int>(k);
    std::sort(order.begin(), order.end(),
        [&](int a, int b) { return cls[a].sumP > cls[b].sumP; });

    const vec2 wmin = air.world_min(), wmax = air.world_max();
    const f32 adx = air.airtight_dx();
    const f32 cell_area = adx * adx;
    const f32 cone_cos = std::cos(params.vent_cone_deg * 3.14159265f / 180.0f);
    auto cell_to_world = [&](f32 cx, f32 cy) {
        vec2 uv((cx + 0.5f) / W, (cy + 0.5f) / H);
        return vec2(wmin.x + uv.x * (wmax.x - wmin.x), wmin.y + uv.y * (wmax.y - wmin.y));
    };
    auto read_out = [&](int x, int y) -> f32 {
        if (x < 0 || y < 0 || x >= W || y >= H) return 1.0f;
        return out[static_cast<size_t>(y) * W + x];
    };

    int spawned = 0;
    for (int oi = 0; oi < static_cast<int>(order.size()) &&
                     spawned < params.vent_max_per_frame; ++oi) {
        const Cluster& cl = cls[order[oi]];
        if (cl.n < 1) continue;
        f32 gcx = static_cast<f32>(cl.sx / cl.n), gcy = static_cast<f32>(cl.sy / cl.n);
        // Aperture direction = outside gradient (same central-difference stencil the
        // legacy vent branch uses in euler_airtight_update.comp).
        int ix = static_cast<int>(gcx + 0.5f), iy = static_cast<int>(gcy + 0.5f);
        vec2 g(read_out(ix + 1, iy) - read_out(ix - 1, iy),
               read_out(ix, iy + 1) - read_out(ix, iy - 1));
        f32 gl = std::sqrt(g.x * g.x + g.y * g.y);
        vec2 dir = (gl > 1e-4f) ? vec2(g.x / gl, g.y / gl) : vec2(0.0f, 1.0f);
        vec2 center = cell_to_world(gcx, gcy);
        vec2 origin = center + dir * (1.5f * adx); // born just outside the aperture
        f32 E = params.k_vent * static_cast<f32>(cl.sumP) * cell_area * params.vent_area_k;
        // Vent fronts deliver impulse only; cavity heat stays with the legacy branch
        // (now operating on the deducted pressure) -> single-owner energy.
        spawn(origin, E, time, dir, cone_cos, /*push_mul*/ 1.0f, /*heat_mul*/ 0.0f);
        f32 rad = std::max(std::sqrt(static_cast<f32>(cl.n)) * adx, 2.0f * adx);
        vent_circles_.emplace_back(center.x, center.y, rad, 1.0f - params.k_vent);
        ++spawned;
    }

    // Roll history forward.
    prev_outside_.assign(out.begin(), out.begin() + N);
    prev_pressure_.assign(prs.begin(), prs.begin() + N);
}

void ShockFrontSolver::spawn_backdraft_fronts(const std::vector<u32>& bins, ivec2 bg,
                                              vec2 wmin, const SDFField* sdf, f32 time) {
    const size_t need = static_cast<size_t>(bg.x) * static_cast<size_t>(bg.y) * 3;
    if (bins.size() < need) return; // feature off / not downloaded this frame
    for (int by = 0; by < bg.y; ++by) for (int bx = 0; bx < bg.x; ++bx) {
        int base = (by * bg.x + bx) * 3;
        f32 sum = bins[base + 0] * 0.001f;
        if (sum < 0.6f) continue;  // ignore tiny / diffuse flashes
        // Mass-weighted centroid (moments stored relative to world_min).
        f32 cx = bins[base + 1] * 0.001f / sum + wmin.x;
        f32 cy = bins[base + 2] * 0.001f / sum + wmin.y;
        vec2 c(cx, cy);
        if (sdf && sdf->sample_cpu(c) < 0.0f) continue; // centroid inside a wall
        spawn(c, params.k_bd * sum, time); // small isotropic gen-0 front
    }
}

void ShockFrontSolver::update(f32 time) {
    const f32 rho0 = params.rho_air;
    const f32 xi = std::max(params.xi, 0.05f);
    const f32 c0 = std::max(params.c0, 1.0f);
    // Sedov speed drops below c0 at t_trans; after that the front coasts at c0.
    // D(t) = 0.5*xi*(E/rho0)^(1/4) * t^(-1/2). Solve D = c0.
    for (auto& f : fronts_) {
        f32 t = std::max(time - f.birth_time, 0.0f);
        f32 k = xi * std::pow(std::max(f.energy, 1e-4f) / rho0, 0.25f);
        f32 t_trans = (k / (2.0f * c0));
        t_trans = t_trans * t_trans;
        f32 r;
        if (t <= t_trans) {
            r = k * std::sqrt(t);
        } else {
            f32 r_trans = k * std::sqrt(t_trans);
            r = r_trans + c0 * (t - t_trans);
        }
        f.r_prev = f.r_now;
        f.r_now = r;
    }
    // Retire fronts whose leading overpressure has faded, or that ran off-world.
    fronts_.erase(std::remove_if(fronts_.begin(), fronts_.end(),
        [&](const Front& f) {
            if (f.r_now > params.max_radius) return true; // trimmed tail (calibrated)
            f32 dp = params.k_p * f.energy / (f.r_now * f.r_now + 0.01f);
            return dp < 0.02f && f.r_now > 0.3f;       // faded below a useful push
        }), fronts_.end());

    upload();
}

void ShockFrontSolver::upload() {
    packed_.clear();
    packed_.reserve(fronts_.size() * 3);
    for (const auto& f : fronts_) {
        packed_.emplace_back(f.origin.x, f.origin.y, f.energy, params.rho_air);
        packed_.emplace_back(f.r_prev, f.r_now, f.cone_cos, f.push_mul);
        packed_.emplace_back(f.dir.x, f.dir.y, f.heat_mul,
                             static_cast<f32>(f.id));
    }
    if (!packed_.empty())
        fronts_buf_.upload(packed_.data(), packed_.size() * sizeof(vec4));
}

f32 ShockFrontSolver::radius_of(u32 id) const {
    for (const auto& f : fronts_) if (f.id == id) return f.r_now;
    // Negative sentinel for absent/retired -- distinct from a LIVE front whose r_now
    // is legitimately 0 on its birth substep (t = time - birth_time = 0). Returning
    // 0 here would make the renderer untag a just-born ring and freeze it at r=0.
    return -1.0f;
}

void ShockFrontSolver::dispatch_source(EulerianFluid& air, const SDFField* sdf) {
    if (fronts_.empty()) return;
    air.bind_fields();
    fronts_buf_.bind_base(BIND_FRONTS);
    if (sdf) sdf->bind_for_read(0);
    source_shader_.bind();
    source_shader_.set_ivec2("u_res", air.resolution());
    source_shader_.set_float("u_dx", air.dx());
    source_shader_.set_vec2("u_world_min", air.world_min());
    source_shader_.set_vec2("u_world_max", air.world_max());
    source_shader_.set_int("u_use_sdf", sdf ? 1 : 0);
    if (sdf) source_shader_.set_int("u_sdf_tex", 0);
    source_shader_.set_int("u_front_count", static_cast<i32>(fronts_.size()));
    source_shader_.set_float("u_xi", params.xi);
    source_shader_.set_float("u_c0", params.c0);
    source_shader_.set_float("u_kp", params.k_p);
    source_shader_.set_float("u_kheat", params.k_heat);
    source_shader_.set_float("u_ksmoke", params.k_smoke);
    source_shader_.set_float("u_thickness", params.thickness);
    source_shader_.set_float("u_wind_frac", params.wind_frac);
    source_shader_.set_float("u_vapor_pressure", air.config().vapor_pressure);
    source_shader_.set_float("u_ambient_temp", air.config().ambient_temp);
    source_shader_.dispatch_1d(air.resolution().x * air.resolution().y);
    ComputeShader::barrier_ssbo();
}

void ShockFrontSolver::dispatch_wind(EulerianFluid& air, const SDFField* sdf) {
    if (fronts_.empty() || !params.wind_enabled) return;
    air.bind_fields();
    fronts_buf_.bind_base(BIND_FRONTS);
    if (sdf) sdf->bind_for_read(0);
    wind_shader_.bind();
    wind_shader_.set_ivec2("u_res", air.resolution());
    wind_shader_.set_float("u_dx", air.dx());
    wind_shader_.set_vec2("u_world_min", air.world_min());
    wind_shader_.set_vec2("u_world_max", air.world_max());
    wind_shader_.set_int("u_use_sdf", sdf ? 1 : 0);
    if (sdf) wind_shader_.set_int("u_sdf_tex", 0);
    wind_shader_.set_int("u_front_count", static_cast<i32>(fronts_.size()));
    wind_shader_.set_float("u_xi", params.xi);
    wind_shader_.set_float("u_c0", params.c0);
    wind_shader_.set_float("u_kp", params.k_p);
    wind_shader_.set_float("u_thickness", params.thickness);
    wind_shader_.set_float("u_wind_frac", params.wind_frac);
    wind_shader_.set_float("u_rho_air", params.rho_air);
    wind_shader_.dispatch_1d(air.resolution().x * air.resolution().y);
    ComputeShader::barrier_ssbo();
}

void ShockFrontSolver::dispatch_particles(ParticleBuffer& particles, u32 offset, u32 count,
                                          const SDFField* sdf, vec2 air_world_min,
                                          vec2 air_world_max, f32 air_dx, f32 physics_dt) {
    if (fronts_.empty() || count == 0) return;
    particles.bind_all();
    fronts_buf_.bind_base(BIND_FRONTS);
    guard_buf_.bind_base(BIND_GUARD);
    if (sdf) sdf->bind_for_read(0);
    particle_shader_.bind();
    particle_shader_.set_uint("u_offset", offset);
    particle_shader_.set_uint("u_count", count);
    particle_shader_.set_int("u_front_count", static_cast<i32>(fronts_.size()));
    particle_shader_.set_int("u_use_sdf", sdf ? 1 : 0);
    if (sdf) particle_shader_.set_int("u_sdf_tex", 0);
    particle_shader_.set_vec2("u_world_min", air_world_min);
    particle_shader_.set_vec2("u_world_max", air_world_max);
    particle_shader_.set_float("u_dx", air_dx);
    particle_shader_.set_float("u_xi", params.xi);
    particle_shader_.set_float("u_c0", params.c0);
    particle_shader_.set_float("u_kp", params.k_p);
    particle_shader_.set_float("u_kimp", params.k_imp);
    particle_shader_.set_float("u_thickness", params.thickness);
    particle_shader_.set_float("u_dt", physics_dt);
    particle_shader_.dispatch_1d(count);
    ComputeShader::barrier_ssbo();
}

} // namespace ng
