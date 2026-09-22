#pragma once
#include "core/types.h"
#include "gpu/buffer.h"
#include "gpu/compute_shader.h"
#include <vector>

// ShockFrontSolver -- analytic Sedov-Taylor blast fronts.
//
// Each explosion/rupture spawns a Front: an expanding ring whose radius follows the
// 2D Sedov-Taylor law r(t) = xi*(E t^2 / rho0)^(1/4), transitioning to a constant
// acoustic speed c0 once the front slows below it. One energy scalar E drives the
// front speed, overpressure, heat and smoke together, rather than independent
// polynomials per effect. Every physics substep the CPU advances r_prev->r_now and
// uploads the front list; three GPU passes then read the annulus r_prev < |x-o| <=
// r_now and deposit:
//   - source  (before divergence): a divergence source + heat + smoke -> the
//     projection turns it into wall-aware outward flow (the punch that survives).
//   - wind    (after project):     a one-frame directional gust on the MAC faces.
//   - particles:                   an outward impulse on MPM/SPH particles, each
//     particle hit at most once per front (persistent guard buffer).
//
// dt convention: front-crossing deposits are IMPULSES consumed by exactly one
// projection per frame, so they are written dt-FREE (applying u_dt would make the
// delivered kick scale with frame rate). Continuous emitters (euler_inject) stay
// rate-based.

namespace ng {

class SDFField;
class EulerianFluid;
class ParticleBuffer;

class ShockFrontSolver {
public:
    struct Params {
        f32 xi          = 1.0f;   // Sedov constant (2D, tunable feel)
        f32 c0          = 22.0f;  // acoustic transition speed (m/s)
        f32 k_p         = 1.0f;   // overpressure gain: dp = k_p*E/(r^2 + r_c^2)
        f32 k_heat      = 90.0f;  // heat gain: dT = k_heat*dp
        f32 k_imp       = 0.9f;   // particle impulse gain
        f32 k_smoke     = 0.35f;  // smoke gain (*sqrt E)
        f32 thickness   = 4.0f;   // front thickness in air cells
        f32 max_radius  = 12.0f;  // retire a front past this radius (world units). The
                                  // world is ~6 wide, so 12 (~2x) trims the wasted tail
                                  // where dp has decayed to nothing without cutting the
                                  // near-field.
        f32 wind_frac   = 0.35f;  // fraction of the momentum budget to the wind kick
        f32 rho_air     = 1.2f;   // air density for the wind kick
        f32 energy_scale = 1.0f;  // master: spawned E = energy_scale * source energy
        bool wind_enabled = true; // wind pass on/off (A/B knob)

        // Non-vessel vent fronts (default OFF): dug-wall / heat-gun-flask
        // ruptures spawn Sedov cone fronts from the airtight pressure release.
        bool vent_fronts_enabled = false;
        f32  k_vent      = 0.5f;   // fraction of stored P converted to a vent front
        f32  vent_p_min  = 40.0f;  // hysteresis: prev cavity pressure must exceed this
        f32  vent_cone_deg = 35.0f;// jet aperture half-angle (deg)
        i32  vent_max_per_frame = 2;
        f32  vent_area_k = 1.0f;   // E_vent = k_vent * sum_P * airtight_dx^2 * vent_area_k

        // Backdraft flash fronts (gated by EulerianFluid::Config::
        // backdraft_fronts_enabled). Energy gain: E = k_bd * bin_sum.
        f32  k_bd = 0.5f;
    };
    Params params;

    void init(u32 particle_capacity);

    // Spawn a front. dir/cone_cos define a directional jet (cone_cos = -1 -> isotropic).
    // Returns the assigned front id (>=1), or 0 if no front was spawned (energy too low).
    u32 spawn(vec2 origin, f32 energy, f32 time,
              vec2 dir = vec2(0.0f, 1.0f), f32 cone_cos = -1.0f,
              f32 push_mul = 1.0f, f32 heat_mul = 1.0f);

    // Pre-seed guard_buf_[offset .. offset+count] with front_id so shock_front_particles
    // skips those particles for THAT front (owner exemption: the rupturing vessel's own
    // casing is carried by the legacy mech_burst pushes and must not be double-kicked).
    void seed_guard(u32 offset, u32 count, u32 front_id);

    // Non-vessel vent fronts: diff the airtight outside/pressure fields
    // (retained on CPU by air.update_airtight_from_particles) with hysteresis to find
    // newly-opened apertures, cluster them, and spawn <=vent_max_per_frame cone fronts
    // pointing out through each aperture. Records the deduct circles for the caller to
    // hand to air.set_vent_deduct_clusters(). No-op unless params.vent_fronts_enabled.
    void detect_vent_fronts(EulerianFluid& air, f32 time);
    const std::vector<vec4>& vent_deduct_circles() const { return vent_circles_; }

    // Backdraft flash fronts: consume the binned flash reduction
    // (air.backdraft_bins(), 3 fixed-point uints/bin) and spawn one small isotropic
    // front per compact flash cluster whose centroid is not inside a wall. No-op if
    // bins is empty (feature off). 1-frame latency (reads last frame's flash).
    void spawn_backdraft_fronts(const std::vector<u32>& bins, ivec2 bin_grid,
                                vec2 world_min, const SDFField* sdf, f32 time);

    // Advance all fronts to `time`; retire faded/expired ones; upload the SSBO.
    void update(f32 time);

    bool empty() const { return fronts_.empty(); }
    i32  active_count() const { return static_cast<i32>(fronts_.size()); }

    // GPU passes. `air` supplies grid geometry + field bindings; `sdf` gives occlusion.
    void dispatch_source(EulerianFluid& air, const SDFField* sdf);   // before divergence
    void dispatch_wind(EulerianFluid& air, const SDFField* sdf);     // after project
    void dispatch_particles(ParticleBuffer& particles, u32 offset, u32 count,
                            const SDFField* sdf, vec2 air_world_min, vec2 air_world_max,
                            f32 air_dx, f32 physics_dt);

    // Radius of a live front (for the renderer ring sync); 0 if not found.
    f32 radius_of(u32 id) const;

    static constexpr u32 BIND_FRONTS = 95;
    // 94, not 96: GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS is 96 on this driver, so the
    // valid binding range is 0..95 (96 fails to compile). Slot 94 is euler's
    // fuel_vapor2, but the particle pass runs AFTER g_air.step() finishes and euler
    // re-binds 94 via bind_all() on the next step, so borrowing it here is safe --
    // the same temporary-slot pattern the source/wind passes use for 95.
    static constexpr u32 BIND_GUARD  = 94;

private:
    struct Front {
        vec2 origin;
        f32  energy;
        f32  birth_time;
        f32  r_prev;
        f32  r_now;
        vec2 dir;
        f32  cone_cos;
        f32  push_mul;
        f32  heat_mul;
        u32  id;
    };
    std::vector<Front> fronts_;
    std::vector<vec4>  packed_;   // 3 vec4 per front, uploaded to BIND_FRONTS
    GPUBuffer fronts_buf_;
    GPUBuffer guard_buf_;         // uint last_hit_front_id per particle (persistent)
    u32 next_id_ = 1;
    u32 particle_capacity_ = 0;
    static constexpr u32 kMaxFronts = 64;

    // Vent-front detection state: prev-frame airtight fields + this frame's
    // deduct circles (xy=world center, z=world radius, w=1-k_vent scale).
    std::vector<f32>  prev_outside_, prev_pressure_;
    std::vector<vec4> vent_circles_;
    bool vent_prev_valid_ = false;

    ComputeShader source_shader_;
    ComputeShader wind_shader_;
    ComputeShader particle_shader_;

    void upload();
};

} // namespace ng
