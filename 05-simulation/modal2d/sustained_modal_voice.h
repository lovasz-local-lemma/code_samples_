#pragma once
// Continuous excitation of ONE mass-normalized modal body, with up to 13
// simultaneous contacts. Pure deterministic CPU DSP; no viewer/CUDA dependency.
//
// NoiseRub is a filtered stochastic force. FrictionBow is a compliant,
// regularized velocity-weakening tangential contact: force depends on bow speed
// MINUS the velocity of this same modal body at the contact. Static/kinetic
// friction give a smooth stick/slip approximation; this is not a full
// finite-width/thermal bow or fluid-finger model. Controls use normalized
// mechanical units, not calibrated newtons or metres/second.
// Blow is a pressure-powered, saturating acoustic feedback drive. A broad
// lip response favors the selected bore pole. It is a reduced negative-
// resistance generator, not a full lip-mass/Bernoulli model; blowing needs
// positive pressure and flow and stops supplying energy on release.
//
// Implicit midpoint integrates q'' + 2*zeta*w*q' + w*w*q = sum b_j F_j.
// Two bounded contact sweeps account for shared-body velocity feedback. The
// linear update is unconditionally stable; friction is force-bounded. Attack
// and release scale FORCE, leaving the accumulated modal state free to decay.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace modal {

enum class SustainedDrive { FrictionBow, NoiseRub, Blow };

struct SustainedContact {
    std::vector<float> weights; // signed, mass-normalized modal projection
    bool gate = false;
    SustainedDrive drive = SustainedDrive::FrictionBow;
    float pressure = 1.0f;
    float speed = 0.2f;         // contact speed; Blow: positive breath-flow multiplier (.2 nominal)
    float variation = 0.08f;   // slow, seeded pressure/speed variation
    float color = 0.35f;       // rubbed-noise bandwidth around the local resonant pole
    float attack_ms = 35.0f;
    float release_ms = 100.0f;
};

struct ContinuousModel {
    std::vector<float> frequencies_hz;
    std::vector<float> zetas;
    std::vector<float> radiation; // optional signed modal pickup; empty = equal
    bool displacement_pickup = false; // true: sum alpha*q, matching libmodal pressure convention
    bool dynamic_radiation_pickup = false; // alpha*v/omega: harmonic-equivalent amplitude, no static sound
    float pickup_gain = 1.0f;         // explicit listening-unit scale; never changes mechanics
};

class SustainedModalVoice {
public:
    static constexpr int max_contacts = 13;
    void configure(const std::vector<float>& frequencies_hz,
                   const std::vector<float>& zetas,
                   float sample_rate = 44100.0f,
                   const std::vector<float>& radiation = {}) {
        fs_ = clean(sample_rate, 44100.0f, 8000.0f, 192000.0f);
        displacement_pickup_ = false;
        dynamic_radiation_pickup_ = false;
        pickup_gain_ = 1.0;
        h_ = 1.0 / fs_;
        modes_.assign(frequencies_hz.size(), Mode{});
        force_.assign(modes_.size(), 0.0);
        midpoint_.assign(modes_.size(), 0.0);
        for (size_t i = 0; i < modes_.size(); ++i) {
            auto& m = modes_[i];
            const double f = frequencies_hz[i];
            // Keep genuine sub-audio swing modes in the mechanical state.
            // They are silent at the pickup; never shift them up to 20 Hz.
            // Zero/rigid and ultrasonic entries retain their array indices.
            if (!std::isfinite(f) || f < .001 || f >= 0.45 * fs_) continue;
            const double z = clean(i < zetas.size() ? zetas[i] : .008f,
                                   .008f, .00001f, .999f);
            m.w = 2.0 * fs_ * std::tan(3.141592653589793 * f / fs_);
            m.hz = f;
            m.admittance = 1.0 / (2.0 * z * (2.0 * 3.141592653589793 * f));
            // Prewarp the oscillation frequency; this factor preserves the
            // low-damping continuous-time decay rate near the Nyquist limit.
            const double warp = 1.0 + .25 * h_ * h_ * m.w * m.w;
            const double decay = z * (2.0 * 3.141592653589793 * f) * warp;
            m.inv_d = 1.0 / (1.0 + h_ * decay + .25 * h_ * h_ * m.w * m.w);
            m.pickup = clean(i < radiation.size() ? radiation[i] : 1.0f,
                             0.0f, -100.0f, 100.0f);
            if (f < 20.0) m.pickup = 0.0;
        }
        for (auto& c : contacts_) {
            c = ContactState{};
            c.weights.assign(modes_.size(), 0.0);
            c.target.weights.assign(modes_.size(), 0.0f);
        }
        control_a_ = 1.0 - std::exp(-h_ / .012);
        slow_a_ = 1.0 - std::exp(-h_ / .09);
        set_seed(seed_);
        normalized_friction(.1); // prewarm lookup before entering the audio callback
        reset();
    }
    void configure(const ContinuousModel& model, float sample_rate = 44100.0f) {
        configure(model.frequencies_hz, model.zetas, sample_rate, model.radiation);
        displacement_pickup_ = model.displacement_pickup;
        dynamic_radiation_pickup_ = model.dynamic_radiation_pickup;
        pickup_gain_ = clean(model.pickup_gain, 1.0f, 0.0f, 100000.0f);
    }
    int num_modes() const { return static_cast<int>(modes_.size()); }
    static double normalized_friction(double relative_speed) {
        // Uniform fine table of the same analytic Stribeck/tanh law, not a
        // different force model. 16KB; avoids millions of exp/tanh calls in
        // 13 simultaneous contacts. Outside the table the exponential tail
        // differs from the kinetic limit by less than 1e-11.
        static const std::array<float, 4097> table = [] {
            std::array<float, 4097> values{};
            for (size_t i = 0; i < values.size(); ++i) {
                const double v = double(i) * .00025;
                values[i] = static_cast<float>((.76 + .24 * std::exp(-v*v/.04)) * std::tanh(v/.012));
            }
            return values;
        }();
        const double x = std::abs(relative_speed) * 4000.0;
        if (x >= 4096.0) return std::copysign(.76, relative_speed);
        const size_t index = static_cast<size_t>(x);
        const double t = x - double(index);
        return std::copysign(table[index] + t * (table[index+1] - table[index]), relative_speed);
    }
    void set_seed(uint32_t seed) {
        seed_ = seed ? seed : 1u;
        for (int j = 0; j < max_contacts; ++j)
            contacts_[j].rng = seed_ ^ (0x9e3779b9u * (j + 1u));
    }
    void reset() {
        for (auto& m : modes_) m.x = m.v = 0.0;
        for (auto& c : contacts_) {
            c.target.gate = false;
            c.envelope = c.pressure = c.speed = c.variation = 0.0;
            c.noise = c.slow_noise = c.force = 0.0;
            c.band_s1 = c.band_s2 = 0.0;
            std::fill(c.weights.begin(), c.weights.end(), 0.0);
        }
        energy_ = 0.0;
    }
    void set_contact(int slot, const SustainedContact& target) {
        if (slot < 0 || slot >= max_contacts) return;
        auto& c = contacts_[slot];
        const bool was_silent = c.envelope < 1e-9 && !c.target.gate;
        c.target = target;
        c.target.pressure = clean(target.pressure, 0.f, 0.f, 8.f);
        c.target.speed = clean(target.speed, 0.f, -4.f, 4.f);
        c.target.variation = clean(target.variation, 0.f, 0.f, 1.f);
        c.target.color = clean(target.color, .35f, .005f, .95f);
        c.target.attack_ms = clean(target.attack_ms, 35.f, 2.f, 4000.f);
        c.target.release_ms = clean(target.release_ms, 100.f, 2.f, 8000.f);
        c.target.weights.resize(modes_.size(), 0.0f);
        for (auto& w : c.target.weights) w = clean(w, 0.f, -4096.f, 4096.f);
        c.attack_a = 1.0 - std::exp(-h_ / (.001 * c.target.attack_ms));
        c.release_a = 1.0 - std::exp(-h_ / (.001 * c.target.release_ms));
        c.noise_a = 1.0 - std::exp(-3.141592653589793 * c.target.color);
        double admittance = 0.0, best_displacement = 0.0, center = 220.0;
        for (size_t i = 0; i < modes_.size(); ++i) {
            const auto& m = modes_[i];
            if (m.hz < 20 || m.w <= 0) continue;
            const double b = c.target.weights[i];
            admittance = std::max(admittance, b*b*m.admittance);
            const double displacement = std::abs(b) / m.w;
            if (displacement > best_displacement) { best_displacement = displacement; center = m.hz; }
        }
        // Pressure is normalized relative to the contacted body's mechanical
        // impedance. This does not renormalize spatial weights or change its
        // eigenmodes. A force calibrated for b=1 can either fail to overcome
        // damping or over-lock a strongly coupled FEM contact. The nominal
        // level includes onset headroom for this regularized friction law;
        // pressure remains a relative control, not a calibrated normal force.
        c.target_force_scale = admittance > 1e-12 ? std::clamp(
            (target.drive==SustainedDrive::Blow?.65:18.0) / admittance, 1e-12, 1e8) : 0.0;
        c.target_band_g = std::tan(3.141592653589793 * center / fs_);
        // Tonal rubbing: a narrow stochastic band, with brightness widening
        // it. A broad band lets a strongly radiating shared support pole bury
        // the selected element's quiet fundamental (notably glass modes).
        c.target_band_k = target.drive==SustainedDrive::Blow ? .55+.5*c.target.color : .008 + .055*c.target.color;
        // A silent contact may adopt its position immediately. Subsequent
        // motion is interpolated in signed modal space rather than snapping.
        if (was_silent) {
            for (size_t i = 0; i < modes_.size(); ++i) c.weights[i] = c.target.weights[i];
            c.force_scale = c.target_force_scale;
            c.band_g = c.target_band_g;
            c.band_k = c.target_band_k;
        }
    }
    void release_contact(int slot) {
        if (slot >= 0 && slot < max_contacts) contacts_[slot].target.gate = false;
    }
    void release_all() { for (auto& c : contacts_) c.target.gate = false; }
    // A mechanical impulse changes velocity without erasing displacement or
    // another note's state. Magnitude is normalized impulse, not legacy sine A.
    void add_impulse(const std::vector<float>& weights, float magnitude) {
        const double a = clean(magnitude, 0.f, -32.f, 32.f);
        for (size_t i = 0; i < modes_.size() && i < weights.size(); ++i)
            if (modes_[i].w > 0.0)
                modes_[i].v += clean(weights[i], 0.f, -4096.f, 4096.f) * a;
        measure_energy();
    }
    // Release a static force: q_i += b_i*F/omega_i^2, v_i is unchanged.
    // This gives a genuine pluck's high-frequency rolloff instead of a softer
    // impact. The idealized loading is instantaneous; no fingertip model.
    void add_pluck(const std::vector<float>& weights, float static_force) {
        const double force = clean(static_force, 0.f, -100000.f, 100000.f);
        for (size_t i = 0; i < modes_.size() && i < weights.size(); ++i)
            if (modes_[i].w > 0.0)
                modes_[i].x += clean(weights[i], 0.f, -4096.f, 4096.f) * force / modes_[i].w;
        measure_energy();
    }
    int copy_modal_positions(float* out, int capacity) const {
        const int count = std::min(std::max(0, capacity), num_modes());
        if (!out) return 0;
        for (int i = 0; i < count; ++i)
            out[i] = modes_[i].w > 0.0 ? static_cast<float>(modes_[i].x / modes_[i].w) : 0.f;
        return count;
    }
    int copy_modal_velocities(float* out, int capacity) const {
        const int count = std::min(std::max(0, capacity), num_modes());
        if (!out) return 0;
        for (int i = 0; i < count; ++i) out[i] = static_cast<float>(modes_[i].v);
        return count;
    }
    double state_energy() const { return energy_; }
    bool has_energy() const {
        if (energy_ > 1e-18) return true;
        for (const auto& c : contacts_)
            if (c.target.gate || c.envelope > 1e-8) return true;
        return false;
    }
    float render_add(float* out, int count, float output_gain = 1.0f) {
        if (!out || count <= 0 || modes_.empty()) return 0.f;
        const double gain = clean(output_gain, 1.f, -16.f, 16.f);
        double peak = 0.0;
        for (int s = 0; s < count; ++s) {
            std::fill(force_.begin(), force_.end(), 0.0);
            std::array<int, max_contacts> bows{};
            int n_bows = 0;
            for (int j = 0; j < max_contacts; ++j) {
                auto& c = contacts_[j];
                const double target_env = c.target.gate ? 1.0 : 0.0;
                c.envelope += (target_env - c.envelope) * (c.target.gate ? c.attack_a : c.release_a);
                if (!c.target.gate && c.envelope < 1e-10) { c.envelope = 0; continue; }
                c.pressure += control_a_ * (c.target.pressure - c.pressure);
                c.speed += control_a_ * (c.target.speed - c.speed);
                c.variation += control_a_ * (c.target.variation - c.variation);
                c.force_scale += control_a_ * (c.target_force_scale - c.force_scale);
                c.band_g += control_a_ * (c.target_band_g - c.band_g);
                c.band_k += control_a_ * (c.target_band_k - c.band_k);
                c.rng = 1664525u * c.rng + 1013904223u;
                const double white = (c.rng >> 8) * (1.0 / 8388608.0) - 1.0;
                c.noise += c.noise_a * (white - c.noise);
                c.slow_noise += slow_a_ * (white - c.slow_noise);
                const double drift = std::clamp(12.0 * c.slow_noise, -1.0, 1.0) * c.variation;
                c.effective_pressure = c.envelope * c.pressure * (1.0 + .35 * drift);
                c.effective_speed = c.speed * (1.0 + .3 * drift);
                for (size_t i = 0; i < modes_.size(); ++i)
                    c.weights[i] += control_a_ * (c.target.weights[i] - c.weights[i]);
                c.force = 0.0;
                if (c.target.drive == SustainedDrive::Blow) {
                    // Reduced lip-valve negative acoustic resistance. A
                    // broad lip response centered on the contacted bore's
                    // first pole feeds pressure-powered energy back into
                    // its velocity, with asymmetric saturation representing
                    // opening/closing. The bore, not a prescribed sinusoid,
                    // determines the resulting pitch. The seed is turbulent
                    // breath, needed to leave the exactly-zero equilibrium.
                    // This is a bounded acoustic-generator approximation,
                    // not a full Bernoulli/lip-mass or nonlinear-bore solve.
                    // Reference: UNSW brass acoustics and Fletcher (1990),
                    // https://www.phys.unsw.edu.au/jw/brassacoustics.html
                    double velocity=0;
                    for(size_t i=0;i<modes_.size();++i)velocity+=c.weights[i]*modes_[i].v;
                    const double a1=1.0/(1.0+c.band_g*(c.band_g+c.band_k));
                    const double a2=c.band_g*a1,a3=c.band_g*a2;
                    const double v3=velocity-c.band_s2;
                    const double band=a1*c.band_s1+a2*v3,low=c.band_s2+a2*c.band_s1+a3*v3;
                    c.band_s1=2*band-c.band_s1;c.band_s2=2*low-c.band_s2;
                    const double lip=std::tanh(c.band_k*band/.12);
                    const double flow=std::sqrt(std::clamp(c.effective_speed/.2,0.0,3.0));
                    const double opening=lip+.18*lip*lip;
                    c.force=c.force_scale*c.effective_pressure*flow*(opening+.004*c.noise);
                    for(size_t i=0;i<modes_.size();++i)force_[i]+=c.weights[i]*c.force;
                } else if (c.target.drive == SustainedDrive::NoiseRub) {
                    // Speed controls roughness-drive amplitude but its sign
                    // does not invert random noise. A stopped rub is silent.
                    const double speed_gain = std::sqrt(std::min(2.0, std::abs(c.effective_speed) / .2));
                    // A real band-limited random force excites the local
                    // resonant pole; this is not an added sine-wave note.
                    const double a1 = 1.0 / (1.0 + c.band_g * (c.band_g+c.band_k));
                    const double a2 = c.band_g*a1, a3 = c.band_g*a2;
                    const double v3 = c.noise-c.band_s2;
                    const double band = a1*c.band_s1+a2*v3;
                    const double low = c.band_s2+a2*c.band_s1+a3*v3;
                    c.band_s1=2*band-c.band_s1; c.band_s2=2*low-c.band_s2;
                    c.force = 90.0 * c.effective_pressure * speed_gain * c.band_k * band;
                    for (size_t i = 0; i < modes_.size(); ++i) force_[i] += c.weights[i] * c.force;
                } else bows[n_bows++] = j;
            }
            for (size_t i = 0; i < modes_.size(); ++i) {
                const auto& m = modes_[i];
                midpoint_[i] = (m.v - .5 * h_ * m.w * m.x + .5 * h_ * force_[i]) * m.inv_d;
            }
            // Projected Gauss-Seidel: each update changes the shared modal
            // velocity seen by every other contact. Fixed iteration count
            // bounds audio work. Finite contact compliance is intentional.
            for (int sweep = 0; sweep < 2; ++sweep) {
                for (int k = 0; k < n_bows; ++k) {
                    auto& c = contacts_[bows[k]];
                    double v = 0.0, compliance = 0.0;
                    for (size_t i = 0; i < modes_.size(); ++i) {
                        v += c.weights[i] * midpoint_[i];
                        compliance += c.weights[i] * c.weights[i] * modes_[i].inv_d;
                    }
                    compliance *= .5 * h_;
                    const double limit = c.force_scale * c.effective_pressure;
                    const double free_v = v - compliance * c.force;
                    double lo = -limit, hi = limit;
                    // Stribeck-like static/kinetic transition. Relative
                    // velocity is solved implicitly, not sampled once from
                    // the previous block. The force bracket always contains
                    // a root because |friction| <= limit. A non-monotone law
                    // can have multiple roots; this finite regularization is
                    // an approximation, not an exact Coulomb contact solve.
                    for (int iteration = 0; iteration < 16; ++iteration) {
                        const double f = .5 * (lo + hi);
                        const double relative = c.effective_speed - free_v - compliance * f;
                        const double friction = limit * normalized_friction(relative);
                        if (std::abs(f - friction) <= 1e-12 * (1.0 + limit)) { lo = hi = f; break; }
                        if (f > friction) hi = f; else lo = f;
                    }
                    const double f = .5 * (lo + hi);
                    const double delta = f - c.force;
                    c.force = f;
                    for (size_t i = 0; i < modes_.size(); ++i)
                        midpoint_[i] += .5 * h_ * modes_[i].inv_d * c.weights[i] * delta;
                }
            }
            // The finite multi-contact iteration need not exactly solve all
            // simultaneous friction equations. Enforce their aggregate work
            // bound: body work <= work supplied by the moving bows. In
            // particular stationary contacts cannot generate modal energy.
            // A scalar reduction is sufficient because contact mobility is
            // positive semidefinite. Normally the converged law already
            // satisfies this and the factor is exactly one.
            if (n_bows > 0) {
                std::fill(force_.begin(), force_.end(), 0.0);
                double supplied = 0.0;
                for (int k = 0; k < n_bows; ++k) {
                    const auto& c = contacts_[bows[k]];
                    supplied += c.force * c.effective_speed;
                    for (size_t i = 0; i < modes_.size(); ++i) force_[i] += c.weights[i] * c.force;
                }
                double linear_work = supplied, quadratic_work = 0.0;
                for (size_t i = 0; i < modes_.size(); ++i) {
                    const double dv = .5 * h_ * modes_[i].inv_d * force_[i];
                    linear_work -= force_[i] * (midpoint_[i] - dv);
                    quadratic_work += force_[i] * dv;
                }
                if (linear_work < quadratic_work && quadratic_work > 1e-30) {
                    const double scale = std::clamp(linear_work / quadratic_work, 0.0, 1.0);
                    for (size_t i = 0; i < modes_.size(); ++i)
                        midpoint_[i] -= (1.0 - scale) * .5 * h_ * modes_[i].inv_d * force_[i];
                }
            }
            double value = 0.0;
            for (size_t i = 0; i < modes_.size(); ++i) {
                auto& m = modes_[i];
                m.x += h_ * m.w * midpoint_[i];
                m.v = 2.0 * midpoint_[i] - m.v;
                // Guard untrusted inputs/accumulation, never an output clip.
                if (!std::isfinite(m.x) || !std::isfinite(m.v)) m.x = m.v = 0.0;
                const double observable = dynamic_radiation_pickup_ ? (m.w > 0 ? m.v/m.w : 0.0) :
                    (displacement_pickup_ ? (m.w > 0 ? m.x/m.w : 0.0) : m.v);
                value += m.pickup * observable;
            }
            const double added = gain * pickup_gain_ * value;
            out[s] += static_cast<float>(added);
            peak = std::max(peak, std::abs(added));
        }
        measure_energy();
        return static_cast<float>(peak);
    }

private:
    static float clean(float x, float fallback, float lo, float hi) {
        return std::isfinite(x) ? std::clamp(x, lo, hi) : fallback;
    }
    struct Mode { double w = 0, hz = 0, admittance = 0, inv_d = 0, pickup = 0, x = 0, v = 0; };
    struct ContactState {
        SustainedContact target;
        std::vector<double> weights;
        double envelope = 0, pressure = 0, speed = 0, variation = 0;
        double attack_a = .001, release_a = .001, noise_a = .6;
        double noise = 0, slow_noise = 0, force = 0;
        double effective_pressure = 0, effective_speed = 0;
        double force_scale = 90, target_force_scale = 90;
        double band_g = .03, target_band_g = .03, band_k = .7, target_band_k = .7;
        double band_s1 = 0, band_s2 = 0;
        uint32_t rng = 1;
    };
    void measure_energy() {
        energy_ = 0;
        for (const auto& m : modes_) energy_ += m.x * m.x + m.v * m.v;
    }
    std::vector<Mode> modes_;
    std::vector<double> force_, midpoint_;
    std::array<ContactState, max_contacts> contacts_;
    double fs_ = 44100, h_ = 1.0 / 44100, control_a_ = .001, slow_a_ = .0001;
    double energy_ = 0;
    uint32_t seed_ = 0x53ab912u;
    bool displacement_pickup_ = false;
    bool dynamic_radiation_pickup_ = false;
    double pickup_gain_ = 1;
};

} // namespace modal
