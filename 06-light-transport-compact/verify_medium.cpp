// path_tracer/src/verify_medium.cpp
//
// Verifies the medium sampling against closed forms and, for the
// heterogeneous path, against direct numerical integration of the density
// field. NOTE: this file must track the medium.h interface -- an earlier
// version kept calling the pre-Phase-C API, no longer compiled, and the
// STALE BINARY kept printing OK on every run. If you change medium.h,
// build this target and watch it fail before trusting anything.
#include "path_tracer/medium.h"
#include <iostream>
#include <random>
#include <cmath>

using namespace nrr;

static int fail_count = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++fail_count; } } while(0)

// ── Homogeneous ────────────────────────────────────────────────────────────

static void test_homogeneous_distance_mean() {
    // With unbounded flight, every draw scatters and E[t] = 1/sigma_t.
    HomogeneousMedium m{Vec3(0.5), Vec3(1.5), 0.0};   // sigma_t = 2
    std::mt19937 rng(42);
    Ray r{Vec3(0), Vec3(0, 0, 1)};
    const int N = 100000;
    double sum = 0;
    int scattered = 0;
    for (int i = 0; i < N; ++i) {
        MediumSample s = sample_interaction(m, r, 1e30, rng);
        if (s.scattered) { sum += s.t; ++scattered; }
    }
    EXPECT(scattered == N, "homogeneous: every unbounded flight scatters");
    EXPECT(std::abs(sum / scattered - 0.5) < 0.01,
           "homogeneous distance mean ~ 1/sigma_t");
}

static void test_homogeneous_weights() {
    // Estimator identities for a grey medium: a scatter event's weight is the
    // single-scattering albedo sigma_s/sigma_t, an escape's weight is 1 --
    // the transmittance cancels against the sampling pdf in both cases.
    HomogeneousMedium m{Vec3(0.5), Vec3(1.5), 0.0};
    std::mt19937 rng(7);
    Ray r{Vec3(0), Vec3(0, 0, 1)};
    const double t_max = 0.6;
    int esc = 0, sca = 0;
    for (int i = 0; i < 200000; ++i) {
        MediumSample s = sample_interaction(m, r, t_max, rng);
        if (s.scattered) {
            ++sca;
            EXPECT(std::abs(s.weight.x - 0.75) < 1e-9,
                   "homogeneous scatter weight == sigma_s/sigma_t");
        } else {
            ++esc;
            EXPECT(std::abs(s.weight.x - 1.0) < 1e-9,
                   "homogeneous escape weight == 1");
        }
    }
    // Escape frequency must reproduce the transmittance exp(-2 * 0.6).
    const double p_esc = double(esc) / (esc + sca);
    EXPECT(std::abs(p_esc - std::exp(-1.2)) < 0.005,
           "homogeneous escape probability == exp(-sigma_t t_max)");
}

static void test_homogeneous_transmittance() {
    HomogeneousMedium m{Vec3(0.5), Vec3(1.5), 0.0};
    std::mt19937 rng(3);
    Ray r{Vec3(0), Vec3(0, 0, 1)};
    Vec3 t = transmittance(m, r, 1.0, rng);
    EXPECT(std::abs(t.x - std::exp(-2.0)) < 1e-9,
           "homogeneous transmittance = exp(-sigma_t t) (closed form)");
}

static void test_isotropic_phase_uniform() {
    HomogeneousMedium m{Vec3(0.5), Vec3(1.5), 0.0};   // g = 0
    std::mt19937 rng(42);
    Vec3 wo(0, 0, 1), sum(0);
    const int N = 100000;
    for (int i = 0; i < N; ++i) sum += sample_phase(m, wo, rng);
    EXPECT(glm::length(sum / double(N)) < 0.05,
           "isotropic phase mean ~ 0");
}

// ── Heterogeneous (Perlin, delta/ratio tracking) ───────────────────────────

// Reference optical depth by fine Riemann sum over the exposed density field.
static double optical_depth_ref(const PerlinMedium& m, const Ray& r, double dist) {
    const int N = 20000;
    const double dt = dist / N;
    double tau = 0.0;
    const double st1 = 0.2126 * (m.sigma_a.x + m.sigma_s.x)
                     + 0.7152 * (m.sigma_a.y + m.sigma_s.y)
                     + 0.0722 * (m.sigma_a.z + m.sigma_s.z);
    for (int i = 0; i < N; ++i)
        tau += st1 * density_at(m, r.at((i + 0.5) * dt)) * dt;
    return tau;
}

static void test_perlin_tracking_unbiased() {
    // Ratio tracking's E[transmittance] and delta tracking's escape rate must
    // BOTH reproduce exp(-integral of sigma_t), computed independently by
    // integrating density_at along the ray. This is the test that catches a
    // wrong majorant, a wrong acceptance ratio, or clamping bugs -- none of
    // which crash; they all just darken or brighten smoke silently.
    PerlinMedium m;
    m.sigma_a = Vec3(0.1); m.sigma_s = Vec3(2.0);
    m.freq = 3.0; m.octaves = 3; m.threshold = 0.35; m.density_mult = 1.0;

    std::mt19937 rng(2024);
    const Ray rays[] = {
        {Vec3(-0.4, 0.1, -0.2), glm::normalize(Vec3(1.0, 0.2, 0.4))},
        {Vec3(0.3, -0.5, 0.6),  glm::normalize(Vec3(-0.5, 1.0, -0.3))},
        {Vec3(0.0, 0.0, -1.0),  Vec3(0.0, 0.0, 1.0)},
    };
    for (const Ray& r : rays) {
        const double dist = 1.5;
        const double ref = std::exp(-optical_depth_ref(m, r, dist));

        const int N = 60000;
        double tr_sum = 0.0;
        int esc = 0;
        for (int i = 0; i < N; ++i) {
            tr_sum += transmittance(m, r, dist, rng).x;
            MediumSample s = sample_interaction(m, r, dist, rng);
            if (!s.scattered) ++esc;
        }
        const double tr_mc = tr_sum / N;
        const double esc_mc = double(esc) / N;
        // Monte Carlo tolerance: se of a [0,1] variable at N=60k is < 0.0021.
        EXPECT(std::abs(tr_mc - ref) < 0.01,
               "perlin ratio-tracking transmittance matches density integral");
        EXPECT(std::abs(esc_mc - ref) < 0.01,
               "perlin delta-tracking escape rate matches density integral");
    }
}

static void test_perlin_majorant_dominates() {
    // Delta tracking is only unbiased if the majorant truly bounds the field.
    PerlinMedium m;
    m.sigma_a = Vec3(0.05); m.sigma_s = Vec3(6.0);
    m.freq = 4.0; m.octaves = 4; m.threshold = 0.42; m.density_mult = 1.0;
    const double maj = majorant_sigma_t(m);
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> uni(-2.0, 2.0);
    double worst = 0.0;
    for (int i = 0; i < 200000; ++i) {
        Vec3 p(uni(rng), uni(rng), uni(rng));
        const double d = density_at(m, p);
        const double st1 = 0.2126 * (m.sigma_a.x + m.sigma_s.x)
                         + 0.7152 * (m.sigma_a.y + m.sigma_s.y)
                         + 0.0722 * (m.sigma_a.z + m.sigma_s.z);
        worst = std::max(worst, st1 * d);
    }
    EXPECT(worst <= maj + 1e-9, "majorant dominates sigma_t everywhere sampled");
}

int main() {
    test_homogeneous_distance_mean();
    test_homogeneous_weights();
    test_homogeneous_transmittance();
    test_isotropic_phase_uniform();
    test_perlin_tracking_unbiased();
    test_perlin_majorant_dominates();
    std::cerr << (fail_count == 0 ? "OK\n" : "FAILED\n");
    return fail_count == 0 ? 0 : 1;
}
