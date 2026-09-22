// path_tracer/src/verify_bsdfs.cpp
#include "path_tracer/bsdf.h"
#include <iostream>
#include <random>
#include <cmath>

using namespace nrr;

static int fail_count = 0;
#define EXPECT(cond, msg) do { if (!(cond)) { std::cerr << "FAIL: " << msg << "\n"; ++fail_count; } } while(0)

static void test_lambert_energy_conservation() {
    Lambert m{Vec3(0.7)};
    Vec3 n(0,0,1), wo(0,0,1);
    std::mt19937 rng(42);
    Vec3 sum(0);
    int N = 10000;
    for (int i=0;i<N;++i) {
        BSDFSample s = bsdf::sample(m, n, wo, rng);
        sum += s.weight;
    }
    Vec3 mean = sum / double(N);
    EXPECT(std::abs(mean.x - 0.7) < 0.05, "Lambert mean weight ~ albedo");
}

static void test_ggx_returns_specular() {
    GGX m{Vec3(1.0), 0.1};
    Vec3 n(0,0,1), wo(0,0,1);
    std::mt19937 rng(42);
    BSDFSample s = bsdf::sample(m, n, wo, rng);
    EXPECT(s.pdf > 0, "GGX sample produces nonzero pdf");
    EXPECT(glm::length(s.weight) > 0, "GGX sample produces nonzero weight");
}

static void test_glass_normal_incidence_fresnel() {
    // At normal incidence, Schlick Fresnel for n1=1, n2=1.5 is:
    // F0 = ((1-1.5)/(1+1.5))^2 = 0.04
    // So at normal incidence, ~96% transmission, ~4% reflection.
    Glass m{Vec3(1.0), 1.5};
    Vec3 n(0,0,1), wo(0,0,1);  // straight on
    std::mt19937 rng(42);
    int reflect = 0, transmit = 0;
    int N = 10000;
    for (int i=0;i<N;++i) {
        BSDFSample s = bsdf::sample(m, n, wo, rng);
        if (glm::dot(s.wi, n) > 0) ++reflect;
        else ++transmit;
    }
    double frac_reflect = double(reflect) / N;
    EXPECT(std::abs(frac_reflect - 0.04) < 0.01, "Glass Fresnel ~ 4% reflect at normal incidence");
}

static void test_null_bsdf_passes_through() {
    NullBSDF m;
    Vec3 n(0,0,1), wo(0,0,1);   // ray going INTO surface (wo = -ray.dir)
    std::mt19937 rng(42);
    BSDFSample s = bsdf::sample(m, n, wo, rng);
    // The sampled direction should continue straight through (i.e., -wo, the original ray direction).
    EXPECT(glm::length(s.wi - (-wo)) < 1e-6, "NullBSDF wi == ray direction (pass-through)");
    EXPECT(s.pdf > 0, "NullBSDF nonzero pdf");
    EXPECT(glm::length(s.weight - Vec3(1)) < 1e-6, "NullBSDF weight = 1 (no attenuation)");
}


// ═══════════════════════════════════════════════════════════════════════════
// Anisotropic GGX verification.
//
// The aniso_brushed audit column was flagged untrusted: heavy fireflies at
// 48 spp, and nobody had proven whether that was a pdf/weight bug (bias) or
// legitimate variance from D-sampling a razor-thin lobe. These tests settle
// the bias question empirically:
//
//   T1 (unbiasedness): for a fixed wo, the mean sampling weight
//       E[f·cos/pdf] over sample() draws must equal the directional albedo
//       A(wo) = ∫ f(wo,wi) cos_i dwi computed by brute-force quadrature of
//       eval(). If sample()'s pdf or weight is wrong in ANY direction the two
//       diverge; if they match, the fireflies are variance, not bias.
//
//   T2 (isotropic reduction): AnisoGGX(a, a) must equal GGX(roughness=√a)
//       pointwise — the two implementations share no code, so agreement
//       cross-validates both D and the (algebraically equal) Smith-G forms.
//
//   T3 (bounded weights, VNDF): with visible-normal sampling the weight
//       simplifies to albedo·G1(wi) ≤ albedo. Asserted so a regression back
//       to an unbounded estimator fails loudly.
//
// The quadrature for T1 runs in half-vector SLOPE space, substituting
// u = x_slope/αx, v = y_slope/αy. In (u,v) the GGX lobe is a unit isotropic
// bump regardless of anisotropy, so one log-polar grid resolves αy = 0.02 as
// reliably as αy = 0.45 — a naive (θ,φ) grid would need θ-resolution finer
// than the lobe width and silently under-integrate razor-thin lobes, i.e. the
// verifier itself would have the bug it is hunting. Measure chain:
// dω_i = 4(wo·h) dω_h,  dω_h = cos³θ_h dx_s dy_s,  dx_s dy_s = αx αy du dv.
// ═══════════════════════════════════════════════════════════════════════════

static double aniso_albedo_bruteforce(const AnisoGGX& m, const Vec3& wo) {
    const int NR = 4096, NPSI = 512;
    const double lr0 = std::log(1e-4), lr1 = std::log(1e4);
    const double dlr = (lr1 - lr0) / NR, dpsi = 2.0 * PI / NPSI;
    const Vec3 n(0, 0, 1);
    double sum = 0.0;
    for (int i = 0; i < NR; ++i) {
        const double r = std::exp(lr0 + (i + 0.5) * dlr);
        const double cell = r * r * dlr * dpsi;   // r dr dψ with dr = r·dlnr
        for (int j = 0; j < NPSI; ++j) {
            const double psi = (j + 0.5) * dpsi;
            const double xs = m.alpha_x * r * std::cos(psi);
            const double ys = m.alpha_y * r * std::sin(psi);
            const double inv = 1.0 / std::sqrt(1.0 + xs * xs + ys * ys);
            const Vec3 h(xs * inv, ys * inv, inv);
            const double voh = glm::dot(wo, h);
            if (voh <= 0.0) continue;
            const Vec3 wi = glm::normalize(2.0 * voh * h - wo);
            if (wi.z <= 0.0) continue;
            const Vec3 f = bsdf::eval(m, n, wo, wi);
            sum += f.x * wi.z * 4.0 * voh
                   * (inv * inv * inv) * m.alpha_x * m.alpha_y * cell;
        }
    }
    return sum;
}

struct McStats { double mean, se, wmax; };

static McStats aniso_mc_weight(const AnisoGGX& m, const Vec3& wo, int N,
                               uint32_t seed) {
    const Vec3 n(0, 0, 1);
    std::mt19937 rng(seed);
    double sum = 0.0, sum2 = 0.0, wmax = 0.0;
    for (int i = 0; i < N; ++i) {
        BSDFSample s = bsdf::sample(m, n, wo, rng);
        // Rejected samples (pdf==0) count as weight 0: the estimator treats
        // below-horizon reflections as absorption, and dropping them from the
        // mean would bias it upward.
        const double w = s.weight.x;
        sum += w; sum2 += w * w;
        if (w > wmax) wmax = w;
    }
    const double mean = sum / N;
    const double var = std::max(0.0, sum2 / N - mean * mean);
    return {mean, std::sqrt(var / N), wmax};
}

static void test_aniso_unbiased_and_bounded() {
    struct Cfg { double ax, ay; };
    const Cfg cfgs[] = {{0.45, 0.02},    // the aniso_brushed scene values
                        {0.20, 0.20},    // isotropic sanity point
                        {0.05, 0.50}};   // anisotropy rotated 90°
    const double thetas[] = {0.0, 30.0, 60.0, 80.0};
    std::cerr << "  aniso T1/T3:  ax    ay   θo(°) φo(°)   brute-A     MC-mean"
                 "      stderr    max-w\n";
    for (const Cfg& c : cfgs) {
        AnisoGGX m{Vec3(1.0), c.ax, c.ay, Vec3(1, 0, 0)};
        for (double deg : thetas) {
            // One off-axis azimuth per config so anisotropy is exercised off
            // the tangent plane, not just in it.
            for (double pdeg : {0.0, 40.0}) {
                if (deg == 0.0 && pdeg != 0.0) continue;  // degenerate at pole
                const double th = deg * PI / 180.0, ph = pdeg * PI / 180.0;
                const Vec3 wo(std::sin(th) * std::cos(ph),
                              std::sin(th) * std::sin(ph), std::cos(th));
                const double brute = aniso_albedo_bruteforce(m, wo);
                const McStats mc = aniso_mc_weight(m, wo, 4'000'000,
                                                   0xC0FFEEu ^ (uint32_t)(deg * 7 + pdeg));
                char line[160];
                std::snprintf(line, sizeof line,
                    "               %.2f  %.2f  %4.0f  %4.0f   %.6f   %.6f   %.6f   %.3f\n",
                    c.ax, c.ay, deg, pdeg, brute, mc.mean, mc.se, mc.wmax);
                std::cerr << line;
                const double tol = std::max(4.0 * mc.se, 0.015 * brute);
                EXPECT(std::abs(mc.mean - brute) < tol,
                       "aniso E[weight] == bruteforce albedo (unbiased sampling)");
                // T3: VNDF weights are albedo·G1(wi) ≤ albedo. An unbounded
                // max here means someone reverted to D-sampling (fireflies).
                EXPECT(mc.wmax <= 1.0 + 1e-9,
                       "aniso VNDF weight bounded by albedo");
            }
        }
    }
}

static void test_aniso_isotropic_reduction() {
    // AnisoGGX(a, a) and GGX(roughness = √a) share no code; pointwise
    // agreement of eval() cross-validates both.
    const double alpha = 0.09;                       // GGX stores roughness, α = r²
    AnisoGGX ma{Vec3(1.0), alpha, alpha, Vec3(1, 0, 0)};
    GGX      mi{Vec3(1.0), std::sqrt(alpha)};
    const Vec3 n(0, 0, 1);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double max_rel = 0.0;
    for (int k = 0; k < 500; ++k) {
        const double t1 = std::acos(uni(rng)), p1 = 2 * PI * uni(rng);
        const double t2 = std::acos(uni(rng)), p2 = 2 * PI * uni(rng);
        const Vec3 wo(std::sin(t1) * std::cos(p1), std::sin(t1) * std::sin(p1), std::cos(t1));
        const Vec3 wi(std::sin(t2) * std::cos(p2), std::sin(t2) * std::sin(p2), std::cos(t2));
        const double fa = bsdf::eval(ma, n, wo, wi).x;
        const double fi = bsdf::eval(mi, n, wo, wi).x;
        if (fi > 1e-9)
            max_rel = std::max(max_rel, std::abs(fa - fi) / fi);
    }
    std::cerr << "  aniso T2: max |aniso(a,a)-iso|/iso over 500 dirs = "
              << max_rel << "\n";
    EXPECT(max_rel < 1e-6, "AnisoGGX(a,a) reduces to isotropic GGX");
}


static void test_pdf_matches_sampler() {
    // bsdf::pdf() must agree with the pdf the sampler itself reports.
    // The integrator's MIS weights are computed from the standalone function,
    // so any drift between the two silently re-biases every glossy highlight
    // -- there is no crash, just wrong pictures.
    const Vec3 n(0, 0, 1);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    Lambert  ml{Vec3(0.8)};
    GGX      mg{Vec3(1.0), 0.35};
    AnisoGGX ma{Vec3(1.0), 0.45, 0.02, Vec3(1, 0, 0)};
    Material mats[] = {Material{ml}, Material{mg}, Material{ma}};
    double max_rel = 0.0;
    int checked = 0;
    for (const Material& mat : mats) {
        for (int k = 0; k < 300; ++k) {
            const double ct = 0.02 + 0.98 * uni(rng);       // avoid exact pole
            const double st = std::sqrt(1.0 - ct * ct);
            const double ph = 2.0 * PI * uni(rng);
            const Vec3 wo(st * std::cos(ph), st * std::sin(ph), ct);
            BSDFSample smp = sample_material(mat, n, wo, rng);
            if (smp.pdf <= 0.0) continue;                   // rejected draw
            const double pd = pdf_material(mat, n, wo, smp.wi);
            max_rel = std::max(max_rel,
                               std::abs(pd - smp.pdf) / std::max(1e-12, smp.pdf));
            ++checked;
        }
    }
    std::cerr << "  pdf-lockstep: max |pdf_fn - sampler_pdf|/pdf over "
              << checked << " draws = " << max_rel << "\n";
    EXPECT(checked > 500, "pdf-lockstep exercised enough draws");
    EXPECT(max_rel < 1e-9, "bsdf::pdf matches sampler-reported pdf on-support");
}

static void test_ggx_vndf_unbiased_and_bounded() {
    // GGX shares the VNDF sampler with AnisoGGX; T2 proved their eval()s are
    // pointwise equal at ax == ay, so the aniso slope-space quadrature is a
    // valid target for the isotropic sampler too.
    const double alpha = 0.2;
    AnisoGGX ref{Vec3(1.0), alpha, alpha, Vec3(1, 0, 0)};
    GGX      g{Vec3(1.0), std::sqrt(alpha)};
    const double th = 60.0 * PI / 180.0;
    const Vec3 n(0, 0, 1), wo(std::sin(th), 0.0, std::cos(th));
    const double brute = aniso_albedo_bruteforce(ref, wo);
    std::mt19937 rng(99);
    double sum = 0.0, sum2 = 0.0, wmax = 0.0;
    const int N = 4'000'000;
    for (int i = 0; i < N; ++i) {
        BSDFSample s = bsdf::sample(g, n, wo, rng);
        sum += s.weight.x; sum2 += s.weight.x * s.weight.x;
        wmax = std::max(wmax, s.weight.x);
    }
    const double mean = sum / N;
    const double se = std::sqrt(std::max(0.0, sum2 / N - mean * mean) / N);
    std::cerr << "  ggx-vndf: brute=" << brute << "  mc=" << mean
              << "  se=" << se << "  max-w=" << wmax << "\n";
    EXPECT(std::abs(mean - brute) < std::max(4.0 * se, 0.015 * brute),
           "GGX VNDF E[weight] == bruteforce albedo");
    EXPECT(wmax <= 1.0 + 1e-9, "GGX VNDF weight bounded by albedo");
}

int main() {
    test_lambert_energy_conservation();
    test_ggx_returns_specular();
    test_glass_normal_incidence_fresnel();
    test_null_bsdf_passes_through();
    test_aniso_isotropic_reduction();
    test_pdf_matches_sampler();
    test_ggx_vndf_unbiased_and_bounded();
    test_aniso_unbiased_and_bounded();
    std::cerr << (fail_count==0 ? "OK\n" : "FAILED\n");
    return fail_count == 0 ? 0 : 1;
}
