// path_tracer/src/medium.cpp
#include "path_tracer/medium.h"
#include "path_tracer/noise.h"
#include <algorithm>
#include <cmath>

namespace nrr {

static double luminance(const Vec3& v) {
    return 0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z;
}

// Draw an exponential free-flight distance for a scalar extinction.
static double exp_dist(double sigma, std::mt19937& rng) {
    std::uniform_real_distribution<double> d(0.0, 1.0);
    double u = std::max(1e-12, 1.0 - d(rng));   // avoid log(0)
    return -std::log(u) / std::max(1e-9, sigma);
}

static Vec3 exp_vec(const Vec3& sigma_t, double t) {
    return Vec3(std::exp(-sigma_t.x * t),
                std::exp(-sigma_t.y * t),
                std::exp(-sigma_t.z * t));
}

// ---------------------------------------------------------------------------
// Homogeneous
// ---------------------------------------------------------------------------
MediumSample sample_interaction(const HomogeneousMedium& m, const Ray& /*r*/,
                                double t_max, std::mt19937& rng) {
    MediumSample s;
    Vec3 sigma_t = m.sigma_a + m.sigma_s;
    double st_lum = luminance(sigma_t);

    if (st_lum <= 1e-9) {          // vacuum
        s.scattered = false; s.t = t_max; s.weight = Vec3(1.0);
        return s;
    }

    double t = exp_dist(st_lum, rng);

    // Distance is sampled from the luminance pdf but the medium may be
    // chromatic, so each event's weight is the true spectral quantity divided
    // by the scalar pdf actually used. For a grey medium these collapse to
    // `albedo` and `1`, which is the textbook result.
    if (t < t_max) {
        s.scattered = true;
        s.t = t;
        double pdf = st_lum * std::exp(-st_lum * t);
        s.weight = (m.sigma_s * exp_vec(sigma_t, t)) / std::max(1e-12, pdf);
    } else {
        s.scattered = false;
        s.t = t_max;
        double pdf = std::exp(-st_lum * t_max);   // P(escape)
        s.weight = exp_vec(sigma_t, t_max) / std::max(1e-12, pdf);
    }
    return s;
}

Vec3 transmittance(const HomogeneousMedium& m, const Ray& /*r*/, double dist,
                   std::mt19937& /*rng*/) {
    return exp_vec(m.sigma_a + m.sigma_s, dist);   // closed form
}

// ---------------------------------------------------------------------------
// Heterogeneous (Perlin fBm)
// ---------------------------------------------------------------------------
double density_at(const PerlinMedium& m, const Vec3& p) {
    double n = noise::fbm01(p * m.freq + m.offset,
                            m.octaves, m.lacunarity, m.gain);
    double thr = std::min(0.999, std::max(0.0, m.threshold));
    double d = (n - thr) / (1.0 - thr);
    d = std::min(1.0, std::max(0.0, d));
    return d * m.density_mult;
}

// Upper bound on scalar extinction anywhere in the medium. density_at is
// clamped to [0, density_mult] by construction, so this bound is exact rather
// than estimated -- important, because delta tracking is only unbiased when the
// majorant genuinely dominates the field.
double majorant_sigma_t(const PerlinMedium& m) {
    return luminance(m.sigma_a + m.sigma_s) * std::max(0.0, m.density_mult);
}

MediumSample sample_interaction(const PerlinMedium& m, const Ray& r,
                                double t_max, std::mt19937& rng) {
    MediumSample s;
    s.weight = Vec3(1.0);

    const double maj = majorant_sigma_t(m);
    if (maj <= 1e-9) {
        s.scattered = false; s.t = t_max;
        return s;
    }

    // Delta (Woodcock) tracking: march in exponential steps drawn against the
    // majorant, then decide at each candidate whether it was a real collision
    // or a fictitious ("null") one. Attenuation is carried by the *rate* of
    // real collisions, so no explicit transmittance factor appears here.
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double t = 0.0;
    for (int guard = 0; guard < 10000; ++guard) {
        t += exp_dist(maj, rng);
        if (t >= t_max) {
            s.scattered = false;
            s.t = t_max;
            return s;                       // weight stays 1
        }
        Vec3 p = r.at(t);
        double d = density_at(m, p);
        Vec3 sigma_t_p = (m.sigma_a + m.sigma_s) * d;
        double st_lum = luminance(sigma_t_p);

        if (uni(rng) < st_lum / maj) {      // real collision
            s.scattered = true;
            s.t = t;
            // Single-scattering albedo at this point.
            Vec3 ss = m.sigma_s * d;
            s.weight = Vec3(
                st_lum > 1e-12 ? ss.x / std::max(1e-12, sigma_t_p.x) : 0.0,
                st_lum > 1e-12 ? ss.y / std::max(1e-12, sigma_t_p.y) : 0.0,
                st_lum > 1e-12 ? ss.z / std::max(1e-12, sigma_t_p.z) : 0.0);
            return s;
        }
        // else: null collision, keep marching.
    }
    // Guard tripped (pathologically dense field); treat as escape rather than
    // spinning forever.
    s.scattered = false;
    s.t = t_max;
    return s;
}

Vec3 transmittance(const PerlinMedium& m, const Ray& r, double dist,
                   std::mt19937& rng) {
    const double maj = majorant_sigma_t(m);
    if (maj <= 1e-9) return Vec3(1.0);

    // Ratio tracking: an unbiased transmittance estimator that accumulates
    // (1 - sigma_t/majorant) at each candidate collision instead of terminating.
    // Lower variance than binary delta tracking and, unlike the closed form,
    // valid for a spatially varying field.
    Vec3 tr(1.0);
    double t = 0.0;
    for (int guard = 0; guard < 10000; ++guard) {
        t += exp_dist(maj, rng);
        if (t >= dist) break;
        double d = density_at(m, r.at(t));
        double st_lum = luminance((m.sigma_a + m.sigma_s) * d);
        tr *= std::max(0.0, 1.0 - st_lum / maj);
        if (tr.x < 1e-4 && tr.y < 1e-4 && tr.z < 1e-4) break;   // fully opaque
    }
    return tr;
}

// ---------------------------------------------------------------------------
// Phase function (shared by both media types)
// ---------------------------------------------------------------------------
static Vec3 sample_hg(double g, const Vec3& wo, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u1 = dist(rng), u2 = dist(rng);
    double cos_theta;
    if (std::abs(g) < 1e-3) {
        cos_theta = 1.0 - 2.0 * u1;                 // isotropic
    } else {
        double sqr = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
        cos_theta = (1.0 + g * g - sqr * sqr) / (2.0 * g);
    }
    double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    double phi = 2.0 * PI * u2;

    Vec3 w = -wo;                                   // forward = incident dir
    Vec3 a = (std::abs(w.x) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    Vec3 t = glm::normalize(glm::cross(a, w));
    Vec3 b = glm::cross(w, t);
    return glm::normalize(t * std::cos(phi) * sin_theta +
                          b * std::sin(phi) * sin_theta +
                          w * cos_theta);
}

Vec3 sample_phase(const HomogeneousMedium& m, const Vec3& wo, std::mt19937& rng) {
    return sample_hg(m.g, wo, rng);
}
Vec3 sample_phase(const PerlinMedium& m, const Vec3& wo, std::mt19937& rng) {
    return sample_hg(m.g, wo, rng);
}

} // namespace nrr
