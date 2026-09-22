#include "path_tracer/bsdf.h"
#include <cmath>

namespace nrr::bsdf {

static Vec3 sample_cosine_hemisphere(const Vec3& normal, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double u1 = dist(rng), u2 = dist(rng);
    double phi = 2.0 * PI * u1;
    double cos_theta = std::sqrt(u2);
    double sin_theta = std::sqrt(1.0 - u2);
    Vec3 w = normal;
    Vec3 a = (std::abs(w.x) > 0.9) ? Vec3(0,1,0) : Vec3(1,0,0);
    Vec3 u = glm::normalize(glm::cross(a, w));
    Vec3 v = glm::cross(w, u);
    return glm::normalize(u * std::cos(phi) * sin_theta +
                          v * std::sin(phi) * sin_theta +
                          w * cos_theta);
}

static Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0 * glm::dot(v, n) * n;
}

BSDFSample sample(const Lambert& m, const Vec3& n, const Vec3& /*wo*/, std::mt19937& rng) {
    BSDFSample s;
    s.wi = sample_cosine_hemisphere(n, rng);
    s.pdf = glm::dot(s.wi, n) * INV_PI;
    s.weight = m.albedo;   // (albedo/pi)*cos / (cos/pi) = albedo
    s.is_specular = false;
    return s;
}

BSDFSample sample(const Mirror& m, const Vec3& n, const Vec3& wo, std::mt19937& /*rng*/) {
    BSDFSample s;
    s.wi = reflect(-wo, n);
    s.pdf = 1.0;
    s.weight = m.albedo;
    s.is_specular = true;
    return s;
}

Vec3 eval(const Lambert& m, const Vec3& /*n*/, const Vec3& /*wo*/, const Vec3& /*wi*/) {
    return m.albedo * INV_PI;
}

// GGX/Trowbridge-Reitz microfacet, isotropic, with cosine-of-half-vector sampling.
// Reference: Walter et al. 2007 ("Microfacet Models for Refraction").

static double ggx_d(double NdotH, double alpha) {
    double a2 = alpha * alpha;
    double d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

static double smith_g1(double NdotV, double alpha) {
    double a2 = alpha * alpha;
    return 2.0 * NdotV / (NdotV + std::sqrt(a2 + (1.0 - a2) * NdotV * NdotV));
}

// Shared VNDF half-vector sampler (Heitz 2018), in the local shading frame
// (x = tangent, y = bitangent, z = normal). Samples only the microfacets
// visible from wo, which is what makes the resulting estimator weight
// collapse to albedo*G1(wi) <= albedo. Isotropic callers pass ax == ay.
static Vec3 vndf_h_local(double ox, double oy, double oz,
                         double ax, double ay, double u1, double u2) {
    // 1. Stretch: the slope ellipsoid becomes the unit hemisphere.
    Vec3 v = glm::normalize(Vec3(ax * ox, ay * oy, oz));
    // 2. Orthonormal basis around the stretched view vector.
    double lensq = v.x * v.x + v.y * v.y;
    Vec3 T1 = lensq > 1e-16 ? Vec3(-v.y, v.x, 0.0) / std::sqrt(lensq)
                            : Vec3(1.0, 0.0, 0.0);
    Vec3 T2 = glm::cross(v, T1);
    // 3. Uniform disk point, warped by the projected area of the hemisphere.
    double r = std::sqrt(u1), phi = 2.0 * PI * u2;
    double p1 = r * std::cos(phi), p2 = r * std::sin(phi);
    double sfac = 0.5 * (1.0 + v.z);
    p2 = (1.0 - sfac) * std::sqrt(std::max(0.0, 1.0 - p1 * p1)) + sfac * p2;
    // 4. Project onto the hemisphere, unstretch back.
    Vec3 Nh = p1 * T1 + p2 * T2 +
              std::sqrt(std::max(0.0, 1.0 - p1 * p1 - p2 * p2)) * v;
    return glm::normalize(Vec3(ax * Nh.x, ay * Nh.y, std::max(1e-9, Nh.z)));
}

BSDFSample sample(const GGX& m, const Vec3& n, const Vec3& wo, std::mt19937& rng) {
    // VNDF sampling -- isotropic case of the AnisoGGX sampler below, for the
    // same reason: D*cos sampling was unbiased but its weight
    // G*(wo.h)/(cos_o*(h.n)) is unbounded at grazing (measured up to 21x
    // albedo at theta_o = 80 deg for roughness 0.45 in verify_bsdfs), which is
    // the noise every glossy capture showed. VNDF weight = albedo*G1(wi).
    BSDFSample s;
    double alpha = std::max(1e-4, m.roughness * m.roughness);
    Vec3 a = (std::abs(n.x) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
    Vec3 t = glm::normalize(glm::cross(a, n));
    Vec3 b = glm::cross(n, t);
    double ox = glm::dot(wo, t), oy = glm::dot(wo, b), oz = glm::dot(wo, n);
    if (oz <= 0.0) { s.pdf = 0; s.weight = Vec3(0); return s; }
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Vec3 hl = vndf_h_local(ox, oy, oz, alpha, alpha, dist(rng), dist(rng));
    Vec3 h = t * hl.x + b * hl.y + n * hl.z;
    Vec3 wi = glm::normalize(2.0 * glm::dot(wo, h) * h - wo);
    double NdotL = glm::dot(wi, n);
    double NdotH = glm::dot(h, n);
    double VdotH = glm::dot(wo, h);
    if (NdotL <= 0 || NdotH <= 0 || VdotH <= 0) {
        s.pdf = 0; s.weight = Vec3(0); return s;
    }
    s.wi = wi;
    // pdf and weight of visible-normal sampling; the pdf is also what
    // bsdf::pdf(GGX) reports, and verify_bsdfs asserts they agree.
    s.pdf = smith_g1(oz, alpha) * ggx_d(NdotH, alpha) / (4.0 * oz);
    s.weight = m.albedo * smith_g1(NdotL, alpha);
    s.is_specular = (m.roughness < 0.05);
    return s;
}

double pdf(const Lambert&, const Vec3& n, const Vec3& /*wo*/, const Vec3& wi) {
    double c = glm::dot(wi, n);
    return c > 0.0 ? c * INV_PI : 0.0;
}

double pdf(const GGX& m, const Vec3& n, const Vec3& wo, const Vec3& wi) {
    double NdotV = glm::dot(wo, n), NdotL = glm::dot(wi, n);
    if (NdotV <= 0.0 || NdotL <= 0.0) return 0.0;
    Vec3 h = wo + wi;
    double hlen = glm::length(h);
    if (hlen < 1e-12) return 0.0;
    h /= hlen;
    double NdotH = glm::dot(h, n);
    if (NdotH <= 0.0) return 0.0;
    double alpha = std::max(1e-4, m.roughness * m.roughness);
    return smith_g1(NdotV, alpha) * ggx_d(NdotH, alpha) / (4.0 * NdotV);
}

Vec3 eval(const GGX& m, const Vec3& n, const Vec3& wo, const Vec3& wi) {
    double NdotL = glm::dot(wi, n);
    double NdotV = glm::dot(wo, n);
    if (NdotL <= 0 || NdotV <= 0) return Vec3(0);
    Vec3 h = glm::normalize(wo + wi);
    double NdotH = glm::dot(h, n);
    if (NdotH <= 0) return Vec3(0);
    double alpha = std::max(1e-4, m.roughness * m.roughness);
    double D = ggx_d(NdotH, alpha);
    double G = smith_g1(NdotL, alpha) * smith_g1(NdotV, alpha);
    return m.albedo * (D * G / (4.0 * NdotL * NdotV));
}

} // namespace nrr::bsdf

namespace nrr::bsdf {

// --- Anisotropic GGX (brushed metal) --------------------------------------
// Tangent frame from the material's world-space brush direction, Gram-Schmidt'd
// onto the surface plane. Falls back to an arbitrary frame when the hint is
// parallel to the normal (otherwise the cross product degenerates and the
// highlight direction becomes undefined).
static void aniso_frame(const Vec3& n, const Vec3& hint, Vec3& t, Vec3& b) {
    Vec3 h = hint - n * glm::dot(hint, n);
    double len = glm::length(h);
    if (len < 1e-6) {
        Vec3 a = (std::abs(n.x) > 0.9) ? Vec3(0, 1, 0) : Vec3(1, 0, 0);
        t = glm::normalize(glm::cross(a, n));
    } else {
        t = h / len;
    }
    b = glm::cross(n, t);
}

static double aniso_d(double hx, double hy, double hz, double ax, double ay) {
    if (hz <= 0.0) return 0.0;
    double q = (hx * hx) / (ax * ax) + (hy * hy) / (ay * ay) + hz * hz;
    return 1.0 / (PI * ax * ay * q * q);
}

// Smith lambda for the anisotropic GGX distribution.
static double aniso_lambda(double vx, double vy, double vz, double ax, double ay) {
    if (vz <= 1e-9) return 0.0;
    double a2 = (ax * ax * vx * vx + ay * ay * vy * vy) / (vz * vz);
    return 0.5 * (-1.0 + std::sqrt(1.0 + a2));
}

BSDFSample sample(const AnisoGGX& m, const Vec3& n, const Vec3& wo, std::mt19937& rng) {
    // Visible-normal (VNDF) sampling, Heitz 2018 "Sampling the GGX
    // Distribution of Visible Normals".
    //
    // The original implementation here sampled h from D*cos(h), which is
    // unbiased (verify_bsdfs T1 confirmed the estimator's mean matches
    // brute-force quadrature everywhere) but has an unbounded weight
    // f*cos/pdf = albedo*G*(wo.h)/(cos_o*h.n): measured maxima of 29.5x albedo
    // at grazing incidence, which is what rendered as dense fireflies in the
    // aniso_brushed captures at 48-256 spp. Sampling only the *visible*
    // microfacets makes the pdf absorb both D and the masking term, and with
    // separable Smith the weight collapses algebraically to
    //
    //     weight = albedo * G1(wi) <= albedo,
    //
    // so no sample can exceed the surface's own reflectance. Same expectation
    // (T1 re-verified against the same quadrature), bounded variance (T3).
    BSDFSample s;
    double ax = std::max(1e-4, m.alpha_x);
    double ay = std::max(1e-4, m.alpha_y);

    Vec3 t, b;
    aniso_frame(n, m.tangent_hint, t, b);

    // Local frame: x=tangent, y=bitangent, z=normal.
    double ox = glm::dot(wo, t), oy = glm::dot(wo, b), oz = glm::dot(wo, n);
    if (oz <= 0.0) { s.pdf = 0; s.weight = Vec3(0); return s; }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    Vec3 hl = vndf_h_local(ox, oy, oz, ax, ay, dist(rng), dist(rng));
    double hx = hl.x, hy = hl.y, hz = hl.z;
    Vec3 h = t * hx + b * hy + n * hz;

    Vec3 wi = glm::normalize(2.0 * glm::dot(wo, h) * h - wo);
    double ix = glm::dot(wi, t), iy = glm::dot(wi, b), iz = glm::dot(wi, n);
    double VdotH = glm::dot(wo, h);
    if (iz <= 0.0 || VdotH <= 0.0) { s.pdf = 0; s.weight = Vec3(0); return s; }

    // pdf of wi under VNDF: G1(wo) * D(h) / (4 * cos_o). Kept exact (the
    // integrator gates on pdf < EPS) even though it cancels out of the weight.
    double D = aniso_d(hx, hy, hz, ax, ay);
    double G1o = 1.0 / (1.0 + aniso_lambda(ox, oy, oz, ax, ay));
    s.wi = wi;
    s.pdf = G1o * D / (4.0 * oz);
    s.weight = m.albedo / (1.0 + aniso_lambda(ix, iy, iz, ax, ay));  // = albedo*G1(wi)
    s.is_specular = (ax < 0.002 && ay < 0.002);
    return s;
}

double pdf(const AnisoGGX& m, const Vec3& n, const Vec3& wo, const Vec3& wi) {
    double ax = std::max(1e-4, m.alpha_x), ay = std::max(1e-4, m.alpha_y);
    Vec3 t, b;
    aniso_frame(n, m.tangent_hint, t, b);
    double oz = glm::dot(wo, n), iz = glm::dot(wi, n);
    if (oz <= 0.0 || iz <= 0.0) return 0.0;
    Vec3 h = wo + wi;
    double hlen = glm::length(h);
    if (hlen < 1e-12) return 0.0;
    h /= hlen;
    double hx = glm::dot(h, t), hy = glm::dot(h, b), hz = glm::dot(h, n);
    if (hz <= 0.0) return 0.0;
    double ox = glm::dot(wo, t), oy = glm::dot(wo, b);
    double G1o = 1.0 / (1.0 + aniso_lambda(ox, oy, oz, ax, ay));
    return G1o * aniso_d(hx, hy, hz, ax, ay) / (4.0 * oz);
}

Vec3 eval(const AnisoGGX& m, const Vec3& n, const Vec3& wo, const Vec3& wi) {
    double ax = std::max(1e-4, m.alpha_x);
    double ay = std::max(1e-4, m.alpha_y);
    Vec3 t, b;
    aniso_frame(n, m.tangent_hint, t, b);

    double oz = glm::dot(wo, n), iz = glm::dot(wi, n);
    if (oz <= 0.0 || iz <= 0.0) return Vec3(0);

    Vec3 h = glm::normalize(wo + wi);
    double hx = glm::dot(h, t), hy = glm::dot(h, b), hz = glm::dot(h, n);
    if (hz <= 0.0) return Vec3(0);

    double D = aniso_d(hx, hy, hz, ax, ay);
    double G = 1.0 / ((1.0 + aniso_lambda(glm::dot(wo,t), glm::dot(wo,b), oz, ax, ay)) *
                      (1.0 + aniso_lambda(glm::dot(wi,t), glm::dot(wi,b), iz, ax, ay)));
    return m.albedo * (D * G / (4.0 * iz * oz));
}

static double fresnel_schlick(double cos_theta, double n1, double n2) {
    double r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    double m = 1.0 - cos_theta;
    return r0 + (1.0 - r0) * m * m * m * m * m;
}

BSDFSample sample(const Glass& g, const Vec3& n_in, const Vec3& wo, std::mt19937& rng) {
    BSDFSample s;
    s.is_specular = true;

    Vec3 n = n_in;
    double cos_i = glm::dot(wo, n);
    double n1 = 1.0, n2 = g.ior;
    if (cos_i < 0) { n = -n; cos_i = -cos_i; std::swap(n1, n2); }

    double eta = n1 / n2;
    double sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    if (sin2_t > 1.0) {
        // Total internal reflection
        s.wi = glm::normalize(2.0 * cos_i * n - wo);
        s.pdf = 1.0;
        s.weight = g.albedo;
        return s;
    }
    double cos_t = std::sqrt(1.0 - sin2_t);
    double F = fresnel_schlick(cos_i, n1, n2);

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    if (dist(rng) < F) {
        // Reflect
        s.wi = glm::normalize(2.0 * cos_i * n - wo);
        s.pdf = F;
        s.weight = g.albedo;   // F cancels with prob F
    } else {
        // Refract
        s.wi = glm::normalize(eta * (-wo) + (eta * cos_i - cos_t) * n);
        s.pdf = 1.0 - F;
        s.weight = g.albedo;   // (1-F) cancels with prob (1-F)
    }
    return s;
}

BSDFSample sample(const NullBSDF&, const Vec3& /*n*/, const Vec3& wo, std::mt19937& /*rng*/) {
    BSDFSample s;
    s.wi = -wo;            // continue straight through
    s.pdf = 1.0;
    s.weight = Vec3(1.0);  // no attenuation
    s.is_specular = true;  // don't NEE; treat as delta transmission
    return s;
}

} // namespace nrr::bsdf
