#pragma once
#include "path_tracer/scene.h"
#include "path_tracer/bsdf.h"
#include "path_tracer/medium.h"
#include <random>
#include <algorithm>
#include <optional>

namespace nrr {

// Veach's power heuristic (beta = 2). Balances the two direct-lighting
// estimators: light-area sampling (great for diffuse, terrible for sharp
// lobes) and BSDF sampling (the reverse). Without it, NEE alone evaluates a
// razor-thin GGX lobe against area samples and occasionally lands near the
// specular ridge where f is enormous -- measured as ~60 isolated fireflies
// per 384x384 frame in the aniso scene, ZERO in the all-Lambert control.
inline double power_heuristic(double pf, double pg) {
    double f2 = pf * pf, g2 = pg * pg;
    double d = f2 + g2;
    return d > 0.0 ? f2 / d : 0.0;
}

class PathIntegrator {
public:
    int max_bounces = 8;

    Vec3 trace(const Scene& scene, Ray ray, std::mt19937& rng) const {
        Vec3 throughput(1.0);
        Vec3 radiance(0.0);
        std::optional<Medium> current_medium;
        // MIS bookkeeping across bounces. nee_at_prev: whether the vertex
        // that produced the current ray performed next-event estimation (only
        // then does an emitter hit compete with a light sample and need
        // downweighting). prev_pdf: the BSDF pdf of that bounce, in solid
        // angle. dist_since_bounce: accumulated distance across NullBSDF
        // pass-throughs, so the emitter-hit light pdf uses the true distance
        // from the last real vertex rather than the last segment only.
        bool nee_at_prev = false;
        double prev_pdf = 0.0;
        double dist_since_bounce = 0.0;

        for (int bounce = 0; bounce <= max_bounces; ++bounce) {
            HitRecord rec;
            bool surface_hit = scene.intersect(ray, rec);
            double t_surface = surface_hit ? rec.t : 1e30;

            // Medium event sampling. sample_interaction returns the full
            // throughput factor for whichever event it picked -- the
            // transmittance is already folded into the estimator (it cancels
            // against the sampling pdf for homogeneous media, and is carried by
            // the real-collision rate for heterogeneous ones). Multiplying by a
            // separate transmittance here, as this used to, attenuated every
            // medium twice and made them render too dark.
            if (current_medium) {
                MediumSample ms = sample_interaction(*current_medium, ray,
                                                      t_surface, rng);
                throughput *= ms.weight;

                if (ms.scattered) {
                    // In-scatter event
                    Vec3 p = ray.at(ms.t);
                    Vec3 wo = -ray.direction;
                    Vec3 wi = sample_phase(*current_medium, wo, rng);

                    // (Skip volumetric NEE for B; rely on path tracing.)

                    ray = Ray{p, wi};
                    // No NEE is performed at medium vertices (Phase B), so a
                    // subsequent emitter hit is the ONLY estimator for this
                    // path and must be credited in full. The old
                    // specular_bounce=false here made the emissive branch drop
                    // it entirely: every "scatter, then hit the lamp" path
                    // contributed zero, and all volume captures were darker
                    // than ground truth.
                    nee_at_prev = false;
                    prev_pdf = 0.0;
                    dist_since_bounce = 0.0;

                    // Russian roulette
                    if (bounce > 3) {
                        double max_comp = std::max({throughput.x, throughput.y, throughput.z});
                        double q = std::max(0.05, 1.0 - max_comp);
                        std::uniform_real_distribution<double> dist(0.0, 1.0);
                        if (dist(rng) < q) break;
                        throughput /= (1.0 - q);
                    }
                    continue;
                }
            }

            if (!surface_hit) break;

            const Material& mat = scene.materials[rec.material_id];

            // NullBSDF: medium boundary; toggle current_medium and continue.
            // Use rec.front_face (set before set_face_normal flips normal to
            // oppose the ray) — checking dot(ray.dir, rec.normal) gives the
            // wrong answer because rec.normal is already ray-aligned.
            if (std::holds_alternative<NullBSDF>(mat.bsdf)) {
                if (mat.interior_medium) {
                    current_medium = rec.front_face ? mat.interior_medium : std::nullopt;
                }
                dist_since_bounce += rec.t;   // same path vertex, longer ray
                ray = Ray{rec.position + ray.direction * 1e-4, ray.direction};
                continue;
            }

            if (mat.is_emissive()) {
                // MIS weight against the light sample the previous vertex
                // could have drawn toward this point. Full credit when no NEE
                // happened there (camera ray, delta bounce, medium scatter) or
                // when NEE could not have reached this side of the emitter.
                double w = 1.0;
                if (nee_at_prev && rec.front_face &&
                    !scene.light_indices.empty() && rec.shape_index >= 0) {
                    double cos_l = glm::dot(-ray.direction, rec.normal);
                    double area = scene.shapes[rec.shape_index]->area();
                    if (cos_l > EPS && area > 1e-12) {
                        double d_total = dist_since_bounce + rec.t;
                        double p_light = (d_total * d_total) /
                            (cos_l * area * double(scene.light_indices.size()));
                        w = power_heuristic(prev_pdf, p_light);
                    }
                }
                radiance += throughput * mat.emission * w;
                break;
            }

            bool did_nee = !scene.light_indices.empty() && !is_specular(mat);
            if (did_nee) {
                radiance += throughput * sample_light(scene, rec, ray, rng, current_medium);
            }

            Vec3 wo = -ray.direction;
            BSDFSample s = sample_material(mat, rec.normal, wo, rng);
            if (s.pdf < EPS) break;
            nee_at_prev = did_nee;
            prev_pdf = s.pdf;
            dist_since_bounce = 0.0;
            throughput *= s.weight;

            if (bounce > 3) {
                double max_comp = std::max({throughput.x, throughput.y, throughput.z});
                double q = std::max(0.05, 1.0 - max_comp);
                std::uniform_real_distribution<double> dist(0.0, 1.0);
                if (dist(rng) < q) break;
                throughput /= (1.0 - q);
            }

            ray = Ray{rec.position + rec.normal * 1e-4, s.wi};
        }
        return radiance;
    }

private:
    Vec3 sample_light(const Scene& scene, const HitRecord& shading_point,
                      const Ray& incoming, std::mt19937& rng,
                      const std::optional<Medium>& current_medium = std::nullopt) const {
        std::uniform_int_distribution<int> light_dist(0, static_cast<int>(scene.light_indices.size()) - 1);
        int li = light_dist(rng);
        const Shape* light = scene.shapes[scene.light_indices[li]].get();
        const Material& light_mat = scene.materials[light->material_id];

        Vec3 light_normal;
        Vec3 light_pos = light->sample_point(rng, light_normal);
        Vec3 to_light = light_pos - shading_point.position;
        double dist2 = glm::dot(to_light, to_light);
        double dist = std::sqrt(dist2);
        Vec3 wi = to_light / dist;

        double cos_shading = glm::dot(wi, shading_point.normal);
        double cos_light = -glm::dot(wi, light_normal);
        if (cos_shading <= 0 || cos_light <= 0) return Vec3(0);

        Vec3 tr = shadow_transmittance(
            scene, shading_point.position + shading_point.normal * 1e-4,
            wi, dist - 1e-3, current_medium, rng);
        if (tr.x <= 0.0 && tr.y <= 0.0 && tr.z <= 0.0) return Vec3(0);

        const Material& shading_mat = scene.materials[shading_point.material_id];
        Vec3 wo = -incoming.direction;
        Vec3 f = eval_material(shading_mat, shading_point.normal, wo, wi);

        // Solid-angle pdf of this light sample, and the pdf with which the
        // BSDF sampler could have produced the same direction; the power
        // heuristic shifts sharp-lobe direct lighting onto the BSDF-sampling
        // estimator, which handles it with bounded weights.
        double n_lights = double(scene.light_indices.size());
        double p_light_sa = dist2 / (cos_light * light->area() * n_lights);
        double p_bsdf = pdf_material(shading_mat, shading_point.normal, wo, wi);
        double w = power_heuristic(p_light_sa, p_bsdf);

        return f * light_mat.emission * cos_shading / p_light_sa * w * tr;
    }

    // Transmittance-aware visibility: walks the shadow ray through NullBSDF
    // boundaries, applying each traversed medium's transmittance per segment,
    // and returns 0 at the first real occluder. Replaces a binary
    // scene.intersect test with two defects: it treated transparent medium
    // shells as opaque (every shading point whose shadow ray crossed the
    // smoke ball read as fully shadowed, darkening the floor around the
    // subject in both volume scenes), and the medium attenuation that was
    // applied afterwards used the shading vertex's medium over the ray's FULL
    // length even though the medium ends at the shell boundary.
    Vec3 shadow_transmittance(const Scene& scene, Vec3 pos, const Vec3& wi,
                              double dist_left,
                              std::optional<Medium> medium,
                              std::mt19937& rng) const {
        Vec3 tr(1.0);
        for (int guard = 0; guard < 64; ++guard) {
            Ray r{pos, wi, 1e-4, dist_left};
            HitRecord hr;
            if (!scene.intersect(r, hr)) {
                if (medium) tr *= transmittance(*medium, r, dist_left, rng);
                return tr;
            }
            const Material& m = scene.materials[hr.material_id];
            if (!std::holds_alternative<NullBSDF>(m.bsdf)) return Vec3(0.0);
            if (medium) tr *= transmittance(*medium, r, hr.t, rng);
            if (m.interior_medium)
                medium = hr.front_face ? m.interior_medium : std::nullopt;
            pos = hr.position + wi * 1e-4;
            dist_left -= hr.t + 1e-4;
            if (dist_left <= 1e-3) return tr;
        }
        return tr;
    }
};

} // namespace nrr
