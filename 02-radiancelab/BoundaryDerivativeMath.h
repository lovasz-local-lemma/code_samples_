#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <glm/glm.hpp>

namespace Prism::Research::Boundary {
constexpr double pi = 3.14159265358979323846;

// A deliberately restricted transport problem: a sharp, uniformly collimated
// quad emits downward onto static Lambertian receivers. Translation changes
// support; its irradiance and the receiver's light visibility do not change.
// No edge blur, finite difference stencil or kernel enters analyticDerivative.
struct Config {
    double sourceX = .10, sourceZ = .02, halfX = 1.02, halfZ = .74;
    glm::dvec3 direction{1, 0, .35};
    double epsilon = .025, yaw = .43, elevation = 2.9;
    double irradiance = 2.8, albedo = .72;
    bool curved = true, shadows = true;
};
struct Hit {
    glm::dvec3 p{}, n{};
    double t = 1e30;
    int object = -1;
};
inline Hit intersect(const glm::dvec3& ro, const glm::dvec3& rd, bool curved) {
    Hit h;
    if (rd.y < -1e-10) {
        const double t = -ro.y / rd.y;
        const auto p = ro + t * rd;
        if (t > 1e-7 && std::abs(p.x) < 2.55 && std::abs(p.z) < 1.9)
            h = {p, {0, 1, 0}, t, 0};
    }
    if (curved) {
        const glm::dvec3 centers[] = {{-.69, .65, -.05}, {.78, .45, -.4}};
        for (int i = 0; i < 2; ++i) {
            const auto v = ro - centers[i];
            const double b = glm::dot(v, rd);
            const double discriminant = b * b - glm::dot(v, v) + centers[i].y * centers[i].y;
            if (discriminant < 0) continue;
            double t = -b - std::sqrt(discriminant);
            if (t <= 1e-7) t = -b + std::sqrt(discriminant);
            if (t > 1e-7 && t < h.t) {
                const auto p = ro + t * rd;
                h = {p, glm::normalize(p - centers[i]), t, i + 1};
            }
        }
    }
    return h;
}
struct Camera {
    glm::dvec3 origin, forward, right, up;
    double scaleX, scaleY;
    Camera(const Config& c, int w, int h) {
        origin = {4.0 * std::sin(c.yaw), c.elevation, 4.0 * std::cos(c.yaw)};
        forward = glm::normalize(glm::dvec3(0, .27, 0) - origin);
        right = glm::normalize(glm::cross(forward, glm::dvec3(0, 1, 0)));
        up = glm::cross(right, forward);
        scaleX = 1.16 * double(w) / h;
        scaleY = 1.16;
    }
    glm::dvec3 ray(double x, double y, int w, int h) const {
        return glm::normalize(forward + right * ((x / w - .5) * scaleX)
                                     + up * ((.5 - y / h) * scaleY));
    }
};

// Affine approximation of receiver coordinates over ONE screen pixel.
// The square [-1/2,1/2]^2 has area one, so integrals are already pixel averages.
struct Patch {
    double u = 0, v = 0;
    glm::dvec2 du{}, dv{};
    double radiance = 0;
    Hit hit;
};
inline Patch patch(const Config& c, const Camera& cam, int x, int y, int w, int h) {
    Patch p;
    const auto rd = cam.ray(x + .5, y + .5, w, h);
    p.hit = intersect(cam.origin, rd, c.curved);
    if (p.hit.object < 0) return p;
    p.u = p.hit.p.x; p.v = p.hit.p.z;
    const double cosine = std::max(0.0, p.hit.n.y);
    p.radiance = c.irradiance * c.albedo * cosine / pi;
    if (c.shadows && intersect(p.hit.p + p.hit.n * 1e-5, {0, 1, 0}, c.curved).object >= 0)
        p.radiance = 0;
    // Differentiate the camera ray's intersection with the local tangent
    // plane, rather than jumping to another object at pixel/silhouette edges.
    auto tangent = [&](double px, double py) {
        const auto d = cam.ray(px, py, w, h);
        double denom = glm::dot(d, p.hit.n);
        if (std::abs(denom) < 1e-10) denom = std::copysign(1e-10, denom);
        return cam.origin + d * (glm::dot(p.hit.p - cam.origin, p.hit.n) / denom);
    };
    const auto dx = tangent(x + 1., y + .5) - tangent(x, y + .5);
    const auto dy = tangent(x + .5, y + 1.) - tangent(x + .5, y);
    p.du = {dx.x, dy.x}; p.dv = {dx.z, dy.z};
    return p;
}

struct Polygon { std::array<glm::dvec2, 16> p{}; int n = 0; };
inline Polygon square() {
    Polygon p; p.n = 4;
    p.p[0] = {-.5, -.5}; p.p[1] = {.5, -.5};
    p.p[2] = {.5, .5}; p.p[3] = {-.5, .5}; return p;
}
// Clip to dot(normal, position) <= upper. Clipping a square to four source
// half planes needs at most eight vertices; the fixed buffer avoids allocation.
inline Polygon clip(const Polygon& input, const glm::dvec2& normal, double upper) {
    Polygon out;
    if (!input.n) return out;
    auto previous = input.p[input.n - 1];
    double dp = glm::dot(normal, previous) - upper;
    for (int i = 0; i < input.n; ++i) {
        const auto current = input.p[i];
        const double dc = glm::dot(normal, current) - upper;
        if ((dc <= 0) != (dp <= 0)) out.p[out.n++] = previous + (current - previous) * (dp / (dp - dc));
        if (dc <= 0) out.p[out.n++] = current;
        previous = current; dp = dc;
    }
    return out;
}
inline double coverage(const Patch& p, const Config& c, double translation = 0) {
    const double cx = c.sourceX + translation * c.direction.x;
    const double cz = c.sourceZ + translation * c.direction.z;
    auto poly = square();
    poly = clip(poly, p.du, cx + c.halfX - p.u);
    poly = clip(poly, -p.du, p.u - cx + c.halfX);
    poly = clip(poly, p.dv, cz + c.halfZ - p.v);
    poly = clip(poly, -p.dv, p.v - cz + c.halfZ);
    double area = 0;
    for (int i = 0; i < poly.n; ++i) {
        const auto a = poly.p[i], b = poly.p[(i + 1) % poly.n];
        area += a.x * b.y - a.y * b.x;
    }
    return std::clamp(std::abs(area) * .5, 0.0, 1.0);
}

// Integral of delta(u-edge) * 1{otherLo <= v <= otherHi} over a screen pixel.
// The coarea factor 1/|grad_screen u| is essential: drawing an arbitrary-width
// colored line would have the wrong derivative normalization.
inline double edgeIntegral(double u, const glm::dvec2& gradU, double edge,
                           double v, const glm::dvec2& gradV, double otherLo, double otherHi) {
    const double length = glm::length(gradU);
    if (length < 1e-14) return 0;
    const auto normal = gradU / length;
    const auto origin = normal * ((edge - u) / length);
    const glm::dvec2 direction(-normal.y, normal.x);
    double t0 = -1e30, t1 = 1e30;
    auto interval = [&](double a, double b, double lo, double hi) {
        if (std::abs(b) < 1e-14) return a >= lo && a <= hi;
        double q0 = (lo - a) / b, q1 = (hi - a) / b;
        if (q0 > q1) std::swap(q0, q1);
        t0 = std::max(t0, q0); t1 = std::min(t1, q1);
        return t1 > t0;
    };
    if (!interval(origin.x, direction.x, -.5, .5) ||
        !interval(origin.y, direction.y, -.5, .5) ||
        !interval(v + glm::dot(gradV, origin), glm::dot(gradV, direction), otherLo, otherHi)) return 0;
    return std::max(0.0, t1 - t0) / length;
}
inline double analyticDerivative(const Patch& p, const Config& c) {
    const double left = c.sourceX - c.halfX, right = c.sourceX + c.halfX;
    const double back = c.sourceZ - c.halfZ, front = c.sourceZ + c.halfZ;
    const double dx = edgeIntegral(p.u, p.du, right, p.v, p.dv, back, front)
                    - edgeIntegral(p.u, p.du, left, p.v, p.dv, back, front);
    const double dz = edgeIntegral(p.v, p.dv, front, p.u, p.du, left, right)
                    - edgeIntegral(p.v, p.dv, back, p.u, p.du, left, right);
    return p.radiance * (c.direction.x * dx + c.direction.z * dz);
}
struct Sample { double base = 0, derivative = 0, finite = 0, halfFinite = 0; Hit hit; };
inline Sample evaluate(const Patch& p, const Config& c) {
    Sample result; result.hit = p.hit;
    if (p.radiance <= 0) return result;
    result.base = p.radiance * coverage(p, c);
    result.derivative = analyticDerivative(p, c);
    const double e = std::max(1e-8, c.epsilon);
    result.finite = p.radiance * (coverage(p, c, e) - coverage(p, c, -e)) / (2 * e);
    result.halfFinite = p.radiance * (coverage(p, c, e * .5) - coverage(p, c, -e * .5)) / e;
    return result;
}
struct Display { int mode = 0; bool overlay = true; double gain = .07, baseline = .16; };
inline glm::dvec3 color(const Sample& s, const Display& d) {
    if (s.hit.object < 0) return {.003, .006, .012};
    const double grid = (std::abs(std::sin(s.hit.p.x * 10)) < .024 || std::abs(std::sin(s.hit.p.z * 10)) < .024) ? .017 : 0;
    const double normalLight = .3 + .7 * std::max(0.0, glm::dot(s.hit.n, glm::normalize(glm::dvec3(-1, 2, 1))));
    const auto context = glm::dvec3(.023 + grid, .036 + grid, .045 + grid) * normalLight;
    const glm::dvec3 material(.88, .80, .63);
    if (d.mode == 1) return context + material * (1.35 * s.base);
    const double value = d.mode == 2 ? s.finite : d.mode == 3 ? s.finite - s.derivative : s.derivative;
    auto c = d.overlay ? context + material * (d.baseline * s.base) : glm::dvec3(.0015);
    c += (value >= 0 ? glm::dvec3(1., .022, .005) : glm::dvec3(.005, .11, 1.)) * (std::abs(value) * d.gain);
    return c;
}
inline void rgba(const glm::dvec3& color, unsigned char* pixel) {
    for (int j = 0; j < 3; ++j) pixel[j] = static_cast<unsigned char>(255 * std::pow(1 - std::exp(-std::max(0., color[j])), 1 / 2.2));
    pixel[3] = 255;
}
inline std::vector<Sample> render(const Config& c, int w, int h) {
    std::vector<Sample> image(size_t(w) * h); const Camera cam(c, w, h);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
        image[size_t(y) * w + x] = evaluate(patch(c, cam, x, y, w, h), c);
    return image;
}
inline std::vector<unsigned char> pixels(const std::vector<Sample>& image, const Display& display) {
    std::vector<unsigned char> result(image.size() * 4);
    for (size_t i = 0; i < image.size(); ++i) rgba(color(image[i], display), result.data() + i * 4);
    return result;
}
} // namespace Prism::Research::Boundary
