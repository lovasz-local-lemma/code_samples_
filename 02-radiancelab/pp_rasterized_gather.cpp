// Excerpt: RadianceLab/src/render/PPIntegrator.cpp, lines 9277-9857 of 11012. Screen-space bounding of frozen photon primitives (monotone-chain hull, supporting-plane cones), counting sort into scanlines, and a parallel row gather with thread-local scratch.
// Not a standalone translation unit; see the folder README for the surrounding types.

static std::vector<PPPrim>   g_ppBatch;
static std::vector<uint32_t> g_ppBatchStart;

// A cheap FNV-1a over the batch bytes. Run either side of the gather: if a cross function ever
// writes through its const& (a const_cast, a union pun, a stray memcpy), the hash moves and the
// "read-only during the camera pass" claim is falsified at runtime instead of being asserted in
// a comment. Costs N*sizeof(PPPrim) bytes of hashing once per pass -- ~18 kB at N = 64.
static uint64_t ppBatchHash(const std::vector<PPPrim>& b) {
    uint64_t hsh = 1469598103934665603ull;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(b.data());
    const size_t n = b.size() * sizeof(PPPrim);
    for (size_t i = 0; i < n; ++i) { hsh ^= p[i]; hsh *= 1099511628211ull; }
    return hsh;
}

// ====================== THE RASTERIZED GATHER (strategy 25, pp.raster) =======================
// The tracing gather asks, once per pixel, "does primitive k cross me?" -- N questions
// per pixel, and for a swept sheet almost every answer is no. The rasterizer inverts the loop:
// bound each FROZEN primitive in screen space, and visit only the pixels inside the bound. The
// crossing that then runs is the same closed form on the same record, so the two gathers
// differ in nothing but the order the pixels are visited in.
//
// BIT-IDENTICAL TO THE TRACING GATHER, and that is the acceptance test -- not a similarity
// claim, not a converged ratio. Two properties buy it and both are load-bearing:
//
//   (1) THE BOUND IS CONSERVATIVE. Every pixel whose crossing would return non-zero is inside
//       it, so every term the raster skips is an exact +0.0f, and dropping an exact zero out
//       of a left-to-right float sum changes nothing. A bound that is nearly conservative is
//       NOT nearly bit-identical: it is a small, camera-dependent energy deficit that reads as
//       shading rather than as noise. The guard band below is the whole defence.
//
//   (2) EACH PIXEL STILL ACCUMULATES IN (draw k, record i) ORDER. The batch is already laid
//       out in that order, so walking records outermost and pixels innermost hands every pixel
//       the same SEQUENCE of additions the tracing loop gave it. The one place the order could
//       still slip is the PER-DRAW GROUPING -- the tracing loop sums a draw's records into `c`
//       and only then does `color += c`, and float addition does not associate, so
//       color + (a+b) is not (color + a) + b. Only the beam emits several records per draw,
//       but that is enough: the row therefore keeps a SECOND accumulator and flushes it
//       whenever a pixel's draw index changes. See lastK[] in ppRasterGather.
//
// Not a splat, deliberately: no atomics, no per-thread frame buffers, no non-deterministic
// summation order. Parallelised over ROWS exactly like the tracing gather, and each row owns
// its own accumulators, so two threads never touch the same pixel.

// ---- Projection ------------------------------------------------------------------------------
// The inverse of the ray the gather actually generates. sampleOnePixel builds its camera ray as
//   uvx = ((x+jx)/w)*2-1,  uvy = 1-((y+jy)/h)*2,  cam.generateRay(uv, lens=0)
// and PinholeCamera::generateRay turns that into normalize(-w + uvx*halfW*u + uvy*halfH*v).
// So a world point's continuous pixel coordinate is (x+jx, y+jy), and pixel X sees every point
// that projects into [X, X+1). halfW carries the aspect factor (halfW = halfH*aspect) --
// Tungsten's binner puts it on the OTHER axis, the single most likely bug in a port, so this
// is written against Prism's own Camera::project convention and then CHECKED against
// cam.generateRay at pass 1 (ppRasterProjCheck).
struct PPProj {
    glm::vec3 pos{0.0f}, U{0.0f}, V{0.0f}, F{0.0f};
    float ax = 1.0f, ay = 1.0f, cx = 0.0f, cy = 0.0f;
    float halfW = 1.0f, halfH = 1.0f;
    int   w = 0, h = 0;
};
static PPProj ppMakeProj(const Camera& cam, int w, int h) {
    PPProj P;
    const Camera::Basis b = cam.computeBasis();
    P.pos = cam.params().position;
    P.U = b.u; P.V = b.v; P.F = -b.w;                     // forward = normalize(lookAt - pos)
    const float halfH = std::tan(0.5f * glm::radians(cam.fovY()));
    const float halfW = halfH * cam.params().aspectRatio;
    P.ax = 0.5f * (float)w / halfW;  P.cx = 0.5f * (float)w;
    P.ay = 0.5f * (float)h / halfH;  P.cy = 0.5f * (float)h;
    P.halfW = halfW; P.halfH = halfH;
    P.w = w; P.h = h;
    // Note that swapping ax and ay is a NO-OP here, because aspectRatio is w/h and the two scales
    // are then equal -- which is also why an axis-convention bug can hide in a square-pixel
    // render. Putting the aspect factor on the OTHER axis (Tungsten's convention) is the port bug
    // ppRasterProjCheck exists to catch.
    return P;
}
// false = "this point cannot be projected safely" (behind or almost on the camera plane, or so
// close to it that the pixel coordinate has lost its low bits). The caller's ONLY legal
// response is to widen the bound, never to drop the primitive.
static inline bool ppProjPt(const PPProj& P, const glm::vec3& p, glm::vec2& s) {
    const glm::vec3 e = p - P.pos;
    const float fz = glm::dot(e, P.F);
    if (!(fz > 1e-3f)) return false;
    const float inv = 1.0f / fz;
    s.x = glm::dot(e, P.U) * inv * P.ax + P.cx;
    s.y = -glm::dot(e, P.V) * inv * P.ay + P.cy;
    return (std::fabs(s.x) < 1.0e5f && std::fabs(s.y) < 1.0e5f);
}

// ---- The bound: a convex screen polygon, tested per PIXEL -------------------------------------
static constexpr int   kPPRastMaxEdges = 24;
// 0.7072 px -- half a pixel diagonal -- is the REQUIREMENT, because the polygon is tested at
// pixel CENTRES while the camera ray is jittered anywhere inside the pixel. The rest is slop
// for the projection's float error. (Tungsten needs 10.66 px for the same job only because it
// tests at 4x4 TILE centres, FrustumBinner.cpp:6; testing per pixel buys the tightness back.)
static constexpr float kPPRastGuard = 2.0f;
struct PPRastBound {
    int   x0 = 0, y0 = 0, x1 = -1, y1 = -1;      // inclusive; empty when x1 < x0 or y1 < y0
    int   ne = 0;                                 // accept a pixel when every edge is >= 0
    float ea[kPPRastMaxEdges], eb[kPPRastMaxEdges], ec[kPPRastMaxEdges];
};
static inline void ppRastFull(const PPProj& P, PPRastBound& B) {
    B.x0 = 0; B.y0 = 0; B.x1 = P.w - 1; B.y1 = P.h - 1; B.ne = 0;
}
static inline float ppCross2(const glm::vec2& o, const glm::vec2& a, const glm::vec2& b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}
// Andrew's monotone chain. Sorts `p` in place; `out` needs 2n+1 slots. Collinear points are
// dropped (the <= 0), so the result is a strict convex polygon in CCW order.
static int ppHull2D(glm::vec2* p, int n, glm::vec2* out) {
    if (n <= 2) { for (int i = 0; i < n; ++i) out[i] = p[i]; return n; }
    std::sort(p, p + n, [](const glm::vec2& A, const glm::vec2& Bv) {
        return (A.x < Bv.x) || (A.x == Bv.x && A.y < Bv.y);
    });
    int m = 0;
    for (int i = 0; i < n; ++i) {
        while (m >= 2 && ppCross2(out[m - 2], out[m - 1], p[i]) <= 0.0f) --m;
        out[m++] = p[i];
    }
    for (int i = n - 2, t = m + 1; i >= 0; --i) {
        while (m >= t && ppCross2(out[m - 2], out[m - 1], p[i]) <= 0.0f) --m;
        out[m++] = p[i];
    }
    return m - 1;
}
// AND one more convex world-space hull into B. Every primitive is handed to this as a POINT
// SET whose convex hull contains it: central projection restricted to the strictly-in-front
// half space is projective, so it maps a convex hull to the convex hull of the projected
// points, and the screen footprint of the primitive is inside the hull of the projected set.
// Returns false and leaves B untouched when some point will not project -- an unusable
// constraint, never a reason to shrink the bound.
static bool ppRastAddHull(const PPProj& P, const glm::vec3* pts, int n, PPRastBound& B) {
    if (n < 1 || n > 32) return false;
    glm::vec2 s[32], hull[66];
    for (int i = 0; i < n; ++i) if (!ppProjPt(P, pts[i], s[i])) return false;
    const int m = ppHull2D(s, n, hull);
    if (m < 1) return false;

    float lox = hull[0].x, hix = hull[0].x, loy = hull[0].y, hiy = hull[0].y;
    for (int i = 1; i < m; ++i) {
        lox = std::min(lox, hull[i].x); hix = std::max(hix, hull[i].x);
        loy = std::min(loy, hull[i].y); hiy = std::max(hiy, hull[i].y);
    }
    // pixel X is tested at X+0.5, so X is in range iff X+0.5 lies in [lo-guard, hi+guard].
    const int nx0 = (int)std::ceil (lox - kPPRastGuard - 0.5f);
    const int nx1 = (int)std::floor(hix + kPPRastGuard - 0.5f);
    const int ny0 = (int)std::ceil (loy - kPPRastGuard - 0.5f);
    const int ny1 = (int)std::floor(hiy + kPPRastGuard - 0.5f);
    B.x0 = std::max(B.x0, nx0); B.x1 = std::min(B.x1, nx1);
    B.y0 = std::max(B.y0, ny0); B.y1 = std::min(B.y1, ny1);

    // Edge half-planes, only when the hull is a real polygon and there is room. A degenerate
    // hull (a point, a segment, a collinear set) keeps the box alone, which is still correct.
    if (m >= 3 && B.ne + m <= kPPRastMaxEdges) {
        double area2 = 0.0;
        for (int i = 0; i < m; ++i) {
            const glm::vec2& a = hull[i]; const glm::vec2& b = hull[(i + 1) % m];
            area2 += (double)a.x * (double)b.y - (double)b.x * (double)a.y;
        }
        const float sgn = (area2 < 0.0) ? -1.0f : 1.0f;   // interior on the >= 0 side either way
        for (int i = 0; i < m; ++i) {
            const glm::vec2& a = hull[i]; const glm::vec2& b = hull[(i + 1) % m];
            const float dx = b.x - a.x, dy = b.y - a.y;
            const float len = std::sqrt(dx * dx + dy * dy);
            // f(p) = (b-a) x (p-a), UNNORMALISED -- so it is scaled by |b-a| and adding
            // guard*|b-a| to the constant slides the line outward by exactly `guard` pixels.
            // (Tungsten's triangleSetup does the same trick, FrustumBinner.cpp:16.)
            B.ea[B.ne] = -dy * sgn;
            B.eb[B.ne] =  dx * sgn;
            B.ec[B.ne] = (dy * a.x - dx * a.y) * sgn + kPPRastGuard * len;
            ++B.ne;
        }
    }
    return true;
}
// ---- The straddle case: bound the DIRECTION cone instead of the projection ------------------
// A primitive with a point behind the camera plane has no finite projected hull -- a point just
// in front of the pinhole projects arbitrarily far out -- and ppRastAddHull correctly refuses,
// which costs that record the whole frame. MEASURED, and it is not a rounding error: on preset
// 48 / strategy 18 at N=64, four records of seventy-eight straddle and they own 71 % of the
// rasterizer's total pixel budget; on preset 47 / strategy 3, ONE record of sixty-four owns 99 %.
//
// The fix needs no projection at all. Every camera ray leaves the same point (`ray.origin` is
// `m_params.position`, PinholeCamera::generateRay), so "which pixels can see this primitive" is
// really "which DIRECTIONS", and the directions that see a convex body form a convex cone whose
// faces are the supporting planes through the camera position. A supporting plane with normal n
// is the constraint dot(d, n) >= 0, and the ray direction is affine in the pixel --
// d proportional to F + uvx*halfW*U + uvy*halfH*V -- so each supporting plane is exactly a
// half-plane in pixel coordinates, finite and well defined whether or not the body crosses the
// camera plane. This is the same cone Tungsten's near-plane clip approximates; taking it
// directly is simpler than clipping and it is exact.
//
// The supporting planes are found by brute force over point PAIRS: the plane through the camera
// and two points of the set supports the hull iff every other point is on one side of it. That
// is O(n^3) at n <= 16 -- a few thousand operations, once per record per pass -- and it is only
// run for the records the projection path refused, so the common case never pays for it.
static bool ppRastAddCone(const PPProj& P, const glm::vec3* pts, int n, PPRastBound& B) {
    if (n < 2 || n > 32) return false;
    glm::dvec3 e[32];
    double scale = 0.0;
    for (int i = 0; i < n; ++i) {
        e[i] = glm::dvec3(pts[i] - P.pos);
        scale = std::max(scale, glm::length(e[i]));
    }
    if (scale <= 0.0) return false;
    const int ne0 = B.ne;
    for (int i = 0; i < n && B.ne < kPPRastMaxEdges; ++i) {
        for (int j = i + 1; j < n && B.ne < kPPRastMaxEdges; ++j) {
            glm::dvec3 nn = glm::cross(e[i], e[j]);
            const double nl = glm::length(nn);
            if (nl < 1e-12 * scale * scale) continue;      // collinear with the camera: no plane
            // Tolerance is RELATIVE and tiny: points i and j sit on the plane to rounding, and a
            // point genuinely on the wrong side is off by far more than this. Accepting a plane
            // that is not quite supporting would make the bound non-conservative, so the test is
            // done in double and the tolerance is 1e-9 of the plane's own scale.
            const double tol = 1e-9 * nl * scale;
            bool pos = true, neg = true;
            for (int k = 0; k < n; ++k) {
                const double v = glm::dot(e[k], nn);
                if (v < -tol) pos = false;
                if (v >  tol) neg = false;
                if (!pos && !neg) break;
            }
            if (!pos && !neg) continue;
            if (neg) nn = -nn;
            nn /= nl;
            // dot(d, n) >= 0 with d = F + uvx*halfW*U + uvy*halfH*V, then uvx = 2*px/w - 1 and
            // uvy = 1 - 2*py/h.
            const double Au = (double)P.halfW * glm::dot(glm::dvec3(P.U), nn);
            const double Bu = (double)P.halfH * glm::dot(glm::dvec3(P.V), nn);
            const double Cu = glm::dot(glm::dvec3(P.F), nn);
            double ea = Au * 2.0 / (double)P.w;
            double eb = -Bu * 2.0 / (double)P.h;
            double ec = Cu - Au + Bu;
            const double len = std::sqrt(ea * ea + eb * eb);
            if (len < 1e-18) {                             // the half-plane is a constant
                if (ec < 0.0) { B.x1 = B.x0 - 1; return true; }   // nothing can see it
                continue;
            }
            ea /= len; eb /= len; ec = ec / len + (double)kPPRastGuard;   // normalised => px
            B.ea[B.ne] = (float)ea; B.eb[B.ne] = (float)eb; B.ec[B.ne] = (float)ec;
            ++B.ne;
        }
    }
    return B.ne > ne0;
}

// The scanline: the exact x-range of the convex polygon on row y. No per-pixel edge test.
static inline bool ppRastRow(const PPRastBound& B, int y, int& xa, int& xb) {
    if (y < B.y0 || y > B.y1 || B.x1 < B.x0) return false;
    if (B.ne == 0) { xa = B.x0; xb = B.x1; return true; }
    const float py = (float)y + 0.5f;
    float lo = (float)B.x0 + 0.5f, hi = (float)B.x1 + 0.5f;
    for (int e = 0; e < B.ne; ++e) {
        const float A = B.ea[e], C = B.eb[e] * py + B.ec[e];
        if      (A >  1e-12f) lo = std::max(lo, -C / A);
        else if (A < -1e-12f) hi = std::min(hi, -C / A);
        else if (C < 0.0f)    return false;
    }
    xa = std::max(B.x0, (int)std::ceil (lo - 0.5f));
    xb = std::min(B.x1, (int)std::floor(hi - 0.5f));
    return xa <= xb;
}

// ---- Which pixels can each record possibly touch? --------------------------------------------
// Per family, a point set whose convex hull contains every scatter point the crossing can
// return. Unbounded families are clipped to the MEDIUM first -- every one of these estimators
// discards a crossing whose scatter point is not inside a participating volume, so the union
// of the volume entities is a legitimate world-space clip and it is what makes a sweep that
// runs to infinity boundable at all.
static bool ppMediumAABB(const Scene& scene, glm::vec3& lo, glm::vec3& hi) {
    bool any = false;
    for (const Entity& e : scene.entities) {
        if (!e.visible) continue;
        if (e.volSigmaS + e.volSigmaA <= 1e-6f) continue;   // ppFindMediumAt skips these
        glm::vec3 l, g;
        if (e.primType == PrimType::VolumeBox) { l = e.primCenter - e.primHalfExt; g = e.primCenter + e.primHalfExt; }
        else if (e.primType == PrimType::VolumeSphere) { l = e.primCenter - glm::vec3(e.primRadius); g = e.primCenter + glm::vec3(e.primRadius); }
        else continue;
        if (!any) { lo = l; hi = g; any = true; } else { lo = glm::min(lo, l); hi = glm::max(hi, g); }
    }
    return any;
}
struct PPRastMedium { bool have = false; glm::vec3 lo{0.0f}, hi{0.0f}, ctr{0.0f}; float halfDiag = 0.0f; glm::vec3 c[8]; };
static PPRastMedium ppRastMedium(const Scene& scene) {
    PPRastMedium M;
    M.have = ppMediumAABB(scene, M.lo, M.hi);
    if (!M.have) return M;
    M.ctr = 0.5f * (M.lo + M.hi);
    M.halfDiag = 0.5f * glm::length(M.hi - M.lo);
    for (int i = 0; i < 8; ++i)
        M.c[i] = glm::vec3((i & 1) ? M.hi.x : M.lo.x, (i & 2) ? M.hi.y : M.lo.y, (i & 4) ? M.hi.z : M.lo.z);
    return M;
}
// true = this kind has a bound; false = the raster gather must refuse and let the tracing
// gather run, which is what keeps an unconverted family correct rather than dark.
static inline bool ppRastKindSupported(PPPrimKind k) {
    return k == PPPrimKind::Quad || k == PPPrimKind::Beam || k == PPPrimKind::Hourglass;
}
static void ppRastBoundOf(const PPProj& P, const PPIntegratorConfig& cfg, int pass, const PPRastMedium& M,
                          PPPrimKind kind, const PPPrim& pr, PPRastBound& B)
{
    ppRastFull(P, B);
    glm::vec3 pts[32]; int n = 0;
    switch (kind) {
        case PPPrimKind::Quad: {
            // Bounded already: the light quad translated by t*omega. The crossing point is a
            // Moller-Trumbore hit with u,v in [0,1], i.e. inside the parallelogram.
            const PPQuadPrim& q = pr.u.quad;
            pts[0] = q.corner; pts[1] = q.corner + q.eu;
            pts[2] = q.corner + q.eu + q.ev; pts[3] = q.corner + q.ev; n = 4;
            break;
        }
        case PPPrimKind::Beam: {
            // The crossing keeps a camera ray only when its closest-approach point xc lies
            // within `rad` of the segment, so xc is in the CAPSULE about [p, p+w*segLen] --
            // and a capsule is the convex hull of its two end balls. A ball of radius r sits
            // inside the cube of half-side r on any orthonormal basis, so 2 x 8 corners on the
            // CAMERA basis (which keeps the cube nearly screen-aligned, hence tight) bound it.
            const PPBeamPrim& b = pr.u.beam;
            const float r = ppBlurRadius(cfg, pass);   // the same tube the gather reads this pass
            const glm::vec3 e0 = b.p, e1 = b.p + b.w * b.segLen;
            for (int i = 0; i < 8; ++i) {
                const glm::vec3 d = ((i & 1) ? r : -r) * P.U + ((i & 2) ? r : -r) * P.V + ((i & 4) ? r : -r) * P.F;
                pts[n++] = e0 + d; pts[n++] = e1 + d;
            }
            break;
        }
        case PPPrimKind::Hourglass: {
            // Unbounded: the exit ray revolved about the axis runs to infinity, so this one is
            // the "clip to the medium, then project" case and nothing else works.
            // A crossing sits at x = yEff + (z2 + r*wz)*a + (rho2 + r*wr)*e_h with r > 0
            // (ppHgRecoverBranch's `return r > 0`) and |e_h| = 1, e_h perpendicular to a. With
            // wz^2 + wr^2 = 1 that gives |x - yEff|^2 = r^2 + 2r(z2*wz + rho2*wr) + z2^2+rho2^2,
            // and the crossing must be INSIDE the medium, so |x - yEff| <= |Mctr - yEff| + D.
            // Solving for r caps the sweep exactly; the capped sweep then fits in a box on the
            // (a,u,v) frame, and the medium's own screen hull is ANDed in on top.
            if (!M.have) { B.x1 = B.x0 - 1; return; }            // no medium => nothing gathers
            const PPHgPrim& H = pr.u.hg;
            const float z2 = H.ch.z2, rho2 = H.ch.rho2, wz = H.ch.wz, wr = H.ch.wr;
            const float A2 = wz * wz + wr * wr;                  // 1 by construction; not assumed
            if (A2 < 1e-12f) break;
            const float beta = z2 * wz + rho2 * wr;
            const float gam2 = z2 * z2 + rho2 * rho2;
            const float Dmax = glm::length(M.ctr - H.yEff) + M.halfDiag;
            const float disc = beta * beta - A2 * (gam2 - Dmax * Dmax);
            if (disc <= 0.0f) { B.x1 = B.x0 - 1; return; }       // the sheet never meets the fog
            const float L = (-beta + std::sqrt(disc)) / A2;
            if (L <= 0.0f) { B.x1 = B.x0 - 1; return; }
            const float Rm  = std::max(std::fabs(rho2), std::fabs(rho2 + L * wr));
            const float zA  = z2, zB = z2 + L * wz;
            const float zLo = std::min(zA, zB), zHi = std::max(zA, zB);
            const glm::vec3 uu = H.u, vv = glm::cross(H.a, H.u);
            for (int i = 0; i < 8; ++i) {
                const float z  = (i & 4) ? zHi : zLo;
                const float su = (i & 1) ? Rm : -Rm;
                const float sv = (i & 2) ? Rm : -Rm;
                pts[n++] = H.yEff + z * H.a + su * uu + sv * vv;
            }
            break;
        }
        default: break;
    }
    // Projection first (cheap, and it gives a pixel box as well as the edges); the direction
    // cone only for the records the projection refuses. Neither can ever shrink the bound
    // wrongly: each is an independent conservative constraint and they are ANDed.
    if (n > 0 && !ppRastAddHull(P, pts, n, B)) ppRastAddCone(P, pts, n, B);
    if (kind == PPPrimKind::Hourglass && M.have && !ppRastAddHull(P, M.c, 8, B))
        ppRastAddCone(P, M.c, 8, B);
}

// ---- The projection self-check ---------------------------------------------------------------
// The bound is only conservative if ppProjPt really is the inverse of the ray the gather
// generates, so this re-derives a handful of rays FROM cam.generateRay and projects points on
// each one back. Failure is not fatal: the raster pass refuses and the tracing gather runs.
// What it is guarding: a camera whose generateRay is not the pinhole formula (PP always passes
// a ZERO lens sample, under which ThinLensCamera happens to collapse onto the pinhole ray --
// CHECKED here rather than assumed, and a finite-aperture camera primitive would break it), and
// any change to the uv convention or the aspect axis.
static bool ppRasterProjCheck(const Camera& cam, const PPProj& P, int w, int h, float& worstPx) {
    worstPx = 0.0f;
    RNG rng(0x9E3779B9u);
    const float fx[5] = { 0.13f, 0.5f, 0.87f, 0.03f, 0.97f };
    const float fy[5] = { 0.11f, 0.5f, 0.91f, 0.96f, 0.07f };
    const float ts[3] = { 0.5f, 7.0f, 250.0f };
    for (int i = 0; i < 5; ++i) {
        const float px = fx[i] * (float)w, py = fy[i] * (float)h;
        const float uvx = px / (float)w * 2.0f - 1.0f;
        const float uvy = 1.0f - py / (float)h * 2.0f;
        Ray r = cam.generateRay(Vec2(uvx, uvy), Vec2(0.0f), rng);
        const glm::vec3 o = r.origin, d = glm::normalize(r.direction);
        for (int j = 0; j < 3; ++j) {
            glm::vec2 s;
            if (!ppProjPt(P, o + ts[j] * d, s)) return false;
            worstPx = std::max(worstPx, std::max(std::fabs(s.x - px), std::fabs(s.y - py)));
        }
    }
    return worstPx < 0.05f;
}

// ---- The gather ------------------------------------------------------------------------------
// Records outermost, pixels innermost -- the inversion. Returns false when it refuses (an
// unsupported kind, a camera whose rays it cannot invert), in which case the caller runs the
// tracing gather instead and the film is unchanged.
static bool ppRasterGather(const Scene& scene, const Camera& cam, int w, int h, int pass,
                           const PPIntegratorConfig& cfg, const PPIntegratorConfig& scfg,
                           float wgt, PPPrimKind kind, int N,
                           const std::vector<PPPrim>& batch, glm::vec3* film,
                           bool logNow)
{
    if (!ppRastKindSupported(kind)) {
        if (logNow) Log::tagged(LogLevel::Warn, "PP", "raster",
            "REFUSED: kind {} has no screen bound (bounded: 3 quad, 18 beam, 20 hourglass) -- tracing gather runs instead",
            (int)kind);
        return false;
    }
    const PPProj P = ppMakeProj(cam, w, h);
    float worst = 0.0f;
    if (!ppRasterProjCheck(cam, P, w, h, worst)) {
        if (logNow) Log::tagged(LogLevel::Warn, "PP", "raster",
            "REFUSED: the projection is not the inverse of cam.generateRay ({} camera, worst {} px) -- tracing gather runs instead",
            cam.typeName(), worst);
        return false;
    }

    const PPRastMedium M = ppRastMedium(scene);
    const size_t nRec = batch.size();
    std::vector<PPRastBound> bounds(nRec);
    double boundPx = 0.0;
    int nFull = 0, nEmpty = 0;
    for (size_t i = 0; i < nRec; ++i) {
        ppRastBoundOf(P, cfg, pass, M, kind, batch[i], bounds[i]);
        const PPRastBound& B = bounds[i];
        if (B.x1 < B.x0 || B.y1 < B.y0) { ++nEmpty; continue; }
        // The SCANLINE count, not the bounding box. A beam is a long thin DIAGONAL band: its box
        // is most of the frame and its polygon is a few hundred pixels, so quoting the box would
        // overstate the work by 3.3x on exactly the family with the largest expected win.
        if (logNow) {
            for (int yy = B.y0; yy <= B.y1; ++yy) {
                int xa = 0, xb = -1;
                if (ppRastRow(B, yy, xa, xb)) boundPx += (double)(xb - xa + 1);
            }
        }
        // What is left after ppRastAddCone: a record the CAMERA IS INSIDE, which has no supporting
        // plane and no finite projection, so the whole frame is the honest answer. Counting them
        // is how the looseness of the bound is read off a real render rather than guessed -- they
        // are the only records that cost more than their own footprint.
        if (B.ne == 0 && B.x0 == 0 && B.y0 == 0 && B.x1 == w - 1 && B.y1 == h - 1) ++nFull;
    }
    if (logNow) {
        const double full = (double)w * (double)h * (double)nRec;
        Log::tagged(LogLevel::Info, "PP", "raster",
            "RASTER gather: kind={} records={} bound-pixels={} of {} ({} pct of the tracing gather's crossings) unbounded={} empty={} proj-err={} px",
            (int)kind, (int)nRec, (long long)boundPx, (long long)full,
            (full > 0.0) ? (100.0 * boundPx / full) : 0.0, nFull, nEmpty, worst);
    }

    // ---- PER-ROW RECORD LISTS -------------------------------------------------------------
    // Without these, every row walks the whole record array to reject most of it on a y test.
    // That reads nRec * sizeof(PPRastBound) bytes PER ROW -- 79 kB at N = 256, times 140 rows,
    // 11 MB per pass -- and it was measured as the rasterizer's entire residual cost on
    // strategy 3: 0.205 ms/pass at N=64 against 0.421 at N=256, a slope of 1.1 us per record
    // per pass, which is 8 ns per (record, row) and nothing but the streaming. A counting sort
    // into h buckets removes it. Records are appended in INCREASING INDEX ORDER within each
    // row, which is what keeps the per-pixel accumulation order -- and therefore the film's
    // bits -- unchanged. It is a bin list in the one dimension the gather needs.
    std::vector<uint32_t> rowStart((size_t)h + 1, 0u);
    for (size_t i = 0; i < nRec; ++i) {
        const PPRastBound& B = bounds[i];
        if (B.x1 < B.x0) continue;
        for (int yy = std::max(0, B.y0); yy <= std::min(h - 1, B.y1); ++yy) ++rowStart[(size_t)yy + 1];
    }
    for (int yy = 0; yy < h; ++yy) rowStart[(size_t)yy + 1] += rowStart[(size_t)yy];
    std::vector<uint32_t> rowIdx(rowStart[(size_t)h]);
    {
        std::vector<uint32_t> cur(rowStart.begin(), rowStart.end() - 1);
        for (size_t i = 0; i < nRec; ++i) {
            const PPRastBound& B = bounds[i];
            if (B.x1 < B.x0) continue;
            for (int yy = std::max(0, B.y0); yy <= std::min(h - 1, B.y1); ++yy)
                rowIdx[cur[(size_t)yy]++] = (uint32_t)i;
        }
    }

    const bool viz = cfg.corrViz;
    ThreadPool::global().parallelFor(0, h, [&](int y) {
        // Per-ROW scratch, so two threads never share a pixel and there is no atomic anywhere.
        // camv/camOk are the SAME lazy PPCamRay the tracing gather keeps per pixel: at most one
        // intersectScene per pixel per pass, and a pixel no bound covers pays nothing at all.
        thread_local std::vector<PPCamRay>  camv;
        thread_local std::vector<uint8_t>   camOk;
        thread_local std::vector<glm::vec3> acc, accK;
        thread_local std::vector<int>       lastK;
        if ((int)acc.size() != w) {
            camv.assign((size_t)w, PPCamRay{}); camOk.assign((size_t)w, 0u);
            acc.assign((size_t)w, glm::vec3(0.0f)); accK.assign((size_t)w, glm::vec3(0.0f));
            lastK.assign((size_t)w, -1);
        }
        std::fill(camOk.begin(), camOk.end(), (uint8_t)0);
        std::fill(acc.begin(),   acc.end(),   glm::vec3(0.0f));
        std::fill(lastK.begin(), lastK.end(), -1);

        for (uint32_t ri = rowStart[(size_t)y]; ri < rowStart[(size_t)y + 1]; ++ri) {
            const uint32_t i = rowIdx[ri];
            int xa = 0, xb = -1;
            if (!ppRastRow(bounds[i], y, xa, xb)) continue;
            const PPPrim& pr = batch[i];
            const int k = (int)pr.seq;
            for (int x = xa; x <= xb; ++x) {
                if (!camOk[(size_t)x]) {
                    // The three-draw camera prefix, verbatim from the tracing gather: two
                    // jitter floats and generateRay's own ray.time draw, off ppHashSeed.
                    RNG rng(ppHashSeed(x, y, pass));
                    const float jx = rng.nextFloat();
                    const float jy = rng.nextFloat();
                    const float uvx = ((float)x + jx) / (float)w * 2.0f - 1.0f;
                    const float uvy = 1.0f - ((float)y + jy) / (float)h * 2.0f;
                    Ray camRay = cam.generateRay(Vec2(uvx, uvy), Vec2(0.0f), rng);
                    camv[(size_t)x] = ppMakeCamRay(camRay);
                    camOk[(size_t)x] = 1u;
                }
                PPCamRay& cr = camv[(size_t)x];
                glm::vec3 c(0.0f);
                switch (kind) {
                    case PPPrimKind::Quad:      c = ppCrossQuadPrim(scene, pr, cr); break;
                    case PPPrimKind::Hourglass: c = ppCrossHgPrim(scene, scfg, pr, cr, nullptr); break;
                    case PPPrimKind::Beam: {
                        glm::vec3 add;
                        if (!ppCrossBeamPrim(scene, scfg, pass, pr, cr, nullptr, add)) continue;
                        c = add; break;
                    }
                    default: break;
                }
                // An exact zero is the additive identity, so skipping it is not an
                // approximation -- it is the same sum. (NaN fails this test and is kept.)
                if (c.x == 0.0f && c.y == 0.0f && c.z == 0.0f) continue;
                if (lastK[(size_t)x] != k) {                       // the per-DRAW regrouping
                    if (lastK[(size_t)x] >= 0) {
                        glm::vec3 t = accK[(size_t)x];
                        if (viz) t *= ppCorrTint(lastK[(size_t)x], N);
                        acc[(size_t)x] += t;
                    }
                    accK[(size_t)x] = glm::vec3(0.0f);
                    lastK[(size_t)x] = k;
                }
                accK[(size_t)x] += c;
            }
        }
        for (int x = 0; x < w; ++x) {
            glm::vec3 color = acc[(size_t)x];
            if (lastK[(size_t)x] >= 0) {
                glm::vec3 t = accK[(size_t)x];
                if (viz) t *= ppCorrTint(lastK[(size_t)x], N);
                color += t;
            }
            if (!viz) color /= (float)N;
            const int idx = y * w + x;
            film[idx] = film[idx] * (1.0f - wgt) + color * wgt;
        }
    });
    return true;
}

// ---- The record gather ----------------------------------------------------------------------
// One build per primitive PER PASS instead of one per primitive per PIXEL. Same primitives,
// same order, same arithmetic -- only the place they are computed moves.
