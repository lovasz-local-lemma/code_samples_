// Excerpt: RadianceLab/src/render/PPIntegrator.cpp, lines 925-1303 of 11012. Photon-hourglass chain geometry (Snell/Fresnel through collinear glass balls), the double-precision ray-centred sheet solver whose discriminant is the co-area Jacobian, and branch recovery.
// Not a standalone translation unit; see the folder README for the surrounding types.

        if (nl < 1e-6f) continue;
        out[n].n = e.primNormal / nl;
        out[n].c = e.primCenter;
        out[n].half = glm::abs(e.primHalfExt);
        out[n].tint = glm::vec3(m.albedo);
        out[n].entity = i;
        ++n;
    }
    return n;
}

// ---------------------------------------------------------------------------
// HOURGLASS CHAIN + SWEPT SHEET, shared by strategies 20 and 22.
//
// One deterministic refraction chain at phi = 0 through a glass SPHERE (Cb, Rb, ior), then
// the surface of revolution of its exit ray about the line (apex, axis), intersected with
// the camera ray o + s*d. The chain is seeded by the incident direction om0 at phi = 0,
// which must lie in the meridian plane span{axis, u} (u = the phi = 0 tangent):
//   strategy 20: apex = the light point y (OUTSIDE the ball), axis = normalize(Cb - y),
//                om0 = cosPsi*axis + sinPsi*u; the entry point x1 is the NEAR ROOT of the
//                sphere along om0 (apexOnSphere = false).
//   strategy 22: apex = x1 itself, a sampled point ON the sphere; axis = n(x1), which passes
//                through Cb so the revolution is again a symmetry of the ball; om0 = the
//                sampled incident direction. No root solve: t1 = 0 (apexOnSphere = true).
// Everything downstream of x1 -- the two refractions, T1*T2, the chord, the revolved
// quadric -- is the SAME map, which is why the two strategies share one function.
// ---------------------------------------------------------------------------
// A chain link is one ball; kPPHgMaxLinks caps how many a single record can carry (the
// record is a flat union, so the cap is a size, not a policy). Link 0 is the ball of the
// one-ball chain and keeps its own fields below; links 1.. live in the small tables.
static constexpr int kPPHgMaxLinks = 4;
struct PPHgLink { glm::vec3 C; float R; float ior; };
struct PPHgChain {
    glm::vec3 x1, n1, om1, x2, n2, om2;   // the phi = 0 chain THROUGH LINK 0: entry, chord, exit
    float t1, ci1, T1, t2, ci2, T2;       // entry distance, incidence cosines, Fresnel transmittances
    float z2, rho2, wz, wr, mp;           // the CHAIN's exit point / direction in the (axis, u) frame
    float roots[2]; int nRoots;           // camera-ray parameters s where it pierces the sheet
    float Jr[2];                          // the sheet's co-area Jacobian at each root, from the
                                          // solver's own discriminant (see ppHgSolveSheet)
    // --- the chain (all one-ball defaults are exact identities, see ppHgBuildChainGeom) ---
    int   nLinks;                         // 1 = one ball
    float Trest;                          // Fresnel T-product of links 1.. (exactly 1.0f at nLinks==1)
    float gapLen;                         // total AIR path BETWEEN balls (exactly 0.0f at nLinks==1)
    float lkZ[kPPHgMaxLinks * 2];         // meridian z of link k's entry [2k] and exit [2k+1]
    float lkR[kPPHgMaxLinks * 2];         // meridian SIGNED rho of the same points
    // PER-SEGMENT prefixes: segment k is the air leg AFTER link k. lkTcum[k] is the Fresnel
    // product of links 1..k (1.0f at k = 0; equals Trest at the last), lkGapCum[k]
    // the fog path before segment k (0.0f at k = 0; equals gapLen at the last). Accumulated in
    // the same loop by the same statements, so the last entries are Trest / gapLen bit for bit.
    float lkTcum[kPPHgMaxLinks];
    float lkGapCum[kPPHgMaxLinks];
};
// The PRIMITIVE half: everything that does not depend on the camera ray. The cut between the
// two halves is `const glm::vec3 q0 = o - apex;`, the first camera-dependent statement, so
// everything above it lives here and everything below is the CROSSING (ppHgSolveSheet);
// ppHgBuildChain calls the two halves back to back.
//
// The ball sequence arrives as two defaulted trailing parameters, so `more == nullptr` is
// the one-ball chain. The extra links are appended AFTER `ch.om2` is stored -- link 0 keeps
// ch.x1..ch.T2 to itself, links 1.. fold their Fresnel product into `Trest` and their air
// gaps into `gapLen`, and only the sheet coefficients (z2, rho2, wz, wr, mp) are taken from
// the CHAIN's exit. At nMore == 0 that is `xe = x2, ome = om2`, so the one-ball chain
// evaluates the same floats bit for bit, without needing a tolerance to say so.
static bool ppHgBuildChainGeom(const glm::vec3& apex, const glm::vec3& axis, const glm::vec3& u,
                               const glm::vec3& om0, const glm::vec3& Cb, float Rb, float ior,
                               bool apexOnSphere, PPHgChain& ch,
                               const PPHgLink* more = nullptr, int nMore = 0)
{
    // --- the deterministic chain at phi = 0 (entirely inside span{axis,u}) ---
    if (!apexOnSphere) {
        // near root of the sphere along om0
        const glm::vec3 oc = apex - Cb;
        const float bq = glm::dot(oc, om0), cq = glm::dot(oc, oc) - Rb * Rb;
        const float disc = bq * bq - cq;
        if (disc <= 0.0f) return false;
        ch.t1 = -bq - std::sqrt(disc);
        if (ch.t1 <= 1e-4f) return false;
        ch.x1 = apex + ch.t1 * om0;
    } else {
        ch.t1 = 0.0f;
        ch.x1 = apex;
    }
    const glm::vec3 x1 = ch.x1, n1 = (x1 - Cb) / Rb;
    ch.n1 = n1;

    // entry refraction (air -> glass); Fresnel TRANSMITTANCE as a deterministic weight
    const float ci1 = -glm::dot(om0, n1);
    if (ci1 <= 1e-6f) return false;
    const glm::vec3 om1v = glm::refract(om0, n1, 1.0f / ior);
    if (glm::dot(om1v, om1v) < 1e-8f) return false;                          // (cannot TIR entering)
    const glm::vec3 om1 = glm::normalize(om1v);
    const float ct1 = std::fabs(glm::dot(om1, n1));
    const float rs1 = (1.0f * ci1 - ior * ct1) / (1.0f * ci1 + ior * ct1);
    const float rp1 = (1.0f * ct1 - ior * ci1) / (1.0f * ct1 + ior * ci1);
    const float T1 = 1.0f - 0.5f * (rs1 * rs1 + rp1 * rp1);

    // chord to the far side: from a point ON the sphere heading inward, t = -2*(oc.dir)
    const glm::vec3 oc1 = x1 - Cb;
    const float t2 = -2.0f * glm::dot(oc1, om1);
    if (t2 <= 1e-5f) return false;
    const glm::vec3 x2 = x1 + t2 * om1, n2 = (x2 - Cb) / Rb;

    // exit refraction (glass -> air). TIR is structurally impossible on a sphere: the internal
    // incidence angle at the exit equals the refracted angle at the entry, which is < theta_c.
    const float ci2 = glm::dot(om1, n2);
    if (ci2 <= 1e-6f) return false;
    const glm::vec3 om2v = glm::refract(om1, -n2, ior);
    if (glm::dot(om2v, om2v) < 1e-8f) return false;
    const glm::vec3 om2 = glm::normalize(om2v);
    const float ct2 = std::fabs(glm::dot(om2, n2));
    const float rs2 = (ior * ci2 - 1.0f * ct2) / (ior * ci2 + 1.0f * ct2);
    const float rp2 = (ior * ct2 - 1.0f * ci2) / (ior * ct2 + 1.0f * ci2);
    const float T2 = 1.0f - 0.5f * (rs2 * rs2 + rp2 * rp2);

    ch.ci1 = ci1; ch.T1 = T1; ch.om1 = om1;
    ch.t2 = t2; ch.x2 = x2; ch.n2 = n2;
    ch.ci2 = ci2; ch.T2 = T2; ch.om2 = om2;

    // --- the chain: refract through links 1.. in turn, all of them collinear with `axis` ---
    ch.nLinks = 1; ch.Trest = 1.0f; ch.gapLen = 0.0f;
    ch.lkTcum[0] = 1.0f; ch.lkGapCum[0] = 0.0f;
    {
        const glm::vec3 e0 = x1 - apex, x0 = x2 - apex;
        ch.lkZ[0] = glm::dot(e0, axis); ch.lkR[0] = glm::dot(e0, u);
        ch.lkZ[1] = glm::dot(x0, axis); ch.lkR[1] = glm::dot(x0, u);
    }
    glm::vec3 xe = x2, ome = om2;
    for (int k = 0; k < nMore && k + 1 < kPPHgMaxLinks; ++k) {
        const glm::vec3 Ck = more[k].C; const float Rk = more[k].R, iork = more[k].ior;
        const glm::vec3 ock = xe - Ck;
        const float bk = glm::dot(ock, ome), ck2 = glm::dot(ock, ock) - Rk * Rk;
        const float dk = bk * bk - ck2;
        if (dk <= 0.0f) return false;
        const float tk = -bk - std::sqrt(dk);
        if (tk <= 1e-4f) return false;                                  // behind, or grazing
        const glm::vec3 ek = xe + tk * ome, nk1 = (ek - Ck) / Rk;
        const float cA = -glm::dot(ome, nk1);
        if (cA <= 1e-6f) return false;
        const glm::vec3 iAv = glm::refract(ome, nk1, 1.0f / iork);
        if (glm::dot(iAv, iAv) < 1e-8f) return false;
        const glm::vec3 iA = glm::normalize(iAv);
        const float ctA = std::fabs(glm::dot(iA, nk1));
        const float rsA = (1.0f * cA - iork * ctA) / (1.0f * cA + iork * ctA);
        const float rpA = (1.0f * ctA - iork * cA) / (1.0f * ctA + iork * cA);
        const float TA  = 1.0f - 0.5f * (rsA * rsA + rpA * rpA);
        const float tch = -2.0f * glm::dot(ek - Ck, iA);
        if (tch <= 1e-5f) return false;
        const glm::vec3 xk = ek + tch * iA, nk2 = (xk - Ck) / Rk;
        const float cB = glm::dot(iA, nk2);
        if (cB <= 1e-6f) return false;
        const glm::vec3 oBv = glm::refract(iA, -nk2, iork);
        if (glm::dot(oBv, oBv) < 1e-8f) return false;
        const glm::vec3 oB = glm::normalize(oBv);
        const float ctB = std::fabs(glm::dot(oB, nk2));
        const float rsB = (iork * cB - 1.0f * ctB) / (iork * cB + 1.0f * ctB);
        const float rpB = (iork * ctB - 1.0f * cB) / (iork * ctB + 1.0f * cB);
        const float TB  = 1.0f - 0.5f * (rsB * rsB + rpB * rpB);
        const glm::vec3 qe = ek - apex, qx = xk - apex;
        ch.lkZ[2 * (k + 1)]     = glm::dot(qe, axis); ch.lkR[2 * (k + 1)]     = glm::dot(qe, u);
        ch.lkZ[2 * (k + 1) + 1] = glm::dot(qx, axis); ch.lkR[2 * (k + 1) + 1] = glm::dot(qx, u);
        ch.Trest *= TA * TB;
        ch.gapLen += tk;                                                // the AIR leg into this ball
        ch.lkTcum[k + 1] = ch.Trest; ch.lkGapCum[k + 1] = ch.gapLen;   // prefixes for segment k+1
        xe = xk; ome = oB; ++ch.nLinks;
    }

    // --- the swept sheet: revolve the exit ray {xe, ome} about the line (apex, axis) ---
    // At nLinks == 1, xe and ome hold the same floats as x2 and om2, so `q2` below is
    // `x2 - apex` bit for bit and nothing downstream can differ.
    const glm::vec3 q2 = xe - apex;
    const float z2 = glm::dot(q2, axis), rho2 = glm::dot(q2, u);
    const float wz = glm::dot(ome, axis), wr = glm::dot(ome, u);
    const float mp = wz * rho2 - wr * z2;
    ch.z2 = z2; ch.rho2 = rho2; ch.wz = wz; ch.wr = wr; ch.mp = mp;
    // implicit: wz^2 * (|x-apex|^2 - z^2) = (wr*z + mp)^2, with z = (x-apex).axis.
    // Multiplied through by wz^2 so an exit ray perpendicular to the axis degenerates to a
    // plane rather than dividing by zero.
    return true;
}

// The CROSSING half: pierce the already-built sheet with one camera ray. Reads `ch` and never
// writes it -- which is what lets a whole batch of frozen chains be shared, read-only, across
// every pixel of a correlated pass. Roots AND their Jacobians go to the caller's own storage.
//
// The sheet is the exit ray revolved about the axis: the double cone wz^2 rho^2 = (wr z + mp)^2
// with its apex ON THE AXIS at z = -mp/wr. Posing its crossing quadric in float32 from the
// camera origin, with dot(d,d) taken as 1, the two roots taken as (-B -+ sqrt)/(2A) unpaired
// and the estimator's J evaluated afterwards as the float32 cross product |((a x q) x om) . d|
// at the recovered root, is not accurate enough: the cancellation inside B^2 - 4AC invents
// and misses crossings near tangency and leaves legitimate crossings with a J orders of
// magnitude too small. Measured against an exact double reference on the renderer's own draws
// (1.49 M solver calls): 4.37 % of the film-reaching crossings carried |J/J_exact - 1| > 1e-3,
// 13 crossings were invented (exact discriminant negative, roots accepted) and 4 missed, and
// 123 of the 137 bad crossings sat above hgJMin, so the guard did not cover the solver's error
// band; with the guard off the float32 solver's energy was 125x the double solver's on the same
// draws, from legitimate crossings whose float32 J came out up to 33 000x too small.
//
// So: double precision, confined to this function exactly as ppConeSolve confines its own --
// the inputs stay float (a float axis defines a sheet rotated by 1e-7 rad and every consumer
// sees that SAME sheet) and the outputs are cast to float for the caller's x = o + s d -- and:
//   (1) the AXIS is normalised exactly, in double. `a` arrives as a float32 normalise
//       (|a|^2 - 1 ~ 1e-7), and assuming |a| = 1 in z = q.a and again in rho^2 = |q|^2 - z^2
//       bakes that error into the sheet. The sheet is the revolution of the stored meridian
//       numbers (wz, wr, mp) about a/|a| -- the only frame in which "revolve" is a rigid
//       motion -- so that is the frame the solver works in. One sqrt, three divisions.
//   (2) dd = dot(d,d) is CARRIED, not assumed 1: A = wz^2 dd - hh^2 W. d is glm::normalize of
//       a float32 direction, so dd = 1 + O(1e-7); assuming dd = 1 drops exactly the
//       wz^2 (dd - 1) s^2 term.
//   (3) the unknown is RAY-CENTRED: t = s - s0, s0 = -(q0.d)/dd the foot of the perpendicular
//       from the apex, so qp = q0 + s0 d is perpendicular to d and the linear coefficient loses
//       its cancelling q0.d term: B = -2 hh (W g + wr mp), W = wz^2 + wr^2, g = qp.a. The
//       constant term is the sheet function at the foot point, C = F(qp) = wz^2 |qp - g a|^2 -
//       (wr g + mp)^2, the perpendicular taken as a VECTOR so |qp|^2 - g^2 is never formed as
//       a difference.
//   (4) the JACOBIAN comes from the discriminant, analytically cancelled and factored:
//           N = wr (qp x d) + mp (a x d),   tau = (qp x d) . a,
//           J^2 = |N|^2 - W tau^2  ==  (B^2 - 4AC) / (4 wz^2)
//       -- the ray's moment about the apex against the cone's own two numbers. dF/ds is
//       2 wz J_signed on the sheet, so the estimator's cross product IS sqrt(disc)/(2|wz|) at
//       either root. This form has no wz in a denominator (an exit ray perpendicular to the
//       axis is a plane, not a 0/0), replaces the seven-term cancellation inside B^2 - 4AC by
//       ONE difference of two squares, and makes the root-existence decision (J^2 < 0: no
//       crossing) and the J the estimator divides by one quantity. Both roots carry the same
//       J -- a property of a quadratic (f'(s+-) = -+sqrt(disc)), not an approximation.
//   (5) citardauq pairing, the "-" root pushed first, so the small root is a division and
//       never a subtraction of near-equals.
// The ray-centred quadratic is the camera-origin quadratic shifted by s0 for a GENERAL dd; at
// dd = 1 it shifts back onto Aq s^2 + Bq s + Cq with discriminant 4 wz^2 J^2, so this form is
// the exact generalisation of the origin-centred one rather than an approximation of it.
// The hgJMin tangency guard and every downstream gate (s > 1e-4, branch recovery, inside-ball,
// occlusion, medium) are applied by the callers, not here.
static void ppHgSolveSheet(float wzf, float wrf, float mpf, const glm::vec3& apexF,
                           const glm::vec3& axisF, const glm::vec3& oF, const glm::vec3& dF,
                           float roots[2], float Js[2], int& nRoots)
{
    const double wz = wzf, wr = wrf, mp = mpf;
    const glm::dvec3 a0(axisF), o(oF), d(dF), apex(apexF);
    const glm::dvec3 a = a0 / std::sqrt(glm::dot(a0, a0));      // (1) the exactly-unit frame
    const glm::dvec3 q0 = o - apex;
    const double dd = glm::dot(d, d);                            // (2) 1 + O(1e-7), carried
    const double s0 = -glm::dot(q0, d) / dd;                     // (3) foot of the perpendicular
    const glm::dvec3 qp = q0 + s0 * d;                           //     qp . d == 0 to one rounding
    const double g = glm::dot(qp, a), hh = glm::dot(d, a);
    const glm::dvec3 qperp = qp - g * a;
    const double P = glm::dot(qperp, qperp);                     //     |qp|^2 - g^2, as a vector norm
    const double W = wz * wz + wr * wr;
    const double m = wr * g + mp;
    const double A = wz * wz * dd - hh * hh * W;
    const double B = -2.0 * hh * (W * g + wr * mp);
    const double C = wz * wz * P - m * m;                        //     = F(qp)
    const glm::dvec3 M = glm::cross(qp, d);                      // (4) the ray's moment about the apex
    const glm::dvec3 N = wr * M + mp * glm::cross(a, d);
    const double tau = glm::dot(M, a);
    const double J2 = glm::dot(N, N) - W * tau * tau;            //     = (B^2 - 4AC) / (4 wz^2)
    nRoots = 0;
    if (J2 < 0.0) return;                                        // no real crossing
    const double Jd = std::sqrt(J2);
    // The |A| < 1e-12 linear branch and the 1e-12 floor on B keep the float-era thresholds; they were
    // not retuned for double.
    if (std::fabs(A) < 1e-12) {                                  // ray parallel to a generator: one crossing
        if (std::fabs(B) > 1e-12) { roots[0] = (float)(s0 - C / B); Js[0] = (float)Jd; nRoots = 1; }
        return;
    }
    const double sq = 2.0 * std::fabs(wz) * Jd;                  // sqrt(B^2 - 4AC), by (4)
    const double qq = -0.5 * (B + ((B >= 0.0) ? sq : -sq));      // (5) citardauq
    double tA, tB;
    if (qq != 0.0) {
        const double r1 = qq / A, r2 = C / qq;
        if (B >= 0.0) { tA = r1; tB = r2; } else { tA = r2; tB = r1; }
    } else {
        tA = tB = -B / (2.0 * A);                                 // B == 0 and J2 == 0: a double root
    }
    roots[0] = (float)(s0 + tA); Js[0] = (float)Jd;              // the "-" root first
    roots[1] = (float)(s0 + tB); Js[1] = (float)Jd;
    nRoots = 2;
}
// The chain-struct entry point every existing caller uses: the same solver on the chain's own
// exit-sheet numbers, roots and Jacobians to the caller's storage.
static inline void ppHgSolveSheet(const PPHgChain& chIn, const glm::vec3& apex, const glm::vec3& axis,
                                  const glm::vec3& o, const glm::vec3& d, float roots[2], float Js[2],
                                  int& nRoots)
{
    ppHgSolveSheet(chIn.wz, chIn.wr, chIn.mp, apex, axis, o, d, roots, Js, nRoots);
}

// The combined entry point: geometry then crossing, in that order.
static bool ppHgBuildChain(const glm::vec3& apex, const glm::vec3& axis, const glm::vec3& u,
                           const glm::vec3& om0, const glm::vec3& Cb, float Rb, float ior,
                           bool apexOnSphere, const glm::vec3& o, const glm::vec3& d, PPHgChain& ch)
{
    if (!ppHgBuildChainGeom(apex, axis, u, om0, Cb, Rb, ior, apexOnSphere, ch)) return false;
    ppHgSolveSheet(ch, apex, axis, o, d, ch.roots, ch.Jr, ch.nRoots);
    return ch.nRoots > 0;
}

// Per sheet root: recover which BRANCH of the revolved quadric the crossing x belongs to and
// the exit-ray distance r there. Revolving a line about an axis squares the signed radial
// coordinate, so the quadric contains BOTH the real sheet AND its mirror image through the
// axis; accepting the wrong one gathers light along a ray the photon never travelled (it
// renders as a plausible-looking second faint cone). The consistency test is what rejects it.
// q = x - apex. Returns false on the axis (azimuth undefined) or when no branch is consistent.
// The segment form: the same test against an explicit exit point (zE, rhoE) and direction
// (wzS, wrS), so every fog segment of a chain can recover its own branch. The PPHgChain
// overload below forwards the chain's FINAL exit.
static bool ppHgRecoverBranchAt(float zE, float rhoE, float wzS, float wrS,
                                const glm::vec3& axis, const glm::vec3& q,
                              float& z, glm::vec3& per, float& rho, float& r, float& sgn)
{
    z = glm::dot(q, axis);
    per = q - z * axis;
    rho = glm::length(per);
    if (rho < 1e-6f) return false;                                   // on the axis: azimuth undefined
    r = -1.0f; sgn = 0.0f;
    for (int b2 = 0; b2 < 2; ++b2) {
        const float sigma = (b2 == 0) ? 1.0f : -1.0f;
        const float rr = (z - zE) * wzS + (sigma * rho - rhoE) * wrS;
        if (rr <= 1e-5f) continue;
        if (std::fabs(std::fabs(rhoE + rr * wrS) - rho) > 1e-3f * std::max(1.0f, rho)) continue;
        r = rr; sgn = sigma; break;
    }
    return r > 0.0f;
}
static bool ppHgRecoverBranch(const PPHgChain& ch, const glm::vec3& axis, const glm::vec3& q,
                              float& z, glm::vec3& per, float& rho, float& r, float& sgn)
{
    return ppHgRecoverBranchAt(ch.z2, ch.rho2, ch.wz, ch.wr, axis, q, z, per, rho, r, sgn);
}

// Strategy 22 per-branch DENSITY per unit camera-ray arclength, conditional on the ball and
// on the sampled x1 (the rho-cancelled closed form, regular on the axis):
//     p22(s | b, x1) = p_c(c) * sin(theta_in) * |m_hat . d| / |M22|
//     m_hat = wz*eh - wr*n,   M22 = 2 R c theta2' + r (2 theta2' - 1),   theta2' = c / (ior cos theta2)
// (= strategy 20's M_psi = R cos(theta1) gamma' + r chi' at L = R). Inputs: c = cos(theta_in),
// p_c its pdf, r the exit distance of the branch, (wz, wr) the exit direction in the (n, eh)
// frame, n = n(x1), eh = the crossing azimuth. M22 is NEVER floored: the fold M22 -> 0 is the
// caustic, an integrable infinity; J -> 0 (tangency) is the other regime, handled by the guard
// or by MIS. At ior -> 1: M22 = |x - x1| and this equals ppDirPrimPdf(9, x, x1, -n, d, ., true)
// * p_c(c), which serves as a consistency check. Not yet referenced by a renderer path; it is
// the MIS ingredient.
[[maybe_unused]] static float ppHgVertexPdf(float c, float p_c, float ior, float Rb, float r,
                                            float wz, float wr, const glm::vec3& n,
                                            const glm::vec3& eh, const glm::vec3& d)
{
    const float sinIn = std::sqrt(std::max(0.0f, 1.0f - c * c));
    const float cos2  = std::sqrt(std::max(0.0f, 1.0f - (1.0f - c * c) / (ior * ior)));
    if (cos2 <= 0.0f) return 0.0f;
    const float th2p = c / (ior * cos2);
    const float M22  = 2.0f * Rb * c * th2p + r * (2.0f * th2p - 1.0f);
    const glm::vec3 mh = wz * eh - wr * n;
    const float num = p_c * sinIn * std::fabs(glm::dot(mh, d));
    const float den = std::fabs(M22);
    if (den <= 0.0f) return (num > 0.0f) ? std::numeric_limits<float>::infinity() : 0.0f;
    return num / den;
}

