// My research branch of Tungsten (Benedikt Bitterli). See CONTRIBUTION.md and LICENSE.txt.
// Formatting/comment cleanup only; host-renderer dependencies are not bundled.

#include "PathTracer.hpp"

#include "bsdfs/TransparencyBsdf.hpp"

#include "integrators/TraceBase.hpp"

#include "core/primitives/TriangleMesh.hpp"

#include <Eigen/Dense>

namespace Tungsten
{

    // first 2 terms are always photon terms
    // last might be the cam term

    static const float sq3 = sqrtf(3.0f);
    static const float one_sq3 = float(1.0f) / sqrtf(3.0f);

    static const Vec3f v_one_sq3 = Vec3f(one_sq3);
    static const Vec3f v_sq3 = Vec3f(sq3);

    typedef StringableEnum<A_ter> A_ter_;

    DEFINE_STRINGABLE_ENUM(A_ter_, "A_ter",
                           ({{"dist", A_ter::A_dist}, {"theta", A_ter::A_theta}, {"phi", A_ter::A_phi}}));

    typedef StringableEnum<S_typ> S_typ_;

    DEFINE_STRINGABLE_ENUM(S_typ_, "S_typ",
                           ({{"plane", S_typ::S_plane},
                             {"cone", S_typ::S_cone},
                             {"cylinder", S_typ::S_cylinder},
                             {"sphere", S_typ::S_sphere},
                             {"disk", S_typ::S_disk},
                             {"torus", S_typ::S_torus},
                             {"hyperb", S_typ::S_hyperb},
                             {"unknown", S_typ::S_unknown}}));

    typedef StringableEnum<C_typ> C_typ_;

    DEFINE_STRINGABLE_ENUM(
        C_typ_, "C_typ",
        ({{"points", C_typ::C_point}, {"beams", C_typ::C_beam}, {"unknown", C_typ::C_unknown}}));

    bool izSphere(Vec2i &locs, arr<A_ter> &terms)
    {
        return (((terms[0] == A_phi) && (terms[1] == A_theta)) ||
                ((terms[1] == A_phi) && (terms[0] == A_theta)));
    }

    bool izPlane(Vec2i &locs, arr<A_ter> &terms)
    {
        return (((terms[0] == A_dist) && (terms[1] == A_dist)));
    }

    bool izCone(Vec2i &locs, arr<A_ter> &terms)
    {
        return ((locs[1] == locs[0]) && (((terms[0] == A_dist) && (terms[1] == A_phi)) ||
                                         ((terms[1] == A_dist) && (terms[0] == A_phi))));
    }

    // index goes backwards
    bool izCylinder(Vec2i &locs, arr<A_ter> &terms)
    {
        return ((((terms[0] == A_dist) && (terms[1] == A_phi) && (locs[0] == locs[1] + 1)) ||
                 ((terms[1] == A_dist) && (terms[0] == A_phi) && (locs[1] == locs[0] + 1))));
    }

    bool iz(Vec2i &locs, arr<A_ter> &terms, S_typ styp)
    {
        switch (styp)
        {
        case S_plane:
            return izPlane(locs, terms);
        case S_cone:
            return izCone(locs, terms);
        case S_cylinder:
            return izCylinder(locs, terms);
        case S_sphere:
            return izSphere(locs, terms);
        default:
            return S_unknown;
        }
    }

    S_typ surf_type(Vec2i &locs, arr<A_ter> &terms)
    {
        for (int ss = S_plane; ss <= S_sphere; ss++)
        {
            if (iz(locs, terms, S_typ(ss)))
                return S_typ(ss);
        }

        return S_unknown;
    }

    C_typ cam_type(int loc, A_ter term)
    {
        if ((loc == 2) && (term == A_dist))
            return C_beam;
        else if (loc != 2)
            return C_point;

        return C_unknown;
    }

    // locations:
    // -1: LS edge1
    // 0: LS edge2 (assoc'ed with LS sample)
    // 1: first bounce
    // 2: cam

    arr<A_ter> P_terms({A_dist, A_dist});
    Vec2i P_locs(-1, 0);

    A_ter CP_term = A_dist;
    int CP_loc = 2;

    C_typ_ ct = C_unknown;
    S_typ_ st = S_unknown;

    PathTracer::PathTracer(TraceableScene *scene, const PathTracerSettings &settings, uint32 threadId)
        : TraceBase(scene, settings, threadId), _settings(settings),
          _trackOutputValues(!scene->rendererSettings().renderOutputs().empty())
    {
        st = surf_type(P_locs, P_terms);
        ct = cam_type(CP_loc, CP_term);

        std::cout << "\n" << st.toString() << " -=> " << ct.toString();
    }

    static Vec2f decomp_plx(const Vec3f &orig, const Vec3f &d0, const Vec3f &d1, const Vec3f &pt_on_pl)
    {

        Vec3f norm = (d0.cross(d1)).normalized();

        float XX = (pt_on_pl - orig).dot(d0);
        Vec3f Yaxis = norm.cross(d0).normalized();
        float YY = (pt_on_pl - orig).dot(Yaxis);

        float costheta = d1.dot(d0);
        float sintheta = abs(d0.cross(d1).length());

        Vec2f coords;

        coords[0] = XX - YY * costheta / sintheta;
        coords[1] = YY / sintheta;

        return coords;
    }

    bool inter(const Ray &eyeray, const S_typ stype, const Vec3f &p0, const Vec3f &v0, const Vec3f &v1,
               bool long0, bool long1, arr<Vec3f> &out_hitPs,
               Vec2f &out_hitUV /*only the plane has UV, thus only 1 UV*/
               ,
               float tMin, float tMax, Vec3f &prevDir)
    {
        Vec3f d0 = v0.normalized();
        Vec3f d1 = v1.normalized();
        float l0 = v0.length();
        float l1 = v1.length();

        switch (stype)
        {
        case S_plane:
        {
            Vec3f norm = (d0.cross(d1)).normalized();

            float dist = (eyeray.pos() - p0).dot(norm) / (-eyeray.dir().dot(norm));

            if (dist <= 0)
                return false;

            if (!finite(dist))
                return false;

            if (dist <= tMin || dist >= tMax)
                return false;

            Vec3f hitP = eyeray.pos() + dist * eyeray.dir();

            float XX = (hitP - p0).dot(d0);
            Vec3f Yaxis = norm.cross(d0).normalized();
            float YY = (hitP - p0).dot(Yaxis);

            float costheta = d1.dot(d0);
            float sintheta = abs(d0.cross(d1).length());

            out_hitUV[0] = XX - YY * costheta / sintheta;
            out_hitUV[1] = YY / sintheta;

            if ((out_hitUV[0] < 0) || (out_hitUV[1] < 0))
                return false;

            if (!long0)
                if (out_hitUV[0] > l0)
                    return false;

            if (!long1)
                if (out_hitUV[1] > l1)
                    return false;

            out_hitPs.clear();
            out_hitPs.emplace_back(hitP);

            return true;
        }

        case S_Q:
        {
            Vec3f norm = (d0.cross(d1)).normalized();

            float dist = (eyeray.pos() - p0).dot(norm) / (-eyeray.dir().dot(norm));

            if (dist <= 0)
                return false;

            if (!finite(dist))
                return false;

            if (dist <= tMin || dist >= tMax)
                return false;

            Vec3f hitP = eyeray.pos() + dist * eyeray.dir();

            float XX = (hitP - p0).dot(d0);
            Vec3f Yaxis = norm.cross(d0).normalized();
            float YY = (hitP - p0).dot(Yaxis);

            float costheta = d1.dot(d0);
            float sintheta = abs(d0.cross(d1).length());

            out_hitUV[0] = XX - YY * costheta / sintheta;
            out_hitUV[1] = YY / sintheta;

            out_hitPs.clear();
            out_hitPs.emplace_back(hitP);

            return true;
        }

        case S_cone:
        {
            Tungsten::Mat4f toLocal;

            Vec3f XYZO[4];

            XYZO[2] = prevDir;
            XYZO[0] = prevDir.cross(v0.normalized()).normalized();
            XYZO[1] = XYZO[2].cross(XYZO[0]);
            XYZO[3] = p0;

            toLocal.setRight(XYZO[0]);
            toLocal.setUp(XYZO[1]);
            toLocal.setFwd(XYZO[2]);

            toLocal = toLocal.transpose();

            Mat4f tl_tr;
            tl_tr = tl_tr.translate(-XYZO[3]);

            toLocal = (toLocal * tl_tr);

            Vec3f raydir = toLocal.transformVector(eyeray.dir());
            Vec3f rayorig = toLocal * eyeray.pos();

            Ray rin(raydir, rayorig);

            Vec3f centerdir = prevDir;
            float r = (v0.cross(centerdir)).length();
            float ratio = abs(r / (v0.dot(centerdir)));

            Vec2i valid12(false, false);
            Vec2f t12(INFINITY, INFINITY);

            float rad = v0.length();

            float A = raydir[0] * raydir[0] + raydir[1] * raydir[1] - raydir[2] * raydir[2] * ratio * ratio;
            float B = raydir[0] * rayorig[0] * 2 + raydir[1] * rayorig[1] * 2 -
                      raydir[2] * rayorig[2] * 2 * ratio * ratio;
            float C =
                rayorig[0] * rayorig[0] + rayorig[1] * rayorig[1] - rayorig[2] * rayorig[2] * ratio * ratio;

            if (B * B >= 4 * A * C) // real solutions exist
            {
                if (A == 0)
                { // no solution
                    return false;
                }

                if (B * B - 4.0f * A * C < 0)
                    std::cout << "\nsqrt neg";

                t12[0] = (-B - std::sqrt(B * B - 4.0f * A * C)) / 2 / A;
                t12[1] = (-B + std::sqrt(B * B - 4.0f * A * C)) / 2 / A;

                float prj = (v0).dot(prevDir);

                prj = copysignf(INFINITY, prj);

                float hx[2];

                if (prj > 0)
                {
                    hx[0] = 0;
                    hx[1] = prj;
                }
                else
                {
                    hx[0] = prj;
                    hx[1] = 0;
                }

                for (int i = 0; i < 2; i++)
                {
                    valid12[i] = ((t12[i] > 0) && (finite(t12[i])));

                    // TODO: the tMin/tMax range clamp is not validated yet
                    valid12[i] = valid12[i] && ((t12[i] > tMin) && (t12[i] < tMax));

                    if (valid12[i])
                    {
                        Vec3f tracev = (eyeray.dir() * t12[i] + eyeray.pos()) - p0;

                        float pjx = tracev.dot(prevDir);

                        if (!((pjx >= hx[0]) && (pjx <= hx[1])))
                            valid12[i] = false;
                    }
                }
            }

            if ((valid12[1]) && (valid12[0]))
            {
                if (t12[0] > t12[1])
                {
                    float tt = t12[0];
                    t12[0] = t12[1];
                    t12[1] = tt;
                }

                out_hitPs.emplace_back(eyeray.pos() + eyeray.dir() * t12[0]);
                out_hitPs.emplace_back(eyeray.pos() + eyeray.dir() * t12[1]);

                return true;
            }
            else
            {
                // only one

                if ((!valid12[0]) && (!valid12[1]))
                {
                    return false;
                }
                else
                {
                    if (valid12[1])
                        t12[0] = t12[1];
                    {
                        out_hitPs.emplace_back(eyeray.pos() + eyeray.dir() * t12[0]);
                        return true;
                    }
                }
            }
        }

        case S_sphere:
        {
            Vec2i valid12(false, false);
            Vec2f t12(INFINITY, INFINITY);

            Vec3f e2 = v0;

            float rad = e2.length();

            Vec3f tocenter = p0 - eyeray.pos();

            float dist = tocenter.cross(eyeray.dir()).length();

            if (dist > rad)
                return false;

            float tocentercut = tocenter.dot(eyeray.dir());

            float ft = std::sqrt(tocenter.lengthSq() - tocentercut * tocentercut);
            float cutrad = std::sqrt(rad * rad - ft * ft);
            // not used
            Vec3f centercut = eyeray.pos() + tocentercut * eyeray.dir();

            t12[0] = tocentercut - cutrad;
            t12[1] = tocentercut + cutrad;

            if (t12[0] > t12[1])
            {
                float tt = t12[0];
                t12[0] = t12[1];
                t12[1] = tt;
            }

            for (int i = 0; i < 2; i++)
            {
                valid12[i] = ((t12[i] > 0) && (finite(t12[i])));

                // TODO: the tMin/tMax range clamp is not validated yet
                valid12[i] = valid12[i] && ((t12[i] > tMin) && (t12[i] < tMax));
            }

            if (valid12[1])
            {
                out_hitPs.emplace_back(eyeray.pos() + eyeray.dir() * t12[0]);
                out_hitPs.emplace_back(eyeray.pos() + eyeray.dir() * t12[1]);

                return true;
            }
            else
            {
                if (valid12[0])
                {
                    out_hitPs.emplace_back(eyeray.pos() + eyeray.dir() * t12[0]);
                    return true;
                }
                else
                {
                    return false;
                }
            }

            return false;
        }

        case S_cylinder:
        {
            Tungsten::Mat4f toLocal;

            Vec3f XYZO[4];

            Vec3f d0 = v0.normalized();
            Vec3f d1 = v1.normalized();

            XYZO[2] = d0;
            XYZO[0] = (d0.cross(d1)).normalized();
            XYZO[1] = XYZO[2].cross(XYZO[0]);
            XYZO[3] = p0;

            toLocal.setRight(XYZO[0]);
            toLocal.setUp(XYZO[1]);
            toLocal.setFwd(XYZO[2]);

            toLocal = toLocal.transpose();

            Mat4f tl_tr;
            tl_tr = tl_tr.translate(-XYZO[3]);

            toLocal = (toLocal * tl_tr);
        }

        default:
            return false;
        }

        return false;
    }

    PathTracerX::PathTracerX(TraceableScene *scene, const PathTracerSettings &settings, uint32 threadId)
        : TraceBase(scene, settings, threadId), _settings(settings),
          _trackOutputValues(!scene->rendererSettings().renderOutputs().empty())
    {
    }

    std::vector<Vec3f> arbline(Vec3f p0, Vec3f &v0, Vec3f &v1, Vec2f &dir, Vec3f &pos)
    {
        Vec3f d = pos - p0;
        Vec3f edge0n = v0.normalized();
        Vec3f edge1n = v1.normalized();

        std::vector<Vec3f> ret;
        float u = d.dot(edge0n);
        float v = d.dot(edge1n);
        float k = dir.y() / dir.x();
        float b = -u * k + v;
        float ty1 = b;
        float ty2 = k * v0.length() + b;
        float tx1 = -b / k;
        float tx2 = (v1.length() - b) / k;
        if (ty1 >= 0 && ty1 < v1.length())
            ret.push_back(p0 + edge1n * ty1);
        if (ty2 > 0 && ty2 <= v1.length())
            ret.push_back(p0 + v0 + edge1n * ty2);
        if (tx1 > 0 && tx1 <= v0.length())
            ret.push_back(p0 + edge0n * tx1);
        if (tx2 >= 0 && tx2 < v0.length())
            ret.push_back(p0 + edge0n * tx2 + v1);

        return ret;
    }

    // 2-step:
    // step 1: path gen + def eval
    // step 2: alt eval

    Vec3f TP(PathSampleGenerator &sampler, Ray &LS_M, Ray &M_C, PositionSample &_LS, DirectionSample &_LSD,
             Medium *med)
    {
        const PhaseFunction *pf = med->phaseFunction(M_C.pos());

        return med->transmittance(sampler, LS_M, false, false) *
               med->transmittance(sampler, M_C, false, false) * pf->eval(LS_M.dir(), M_C.dir()) * _LS.weight;
    }

    Vec3f PDF_t(PathSampleGenerator &sampler, Ray &LS_M, Medium *med)
    {
        // sigT at starting pos
        return med->transmittance(sampler, LS_M, false, false) * med->sigmaT(LS_M.pos());
    }

    const Vec3f endP(Ray &r)
    {
        return r.pos() + r.dir() * r.farT();
    }

    int intpoints(S_typ st, Ray &LS_M, Ray &M_C, Vec3f &lsnorm)
    {
        arr<Vec3f> hps;
        Vec2f uv;

        Ray eray = M_C;

        eray.setPos(endP(M_C));
        eray.setDir(-M_C.dir());
        eray.setFarT(10000000);

        bool hit = inter(eray, st, LS_M.pos(), LS_M.dir() * LS_M.farT(), LS_M.dir() * LS_M.farT(), false,
                         false, hps, uv, eray.nearT(), eray.farT(), lsnorm);

        if (hit)
            return hps.size();
        else
            return 0;
    }

    Vec3f PathTracer::reeval(int which, PathSampleGenerator &sampler, Ray &LS_M, Ray &M_C,
                             PositionSample &_LS)
    {

        float lightPdf;
        const Primitive *light = chooseLightAdjoint(sampler, lightPdf);
        const Medium *mediumx = light->extMedium().get();

        const PhaseFunction *phase = mediumx->phaseFunction(Vec3f(0.0f));
        Vec3f sigT = mediumx->sigmaT(Vec3f(0.0f));
        Vec3f sigS = mediumx->sigmaS(Vec3f(0.0f));
        Vec3f sigA = mediumx->sigmaA(Vec3f(0.0f));

        arr<Vec3f> corners = light->getCorner();
        Vec3f LS_orig = corners[0];
        Vec3f LS_u = corners[1] - corners[0];
        Vec3f LS_v = corners[2] - corners[1];
        Vec3f LS_udir = LS_u.normalized();
        Vec3f LS_vdir = LS_v.normalized();
        Vec3f LS_norm = -LS_udir.cross(LS_vdir).normalized();
        PositionSample psmp_LS;
        DirectionSample dsmp_LS;
        PhaseSample fsmp_mid;
        MediumSample vsmp_LS2m, vsmp_m2cam;
        Vec3f lsV, connectV, camV;
        float D = _scene->bounds().diagonal().length();

        Vec3f vmid = M_C.pos();
        psmp_LS.p = LS_M.pos();
        dsmp_LS.d = LS_M.dir();

        // LS is a new sample: its sampled data are not meaningful here,
        // so only the weight is used

        psmp_LS.weight = Vec3f(LS_u.cross(LS_v).length());
        dsmp_LS.d = LS_M.dir();

        switch (which)
        {
        case 0:
        {
            return light->evalPositionalEmission(psmp_LS)

                   * mediumx->transmittance(sampler, M_C, false, false) * phase->eval(LS_M.dir(), M_C.dir()) *
                   sigS / sigT / abs(LS_udir.cross(LS_vdir).dot(-M_C.dir()));
        }

        case 1:
        {
            return light->evalPositionalEmission(psmp_LS)

                   * LS_v.length()

                   * mediumx->transmittance(sampler, M_C, false, false) *
                   mediumx->transmittance(sampler, LS_M, false, false)

                   * phase->eval(LS_M.dir(), M_C.dir()) * sigS / sigT /
                   abs(LS_udir.cross(LS_M.dir()).dot(-M_C.dir()));
        }
        case 2:
        {
            return light->evalPositionalEmission(psmp_LS)

                   * LS_v.length()

                   * mediumx->transmittance(sampler, M_C, false, false) *
                   mediumx->transmittance(sampler, LS_M, false, false)

                   * phase->eval(LS_M.dir(), M_C.dir()) * sigS / sigT /
                   abs(LS_vdir.cross(LS_M.dir()).dot(-M_C.dir()));
        }
        case 3:
        {

            int hitpnum_v = intpoints(S_sphere, LS_M, M_C, LS_norm);

            return _LS.weight * mediumx->transmittance(sampler, M_C, false, false)

                   * light->evalDirectionalEmission(_LS, dsmp_LS) * phase->eval(LS_M.dir(), M_C.dir()) *
                   sigS / sigT / LS_M.farT() / LS_M.farT() / abs(LS_M.dir().dot(-M_C.dir())) * hitpnum_v;
        }
        case 4:
        {
            int hitpnum_v = intpoints(S_cone, LS_M, M_C, LS_norm);

            return _LS.weight * mediumx->transmittance(sampler, M_C, false, false) *
                   mediumx->transmittance(sampler, LS_M, false, false)

                   / M_PI / 2

                   * phase->eval(LS_M.dir(), M_C.dir()) * sigS / sigT / LS_M.farT() /
                   LS_M.dir().cross(LS_norm).length() /
                   abs(LS_M.dir().cross(LS_M.dir().cross(LS_norm)).normalized().dot(-M_C.dir())) * hitpnum_v;
        }
        case 5:
        {
            return _LS.weight * mediumx->transmittance(sampler, M_C, false, false) *
                   mediumx->transmittance(sampler, LS_M, false, false)

                   * light->evalDirectionalEmission(_LS, dsmp_LS) * M_PI *
                   phase->eval(LS_M.dir(), M_C.dir()) * sigS / LS_M.farT() *
                   (LS_M.dir().cross(LS_norm).length()) /
                   abs((LS_M.dir().cross(LS_norm)).normalized().dot(-M_C.dir()));
        }

        case 6:
        {
            return _LS.weight * mediumx->transmittance(sampler, LS_M, false, false)

                   * light->evalDirectionalEmission(_LS, dsmp_LS) * phase->eval(LS_M.dir(), M_C.dir()) *
                   sigS / sigT / LS_M.farT() / LS_M.farT();
        }

        case 7:
        {

            // some dims are not filled yet
            return light->evalDirectionalEmission(psmp_LS, dsmp_LS) * light->evalPositionalEmission(psmp_LS) *
                   mediumx->transmittance(sampler, LS_M, false, false) * phase->eval(LS_M.dir(), M_C.dir()) /
                   phase->pdf(LS_M.dir(), M_C.dir()) * sigS / sigT / abs(LS_norm.dot(dsmp_LS.d));
        }

        case 8:
        {
            psmp_LS.p = LS_M.pos();

            Vec3f psi1 = -M_C.dir();
            Vec3f omega1 = LS_M.dir();
            Vec3f clt = psi1.cross(omega1);

            float rj = 2 * sqrtf(powf(clt.dot(LS_udir), 2) + powf(clt.dot(LS_vdir), 2)) / M_PI;

            return light->evalPositionalEmission(psmp_LS) * LS_M.farT() *
                   mediumx->transmittance(sampler, M_C, false, false) *
                   mediumx->transmittance(sampler, LS_M, false, false)

                   * phase->eval(LS_M.dir(), M_C.dir()) * sigS / sigT / rj;
        }

        default:
            return Vec3f(0.0f);
        }

        return Vec3f(0.0f);
    }

    Vec3f PathTracer::routine(int which, Vec2u pixel, PathSampleGenerator &sampler, Ray &LS_M, Ray &M_C,
                              bool &hit_or_occluded)
    {

        Vec3f shiftD = Vec3f(1.0f, 0.0f, 0.0f);

        hit_or_occluded = true;
        // default: no contribution
        // TODO: Put diagnostic colors in JSON?
        const Vec3f nanDirColor = Vec3f(0.0f);
        const Vec3f nanEnvDirColor = Vec3f(0.0f);
        const Vec3f nanBsdfColor = Vec3f(0.0f);

        try
        {
            PositionSample point;
            if (!_scene->cam().samplePosition(sampler, point))
                return Vec3f(0.0f);
            DirectionSample direction;
            if (!_scene->cam().sampleDirection(sampler, point, pixel, direction))
                return Vec3f(0.0f);

            Vec3f throughput = point.weight * direction.weight;
            Ray ray(point.p, direction.d);
            ray.setPrimaryRay(true);

            MediumSample mediumSample;
            SurfaceScatterEvent surfaceEvent;
            IntersectionTemporary data;
            Medium::MediumState state;
            state.reset();
            IntersectionInfo info;
            Vec3f emission(0.0f);
            const Medium *medium = _scene->cam().medium().get();

            bool recordedOutputValues = false;
            float hitDistance = 0.0f;

            int mediumBounces = 0;
            int bounce = 0;
            bool didHit = _scene->intersect(ray, data, info);
            bool wasSpecular = true;
            while ((didHit || medium) && bounce < _settings.maxBounces)
            {
                bool hitSurface = true;
                if (medium)
                {
                    mediumSample.continuedWeight = throughput;
                    if (!medium->sampleDistance(sampler, ray, state, mediumSample))
                        return emission;
                    throughput *= mediumSample.weight;
                    hitSurface = mediumSample.exited;
                    if (hitSurface && !didHit)
                        break;
                }

                if (hitSurface)
                {
                    hitDistance += ray.farT();

                    surfaceEvent = makeLocalScatterEvent(data, info, ray, &sampler);
                    Vec3f transmittance(-1.0f);

                    const Bsdf &bsdf = *info.bsdf;

                    Vec3f transparency = bsdf.eval(surfaceEvent.makeForwardEvent(), false);
                    float transparencyScalar = transparency.avg();

                    Vec3f wo;

                    wo = ray.dir();

                    surfaceEvent.pdf = transparencyScalar;
                    surfaceEvent.weight = transparency / transparencyScalar;
                    surfaceEvent.sampledLobe = BsdfLobes::ForwardLobe;
                    throughput *= surfaceEvent.weight;

                    bool geometricBackside = (wo.dot(info.Ng) < 0.0f);
                    medium = info.primitive->selectMedium(medium, geometricBackside);
                    state.reset();

                    ray = ray.scatter(ray.hitpoint(), wo, info.epsilon);

                    bool terminate = (mediumBounces > 0);

                    if (!info.bsdf->lobes().isPureDirac())
                        if (mediumBounces == 0 && !_settings.includeSurfaces)
                            return emission;

                    if (terminate)
                        return emission;
                }
                else
                {
                    mediumBounces++;

                    wasSpecular = true;

                    float lightPdf;
                    const Primitive *light = chooseLightAdjoint(sampler, lightPdf);
                    const Medium *mediumx = light->extMedium().get();

                    const PhaseFunction *phase = mediumx->phaseFunction(Vec3f(0.0f));
                    Vec3f sigT = mediumx->sigmaT(Vec3f(0.0f));
                    Vec3f sigS = mediumx->sigmaS(Vec3f(0.0f));
                    Vec3f sigA = mediumx->sigmaA(Vec3f(0.0f));

                    arr<Vec3f> corners = light->getCorner();
                    Vec3f LS_orig = corners[0];
                    Vec3f LS_u = corners[1] - corners[0];
                    Vec3f LS_v = corners[2] - corners[1];
                    Vec3f LS_udir = LS_u.normalized();
                    Vec3f LS_vdir = LS_v.normalized();
                    Vec3f LS_norm = -LS_udir.cross(LS_vdir).normalized();
                    PositionSample psmp_LS;
                    DirectionSample dsmp_LS;
                    PhaseSample fsmp_mid;
                    MediumSample vsmp_LS2m, vsmp_m2cam;
                    Vec3f lsV, connectV, camV;
                    float D = _scene->bounds().diagonal().length();

                    //	LS pdf == LS emission
                    //  LS weight no need to / LS pdf

                    switch (which)
                    {
                        // UV
                    case 0:
                    {
                        emission = Vec3f(0.0f);

                        PositionSample _LS;
                        if (!light->samplePosition(sampler, _LS))
                            return emission;

                        DirectionSample _LSD;
                        if (!light->sampleDirection(sampler, _LS, _LSD))
                            return emission;

                        Ray LSray(_LS.p, _LSD.d);

                        MediumSample _Cdist;
                        Medium::MediumState state_C;
                        state_C.reset();

                        if (!mediumx->sampleDistance(sampler, LSray, state_C, _Cdist))
                            return emission;

                        LSray.setFarT(_Cdist.continuedT);

                        arr<Vec3f> hps;
                        Vec2f uv;

                        bool hit = inter(ray, S_plane, LS_orig + LSray.dir() * LSray.farT(), LS_u, LS_v,
                                         false, false, hps, uv, ray.nearT(), ray.farT(), LS_norm);

                        if (!hit)
                            return Vec3f(0.0f);
                        else
                        {
                            Vec3f vmid = hps[0];

                            ray_from_to(vmid - LSray.dir() * (LSray.farT() - LSray.nearT()), vmid, LS_M);
                            ray_from_to(vmid, ray.pos(), M_C);

                            LS_M.setPrimaryRay(false);
                            M_C.setPrimaryRay(false);

                            if (_scene->occluded(LS_M) || _scene->occluded(M_C))
                            {
                                emission = Vec3f(0.0f);
                                return emission;
                            }

                            psmp_LS.p = vmid - LSray.dir() * (LSray.farT());

                            hit_or_occluded = false;

                            emission = light->evalPositionalEmission(psmp_LS)

                                       * medium->transmittance(sampler, M_C, false, false)

                                       * phase->eval(LS_M.dir(), M_C.dir()) * sigS / sigT /
                                       abs(LS_udir.cross(LS_vdir).dot(ray.dir()));

                            return emission;
                        }
                    }

                    case 1:
                        // UT
                        {
                            emission = Vec3f(0.0f);

                            PositionSample _LS;
                            if (!light->samplePosition(sampler, _LS))
                                return emission;

                            float V = _LS.uv[1] * LS_v.length();

                            DirectionSample _LSD;
                            if (!light->sampleDirection(sampler, _LS, _LSD))
                                return emission;

                            Ray LSray(_LS.p, _LSD.d);

                            MediumSample _Cdist;

                            arr<Vec3f> hps, hps2;
                            Vec2f uv;

                            bool hit = inter(ray, S_plane, LS_orig + LS_vdir * V, LS_u, _LSD.d, false, true,
                                             hps, uv, ray.nearT(), ray.farT(), LS_norm);

                            if (!hit)
                            {
                                return Vec3f(0.0f, 0.0f, 0.0f);
                            }
                            else
                            {
                                Ray sray(hps[0], -_LSD.d);

                                bool hit2 = inter(sray, S_plane, LS_orig, LS_u, LS_v, false, false, hps2, uv,
                                                  ray.nearT(), D, LS_norm);

                                if (!hit2)
                                    return Vec3f(0.0f, 1.0f, 0.0f);

                                Vec3f vmid = hps[0];
                                Vec3f vLS = hps2[0];

                                ray_from_to(vLS, vmid, LS_M);
                                ray_from_to(vmid, ray.pos(), M_C);

                                LS_M.setPrimaryRay(false);
                                M_C.setPrimaryRay(false);

                                if (_scene->occluded(LS_M) || _scene->occluded(M_C))
                                {
                                    emission = Vec3f(0.0f);
                                    return emission;
                                }

                                psmp_LS.p = vLS;

                                hit_or_occluded = false;

                                emission = light->evalPositionalEmission(psmp_LS)

                                           * LS_v.length()

                                           * medium->transmittance(sampler, M_C, false, false) *
                                           medium->transmittance(sampler, LS_M, false, false)

                                           * phase->eval(LS_M.dir(), M_C.dir()) * sigS / sigT /
                                           abs(LS_udir.cross(LSray.dir()).dot(ray.dir()));

                                return emission;
                            }
                        }

                    case 2:
                        // VT
                        {
                            emission = Vec3f(0.0f);

                            PositionSample _LS;
                            if (!light->samplePosition(sampler, _LS))
                                return emission;

                            float U = _LS.uv[0] * LS_u.length();

                            DirectionSample _LSD;
                            if (!light->sampleDirection(sampler, _LS, _LSD))
                                return emission;

                            Ray LSray(_LS.p, _LSD.d);

                            MediumSample _Cdist;

                            arr<Vec3f> hps, hps2;
                            Vec2f uv;

                            bool hit = inter(ray, S_plane, LS_orig + LS_udir * U, LS_v, _LSD.d, false, true,
                                             hps, uv, ray.nearT(), ray.farT(), LS_norm);

                            if (!hit)
                            {
                                return Vec3f(0.0f, 0.0f, 0.0f);
                            }
                            else
                            {
                                Ray sray(hps[0], -_LSD.d);

                                bool hit2 = inter(sray, S_plane, LS_orig, LS_u, LS_v, false, false, hps2, uv,
                                                  ray.nearT(), D, LS_norm);

                                if (!hit2)
                                    return Vec3f(0.0f, 1.0f, 0.0f);

                                Vec3f vmid = hps[0];
                                Vec3f vLS = hps2[0];

                                ray_from_to(vLS, vmid, LS_M);
                                ray_from_to(vmid, ray.pos(), M_C);

                                LS_M.setPrimaryRay(false);
                                M_C.setPrimaryRay(false);

                                if (_scene->occluded(LS_M) || _scene->occluded(M_C))
                                {
                                    emission = Vec3f(0.0f);
                                    return emission;
                                }

                                hit_or_occluded = false;

                                psmp_LS.p = vLS;

                                emission = light->evalPositionalEmission(psmp_LS)

                                           * LS_v.length()

                                           * medium->transmittance(sampler, M_C, false, false) *
                                           medium->transmittance(sampler, LS_M, false, false)

                                           * phase->eval(LS_M.dir(), M_C.dir()) * sigS / sigT /
                                           abs(LS_vdir.cross(LSray.dir()).dot(ray.dir()));

                                return emission;
                            }
                        }

                    case 3:
                        // SPH
                        {
                            emission = Vec3f(0.0f);

                            PositionSample _LS;
                            if (!light->samplePosition(sampler, _LS))
                                return emission;

                            DirectionSample _LSD;
                            if (!light->sampleDirection(sampler, _LS, _LSD))
                                return emission;

                            Ray LSray(_LS.p, _LSD.d);

                            MediumSample _Cdist;
                            Medium::MediumState state_C;
                            state_C.reset();

                            if (!mediumx->sampleDistance(sampler, LSray, state_C, _Cdist))
                                return emission;

                            if (_Cdist.exited)
                                return emission;

                            LSray.setFarT(_Cdist.continuedT);

                            arr<Vec3f> hps;
                            Vec2f uv;

                            bool hit = inter(ray, S_sphere, _LS.p, LSray.dir() * LSray.farT(),
                                             LSray.dir() * LSray.farT(), false, false, hps, uv, ray.nearT(),
                                             ray.farT(), LS_norm);

                            if (hit)
                            {
                                Vec3f res(0.0f);

                                Vec3f vmid;

                                if (hps.size() > 1)
                                {
                                    float sss = sampler.next1D();
                                    if (sss > 0.5)
                                        vmid = hps[0];
                                    else
                                        vmid = hps[1];
                                }
                                else
                                {
                                    vmid = hps[0];
                                }

                                {

                                    ray_from_to(_LS.p, vmid, LS_M);
                                    ray_from_to(vmid, ray.pos(), M_C);

                                    LS_M.setPrimaryRay(false);
                                    M_C.setPrimaryRay(false);

                                    dsmp_LS.d = LS_M.dir();

                                    if (_scene->occluded(LS_M) || _scene->occluded(M_C))
                                    {
                                        return Vec3f(0.0f);
                                    }

                                    hit_or_occluded = false;

                                    res += _LS.weight * medium->transmittance(sampler, M_C, false, false)

                                           * light->evalDirectionalEmission(_LS, dsmp_LS) *
                                           phase->eval(LS_M.dir(), M_C.dir()) * sigS / sigT / LS_M.farT() /
                                           LS_M.farT() / abs(LS_M.dir().dot(ray.dir()));
                                }

                                return res * hps.size();
                            }
                            else
                                return Vec3f(0.0f, 0.0f, 0.0f);
                        }
                    case 4:
                        // cone
                        {
                            emission = Vec3f(0.0f);

                            PositionSample _LS;
                            if (!light->samplePosition(sampler, _LS))
                                return emission;

                            DirectionSample _LSD;
                            if (!light->sampleDirection(sampler, _LS, _LSD))
                                return emission;

                            Ray LSray(_LS.p, _LSD.d);

                            MediumSample _Cdist;
                            Medium::MediumState state_C;
                            state_C.reset();

                            if (!mediumx->sampleDistance(sampler, LSray, state_C, _Cdist))
                                return emission;

                            if (_Cdist.exited)
                                return emission;

                            LSray.setFarT(_Cdist.continuedT);

                            arr<Vec3f> hps;
                            Vec2f uv;

                            bool hit = inter(ray, S_cone, _LS.p, LSray.dir() * LSray.farT(),
                                             LSray.dir() * LSray.farT(), false, false, hps, uv, ray.nearT(),
                                             ray.farT(), LS_norm);

                            if (hit)
                            {
                                Vec3f res(0.0f);

                                Vec3f vmid;

                                if (hps.size() > 1)
                                {
                                    float sss = sampler.next1D();
                                    if (sss > 0.5)
                                        vmid = hps[0];
                                    else
                                        vmid = hps[1];
                                }
                                else
                                {
                                    vmid = hps[0];
                                }

                                {

                                    ray_from_to(_LS.p, vmid, LS_M);
                                    ray_from_to(vmid, ray.pos(), M_C);

                                    LS_M.setPrimaryRay(false);
                                    M_C.setPrimaryRay(false);

                                    dsmp_LS.d = LS_M.dir();

                                    if (_scene->occluded(LS_M) || _scene->occluded(M_C))
                                    {
                                        return Vec3f(0.0f);
                                    }

                                    hit_or_occluded = false;

                                    res = _LS.weight * medium->transmittance(sampler, M_C, false, false) *
                                          medium->transmittance(sampler, LS_M, false, false)

                                          / M_PI / 2

                                          * phase->eval(LS_M.dir(), M_C.dir()) * sigS / sigT / LS_M.farT() /
                                          LS_M.dir().cross(LS_norm).length() /
                                          abs(LS_M.dir()
                                                  .cross(LS_M.dir().cross(LS_norm))
                                                  .normalized()
                                                  .dot(ray.dir()));
                                }

                                return res * hps.size();
                            }
                            else
                                return Vec3f(0.0f, 0.0f, 0.0f);
                        }

                    case 5:
                        // theta-t disk
                        {
                            emission = Vec3f(0.0f);

                            PositionSample _LS;
                            if (!light->samplePosition(sampler, _LS))
                                return emission;

                            DirectionSample _LSD;
                            if (!light->sampleDirection(sampler, _LS, _LSD))
                                return emission;

                            Ray LSray(_LS.p, _LSD.d);

                            Vec3f pj = _LSD.d - LS_norm * (_LSD.d.dot(LS_norm));
                            Vec3f dir_on_LS = pj.normalized();

                            arr<Vec3f> hps;
                            Vec2f uv;

                            bool hit = inter(ray, S_Q, _LS.p, LSray.dir(), LS_norm, true, true, hps, uv,
                                             ray.nearT(), ray.farT(), LS_norm);

                            if (hit)
                            {
                                Vec3f res(0.0f);

                                Vec3f vmid = hps[0];

                                ray_from_to(_LS.p, vmid, LS_M);
                                ray_from_to(vmid, ray.pos(), M_C);

                                LS_M.setPrimaryRay(false);
                                M_C.setPrimaryRay(false);

                                dsmp_LS.d = LS_M.dir();

                                if (_scene->occluded(LS_M) || _scene->occluded(M_C))
                                {
                                    return Vec3f(0.0f, 0.0f, 0.0f);
                                }

                                hit_or_occluded = false;

                                res = _LS.weight * medium->transmittance(sampler, M_C, false, false) *
                                      medium->transmittance(sampler, LS_M, false, false)

                                      * light->evalDirectionalEmission(_LS, dsmp_LS) * M_PI *
                                      phase->eval(LS_M.dir(), M_C.dir()) * sigS / LS_M.farT() *
                                      (LS_M.dir().cross(LS_norm).length()) /
                                      abs((LS_M.dir().cross(LS_norm)).normalized().dot(ray.dir()));

                                return res;
                            }
                            else
                                return Vec3f(0.0f, 0.0f, 0.0f);
                        }

                    case 6:

                        // NEE
                        {
                            emission = Vec3f(0.0f);

                            PositionSample _LS;
                            if (!light->samplePosition(sampler, _LS))
                                return emission;

                            MediumSample _Cdist;
                            Medium::MediumState state_C;
                            state_C.reset();

                            if (!mediumx->sampleDistance(sampler, ray, state_C, _Cdist))
                                return emission;

                            if (_Cdist.exited)
                                return emission;

                            Vec3f vmid = _Cdist.p2;

                            ray_from_to(_LS.p, vmid, LS_M);
                            ray_from_to(vmid, ray.pos(), M_C);

                            LS_M.setPrimaryRay(false);
                            M_C.setPrimaryRay(false);

                            if (_scene->occluded(LS_M) || _scene->occluded(M_C))
                            {
                                emission = Vec3f(0.0f);
                                return emission;
                            }
                            else
                            {

                                hit_or_occluded = false;

                                dsmp_LS.weight = Vec3f(1.0f);
                                dsmp_LS.pdf = 1.0f;
                                dsmp_LS.d = LS_M.dir();

                                Vec3f trp = _LS.weight * medium->transmittance(sampler, LS_M, false, false)

                                            * light->evalDirectionalEmission(_LS, dsmp_LS) *
                                            phase->eval(LS_M.dir(), M_C.dir()) * sigS / sigT / LS_M.farT() /
                                            LS_M.farT();

                                emission = trp;
                            }

                            return emission;
                        }

                    case 7:
                        // NAIVE

                        {
                            emission = Vec3f(0.0f);

                            MediumSample _Cdist;
                            Medium::MediumState state_C;
                            state_C.reset();

                            if (!mediumx->sampleDistance(sampler, ray, state_C, _Cdist))
                                return emission;

                            if (_Cdist.exited)
                                return emission;

                            Vec3f vmid = _Cdist.p2;
                            PhaseSample _phase;

                            if (!phase->sample(sampler, ray.dir(), _phase))
                                return emission;

                            Ray phaseRay(vmid, _phase.w);

                            arr<Vec3f> hps;
                            Vec2f uv;

                            bool hitLS = inter(phaseRay, S_plane, LS_orig, LS_u, LS_v, false, false, hps, uv,
                                               phaseRay.nearT(), phaseRay.farT(), LS_norm);

                            if (!hitLS)
                                return emission;

                            ray_from_to(hps[0], vmid, LS_M);
                            ray_from_to(vmid, ray.pos(), M_C);

                            LS_M.setPrimaryRay(false);
                            M_C.setPrimaryRay(false);

                            if (_scene->occluded(LS_M) || _scene->occluded(M_C))
                            {
                                emission = Vec3f(0.0f);
                                return emission;
                            }
                            else
                            {

                                hit_or_occluded = false;

                                psmp_LS.p = hps[0];
                                psmp_LS.uv = uv;
                                psmp_LS.weight = Vec3f(LS_u.cross(LS_v).length());

                                dsmp_LS.d = -phaseRay.dir();

                                Vec3f trp = light->evalDirectionalEmission(psmp_LS, dsmp_LS) *
                                            light->evalPositionalEmission(psmp_LS) *
                                            medium->transmittance(sampler, LS_M, false, false) *
                                            phase->eval(LS_M.dir(), M_C.dir()) /
                                            phase->pdf(LS_M.dir(), M_C.dir()) * sigS / sigT /
                                            abs(LS_norm.dot(dsmp_LS.d));

                                emission = trp;
                            }

                            return emission;
                        }

                    case 8:
                        // ARB_plane

                        {

                            emission = Vec3f(0.0f);

                            PositionSample _LS;
                            if (!light->samplePosition(sampler, _LS))
                                return emission;

                            float alpha = sampler.next1D() * M_PI;
                            Vec2f dir = Vec2f(cosf(alpha), sinf(alpha));
                            std::vector<Vec3f> line = light->sampleLine(_LS.p, dir);

                            Vec3f p_arbline = line[0];
                            Vec3f v_arbline = line[1] - line[0];
                            Vec3f d_arbline = v_arbline.normalized();

                            float V = _LS.uv[1] * LS_v.length();

                            DirectionSample _LSD;
                            if (!light->sampleDirection(sampler, _LS, _LSD))
                                return emission;

                            Ray LSray(_LS.p, _LSD.d);

                            MediumSample _Cdist;

                            arr<Vec3f> hps, hps2;
                            Vec2f uv;

                            bool hit = inter(ray, S_plane, p_arbline, v_arbline, _LSD.d, false, true, hps, uv,
                                             ray.nearT(), ray.farT(), LS_norm);

                            if (!hit)
                            {
                                return Vec3f(0.0f, 0.0f, 0.0f);
                            }
                            else
                            {
                                hit_or_occluded = false;

                                Ray sray(hps[0], -_LSD.d);

                                bool hit2 = inter(sray, S_plane, LS_orig, LS_u, LS_v, false, false, hps2, uv,
                                                  ray.nearT(), D, LS_norm);

                                if (!hit2)
                                    return Vec3f(0.0f, 1.0f, 0.0f);

                                Vec3f vmid = hps[0];
                                Vec3f vLS = hps2[0];

                                ray_from_to(vLS, vmid, LS_M);
                                ray_from_to(vmid, ray.pos(), M_C);

                                LS_M.setPrimaryRay(false);
                                M_C.setPrimaryRay(false);

                                if (_scene->occluded(LS_M) || _scene->occluded(M_C))
                                {
                                    emission = Vec3f(0.0f);
                                    return emission;
                                }

                                psmp_LS.p = vLS;

                                Vec3f psi1 = ray.dir();
                                Vec3f omega1 = LSray.dir();
                                Vec3f clt = psi1.cross(omega1);

                                float rj =
                                    2 * sqrtf(powf(clt.dot(LS_udir), 2) + powf(clt.dot(LS_vdir), 2)) / M_PI;

                                emission = light->evalPositionalEmission(psmp_LS)

                                           * LS_v.length()

                                           * medium->transmittance(sampler, M_C, false, false) *
                                           medium->transmittance(sampler, LS_M, false, false)

                                           * phase->eval(LS_M.dir(), M_C.dir()) * sigS / sigT / rj;

                                return emission;
                            }
                        }

                    // differential edge plane VVT
                    case 9:
                    {

                        emission = Vec3f(0.0f);

                        PositionSample _LS;
                        if (!light->samplePosition(sampler, _LS))
                            return emission;

                        std::vector<Vec3f> poses = light->getCorner();
                        Vec3f orig = poses[0];

                        Vec3f d0 = (poses[1] - poses[0]).normalized();
                        Vec3f d1 = (poses[2] - poses[0]).normalized();

                        Vec2f decomposed_LS = decomp_plx(orig, d0, d1, _LS.p);

                        Edge ed = _scene->sample_edge(sampler);

                        float Edist = sampler.next1D() * (ed.p1 - ed.p0).length();

                        Vec3f smped_edge_V = ed.p0 + (ed.p1 - ed.p0).normalized() * Edist;

                        arr<Vec3f> hps;
                        Vec2f uv;

                        Vec3f p1 = poses[0] + decomposed_LS[0] * d0;

                        bool hit = inter(ray, S_Q, p1, _LS.p - p1, smped_edge_V - _LS.p, true, true, hps, uv,
                                         ray.nearT(), ray.farT(), LS_norm);

                        if (hit)
                        {

                            Vec3f res(0.0f);

                            Vec3f vhit = hps[0];

                            Ray LSrayBK, hit2E;

                            // only difference: LSrayBK extends to infinity
                            ray_from_to(vhit, smped_edge_V, LSrayBK);
                            LSrayBK.setFarT(INFINITY);

                            ray_from_to(vhit, smped_edge_V, hit2E);

                            bool hit2 =
                                inter(LSrayBK, S_Q, poses[0], poses[1] - poses[0], poses[2] - poses[0], true,
                                      true, hps, uv, ray.nearT(), ray.farT(), LS_norm);

                            // did not hit the LS plane
                            if (!hit2)
                                return Vec3f(0.0f);

                            // now truncated at LS
                            LSrayBK.setFarT((hps[0] - LSrayBK.pos()).length());

                            if ((_scene->occluded(LSrayBK)))
                                return Vec3f(0.0f);

                            if ((_scene->occluded(hit2E)))
                                return Vec3f(0.0f);

                            // hit outside LS
                            if (uv[0] < 0 || uv[0] > (poses[1] - poses[0]).length())
                                return Vec3f(0.0f);

                            if (uv[1] < 0 || uv[1] > (poses[2] - poses[0]).length())
                                return Vec3f(0.0f);

                            // hps[0] now updated to hitpoint on LS
                            Ray LSP2EdgeV;
                            ray_from_to(smped_edge_V, hps[0], LSP2EdgeV);

                            Ray hitP2LSV;
                            ray_from_to(vhit, hps[0], hitP2LSV);

                            if ((_scene->occluded(LSP2EdgeV)))
                                return Vec3f(0.0f);

                            LSP2EdgeV.setPrimaryRay(false);

                            dsmp_LS.d = -(hps[0] - smped_edge_V).normalized();
                            hit_or_occluded = false;

                            // primitive norm
                            Vec3f norm = LSP2EdgeV.dir().cross(d1).normalized();

                            // discontinuity boundary norm
                            Vec3f norm2 = LSP2EdgeV.dir().cross(ed.p1 - ed.p0).normalized();

                            float norm_sign2 = norm2.dot(smped_edge_V - Vec3f(0.0f, 1.6f, 0.0f));

                            if (norm_sign2 < 0)
                                norm2 = -norm2;

                            Vec3f tangent = norm.cross(dsmp_LS.d).normalized();
                            float rad = LSP2EdgeV.farT();
                            float rad0 = abs(hitP2LSV.farT());

                            float rad2 = abs(hitP2LSV.farT());

                            float sinn = (dsmp_LS.d.cross(ed.dir_)).length();
                            float deriv_acos = 1 / sinn;

                            Vec3f pjv = LSP2EdgeV.dir().dot(ed.dir_) * LSP2EdgeV.farT() * ed.dir_;
                            Vec3f to_pjpoint = LSP2EdgeV.dir() * LSP2EdgeV.farT() - pjv;

                            float deriv_len = deriv_acos * to_pjpoint.length();

                            Vec3f real_tangent = norm.cross(dsmp_LS.d) * hit2E.farT() + dsmp_LS.d * deriv_len;

                            tangent = real_tangent.normalized();

                            float JACX = real_tangent.cross(dsmp_LS.d).dot(ray.dir());
                            JACX *= tangent.dot(ed.dir_);

                            // wE: edge seg
                            // we: edge

                            Vec3f We = ed.dir_;
                            Vec3f WE = hitP2LSV.dir();

                            Vec3f tan1 = We.cross(WE).normalized();
                            Vec3f tan2 = WE.cross(tan1).normalized();

                            float Rd = LSP2EdgeV.farT();
                            float Rs = hitP2LSV.farT() - Rd;

                            float sintheta = ed.dir_.cross(LSP2EdgeV.dir()).length();

                            Vec3f RsT1 = tan1 * Rs;
                            Vec3f RdT1 = tan1 * Rd;

                            Vec3f RsT2S = tan2 * Rs * sintheta;
                            Vec3f RdT2S = tan2 * Rd * sintheta;

                            Eigen::Matrix<float, 6, 6> M6;

                            Eigen::VectorXf R0(6), R1(6), R2(6), R3(6), R4(6), R5(6);

                            R0 << -WE[0], -WE[1], -WE[2], 0.0f, 0.0f, 0.0f;
                            R1 << 0.0f, 0.0f, 0.0f, WE[0], WE[1], WE[2];

                            R2 << RsT1[0], RsT1[1], RsT1[2], -RdT1[0], -RdT1[1], -RdT1[2];
                            R3 << RsT2S[0], RsT2S[1], RsT2S[2], -RdT2S[0], -RdT2S[1], -RdT2S[2];

                            R4 << We[0], We[1], We[2], We[0], We[1], We[2];
                            R5 << 0.0f, 0.0f, 0.0f, ray.dir()[0], ray.dir()[1], ray.dir()[2];

                            M6 << R0, R1, R2, R3, R4, R5;

                            float jacz = abs(M6.determinant());

                            JACX = abs(JACX);

                            float boundary_V = (norm2.dot(shiftD));

                            res = _LS.weight * medium->transmittance(sampler, hitP2LSV, false, false) *
                                  light->evalDirectionalEmission(_LS, dsmp_LS) * M_PI *
                                  phase->eval(LSP2EdgeV.dir(), ray.dir()) * sigS / jacz * boundary_V /
                                  ed.len_;

                            return res;
                        }
                        else
                            return Vec3f(0.0f, 0.0f, 0.0f);
                    }

                    // differential edge plane VE
                    case 10:

                    {

                        emission = Vec3f(0.0f);

                        PositionSample _LS;
                        if (!light->samplePosition(sampler, _LS))
                            return emission;

                        std::vector<Vec3f> poses = light->getCorner();
                        Vec3f orig = poses[0];

                        Vec3f d0 = (poses[1] - poses[0]).normalized();
                        Vec3f d1 = (poses[2] - poses[0]).normalized();

                        Vec2f decomposed_LS = decomp_plx(orig, d0, d1, _LS.p);

                        Edge ed = _scene->sample_edge(sampler);

                        arr<Vec3f> hps;
                        Vec2f uv;

                        Vec3f p1 = poses[0] + decomposed_LS[0] * d0;

                        bool hit = inter(ray, S_Q, _LS.p, ed.p0 - _LS.p, ed.p1 - _LS.p, true, true, hps, uv,
                                         ray.nearT(), ray.farT(), LS_norm);

                        if (hit)
                        {

                            Vec3f res(0.0f);

                            Vec3f vhit = hps[0];

                            Ray LSrayBK;

                            ray_from_to(vhit, _LS.p, LSrayBK);

                            float lenx = (vhit - _LS.p).length();

                            if ((_scene->occluded(LSrayBK)))
                                return Vec3f(0.0f);

                            Vec3f L_dir = (vhit - _LS.p).normalized();

                            Vec3f dir_to_v0 = (ed.p0 - _LS.p).normalized();

                            Vec2f decomp_proj_onto_edge = decomp_plx(_LS.p, dir_to_v0, ed.dir_, vhit);

                            float dist_to_v0 = (ed.p0 - _LS.p).length();
                            float dist_on_edge =
                                decomp_proj_onto_edge[1] * dist_to_v0 / decomp_proj_onto_edge[0];
                            float dist_to_edge = lenx * dist_to_v0 / decomp_proj_onto_edge[0];

                            if (lenx < dist_to_edge)
                                return Vec3f(0.0f);

                            if (dist_on_edge < 0 || dist_on_edge > ed.len_)
                                return Vec3f(0.0f);

                            dsmp_LS.d = (hps[0] - _LS.p).normalized();
                            hit_or_occluded = false;

                            Vec3f norm = dsmp_LS.d.cross(ed.dir_);

                            Vec3f smped_edge_V = ed.p0 + ed.dir_ * dist_on_edge;

                            float norm_sign = norm.dot(smped_edge_V - Vec3f(0.0f, 1.6f, 0.0f));

                            if (norm_sign < 0)
                                norm = -norm;

                            Vec3f norm_normalized = norm.normalized();

                            float boundary_V = (norm.dot(shiftD));

                            Vec3f tangent = (hps[0] - _LS.p).cross(norm_normalized).normalized();

                            float JAC = abs(tangent.cross(dsmp_LS.d).dot(ray.dir()));

                            JAC *= (hps[0] - _LS.p).length();

                            JAC /= dist_to_edge;

                            JAC = abs(JAC);

                            res = _LS.weight * medium->transmittance(sampler, LSrayBK, false, false) *
                                  light->evalDirectionalEmission(_LS, dsmp_LS) * M_PI *
                                  phase->eval(LSrayBK.dir(), ray.dir()) * sigS * boundary_V / JAC /
                                  (poses[2] - poses[0]).length();

                            return res;
                        }
                        else
                            return Vec3f(0.0f, 0.0f, 0.0f);
                    }
                    }

                    return emission;
                }

                bounce++;
                if (bounce < _settings.maxBounces)
                    didHit = _scene->intersect(ray, data, info);
            }

            return emission;
        }
        catch (std::runtime_error &e)
        {
            std::cout << tfm::format("Caught an internal error at pixel %s: %s", pixel, e.what())
                      << std::endl;

            return Vec3f(0.0f);
        }
    }

    Vec3f PathTracer::traceSample(Vec2u pixel, PathSampleGenerator &sampler)
    {
        Ray LS_M, M_C;

        float lightPdf;
        const Primitive *light = chooseLightAdjoint(sampler, lightPdf);

        PositionSample point;

        if (!light->samplePosition(sampler, point))
            return Vec3f(0.0f);

        Vec3f ev1, ev2;

        bool path_unclear = true;

        Vec3f v2 = routine(1, pixel, sampler, LS_M, M_C, path_unclear);

        return v2;

        Vec3f res1 = routine(9, pixel, sampler, LS_M, M_C, path_unclear);
        if (!path_unclear)
        {
            Vec3f res2 = reeval(10, sampler, LS_M, M_C, point);
            ev1 = Vec3f(2.0f) / ((Vec3f(1.0f) / res1 + Vec3f(1.0f) / res2));
        }
        else
            ev1 = Vec3f(0.0f);

        path_unclear = true;

        Vec3f res3 = routine(10, pixel, sampler, LS_M, M_C, path_unclear);
        if (!path_unclear)
        {
            Vec3f res4 = reeval(9, sampler, LS_M, M_C, point);
            ev2 = Vec3f(2.0f) / ((Vec3f(1.0f) / res3 + Vec3f(1.0f) / res4));
        }
        else
            ev2 = Vec3f(0.0f);

        return (ev1 + ev2) / 2.0f;
    }

} // namespace Tungsten