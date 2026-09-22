// My research branch of Tungsten (Benedikt Bitterli). See CONTRIBUTION.md and LICENSE.txt.
// Formatting/comment cleanup only; host-renderer dependencies are not bundled.

#include "PhotonMapIntegrator.hpp"
#include "PhotonTracer.hpp"

#include "sampling/UniformPathSampler.hpp"
#include "sampling/SobolPathSampler.hpp"

#include "cameras/PinholeCamera.hpp"

#include "thread/ThreadUtils.hpp"
#include "thread/ThreadPool.hpp"

#include "bvh/BinaryBvh.hpp"

#include <core/renderer/TraceableScene.hpp>

namespace Tungsten
{

    CONSTEXPR uint32 PhotonMapIntegrator::TileSize;

    PhotonMapIntegrator::PhotonMapIntegrator() : _w(0), _h(0), _sampler(0xBA5EBA11) {}

    PhotonMapIntegrator::~PhotonMapIntegrator() {}

    void PhotonMapIntegrator::diceTiles()
    {
        for (uint32 y = 0; y < _h; y += TileSize)
        {
            for (uint32 x = 0; x < _w; x += TileSize)
            {
                _tiles.emplace_back(x, y, min(TileSize, _w - x), min(TileSize, _h - y),
                                    _scene->rendererSettings().useSobol()
                                        ? std::unique_ptr<PathSampleGenerator>(
                                              new SobolPathSampler(MathUtil::hash32(_sampler.nextI())))
                                        : std::unique_ptr<PathSampleGenerator>(
                                              new UniformPathSampler(MathUtil::hash32(_sampler.nextI()))));
            }
        }
    }

    void PhotonMapIntegrator::saveState(OutputStreamHandle & /*out*/) {}

    void PhotonMapIntegrator::loadState(InputStreamHandle & /*in*/) {}

    void PhotonMapIntegrator::tracePhotons(uint32 taskId, uint32 numSubTasks, uint32 threadId,
                                           uint32 sampleBase)
    {
        SubTaskData &data = _taskData[taskId];
        PathSampleGenerator &sampler = *_samplers[taskId];

        uint32 photonBase = intLerp(0, _settings.photonCount, taskId + 0, numSubTasks);
        uint32 photonsToCast = intLerp(0, _settings.photonCount, taskId + 1, numSubTasks) - photonBase;

        uint32 totalSurfaceCast = 0;
        uint32 totalVolumeCast = 0;
        uint32 totalPathsCast = 0;
        for (uint32 i = 0; i < photonsToCast; ++i)
        {
            sampler.startPath(0, sampleBase + photonBase + i);
            _tracers[threadId]->tracePhotonPath(data.surfaceRange, data.volumeRange, data.pathRange, sampler);
            if (!data.surfaceRange.full())
                totalSurfaceCast++;
            if (!data.volumeRange.full())
                totalVolumeCast++;
            if (!data.pathRange.full())
                totalPathsCast++;
            if (data.surfaceRange.full() && data.volumeRange.full() && data.pathRange.full())
                break;

            if (_group->isAborting())
                break;
        }

        _totalTracedSurfacePaths += totalSurfaceCast;
        _totalTracedVolumePaths += totalVolumeCast;
        _totalTracedPaths += totalPathsCast;
    }

    void PhotonMapIntegrator::tracePixels(uint32 tileId, uint32 threadId, float surfaceRadius,
                                          float volumeRadius)
    {
        int spp = _nextSpp - _currentSpp;

        ImageTile &tile = _tiles[tileId];
        for (uint32 y = 0; y < tile.h; ++y)
        {
            for (uint32 x = 0; x < tile.w; ++x)
            {
                Vec2u pixel(tile.x + x, tile.y + y);
                uint32 pixelIndex = pixel.x() + pixel.y() * _w;

                Ray dummyRay;
                Ray *depthRay = _depthBuffer ? &_depthBuffer[pixel.x() + pixel.y() * _w] : &dummyRay;
                for (int i = 0; i < spp; ++i)
                {
                    tile.sampler->startPath(pixelIndex, _currentSpp + i);
                    Vec3f c = _tracers[threadId]->traceSensorPath(
                        pixel, *_surfaceTree, _volumeTree.get(), _volumeBvh.get(), _volumeGrid.get(),
                        _beams.get(), _planes0D.get(), _planes1D.get(), _primitives.get(),
                        _volumeBvh_DBG.get(), _volumeGrid_DBG.get(), _volumeBvh_dangling.get(), *tile.sampler,
                        surfaceRadius, volumeRadius, _settings.volumePhotonType, *depthRay, _useFrustumGrid,
                        _volumes.get());
                    _scene->cam().colorBuffer()->addSample(pixel, c);
                }
            }
        }
    }

    template <typename PhotonType>
    std::unique_ptr<KdTree<PhotonType>> streamCompactAndBuild(std::vector<PhotonRange<PhotonType>> ranges,
                                                              std::vector<PhotonType> &photons,
                                                              uint32 totalTraced)
    {
        uint32 tail = streamCompact(ranges);

        float scale = 1.0f / totalTraced;
        for (uint32 i = 0; i < tail; ++i)
            photons[i].power *= scale;

        return std::unique_ptr<KdTree<PhotonType>>(new KdTree<PhotonType>(&photons[0], tail));
    }

    static void precomputeBeam(PhotonBeam &beam, const PathPhoton &p0, const PathPhoton &p1)
    {
        beam.p0 = p0.pos;
        beam.p1 = p1.pos;
        beam.dir = p0.dir;
        beam.length = p0.length;
        beam.power = p1.power;
        beam.bounce = p0.bounce();
        beam.valid = true;
    }

    static void precomputeBeam_info(PhotonBeam &beam, const PathPhoton &p0, const PathPhoton &p1, Vec3f info)
    {
        beam.p0 = p0.pos;
        beam.p1 = p1.pos;
        beam.dir = p0.dir;
        beam.length = p0.length;
        beam.power = info;
        beam.bounce = p0.bounce();
        beam.valid = true;
    }

    // precomp beam data and also insert into bound array
    static void insertDicedBeam_info(Bvh::PrimVector &beams, PhotonBeam &beam, uint32 i, const PathPhoton &p0,
                                     const PathPhoton &p1, float radius, Vec3f info)
    {
        precomputeBeam_info(beam, p0, p1, info);

        Vec3f absDir = std::abs(p0.dir);
        int majorAxis = absDir.maxDim();
        int numSteps = min(64, max(1, int(absDir[majorAxis] * 16.0f)));

        Vec3f minExtend = Vec3f(radius);
        for (int j = 0; j < 3; ++j)
        {
            minExtend[j] = std::copysign(minExtend[j], p0.dir[j]);
            if (j != majorAxis)
                minExtend[j] /= std::sqrt(max(0.0f, 1.0f - sqr(p0.dir[j])));
        }
        for (int j = 0; j < numSteps; ++j)
        {
            Vec3f v0 = p0.pos + p0.dir * p0.length * (j + 0) / numSteps;
            Vec3f v1 = p0.pos + p0.dir * p0.length * (j + 1) / numSteps;
            for (int k = 0; k < 3; ++k)
            {
                if (k != majorAxis || j == 0)
                    v0[k] -= minExtend[k];
                if (k != majorAxis || j == numSteps - 1)
                    v1[k] += minExtend[k];
            }
            Box3f bounds;
            bounds.grow(v0);
            bounds.grow(v1);

            beams.emplace_back(Bvh::Primitive(bounds, bounds.center(), i));
        }
    }

    static void precomputePlane0D(PhotonPlane0D &plane, const PathPhoton &p0, const PathPhoton &p1,
                                  const PathPhoton &p2)
    {
        Vec3f d1 = p1.dir * p1.sampledLength;
        plane = PhotonPlane0D{
            p0.pos, p1.pos,           p1.pos + d1,      p0.pos + d1, p0.length * p1.sampledLength * p2.power,
            p1.dir, p1.sampledLength, int(p1.bounce()), true};
    }
    static void precomputePlane1D(PhotonPlane1D &plane, const PathPhoton &p0, const PathPhoton &p1,
                                  const PathPhoton &p2, float radius)
    {
        Vec3f a = p1.pos - p0.pos;
        Vec3f b = p1.dir * p1.sampledLength;
        Vec3f c = 2.0f * a.cross(p1.dir).normalized() * radius;
        float det = std::abs(a.dot(b.cross(c)));

        if (std::isnan(c.sum()) || det < 1e-8f)
            return;

        float invDet = 1.0f / det;
        Vec3f u = invDet * b.cross(c);
        Vec3f v = invDet * c.cross(a);
        Vec3f w = invDet * a.cross(b);

        plane.p = p0.pos - c * 0.5f;
        plane.invDet = invDet;
        plane.invU = u;
        plane.invV = v;
        plane.invW = w;
        plane.binCount = a.length() / (2.0f * radius);
        plane.valid = true;

        plane.center = p0.pos + a * 0.5f + b * 0.5f;
        plane.a = a * 0.5f;
        plane.b = b * 0.5f;
        plane.c = c * 0.5f;

        plane.d1 = p1.dir;
        plane.l1 = p1.sampledLength;
        plane.power = p0.length * p1.sampledLength * p2.power * std::abs(invDet);
        plane.bounce = p1.bounce();
    }

    static void precompute_make_DicedBeam(Bvh::PrimVector &beams, PhotonBeam &beam, uint32 i,
                                          const PathPhoton &p0, const PathPhoton &p1, float radius)
    {
        precomputeBeam(beam, p0, p1);

        Vec3f absDir = std::abs(p0.dir);
        int majorAxis = absDir.maxDim();
        int numSteps = min(64, max(1, int(absDir[majorAxis] * 16.0f)));

        Vec3f minExtend = Vec3f(radius);
        for (int j = 0; j < 3; ++j)
        {
            minExtend[j] = std::copysign(minExtend[j], p0.dir[j]);
            if (j != majorAxis)
                minExtend[j] /= std::sqrt(max(0.0f, 1.0f - sqr(p0.dir[j])));
        }
        for (int j = 0; j < numSteps; ++j)
        {
            Vec3f v0 = p0.pos + p0.dir * p0.length * (j + 0) / numSteps;
            Vec3f v1 = p0.pos + p0.dir * p0.length * (j + 1) / numSteps;
            for (int k = 0; k < 3; ++k)
            {
                if (k != majorAxis || j == 0)
                    v0[k] -= minExtend[k];
                if (k != majorAxis || j == numSteps - 1)
                    v1[k] += minExtend[k];
            }
            Box3f bounds;
            bounds.grow(v0);
            bounds.grow(v1);

            beams.emplace_back(Bvh::Primitive(bounds, bounds.center(), i));
        }
    }

    void PhotonMapIntegrator::buildPointBvh(uint32 tail, float volumeRadiusScale)
    {
        float radius = _settings.volumeGatherRadius * volumeRadiusScale;

        Bvh::PrimVector points;
        for (uint32 i = 0; i < tail; ++i)
        {
            Box3f bounds(_pathPhotons[i].pos);
            bounds.grow(radius);
            points.emplace_back(Bvh::Primitive(bounds, _pathPhotons[i].pos, i));
        }

        _volumeBvh.reset(new Bvh::BinaryBvh(std::move(points), 1));
    }
    void PhotonMapIntegrator::buildBeamBvh(uint32 tail, float volumeRadiusScale)
    {
        float radius = _settings.volumeGatherRadius * volumeRadiusScale;

        Bvh::PrimVector beams;
        for (uint32 i = 0; i < tail; ++i)
        {
            if (_pathPhotons[i].bounce() < 0 || i == 0)
                continue;

            if (!_pathPhotons[i - 1].onSurface() || _settings.lowOrderScattering)
                precompute_make_DicedBeam(beams, _beams[i], i, _pathPhotons[i - 1], _pathPhotons[i], radius);
        }

        _volumeBvh.reset(new Bvh::BinaryBvh(std::move(beams), 1));
    }
    void PhotonMapIntegrator::buildPlaneBvh(uint32 tail, float volumeRadiusScale)
    {
        float radius = _settings.volumeGatherRadius * volumeRadiusScale;

        Bvh::PrimVector planes;
        for (uint32 i = 0; i < tail; ++i)
        {
            const PathPhoton &p0 = _pathPhotons[i - 2];
            const PathPhoton &p1 = _pathPhotons[i - 1];
            const PathPhoton &p2 = _pathPhotons[i - 0];

            if (p2.bounce() > 0 && p2.bounce() > p1.bounce() && p1.onSurface() &&
                _settings.lowOrderScattering)
                precompute_make_DicedBeam(planes, _beams[i], i, p1, p2, radius);
            if (p2.bounce() > 1 && !p1.onSurface() && p1.sampledLength > 0.0f)
            {
                if (_settings.volumePhotonType == PhotonMapSettings::VOLUME_PLANES)
                {
                    precomputePlane0D(_planes0D[i], p0, p1, p2);
                    Box3f bounds = _planes0D[i].bounds();
                    planes.emplace_back(Bvh::Primitive(bounds, bounds.center(), i));
                }
                else
                {
                    precomputePlane1D(_planes1D[i], p0, p1, p2, radius);
                    if (_planes1D[i].valid)
                    {
                        Box3f bounds = _planes1D[i].bounds();
                        planes.emplace_back(Bvh::Primitive(bounds, bounds.center(), i));
                    }
                }
            }
        }

        _volumeBvh.reset(new Bvh::BinaryBvh(std::move(planes), 1));
    }

    void PhotonMapIntegrator::buildBeamGrid(uint32 tail, float volumeRadiusScale)
    {
        float radius = _settings.volumeGatherRadius * volumeRadiusScale;

        std::vector<GridAccel::Primitive> beams;
        for (uint32 i = 1; i < tail; ++i)
        {
            const PathPhoton &p0 = _pathPhotons[i - 1];
            const PathPhoton &p1 = _pathPhotons[i - 0];
            if (_pathPhotons[i].bounce() == 0)
                continue;

            // beams: always built
            {
                precomputeBeam(_beams[i], p0, p1);
                beams.emplace_back(
                    GridAccel::Primitive(i, p0.pos, p1.pos, Vec3f(0.0f), Vec3f(0.0f), radius, true));
            }
        }

        _volumeGrid.reset(new GridAccel(_scene->bounds(), _settings.gridMemBudgetKb, std::move(beams)));
    }
    void PhotonMapIntegrator::buildPlaneGrid(uint32 tail, float volumeRadiusScale)
    {
        float radius = _settings.volumeGatherRadius * volumeRadiusScale;

        std::vector<GridAccel::Primitive> prims;
        for (uint32 i = 0; i < tail; ++i)
        {
            const PathPhoton &p0 = _pathPhotons[i - 2];
            const PathPhoton &p1 = _pathPhotons[i - 1];
            const PathPhoton &p2 = _pathPhotons[i - 0];

            if (p2.bounce() > 0 && p2.bounce() > p1.bounce() && p1.onSurface() &&
                _settings.lowOrderScattering)
            {
                precomputeBeam(_beams[i], p1, p2);
                prims.emplace_back(
                    GridAccel::Primitive(i, p1.pos, p2.pos, Vec3f(0.0f), Vec3f(0.0f), radius, true));
            }
            if (p2.bounce() > 1 && !p1.onSurface() && p1.sampledLength > 0.0f)
            {
                if (_settings.volumePhotonType == PhotonMapSettings::VOLUME_PLANES)
                {
                    precomputePlane0D(_planes0D[i], p0, p1, p2);
                    prims.emplace_back(GridAccel::Primitive(i, _planes0D[i].p0, _planes0D[i].p1,
                                                            _planes0D[i].p2, _planes0D[i].p3, 0.0f, false));
                }
                else
                {
                    precomputePlane1D(_planes1D[i], p0, p1, p2, radius);
                    if (_planes1D[i].valid)
                    {
                        Vec3f p = _planes1D[i].center, a = _planes1D[i].a, b = _planes1D[i].b;
                        prims.emplace_back(GridAccel::Primitive(i, p - a - b, p + a - b, p + a + b, p - a + b,
                                                                radius, false));
                    }
                }
            }
        }

        _volumeGrid.reset(new GridAccel(_scene->bounds(), _settings.gridMemBudgetKb, std::move(prims)));
    }

    void PhotonMapIntegrator::makePrimitiveBVH(uint32 tail, float volumeRadiusScale,
                                               Bvh::PrimVector &prims_out,
                                               PhotonMapSettings::Primitive_Policy &masterPolicy)
    {
        prims_out.clear();

        for (uint32 curpp = 0; curpp < tail; ++curpp)
        {
            if (_primitives[curpp].o_valid())
            {
                Box3f single_box;

                single_box = _primitives[curpp].io_bbox();

                // debug override
                if (_settings.D_field.is_on(_settings.D_force_sceneBBOX))
                    single_box = _scene->bounds();

                prims_out.emplace_back(Bvh::Primitive(single_box, single_box.center(), curpp));
            }
        }
    }

    void PhotonMapIntegrator::makePathBVH(uint32 tail, float volumeRadiusScale, Bvh::PrimVector &prims_out,
                                          PhotonMapSettings::Primitive_Policy &masterPolicy)
    {

        float brightness = 0.7f;

        if (_settings.D_field.is_on(_settings.D_viz_path))
        {

            float radius = _settings.volumeGatherRadius * volumeRadiusScale;

            // only the main path is covered here
            for (uint32 i = 1; i < tail; ++i)
            {

                if (_pathPhotons[i].first_in_path())
                    continue;

                if (_settings.D_field.is_on(_settings.D_CC_bc))
                {
                    int which = _pathPhotons[i].bounce() % 3;

                    if (_pathPhotons[i].bounce() <= 0)
                        which = 4;

                    Vec3f col(0.0f, 0.0f, 0.0f);
                    if (which == 0)
                        col = Vec3f(brightness, 0.0f, 0.0f);
                    else if (which == 1)
                        col = Vec3f(0.0f, brightness, 0.0f);
                    else if (which == 2)
                        col = Vec3f(0.0f, 0.0f, brightness);
                    else
                        col = Vec3f(brightness, brightness, brightness);

                    insertDicedBeam_info(prims_out, _beams[i], i, _pathPhotons[i - 1], _pathPhotons[i],
                                         radius, col);
                }
                else if (_settings.D_field.is_on(_settings.D_CC_bcSS))
                {

                    int which = _pathPhotons[i].bounceSS() % 3;
                    if (_pathPhotons[i].bounceSS() <= 0)
                        which = 4;

                    Vec3f col(0.0f, 0.0f, 0.0f);
                    if (which == 0)
                        col = Vec3f(brightness, 0.0f, 0.0f);
                    else if (which == 1)
                        col = Vec3f(0.0f, brightness, 0.0f);
                    else if (which == 2)
                        col = Vec3f(0.0f, 0.0f, brightness);
                    else
                        col = Vec3f(brightness, brightness, brightness);

                    insertDicedBeam_info(prims_out, _beams[i], i, _pathPhotons[i - 1], _pathPhotons[i],
                                         radius, col);
                }
                else
                    precompute_make_DicedBeam(prims_out, _beams[i], i, _pathPhotons[i - 1], _pathPhotons[i],
                                              radius);
            }
        }
    }

    void PhotonMapIntegrator::makeDanglingBVH(uint32 tail, float volumeRadiusScale,
                                              Bvh::PrimVector &prims_out,
                                              PhotonMapSettings::Primitive_Policy &masterPolicy)
    {

        float brightness = 0.4f;

        if (_settings.D_field.is_on(_settings.D_viz_dangling))
        {

            float radius = _settings.volumeGatherRadius * volumeRadiusScale;

            for (uint32 i = 1; i < tail; ++i)
            {

                if (!_primitives[i].o_valid())
                    continue;

                // use the segs in pathPhoton to generate beams and store in PP

                PhotonPrimitive &cur_prim = _primitives[i];
                PathPhoton &cur_photon = _pathPhotons[i];

                int lookup = cur_photon.extra.dangling_index;
                if (lookup < 0)
                    continue;
                PathPhoton &data_photon = _pathPhotons[lookup];

                arr<PhotonBeam> &beams = cur_prim.io_dbg_beams();
                arr<PathPhoton> &segs = data_photon.extra.danglings;

                if (segs.size() == 0)
                    continue;

                cur_prim.enable_feature(PhotonPrimitive::PS_frame);

                Vec3f color = Vec3f(brightness, brightness, 0.0f);

                for (int j = 0; j < segs.size() - 1; j++)
                {
                    PhotonBeam PB;

                    PB.bounce =
                        cur_prim.o_bounce() - (cur_prim.o_I_mybounce() - cur_prim.getIndex(0)) + j + 1;

                    insertDicedBeam_info(prims_out, PB, i, segs[j], segs[j + 1], _settings.volumeGatherRadius,
                                         color);

                    beams.emplace_back(PB);
                }
            }
        }
    }

    void PhotonMapIntegrator::makePrimitiveGrid(uint32 tail, float volumeRadiusScale,
                                                Bvh::PrimVector &prims_out,
                                                PhotonMapSettings::Primitive_Policy &masterPolicy)
    {
    }

    void PhotonMapIntegrator::buildPrimitiveAccStructure(uint32 tail, float volumeRadiusScale)
    {
        float radius = _settings.volumeGatherRadius * volumeRadiusScale;

        if (!_settings.useGrid)
        {
            Bvh::PrimVector primBounds;
            makePrimitiveBVH(tail, volumeRadiusScale, primBounds, _settings.surf_EST);
            Bvh::PrimVector debug_primBounds;
            makePathBVH(tail, volumeRadiusScale, debug_primBounds, _settings.surf_EST);

            Bvh::PrimVector dangling_primBounds;
            makeDanglingBVH(tail, volumeRadiusScale, dangling_primBounds, _settings.surf_EST);

            _volumeBvh.reset(new Bvh::BinaryBvh(std::move(primBounds), 1));
            _volumeBvh_DBG.reset(new Bvh::BinaryBvh(std::move(debug_primBounds), 1));
            _volumeBvh_dangling.reset(new Bvh::BinaryBvh(std::move(dangling_primBounds), 1));
        }
        else
        {
        }
    }

    void PhotonMapIntegrator::Add_Dangling_Segments(uint32 tail)
    {
        bool homo_trans = _settings.S_field.is_on(_settings.S_homo_transmittance);
        bool homo_phase = _settings.S_field.is_on(_settings.S_homo_phase);
        bool homo = homo_trans && homo_phase;

        for (uint32 index = 0; index < tail; ++index)
        {
            PathPhoton &p = _pathPhotons[index];

            // a negative index means infinity here
            if (p.extra.dangling_index < tail && p.extra.dangling_index >= p.extra.data_bracketL_far)
            {
                uint32 pos_store = p.extra.dangling_index;
                uint32 pos_LT = p.index_LT;

                PathPhoton &p_storage = _pathPhotons[pos_store];

                // should be the total EST length
                int segnum = _settings.surf_EST.nth_comb(p.handler)->seg_involved();

                int segs_nodangling = pos_store - p.index_comb_start;

                if (segnum == 0)
                {
                }

                // already have enough
                if (segnum <= p_storage.extra.danglings.size() + segs_nodangling)
                    continue;

                p_storage.extra.danglings.clear();

                // negative index (infinity)
                if (p.index_comb_start > tail)
                    continue;

                // here it should be dangling only
                for (int k = p.index_comb_start; k <= pos_store; k++)
                    p_storage.extra.danglings.emplace_back(_pathPhotons[k]);

                PathPhoton &prevlast = p_storage.extra.danglings.back();

                int BC = prevlast.BC;
                int BCSS = prevlast.BCSS;
                Vec3f curpos = prevlast.pos;

                for (int i = 0; i < segnum - segs_nodangling; i++)
                {

                    // use prev one to replace
                    int pos_curSeg = pos_store + i;

                    // use the data stored in the cur for the next
                    if (pos_curSeg >= tail)
                        break;

                    curpos += _pathPhotons[pos_curSeg].dir_marg * _pathPhotons[pos_curSeg].length_marg;

                    PathPhoton ppx;

                    // incomplete, using pos only
                    ppx.pos = curpos;

                    BC++;
                    BCSS++;

                    ppx.BC = BC;
                    ppx.BCSS = BCSS;

                    p_storage.extra.danglings.emplace_back(ppx);
                }

                for (int i = 0; i < segnum; i++)
                {
                    PathPhoton &pp = p_storage.extra.danglings[i];
                    PathPhoton &ppnext = p_storage.extra.danglings[i + 1];
                    pp.dir = ppnext.pos - pp.pos;
                    // might need to change this later
                    // to create a shorter plane
                    pp.length_marg = pp.length = pp.sampledLength = pp.dir.length();
                    pp.dir /= pp.length_marg;
                    pp.dir_marg = pp.dir;
                }
            }
        }
    }

    void PhotonMapIntegrator::showPhotonInfo(uint32 k, uint32 tail)
    {
        char buf[10];
        sprintf(buf, "%d", tail);
        int maxlen = strlen(buf);

        sst stringBuffer;

        sprintf(buf, "%d", k);
        str p1 = str("<") + str(buf) + str("> ");

        prtln std::setw(maxlen + 3) << std::left << p1;

        prt(_pathPhotons[k].first_in_path() ? "L" : "_");
        prt(_pathPhotons[k].initial() ? "I" : "_");
        prt(_pathPhotons[k].last_in_path() ? "R" : "_");
        prt(_pathPhotons[k].LS_dims() ? "O" : "_");
        prt(_pathPhotons[k].onSurface() ? "S" : "_");

        prt "[L ";
        prt(_pathPhotons[k].index_LT);
        prt ",D ";
        prt(_pathPhotons[k].extra.dangling_using);
        prt ",D= ";
        prt(_pathPhotons[k].extra.danglings.size());
        prt "]";

        // bounce brackets:
        // indicating: [ )

        prt " ";

        stringBuffer.str("");
        stringBuffer << _pathPhotons[k].extra.data_bracketL_far << "-"
                     << _pathPhotons[k].extra.data_bracketR_far << " " << _pathPhotons[k].extra.data_bracketL
                     << "-" << _pathPhotons[k].extra.data_bracketR;
        prt std::setw(6) << std::left << stringBuffer.str();

        stringBuffer.str("");

        if (_pathPhotons[k].LS_dims())
            stringBuffer << "|";
        else
        {
            if (_pathPhotons[k].initial())
                stringBuffer << "[";
            else
                stringBuffer << " ";

            if (_pathPhotons[k].onSurface())
                stringBuffer << "*";
            else if (k != tail - 1)
            {
                if (!_pathPhotons[k].initial() && !_pathPhotons[k + 1].initial())
                    stringBuffer << ".";
                else
                    stringBuffer << " ";
            }
            else
                stringBuffer << " ";

            if (k == tail - 1)
                stringBuffer << "]";
            else if (_pathPhotons[k + 1].initial() || _pathPhotons[k + 1].LS_dims())
                stringBuffer << "]";
            else
                stringBuffer << " ";
        }

        prt std::setw(5) << std::left << stringBuffer.str();

        if (_pathPhotons[k].handler > -1)
            prt _pathPhotons[k].handler;
        else if (_pathPhotons[k].handler == -1)
            prt "X";
        else if (_pathPhotons[k].handler == -2)
            prt "-";

        stringBuffer.str("");
        prt std::setw(10) << std::left << stringBuffer.str();
        stringBuffer.str("");
        stringBuffer << _pathPhotons[k];
        prt std::setw(10) << std::left << stringBuffer.str();
    }

    void PhotonMapIntegrator::Add_marg_segments(uint32 tail)
    {
        bool homo_trans = _settings.S_field.is_on(_settings.S_homo_transmittance);
        bool homo_phase = _settings.S_field.is_on(_settings.S_homo_phase);
        bool homo = homo_trans && homo_phase;

        for (uint32 index = 1; index < tail; ++index)
        {
            PathPhoton &p = _pathPhotons[index];
            PathPhoton &pprev = _pathPhotons[index - 1];

            // for now, always sample new dirs after hitting a surface
            // dir: to the next

            Vec3f init_dir(0.0f, -1.0f, 0.0f); // -Y from LS

            if (index > 1)
                init_dir = pprev.dir;

            // just generate for all segment
            // can't really tell where it's needed
            // even on the marginalizable ones, since u might need multiple stored on other location

            if (!homo_phase || p.blocking())
            {
                p.dir_marg = _settings.smp_dir_homo_phase(init_dir);
                Ray rx(p.pos, p.dir_marg);
                p.length_marg = _settings.smp_dist_homo_trans(rx);
            }
        }

        // len_marg might differ from len (one of them is the next bounce)
        // only case needing a non-const phase: non-homo phase
    }

    void PhotonMapIntegrator::PostProcess_PathPhotons(uint32 tail, bool disp)
    {

        // cause still unknown, but this sometimes happens with multithreading (MT) on
        for (uint32 i = 0; i < tail - 1; ++i)
        {
            PathPhoton &cur = _pathPhotons[i];
            PathPhoton &next = _pathPhotons[i];

            if (cur.extra.profile.is_on(cur.extra.EF_SF) && next.BC == 0)
                cur.BCSS = 0;
        }

        // by default: brackets enabled
        for (uint32 i = 0; i < tail; ++i)
            _pathPhotons[i].extra.enable_feature(PathPhoton::extra_feature_blob::EF_brackets);

        for (uint32 i = 0; i < tail; ++i)
        {
            for (uint32 j = i + 1; j <= tail; ++j)
            {
                bool rdy = false;
                if (j == tail)
                    rdy = true;
                else if (_pathPhotons[j].BCSS <= _pathPhotons[j - 1].BCSS)
                    rdy = true;

                if (rdy)
                {
                    for (int k = i; k < j; k++)
                    {
                        _pathPhotons[k].extra.data_bracketL = i;
                        _pathPhotons[k].extra.data_bracketR = j - 1;
                    }
                    i = j - 1;
                    break;
                }
            }
        }

        for (uint32 i = 0; i < tail; ++i)
        {
            for (uint32 j = i + 1; j <= tail; ++j)
            {
                bool rdy = false;
                if (j == tail)
                    rdy = true;
                else if (_pathPhotons[j].BC <= _pathPhotons[j - 1].BC)
                    rdy = true;

                if (rdy)
                {
                    for (int k = i; k < j; k++)
                    {
                        _pathPhotons[k].extra.data_bracketL_far = i;
                        _pathPhotons[k].extra.data_bracketR_far = j - 1;
                    }
                    i = j - 1;
                    break;
                }
            }
        }

        for (uint32 k = 0; k < tail; ++k)
        {
            PhotonMapSettings::Primitive_Combination *curStrat = nullptr;

            if (_pathPhotons[k].BCSS >= 0)
            {
                if (_settings.S_field.is_on(_settings.S_homo_transmittance))
                    curStrat = _settings.surf_EST.B_BSS_strat(_pathPhotons[k].BC, _pathPhotons[k].BCSS);
                else if (_pathPhotons[k].BCSS < 1)
                    curStrat = _settings.surf_EST.B_BSS_strat(_pathPhotons[k].BC, _pathPhotons[k].BCSS);
            }
            else
                _pathPhotons[k].handler = -2;

            if (curStrat)
            {
                _pathPhotons[k].handler = _settings.surf_EST.strat_id(curStrat);
                _pathPhotons[k].index_comb_start = k - curStrat->o_comblen();
            }

            if (disp)
                showPhotonInfo(k, tail);
        }

        PhotonMapSettings::Primitive_Policy &EST = _settings.surf_EST;
        arr<PhotonMapSettings::Primitive_Combination> &strats = EST.io_strategies();

        int max_EST_len = _settings.surf_EST.max_BSS_involved();

        // first photon not handled by anything; to be fixed later

        auto prep_danglings_placeholders =
            [&](uint32 index_store, int missing_bounces, uint32 index, int which_strat)
        {
            bool inserted_anything = false;

            if (missing_bounces > 0)
            {
                inserted_anything = true;

                _pathPhotons[index].extra.dangling_index = index_store;
                _pathPhotons[index_store].extra.enable_feature(
                    PathPhoton::extra_feature_blob::EF_dangling_store);
                _pathPhotons[index].extra.enable_feature(PathPhoton::extra_feature_blob::EF_dangling_using);

                _pathPhotons[index].extra.dangling_using = missing_bounces;

                if (missing_bounces > _pathPhotons[index_store].extra.dangling_storing)
                {
                    _pathPhotons[index_store].extra.dangling_storing = missing_bounces;
                    _pathPhotons[index_store].extra.danglings.clear();
                    _pathPhotons[index_store].extra.danglings.reserve(missing_bounces);
                }
            }

            _pathPhotons[index].handler = which_strat;

            _pathPhotons[index].index_comb_start = index - strats[which_strat].o_comblen();

            return inserted_anything;
        };

        int64 prev_block = -1;

        for (uint32 index = 0; index < tail; ++index)
        {
            if (_pathPhotons[index].BCSS < 0)
                continue;

            uint32 prev_light = index - _pathPhotons[index].BC;

            bool trying_to_fix = (_pathPhotons[index].handler == -1);
            if (!trying_to_fix)
                continue;

            PathPhoton &LT_photon = _pathPhotons[prev_light];

            int LTdim = LT_photon.extra.LS.dim();

            uint32 prev_light_ext = index - _pathPhotons[index].BC - LTdim;

            if (_pathPhotons[index].blocking())
                prev_block = index;

            for (int which_strat = strats.size() - 1; which_strat >= 0; which_strat--)
            {
                PhotonMapSettings::Primitive_Combination &cur_strat = strats[which_strat];
                int stratLen = cur_strat.seg_involved();
                // minimum requirement: enough absolute bounces
                // before knowing if LS is allowed, assume yes
                // the LS that's directly accessible
                bool fixing;

                int BC_MX = cur_strat.use_LT_dims() ? _pathPhotons[index].BC_M : _pathPhotons[index].BC_M_NL;
                int LTdim_effective = cur_strat.use_LT_dims() ? LTdim : 0;
                uint32 prev_LT_ext_effective = cur_strat.use_LT_dims() ? prev_light_ext : prev_light;

                // extra and not included
                // only inserting when not enough
                if (cur_strat.seg_involved() < BC_MX)
                {
                    // has enough
                    // either handled or not needed
                    // either way, nothing to fix
                    // in this case, BCSS won't change
                    if (!cur_strat.BCSS_in_range(_pathPhotons[index].BCSS))
                        fixing = false;
                    else
                    {
                        // already handling
                        if (_pathPhotons[index].handler > 0)
                            fixing = false;
                        else
                            fixing = true;
                    }
                }
                else
                {
                    // no need to store another noclip BC_M
                    fixing = _pathPhotons[index].BC + LTdim_effective >= stratLen;
                }

                bool hasLT = cur_strat.use_LT_dims();

                if (fixing)
                {
                    // seg_involved is actually #bounces
                    // TODO: change this name
                    int strat_bounces = cur_strat.seg_involved();

                    // photons stored where orig path still valid
                    // array stores all the changed vertices

                    if (_settings.S_field.is_on(_settings.S_homo_transmittance))
                    {

                        int farthest_store_pos = index - strat_bounces;
                        if (farthest_store_pos < prev_LT_ext_effective)
                            farthest_store_pos = prev_LT_ext_effective;

                        bool already_found = false;
                        for (int curpos = index - 1; curpos >= farthest_store_pos; curpos--)
                        {
                            int to_insert = index - curpos + 1;

                            bool can_handle = (BC_MX + to_insert) >= stratLen;

                            if (can_handle)
                            {
                                already_found = true;
                                prep_danglings_placeholders(curpos, to_insert, index, which_strat);
                                break;
                            }
                        }
                        if (already_found)
                            break;
                    }
                    else
                    {

                        bool can_handle = index - stratLen >= prev_LT_ext_effective;

                        if (!can_handle)
                            continue;

                        // 1 non-marg edge is allowed
                        int to_insert = stratLen;
                        int curpos = index - to_insert;

                        if (curpos <= prev_light)
                        {
                            // LT dims need not be made marginalizable
                            to_insert -= (prev_light - curpos);
                            curpos += (prev_light - curpos);
                        }

                        if (can_handle)
                        {
                            prep_danglings_placeholders(curpos, to_insert, index, which_strat);
                            break;
                        }
                    }
                }
            }
        }

        if (disp)
        {
            for (int k = 0; k < tail; k++)
                showPhotonInfo(k, tail);
        }
    }

    // assume biased ESTs would also be supported
    void PhotonMapIntegrator::prepare_PhotonPrimitives(uint32 tail, float volumeRadiusScale)
    {
        UniformSampler smp_prim; // for selecting type of est to draw
        UniformSampler smp_dir;  // for selecting arbitrary dir

        PhotonPrimitive::initialize_globals(_pathPhotons.data(), _scene->bounds(), &_settings);

        for (uint32 curpp = 0; curpp < tail; ++curpp)
        {
            PhotonPrimitive &curprim = _primitives[curpp];
            PathPhoton &curphoton = _pathPhotons[curpp];
            int chosen_strategy = curphoton.handler;

            if (chosen_strategy < 0)
            {
                curprim.set_invalid();
                continue;
            }

            curprim.set_valid();

            arr<PhotonMapSettings::Primitive_Combination> strats = _settings.surf_EST.o_strategies();

            if (chosen_strategy != -1)
            {
                if (strats[chosen_strategy].o_combine() ==
                    PhotonMapSettings::Primitive_Combination::COMB_1PRIM)
                {
                    int which_prim = _sampler.next1D() * strats[chosen_strategy].io_primitives().size();
                    _primitives[curpp].precompute(chosen_strategy, which_prim, curpp);
                }
                else if (strats[chosen_strategy].o_combine() ==
                         PhotonMapSettings::Primitive_Combination::COMB_ALL)
                {
                    // not handled yet
                }
            }
        }
    }

    void PhotonMapIntegrator::buildPhotonDataStructures(float volumeRadiusScale)
    {
        // don't open a shared_ptr
        _settings.sampling_medium = _scene->media()[0];

        // classic photon maps
        // surface: geometry in the scene, not PS
        std::vector<SurfacePhotonRange> surfaceRanges;
        std::vector<VolumePhotonRange> volumeRanges;
        std::vector<PathPhotonRange> pathRanges;

        for (const SubTaskData &data : _taskData)
        {
            surfaceRanges.emplace_back(data.surfaceRange);
            volumeRanges.emplace_back(data.volumeRange);
            pathRanges.emplace_back(data.pathRange);
        }

        _surfaceTree = streamCompactAndBuild(surfaceRanges, _surfacePhotons, _totalTracedSurfacePaths);

        const Vec3f &vmax = _scene->bounds().max();
        const Vec3f &vmin = _scene->bounds().min();

        // fixing invalid bbox
        for (int i = 0; i < 3; i++)
        {
            if (vmax[i] < vmin[i])
            {
                prt "\n>>>   #Scene bounds invalid#";
                prt "\n>>>>   #Quick Fix#\n";
                const_cast<TraceableScene *>(_scene)->fix_bounds();
                break;
            }
            assert(vmax[i] >= vmin[i]);
        }

        if (!_volumePhotons.empty())
        {
            _volumeTree = streamCompactAndBuild(volumeRanges, _volumePhotons, _totalTracedVolumePaths);
            float volumeRadius = _settings.fixedVolumeRadius ? _settings.volumeGatherRadius : 1.0f;
            _volumeTree->buildVolumeHierarchy(_settings.fixedVolumeRadius, volumeRadius * volumeRadiusScale);
        }
        else if (!_pathPhotons.empty())
        {
            uint32 tail = streamCompact(pathRanges);
            for (uint32 i = 0; i < tail; ++i)
            {
                _pathPhotons[i].power *= (1.0 / _totalTracedPaths);
            }

            PostProcess_PathPhotons(tail, _settings.D_field.is_on(PhotonMapSettings::D_print_photoninfo));

            _beams.reset(new PhotonBeam[tail]);
            for (uint32 i = 0; i < tail; ++i)
                _beams[i].valid = false;

            if (_settings.volumePhotonType == PhotonMapSettings::VOLUME_PRIMITIVES)
            {
                _primitives.reset(new PhotonPrimitive[tail]);
                for (uint32 i = 0; i < tail; ++i)
                    _primitives[i].set_invalid();

                Add_marg_segments(tail);

                prtln "Marginalizable edges added.";

                Add_Dangling_Segments(tail);

                prtln "Dangling segments added.";

                if (_settings.D_field.is_on(_settings.D_print_photoninfo))
                {
                    for (uint32 k = 0; k < tail; ++k)
                        showPhotonInfo(k, tail);
                }

                prepare_PhotonPrimitives(tail, volumeRadiusScale);
                prtln "Photon primitives prepared.";

                buildPrimitiveAccStructure(tail, volumeRadiusScale);
            }

            if (_settings.volumePhotonType == PhotonMapSettings::VOLUME_BEAMS)
            {
                if (_settings.useGrid)
                    buildBeamGrid(tail, volumeRadiusScale);
                else
                    buildBeamBvh(tail, volumeRadiusScale);
            }

            else if (_settings.volumePhotonType == PhotonMapSettings::VOLUME_PLANES ||
                     _settings.volumePhotonType == PhotonMapSettings::VOLUME_PLANES_1D)
            {
                if (_settings.volumePhotonType == PhotonMapSettings::VOLUME_PLANES)
                {
                    _planes0D.reset(new PhotonPlane0D[tail]);
                    for (uint32 i = 0; i < tail; ++i)
                        _planes0D[i].valid = false;
                }
                if (_settings.volumePhotonType == PhotonMapSettings::VOLUME_PLANES_1D)
                {
                    _planes1D.reset(new PhotonPlane1D[tail]);
                    for (uint32 i = 0; i < tail; ++i)
                        _planes1D[i].valid = false;
                }

                if (_settings.useGrid)
                    buildPlaneGrid(tail, volumeRadiusScale);
                else
                    buildPlaneBvh(tail, volumeRadiusScale);
            }

            else if (_settings.volumePhotonType == PhotonMapSettings::VOLUME_PRIMITIVES)
            {
                prepare_PhotonPrimitives(tail, volumeRadiusScale);
            }

            else if (_settings.volumePhotonType == PhotonMapSettings::VOLUME_VOLUME)
            {
                _volumes.reset(new PhotonVolume[tail]);
                for (uint32 i = 0; i < tail; ++i)
                    _volumes[i].valid = false;

                if (_settings.useGrid)
                {
                }
                else
                {
                    Bvh::PrimVector debug_primBounds;
                    makePathBVH(tail, volumeRadiusScale, debug_primBounds, _settings.surf_EST);
                    _volumeBvh_DBG.reset(new Bvh::BinaryBvh(std::move(debug_primBounds), 1));

                    buildVolumeBvh(tail, volumeRadiusScale);
                }
            }

            _pathPhotonCount = tail;
        }
    }

    void PhotonMapIntegrator::fromJson(JsonPtr value, const Scene & /*scene*/)
    {
        _settings.fromJson(value);
    }

    rapidjson::Value PhotonMapIntegrator::toJson(Allocator &allocator) const
    {
        return _settings.toJson(allocator);
    }

    void PhotonMapIntegrator::prepareForRender(TraceableScene &scene, uint32 seed)
    {
        _sampler = UniformSampler(MathUtil::hash32(seed));
        _currentSpp = 0;
        _totalTracedSurfacePaths = 0;
        _totalTracedVolumePaths = 0;
        _totalTracedPaths = 0;
        _pathPhotonCount = 0;
        _scene = &scene;
        advanceSpp();
        scene.cam().requestColorBuffer();
        scene.cam().requestSplatBuffer();

        _useFrustumGrid = _settings.useFrustumGrid;
        if (_useFrustumGrid && !dynamic_cast<const PinholeCamera *>(&scene.cam()))
        {
            std::cout
                << "Warning: Frustum grid acceleration structure is only supported for a pinhole camera. "
                   "Frustum grid will be disabled for this render."
                << std::endl;
            _useFrustumGrid = false;
        }

        if (_settings.includeSurfaces)
            _surfacePhotons.resize(_settings.photonCount);
        if (!_scene->media().empty())
        {
            if (_settings.volumePhotonType == PhotonMapSettings::VOLUME_POINTS)
                _volumePhotons.resize(_settings.volumePhotonCount);
            else
                _pathPhotons.resize(_settings.volumePhotonCount);
        }

        int numThreads = ThreadUtils::pool->threadCount();
        for (int i = 0; i < numThreads; ++i)
        {
            uint32 surfaceRangeStart = intLerp(0, uint32(_surfacePhotons.size()), i + 0, numThreads);
            uint32 surfaceRangeEnd = intLerp(0, uint32(_surfacePhotons.size()), i + 1, numThreads);
            uint32 volumeRangeStart = intLerp(0, uint32(_settings.volumePhotonCount), i + 0, numThreads);
            uint32 volumeRangeEnd = intLerp(0, uint32(_settings.volumePhotonCount), i + 1, numThreads);
            _taskData.emplace_back(
                SubTaskData{SurfacePhotonRange(_surfacePhotons.empty() ? nullptr : &_surfacePhotons[0],
                                               surfaceRangeStart, surfaceRangeEnd),
                            VolumePhotonRange(_volumePhotons.empty() ? nullptr : &_volumePhotons[0],
                                              volumeRangeStart, volumeRangeEnd),
                            PathPhotonRange(_pathPhotons.empty() ? nullptr : &_pathPhotons[0],
                                            volumeRangeStart, volumeRangeEnd)});
            _samplers.emplace_back(_scene->rendererSettings().useSobol()
                                       ? std::unique_ptr<PathSampleGenerator>(
                                             new SobolPathSampler(MathUtil::hash32(_sampler.nextI())))
                                       : std::unique_ptr<PathSampleGenerator>(
                                             new UniformPathSampler(MathUtil::hash32(_sampler.nextI()))));

            _tracers.emplace_back(new PhotonTracer(&scene, _settings, i));
        }

        Vec2u res = _scene->cam().resolution();
        _w = res.x();
        _h = res.y();

        if (_useFrustumGrid)
            _depthBuffer.reset(new Ray[_w * _h]);

        diceTiles();
    }

    void PhotonMapIntegrator::teardownAfterRender()
    {
        _group.reset();
        _depthBuffer.reset();

        _beams.reset();
        _planes0D.reset();
        _planes1D.reset();
        _primitives.reset();

        _surfacePhotons.clear();
        _volumePhotons.clear();
        _pathPhotons.clear();
        _taskData.clear();
        _samplers.clear();
        _tracers.clear();

        _surfacePhotons.shrink_to_fit();
        _volumePhotons.shrink_to_fit();
        _pathPhotons.shrink_to_fit();
        _taskData.shrink_to_fit();
        _samplers.shrink_to_fit();
        _tracers.shrink_to_fit();

        _surfaceTree.reset();
        _volumeTree.reset();
        _volumeGrid.reset();
        _volumeBvh.reset();

        _volumeBvh_DBG.reset();
        _volumeGrid_DBG.reset();

        _volumeBvh_dangling.reset();

        _volumes.reset();
    }

    void PhotonMapIntegrator::renderSegment(std::function<void()> completionCallback)
    {
        using namespace std::placeholders;

        _scene->cam().setSplatWeight(1.0 / _nextSpp);

        if (!_surfaceTree)
        {
            ThreadUtils::pool->yield(*ThreadUtils::pool->enqueue(
                std::bind(&PhotonMapIntegrator::tracePhotons, this, _1, _2, _3, 0), _tracers.size(),
                []() {}));

            buildPhotonDataStructures(1.0f);
        }

        ThreadUtils::pool->yield(
            *ThreadUtils::pool->enqueue(std::bind(&PhotonMapIntegrator::tracePixels, this, _1, _3,
                                                  _settings.gatherRadius, _settings.volumeGatherRadius),
                                        _tiles.size(), []() {}));

        // no frustum grid for primitives for now
        if (_useFrustumGrid)
        {
            ThreadUtils::pool->yield(*ThreadUtils::pool->enqueue(
                [&](uint32 tracerId, uint32 numTracers, uint32)
                {
                    uint32 start = intLerp(0, _pathPhotonCount, tracerId, numTracers);
                    uint32 end = intLerp(0, _pathPhotonCount, tracerId + 1, numTracers);
                    _tracers[tracerId]->evalPrimaryRays(_beams.get(), _planes0D.get(), _planes1D.get(), start,
                                                        end, _settings.volumeGatherRadius, _depthBuffer.get(),
                                                        *_samplers[tracerId], _nextSpp - _currentSpp);
                },
                _tracers.size(), []() {}));
        }

        _currentSpp = _nextSpp;
        advanceSpp();

        completionCallback();
    }

    void PhotonMapIntegrator::startRender(std::function<void()> completionCallback)
    {
        if (done())
        {
            completionCallback();
            return;
        }

        _group = ThreadUtils::pool->enqueue([&, completionCallback](uint32, uint32, uint32)
                                            { renderSegment(completionCallback); }, 1, []() {});
    }

    void PhotonMapIntegrator::waitForCompletion()
    {
        if (_group)
        {
            _group->wait();
            _group.reset();
        }
    }

    void PhotonMapIntegrator::abortRender()
    {
        if (_group)
        {
            _group->abort();
            _group->wait();
            _group.reset();
        }
    }

    void PhotonMapIntegrator::precomputeVolume(PhotonVolume &volume, const PathPhoton &p0,
                                               const PathPhoton &p1, const PathPhoton &p2,
                                               const PathPhoton &p3)
    {
        Vec3f a = p0.dir * p0.sampledLength;
        Vec3f b = p1.dir * p1.sampledLength;
        Vec3f c = p2.dir * p2.sampledLength;
        // TODO: optimization: avoid normalizing determinant (like precomputePlane1D())
        float det = std::abs(a.dot(b.cross(c))) / (a.length() * b.length() * c.length());

        if (det < 1e-4f)
        {
            return;
        }

        float invDet = 1.0f / det;

        volume.p = p0.pos;
        volume.a = a;
        volume.b = b;
        volume.c = c;

        volume.aDir = p0.dir;
        volume.bDir = p1.dir;
        volume.cDir = p2.dir;
        volume.aLen = p0.sampledLength;
        volume.bLen = p1.sampledLength;
        volume.cLen = p2.sampledLength;

        volume.powerOverDet = p3.power * invDet;
        volume.bounce = int(p2.bounce());
        volume.valid = true;
    }

    void PhotonMapIntegrator::PhotonMapIntegrator::buildVolumeBvh(uint32 tail, float volumeRadiusScale)
    {
        Bvh::PrimVector volumes;
        for (uint32 i = 3; i < tail; ++i)
        {
            // TODO: support lowOrderScattering
            const PathPhoton &p0 = _pathPhotons[i - 3];
            const PathPhoton &p1 = _pathPhotons[i - 2];
            const PathPhoton &p2 = _pathPhotons[i - 1];
            const PathPhoton &p3 = _pathPhotons[i - 0];

            // TODO: check if these two tests are valid (considering cases w. surfaces)
            bool onSurface = p2.onSurface() || p1.onSurface();
            bool isSampledLengthValid = p1.sampledLength > 0.0f && p2.sampledLength > 0.0f;
            if (p3.bounce() > 2 && !onSurface && isSampledLengthValid)
            {
                precomputeVolume(_volumes[i], p0, p1, p2, p3);
                Box3f bounds = _volumes[i].bounds();
                volumes.emplace_back(Bvh::Primitive(bounds, bounds.center(), i));
            }
        }

        _volumeBvh.reset(new Bvh::BinaryBvh(std::move(volumes), 1));
    }

} // namespace Tungsten
