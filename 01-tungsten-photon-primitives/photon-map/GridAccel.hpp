// My research branch of Tungsten (Benedikt Bitterli). See CONTRIBUTION.md and LICENSE.txt.
// Formatting/comment cleanup only; host-renderer dependencies are not bundled.

#ifndef GRIDACCEL_HPP_
#define GRIDACCEL_HPP_

#include "thread/ThreadUtils.hpp"
#include "thread/ThreadPool.hpp"
#include "Timer.hpp"

#include <tribox/tribox.hpp>
#include <atomic>
#include <cmath>

namespace Tungsten
{

    class GridAccel
    {
      public:
        struct Primitive
        {
            uint32 idx;
            Vec3f p0, p1, p2, p3;
            float r;

            Primitive() = default;
            Primitive(uint32 idx_, Vec3f p0_, Vec3f p1_, Vec3f p2_, Vec3f p3_, float r_, bool beam)
                : idx(beam ? idx_ | 0x80000000u : idx_), p0(p0_), p1(p1_), p2(p2_), p3(p3_), r(r_)
            {
            }

            bool isBeam()
            {
                return idx & 0x80000000u;
            }
        };

        GridAccel(Box3f max_bounds, int memBudgetKb, arr<PP_PRIM *> &surfs)
        {
            Timer timer;

            Vec3f diag = max_bounds.diagonal();
            Vec3f relDiag = diag / diag.max();
            float maxCells = std::cbrt(double(int64(memBudgetKb) << 10) / (4.0 * relDiag.product()));
            _sizes = max(Vec3i(relDiag * maxCells), Vec3i(1));
            _offset = max_bounds.min();
            _scale = Vec3f(_sizes) / diag;
            _invScale = 1.0f / _scale;
            _fSizes = Vec3f(_sizes);

            _yStride = _sizes.x();
            _zStride = _sizes.x() * _sizes.y();

            _cellCount = _zStride * _sizes.z();

            std::cout << "Building grid accelerator with bounds " << max_bounds << " and size " << _sizes
                      << " (" << (_sizes.product() * 4) / (1024 * 1024) << "mb)" << std::endl;

            timer.bench("Initialization");

            buildAccel_primitives(std::move(surfs));
        }

      private:
        std::unique_ptr<std::atomic<uint32>[]> _atomicListOffsets;
        const uint32 *_listOffsets;
        std::unique_ptr<uint32[]> _lists;
        Vec3f _offset;
        Vec3f _scale;
        Vec3f _invScale;
        Vec3i _sizes;
        Vec3f _fSizes;

        int64 _yStride;
        int64 _zStride;
        uint64 _cellCount;

        uint64 idx(int x, int y, int z) const
        {
            return x + y * _yStride + z * _zStride;
        }

        template <typename LoopBody> void iterateBounds(Box3f bounds, LoopBody body)
        {
            Vec3i minI = max(Vec3i(bounds.min()), Vec3i(0));
            Vec3i maxI = min(Vec3i(bounds.max()), _sizes - 1);

            for (int z = minI.z(); z <= maxI.z(); ++z)
                for (int y = minI.y(); y <= maxI.y(); ++y)
                    for (int x = minI.x(); x <= maxI.x(); ++x)
                        body(x, y, z);
        }

        template <typename LoopBody>
        void iterateTrapezoid(Vec3f p0, Vec3f p1, Vec3f p2, Vec3f p3, float r, LoopBody body)
        {
            Vec3f radius = r * _scale;
            p0 = (p0 - _offset) * _scale;
            p1 = (p1 - _offset) * _scale;
            p2 = (p2 - _offset) * _scale;
            p3 = (p3 - _offset) * _scale;
            float vertsA[3][3] = {
                {p0.x(), p0.y(), p0.z()},
                {p1.x(), p1.y(), p1.z()},
                {p2.x(), p2.y(), p2.z()},
            };
            float vertsB[3][3] = {
                {p0.x(), p0.y(), p0.z()},
                {p2.x(), p2.y(), p2.z()},
                {p3.x(), p3.y(), p3.z()},
            };
            Box3f bounds;
            bounds.grow(p0 + radius);
            bounds.grow(p0 - radius);
            bounds.grow(p1 + radius);
            bounds.grow(p1 - radius);
            bounds.grow(p2 + radius);
            bounds.grow(p2 - radius);
            bounds.grow(p3 + radius);
            bounds.grow(p3 - radius);
            iterateBounds(bounds,
                          [&](int x, int y, int z)
                          {
                              Vec3f boxCenter = Vec3f(Vec3i(x, y, z)) + 0.5f;
                              Vec3f boxHalfSize(0.5f + radius);
                              if (triBoxOverlap(boxCenter.data(), boxHalfSize.data(), vertsA) ||
                                  triBoxOverlap(boxCenter.data(), boxHalfSize.data(), vertsB))
                                  body(x, y, z);
                          });
        }

        template <typename LoopBody> void iterateBeam(Vec3f p0, Vec3f p1, float r, LoopBody body)
        {
            Vec3f radius = r * _scale;
            Box3f bounds;
            bounds.grow((p0 - _offset) * _scale + radius);
            bounds.grow((p0 - _offset) * _scale - radius);
            bounds.grow((p1 - _offset) * _scale + radius);
            bounds.grow((p1 - _offset) * _scale - radius);

            Vec3f d = p1 - p0;
            Vec3f invD = 1.0f / d;
            Vec3f coordScale = _invScale * invD;

            Vec3f relMin(-r + _offset - p0);
            Vec3f relMax((_invScale + r) + _offset - p0);
            Vec3f tMins, tMaxs;
            for (int i = 0; i < 3; ++i)
            {
                if (invD[i] >= 0.0f)
                {
                    tMins[i] = relMin[i] * invD[i];
                    tMaxs[i] = relMax[i] * invD[i];
                }
                else
                {
                    tMins[i] = relMax[i] * invD[i];
                    tMaxs[i] = relMin[i] * invD[i];
                }
            }

            iterateBounds(bounds,
                          [&](int x, int y, int z)
                          {
                              Vec3f boxTs = Vec3f(Vec3i(x, y, z)) * coordScale;
                              float tMin = max((tMins + boxTs).max(), 0.0f);
                              float tMax = min((tMaxs + boxTs).min(), 1.0f);
                              if (tMin <= tMax)
                                  body(x, y, z);
                          });
        }

        template <typename LoopBody> void iteratePrimitive(PP_PRIM &pf, LoopBody body)
        {
            PrimEnum pt = pf.cache.ptype;
            Box3f bounds;
            // TODO: cache the bbox
            Box3f bounds_raw = pf.bbox();

            bounds.grow((bounds_raw.min() - _offset) * _scale);
            bounds.grow((bounds_raw.max() - _offset) * _scale);

            int prim_segs = pf.cache.prim->o_seg_involved();

            arr<Vec3f> orig_vct;
            orig_vct.emplace_back((pf.orig - _offset) * _scale);

            for (int ii = 0; ii < prim_segs; ii++)
            {
                orig_vct.emplace_back(pf.PSP_geo.V[ii] * _scale);
            }

            // TODO: BMAX might be unreliable; the scene box seems to include only the surfaces
            float scenelen = (pf.BMAX.max() - pf.BMAX.min()).length();

            Vec3f boxHalfSize(0.5f);
            float sqrt3 = sqrtf(3.01f);

            switch (pt)
            {
            case PrimEnum::PRIM2_PLANE:
            {
                Vec3f radius(0.0f);
                Vec3f p0 = (orig_vct[0]);
                Vec3f p1 = (orig_vct[0] + orig_vct[1]);
                Vec3f p2 = (orig_vct[0] + orig_vct[1] + orig_vct[2]);
                Vec3f p3 = (orig_vct[0] + orig_vct[2]);

                float vertsA[3][3] = {
                    {p0.x(), p0.y(), p0.z()},
                    {p1.x(), p1.y(), p1.z()},
                    {p2.x(), p2.y(), p2.z()},
                };
                float vertsB[3][3] = {
                    {p0.x(), p0.y(), p0.z()},
                    {p2.x(), p2.y(), p2.z()},
                    {p3.x(), p3.y(), p3.z()},
                };

                iterateBounds(bounds,
                              [&](int x, int y, int z)
                              {
                                  Vec3f boxCenter = Vec3f(Vec3i(x, y, z)) + 0.5f;

                                  if (triBoxOverlap(boxCenter.data(), boxHalfSize.data(), vertsA) ||
                                      triBoxOverlap(boxCenter.data(), boxHalfSize.data(), vertsB))
                                      body(x, y, z);
                              });

                break;
            }
            case PrimEnum::PRIM3_PARALLELEPIPED:
            {
                Vec3f radius(0.0f);
                Vec3f p0 = (orig_vct[0]);
                Vec3f p1 = (p0 + orig_vct[1]);
                Vec3f p2 = (p1 + orig_vct[2]);
                Vec3f p3 = (p0 + orig_vct[2]);

                Vec3f p4 = (orig_vct[0] + orig_vct[3]);
                Vec3f p5 = (p4 + orig_vct[1]);
                Vec3f p6 = (p5 + orig_vct[2]);
                Vec3f p7 = (p4 + orig_vct[2]);

                auto AddQuad = [&](Vec3f &px, Vec3f &py, Vec3f &pz, Vec3f &pw)
                {
                    float vertsA[3][3] = {
                        {px.x(), px.y(), px.z()},
                        {py.x(), py.y(), py.z()},
                        {pz.x(), pz.y(), pz.z()},
                    };
                    float vertsB[3][3] = {
                        {px.x(), px.y(), px.z()},
                        {pz.x(), pz.y(), pz.z()},
                        {pw.x(), pw.y(), pw.z()},
                    };

                    iterateBounds(bounds,
                                  [&](int x, int y, int z)
                                  {
                                      Vec3f boxCenter = Vec3f(Vec3i(x, y, z)) + 0.5f;

                                      if (triBoxOverlap(boxCenter.data(), boxHalfSize.data(), vertsA) ||
                                          triBoxOverlap(boxCenter.data(), boxHalfSize.data(), vertsB))
                                          body(x, y, z);
                                  });
                };

                AddQuad(p0, p1, p2, p3);
                AddQuad(p4, p5, p6, p7);
                AddQuad(p0, p4, p5, p1);
                AddQuad(p3, p7, p6, p2);
                AddQuad(p0, p4, p7, p3);
                AddQuad(p1, p5, p6, p2);

                break;
            }

            case PrimEnum::PRIM2_SPHERE:
            {
                iterateBounds(bounds,
                              [&](int x, int y, int z)
                              {
                                  Vec3f boxCenter = Vec3f(Vec3i(x, y, z)) + 0.5f;
                                  float Rmin = -0.5 * sqrt3 + orig_vct[1].length();
                                  float Rmax = 0.5 * sqrt3 + orig_vct[1].length();

                                  float RRR = (boxCenter - orig_vct[0]).length();

                                  if ((RRR < Rmax) && (RRR > Rmin))
                                  {
                                      body(x, y, z);
                                  }
                              });
                break;
            }
            case PrimEnum::PRIM3_SSPHERE:
            {
                iterateBounds(bounds,
                              [&](int x, int y, int z)
                              {
                                  Vec3f boxCenter = Vec3f(Vec3i(x, y, z)) + 0.5f;
                                  float Rmax = 0.5 * sqrt3 + orig_vct[1].length();

                                  float RRR = (boxCenter - orig_vct[0]).length();

                                  if ((RRR < Rmax))
                                  {
                                      body(x, y, z);
                                  }
                              });
                break;
            }
            case PrimEnum::PRIM2_CONE:
            {
                Vec3f axisdir = pf.cache.dprev;
                float coss_ref = orig_vct[1].normalized().dot(axisdir);

                iterateBounds(bounds,
                              [&](int x, int y, int z)
                              {
                                  Vec3f boxCenter = Vec3f(Vec3i(x, y, z)) + 0.5f;
                                  Vec3f vt = boxCenter - orig_vct[0];

                                  float coss = vt.normalized().dot(axisdir);
                                  float len = vt.length();

                                  float len_range = vt.dot(axisdir);

                                  if ((sgn(coss_ref) == sgn(coss)) || (abs(len_range - len) < 0.5f * sqrt3) ||
                                      (abs(len_range) < 0.5f * sqrt3))
                                  {

                                      float htmin = abs(vt.dot(axisdir)) - 0.5f * sqrt3;
                                      float htmax = abs(vt.dot(axisdir)) + 0.5f * sqrt3;

                                      float len_ref_min = htmin / abs(coss_ref);
                                      float len_ref_max = htmax / abs(coss_ref);

                                      if ((len_ref_min <= len) && (len_ref_max >= len))
                                          body(x, y, z);
                                  }
                              }

                );

                break;
            }
            }
        }

        void buildAccel_primitives(arr<PP_PRIM *> &surfaces2eval)
        {
            _atomicListOffsets = zeroAlloc<std::atomic<uint32>>(_cellCount + 1);

            ThreadUtils::parallelFor(0, surfaces2eval.size(), ThreadUtils::pool->threadCount() + 1,
                                     [&](uint32 i)
                                     {
                                         iteratePrimitive(*surfaces2eval[i], [&](int x, int y, int z)
                                                          { _atomicListOffsets[idx(x, y, z)]++; });
                                     });

            uint32 prefixSum = 0;
            for (uint64 i = 0; i <= _cellCount; ++i)
            {
                prefixSum += _atomicListOffsets[i];
                _atomicListOffsets[i] = prefixSum;
            }

            _lists.reset(new uint32[prefixSum]);

            ThreadUtils::parallelFor(
                0, surfaces2eval.size(), ThreadUtils::pool->threadCount() + 1,
                [&](uint32 i)
                {
                    iteratePrimitive(
                        *surfaces2eval[i], [&](int x, int y, int z)
                        { _lists[--_atomicListOffsets[idx(x, y, z)]] = surfaces2eval[i]->index_on_array; });
                });

            _listOffsets = reinterpret_cast<const uint32 *>(&_atomicListOffsets[0]);
        }

        void buildAccel(std::vector<Primitive> prims)
        {
            _atomicListOffsets = zeroAlloc<std::atomic<uint32>>(_cellCount + 1);

            ThreadUtils::parallelFor(
                0, prims.size(), ThreadUtils::pool->threadCount() + 1,
                [&](uint32 i)
                {
                    if (prims[i].isBeam())
                    {
                        iterateBeam(prims[i].p0, prims[i].p1, prims[i].r,
                                    [&](int x, int y, int z) { _atomicListOffsets[idx(x, y, z)]++; });
                    }
                    else
                    {
                        iterateTrapezoid(prims[i].p0, prims[i].p1, prims[i].p2, prims[i].p3, prims[i].r,
                                         [&](int x, int y, int z) { _atomicListOffsets[idx(x, y, z)]++; });
                    }
                });

            uint32 prefixSum = 0;
            for (uint64 i = 0; i <= _cellCount; ++i)
            {
                prefixSum += _atomicListOffsets[i];
                _atomicListOffsets[i] = prefixSum;
            }

            _lists.reset(new uint32[prefixSum]);

            ThreadUtils::parallelFor(
                0, prims.size(), ThreadUtils::pool->threadCount() + 1,
                [&](uint32 i)
                {
                    if (prims[i].isBeam())
                    {
                        iterateBeam(
                            prims[i].p0, prims[i].p1, prims[i].r, [&](int x, int y, int z)
                            { _lists[--_atomicListOffsets[idx(x, y, z)]] = prims[i].idx & 0x7FFFFFFFu; });
                    }
                    else
                    {
                        iterateTrapezoid(prims[i].p0, prims[i].p1, prims[i].p2, prims[i].p3, prims[i].r,
                                         [&](int x, int y, int z)
                                         { _lists[--_atomicListOffsets[idx(x, y, z)]] = prims[i].idx; });
                    }
                });

            _listOffsets = reinterpret_cast<const uint32 *>(&_atomicListOffsets[0]);
        }

      public:
        GridAccel(Box3f bounds, int memBudgetKb, std::vector<Primitive> prims)
        {
            Timer timer;

            Vec3f diag = bounds.diagonal();
            Vec3f relDiag = diag / diag.max();
            float maxCells = std::cbrt(double(int64(memBudgetKb) << 10) / (4.0 * relDiag.product()));
            _sizes = max(Vec3i(relDiag * maxCells), Vec3i(1));
            _offset = bounds.min();
            _scale = Vec3f(_sizes) / diag;
            _invScale = 1.0f / _scale;
            _fSizes = Vec3f(_sizes);

            _yStride = _sizes.x();
            _zStride = _sizes.x() * _sizes.y();

            _cellCount = _zStride * _sizes.z();

            std::cout << "Building grid accelerator with bounds " << bounds << " and size " << _sizes << " ("
                      << (_sizes.product() * 4) / (1024 * 1024) << "mb)" << std::endl;

            timer.bench("Initialization");

            buildAccel(std::move(prims));
        }

        template <typename Iterator> void trace(Ray ray, Iterator iterator) const
        {
            Vec3f o = (ray.pos() - _offset) * _scale;
            Vec3f d = ray.dir() * _scale;

            Vec3f relMin = -o;
            Vec3f relMax = _fSizes - o;

            float tMin = ray.nearT(), tMax = ray.farT();
            Vec3f tMins;
            Vec3f tStep = 1.0f / d;
            for (int i = 0; i < 3; ++i)
            {
                if (d[i] >= 0.0f)
                {
                    tMins[i] = relMin[i] * tStep[i];
                    tMin = max(tMin, tMins[i]);
                    tMax = min(tMax, relMax[i] * tStep[i]);
                }
                else
                {
                    tMins[i] = relMax[i] * tStep[i];
                    tMin = max(tMin, tMins[i]);
                    tMax = min(tMax, relMin[i] * tStep[i]);
                }
            }
            if (tMin >= tMax)
                return;

            tStep = std::abs(tStep);

            Vec3f p = o + d * tMin;
            Vec3f nextT;
            Vec3i iStep, iP;
            for (int i = 0; i < 3; ++i)
            {
                if (d[i] >= 0.0f)
                {
                    iP[i] = max(int(p[i]), 0);
                    nextT[i] = tMin + (float(iP[i] + 1) - p[i]) * tStep[i];
                    iStep[i] = 1;
                }
                else
                {
                    iP[i] = min(int(p[i]), _sizes[i] - 1);
                    nextT[i] = tMin + (p[i] - float(iP[i])) * tStep[i];
                    iStep[i] = -1;
                }
            }

            while (tMin < tMax)
            {
                int minIdx = nextT.minDim();
                float cellTmax = nextT[minIdx];

                uint64 i = idx(iP.x(), iP.y(), iP.z());
                uint32 listStart = _listOffsets[i];
                uint32 count = _listOffsets[i + 1] - listStart;
                if (count)
                    for (uint32 t = 0; t < count; ++t)
                        iterator(_lists[listStart + t], tMin, min(cellTmax, tMax));

                tMin = cellTmax;
                nextT[minIdx] += tStep[minIdx];
                iP[minIdx] += iStep[minIdx];

                if (iP[minIdx] < 0 || iP[minIdx] >= _sizes[minIdx])
                    return;
            }
        }
    };

} // namespace Tungsten

#endif /* GRIDACCEL_HPP_ */
