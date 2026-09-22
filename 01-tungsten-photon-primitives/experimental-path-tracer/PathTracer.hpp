// My research branch of Tungsten (Benedikt Bitterli). See CONTRIBUTION.md and LICENSE.txt.
// Formatting/comment cleanup only; host-renderer dependencies are not bundled.

#ifndef PATHTRACER_HPP_
#define PATHTRACER_HPP_

#include "PathTracerSettings.hpp"

#include "integrators/TraceBase.hpp"

namespace Tungsten
{

    class PathTracer : public TraceBase
    {
        PathTracerSettings _settings;
        bool _trackOutputValues;

      public:
        Vec3f routine(int which, Vec2u pixel, PathSampleGenerator &sampler, Ray &LS_M, Ray &M_C,
                      bool &path_unclear);

        Vec3f traceSample2(Vec2u pixel, PathSampleGenerator &sampler);

        Vec3f reeval(int which, PathSampleGenerator &sampler, Ray &LS_M, Ray &M_C, PositionSample &_LS);

        PathTracer(TraceableScene *scene, const PathTracerSettings &settings, uint32 threadId);

        Vec3f traceSample(Vec2u pixel, PathSampleGenerator &sampler);
    };

    class PathTracerX : public TraceBase
    {
        PathTracerSettings _settings;
        bool _trackOutputValues;

      public:
        PathTracerX(TraceableScene *scene, const PathTracerSettings &settings, uint32 threadId);

        Vec3f traceSample(Vec2u pixel, PathSampleGenerator &sampler);
    };

} // namespace Tungsten

#endif /* PATHTRACER_HPP_ */
