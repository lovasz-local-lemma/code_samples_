// My research branch of Tungsten (Benedikt Bitterli). See CONTRIBUTION.md and LICENSE.txt.
// Formatting/comment cleanup only; host-renderer dependencies are not bundled.

#ifndef PATHTRACERSETTINGS_HPP_
#define PATHTRACERSETTINGS_HPP_

#include "integrators/TraceSettings.hpp"

#include "io/JsonObject.hpp"

#include <iostream>
#include <vector>

#include "StringableEnum.hpp"

namespace Tungsten
{

#define br "\n"
#define varName(Vname) (str(#Vname))
#define varNameC(Vname) (#Vname)
#define fetch(XX) value.getField(varNameC(XX), XX)
#define def_if_no(XX, val)  if(!fetch(XX)) XX=val

#ifndef puts
#define puts std::cout<<
#endif

#ifndef iss
    using str = std::string;
    using ost = std::ostream;

    template <class T> using arr = std::vector<T>;
    using iss = std::istringstream;
#endif

    struct PathTracerSettings : public TraceSettings
    {
        bool enableLightSampling;
        bool enableVolumeLightSampling;
        bool lowOrderScattering;
        bool includeSurfaces;

        PathTracerSettings()
            : enableLightSampling(true), enableVolumeLightSampling(true), lowOrderScattering(true),
              includeSurfaces(true)
        {
        }

        void fromJson(JsonPtr value)
        {
            TraceSettings::fromJson(value);
            value.getField("enable_light_sampling", enableLightSampling);
            value.getField("enable_volume_light_sampling", enableVolumeLightSampling);
            value.getField("low_order_scattering", lowOrderScattering);
            value.getField("include_surfaces", includeSurfaces);
        }

        rapidjson::Value toJson(rapidjson::Document::AllocatorType &allocator) const
        {
            return JsonObject{TraceSettings::toJson(allocator),
                              allocator,
                              "type",
                              "path_tracer",
                              "enable_light_sampling",
                              enableLightSampling,
                              "enable_volume_light_sampling",
                              enableVolumeLightSampling,
                              "low_order_scattering",
                              lowOrderScattering,
                              "include_surfaces",
                              includeSurfaces};
        }
    };

    enum A_ter
    {
        A_dist,
        A_theta,
        A_phi
    };

    //surf type
    enum S_typ
    {
        S_plane = 0,
        S_cone = 1,
        S_cylinder = 2,
        S_sphere = 3,
        S_disk,
        S_torus,
        S_hyperb,
        S_Q,
        S_unknown
    };

    enum C_typ
    {
        C_point = 0,
        C_beam = 1,
        C_unknown = 65535
    };

}

#endif /* PATHTRACERSETTINGS_HPP_ */