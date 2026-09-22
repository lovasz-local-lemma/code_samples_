// My research branch of Tungsten (Benedikt Bitterli). See CONTRIBUTION.md and LICENSE.txt.
// Formatting/comment cleanup only; host-renderer dependencies are not bundled.

#include "photon.hpp"

namespace Tungsten
{

    // statics of the old class, PhotonPrimitive
    Box3f PhotonPrimitive::BMAX = Box3f(Vec3f(0.0), Vec3f(0.0));
    PathPhoton *PhotonPrimitive::RAW_ARRAY = nullptr;
    bool PhotonPrimitive::GLOBALS_READY = false;
    PhotonMapSettings *PhotonPrimitive::SETTINGS = nullptr;
    PhotonMapSettings::Primitive_Policy *PhotonPrimitive::EST = nullptr;

    // statics of the new class, PP_PRIM
    Box3f PP_PRIM::BMAX = Box3f(Vec3f(0.0), Vec3f(0.0));
    Box3f PP_PRIM::BSCENE = Box3f(Vec3f(0.0), Vec3f(0.0));

    PP *PP_PRIM::RAW_ARRAY = nullptr;
    bool PP_PRIM::GLOBALS_READY = false;
    PhotonPrimitiveSettings *PP_PRIM::SETTINGS = nullptr;
    UniformPathSampler PP_PRIM::smp = UniformPathSampler(0);

    // primitive type is hard-coded for now
    PP_PRIM::WIP_diffprim PP_PRIM::myprim = PP_PRIM::WIP_diffprim::diff_uni_photon;

    void PhotonPrimitive::initialize_globals(PathPhoton *headpt, const Box3f &scenebox,
                                             PhotonMapSettings *P_setting)
    {
        RAW_ARRAY = headpt;
        BMAX = scenebox;
        SETTINGS = P_setting;
        EST = &(P_setting->surf_EST);
    }

    // TODO: not called yet
    void PP_PRIM::initialize_globals(PP *headpt, const Box3f &scenebox, const Box3f &infbox,
                                     PhotonPrimitiveSettings *PP_settings)
    {
        prtln "INITIALIZING PP GLOBALS";
        prtln "PRIMTYPE: hard-coded diff_uni_photon";

        GLOBALS_READY = true;

        BMAX = infbox;

        SETTINGS = PP_settings;
        RAW_ARRAY = headpt;
    }

    // Summarize the bounce requirement for a primitive combination.
    // Assumes enough dangling segments already exist and the processed edges are already stored.
    void PhotonPrimitive::precompute(int which_strat, int which_prim, int I_curbounce)
    {
        set_invalid();

        strat_prim[0] = which_strat;
        strat_prim[1] = which_prim;

        I_mybounce = I_curbounce;

        PhotonMapSettings::Primitive_Combination &cur_comb = EST->io_strategies()[which_strat];
        PhotonMapSettings::Primitive_Description &cur_surf = cur_comb.io_primitives()[which_prim];

        // segment counts; for now the strategy length is always the one used
        int segnum_strat = cur_comb.seg_involved();
        int segnum_prim = cur_surf.o_seg_involved();

        // some EST needs extra dimensions added first
        int extended_dim = segnum_strat - segnum_prim;

        this->prep_size(segnum_strat + 1);

        // went past the head for some reason; bail out
        if (segnum_strat > I_mybounce)
            return;

        int32 pos_prim_start = I_mybounce - segnum_strat;

        I_mis = pos_prim_start;

        bounce = RAW_ARRAY[I_curbounce].bounce();
        bounce_SS = RAW_ARRAY[I_curbounce].bounceSS();

        int32 LTpos = -1;

        // find related LT
        for (int j = pos_prim_start; j < RAW_ARRAY[I_curbounce].extra.data_bracketR_far; j++)
        {
            if (RAW_ARRAY[j].extra.profile.is_on(PathPhoton::extra_feature_blob::EF_LT))
            {
                LTpos = j;
                break;
            }
        }

        I_LT = LTpos;

        processed_origpath.clear();

        // even when extended, pos_prim_start will never be negative
        {
            // has dangling segments
            if (RAW_ARRAY[I_mybounce].extra.dangling_index >= 0)
            {
                arr<PathPhoton> &pts = RAW_ARRAY[RAW_ARRAY[I_mybounce].extra.dangling_index].extra.danglings;
                for (int ix = 0; ix < pts.size(); ix++)
                    processed_origpath.emplace_back(pts[ix].pos);
            }
            else
            {
                for (int ix = pos_prim_start; ix <= I_mybounce; ix++)
                    processed_origpath.emplace_back(RAW_ARRAY[ix].pos);
            }
        }

        const int_set &affected = cur_surf.o_affected_bounces_rel();

        for (auto curpos : affected)
        {
            P.emplace_back(processed_origpath[segnum_strat + curpos]);
            V.emplace_back(processed_origpath[segnum_strat + curpos + 1] -
                           processed_origpath[segnum_strat + curpos]);
            I_vtx.emplace_back(I_mybounce + curpos);
        }

        P.emplace_back(RAW_ARRAY[I_mybounce].pos);

        for (int i = 0; i < cur_surf.o_affected_bounces_rel().size(); i++)
        {
            D.emplace_back(V[i].normalized());
            L.emplace_back(V[i].length());
        }

        power = RAW_ARRAY[I_vtx[0]].power;

        set_origin();
        set_valid();

        _bbox = bounds();
    }

} // namespace Tungsten
