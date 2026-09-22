// Excerpt: RadianceLab/shaders/pathTrace.comp, lines 2594-2802 of 4327. Forward/reverse area-measure pdfs for assembled bidirectional paths (GGX with folded preimage, HG phase, medium transmittance, full-film camera density) feeding the log-space MIS weights.
// Not a standalone translation unit; see the folder README for the surrounding types.

float bsdfPdfW(int matId, vec3 nrm, vec3 wo, vec3 wi) {
    if (matId < 0 || matId >= uNumMaterials) return 0.0;
    int bt = int(materials[matId].emissionStrength.z);
    float rough = materials[matId].albedoRoughness.w;
    bool spec = (bt==6||bt==7||bt==8||bt==10) || (rough < 0.001 && (bt==1||bt==2||bt==5||bt==11||bt==13));
    if (spec) return 0.0;                                  // delta -> skipped in MIS
    float c = max(dot(nrm, wi), 0.0);
    if (c <= 0.0) return 0.0;
    if (bt==1||bt==5||bt==11) {
        // The sampler reflects a GGX half vector, then folds below-surface
        // directions back above the surface. Both preimages contribute density.
        vec3 h=normalize(wo+wi);
        float p=ggxPdf(max(abs(dot(nrm,h)),0.0),rough)/max(4.0*abs(dot(wo,h)),1e-20);
        vec3 folded=reflect(wi,nrm), hf=normalize(wo+folded);
        p+=ggxPdf(max(abs(dot(nrm,hf)),0.0),rough)/max(4.0*abs(dot(wo,hf)),1e-20);
        return p;
    }
    return c / 3.14159265;                                 // cosine
}

// Method B is active only with a non-flat scheme, method B selected, AND the media
// showcase off (mirrors the C++ feature mask that compiled this block in).
bool bdptUseB() { return uBDPTMISScheme != 0 && uBdptMISMethod == 1 && (uBdptVolume == 0 || bdptHomogeneous()); }

// Assembled connection path. x[0] = camera, x[gBn-1] = light emitter.
#define BDPT_MAXP 18
vec3  gBpos[BDPT_MAXP];
vec3  gBnrm[BDPT_MAXP];
int   gBmat[BDPT_MAXP];
bool  gBspec[BDPT_MAXP];
bool gBmedium[BDPT_MAXP];
float gBphase[BDPT_MAXP];
int   gBn;            // number of vertices in the assembled path
float gBoriginPdf;    // light-origin AREA pdf of x[gBn-1] (areaPdf*selectPdf, consistent)
int gBoriginPrim;

// Forward (camera-ward) area pdf of x[k] sampled from x[k-1].  k = 1 .. gBn-1.
float bdptPdfA(int k) {
    vec3 d = gBpos[k] - gBpos[k-1];
    float l2 = max(dot(d, d), 1e-12);
    vec3 dir = d * inversesqrt(l2);
    float pW;
    if (k == 1) {
        // Full-film camera density: LT samples are normalized across all pixels.
        // Using pixel area here would suppress LT by the film's pixel count.
        float cosCam = max(dot(dir, uCamForward), 1e-4);
        float filmArea = 4.0*uAspectRatio*uFovTanHalf*uFovTanHalf;
        pW = 1.0 / max(filmArea * cosCam * cosCam * cosCam, 1e-20);
    } else {
        pW = gBmedium[k-1] ? evalHGPhase(dot(normalize(gBpos[k-1]-gBpos[k-2]),dir),gBphase[k-1]) : bsdfPdfW(gBmat[k-1], gBnrm[k-1], normalize(gBpos[k-2]-gBpos[k-1]), dir);
    }
    return pW * (gBmedium[k]?bdptSigmaT(gBpos[k]):max(abs(dot(gBnrm[k],dir)),1e-4)) / l2 * (bdptHomogeneous()?evalVolumeTr(gBpos[k-1],dir,sqrt(l2)).x:1.0);
}

// Reverse (light-ward) area pdf of x[k] sampled from x[k+1].  k = 0 .. gBn-2;
// k == gBn-1 returns the stored light-origin pdf (used by the s=0 transition).
float bdptPdfB(int k) {
    if (k >= gBn - 1) return gBoriginPdf;
    vec3 d = gBpos[k] - gBpos[k+1];
    float l2 = max(dot(d, d), 1e-12);
    vec3 dir = d * inversesqrt(l2);
    float pW;
    if (k + 1 == gBn - 1) {
        // emitter x[gBn-1] emits toward x[k]: cosine-weighted emission direction pdf
        float cosE = max(dot(gBnrm[gBn-1], dir), 1e-4);
        pW = cosE / 3.14159265;
    } else {
        pW = gBmedium[k+1] ? evalHGPhase(dot(normalize(gBpos[k+1]-gBpos[k+2]),dir),gBphase[k+1]) : bsdfPdfW(gBmat[k+1], gBnrm[k+1], normalize(gBpos[k+2]-gBpos[k+1]), dir);
    }
    return pW * (gBmedium[k]?bdptSigmaT(gBpos[k]):max(abs(dot(gBnrm[k],dir)),1e-4)) / l2 * (bdptHomogeneous()?evalVolumeTr(gBpos[k+1],dir,sqrt(l2)).x:1.0);
}

// Balance/power MIS weight for the strategy that uses t camera-side vertices (s = gBn - t).
// Build prefix/suffix densities once per connection, with stable normalization.
#include "modules/pt_bdpt_weights.glsl"
float bdptNEEAreaPdf(int primIdx,vec3 receiver,vec3 lp,vec3 ln);
float bdptWeightB(int t) {
    float a[BDPT_MAXP],b[BDPT_MAXP];
    for(int k=1;k<gBn;++k) a[k]=bdptPdfA(k);
    for(int k=1;k<gBn-1;++k) b[k]=bdptPdfB(k);
    float nee=bdptNEEAreaPdf(gBoriginPrim,gBpos[gBn-2],gBpos[gBn-1],gBnrm[gBn-1]);
    return bdptWeightFromDensities(gBn,t,a,b,gBspec,gBoriginPdf,nee,uBDPTMISScheme,float(max(uLTPathsPerPixel,1)));
}

// Match sampleLight's conditional distribution: light selection and the
// sphere's visible-hemisphere / solid-angle sampling differ from emission.
float bdptNEEAreaPdf(int primIdx,vec3 receiver,vec3 lp,vec3 ln) {
    if(primIdx<0||primIdx>=uNumPrimitives) return 0.0;
    float total=0, chosen=0, area=0; int count=0;
    for(int i=0;i<uNumPrimitives;++i) {
        int type=int(prims[i].matIdType.y), mi=int(prims[i].matIdType.x);
        if((type!=0&&type!=1)||mi<0||mi>=uNumMaterials||materials[mi].emissionStrength.x<=.01) continue;
        float r=prims[i].posRadius.w, z=prims[i].matIdType.w;
        if(z<.001) z=prims[i].matIdType.z;
        float ar=type==0?4.0*3.14159*r*r:4.0*(r<.001?1.0:r)*(z<.001?1.0:z);
        float weight=uLightSelect==0?1.0:ar;
        if(uLightSelect==2) weight*=max(dot(materials[mi].emissionMetallic.xyz,vec3(.2126,.7152,.0722))*materials[mi].emissionStrength.x,.01);
        total+=weight; ++count;
        if(i==primIdx) { chosen=weight; area=ar; }
    }
    if(chosen<=0||area<=0) return 0;
    float selection=uLightSelect>=1&&total<=.001?1.0/float(count):chosen/total;
    vec3 d=lp-receiver; float d2=dot(d,d); float cosine=dot(ln,-normalize(d));
    if(cosine<=0) return 0;
    if(int(prims[primIdx].matIdType.y)==0) {
        float r=prims[primIdx].posRadius.w;
        vec3 toCenter=prims[primIdx].posRadius.xyz-receiver; float dc2=dot(toCenter,toCenter);
        if(uSphereCone>0&&dc2>r*r*1.0001) {
            float cmax=sqrt(max(0.0,1.0-r*r/dc2));
            return selection*cosine/(max(6.283185*(1-cmax),1e-7)*d2);
        }
        if(dot(ln,normalize(toCenter))>0) return 0;
        return selection*2.0/area;
    }
    return selection/area;
}

// Consistent light-origin AREA pdf for a light primitive (full-area pdf * uniform select),
// matching traceLightPath. Unsampled emitter types must have zero density.
float bdptLightOriginPdfPrim(int primIdx) {
    if (primIdx < 0 || primIdx >= uNumPrimitives) return 0.0;
    int pt = int(prims[primIdx].matIdType.y);
    if(pt!=0 && pt!=1) return 0.0;
    float areaPdf;
    if (pt == 0) {
        float r = prims[primIdx].posRadius.w;
        areaPdf = 1.0 / (4.0 * 3.14159 * r * r);
    } else {
        float ex = prims[primIdx].posRadius.w;
        float ez = prims[primIdx].matIdType.w;
        if (ez < 0.001) ez = prims[primIdx].matIdType.z;
        if (ex < 0.001) ex = 1.0;
        if (ez < 0.001) ez = 1.0;
        areaPdf = 1.0 / (4.0 * ex * ez);
    }
    int lc = 0;
    for (int i = 0; i < uNumPrimitives; i++) {
        int p = int(prims[i].matIdType.y);
        if (p != 0 && p != 1) continue;
        int m = int(prims[i].matIdType.x);
        if (m < 0 || m >= uNumMaterials) continue;
        if (materials[m].emissionStrength.x < 0.01) continue;
        lc++;
    }
    if (uProjectorOn > 0) lc++;
    return areaPdf / float(max(lc, 1));
}

// Fill the camera side of the path: x[0]=camera, x[1..eyeC]=g_eyeVerts[0..eyeC-1].
void bdptFillCam(int eyeC) {
    gBpos[0] = uCamPos; gBnrm[0] = uCamForward; gBmat[0] = -1; gBspec[0] = false; gBmedium[0]=false; gBphase[0]=0;
    for (int m = 0; m < eyeC; m++) {
        gBpos[1+m]  = g_eyeVerts[m].pos;
        gBnrm[1+m]  = g_eyeVerts[m].nrm;
        gBmat[1+m]  = g_eyeVerts[m].matId;
        gBspec[1+m] = g_eyeVerts[m].isSpecular;
        gBmedium[1+m]=g_eyeVerts[m].isMedium;gBphase[1+m]=g_eyeVerts[m].phaseG;
    }
}

// Append the traced light subpath g_lightVerts[topIdx..0] (emitter last) starting at x[startK].
void bdptFillLightVerts(int startK, int topIdx) {
    for (int m = 0; m <= topIdx; m++) {
        int li = topIdx - m;
        gBpos[startK+m]  = g_lightVerts[li].pos;
        gBnrm[startK+m]  = g_lightVerts[li].nrm;
        gBmat[startK+m]  = g_lightVerts[li].matId;
        gBspec[startK+m] = g_lightVerts[li].isSpecular;
        gBmedium[startK+m]=g_lightVerts[li].isMedium;gBphase[startK+m]=g_lightVerts[li].phaseG;
    }
    gBoriginPdf = max(g_lightVerts[0].fwdPdf, 1e-20);  // areaPdf*selectPdf from emission
    gBoriginPrim = g_ltOriginPrim;
}

// ---- Per-site weight wrappers: assemble the path, then return the method-B weight ----
float bdptWeightB_IC(int i, int j) {            // light[i] <-> eye[j]   (s=i+1, t=j+2)
    bdptFillCam(j + 1);
    bdptFillLightVerts(j + 2, i);
    gBn = i + j + 3;
    return bdptWeightB(j + 2);
}
float bdptWeightB_cam(int i) {                  // light[i] -> camera    (s=i+1, t=1)
    bdptFillCam(0);
    bdptFillLightVerts(1, i);
    gBn = i + 2;
    return bdptWeightB(1);
}
float bdptWeightB_nee(int j, vec3 lp, vec3 ln, int primIdx) {  // eye[j] -> NEE light (s=1, t=j+2)
    bdptFillCam(j + 1);
    int n = j + 3;
    gBpos[n-1] = lp; gBnrm[n-1] = ln; gBmat[n-1] = -1; gBspec[n-1] = false; gBmedium[n-1]=false;gBphase[n-1]=0;
    gBn = n;
    gBoriginPdf = bdptLightOriginPdfPrim(primIdx);
    gBoriginPrim = primIdx;
    return bdptWeightB(j + 2);
}
float bdptWeightB_emit(int eyeC, vec3 lp, vec3 ln, int primIdx) {  // eye hits emitter (s=0, t=eyeC+2)
    bdptFillCam(eyeC);
    int n = eyeC + 2;
    gBpos[n-1] = lp; gBnrm[n-1] = ln; gBmat[n-1] = -1; gBspec[n-1] = false; gBmedium[n-1]=false;gBphase[n-1]=0;
    gBn = n;
    gBoriginPdf = bdptLightOriginPdfPrim(primIdx);
    gBoriginPrim = primIdx;
    return bdptWeightB(n);
}
#endif // FEAT_BDPT_MIS_B

#ifdef FEAT_BDPT
void traceLightPath() {
