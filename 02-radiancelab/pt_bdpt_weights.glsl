// Strategy densities share a full-film path measure. Evaluate each directional
// density once, then build prefix/suffix products in log space. NEE samples a
// conditional emitter distribution, which need not equal photon emission.
float bdptStrategyLog[BDPT_MAXP+1];
float bdptWeightFromDensities(int n,int selected,float a[BDPT_MAXP],float b[BDPT_MAXP],
                            bool delta[BDPT_MAXP],float origin,float nee,int scheme,float lightSamples) {
    float suffix[BDPT_MAXP+1];
    suffix[n-1]=log(max(origin,1e-30));
    for(int k=n-2;k>=1;--k) suffix[k]=suffix[k+1]+(b[k]>0.0?log(b[k]):(delta[k+1]?0.0:-1e30));
    float prefix=0.0, top=-1e30;
    for(int t=1;t<=n;++t) {
        bool valid=t==n || (!delta[t-1]&&!delta[t]);
        // Camera builder stores eight scatter vertices; photon builder stores
        // its emitter plus seven. Do not weight nonexistent longer strategies.
        valid=valid && (t==n ? n<=MAX_EYE_VERTS+2 : t<=MAX_EYE_VERTS+1 && n-t<=MAX_LIGHT_VERTS);
        float p=prefix;
        if(t==n-1 && t>=2) { valid=valid&&nee>0.0; p+=log(max(nee,1e-30)); }
        else if(t<n) { valid=valid&&origin>0.0; p+=suffix[t]+log(max(lightSamples,1.0)); }
        bdptStrategyLog[t]=valid?p:-1e30;
        top=max(top,bdptStrategyLog[t]);
        if(t<n) prefix+=(a[t]>0.0?log(a[t]):(delta[t-1]?0.0:-1e30));
    }
    if(scheme==0) {
        float count=0; for(int t=1;t<=n;++t) if(bdptStrategyLog[t]>-1e29) count+=1;
        return bdptStrategyLog[selected]>-1e29?1.0/max(count,1.0):0.0;
    }
    float beta=scheme==2?2.0:1.0, sum=0.0;
    for(int t=1;t<=n;++t) sum+=exp(beta*(bdptStrategyLog[t]-top));
    return exp(beta*(bdptStrategyLog[selected]-top))/max(sum,1e-30);
}
