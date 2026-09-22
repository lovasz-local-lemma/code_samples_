// Progressive Path Tracer with Temporal Accumulation
// Time-resolved rendering with motion blur and DOF

// Random number generation
__device__ unsigned int tea(unsigned int val0, unsigned int val1) {
    unsigned int v0 = val0, v1 = val1, s0 = 0;
    for (unsigned int n = 0; n < 4; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
}

__device__ float rnd(unsigned int& seed) {
    seed = tea(seed, seed);
    return (float)(seed & 0x00FFFFFF) / (float)0x01000000;
}

// SDF primitives
__device__ float sdSphere(float3 p, float r) {
    return sqrtf(p.x*p.x + p.y*p.y + p.z*p.z) - r;
}

__device__ float sdBox(float3 p, float3 b) {
    float3 d = make_float3(fabsf(p.x) - b.x, fabsf(p.y) - b.y, fabsf(p.z) - b.z);
    float outside = sqrtf(fmaxf(d.x, 0.0f) * fmaxf(d.x, 0.0f) + 
                          fmaxf(d.y, 0.0f) * fmaxf(d.y, 0.0f) + 
                          fmaxf(d.z, 0.0f) * fmaxf(d.z, 0.0f));
    float inside = fminf(fmaxf(d.x, fmaxf(d.y, d.z)), 0.0f);
    return outside + inside;
}

// Material structure
struct Material {
    float3 albedo;
    float3 emission;
    float roughness;
    float metallic;
    int type; // 0=diffuse, 1=metal, 2=glass, 3=emissive
};

// Scene with materials and motion
__device__ float sceneSDF(float3 p, float time, Material& mat) {
    float d = 1e10f;
    mat = {{0.8f, 0.8f, 0.8f}, {0.0f, 0.0f, 0.0f}, 0.5f, 0.0f, 0};
    
    // Ground plane
    float ground = p.y + 1.0f;
    if (ground < d) {
        d = ground;
        float checker = fmodf(floorf(p.x) + floorf(p.z), 2.0f);
        mat.albedo = make_float3(0.4f + checker * 0.4f, 0.4f + checker * 0.4f, 0.4f + checker * 0.3f);
        mat.type = 0;
    }
    
    // Moving metallic sphere
    float3 sp1 = make_float3(p.x - sinf(time) * 0.5f, p.y - 0.3f, p.z - cosf(time * 0.7f) * 0.5f);
    float sphere1 = sdSphere(sp1, 0.5f);
    if (sphere1 < d) {
        d = sphere1;
        mat.albedo = make_float3(0.95f, 0.8f, 0.6f);
        mat.roughness = 0.1f;
        mat.metallic = 1.0f;
        mat.type = 1;
    }
    
    // Glass sphere
    float3 sp2 = make_float3(p.x + 0.8f, p.y, p.z + 0.5f);
    float sphere2 = sdSphere(sp2, 0.4f);
    if (sphere2 < d) {
        d = sphere2;
        mat.albedo = make_float3(1.0f, 1.0f, 1.0f);
        mat.type = 2;
    }
    
    // Diffuse sphere
    float3 sp3 = make_float3(p.x - 0.9f, p.y + 0.2f, p.z + 0.3f);
    float sphere3 = sdSphere(sp3, 0.35f);
    if (sphere3 < d) {
        d = sphere3;
        mat.albedo = make_float3(0.8f, 0.2f, 0.2f);
        mat.type = 0;
    }
    
    // Light box
    float3 lp = make_float3(p.x, p.y - 2.5f, p.z);
    float light = sdBox(lp, make_float3(0.8f, 0.1f, 0.8f));
    if (light < d) {
        d = light;
        mat.emission = make_float3(15.0f, 14.0f, 12.0f);
        mat.type = 3;
    }
    
    return d;
}

__device__ float3 calcNormal(float3 p, float time) {
    Material dummy;
    float eps = 0.001f;
    float3 n;
    n.x = sceneSDF(make_float3(p.x + eps, p.y, p.z), time, dummy) - 
          sceneSDF(make_float3(p.x - eps, p.y, p.z), time, dummy);
    n.y = sceneSDF(make_float3(p.x, p.y + eps, p.z), time, dummy) - 
          sceneSDF(make_float3(p.x, p.y - eps, p.z), time, dummy);
    n.z = sceneSDF(make_float3(p.x, p.y, p.z + eps), time, dummy) - 
          sceneSDF(make_float3(p.x, p.y, p.z - eps), time, dummy);
    return normalize(n);
}

// Cosine-weighted hemisphere sampling
__device__ float3 sampleHemisphere(float3 n, unsigned int& seed) {
    float r1 = rnd(seed);
    float r2 = rnd(seed);
    
    float phi = 2.0f * 3.14159f * r1;
    float cosTheta = sqrtf(r2);
    float sinTheta = sqrtf(1.0f - r2);
    
    float3 w = n;
    float3 u = normalize(fabsf(w.x) > 0.1f ? cross(make_float3(0, 1, 0), w) : cross(make_float3(1, 0, 0), w));
    float3 v = cross(w, u);
    
    return normalize(u * cosf(phi) * sinTheta + v * sinf(phi) * sinTheta + w * cosTheta);
}

// Fresnel (Schlick approximation)
__device__ float fresnel(float cosTheta, float ior) {
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);
}

// Path trace
__device__ float3 pathTrace(float3 ro, float3 rd, float time, unsigned int& seed) {
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    
    for (int bounce = 0; bounce < 6; bounce++) {
        float t = 0.0f;
        Material mat;
        bool hit = false;
        
        for (int i = 0; i < 100; i++) {
            float3 p = ro + rd * t;
            float d = sceneSDF(p, time, mat);
            if (d < 0.001f) { hit = true; break; }
            t += d;
            if (t > 50.0f) break;
        }
        
        if (!hit) {
            // Sky
            float3 sky = make_float3(0.5f, 0.7f, 1.0f) * (0.5f + 0.5f * rd.y);
            color = color + throughput * sky * 0.5f;
            break;
        }
        
        float3 hitPos = ro + rd * t;
        float3 n = calcNormal(hitPos, time);
        
        // Emission
        color = color + throughput * mat.emission;
        if (mat.type == 3) break;
        
        // BRDF sampling
        if (mat.type == 0) {
            // Diffuse
            rd = sampleHemisphere(n, seed);
            throughput = make_float3(throughput.x * mat.albedo.x, 
                                     throughput.y * mat.albedo.y, 
                                     throughput.z * mat.albedo.z);
        } else if (mat.type == 1) {
            // Metal
            float3 reflected = rd - n * 2.0f * dot(rd, n);
            float3 rough = sampleHemisphere(n, seed);
            rd = normalize(reflected + rough * mat.roughness);
            if (dot(rd, n) < 0.0f) rd = reflected;
            throughput = make_float3(throughput.x * mat.albedo.x, 
                                     throughput.y * mat.albedo.y, 
                                     throughput.z * mat.albedo.z);
        } else if (mat.type == 2) {
            // Glass
            float ior = 1.5f;
            float cosi = -dot(rd, n);
            bool entering = cosi > 0.0f;
            float3 nn = entering ? n : n * (-1.0f);
            float eta = entering ? (1.0f / ior) : ior;
            cosi = fabsf(cosi);
            
            float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
            float F = fresnel(cosi, ior);
            
            if (k < 0.0f || rnd(seed) < F) {
                // Reflect
                rd = rd - nn * 2.0f * dot(rd, nn);
            } else {
                // Refract
                rd = rd * eta + nn * (eta * cosi - sqrtf(k));
            }
        }
        
        ro = hitPos + rd * 0.01f;
        
        // Russian roulette
        if (bounce > 2) {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (rnd(seed) > p) break;
            throughput = throughput * (1.0f / p);
        }
    }
    
    return color;
}

extern "C" __global__ void __raygen__main() {
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();
    
    unsigned int seed = tea(idx.x + idx.y * dim.x, params.frameIndex);
    
    // Jittered UV for anti-aliasing
    float u = ((float)idx.x + rnd(seed)) / (float)dim.x;
    float v = ((float)idx.y + rnd(seed)) / (float)dim.y;
    
    float2 ndc = make_float2(u * 2.0f - 1.0f, v * 2.0f - 1.0f);
    ndc.x *= (float)dim.x / (float)dim.y;
    
    // Depth of field
    float focalDist = 3.0f;
    float aperture = 0.05f;
    
    float fovScale = tanf(params.fov * 0.5f * 3.14159f / 180.0f);
    float3 focalPoint = params.camPos + 
        (params.camDir + params.camRight * ndc.x * fovScale + params.camUp * ndc.y * fovScale) * focalDist;
    
    // Aperture jitter
    float angle = rnd(seed) * 2.0f * 3.14159f;
    float radius = sqrtf(rnd(seed)) * aperture;
    float3 apertureOffset = params.camRight * cosf(angle) * radius + params.camUp * sinf(angle) * radius;
    
    float3 rayOrigin = params.camPos + apertureOffset;
    float3 rayDir = normalize(focalPoint - rayOrigin);
    
    // Motion blur - sample time within frame
    float frameTime = params.time + rnd(seed) * params.deltaTime;
    
    // Path trace
    float3 color = pathTrace(rayOrigin, rayDir, frameTime, seed);
    
    // Temporal accumulation
    if (params.frameIndex > 0) {
        uchar4 prev = params.outputBuffer[idx.y * dim.x + idx.x];
        float3 prevColor = make_float3((float)prev.x / 255.0f, (float)prev.y / 255.0f, (float)prev.z / 255.0f);
        prevColor.x = powf(prevColor.x, 2.2f);
        prevColor.y = powf(prevColor.y, 2.2f);
        prevColor.z = powf(prevColor.z, 2.2f);
        
        float weight = 1.0f / (float)(params.frameIndex + 1);
        color = prevColor * (1.0f - weight) + color * weight;
    }
    
    // Tone mapping (ACES)
    float3 mapped;
    mapped.x = (color.x * (2.51f * color.x + 0.03f)) / (color.x * (2.43f * color.x + 0.59f) + 0.14f);
    mapped.y = (color.y * (2.51f * color.y + 0.03f)) / (color.y * (2.43f * color.y + 0.59f) + 0.14f);
    mapped.z = (color.z * (2.51f * color.z + 0.03f)) / (color.z * (2.43f * color.z + 0.59f) + 0.14f);
    
    // Gamma
    mapped.x = powf(fmaxf(0.0f, fminf(1.0f, mapped.x)), 1.0f / 2.2f);
    mapped.y = powf(fmaxf(0.0f, fminf(1.0f, mapped.y)), 1.0f / 2.2f);
    mapped.z = powf(fmaxf(0.0f, fminf(1.0f, mapped.z)), 1.0f / 2.2f);
    
    params.outputBuffer[idx.y * dim.x + idx.x] = make_color(mapped);
}

extern "C" __global__ void __miss__main() {
    setPayload({make_float3(0, 0, 0), 0});
}

extern "C" __global__ void __closesthit__main() {
    setPayload({make_float3(1, 0, 1), 0});
}
