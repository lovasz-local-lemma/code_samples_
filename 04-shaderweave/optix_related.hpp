#pragma once

// OptiX 7.x+ Runtime Infrastructure
// Supports runtime compilation of PTX via NVRTC and dynamic pipeline creation

// Prevent Windows min/max macros from conflicting with std::min/std::max
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <cuda_runtime.h>
#include <nvrtc.h>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

#include "shorthands.hpp"

namespace fs = std::filesystem;

// ============================================================================
// OptiX Error Checking
// ============================================================================

#define OPTIX_CHECK(call)                                                      \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            std::cerr << "OptiX error: " << optixGetErrorName(res)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            throw std::runtime_error("OptiX call failed");                     \
        }                                                                      \
    } while (0)

#define OPTIX_CHECK_LOG(call)                                                  \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            std::cerr << "OptiX error: " << optixGetErrorName(res)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            if (log_size > 1) std::cerr << "Log: " << log << std::endl;        \
            throw std::runtime_error("OptiX call failed");                     \
        }                                                                      \
    } while (0)

// ============================================================================
// Launch Parameters - shared between host and device
// ============================================================================

struct OptixLaunchParams {
    // Output buffer
    uchar4* outputBuffer;
    unsigned int width;
    unsigned int height;
    
    // Time and animation
    float time;
    float deltaTime;
    
    // Mouse input
    float mouseX;
    float mouseY;
    int mouseButtons;
    
    // Camera (for ray generation)
    float3 camPos;
    float3 camDir;
    float3 camUp;
    float3 camRight;
    float fov;
    
    // Frame counter
    unsigned int frameIndex;
    
    // Optional input textures/buffers
    cudaTextureObject_t inputTexture;
    float* inputBuffer;
    unsigned int inputWidth;
    unsigned int inputHeight;
    
    // Scene data handle (for BVH traversal)
    OptixTraversableHandle traversable;
};

// ============================================================================
// OptiX Program Types
// ============================================================================

enum class OptixProgramType {
    RayGen,
    Miss,
    ClosestHit,
    AnyHit,
    Intersection
};

// ============================================================================
// OptiX Module Manager - compiles and caches PTX modules
// ============================================================================

class OptixModuleManager {
public:
    struct CompiledModule {
        OptixModule module = nullptr;
        str ptxCode;
        str sourceCode;
        bool valid = false;
    };
    
private:
    OptixDeviceContext context = nullptr;
    std::map<str, CompiledModule> modules;
    OptixModuleCompileOptions moduleCompileOptions = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    
public:
    void init(OptixDeviceContext ctx) {
        context = ctx;
        
        // Module compile options
        moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        
        // Pipeline compile options
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipelineCompileOptions.numPayloadValues = 3;
        pipelineCompileOptions.numAttributeValues = 2;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    }
    
    void cleanup() {
        for (auto& [name, mod] : modules) {
            if (mod.module) {
                optixModuleDestroy(mod.module);
            }
        }
        modules.clear();
    }
    
    // Compile CUDA source to PTX using NVRTC
    bool compileToPTX(const str& source, str& ptxOut, str& errorLog) {
        nvrtcProgram prog;
        nvrtcResult res = nvrtcCreateProgram(&prog, source.c_str(), "optix_program.cu", 0, nullptr, nullptr);
        if (res != NVRTC_SUCCESS) {
            errorLog = "Failed to create NVRTC program";
            return false;
        }
        
        // Compile options for OptiX
        // Note: Update the OptiX SDK path if installed in a different location
        // NVRTC doesn't want quotes around paths - use raw paths
        std::vector<const char*> options = {
            "--gpu-architecture=compute_70",
            "--relocatable-device-code=true",
            "--std=c++17",
            "-IU:/_DEV_SDKS+_/OptiX SDK 9.0.0/include",
            "-IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include",
            "-D__CUDACC__",
            "-DOPTIX_PROGRAM"
        };
        
        res = nvrtcCompileProgram(prog, (int)options.size(), options.data());
        
        // Get log
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        if (logSize > 1) {
            errorLog.resize(logSize);
            nvrtcGetProgramLog(prog, errorLog.data());
        }
        
        if (res != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&prog);
            return false;
        }
        
        // Get PTX
        size_t ptxSize;
        nvrtcGetPTXSize(prog, &ptxSize);
        ptxOut.resize(ptxSize);
        nvrtcGetPTX(prog, ptxOut.data());
        
        nvrtcDestroyProgram(&prog);
        return true;
    }
    
    // Create OptiX module from PTX
    bool createModule(const str& name, const str& ptx, str& errorLog) {
        if (modules.count(name) && modules[name].module) {
            optixModuleDestroy(modules[name].module);
        }
        
        char log[2048];
        size_t log_size = sizeof(log);
        
        OptixModule module;
        OptixResult res = optixModuleCreate(
            context,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            ptx.c_str(),
            ptx.size(),
            log,
            &log_size,
            &module
        );
        
        if (log_size > 1) {
            errorLog = log;
        }
        
        if (res != OPTIX_SUCCESS) {
            return false;
        }
        
        modules[name] = {module, ptx, "", true};
        return true;
    }
    
    // Compile source and create module in one step
    bool compileAndCreate(const str& name, const str& source, str& errorLog) {
        str ptx;
        if (!compileToPTX(source, ptx, errorLog)) {
            return false;
        }
        
        if (!createModule(name, ptx, errorLog)) {
            return false;
        }
        
        modules[name].sourceCode = source;
        return true;
    }
    
    OptixModule getModule(const str& name) {
        if (modules.count(name) && modules[name].valid) {
            return modules[name].module;
        }
        return nullptr;
    }
    
    const OptixPipelineCompileOptions& getPipelineCompileOptions() const {
        return pipelineCompileOptions;
    }
};

// ============================================================================
// OptiX Pipeline Builder
// ============================================================================

class OptixPipelineBuilder {
private:
    OptixDeviceContext context = nullptr;
    OptixPipeline pipeline = nullptr;
    
    std::vector<OptixProgramGroup> programGroups;
    OptixProgramGroup raygenPG = nullptr;
    OptixProgramGroup missPG = nullptr;
    OptixProgramGroup hitgroupPG = nullptr;
    
    OptixShaderBindingTable sbt = {};
    CUdeviceptr raygenRecord = 0;
    CUdeviceptr missRecord = 0;
    CUdeviceptr hitgroupRecord = 0;
    CUdeviceptr launchParamsBuffer = 0;
    size_t launchParamsBufferSize = 0;
    
public:
    void init(OptixDeviceContext ctx) {
        context = ctx;
    }
    
    void cleanup() {
        if (raygenRecord) cudaFree((void*)raygenRecord);
        if (missRecord) cudaFree((void*)missRecord);
        if (hitgroupRecord) cudaFree((void*)hitgroupRecord);
        if (launchParamsBuffer) cudaFree((void*)launchParamsBuffer);
        
        for (auto& pg : programGroups) {
            if (pg) optixProgramGroupDestroy(pg);
        }
        programGroups.clear();
        
        if (pipeline) {
            optixPipelineDestroy(pipeline);
            pipeline = nullptr;
        }
        
        raygenRecord = missRecord = hitgroupRecord = 0;
        launchParamsBuffer = 0;
        launchParamsBufferSize = 0;
        raygenPG = missPG = hitgroupPG = nullptr;
        memset(&sbt, 0, sizeof(sbt));
    }
    
    bool buildPipeline(
        OptixModule module,
        const str& raygenEntry,
        const str& missEntry,
        const str& closestHitEntry,
        const OptixPipelineCompileOptions& pipelineCompileOptions,
        str& errorLog
    ) {
        cleanup();
        
        char log[2048];
        size_t log_size = sizeof(log);
        
        OptixProgramGroupOptions pgOptions = {};
        
        // Ray generation program
        OptixProgramGroupDesc raygenDesc = {};
        raygenDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenDesc.raygen.module = module;
        raygenDesc.raygen.entryFunctionName = raygenEntry.c_str();
        
        log_size = sizeof(log);
        OptixResult res = optixProgramGroupCreate(
            context, &raygenDesc, 1, &pgOptions, log, &log_size, &raygenPG
        );
        if (res != OPTIX_SUCCESS) {
            errorLog = str("RayGen PG: ") + log;
            return false;
        }
        programGroups.push_back(raygenPG);
        
        // Miss program
        OptixProgramGroupDesc missDesc = {};
        missDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missDesc.miss.module = module;
        missDesc.miss.entryFunctionName = missEntry.c_str();
        
        log_size = sizeof(log);
        res = optixProgramGroupCreate(
            context, &missDesc, 1, &pgOptions, log, &log_size, &missPG
        );
        if (res != OPTIX_SUCCESS) {
            errorLog = str("Miss PG: ") + log;
            return false;
        }
        programGroups.push_back(missPG);
        
        // Hit group (closest hit)
        OptixProgramGroupDesc hitgroupDesc = {};
        hitgroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroupDesc.hitgroup.moduleCH = module;
        hitgroupDesc.hitgroup.entryFunctionNameCH = closestHitEntry.c_str();
        
        log_size = sizeof(log);
        res = optixProgramGroupCreate(
            context, &hitgroupDesc, 1, &pgOptions, log, &log_size, &hitgroupPG
        );
        if (res != OPTIX_SUCCESS) {
            errorLog = str("HitGroup PG: ") + log;
            return false;
        }
        programGroups.push_back(hitgroupPG);
        
        // Link pipeline
        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = 2;
        
        log_size = sizeof(log);
        res = optixPipelineCreate(
            context,
            &pipelineCompileOptions,
            &pipelineLinkOptions,
            programGroups.data(),
            (unsigned int)programGroups.size(),
            log,
            &log_size,
            &pipeline
        );
        if (res != OPTIX_SUCCESS) {
            errorLog = str("Pipeline: ") + log;
            return false;
        }
        
        // Set stack sizes
        OptixStackSizes stackSizes = {};
        for (auto& pg : programGroups) {
            OptixStackSizes ss;
            optixProgramGroupGetStackSize(pg, &ss, pipeline);
            stackSizes.cssRG = std::max(stackSizes.cssRG, ss.cssRG);
            stackSizes.cssMS = std::max(stackSizes.cssMS, ss.cssMS);
            stackSizes.cssCH = std::max(stackSizes.cssCH, ss.cssCH);
        }
        
        unsigned int maxTraceDepth = 2;
        unsigned int directCallableStackSizeFromTraversal = 0;
        unsigned int directCallableStackSizeFromState = 0;
        unsigned int continuationStackSize = stackSizes.cssRG + 
            maxTraceDepth * std::max(stackSizes.cssCH, stackSizes.cssMS);
        
        optixPipelineSetStackSize(
            pipeline,
            directCallableStackSizeFromTraversal,
            directCallableStackSizeFromState,
            continuationStackSize,
            2  // maxTraversableGraphDepth
        );
        
        // Build SBT
        return buildSBT(errorLog);
    }
    
    bool buildSBT(str& errorLog) {
        // SBT record structures
        struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) RayGenRecord {
            char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        };
        struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
            char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        };
        struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord {
            char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        };
        
        // RayGen
        RayGenRecord rg;
        optixSbtRecordPackHeader(raygenPG, &rg);
        cudaMalloc((void**)&raygenRecord, sizeof(RayGenRecord));
        cudaMemcpy((void*)raygenRecord, &rg, sizeof(RayGenRecord), cudaMemcpyHostToDevice);
        
        // Miss
        MissRecord ms;
        optixSbtRecordPackHeader(missPG, &ms);
        cudaMalloc((void**)&missRecord, sizeof(MissRecord));
        cudaMemcpy((void*)missRecord, &ms, sizeof(MissRecord), cudaMemcpyHostToDevice);
        
        // HitGroup
        HitGroupRecord hg;
        optixSbtRecordPackHeader(hitgroupPG, &hg);
        cudaMalloc((void**)&hitgroupRecord, sizeof(HitGroupRecord));
        cudaMemcpy((void*)hitgroupRecord, &hg, sizeof(HitGroupRecord), cudaMemcpyHostToDevice);
        
        // Fill SBT
        sbt.raygenRecord = raygenRecord;
        sbt.missRecordBase = missRecord;
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = hitgroupRecord;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
        sbt.hitgroupRecordCount = 1;
        
        return true;
    }
    
    bool launch(const OptixLaunchParams& params, CUstream stream = 0) {
        if (!pipeline) return false;

        if (launchParamsBuffer == 0 || launchParamsBufferSize < sizeof(OptixLaunchParams)) {
            if (launchParamsBuffer) {
                cudaFree((void*)launchParamsBuffer);
            }
            cudaMalloc((void**)&launchParamsBuffer, sizeof(OptixLaunchParams));
            launchParamsBufferSize = sizeof(OptixLaunchParams);
        }

        cudaMemcpy((void*)launchParamsBuffer, &params, sizeof(OptixLaunchParams), cudaMemcpyHostToDevice);
        
        OptixResult res = optixLaunch(
            pipeline,
            stream,
            launchParamsBuffer,
            sizeof(OptixLaunchParams),
            &sbt,
            params.width,
            params.height,
            1  // depth
        );

        return res == OPTIX_SUCCESS;
    }
    
    OptixPipeline getPipeline() const { return pipeline; }
    const OptixShaderBindingTable& getSBT() const { return sbt; }
};

// ============================================================================
// OptiX Context Manager
// ============================================================================

class OptixContextManager {
private:
    CUcontext cuContext = nullptr;
    OptixDeviceContext optixContext = nullptr;
    bool initialized = false;
    
    static void contextLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata*/) {
        std::cerr << "[OptiX][" << level << "][" << tag << "]: " << message << std::endl;
    }
    
public:
    bool init() {
        if (initialized) return true;
        
        CUresult cuRes = cuCtxGetCurrent(&cuContext);
        if (cuRes != CUDA_SUCCESS || cuContext == nullptr) {
            // Fall back to the runtime path only when no driver context is current.
            cudaFree(0);  // Force CUDA initialization
            cuRes = cuCtxGetCurrent(&cuContext);
        }

        if (cuRes != CUDA_SUCCESS || cuContext == nullptr) {
            std::cerr << "Failed to get CUDA context" << std::endl;
            return false;
        }
        
        // Initialize OptiX
        OptixResult res = optixInit();
        if (res != OPTIX_SUCCESS) {
            std::cerr << "Failed to initialize OptiX: " << optixGetErrorName(res) << std::endl;
            return false;
        }
        
        // Create OptiX context
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &contextLogCallback;
        options.logCallbackLevel = 4;
        
        res = optixDeviceContextCreate(cuContext, &options, &optixContext);
        if (res != OPTIX_SUCCESS) {
            std::cerr << "Failed to create OptiX context: " << optixGetErrorName(res) << std::endl;
            return false;
        }
        
        initialized = true;
        std::cout << "OptiX initialized successfully" << std::endl;
        return true;
    }
    
    void cleanup() {
        if (optixContext) {
            optixDeviceContextDestroy(optixContext);
            optixContext = nullptr;
        }
        initialized = false;
    }
    
    OptixDeviceContext getContext() const { return optixContext; }
    bool isInitialized() const { return initialized; }
};

// ============================================================================
// OptiX Program Header/Footer for NVRTC compilation
// ============================================================================

namespace OPTIX_RT {

// Header injected before user code
const str OPTIX_PROGRAM_HEADER = R"(
#include <optix.h>

// Launch parameters structure - must match host-side definition
struct OptixLaunchParams {
    // Output buffer
    uchar4* outputBuffer;
    unsigned int width;
    unsigned int height;
    
    // Time and animation
    float time;
    float deltaTime;
    
    // Mouse input
    float mouseX;
    float mouseY;
    int mouseButtons;
    
    // Camera (for ray generation)
    float3 camPos;
    float3 camDir;
    float3 camUp;
    float3 camRight;
    float fov;
    
    // Frame counter
    unsigned int frameIndex;
    
    // Optional input textures/buffers
    cudaTextureObject_t inputTexture;
    float* inputBuffer;
    unsigned int inputWidth;
    unsigned int inputHeight;
    
    // Scene data handle (for BVH traversal)
    OptixTraversableHandle traversable;
};

extern "C" {
    __constant__ OptixLaunchParams params;
}

// Payload structure
struct PayloadRadiance {
    float3 color;
    float depth;
};

// Helper functions - float2
__forceinline__ __device__ float2 make_float2(float x, float y) {
    float2 v; v.x = x; v.y = y; return v;
}

__forceinline__ __device__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__forceinline__ __device__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__forceinline__ __device__ float2 operator*(float2 a, float s) {
    return make_float2(a.x * s, a.y * s);
}

__forceinline__ __device__ float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

__forceinline__ __device__ float length(float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}

// Helper functions - float3
__forceinline__ __device__ float3 make_float3(float x, float y, float z) {
    float3 v; v.x = x; v.y = y; v.z = z; return v;
}

__forceinline__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__forceinline__ __device__ float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__forceinline__ __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__forceinline__ __device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__forceinline__ __device__ float3 normalize(float3 v) {
    float len = length(v);
    return v * (1.0f / len);
}

__forceinline__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Utility functions
__forceinline__ __device__ float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__forceinline__ __device__ float clamp(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__forceinline__ __device__ float3 clamp(float3 v, float lo, float hi) {
    return make_float3(clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi));
}

__forceinline__ __device__ float3 mix(float3 a, float3 b, float t) {
    return a * (1.0f - t) + b * t;
}

__forceinline__ __device__ float3 reflect(float3 I, float3 N) {
    return I - N * 2.0f * dot(N, I);
}

__forceinline__ __device__ uchar4 make_color(float3 c) {
    return make_uchar4(
        (unsigned char)(__saturatef(c.x) * 255.0f),
        (unsigned char)(__saturatef(c.y) * 255.0f),
        (unsigned char)(__saturatef(c.z) * 255.0f),
        255
    );
}

// Get payload
__forceinline__ __device__ PayloadRadiance getPayload() {
    PayloadRadiance p;
    p.color.x = __uint_as_float(optixGetPayload_0());
    p.color.y = __uint_as_float(optixGetPayload_1());
    p.color.z = __uint_as_float(optixGetPayload_2());
    return p;
}

// Set payload
__forceinline__ __device__ void setPayload(PayloadRadiance p) {
    optixSetPayload_0(__float_as_uint(p.color.x));
    optixSetPayload_1(__float_as_uint(p.color.y));
    optixSetPayload_2(__float_as_uint(p.color.z));
}

)";

// Footer (empty for now, but can add utility functions)
const str OPTIX_PROGRAM_FOOTER = R"(
)";

// Store named OptiX programs
inline std::map<str, str> named_optix_code;

// Load OptiX programs from folder
inline void load_optix_programs(const str& folder) {
    named_optix_code.clear();
    const fs::path resolved_folder = SHORTHANDS::resolve_project_path(folder);
    
    if (!fs::exists(resolved_folder)) {
        std::cout << "OptiX programs folder not found: " << resolved_folder.string() << std::endl;
        return;
    }
    
    for (const auto& entry : fs::directory_iterator(resolved_folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".cu") {
            std::ifstream file(entry.path());
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                str name = entry.path().stem().string();
                named_optix_code[name] = buffer.str();
                std::cout << "Loaded OptiX program: " << name << std::endl;
            }
        }
    }
}

} // namespace OPTIX_RT

// ============================================================================
// Global instances
// ============================================================================

inline OptixContextManager g_optixContext;
inline OptixModuleManager g_optixModules;
inline OptixPipelineBuilder g_optixPipeline;
