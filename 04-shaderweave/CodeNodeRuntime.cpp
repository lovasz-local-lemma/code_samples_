#include "CodeNodeRuntime.hpp"

#include "../shorthands.hpp"
using namespace SHORTHANDS;
#include "../optix_related.hpp"

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <utility>
#include <vector>

namespace LegacyNodeRuntime {

namespace {

namespace fs = std::filesystem;

const std::string& empty_string()
{
    static const std::string value;
    return value;
}

namespace CudaBridge {

std::map<std::string, std::string> g_named_cuda_code;

const char* kKernelHeader = R"(
struct Uniforms {
    float iTime;
    float iTimeDelta;
    int iFrame;
    float iMouse[4];
    float iResolution[3];
    float iDate[4];
};

__device__ inline float fract(float x) { return x - floorf(x); }
__device__ inline float mix(float a, float b, float t) { return a + (b - a) * t; }
__device__ inline float clamp(float x, float minVal, float maxVal) { return fminf(fmaxf(x, minVal), maxVal); }
__device__ inline float smoothstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
__device__ inline float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
__device__ inline float mod(float x, float y) { return x - y * floorf(x / y); }

__device__ inline float dot2(float x1, float y1, float x2, float y2) { return x1*x2 + y1*y2; }
__device__ inline float dot3(float x1, float y1, float z1, float x2, float y2, float z2) { return x1*x2 + y1*y2 + z1*z2; }
__device__ inline float length2(float x, float y) { return sqrtf(x*x + y*y); }
__device__ inline float length3(float x, float y, float z) { return sqrtf(x*x + y*y + z*z); }
__device__ inline void normalize2(float& x, float& y) { float len = length2(x, y); if (len > 0) { x /= len; y /= len; } }
__device__ inline void normalize3(float& x, float& y, float& z) { float len = length3(x, y, z); if (len > 0) { x /= len; y /= len; z /= len; } }

__device__ inline void hsv2rgb(float h, float s, float v, float& r, float& g, float& b) {
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    int hi = (int)(h * 6.0f) % 6;
    if (hi == 0) { r = c; g = x; b = 0; }
    else if (hi == 1) { r = x; g = c; b = 0; }
    else if (hi == 2) { r = 0; g = c; b = x; }
    else if (hi == 3) { r = 0; g = x; b = c; }
    else if (hi == 4) { r = x; g = 0; b = c; }
    else { r = c; g = 0; b = x; }
    r += m; g += m; b += m;
}
)";

const char* kKernelFooter = R"(
extern "C" __global__ void mainImageKernel(uchar4* output, int width, int height, Uniforms uniforms) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int flipped_y = height - 1 - y;

    float fragColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float fragCoord[2] = {(float)x, (float)flipped_y};

    mainImage(fragColor, fragCoord, uniforms);

    int idx = y * width + x;
    output[idx] = make_uchar4(
        (unsigned char)(clamp(fragColor[0], 0.0f, 1.0f) * 255.0f),
        (unsigned char)(clamp(fragColor[1], 0.0f, 1.0f) * 255.0f),
        (unsigned char)(clamp(fragColor[2], 0.0f, 1.0f) * 255.0f),
        (unsigned char)(clamp(fragColor[3], 0.0f, 1.0f) * 255.0f)
    );
}
)";

std::string default_main_image()
{
    return R"(
__device__ void mainImage(float fragColor[4], float fragCoord[2], const Uniforms& u) {
    float2 uv;
    uv.x = fragCoord[0] / u.iResolution[0];
    uv.y = fragCoord[1] / u.iResolution[1];

    float scale = 20.0f;
    float checker = fmodf(floorf(uv.x * scale) + floorf(uv.y * scale), 2.0f);
    float wave = 0.5f + 0.5f * sinf(u.iTime * 2.0f);

    if (checker > 0.5f) {
        fragColor[0] = 0.55f + 0.25f * wave;
        fragColor[1] = 0.03f;
        fragColor[2] = 0.03f;
    } else {
        fragColor[0] = 0.05f;
        fragColor[1] = 0.01f;
        fragColor[2] = 0.01f;
    }
    fragColor[3] = 1.0f;
}
)";
}

bool init_cuda(std::string& error_out)
{
    static bool initialized = false;
    if (initialized) {
        return true;
    }

    if (const CUresult init_result = cuInit(0); init_result != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(init_result, &err_str);
        error_out = std::string("cuInit failed: ") + (err_str ? err_str : "unknown error");
        return false;
    }

    int device_count = 0;
    if (const CUresult count_result = cuDeviceGetCount(&device_count); count_result != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(count_result, &err_str);
        error_out = std::string("cuDeviceGetCount failed: ") + (err_str ? err_str : "unknown error");
        return false;
    }

    if (device_count <= 0) {
        error_out = "No CUDA devices found.";
        return false;
    }

    CUdevice device = 0;
    if (const CUresult device_result = cuDeviceGet(&device, 0); device_result != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(device_result, &err_str);
        error_out = std::string("cuDeviceGet failed: ") + (err_str ? err_str : "unknown error");
        return false;
    }

    CUcontext context = nullptr;
    if (const CUresult context_result = cuCtxCreate(&context, 0, device); context_result != CUDA_SUCCESS) {
        const char* err_str = nullptr;
        cuGetErrorString(context_result, &err_str);
        error_out = std::string("cuCtxCreate failed: ") + (err_str ? err_str : "unknown error");
        return false;
    }

    initialized = true;
    return true;
}

struct Globals {
    float iTime = 0.0f;
    float iTimeDelta = 0.0f;
    int iFrame = 0;
    float iMouse[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float iResolution[3] = { 800.0f, 800.0f, 1.0f };
    float iDate[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point last_frame_time = start_time;

    void update()
    {
        const auto current_time = std::chrono::high_resolution_clock::now();
        iTime = std::chrono::duration<float>(current_time - start_time).count();
        iTimeDelta = std::chrono::duration<float>(current_time - last_frame_time).count();
        last_frame_time = current_time;
        iFrame++;

        std::time_t t = std::time(nullptr);
        std::tm now = {};
        localtime_s(&now, &t);
        iDate[0] = static_cast<float>(now.tm_year + 1900);
        iDate[1] = static_cast<float>(now.tm_mon + 1);
        iDate[2] = static_cast<float>(now.tm_mday);
        iDate[3] = static_cast<float>(now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec);
    }
};

class Kernel {
public:
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    std::string raw_code;
    std::string processed_code;
    std::string compile_log;
    bool compiled = false;

    ~Kernel()
    {
        if (module != nullptr) {
            cuModuleUnload(module);
            module = nullptr;
        }
    }

    std::string process_code(const std::string& user_code)
    {
        std::string result = kKernelHeader;
        result += "\n// ========== USER CODE BEGIN ==========\n";
        result += user_code;
        result += "\n// ========== USER CODE END ==========\n";

        if (user_code.find("mainImage") == std::string::npos) {
            result += default_main_image();
        }

        result += kKernelFooter;
        return result;
    }

    bool compile(const std::string& user_code, std::string& error_out)
    {
        raw_code = user_code;
        processed_code = process_code(user_code);

        nvrtcProgram program = nullptr;
        if (const nvrtcResult create_result = nvrtcCreateProgram(&program, processed_code.c_str(), "kernel.cu", 0, nullptr, nullptr);
            create_result != NVRTC_SUCCESS) {
            error_out = std::string("Failed to create NVRTC program: ") + nvrtcGetErrorString(create_result);
            compiled = false;
            return false;
        }

        const char* options[] = {
            "--gpu-architecture=compute_52",
            "-default-device"
        };

        const nvrtcResult compile_result = nvrtcCompileProgram(program, static_cast<int>(std::size(options)), options);

        size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        compile_log.resize(log_size > 0 ? log_size : 1);
        nvrtcGetProgramLog(program, compile_log.data());

        if (compile_result != NVRTC_SUCCESS) {
            error_out = compile_log.empty() ? std::string("CUDA kernel compilation failed.") : compile_log;
            nvrtcDestroyProgram(&program);
            compiled = false;
            return false;
        }

        size_t ptx_size = 0;
        if (const nvrtcResult ptx_size_result = nvrtcGetPTXSize(program, &ptx_size); ptx_size_result != NVRTC_SUCCESS) {
            error_out = std::string("Failed to get PTX size: ") + nvrtcGetErrorString(ptx_size_result);
            nvrtcDestroyProgram(&program);
            compiled = false;
            return false;
        }

        std::vector<char> ptx(ptx_size);
        if (const nvrtcResult ptx_result = nvrtcGetPTX(program, ptx.data()); ptx_result != NVRTC_SUCCESS) {
            error_out = std::string("Failed to get PTX: ") + nvrtcGetErrorString(ptx_result);
            nvrtcDestroyProgram(&program);
            compiled = false;
            return false;
        }

        nvrtcDestroyProgram(&program);

        if (module != nullptr) {
            cuModuleUnload(module);
            module = nullptr;
        }

        if (const CUresult module_result = cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr); module_result != CUDA_SUCCESS) {
            const char* err_str = nullptr;
            cuGetErrorString(module_result, &err_str);
            error_out = std::string("Failed to load CUDA module: ") + (err_str ? err_str : "unknown error");
            compiled = false;
            return false;
        }

        if (const CUresult function_result = cuModuleGetFunction(&kernel, module, "mainImageKernel"); function_result != CUDA_SUCCESS) {
            const char* err_str = nullptr;
            cuGetErrorString(function_result, &err_str);
            error_out = std::string("Failed to get kernel function: ") + (err_str ? err_str : "unknown error");
            compiled = false;
            return false;
        }

        compiled = true;
        error_out.clear();
        return true;
    }

    bool launch(uchar4* device_output, int width, int height, const Globals& globals, std::string& error_out)
    {
        if (!compiled || kernel == nullptr) {
            error_out = "CUDA kernel is not compiled.";
            return false;
        }

        struct Uniforms {
            float iTime;
            float iTimeDelta;
            int iFrame;
            float iMouse[4];
            float iResolution[3];
            float iDate[4];
        } uniforms = {};

        uniforms.iTime = globals.iTime;
        uniforms.iTimeDelta = globals.iTimeDelta;
        uniforms.iFrame = globals.iFrame;
        std::memcpy(uniforms.iMouse, globals.iMouse, sizeof(uniforms.iMouse));
        std::memcpy(uniforms.iResolution, globals.iResolution, sizeof(uniforms.iResolution));
        std::memcpy(uniforms.iDate, globals.iDate, sizeof(uniforms.iDate));

        const dim3 block_size(16, 16);
        const dim3 grid_size(
            (width + block_size.x - 1) / block_size.x,
            (height + block_size.y - 1) / block_size.y
        );

        void* args[] = { &device_output, &width, &height, &uniforms };
        if (const CUresult launch_result = cuLaunchKernel(
            kernel,
            grid_size.x, grid_size.y, 1,
            block_size.x, block_size.y, 1,
            0, nullptr, args, nullptr
        ); launch_result != CUDA_SUCCESS) {
            const char* err_str = nullptr;
            cuGetErrorString(launch_result, &err_str);
            error_out = std::string("Failed to launch CUDA kernel: ") + (err_str ? err_str : "unknown error");
            return false;
        }

        if (const cudaError_t sync_result = cudaDeviceSynchronize(); sync_result != cudaSuccess) {
            error_out = std::string("CUDA device synchronize failed: ") + cudaGetErrorString(sync_result);
            return false;
        }

        error_out.clear();
        return true;
    }
};

class GLTexture {
public:
    GLuint texture_id = 0;
    GLuint pbo_id = 0;
    cudaGraphicsResource_t cuda_resource = nullptr;
    int width = 0;
    int height = 0;
    uchar4* device_pixels = nullptr;

    ~GLTexture()
    {
        cleanup();
    }

    void cleanup()
    {
        if (cuda_resource != nullptr) {
            cudaGraphicsUnregisterResource(cuda_resource);
            cuda_resource = nullptr;
        }
        if (pbo_id != 0) {
            glDeleteBuffers(1, &pbo_id);
            pbo_id = 0;
        }
        if (texture_id != 0) {
            glDeleteTextures(1, &texture_id);
            texture_id = 0;
        }
        device_pixels = nullptr;
    }

    bool init(int w, int h, std::string& error_out)
    {
        cleanup();

        width = std::max(w, 1);
        height = std::max(h, 1);

        glGenTextures(1, &texture_id);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        glGenBuffers(1, &pbo_id);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        if (const cudaError_t register_result = cudaGraphicsGLRegisterBuffer(&cuda_resource, pbo_id, cudaGraphicsMapFlagsWriteDiscard);
            register_result != cudaSuccess) {
            error_out = std::string("Failed to register CUDA/GL buffer: ") + cudaGetErrorString(register_result);
            cleanup();
            return false;
        }

        error_out.clear();
        return true;
    }

    void resize(int w, int h, std::string& error_out)
    {
        if (w != width || h != height || texture_id == 0 || pbo_id == 0) {
            init(w, h, error_out);
        }
        else {
            error_out.clear();
        }
    }

    uchar4* map_cuda(std::string& error_out)
    {
        if (const cudaError_t map_result = cudaGraphicsMapResources(1, &cuda_resource, 0); map_result != cudaSuccess) {
            error_out = std::string("Failed to map CUDA graphics resource: ") + cudaGetErrorString(map_result);
            return nullptr;
        }

        size_t size = 0;
        if (const cudaError_t pointer_result = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&device_pixels), &size, cuda_resource);
            pointer_result != cudaSuccess) {
            error_out = std::string("Failed to get mapped CUDA pointer: ") + cudaGetErrorString(pointer_result);
            cudaGraphicsUnmapResources(1, &cuda_resource, 0);
            device_pixels = nullptr;
            return nullptr;
        }

        error_out.clear();
        return device_pixels;
    }

    void unmap_cuda()
    {
        if (cuda_resource != nullptr) {
            cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        }
        device_pixels = nullptr;
    }

    void update_texture() const
    {
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id);
        glBindTexture(GL_TEXTURE_2D, texture_id);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
};

void load_programs_from_folder(const std::string& folder)
{
    g_named_cuda_code.clear();

    const fs::path resolved_folder = SHORTHANDS::resolve_project_path(folder);
    if (!fs::exists(resolved_folder)) {
        fs::create_directories(resolved_folder);
        return;
    }

    for (const auto& entry : fs::directory_iterator(resolved_folder)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string extension = entry.path().extension().string();
        if (extension != ".cu" && extension != ".cuh") {
            continue;
        }

        std::ifstream file(entry.path(), std::ios::in | std::ios::binary);
        if (!file) {
            continue;
        }

        const std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        g_named_cuda_code[entry.path().stem().string()] = content;
    }
}

} // namespace CudaBridge

} // namespace

void load_cuda_programs(const std::string& folder)
{
    CudaBridge::load_programs_from_folder(folder);
}

void load_optix_programs(const std::string& folder)
{
    OPTIX_RT::load_optix_programs(folder);
}

std::vector<std::string> cuda_program_names()
{
    std::vector<std::string> names;
    names.reserve(CudaBridge::g_named_cuda_code.size());
    for (const auto& entry : CudaBridge::g_named_cuda_code) {
        names.push_back(entry.first);
    }
    return names;
}

std::vector<std::string> optix_program_names()
{
    std::vector<std::string> names;
    names.reserve(OPTIX_RT::named_optix_code.size());
    for (const auto& entry : OPTIX_RT::named_optix_code) {
        names.push_back(entry.first);
    }
    return names;
}

std::string cuda_program_source(const std::string& name)
{
    const auto it = CudaBridge::g_named_cuda_code.find(name);
    return it == CudaBridge::g_named_cuda_code.end() ? std::string() : it->second;
}

std::string optix_program_source(const std::string& name)
{
    const auto it = OPTIX_RT::named_optix_code.find(name);
    return it == OPTIX_RT::named_optix_code.end() ? std::string() : it->second;
}

struct CUDA_Node_Runtime::Impl {
    CudaBridge::Kernel saved_kernel;
    CudaBridge::Kernel project_kernel;
    CudaBridge::GLTexture output_texture;
    CudaBridge::Globals globals;
    std::string last_error;
    bool initialized = false;
    bool saved_valid = false;
    bool project_active = false;

    bool ensure_ready(int width, int height)
    {
        width = std::max(width, 1);
        height = std::max(height, 1);

        if (!CudaBridge::init_cuda(last_error)) {
            return false;
        }

        if (!initialized) {
            if (!output_texture.init(width, height, last_error)) {
                return false;
            }
            initialized = true;
        }
        else {
            output_texture.resize(width, height, last_error);
            if (!last_error.empty()) {
                return false;
            }
        }

        globals.iResolution[0] = static_cast<float>(width);
        globals.iResolution[1] = static_cast<float>(height);
        globals.iResolution[2] = 1.0f;
        return true;
    }
};

CUDA_Node_Runtime::CUDA_Node_Runtime()
    : impl_(std::make_unique<Impl>())
{
}

CUDA_Node_Runtime::~CUDA_Node_Runtime() = default;
CUDA_Node_Runtime::CUDA_Node_Runtime(CUDA_Node_Runtime&&) noexcept = default;
CUDA_Node_Runtime& CUDA_Node_Runtime::operator=(CUDA_Node_Runtime&&) noexcept = default;

bool CUDA_Node_Runtime::compile_saved(const std::string& source, int width, int height)
{
    if (!impl_->ensure_ready(width, height)) {
        return false;
    }

    if (!impl_->saved_kernel.compile(source, impl_->last_error)) {
        impl_->saved_valid = false;
        return false;
    }

    impl_->saved_valid = true;
    impl_->project_active = false;
    impl_->last_error.clear();
    return true;
}

bool CUDA_Node_Runtime::compile_project(const std::string& source, int width, int height)
{
    if (!impl_->ensure_ready(width, height)) {
        return false;
    }

    if (!impl_->project_kernel.compile(source, impl_->last_error)) {
        return false;
    }

    impl_->project_active = true;
    impl_->last_error.clear();
    return true;
}

void CUDA_Node_Runtime::clear_project()
{
    if (!impl_->project_active) {
        return;
    }
    impl_->project_active = false;
}

bool CUDA_Node_Runtime::has_project_override() const
{
    return impl_ && impl_->project_active;
}

bool CUDA_Node_Runtime::render(int width, int height)
{
    if (!impl_->ensure_ready(width, height)) {
        return false;
    }

    if (!impl_->saved_valid && !impl_->project_active) {
        if (!compile_saved("", width, height)) {
            return false;
        }
    }

    CudaBridge::Kernel* active_kernel =
        (impl_->project_active && impl_->project_kernel.compiled) ? &impl_->project_kernel : &impl_->saved_kernel;
    if (!active_kernel->compiled) {
        impl_->last_error = "CUDA kernel is not compiled.";
        return false;
    }

    impl_->globals.update();

    uchar4* device_pixels = impl_->output_texture.map_cuda(impl_->last_error);
    if (device_pixels == nullptr) {
        return false;
    }

    const bool launched = active_kernel->launch(
        device_pixels,
        impl_->output_texture.width,
        impl_->output_texture.height,
        impl_->globals,
        impl_->last_error
    );
    impl_->output_texture.unmap_cuda();

    if (!launched) {
        return false;
    }

    impl_->output_texture.update_texture();
    glFlush();
    return true;
}

GLuint CUDA_Node_Runtime::texture() const
{
    return impl_->output_texture.texture_id;
}

void CUDA_Node_Runtime::update_mouse(double xpos, double ypos, bool pressed, int height)
{
    const float mouse_x = static_cast<float>(xpos);
    const float mouse_y = static_cast<float>(ypos);

    if (pressed) {
        impl_->globals.iMouse[0] = mouse_x;
        impl_->globals.iMouse[1] = mouse_y;
        if (impl_->globals.iMouse[2] == 0.0f && impl_->globals.iMouse[3] == 0.0f) {
            impl_->globals.iMouse[2] = mouse_x;
            impl_->globals.iMouse[3] = mouse_y;
        }
    }
}

void CUDA_Node_Runtime::update_mouse_click(double xpos, double ypos, bool pressed, int height)
{
    const float mouse_x = static_cast<float>(xpos);
    const float mouse_y = static_cast<float>(ypos);

    impl_->globals.iMouse[0] = mouse_x;
    impl_->globals.iMouse[1] = mouse_y;

    if (pressed) {
        impl_->globals.iMouse[2] = mouse_x;
        impl_->globals.iMouse[3] = mouse_y;
    }
    else {
        impl_->globals.iMouse[2] = -std::abs(impl_->globals.iMouse[2]);
        impl_->globals.iMouse[3] = -std::abs(impl_->globals.iMouse[3]);
    }
}

const std::string& CUDA_Node_Runtime::last_error() const
{
    return impl_ ? impl_->last_error : empty_string();
}

struct OptiX_Node_Runtime::Impl {
    OptixModuleManager module_manager;
    OptixPipelineBuilder pipeline_builder;
    CudaBridge::GLTexture output_texture;

    uchar4* device_output = nullptr;
    unsigned int output_width = 0;
    unsigned int output_height = 0;

    float current_time = 0.0f;
    float delta_time = 0.0f;
    std::chrono::steady_clock::time_point last_frame_time;
    unsigned int frame_index = 0;

    float mouse_x = 0.0f;
    float mouse_y = 0.0f;
    int mouse_buttons = 0;

    float3 camera_pos = { 0.0f, 0.0f, -3.0f };
    float3 camera_target = { 0.0f, 0.0f, 0.0f };
    float camera_fov = 45.0f;
    cudaGraphicsResource_t input_resource = nullptr;
    cudaTextureObject_t input_texture_object = 0;
    bool input_resource_mapped = false;
    GLuint input_gl_texture = 0;
    unsigned int input_width = 0;
    unsigned int input_height = 0;

    std::string saved_label = "saved_optix";
    std::string saved_source;
    std::string project_label = "project_optix";
    std::string project_source;
    std::string last_error;

    bool initialized = false;
    bool program_valid = false;
    bool saved_valid = false;
    bool project_active = false;

    ~Impl()
    {
        cleanup();
    }

    void cleanup()
    {
        release_input_texture();

        if (device_output != nullptr) {
            cudaFree(device_output);
            device_output = nullptr;
        }

        pipeline_builder.cleanup();
        module_manager.cleanup();
        output_texture.cleanup();
        initialized = false;
        program_valid = false;
        saved_valid = false;
        project_active = false;
    }

    bool ensure_ready(int width, int height)
    {
        width = std::max(width, 1);
        height = std::max(height, 1);

        if (!CudaBridge::init_cuda(last_error)) {
            return false;
        }

        if (!initialized) {
            if (!g_optixContext.init()) {
                last_error = "Failed to initialize OptiX context.";
                return false;
            }

            module_manager.init(g_optixContext.getContext());
            pipeline_builder.init(g_optixContext.getContext());
            last_frame_time = std::chrono::steady_clock::now();
            initialized = true;
        }

        if (width != static_cast<int>(output_width) || height != static_cast<int>(output_height) || device_output == nullptr) {
            resize(width, height);
        }

        return last_error.empty();
    }

    void resize(int width, int height)
    {
        last_error.clear();

        if (device_output != nullptr) {
            cudaFree(device_output);
            device_output = nullptr;
        }

        output_width = static_cast<unsigned int>(width);
        output_height = static_cast<unsigned int>(height);

        if (const cudaError_t alloc_result = cudaMalloc(&device_output, output_width * output_height * sizeof(uchar4)); alloc_result != cudaSuccess) {
            last_error = std::string("Failed to allocate OptiX output buffer: ") + cudaGetErrorString(alloc_result);
            return;
        }

        if (const cudaError_t memset_result = cudaMemset(device_output, 0, output_width * output_height * sizeof(uchar4)); memset_result != cudaSuccess) {
            last_error = std::string("Failed to clear OptiX output buffer: ") + cudaGetErrorString(memset_result);
            return;
        }

        output_texture.cleanup();
        if (!output_texture.init(width, height, last_error)) {
            return;
        }

        frame_index = 0;
    }

    void release_input_texture_object()
    {
        if (input_texture_object != 0) {
            cudaDestroyTextureObject(input_texture_object);
            input_texture_object = 0;
        }
    }

    void release_input_texture()
    {
        release_input_texture_object();

        if (input_resource_mapped && input_resource != nullptr) {
            cudaGraphicsUnmapResources(1, &input_resource, 0);
            input_resource_mapped = false;
        }

        if (input_resource != nullptr) {
            cudaGraphicsUnregisterResource(input_resource);
            input_resource = nullptr;
        }

        input_gl_texture = 0;
        input_width = 0;
        input_height = 0;
    }

    bool set_input_texture(GLuint texture, int width, int height)
    {
        width = std::max(width, 0);
        height = std::max(height, 0);

        if (texture == 0 || width <= 0 || height <= 0) {
            release_input_texture();
            last_error.clear();
            return true;
        }

        if (texture != input_gl_texture || input_resource == nullptr) {
            release_input_texture();

            if (const cudaError_t register_result = cudaGraphicsGLRegisterImage(
                &input_resource,
                texture,
                GL_TEXTURE_2D,
                cudaGraphicsRegisterFlagsReadOnly
            ); register_result != cudaSuccess) {
                last_error = std::string("Failed to register OptiX input texture: ") + cudaGetErrorString(register_result);
                input_resource = nullptr;
                return false;
            }

            input_gl_texture = texture;
        }

        input_width = static_cast<unsigned int>(width);
        input_height = static_cast<unsigned int>(height);
        last_error.clear();
        return true;
    }

    bool map_input_texture(OptixLaunchParams& params)
    {
        params.inputTexture = 0;
        params.inputWidth = input_width;
        params.inputHeight = input_height;

        if (input_resource == nullptr) {
            return true;
        }

        if (const cudaError_t map_result = cudaGraphicsMapResources(1, &input_resource, 0); map_result != cudaSuccess) {
            last_error = std::string("Failed to map OptiX input texture: ") + cudaGetErrorString(map_result);
            return false;
        }

        input_resource_mapped = true;

        cudaArray_t mapped_array = nullptr;
        if (const cudaError_t array_result = cudaGraphicsSubResourceGetMappedArray(&mapped_array, input_resource, 0, 0); array_result != cudaSuccess) {
            last_error = std::string("Failed to access OptiX input texture array: ") + cudaGetErrorString(array_result);
            release_input_texture_object();
            cudaGraphicsUnmapResources(1, &input_resource, 0);
            input_resource_mapped = false;
            return false;
        }

        cudaResourceDesc resource_desc = {};
        resource_desc.resType = cudaResourceTypeArray;
        resource_desc.res.array.array = mapped_array;

        cudaTextureDesc texture_desc = {};
        texture_desc.addressMode[0] = cudaAddressModeClamp;
        texture_desc.addressMode[1] = cudaAddressModeClamp;
        texture_desc.filterMode = cudaFilterModeLinear;
        texture_desc.readMode = cudaReadModeElementType;
        texture_desc.normalizedCoords = 1;

        release_input_texture_object();
        if (const cudaError_t texture_result = cudaCreateTextureObject(&input_texture_object, &resource_desc, &texture_desc, nullptr); texture_result != cudaSuccess) {
            last_error = std::string("Failed to create OptiX input sampler: ") + cudaGetErrorString(texture_result);
            cudaGraphicsUnmapResources(1, &input_resource, 0);
            input_resource_mapped = false;
            return false;
        }

        params.inputTexture = input_texture_object;
        return true;
    }

    void unmap_input_texture()
    {
        release_input_texture_object();

        if (input_resource_mapped && input_resource != nullptr) {
            cudaGraphicsUnmapResources(1, &input_resource, 0);
            input_resource_mapped = false;
        }
    }

    static float3 cross3(float3 a, float3 b)
    {
        return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    static float3 normalize3(float3 v)
    {
        const float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        if (len <= 0.0f) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }
        return make_float3(v.x / len, v.y / len, v.z / len);
    }

    bool build_pipeline(const std::string& module_name, const std::string& source)
    {
        const std::string full_source = OPTIX_RT::OPTIX_PROGRAM_HEADER + source + OPTIX_RT::OPTIX_PROGRAM_FOOTER;
        std::string error_log;

        if (!module_manager.compileAndCreate(module_name, full_source, error_log)) {
            last_error = error_log.empty() ? "OptiX compilation failed." : error_log;
            program_valid = false;
            return false;
        }

        const OptixModule module = module_manager.getModule(module_name);
        if (!pipeline_builder.buildPipeline(
            module,
            "__raygen__main",
            "__miss__main",
            "__closesthit__main",
            module_manager.getPipelineCompileOptions(),
            error_log
        )) {
            last_error = error_log.empty() ? "OptiX pipeline creation failed." : error_log;
            program_valid = false;
            return false;
        }

        last_error.clear();
        program_valid = true;
        frame_index = 0;
        return true;
    }
};

OptiX_Node_Runtime::OptiX_Node_Runtime()
    : impl_(std::make_unique<Impl>())
{
}

OptiX_Node_Runtime::~OptiX_Node_Runtime() = default;
OptiX_Node_Runtime::OptiX_Node_Runtime(OptiX_Node_Runtime&&) noexcept = default;
OptiX_Node_Runtime& OptiX_Node_Runtime::operator=(OptiX_Node_Runtime&&) noexcept = default;

bool OptiX_Node_Runtime::compile_saved(const std::string& label, const std::string& source, int width, int height)
{
    if (!impl_->ensure_ready(width, height)) {
        return false;
    }

    impl_->saved_label = label.empty() ? "saved_optix" : label;
    impl_->saved_source = source;
    impl_->saved_valid = impl_->build_pipeline(impl_->saved_label, impl_->saved_source);
    impl_->project_active = false;
    return impl_->saved_valid;
}

bool OptiX_Node_Runtime::compile_project(const std::string& label, const std::string& source, int width, int height)
{
    if (!impl_->ensure_ready(width, height)) {
        return false;
    }

    impl_->project_label = label.empty() ? "project_optix" : label;
    impl_->project_source = source;
    if (!impl_->build_pipeline(impl_->project_label, impl_->project_source)) {
        return false;
    }

    impl_->project_active = true;
    return true;
}

void OptiX_Node_Runtime::clear_project()
{
    if (!impl_->project_active) {
        return;
    }
    impl_->project_active = false;
    if (impl_->saved_valid) {
        impl_->build_pipeline(impl_->saved_label, impl_->saved_source);
    }
}

bool OptiX_Node_Runtime::has_project_override() const
{
    return impl_ && impl_->project_active;
}

bool OptiX_Node_Runtime::render(int width, int height)
{
    if (!impl_->ensure_ready(width, height)) {
        return false;
    }

    if (!impl_->program_valid && impl_->saved_valid) {
        if (!impl_->build_pipeline(impl_->saved_label, impl_->saved_source)) {
            return false;
        }
    }

    if (!impl_->program_valid || impl_->device_output == nullptr) {
        impl_->last_error = "OptiX program is not ready.";
        return false;
    }

    const auto now = std::chrono::steady_clock::now();
    impl_->delta_time = std::chrono::duration<float>(now - impl_->last_frame_time).count();
    impl_->last_frame_time = now;
    impl_->current_time += impl_->delta_time;

    const float3 cam_dir = Impl::normalize3(make_float3(
        impl_->camera_target.x - impl_->camera_pos.x,
        impl_->camera_target.y - impl_->camera_pos.y,
        impl_->camera_target.z - impl_->camera_pos.z
    ));
    const float3 world_up = make_float3(0.0f, 1.0f, 0.0f);
    const float3 cam_right = Impl::normalize3(Impl::cross3(cam_dir, world_up));
    const float3 cam_up = Impl::cross3(cam_right, cam_dir);

    OptixLaunchParams params = {};
    params.outputBuffer = impl_->device_output;
    params.width = impl_->output_width;
    params.height = impl_->output_height;
    params.time = impl_->current_time;
    params.deltaTime = impl_->delta_time;
    params.mouseX = impl_->mouse_x;
    params.mouseY = impl_->mouse_y;
    params.mouseButtons = impl_->mouse_buttons;
    params.camPos = impl_->camera_pos;
    params.camDir = cam_dir;
    params.camUp = cam_up;
    params.camRight = cam_right;
    params.fov = impl_->camera_fov;
    params.frameIndex = impl_->frame_index;
    params.traversable = 0;

    if (!impl_->map_input_texture(params)) {
        return false;
    }

    if (!impl_->pipeline_builder.launch(params)) {
        impl_->unmap_input_texture();
        impl_->last_error = "OptiX launch failed.";
        return false;
    }

    if (const cudaError_t sync_result = cudaDeviceSynchronize(); sync_result != cudaSuccess) {
        impl_->unmap_input_texture();
        impl_->last_error = std::string("OptiX synchronize failed: ") + cudaGetErrorString(sync_result);
        return false;
    }

    uchar4* mapped = impl_->output_texture.map_cuda(impl_->last_error);
    if (mapped == nullptr) {
        impl_->unmap_input_texture();
        return false;
    }

    const cudaError_t copy_result = cudaMemcpy(
        mapped,
        impl_->device_output,
        impl_->output_width * impl_->output_height * sizeof(uchar4),
        cudaMemcpyDeviceToDevice
    );
    impl_->output_texture.unmap_cuda();
    impl_->unmap_input_texture();

    if (copy_result != cudaSuccess) {
        impl_->last_error = std::string("Failed to copy OptiX output texture: ") + cudaGetErrorString(copy_result);
        return false;
    }

    impl_->output_texture.update_texture();
    glFlush();
    impl_->frame_index++;
    return true;
}

GLuint OptiX_Node_Runtime::texture() const
{
    return impl_->output_texture.texture_id;
}

bool OptiX_Node_Runtime::set_input_texture(GLuint texture, int width, int height)
{
    if (!impl_) {
        return false;
    }

    return impl_->set_input_texture(texture, width, height);
}

void OptiX_Node_Runtime::update_mouse(double xpos, double ypos, int width, int height)
{
    if (width <= 0 || height <= 0) {
        return;
    }

    impl_->mouse_x = static_cast<float>(xpos) / static_cast<float>(width);
    impl_->mouse_y = 1.0f - static_cast<float>(ypos) / static_cast<float>(height);
}

void OptiX_Node_Runtime::update_mouse_button(int button, int action)
{
    if (action == GLFW_PRESS) {
        impl_->mouse_buttons |= (1 << button);
    }
    else {
        impl_->mouse_buttons &= ~(1 << button);
    }
}

const std::string& OptiX_Node_Runtime::last_error() const
{
    return impl_ ? impl_->last_error : empty_string();
}

} // namespace LegacyNodeRuntime
