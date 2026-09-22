#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <nvrtc.h>
#include <cufft.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>

#include <fmt_related.hpp>
#include <shorthands.hpp>

using namespace SHORTHANDS;
using namespace FMT;

namespace CUDA_RT
{
    // Error checking macros
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                bad("CUDA Error: {} at {}:{}", cudaGetErrorString(err), __FILE__, __LINE__); \
            } \
        } while(0)

    #define NVRTC_CHECK(call) \
        do { \
            nvrtcResult err = call; \
            if (err != NVRTC_SUCCESS) { \
                bad("NVRTC Error: {} at {}:{}", nvrtcGetErrorString(err), __FILE__, __LINE__); \
            } \
        } while(0)

    #define CU_CHECK(call) \
        do { \
            CUresult err = call; \
            if (err != CUDA_SUCCESS) { \
                const char* errStr; \
                cuGetErrorString(err, &errStr); \
                bad("CUDA Driver Error: {} at {}:{}", errStr, __FILE__, __LINE__); \
            } \
        } while(0)

    // Shared uniforms for CUDA kernels (similar to shader uniforms)
    struct CUDAGlobals {
        float iTime = 0.0f;
        float iTimeDelta = 0.0f;
        int iFrame = 0;
        float iMouse[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float iResolution[3] = {800.0f, 800.0f, 1.0f};
        float iDate[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        
        std::chrono::high_resolution_clock::time_point startTime;
        std::chrono::high_resolution_clock::time_point lastFrameTime;
        
        CUDAGlobals() {
            startTime = std::chrono::high_resolution_clock::now();
            lastFrameTime = startTime;
        }
        
        void update() {
            auto currentTime = std::chrono::high_resolution_clock::now();
            iTime = std::chrono::duration<float>(currentTime - startTime).count();
            iTimeDelta = std::chrono::duration<float>(currentTime - lastFrameTime).count();
            lastFrameTime = currentTime;
            iFrame++;
            
            // Update date
            std::time_t t = std::time(nullptr);
            std::tm now;
            localtime_s(&now, &t);
            iDate[0] = (float)(now.tm_year + 1900);
            iDate[1] = (float)(now.tm_mon + 1);
            iDate[2] = (float)now.tm_mday;
            iDate[3] = (float)(now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec);
        }
    };

    // Default kernel code that will be prepended to user code
    inline const char* CUDA_KERNEL_HEADER = R"(
// Shader-like uniforms available to the kernel
struct Uniforms {
    float iTime;
    float iTimeDelta;
    int iFrame;
    float iMouse[4];
    float iResolution[3];
    float iDate[4];
};

// Helper functions similar to GLSL
__device__ inline float fract(float x) { return x - floorf(x); }
__device__ inline float mix(float a, float b, float t) { return a + (b - a) * t; }
__device__ inline float clamp(float x, float minVal, float maxVal) { return fminf(fmaxf(x, minVal), maxVal); }
__device__ inline float smoothstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
__device__ inline float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
__device__ inline float mod(float x, float y) { return x - y * floorf(x / y); }

// Vector math helpers
__device__ inline float dot2(float x1, float y1, float x2, float y2) { return x1*x2 + y1*y2; }
__device__ inline float dot3(float x1, float y1, float z1, float x2, float y2, float z2) { return x1*x2 + y1*y2 + z1*z2; }
__device__ inline float length2(float x, float y) { return sqrtf(x*x + y*y); }
__device__ inline float length3(float x, float y, float z) { return sqrtf(x*x + y*y + z*z); }
__device__ inline void normalize2(float& x, float& y) { float len = length2(x, y); if (len > 0) { x /= len; y /= len; } }
__device__ inline void normalize3(float& x, float& y, float& z) { float len = length3(x, y, z); if (len > 0) { x /= len; y /= len; z /= len; } }

// Color conversion
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

    // Default kernel footer with main kernel entry point
    inline const char* CUDA_KERNEL_FOOTER = R"(

// Main kernel - launches one thread per pixel
extern "C" __global__ void mainImageKernel(uchar4* output, int width, int height, Uniforms uniforms) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Flip y to match OpenGL coordinate system
    int flipped_y = height - 1 - y;
    
    float fragColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    float fragCoord[2] = {(float)x, (float)flipped_y};
    
    // Call user-defined mainImage function
    mainImage(fragColor, fragCoord, uniforms);
    
    // Convert to uchar4 (RGBA)
    int idx = y * width + x;
    output[idx] = make_uchar4(
        (unsigned char)(clamp(fragColor[0], 0.0f, 1.0f) * 255.0f),
        (unsigned char)(clamp(fragColor[1], 0.0f, 1.0f) * 255.0f),
        (unsigned char)(clamp(fragColor[2], 0.0f, 1.0f) * 255.0f),
        (unsigned char)(clamp(fragColor[3], 0.0f, 1.0f) * 255.0f)
    );
}
)";

    class CUDAKernel {
    public:
        CUmodule module = nullptr;
        CUfunction kernel = nullptr;
        str rawCode;
        str processedCode;
        str compileLog;
        bool compiled = false;
        
        ~CUDAKernel() {
            if (module) {
                cuModuleUnload(module);
                module = nullptr;
            }
        }
        
        str processCode(const str& userCode) {
            // Check if user code has mainImage function
            bool hasMainImage = userCode.find("mainImage") != std::string::npos;
            
            str result = CUDA_KERNEL_HEADER;
            result += "\n// ========== USER CODE BEGIN ==========\n";
            result += userCode;
            result += "\n// ========== USER CODE END ==========\n";
            
            if (!hasMainImage) {
                // Add default mainImage if not present
                result += R"(
// Default mainImage - checkerboard pattern
__device__ void mainImage(float fragColor[4], float fragCoord[2], const Uniforms& u) {
    float2 uv;
    uv.x = fragCoord[0] / u.iResolution[0];
    uv.y = fragCoord[1] / u.iResolution[1];
    
    // Animated black-red checkerboard
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
            
            result += CUDA_KERNEL_FOOTER;
            return result;
        }
        
        bool compile(const str& userCode) {
            highlight("=== Starting CUDA kernel compilation ===");
            rawCode = userCode;
            processedCode = processCode(userCode);
            
            highlight("Processed code length: {} chars", processedCode.length());
            
            // Create NVRTC program
            nvrtcProgram prog;
            nvrtcResult createResult = nvrtcCreateProgram(&prog, processedCode.c_str(), "kernel.cu", 0, nullptr, nullptr);
            if (createResult != NVRTC_SUCCESS) {
                bad("Failed to create NVRTC program: {}", nvrtcGetErrorString(createResult));
                compiled = false;
                return false;
            }
            highlight("NVRTC program created");
            
            // Compile options - use compute_52 for broader compatibility
            // (works on Maxwell and newer: GTX 900+, all RTX cards)
            const char* opts[] = {
                "--gpu-architecture=compute_52",
                "-default-device"
            };
            int numOpts = sizeof(opts) / sizeof(opts[0]);
            
            highlight("Compiling with {} options...", numOpts);
            nvrtcResult compileResult = nvrtcCompileProgram(prog, numOpts, opts);
            
            // Get compile log
            size_t logSize;
            nvrtcGetProgramLogSize(prog, &logSize);
            compileLog.resize(logSize);
            nvrtcGetProgramLog(prog, &compileLog[0]);
            
            // Always print compile log for debugging
            if (logSize > 1) {
                highlight("NVRTC Compile log ({} bytes):\n{}", logSize, compileLog);
            }
            
            if (compileResult != NVRTC_SUCCESS) {
                bad("CUDA Kernel compilation failed: {}", nvrtcGetErrorString(compileResult));
                bad("Full compile log:\n{}", compileLog);
                nvrtcDestroyProgram(&prog);
                compiled = false;
                return false;
            }
            
            good("NVRTC compilation successful");
            
            // Get PTX
            size_t ptxSize;
            nvrtcResult ptxSizeResult = nvrtcGetPTXSize(prog, &ptxSize);
            if (ptxSizeResult != NVRTC_SUCCESS) {
                bad("Failed to get PTX size: {}", nvrtcGetErrorString(ptxSizeResult));
                nvrtcDestroyProgram(&prog);
                compiled = false;
                return false;
            }
            
            std::vector<char> ptx(ptxSize);
            nvrtcResult ptxResult = nvrtcGetPTX(prog, ptx.data());
            if (ptxResult != NVRTC_SUCCESS) {
                bad("Failed to get PTX: {}", nvrtcGetErrorString(ptxResult));
                nvrtcDestroyProgram(&prog);
                compiled = false;
                return false;
            }
            
            highlight("PTX generated: {} bytes", ptxSize);
            nvrtcDestroyProgram(&prog);
            
            // Load module
            if (module) {
                cuModuleUnload(module);
                module = nullptr;
            }
            
            CUresult modResult = cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr);
            if (modResult != CUDA_SUCCESS) {
                const char* errStr;
                cuGetErrorString(modResult, &errStr);
                bad("Failed to load CUDA module: {}", errStr);
                compiled = false;
                return false;
            }
            
            CUresult funcResult = cuModuleGetFunction(&kernel, module, "mainImageKernel");
            if (funcResult != CUDA_SUCCESS) {
                const char* errStr;
                cuGetErrorString(funcResult, &errStr);
                bad("Failed to get kernel function: {}", errStr);
                compiled = false;
                return false;
            }
            
            compiled = true;
            good("=== CUDA Kernel compiled successfully! ===");
            return true;
        }
        
        void launch(uchar4* d_output, int width, int height, const CUDAGlobals& globals) {
            if (!compiled || !kernel) {
                bad("Kernel not compiled!");
                return;
            }
            
            // Prepare uniforms struct
            struct Uniforms {
                float iTime;
                float iTimeDelta;
                int iFrame;
                float iMouse[4];
                float iResolution[3];
                float iDate[4];
            } uniforms;
            
            uniforms.iTime = globals.iTime;
            uniforms.iTimeDelta = globals.iTimeDelta;
            uniforms.iFrame = globals.iFrame;
            memcpy(uniforms.iMouse, globals.iMouse, sizeof(uniforms.iMouse));
            memcpy(uniforms.iResolution, globals.iResolution, sizeof(uniforms.iResolution));
            memcpy(uniforms.iDate, globals.iDate, sizeof(uniforms.iDate));
            
            // Launch kernel
            dim3 blockSize(16, 16);
            dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                          (height + blockSize.y - 1) / blockSize.y);
            
            void* args[] = { &d_output, &width, &height, &uniforms };
            
            CU_CHECK(cuLaunchKernel(kernel,
                gridSize.x, gridSize.y, 1,
                blockSize.x, blockSize.y, 1,
                0, nullptr,
                args, nullptr));
            
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    };

    // CUDA-OpenGL interop texture
    class CUDAGLTexture {
    public:
        GLuint textureID = 0;
        GLuint pboID = 0;
        cudaGraphicsResource_t cudaResource = nullptr;
        int width = 0;
        int height = 0;
        uchar4* d_pixels = nullptr;
        
        ~CUDAGLTexture() {
            cleanup();
        }
        
        void cleanup() {
            if (cudaResource) {
                cudaGraphicsUnregisterResource(cudaResource);
                cudaResource = nullptr;
            }
            if (pboID) {
                glDeleteBuffers(1, &pboID);
                pboID = 0;
            }
            if (textureID) {
                glDeleteTextures(1, &textureID);
                textureID = 0;
            }
            d_pixels = nullptr;
        }
        
        bool init(int w, int h) {
            cleanup();
            
            width = w;
            height = h;
            
            // Create OpenGL texture
            glGenTextures(1, &textureID);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindTexture(GL_TEXTURE_2D, 0);
            
            // Create PBO for CUDA-GL interop
            glGenBuffers(1, &pboID);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
            glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            
            // Register PBO with CUDA
            CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaResource, pboID, cudaGraphicsMapFlagsWriteDiscard));
            
            good("CUDA-GL texture initialized: {}x{}", width, height);
            return true;
        }
        
        void resize(int w, int h) {
            if (w != width || h != height) {
                init(w, h);
            }
        }
        
        uchar4* mapCUDA() {
            CUDA_CHECK(cudaGraphicsMapResources(1, &cudaResource, 0));
            size_t size;
            CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &size, cudaResource));
            return d_pixels;
        }
        
        void unmapCUDA() {
            CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaResource, 0));
            d_pixels = nullptr;
        }
        
        void updateTexture() {
            // Copy from PBO to texture
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    };

    // Simple shader for displaying the CUDA output texture
    class DisplayShader {
    public:
        GLuint program = 0;
        GLuint VAO = 0, VBO = 0, EBO = 0;
        
        ~DisplayShader() {
            cleanup();
        }
        
        void cleanup() {
            if (program) {
                glDeleteProgram(program);
                program = 0;
            }
            if (VAO) {
                glDeleteVertexArrays(1, &VAO);
                VAO = 0;
            }
            if (VBO) {
                glDeleteBuffers(1, &VBO);
                VBO = 0;
            }
            if (EBO) {
                glDeleteBuffers(1, &EBO);
                EBO = 0;
            }
        }
        
        bool init() {
            const char* vertexShaderSrc = R"(
                #version 330 core
                layout (location = 0) in vec3 aPos;
                layout (location = 1) in vec2 aTexCoord;
                out vec2 TexCoord;
                void main() {
                    gl_Position = vec4(aPos, 1.0);
                    TexCoord = aTexCoord;
                }
            )";
            
            const char* fragmentShaderSrc = R"(
                #version 330 core
                in vec2 TexCoord;
                out vec4 FragColor;
                uniform sampler2D screenTexture;
                void main() {
                    FragColor = texture(screenTexture, TexCoord);
                }
            )";
            
            // Compile shaders
            GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(vertexShader, 1, &vertexShaderSrc, nullptr);
            glCompileShader(vertexShader);
            
            int success;
            char infoLog[512];
            glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
                bad("Display vertex shader error: {}", infoLog);
                return false;
            }
            
            GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(fragmentShader, 1, &fragmentShaderSrc, nullptr);
            glCompileShader(fragmentShader);
            
            glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
                bad("Display fragment shader error: {}", infoLog);
                return false;
            }
            
            // Link program
            program = glCreateProgram();
            glAttachShader(program, vertexShader);
            glAttachShader(program, fragmentShader);
            glLinkProgram(program);
            
            glGetProgramiv(program, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(program, 512, nullptr, infoLog);
                bad("Display shader program link error: {}", infoLog);
                return false;
            }
            
            glDeleteShader(vertexShader);
            glDeleteShader(fragmentShader);
            
            // Setup fullscreen quad
            float vertices[] = {
                // positions        // texture coords
                -1.0f,  1.0f, 0.0f, 0.0f, 1.0f, // top left
                 1.0f,  1.0f, 0.0f, 1.0f, 1.0f, // top right
                 1.0f, -1.0f, 0.0f, 1.0f, 0.0f, // bottom right
                -1.0f, -1.0f, 0.0f, 0.0f, 0.0f  // bottom left
            };
            unsigned int indices[] = {
                0, 1, 3,
                1, 2, 3
            };
            
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);
            glGenBuffers(1, &EBO);
            
            glBindVertexArray(VAO);
            
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
            
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            
            glBindVertexArray(0);
            
            good("Display shader initialized!");
            return true;
        }
        
        void draw(GLuint textureID) {
            glUseProgram(program);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textureID);
            glUniform1i(glGetUniformLocation(program, "screenTexture"), 0);
            
            glBindVertexArray(VAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }
    };

    // Initialize CUDA context
    inline bool initCUDA() {
        static bool initialized = false;
        if (initialized) {
            highlight("CUDA already initialized");
            return true;
        }
        
        highlight("=== Initializing CUDA ===");
        
        CUresult initResult = cuInit(0);
        if (initResult != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(initResult, &errStr);
            bad("cuInit failed: {}", errStr ? errStr : "unknown error");
            return false;
        }
        highlight("cuInit successful");
        
        int deviceCount;
        CUresult countResult = cuDeviceGetCount(&deviceCount);
        if (countResult != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(countResult, &errStr);
            bad("cuDeviceGetCount failed: {}", errStr ? errStr : "unknown error");
            return false;
        }
        
        if (deviceCount == 0) {
            bad("No CUDA devices found!");
            return false;
        }
        highlight("Found {} CUDA device(s)", deviceCount);
        
        CUdevice device;
        CUresult devResult = cuDeviceGet(&device, 0);
        if (devResult != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(devResult, &errStr);
            bad("cuDeviceGet failed: {}", errStr ? errStr : "unknown error");
            return false;
        }
        
        char name[256];
        cuDeviceGetName(name, 256, device);
        good("Using CUDA device: {}", name);
        
        // Get compute capability
        int major, minor;
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        highlight("Compute capability: {}.{}", major, minor);
        
        CUcontext context;
        CUresult ctxResult = cuCtxCreate(&context, 0, device);
        if (ctxResult != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(ctxResult, &errStr);
            bad("cuCtxCreate failed: {}", errStr ? errStr : "unknown error");
            return false;
        }
        
        initialized = true;
        good("=== CUDA initialized successfully ===");
        return true;
    }

    // cuFFT error checking
    #define CUFFT_CHECK(call) \
        do { \
            cufftResult err = call; \
            if (err != CUFFT_SUCCESS) { \
                bad("cuFFT Error: {} at {}:{}", (int)err, __FILE__, __LINE__); \
            } \
        } while(0)

    // FFT Convolution helper class for SmoothLife and similar simulations
    class FFTConvolver {
    public:
        int width = 0;
        int height = 0;
        cufftHandle planForward = 0;
        cufftHandle planInverse = 0;
        
        // Device buffers
        float2* d_state = nullptr;        // Current state (complex)
        float2* d_stateFFT = nullptr;     // FFT of state
        float2* d_kernelFFT = nullptr;    // Precomputed FFT of kernel
        float2* d_temp = nullptr;         // Temporary buffer
        float* d_stateReal = nullptr;     // Real-valued state for display
        
        bool initialized = false;
        
        ~FFTConvolver() {
            cleanup();
        }
        
        void cleanup() {
            if (planForward) { cufftDestroy(planForward); planForward = 0; }
            if (planInverse) { cufftDestroy(planInverse); planInverse = 0; }
            if (d_state) { cudaFree(d_state); d_state = nullptr; }
            if (d_stateFFT) { cudaFree(d_stateFFT); d_stateFFT = nullptr; }
            if (d_kernelFFT) { cudaFree(d_kernelFFT); d_kernelFFT = nullptr; }
            if (d_temp) { cudaFree(d_temp); d_temp = nullptr; }
            if (d_stateReal) { cudaFree(d_stateReal); d_stateReal = nullptr; }
            initialized = false;
        }
        
        bool init(int w, int h) {
            if (w == width && h == height && initialized) return true;
            
            cleanup();
            width = w;
            height = h;
            
            size_t complexSize = width * height * sizeof(float2);
            size_t realSize = width * height * sizeof(float);
            
            // Allocate buffers
            CUDA_CHECK(cudaMalloc(&d_state, complexSize));
            CUDA_CHECK(cudaMalloc(&d_stateFFT, complexSize));
            CUDA_CHECK(cudaMalloc(&d_kernelFFT, complexSize));
            CUDA_CHECK(cudaMalloc(&d_temp, complexSize));
            CUDA_CHECK(cudaMalloc(&d_stateReal, realSize));
            
            // Initialize state to zero
            CUDA_CHECK(cudaMemset(d_state, 0, complexSize));
            CUDA_CHECK(cudaMemset(d_kernelFFT, 0, complexSize));
            
            // Create FFT plans
            CUFFT_CHECK(cufftPlan2d(&planForward, height, width, CUFFT_C2C));
            CUFFT_CHECK(cufftPlan2d(&planInverse, height, width, CUFFT_C2C));
            
            initialized = true;
            good("FFTConvolver initialized: {}x{}", width, height);
            return true;
        }
        
        void forward() {
            if (!initialized) return;
            CUFFT_CHECK(cufftExecC2C(planForward, (cufftComplex*)d_state, (cufftComplex*)d_stateFFT, CUFFT_FORWARD));
        }
        
        void inverse() {
            if (!initialized) return;
            CUFFT_CHECK(cufftExecC2C(planInverse, (cufftComplex*)d_stateFFT, (cufftComplex*)d_state, CUFFT_INVERSE));
        }
        
        // Get device pointers for kernel access
        float2* getState() { return d_state; }
        float2* getStateFFT() { return d_stateFFT; }
        float2* getKernelFFT() { return d_kernelFFT; }
        float2* getTemp() { return d_temp; }
        float* getStateReal() { return d_stateReal; }
        int getWidth() { return width; }
        int getHeight() { return height; }
    };

    // Global FFT convolver instance
    inline FFTConvolver g_fftConvolver;

    // Load CUDA source files from folder
    inline M<str, str> LoadCUDAFromFolder(const fs::path& folderPath) {
        M<str, str> fileMap;
        if (!fs::exists(folderPath)) {
            fs::create_directory(folderPath);
            good("Created CUDA folder: {}", folderPath.string());
        }
        
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file()) {
                const std::string filename = entry.path().filename().string();
                if (filename.ends_with(".cu") || filename.ends_with(".cuh")) {
                    std::ifstream file(entry.path(), std::ios::in | std::ios::binary);
                    if (file) {
                        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
                        str filename_x = filename;
                        size_t pos = filename_x.rfind('.');
                        if (pos != std::string::npos) {
                            filename_x = filename_x.substr(0, pos);
                        }
                        fileMap[filename_x] = std::move(content);
                        good("Loaded CUDA file: {}", filename_x);
                    }
                }
            }
        }
        return fileMap;
    }

    // Global CUDA code storage
    inline M<str, str> named_cuda_code;

    inline void read_cuda_from_folder(const str& src_folder = "cuda_kernels") {
        fs::path cuda_folder_path = resolve_project_path(src_folder);
        named_cuda_code = LoadCUDAFromFolder(cuda_folder_path);
        
        for (auto& x : named_cuda_code) {
            good("Found CUDA kernel: {}", x.first);
        }
    }
}
