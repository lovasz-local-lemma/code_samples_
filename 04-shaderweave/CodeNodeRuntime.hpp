#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <memory>
#include <string>
#include <vector>

namespace LegacyNodeRuntime {

void load_cuda_programs(const std::string& folder = "cuda_kernels");
void load_optix_programs(const std::string& folder = "optix_programs");

std::vector<std::string> cuda_program_names();
std::vector<std::string> optix_program_names();

std::string cuda_program_source(const std::string& name);
std::string optix_program_source(const std::string& name);

class CUDA_Node_Runtime {
public:
    CUDA_Node_Runtime();
    ~CUDA_Node_Runtime();

    CUDA_Node_Runtime(CUDA_Node_Runtime&&) noexcept;
    CUDA_Node_Runtime& operator=(CUDA_Node_Runtime&&) noexcept;

    CUDA_Node_Runtime(const CUDA_Node_Runtime&) = delete;
    CUDA_Node_Runtime& operator=(const CUDA_Node_Runtime&) = delete;

    bool compile_saved(const std::string& source, int width, int height);
    bool compile_project(const std::string& source, int width, int height);
    void clear_project();
    bool has_project_override() const;
    bool render(int width, int height);
    GLuint texture() const;
    void update_mouse(double xpos, double ypos, bool pressed, int height);
    void update_mouse_click(double xpos, double ypos, bool pressed, int height);
    const std::string& last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class OptiX_Node_Runtime {
public:
    OptiX_Node_Runtime();
    ~OptiX_Node_Runtime();

    OptiX_Node_Runtime(OptiX_Node_Runtime&&) noexcept;
    OptiX_Node_Runtime& operator=(OptiX_Node_Runtime&&) noexcept;

    OptiX_Node_Runtime(const OptiX_Node_Runtime&) = delete;
    OptiX_Node_Runtime& operator=(const OptiX_Node_Runtime&) = delete;

    bool compile_saved(const std::string& label, const std::string& source, int width, int height);
    bool compile_project(const std::string& label, const std::string& source, int width, int height);
    void clear_project();
    bool has_project_override() const;
    bool render(int width, int height);
    GLuint texture() const;
    bool set_input_texture(GLuint texture, int width, int height);
    void update_mouse(double xpos, double ypos, int width, int height);
    void update_mouse_button(int button, int action);
    const std::string& last_error() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace LegacyNodeRuntime
