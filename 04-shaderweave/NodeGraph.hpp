#pragma once

#include <algorithm>
#include <memory>

#include <prelude.hpp>
#include <ImageLoader.hpp>
#include "ext+/imgui-node-editor/imgui_node_editor.h"
#include <cuda_related.hpp>
#include <optix_related.hpp>

using namespace SHORTHANDS;
using namespace FMT;

namespace LGRAPH {
    namespace ed = ax::NodeEditor;

    enum class NodeKind {
        Shader,
        Image,
        CUDA,
        OptiX,
        TextEditor
    };

    enum class SocketKind {
        Input,
        Output
    };

    struct Socket {
        ed::PinId id = 0;
        ed::NodeId node_id = 0;
        str name;
        SocketKind kind = SocketKind::Input;
        int channel_index = -1;
    };

    struct Link {
        ed::LinkId id = 0;
        ed::PinId start_pin_id = 0;
        ed::PinId end_pin_id = 0;
        ImColor color = ImColor(220, 220, 220, 220);
    };

    struct RenderTarget {
        GLuint fbo = 0;
        GLuint color_tex = 0;
        int width = 0;
        int height = 0;

        void cleanup() {
            if (fbo != 0) {
                glDeleteFramebuffers(1, &fbo);
                fbo = 0;
            }
            if (color_tex != 0) {
                glDeleteTextures(1, &color_tex);
                color_tex = 0;
            }
            width = 0;
            height = 0;
        }

        bool ensure(int w, int h) {
            if (w <= 0 || h <= 0) {
                return false;
            }

            if (fbo != 0 && color_tex != 0 && width == w && height == h) {
                return true;
            }

            cleanup();
            width = w;
            height = h;

            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);

            glGenTextures(1, &color_tex);
            glBindTexture(GL_TEXTURE_2D, color_tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindTexture(GL_TEXTURE_2D, 0);

            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex, 0);
            bool ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
            glBindFramebuffer(GL_FRAMEBUFFER, 0);

            if (!ok) {
                bad("Failed to create an offscreen framebuffer");
                cleanup();
            }

            return ok;
        }
    };

    struct ShaderPreviewRuntime {
        SHADER::Shader shader;
        SHADER::Canvas canvas;
        RenderTarget target;
        RenderTarget feedback_target;
        GLFWwindow* owner_context = nullptr;
        bool feedback_flip = false;
        bool feedback_seeded = false;
        int attempted_revision = -1;
        bool compiled = false;
        str last_error;

        void cleanup() {
            if (shader.program != 0) {
                glDeleteProgram(shader.program);
                shader.program = 0;
            }
            shader.stages.clear();
            shader._UNIFORMS.uniforms.clear();
            target.cleanup();
            feedback_target.cleanup();
            owner_context = nullptr;
            feedback_flip = false;
            feedback_seeded = false;
            attempted_revision = -1;
            compiled = false;
            last_error.clear();
        }
    };

    struct CUDAPreviewRuntime {
        CUDA_RT::CUDAKernel kernel;
        CUDA_RT::CUDAGLTexture texture;
        CUDA_RT::CUDAGlobals globals;
        GLFWwindow* owner_context = nullptr;
        int attempted_revision = -1;
        bool compiled = false;
        str last_error;

        void cleanup() {
            texture.cleanup();
            owner_context = nullptr;
            attempted_revision = -1;
            compiled = false;
            last_error.clear();
        }
    };

    struct OptixPreviewRuntime {
        OptixModuleManager module_manager;
        OptixPipelineBuilder pipeline_builder;
        CUDA_RT::CUDAGLTexture texture;
        uchar4* d_output_buffer = nullptr;
        unsigned int output_width = 0;
        unsigned int output_height = 0;
        float current_time = 0.0f;
        float delta_time = 0.0f;
        std::chrono::steady_clock::time_point last_frame_time = std::chrono::steady_clock::now();
        unsigned int frame_index = 0;
        GLFWwindow* owner_context = nullptr;
        bool initialized = false;
        int attempted_revision = -1;
        bool compiled = false;
        str last_error;

        void cleanup() {
            if (d_output_buffer != nullptr) {
                cudaFree(d_output_buffer);
                d_output_buffer = nullptr;
            }
            pipeline_builder.cleanup();
            module_manager.cleanup();
            texture.cleanup();
            output_width = 0;
            output_height = 0;
            current_time = 0.0f;
            delta_time = 0.0f;
            frame_index = 0;
            owner_context = nullptr;
            initialized = false;
            attempted_revision = -1;
            compiled = false;
            last_error.clear();
            last_frame_time = std::chrono::steady_clock::now();
        }
    };

    struct GraphNode {
        ed::NodeId id = 0;
        str label;
        NodeKind kind = NodeKind::Shader;
        ImColor color = ImColor(120, 160, 220);
        str raw_source;
        str processed_source;
        str resource_path;
        GLuint resource_texture = 0;
        int resource_width = 0;
        int resource_height = 0;
        V<Socket> inputs;
        V<Socket> outputs;
        bool enabled = true;
        bool use_double_buffer = false;
        bool use_pingpong = false;
        int revision = 0;
        str last_error;
        ShaderPreviewRuntime shader_preview;
        CUDAPreviewRuntime cuda_preview;
        OptixPreviewRuntime optix_preview;
    };

    struct ProjectOverride {
        bool active = false;
        ed::NodeId node_id = 0;
        str raw_source;
        int serial = 0;
    };

    inline ed::EditorContext* editor_context = nullptr;
    inline V<uptr<GraphNode>> nodes;
    inline V<Link> links;
    inline int next_id = 1;
    inline ed::NodeId selected_node_id = 0;
    inline ed::NodeId projected_node_id = 0;
    inline ProjectOverride projected_override;
    inline GLuint projected_cuda_texture = 0;
    inline GLuint projected_optix_texture = 0;
    inline GLuint checkerboard_texture = 0;
    inline ImVec2 next_spawn_pos = ImVec2(50.0f, 50.0f);
    inline int preview_width = 800;
    inline int preview_height = 800;
    inline RenderTarget editor_capture;

    inline int next_raw_id() {
        return next_id++;
    }

    inline ImVec2 consume_spawn_pos() {
        ImVec2 out = next_spawn_pos;
        next_spawn_pos.x += 45.0f;
        next_spawn_pos.y += 30.0f;
        if (next_spawn_pos.x > 400.0f) {
            next_spawn_pos.x = 50.0f;
        }
        return out;
    }

    inline bool is_shader_like(NodeKind kind) {
        return kind == NodeKind::Shader || kind == NodeKind::Image || kind == NodeKind::TextEditor;
    }

    inline bool uses_embedded_texture(NodeKind kind) {
        return kind == NodeKind::Image || kind == NodeKind::TextEditor;
    }

    inline int embedded_texture_channel(NodeKind kind) {
        switch (kind) {
        case NodeKind::Image:
            return 0;
        case NodeKind::TextEditor:
            return 0;
        default:
            return -1;
        }
    }

    inline bool reserves_embedded_socket(NodeKind kind) {
        return kind == NodeKind::Image;
    }

    inline str kind_name(NodeKind kind) {
        switch (kind) {
        case NodeKind::Shader:
            return "Shader";
        case NodeKind::Image:
            return "Image";
        case NodeKind::CUDA:
            return "CUDA";
        case NodeKind::OptiX:
            return "OptiX";
        case NodeKind::TextEditor:
            return "Text";
        }
        return "Unknown";
    }

    inline ImColor kind_color(NodeKind kind) {
        switch (kind) {
        case NodeKind::Shader:
            return ImColor(92, 160, 115, 255);
        case NodeKind::Image:
            return ImColor(214, 108, 108, 255);
        case NodeKind::CUDA:
            return ImColor(224, 143, 71, 255);
        case NodeKind::OptiX:
            return ImColor(71, 126, 224, 255);
        case NodeKind::TextEditor:
            return ImColor(183, 110, 214, 255);
        }
        return ImColor(140, 140, 140, 255);
    }

    inline str embedded_texture_shader_source() {
        return R"(
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    fragColor = texture(iChannel0, uv);
}
)";
    }

    inline str text_editor_shader_source() {
        return R"(
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    fragColor = texture(iChannel0, uv);
}
)";
    }

    inline GraphNode* find_node(ed::NodeId id) {
        for (auto& node : nodes) {
            if (node->id == id) {
                return node.get();
            }
        }
        return nullptr;
    }

    inline GraphNode* selected_node() {
        return find_node(selected_node_id);
    }

    inline GraphNode* projected_node() {
        return find_node(projected_node_id);
    }

    inline bool has_project_override(ed::NodeId node_id) {
        return projected_override.active && projected_override.node_id == node_id;
    }

    inline void clear_project_override() {
        projected_override.active = false;
        projected_override.node_id = 0;
        projected_override.raw_source.clear();
        projected_override.serial += 1;
    }

    inline str effective_project_source(const GraphNode& node) {
        if (has_project_override(node.id)) {
            return projected_override.raw_source;
        }
        return node.raw_source;
    }

    inline int effective_project_revision(const GraphNode& node) {
        if (has_project_override(node.id)) {
            return 1000000 + projected_override.serial;
        }
        return node.revision;
    }

    inline GraphNode* projected_shader_node() {
        auto* node = projected_node();
        if (node && is_shader_like(node->kind)) {
            return node;
        }
        return nullptr;
    }

    inline GraphNode* projected_cuda_node() {
        auto* node = projected_node();
        if (node && node->kind == NodeKind::CUDA) {
            return node;
        }
        return nullptr;
    }

    inline GraphNode* projected_optix_node() {
        auto* node = projected_node();
        if (node && node->kind == NodeKind::OptiX) {
            return node;
        }
        return nullptr;
    }

    inline Socket* find_socket(ed::PinId id) {
        for (auto& node : nodes) {
            for (auto& socket : node->inputs) {
                if (socket.id == id) {
                    return &socket;
                }
            }
            for (auto& socket : node->outputs) {
                if (socket.id == id) {
                    return &socket;
                }
            }
        }
        return nullptr;
    }

    inline Link* find_link(ed::LinkId id) {
        for (auto& link : links) {
            if (link.id == id) {
                return &link;
            }
        }
        return nullptr;
    }

    inline Link* find_link_for_input(ed::PinId input_id) {
        for (auto& link : links) {
            if (link.end_pin_id == input_id) {
                return &link;
            }
        }
        return nullptr;
    }

    inline str build_processed_source(NodeKind kind, const str& raw_source) {
        switch (kind) {
        case NodeKind::Shader:
        case NodeKind::Image:
        case NodeKind::TextEditor: {
            SHADER::Shader temp_shader;
            temp_shader.add_stage(GL_FRAGMENT_SHADER, raw_source, "");
            return get<2>(temp_shader.stages[GL_FRAGMENT_SHADER]);
        }
        case NodeKind::CUDA: {
            CUDA_RT::CUDAKernel temp_kernel;
            return temp_kernel.processCode(raw_source);
        }
        case NodeKind::OptiX:
            return OPTIX_RT::OPTIX_PROGRAM_HEADER + raw_source + OPTIX_RT::OPTIX_PROGRAM_FOOTER;
        }
        return raw_source;
    }

    struct SocketSpec {
        str name;
        SocketKind kind = SocketKind::Input;
        int channel_index = -1;
    };

    inline V<int> extract_texture_channels(NodeKind kind, const str& processed_source) {
        std::set<int> unique_channels;
        std::regex channel_regex(R"(\biChannel(\d+)\b)");
        std::smatch match;
        auto begin = processed_source.cbegin();
        while (std::regex_search(begin, processed_source.cend(), match, channel_regex)) {
            const int channel = std::stoi(match[1]);
            if (channel != embedded_texture_channel(kind) || !reserves_embedded_socket(kind)) {
                unique_channels.insert(channel);
            }
            begin = match.suffix().first;
        }

        return V<int>(unique_channels.begin(), unique_channels.end());
    }

    inline V<SocketSpec> extract_named_texture_sockets(NodeKind kind, const str& source) {
        V<SocketSpec> specs;
        std::set<str> used_names;
        std::set<int> used_units;

        auto reserve_input = [&](const str& name, int unit) {
            if (name.empty() || used_names.count(name)) {
                return;
            }

            SocketSpec spec;
            spec.name = name;
            spec.kind = SocketKind::Input;
            spec.channel_index = unit;
            specs.push_back(spec);
            used_names.insert(name);
            if (unit >= 0) {
                used_units.insert(unit);
            }
        };

        for (int channel : extract_texture_channels(kind, source)) {
            reserve_input(fstr("iChannel{}", channel), channel);
        }

        int next_unit = reserves_embedded_socket(kind) ? 1 : 0;
        auto next_free_unit = [&]() {
            while (used_units.count(next_unit) != 0) {
                next_unit += 1;
            }
            return next_unit++;
        };

        std::istringstream stream(source);
        str line;
        while (std::getline(stream, line)) {
            str lowered = line;
            std::transform(
                lowered.begin(),
                lowered.end(),
                lowered.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
            );

            if (lowered.find("uniform") == str::npos && lowered.find("layout") == str::npos) {
                continue;
            }

            const bool is_sampler2d = lowered.find("sampler2d") != str::npos;
            const bool is_sampler3d = lowered.find("sampler3d") != str::npos;
            const bool is_sampler_cube = lowered.find("samplercube") != str::npos;
            const bool is_image2d = lowered.find("image2d") != str::npos;
            const bool is_image3d = lowered.find("image3d") != str::npos;
            if (!(is_sampler2d || is_sampler3d || is_sampler_cube || is_image2d || is_image3d)) {
                continue;
            }

            std::istringstream tokens(line);
            str token;
            str name;
            while (tokens >> token) {
                name = token;
            }

            if (name.empty()) {
                continue;
            }

            while (!name.empty() && (name.back() == ';' || name.back() == ',')) {
                name.pop_back();
            }
            const size_t bracket = name.find('[');
            if (bracket != str::npos) {
                name = name.substr(0, bracket);
            }
            if (name.empty()) {
                continue;
            }

            if (name == "iChannel0" && reserves_embedded_socket(kind)) {
                continue;
            }

            if (lowered.find("writeonly") != str::npos) {
                if (!used_names.count(name)) {
                    SocketSpec spec;
                    spec.name = name;
                    spec.kind = SocketKind::Output;
                    specs.push_back(spec);
                    used_names.insert(name);
                }
            }
            else {
                reserve_input(name, next_free_unit());
            }
        }

        return specs;
    }

    inline bool socket_exists(const GraphNode& node, ed::PinId pin_id) {
        for (const auto& socket : node.inputs) {
            if (socket.id == pin_id) {
                return true;
            }
        }
        for (const auto& socket : node.outputs) {
            if (socket.id == pin_id) {
                return true;
            }
        }
        return false;
    }

    inline void prune_invalid_links() {
        links.erase(
            std::remove_if(
                links.begin(),
                links.end(),
                [](const Link& link) {
                    return find_socket(link.start_pin_id) == nullptr || find_socket(link.end_pin_id) == nullptr;
                }
            ),
            links.end()
        );
    }

    inline void rebuild_sockets(GraphNode& node) {
        M<str, ed::PinId> old_input_ids;
        for (const auto& socket : node.inputs) {
            old_input_ids[socket.name] = socket.id;
        }

        M<str, ed::PinId> old_output_ids;
        for (const auto& socket : node.outputs) {
            old_output_ids[socket.name] = socket.id;
        }

        node.inputs.clear();
        node.outputs.clear();

        if (is_shader_like(node.kind)) {
            const auto specs = extract_named_texture_sockets(node.kind, node.processed_source);
            for (const auto& spec : specs) {
                Socket socket;
                socket.id = spec.kind == SocketKind::Input
                    ? (old_input_ids.count(spec.name) ? old_input_ids[spec.name] : ed::PinId(next_raw_id()))
                    : (old_output_ids.count(spec.name) ? old_output_ids[spec.name] : ed::PinId(next_raw_id()));
                socket.node_id = node.id;
                socket.name = spec.name;
                socket.kind = spec.kind;
                socket.channel_index = spec.channel_index;

                if (socket.kind == SocketKind::Input) {
                    node.inputs.push_back(socket);
                }
                else {
                    node.outputs.push_back(socket);
                }
            }

            if (node.outputs.empty()) {
                Socket output;
                output.id = old_output_ids.count("Out") ? old_output_ids["Out"] : ed::PinId(next_raw_id());
                output.node_id = node.id;
                output.name = "Out";
                output.kind = SocketKind::Output;
                node.outputs.push_back(output);
            }
        }
        else {
            Socket output;
            output.id = old_output_ids.count("Project") ? old_output_ids["Project"] : ed::PinId(next_raw_id());
            output.node_id = node.id;
            output.name = "Project";
            output.kind = SocketKind::Output;
            node.outputs.push_back(output);
        }

        prune_invalid_links();
    }

    inline void refresh_node(GraphNode& node, const str* raw_override = nullptr) {
        if (raw_override != nullptr) {
            node.raw_source = *raw_override;
        }
        node.processed_source = build_processed_source(node.kind, node.raw_source);
        rebuild_sockets(node);
        node.revision += 1;
        node.last_error.clear();

        if (has_project_override(node.id) && projected_override.raw_source == node.raw_source) {
            clear_project_override();
        }
    }

    inline void warm_validate_backend_node(GraphNode& node);

    inline GraphNode& spawn_code_node(NodeKind kind, const str& label, const str& raw_source) {
        auto node = std::make_unique<GraphNode>();
        node->id = ed::NodeId(next_raw_id());
        node->label = label.empty() ? kind_name(kind) : label;
        node->kind = kind;
        node->color = kind_color(kind);
        if (kind == NodeKind::TextEditor && raw_source.empty()) {
            node->raw_source = text_editor_shader_source();
        }
        else {
            node->raw_source = raw_source;
        }
        refresh_node(*node);
        nodes.push_back(std::move(node));
        warm_validate_backend_node(*nodes.back());
        return *nodes.back();
    }

    inline void warm_validate_backend_node(GraphNode& node) {
        if (node.kind == NodeKind::CUDA) {
            auto& runtime = node.cuda_preview;
            runtime.compiled = runtime.kernel.compile(node.raw_source);
            runtime.attempted_revision = node.revision;
            if (runtime.compiled) {
                runtime.last_error.clear();
                node.last_error.clear();
            }
            else {
                runtime.last_error = runtime.kernel.compileLog.empty()
                    ? "CUDA node failed to compile on creation. See the terminal for details."
                    : runtime.kernel.compileLog;
                node.last_error = runtime.last_error;
            }
            return;
        }

        if (node.kind == NodeKind::OptiX) {
            if (!g_optixContext.init()) {
                node.last_error = "OptiX context initialization failed during node creation.";
                return;
            }

            auto& runtime = node.optix_preview;
            runtime.module_manager.init(g_optixContext.getContext());
            runtime.pipeline_builder.init(g_optixContext.getContext());
            runtime.initialized = true;
            str error_log;
            const str module_name = fstr("node_preview_{}", static_cast<int>(node.id.Get()));
            const str full_source = OPTIX_RT::OPTIX_PROGRAM_HEADER + node.raw_source + OPTIX_RT::OPTIX_PROGRAM_FOOTER;
            runtime.compiled = runtime.module_manager.compileAndCreate(module_name, full_source, error_log);
            if (runtime.compiled) {
                OptixModule module = runtime.module_manager.getModule(module_name);
                runtime.compiled = runtime.pipeline_builder.buildPipeline(
                    module,
                    "__raygen__main",
                    "__miss__main",
                    "__closesthit__main",
                    runtime.module_manager.getPipelineCompileOptions(),
                    error_log
                );
            }
            runtime.attempted_revision = node.revision;
            if (runtime.compiled) {
                runtime.last_error.clear();
                node.last_error.clear();
            }
            else {
                runtime.last_error = error_log.empty()
                    ? "OptiX node failed to compile on creation. See the terminal for details."
                    : error_log;
                node.last_error = runtime.last_error;
            }
        }
    }

    inline GraphNode* spawn_image_node_from_file(const fs::path& path) {
        const fs::path resolved_path = resolve_project_path(path);
        auto loaded = LIMG::load_texture_2d_from_file(resolved_path);
        if (!loaded) {
            bad("Failed to load image {}: {}", resolved_path.string(), loaded.error);
            return nullptr;
        }

        auto node = std::make_unique<GraphNode>();
        node->id = ed::NodeId(next_raw_id());
        node->label = resolved_path.stem().string();
        node->kind = NodeKind::Image;
        node->color = kind_color(node->kind);
        node->raw_source = embedded_texture_shader_source();
        node->resource_path = resolved_path.string();
        node->resource_texture = loaded.texture;
        node->resource_width = loaded.width;
        node->resource_height = loaded.height;
        refresh_node(*node);
        nodes.push_back(std::move(node));
        return nodes.back().get();
    }

    inline void cleanup_shader_preview(GraphNode& node) {
        node.shader_preview.cleanup();
        node.cuda_preview.cleanup();
        node.optix_preview.cleanup();
    }

    inline void cleanup_resource_texture(GraphNode& node) {
        LIMG::destroy_texture(node.resource_texture);
        node.resource_width = 0;
        node.resource_height = 0;
        node.resource_path.clear();
    }

    inline void disconnect_link(ed::LinkId link_id);

    inline void delete_node(ed::NodeId node_id) {
        auto it = std::find_if(nodes.begin(), nodes.end(), [&](const uptr<GraphNode>& node) {
            return node->id == node_id;
        });

        if (it == nodes.end()) {
            return;
        }

        V<ed::PinId> pin_ids;
        for (const auto& input : (*it)->inputs) {
            pin_ids.push_back(input.id);
        }
        for (const auto& output : (*it)->outputs) {
            pin_ids.push_back(output.id);
        }

        V<ed::LinkId> link_ids;
        for (const auto& pin_id : pin_ids) {
            for (const auto& link : links) {
                if (link.start_pin_id == pin_id || link.end_pin_id == pin_id) {
                    if (std::find_if(link_ids.begin(), link_ids.end(), [&](const ed::LinkId& existing) {
                        return existing == link.id;
                    }) == link_ids.end()) {
                        link_ids.push_back(link.id);
                    }
                }
            }
        }

        for (const auto& link_id : link_ids) {
            disconnect_link(link_id);
        }

        cleanup_shader_preview(*it->get());
        cleanup_resource_texture(*it->get());

        if (selected_node_id == node_id) {
            selected_node_id = 0;
        }
        if (projected_node_id == node_id) {
            projected_node_id = 0;
            projected_cuda_texture = 0;
            projected_optix_texture = 0;
        }
        if (has_project_override(node_id)) {
            clear_project_override();
        }

        nodes.erase(it);
        prune_invalid_links();
    }

    inline void clear_graph() {
        for (auto& node : nodes) {
            cleanup_shader_preview(*node);
            cleanup_resource_texture(*node);
        }
        nodes.clear();
        links.clear();
        selected_node_id = 0;
        projected_node_id = 0;
        clear_project_override();
        projected_cuda_texture = 0;
        projected_optix_texture = 0;
        next_spawn_pos = ImVec2(50.0f, 50.0f);
    }

    inline bool can_emit_texture(NodeKind kind);

    inline bool can_create_link(const Socket* a, const Socket* b, str* reason = nullptr) {
        if (a == nullptr || b == nullptr) {
            if (reason) {
                *reason = "One of the sockets does not exist.";
            }
            return false;
        }

        if (a->kind == b->kind) {
            if (reason) {
                *reason = "Links require one input and one output.";
            }
            return false;
        }

        const Socket* output = a->kind == SocketKind::Output ? a : b;
        const Socket* input = a->kind == SocketKind::Input ? a : b;
        auto* source_node = find_node(output->node_id);
        auto* dest_node = find_node(input->node_id);
        if (source_node == nullptr || dest_node == nullptr) {
            if (reason) {
                *reason = "Link endpoints are missing.";
            }
            return false;
        }

        if (!can_emit_texture(source_node->kind) || !is_shader_like(dest_node->kind)) {
            if (reason) {
                *reason = "This destination expects a texture-producing node output.";
            }
            return false;
        }

        return true;
    }

    inline void disconnect_link(ed::LinkId link_id) {
        links.erase(
            std::remove_if(
                links.begin(),
                links.end(),
                [&](const Link& link) {
                    return link.id == link_id;
                }
            ),
            links.end()
        );
    }

    inline void connect(ed::PinId a_id, ed::PinId b_id) {
        Socket* a = find_socket(a_id);
        Socket* b = find_socket(b_id);
        if (a == nullptr || b == nullptr) {
            return;
        }

        if (a->kind == SocketKind::Input) {
            std::swap(a, b);
        }

        auto* existing = find_link_for_input(b->id);
        if (existing != nullptr) {
            disconnect_link(existing->id);
        }

        links.push_back({ ed::LinkId(next_raw_id()), a->id, b->id });

        auto* source_node = find_node(a->node_id);
        auto* dest_node = find_node(b->node_id);
        if (source_node != nullptr && dest_node != nullptr && source_node == dest_node && is_shader_like(dest_node->kind)) {
            dest_node->use_double_buffer = true;
            dest_node->shader_preview.feedback_seeded = false;
        }
    }

    inline void select_node(ed::NodeId node_id) {
        selected_node_id = node_id;
    }

    inline bool project_selected_saved() {
        if (selected_node_id.Get() == 0) {
            return false;
        }

        projected_node_id = selected_node_id;
        if (has_project_override(selected_node_id)) {
            clear_project_override();
        }
        return true;
    }

    inline bool project_selected_temp(const str& raw_source) {
        if (selected_node_id.Get() == 0) {
            return false;
        }

        projected_node_id = selected_node_id;
        if (projected_override.active
            && projected_override.node_id == selected_node_id
            && projected_override.raw_source == raw_source)
        {
            return true;
        }
        projected_override.active = true;
        projected_override.node_id = selected_node_id;
        projected_override.raw_source = raw_source;
        projected_override.serial += 1;
        return true;
    }

    inline bool project_selected() {
        return project_selected_saved();
    }

    inline void ensure_default_graph() {
        if (!nodes.empty()) {
            return;
        }

        const str default_key = G::named_code.count("default_F") ? "default_F" : "volumex";
        if (!G::named_code.count(default_key)) {
            return;
        }

        auto& node = spawn_code_node(NodeKind::Shader, default_key, G::named_code[default_key]);
        if (editor_context != nullptr) {
            ed::SetCurrentEditor(editor_context);
            ed::SetNodePosition(node.id, consume_spawn_pos());
        }
        select_node(node.id);
        projected_node_id = node.id;
        clear_project_override();
    }

    inline void set_preview_size(int width, int height) {
        preview_width = std::max(width, 1);
        preview_height = std::max(height, 1);
    }

    inline void capture_editor_surface(int width, int height) {
        if (width <= 0 || height <= 0) {
            return;
        }

        if (!editor_capture.ensure(width, height)) {
            return;
        }

        glBindTexture(GL_TEXTURE_2D, editor_capture.color_tex);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    inline GLuint embedded_texture_id(GraphNode& node) {
        switch (node.kind) {
        case NodeKind::Image:
            return node.resource_texture;
        case NodeKind::TextEditor:
            return editor_capture.color_tex;
        default:
            return 0;
        }
    }

    inline GLuint ensure_checkerboard_texture() {
        if (checkerboard_texture != 0) {
            return checkerboard_texture;
        }

        constexpr int side = 64;
        std::array<unsigned char, side * side * 4> pixels {};
        for (int y = 0; y < side; ++y) {
            for (int x = 0; x < side; ++x) {
                const bool bright = ((x / 8) + (y / 8)) % 2 == 0;
                const unsigned char v = bright ? 210 : 70;
                const int idx = (y * side + x) * 4;
                pixels[idx + 0] = v;
                pixels[idx + 1] = v;
                pixels[idx + 2] = v;
                pixels[idx + 3] = 255;
            }
        }

        glGenTextures(1, &checkerboard_texture);
        glBindTexture(GL_TEXTURE_2D, checkerboard_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, side, side, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);
        return checkerboard_texture;
    }

    inline bool input_has_link(const GraphNode& node, const str& input_name) {
        for (const auto& input : node.inputs) {
            if (input.name == input_name && find_link_for_input(input.id) != nullptr) {
                return true;
            }
        }
        return false;
    }

    inline void ensure_preview_runtime_context(ShaderPreviewRuntime& runtime) {
        GLFWwindow* current = glfwGetCurrentContext();
        if (runtime.owner_context == current) {
            return;
        }

        runtime.canvas.reset();
        runtime.target.cleanup();
        runtime.feedback_target.cleanup();
        runtime.feedback_flip = false;
        runtime.owner_context = current;
    }

    inline bool ensure_shader_preview_compiled(GraphNode& node) {
        if (!is_shader_like(node.kind)) {
            return false;
        }

        auto& runtime = node.shader_preview;
        ensure_preview_runtime_context(runtime);
        if (runtime.attempted_revision == node.revision) {
            return runtime.compiled;
        }

        runtime.cleanup();
        runtime.shader.add_stage(GL_VERTEX_SHADER, G::named_code["vertex_shader"], "");
        runtime.shader.add_stage(GL_FRAGMENT_SHADER, node.raw_source, "");
        runtime.compiled = runtime.shader.compile_program();
        runtime.attempted_revision = node.revision;
        runtime.shader._UNIFORMS.uniforms.clear();

        if (runtime.compiled) {
            runtime.shader._UNIFORMS.add_all_from_program(runtime.shader.program);
            node.last_error.clear();
        }
        else {
            runtime.last_error = "Shader preview compilation failed. See the terminal for details.";
            node.last_error = runtime.last_error;
        }

        return runtime.compiled;
    }

    inline GLuint current_preview_texture(const GraphNode& node);
    inline GLuint render_dependency_texture(GraphNode& node, std::set<int>& visiting);

    inline void ensure_cuda_preview_context(CUDAPreviewRuntime& runtime) {
        GLFWwindow* current = glfwGetCurrentContext();
        if (runtime.owner_context != current) {
            runtime.texture.cleanup();
            runtime.owner_context = current;
        }

        if (runtime.texture.textureID == 0 || runtime.texture.width != preview_width || runtime.texture.height != preview_height) {
            runtime.texture.init(preview_width, preview_height);
        }
    }

    inline bool ensure_cuda_preview_compiled(GraphNode& node) {
        if (node.kind != NodeKind::CUDA) {
            return false;
        }

        if (!CUDA_RT::initCUDA()) {
            node.last_error = "CUDA initialization failed for node preview.";
            return false;
        }

        auto& runtime = node.cuda_preview;
        ensure_cuda_preview_context(runtime);
        if (runtime.attempted_revision == node.revision) {
            return runtime.compiled;
        }

        runtime.compiled = runtime.kernel.compile(node.raw_source);
        runtime.attempted_revision = node.revision;
        if (runtime.compiled) {
            runtime.last_error.clear();
            node.last_error.clear();
        }
        else {
            runtime.last_error = runtime.kernel.compileLog.empty()
                ? "CUDA preview compilation failed. See the terminal for details."
                : runtime.kernel.compileLog;
            node.last_error = runtime.last_error;
        }
        return runtime.compiled;
    }

    inline GLuint render_cuda_preview(GraphNode& node) {
        if (!ensure_cuda_preview_compiled(node)) {
            return 0;
        }

        auto& runtime = node.cuda_preview;
        ensure_cuda_preview_context(runtime);
        runtime.globals.iResolution[0] = static_cast<float>(preview_width);
        runtime.globals.iResolution[1] = static_cast<float>(preview_height);
        runtime.globals.iResolution[2] = 1.0f;
        runtime.globals.update();

        uchar4* d_pixels = runtime.texture.mapCUDA();
        runtime.kernel.launch(d_pixels, runtime.texture.width, runtime.texture.height, runtime.globals);
        runtime.texture.unmapCUDA();
        runtime.texture.updateTexture();
        glFlush();
        return runtime.texture.textureID;
    }

    inline void ensure_optix_preview_context(OptixPreviewRuntime& runtime) {
        GLFWwindow* current = glfwGetCurrentContext();
        if (runtime.owner_context != current) {
            runtime.texture.cleanup();
            runtime.owner_context = current;
        }

        if (runtime.texture.textureID == 0 || runtime.texture.width != preview_width || runtime.texture.height != preview_height) {
            runtime.texture.init(preview_width, preview_height);
        }

        if (runtime.output_width != static_cast<unsigned int>(preview_width)
            || runtime.output_height != static_cast<unsigned int>(preview_height))
        {
            if (runtime.d_output_buffer != nullptr) {
                cudaFree(runtime.d_output_buffer);
                runtime.d_output_buffer = nullptr;
            }
            runtime.output_width = static_cast<unsigned int>(preview_width);
            runtime.output_height = static_cast<unsigned int>(preview_height);
            cudaMalloc(&runtime.d_output_buffer, runtime.output_width * runtime.output_height * sizeof(uchar4));
            cudaMemset(runtime.d_output_buffer, 0, runtime.output_width * runtime.output_height * sizeof(uchar4));
            runtime.frame_index = 0;
        }
    }

    inline bool ensure_optix_preview_initialized(OptixPreviewRuntime& runtime) {
        if (runtime.initialized) {
            return true;
        }

        if (!g_optixContext.init()) {
            return false;
        }

        runtime.module_manager.init(g_optixContext.getContext());
        runtime.pipeline_builder.init(g_optixContext.getContext());
        runtime.initialized = true;
        runtime.last_frame_time = std::chrono::steady_clock::now();
        return true;
    }

    inline bool ensure_optix_preview_compiled(GraphNode& node) {
        if (node.kind != NodeKind::OptiX) {
            return false;
        }

        auto& runtime = node.optix_preview;
        if (!ensure_optix_preview_initialized(runtime)) {
            node.last_error = "OptiX initialization failed for node preview.";
            return false;
        }

        if (runtime.attempted_revision == node.revision) {
            return runtime.compiled;
        }

        str error_log;
        const str module_name = fstr("node_preview_{}", static_cast<int>(node.id.Get()));
        const str full_source = OPTIX_RT::OPTIX_PROGRAM_HEADER + node.raw_source + OPTIX_RT::OPTIX_PROGRAM_FOOTER;
        runtime.compiled = runtime.module_manager.compileAndCreate(module_name, full_source, error_log);
        if (runtime.compiled) {
            OptixModule module = runtime.module_manager.getModule(module_name);
            runtime.compiled = runtime.pipeline_builder.buildPipeline(
                module,
                "__raygen__main",
                "__miss__main",
                "__closesthit__main",
                runtime.module_manager.getPipelineCompileOptions(),
                error_log
            );
        }

        runtime.attempted_revision = node.revision;
        if (runtime.compiled) {
            runtime.last_error.clear();
            node.last_error.clear();
        }
        else {
            runtime.last_error = error_log.empty()
                ? "OptiX preview compilation failed. See the terminal for details."
                : error_log;
            node.last_error = runtime.last_error;
        }
        return runtime.compiled;
    }

    inline GLuint render_optix_preview(GraphNode& node) {
        auto& runtime = node.optix_preview;
        if (!ensure_optix_preview_compiled(node)) {
            return 0;
        }

        ensure_optix_preview_context(runtime);
        auto now = std::chrono::steady_clock::now();
        runtime.delta_time = std::chrono::duration<float>(now - runtime.last_frame_time).count();
        runtime.last_frame_time = now;
        runtime.current_time += runtime.delta_time;

        OptixLaunchParams params = {};
        params.outputBuffer = runtime.d_output_buffer;
        params.width = runtime.output_width;
        params.height = runtime.output_height;
        params.time = runtime.current_time;
        params.deltaTime = runtime.delta_time;
        params.mouseX = 0.0f;
        params.mouseY = 0.0f;
        params.mouseButtons = 0;
        params.camPos = make_float3(0.0f, 0.0f, -3.0f);
        params.camDir = make_float3(0.0f, 0.0f, 1.0f);
        params.camUp = make_float3(0.0f, 1.0f, 0.0f);
        params.camRight = make_float3(1.0f, 0.0f, 0.0f);
        params.fov = 45.0f;
        params.frameIndex = runtime.frame_index;
        params.traversable = 0;

        if (!runtime.pipeline_builder.launch(params)) {
            node.last_error = "OptiX preview launch failed.";
            return 0;
        }

        cudaDeviceSynchronize();
        uchar4* mapped = runtime.texture.mapCUDA();
        if (mapped != nullptr) {
            cudaMemcpy(mapped, runtime.d_output_buffer, runtime.output_width * runtime.output_height * sizeof(uchar4), cudaMemcpyDeviceToDevice);
            runtime.texture.unmapCUDA();
        }
        runtime.texture.updateTexture();
        glFlush();
        runtime.frame_index += 1;
        return runtime.texture.textureID;
    }

    inline bool can_emit_texture(NodeKind kind) {
        return kind == NodeKind::Shader
            || kind == NodeKind::Image
            || kind == NodeKind::TextEditor
            || kind == NodeKind::CUDA
            || kind == NodeKind::OptiX;
    }

    inline bool node_has_self_link(const GraphNode& node);

    inline GLuint current_node_output_texture(GraphNode& node) {
        switch (node.kind) {
        case NodeKind::Shader:
        case NodeKind::Image:
        case NodeKind::TextEditor:
            return current_preview_texture(node);
        case NodeKind::CUDA:
            return node.cuda_preview.texture.textureID;
        case NodeKind::OptiX:
            return node.optix_preview.texture.textureID;
        }
        return 0;
    }

    inline bool node_uses_feedback(const GraphNode& node) {
        if (!is_shader_like(node.kind)) {
            return false;
        }
        return node.use_double_buffer || node_has_self_link(node);
    }

    inline GLuint render_node_output_texture(GraphNode& node, std::set<int>& visiting, bool render_dependencies) {
        switch (node.kind) {
        case NodeKind::Shader:
        case NodeKind::Image:
        case NodeKind::TextEditor:
            if (render_dependencies) {
                return render_dependency_texture(node, visiting);
            }
            return current_preview_texture(node);
        case NodeKind::CUDA:
            return render_dependencies ? render_cuda_preview(node) : node.cuda_preview.texture.textureID;
        case NodeKind::OptiX:
            return render_dependencies ? render_optix_preview(node) : node.optix_preview.texture.textureID;
        }
        return 0;
    }

    inline bool node_has_self_link(const GraphNode& node) {
        for (const auto& input : node.inputs) {
            auto* link = find_link_for_input(input.id);
            if (link == nullptr) {
                continue;
            }

            auto* output_socket = find_socket(link->start_pin_id);
            if (output_socket != nullptr && output_socket->node_id == node.id) {
                return true;
            }
        }

        return false;
    }

    inline GLuint current_preview_texture(const GraphNode& node) {
        if (node_uses_feedback(node)) {
            return node.shader_preview.feedback_flip
                ? node.shader_preview.feedback_target.color_tex
                : node.shader_preview.target.color_tex;
        }
        return node.shader_preview.target.color_tex;
    }

    inline void bind_shader_inputs(GraphNode& node, SHADER::Shader& shader, std::set<int>& visiting, bool render_dependencies = true) {
        for (int unit = 0; unit < 8; ++unit) {
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        const GLuint reserved_texture = embedded_texture_id(node);
        if (reserved_texture != 0 && uses_embedded_texture(node.kind)) {
            const bool fallback_input = node.kind == NodeKind::TextEditor
                ? !input_has_link(node, "iChannel0")
                : true;
            if (fallback_input) {
                shader._UNIFORMS.set_uniform(
                    "iChannel0",
                    std::pair<GLuint, GLuint> { 0u, reserved_texture }
                );
            }
        }

        for (const auto& input : node.inputs) {
            auto* link = find_link_for_input(input.id);
            GLuint texture_id = 0;

            if (link != nullptr) {
                auto* output_socket = find_socket(link->start_pin_id);
                auto* source_node = output_socket != nullptr ? find_node(output_socket->node_id) : nullptr;
                if (source_node != nullptr && can_emit_texture(source_node->kind)) {
                    if (source_node == &node && node_uses_feedback(node) && !node.shader_preview.feedback_seeded) {
                        texture_id = 0;
                    }
                    else {
                        texture_id = source_node == &node
                            ? current_node_output_texture(*source_node)
                            : render_node_output_texture(*source_node, visiting, render_dependencies);
                    }
                }
            }

            if (texture_id == 0) {
                if (node.kind == NodeKind::TextEditor && input.name == "iChannel0" && reserved_texture != 0 && !input_has_link(node, input.name)) {
                    continue;
                }
                texture_id = ensure_checkerboard_texture();
            }

            if (texture_id == 0) {
                continue;
            }

            shader._UNIFORMS.set_uniform(
                input.name,
                std::pair<GLuint, GLuint> {
                    static_cast<GLuint>(std::max(0, input.channel_index)),
                    texture_id
                }
            );
        }
    }

    inline GLuint render_dependency_texture(GraphNode& node, std::set<int>& visiting) {
        if (!is_shader_like(node.kind)) {
            return 0;
        }

        const int node_key = node.id.Get();
        if (visiting.count(node_key) != 0) {
            node.last_error = "Cycle detected while resolving shader node dependencies.";
            return 0;
        }

        if (!ensure_shader_preview_compiled(node)) {
            return 0;
        }

        auto& runtime = node.shader_preview;
        ensure_preview_runtime_context(runtime);
        const bool uses_feedback = node_uses_feedback(node);
        if (!runtime.target.ensure(preview_width, preview_height)) {
            return 0;
        }
        if (uses_feedback && !runtime.feedback_target.ensure(preview_width, preview_height)) {
            return 0;
        }

        visiting.insert(node_key);

        RenderTarget* write_target = &runtime.target;
        if (uses_feedback) {
            write_target = runtime.feedback_flip ? &runtime.target : &runtime.feedback_target;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, write_target->fbo);
        glViewport(0, 0, preview_width, preview_height);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (node.enabled) {
            runtime.shader.use_context_glfw();
            bind_shader_inputs(node, runtime.shader, visiting);
            SHADER::Shader::globals.iResolution = glm::vec3(
                static_cast<float>(preview_width),
                static_cast<float>(preview_height),
                1.0f
            );
            runtime.shader.update_shared_uniforms();
            runtime.canvas.draw();
            glFlush();
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        visiting.erase(node_key);

        if (uses_feedback) {
            runtime.feedback_flip = !runtime.feedback_flip;
            runtime.feedback_seeded = true;
        }

        return current_preview_texture(node);
    }
}
