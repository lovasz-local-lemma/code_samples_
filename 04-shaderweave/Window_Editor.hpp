#pragma once

#include <algorithm>
#include <cctype>

#include <Window_ImGui.hpp>
#include <NodeGraph.hpp>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "ext+/ImGuiColorTextEdit/TextEditor.h"
#include <FancyKnob.hpp>

using namespace SHORTHANDS;
using namespace FMT;

class Editor_Window : public ImGui_Window
{
public:
    TextEditor editor;
    ax::NodeEditor::NodeId loaded_node_id = 0;
    int loaded_revision = -1;
    str source_raw;
    str source_processed;
    bool show_processed_tab = false;
    bool live_edit = false;
    bool show_search_bar = false;
    bool focus_search_bar = false;
    bool case_sensitive = false;
    char search_buf[128] = "";

    struct Bookmark {
        ax::NodeEditor::NodeId nodeId = {};
        TextEditor::Coordinates coords = {};
        bool valid = false;
    };
    Bookmark bookmarks[10];
    ax::NodeEditor::NodeId pending_jump_node_id = {};
    TextEditor::Coordinates pending_jump_coords = {};
    bool has_pending_jump = false;

    static TextEditor::LanguageDefinition GLSLDefinition()
    {
        auto lang = TextEditor::LanguageDefinition::GLSL();
        lang.mAutoIndentation = true;
        return lang;
    }

    static TextEditor::LanguageDefinition CppLikeDefinition()
    {
        auto lang = TextEditor::LanguageDefinition::CPlusPlus();
        lang.mAutoIndentation = true;
        return lang;
    }

    static TextEditor::LanguageDefinition shader_language_for(LGRAPH::NodeKind kind)
    {
        switch (kind) {
        case LGRAPH::NodeKind::Shader:
        case LGRAPH::NodeKind::Image:
        case LGRAPH::NodeKind::TextEditor:
            return GLSLDefinition();
        case LGRAPH::NodeKind::CUDA:
        case LGRAPH::NodeKind::OptiX:
            return CppLikeDefinition();
        }
        return CppLikeDefinition();
    }

    static bool supports_live_edit(LGRAPH::NodeKind kind)
    {
        return kind != LGRAPH::NodeKind::OptiX;
    }

    void sync_from_selection()
    {
        auto* node = LGRAPH::selected_node();
        if (node == nullptr) {
            loaded_node_id = 0;
            loaded_revision = -1;
            source_raw.clear();
            source_processed.clear();
            title = "LumenFlux Editor";
            editor.SetText("");
            return;
        }

        if (loaded_node_id == node->id
            && loaded_revision == node->revision
            && source_raw == node->raw_source
            && source_processed == node->processed_source)
        {
            if (has_pending_jump && loaded_node_id == pending_jump_node_id) {
                editor.SetCursorPosition(pending_jump_coords);
                has_pending_jump = false;
            }
            return;
        }

        loaded_node_id = node->id;
        loaded_revision = node->revision;
        source_raw = node->raw_source;
        source_processed = node->processed_source;
        title = fstr("{} Editor", node->label);
        if (!supports_live_edit(node->kind)) {
            live_edit = false;
        }

        editor.SetLanguageDefinition(shader_language_for(node->kind));
        editor.SetReadOnly(show_processed_tab);
        editor.SetText(show_processed_tab ? source_processed : source_raw);

        if (has_pending_jump && loaded_node_id == pending_jump_node_id) {
            editor.SetCursorPosition(pending_jump_coords);
            has_pending_jump = false;
        }
    }

    void refresh_processed_preview()
    {
        auto* node = LGRAPH::selected_node();
        if (node == nullptr) {
            return;
        }

        source_processed = LGRAPH::build_processed_source(node->kind, source_raw);
        if (show_processed_tab) {
            editor.SetText(source_processed);
        }
    }

    bool commit_to_selected()
    {
        auto* node = LGRAPH::selected_node();
        if (node == nullptr) {
            return false;
        }

        if (!show_processed_tab) {
            source_raw = editor.GetText();
        }

        LGRAPH::refresh_node(*node, &source_raw);
        source_processed = node->processed_source;
        loaded_node_id = node->id;
        loaded_revision = node->revision;

        editor.SetLanguageDefinition(shader_language_for(node->kind));
        editor.SetReadOnly(show_processed_tab);
        editor.SetText(show_processed_tab ? source_processed : source_raw);
        return true;
    }

    bool project_current_to_viewer()
    {
        auto* node = LGRAPH::selected_node();
        if (node == nullptr) {
            return false;
        }

        if (!show_processed_tab) {
            source_raw = editor.GetText();
        }

        refresh_processed_preview();
        return LGRAPH::project_selected_temp(source_raw);
    }

    void switch_tab(bool processed)
    {
        if (show_processed_tab == processed) {
            return;
        }

        if (!show_processed_tab) {
            source_raw = editor.GetText();
            refresh_processed_preview();
        }

        show_processed_tab = processed;
        editor.SetReadOnly(show_processed_tab);
        editor.SetText(show_processed_tab ? source_processed : source_raw);
    }

    void toggle_live_edit(bool enabled)
    {
        auto* node = LGRAPH::selected_node();
        if (node == nullptr) {
            live_edit = enabled;
            return;
        }

        if (!supports_live_edit(node->kind)) {
            live_edit = false;
            return;
        }

        live_edit = enabled;
        if (live_edit) {
            project_current_to_viewer();
        }
        else if (LGRAPH::has_project_override(node->id)) {
            LGRAPH::project_selected_saved();
        }
    }

    void draw_mode_tabs()
    {
        int next_tab = show_processed_tab ? 1 : 0;
        if (ImGui::BeginTabBar("ShaderParts")) {
            if (ImGui::BeginTabItem("RawCode")) {
                next_tab = 0;
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("+Injected")) {
                next_tab = 1;
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }

        switch_tab(next_tab == 1);
    }

    void draw_toolbar()
    {
        auto* node = LGRAPH::selected_node();
        const bool has_selection = node != nullptr;

        if (!has_selection) {
            ImGui::TextDisabled("Select a node to edit its source.");
            return;
        }

        constexpr float kRowH = 30.0f;
        const float textOffset = (kRowH - ImGui::GetTextLineHeight()) * 0.5f;

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.24f, 0.16f, 0.58f, 0.92f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.33f, 0.24f, 0.76f, 1.0f));
        if (ImGui::Button("Save", ImVec2(72.0f, kRowH))) {
            commit_to_selected();
        }
        ImGui::PopStyleColor(2);
        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.54f, 0.28f, 0.96f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.66f, 0.34f, 1.0f));
        if (ImGui::Button("Project", ImVec2(96.0f, kRowH))) {
            project_current_to_viewer();
        }
        ImGui::PopStyleColor(2);

        ImGui::SameLine();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + textOffset);
        ImGui::TextDisabled("Live");
        ImGui::SameLine();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - textOffset);

        bool want_live = live_edit;
        const bool can_live_edit = supports_live_edit(node->kind);
        if (!can_live_edit) {
            ImGui::BeginDisabled();
        }
        if (MyWidgets::FancyKnob("##LiveEdit", &want_live, ImVec2(54.0f, kRowH))) {
            toggle_live_edit(want_live);
        }
        if (!can_live_edit) {
            ImGui::EndDisabled();
        }

        ImGui::SameLine();
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + textOffset);
        ImGui::TextDisabled("%s", LGRAPH::kind_name(node->kind).c_str());
        ImGui::SameLine();
        ImGui::TextDisabled("Ctrl+F");
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - textOffset);
    }

    void handle_bookmarks()
    {
        if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows))
            return;

        auto& io = ImGui::GetIO();
        for (int i = 0; i <= 9; ++i)
        {
            const ImGuiKey numKey = (ImGuiKey)(ImGuiKey_0 + i);
            if (io.KeyCtrl && io.KeyShift && !io.KeyAlt && ImGui::IsKeyPressed(numKey, false))
            {
                bookmarks[i] = { loaded_node_id, editor.GetCursorPosition(), true };
            }
            else if (io.KeyCtrl && !io.KeyShift && !io.KeyAlt && ImGui::IsKeyPressed(numKey, false))
            {
                if (bookmarks[i].valid)
                {
                    if (bookmarks[i].nodeId == loaded_node_id)
                    {
                        editor.SetCursorPosition(bookmarks[i].coords);
                    }
                    else if (bookmarks[i].nodeId.Get() != 0)
                    {
                        LGRAPH::select_node(bookmarks[i].nodeId);
                        pending_jump_node_id = bookmarks[i].nodeId;
                        pending_jump_coords  = bookmarks[i].coords;
                        has_pending_jump     = true;
                    }
                }
            }
        }
    }

    void draw_search_bar_line_highlight()
    {
        auto& io = ImGui::GetIO();
        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)
            && io.KeyCtrl
            && ImGui::IsKeyPressed(ImGuiKey_F))
        {
            show_search_bar = true;
            focus_search_bar = true;
        }

        if (show_search_bar && ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            show_search_bar = false;
            search_buf[0] = '\0';
            editor.SetErrorMarkers({});
            editor.ClearSearchHighlights();
        }

        if (show_search_bar) {
            if (focus_search_bar) {
                ImGui::SetKeyboardFocusHere();
                focus_search_bar = false;
            }

            ImGui::SetNextItemWidth(220.0f);
            ImGui::InputTextWithHint("##Find", "Find...", search_buf, sizeof(search_buf));

            ImGui::SameLine();
            if (ImGui::Button(case_sensitive ? "A!a" : "A~a")) {
                case_sensitive = !case_sensitive;
            }

            ImGui::SameLine();
            if (ImGui::Button("X")) {
                show_search_bar = false;
                search_buf[0] = '\0';
                editor.SetErrorMarkers({});
                editor.ClearSearchHighlights();
            }
        }

        if (!show_search_bar || search_buf[0] == '\0') {
            editor.SetErrorMarkers({});
            editor.ClearSearchHighlights();
            return;
        }

        const str buffer = editor.GetText();
        V<str> lines;
        size_t start = 0;
        while (start <= buffer.size()) {
            const size_t end = buffer.find('\n', start);
            if (end == str::npos) {
                lines.push_back(buffer.substr(start));
                break;
            }

            lines.push_back(buffer.substr(start, end - start));
            start = end + 1;
        }

        str needle = search_buf;
        if (!case_sensitive) {
            std::transform(
                needle.begin(),
                needle.end(),
                needle.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
            );
        }

        TextEditor::ErrorMarkers markers;
        std::vector<TextEditor::Highlight> highlights;
        for (int line_index = 0; line_index < static_cast<int>(lines.size()); ++line_index) {
            str haystack = lines[line_index];
            if (!case_sensitive) {
                std::transform(
                    haystack.begin(),
                    haystack.end(),
                    haystack.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
                );
            }

            size_t pos = 0;
            bool found_on_line = false;
            while ((pos = haystack.find(needle, pos)) != str::npos) {
                highlights.push_back({ line_index, static_cast<int>(pos), static_cast<int>(pos + needle.size()) });
                found_on_line = true;
                pos += std::max<size_t>(needle.size(), 1);
            }

            if (found_on_line) {
                markers[line_index + 1] = "match";
            }
        }

        editor.SetErrorMarkers(markers);
        editor.SetSearchHighlights(highlights);
    }

    void render_editor()
    {
        use_context_glfw();
        use_context_imgui();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        sync_from_selection();

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(my_handle(), &width, &height);

        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(static_cast<float>(width), static_cast<float>(height)));
        ImGui::Begin(
            title.c_str(),
            nullptr,
            ImGuiWindowFlags_NoResize
                | ImGuiWindowFlags_NoMove
                | ImGuiWindowFlags_NoCollapse
                | ImGuiWindowFlags_NoTitleBar
        );

        draw_mode_tabs();
        draw_toolbar();
        handle_bookmarks();
        ImGui::Separator();

        if (loaded_node_id.Get() == 0) {
            ImGui::TextDisabled("Click a node in the graph and its code will appear here.");
        }
        else {
            auto* node = LGRAPH::selected_node();
            if (node != nullptr && !show_processed_tab && editor.IsTextChanged()) {
                source_raw = editor.GetText();
                refresh_processed_preview();
                if (live_edit && supports_live_edit(node->kind)) {
                    LGRAPH::project_selected_temp(source_raw);
                }
            }

            draw_search_bar_line_highlight();

            if (node != nullptr && !show_processed_tab && node->raw_source.empty() && node->kind == LGRAPH::NodeKind::CUDA) {
                ImGui::TextDisabled("This CUDA source file is empty. The runtime fallback only appears in +Injected.");
                ImGui::Separator();
            }

            const ImVec2 editor_min = ImGui::GetCursorScreenPos();
            const ImVec2 editor_size = ImGui::GetContentRegionAvail();
            ImDrawList* dl = ImGui::GetWindowDrawList();
            dl->AddRectFilled(
                ImVec2(editor_min.x - 8.0f, editor_min.y - 8.0f),
                ImVec2(editor_min.x + editor_size.x + 8.0f, editor_min.y + editor_size.y + 8.0f),
                IM_COL32(64, 116, 214, 36),
                8.0f
            );
            dl->AddRectFilled(
                ImVec2(editor_min.x - 4.0f, editor_min.y - 4.0f),
                ImVec2(editor_min.x + editor_size.x + 4.0f, editor_min.y + editor_size.y + 4.0f),
                IM_COL32(88, 156, 255, 54),
                6.0f
            );

            const bool pushed_font = G::font_mgr.push_approx("bold", 20);
            editor.Render("LumenFluxTextEditor");
            if (pushed_font) {
                G::font_mgr.pop();
            }
        }

        ImGui::End();

        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        glViewport(0, 0, width, height);
        glClearColor(0.08f, 0.09f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(draw_data);
        LGRAPH::capture_editor_surface(width, height);
    }

    Editor_Window(int w, int h, const str& title, GLFWmonitor* monitor = nullptr, GLFWwindow* share = nullptr)
        : ImGui_Window(w, h, title, monitor, share)
    {
        this->title = title;
        editor.SetLanguageDefinition(GLSLDefinition());
        editor.SetPalette(TextEditor::GetDarkPalette());
        editor.SetShowWhitespaces(false);
        editor.SetTabSize(4);
        editor.SetColorizerEnable(true);
        set_render_callback([this]() { render_editor(); });
    }

    str title = "LumenFlux Editor";
};
