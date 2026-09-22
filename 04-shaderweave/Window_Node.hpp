#pragma once

#include <algorithm>
#include <cmath>

#include <Window_ImGui.hpp>
#include <NodeGraph.hpp>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

using namespace SHORTHANDS;
using namespace FMT;

namespace ed = ax::NodeEditor;

class Node_Window : public ImGui_Window
{
public:
    enum class VisualStyle {
        Modern = 0,
        Legacy = 1,
        Prism = 2,
        Minimal = 3
    };

    inline static VisualStyle visual_style = VisualStyle::Modern;

    struct PendingDrop {
        fs::path path;
        ImVec2 mouse_pos = ImVec2(0.0f, 0.0f);
    };

    V<PendingDrop> pending_drops;
    V<ed::LinkId> pending_deleted_links;
    V<ed::NodeId> pending_deleted_nodes;

    static bool contains_link_id(const V<ed::LinkId>& ids, ed::LinkId id)
    {
        return std::find_if(ids.begin(), ids.end(), [&](const ed::LinkId& existing) {
            return existing == id;
        }) != ids.end();
    }

    static bool contains_node_id(const V<ed::NodeId>& ids, ed::NodeId id)
    {
        return std::find_if(ids.begin(), ids.end(), [&](const ed::NodeId& existing) {
            return existing == id;
        }) != ids.end();
    }

    void queue_link_delete(ed::LinkId link_id)
    {
        if (!contains_link_id(pending_deleted_links, link_id)) {
            pending_deleted_links.push_back(link_id);
        }
    }

    void queue_node_delete(ed::NodeId node_id)
    {
        if (!contains_node_id(pending_deleted_nodes, node_id)) {
            pending_deleted_nodes.push_back(node_id);
        }
    }

    void process_pending_graph_mutations()
    {
        if (pending_deleted_links.empty() && pending_deleted_nodes.empty()) {
            return;
        }

        if (!pending_deleted_nodes.empty()) {
            ed::ClearSelection();
        }

        for (const auto& link_id : pending_deleted_links) {
            LGRAPH::disconnect_link(link_id);
        }
        pending_deleted_links.clear();

        for (const auto& node_id : pending_deleted_nodes) {
            LGRAPH::delete_node(node_id);
        }
        pending_deleted_nodes.clear();
    }

    void init_node_editor()
    {
        if (LGRAPH::editor_context == nullptr) {
            ed::Config config;
            config.SettingsFile = "NodeEditor.json";
            LGRAPH::editor_context = ed::CreateEditor(&config);
            ed::SetCurrentEditor(LGRAPH::editor_context);

            ed::Style& style = ed::GetStyle();
            style.NodePadding = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
            style.NodeRounding = 16.0f;
            style.NodeBorderWidth = 0.0f;
            style.HoveredNodeBorderWidth = 0.0f;
            style.SelectedNodeBorderWidth = 0.0f;
            style.LinkStrength = 110.0f;
            style.FlowMarkerDistance = 36.0f;
            style.FlowSpeed = 130.0f;
            style.FlowDuration = 1.15f;
            style.PinRadius = 0.0f;
            style.PinArrowSize = 0.0f;
            style.PinArrowWidth = 0.0f;
            style.Colors[ed::StyleColor_NodeBg] = ImColor(0, 0, 0, 0);
            style.Colors[ed::StyleColor_NodeBorder] = ImColor(0, 0, 0, 0);
            style.Colors[ed::StyleColor_HovNodeBorder] = ImColor(0, 0, 0, 0);
            style.Colors[ed::StyleColor_SelNodeBorder] = ImColor(0, 0, 0, 0);
            style.Colors[ed::StyleColor_Bg] = ImColor(18, 22, 29, 255);
            style.Colors[ed::StyleColor_Grid] = ImColor(68, 78, 90, 80);
            style.Colors[ed::StyleColor_PinRect] = ImColor(0, 0, 0, 0);
            style.Colors[ed::StyleColor_PinRectBorder] = ImColor(0, 0, 0, 0);
            style.Colors[ed::StyleColor_Flow] = ImColor(255, 166, 84, 255);
            style.Colors[ed::StyleColor_FlowMarker] = ImColor(255, 227, 154, 255);
            style.Colors[ed::StyleColor_HovLinkBorder] = ImColor(186, 211, 255, 255);
            style.Colors[ed::StyleColor_SelLinkBorder] = ImColor(255, 221, 118, 255);
        }
    }

    static float node_width_for(const LGRAPH::GraphNode& node)
    {
        return 232.0f;
    }

    static float node_height_for(const LGRAPH::GraphNode& node)
    {
        return node_width_for(node);
    }

    static const char* visual_style_name(VisualStyle style)
    {
        switch (style) {
        case VisualStyle::Modern:
            return "Modern";
        case VisualStyle::Legacy:
            return "Legacy Glow";
        case VisualStyle::Prism:
            return "Prism";
        case VisualStyle::Minimal:
            return "Minimal";
        }
        return "Modern";
    }

    static ImVec2 pin_center_left(const ImRect& node_rect, int index, int count)
    {
        const float top = node_rect.Min.y + 52.0f;
        const float span = std::max(42.0f, node_rect.GetHeight() - 92.0f);
        const float step = span / static_cast<float>(std::max(count, 1));
        return ImVec2(node_rect.Min.x + 6.0f, top + step * (index + 0.5f));
    }

    static ImVec2 pin_center_right(const ImRect& node_rect, int index, int count)
    {
        const float top = node_rect.Min.y + 52.0f;
        const float span = std::max(42.0f, node_rect.GetHeight() - 92.0f);
        const float step = span / static_cast<float>(std::max(count, 1));
        return ImVec2(node_rect.Max.x - 6.0f, top + step * (index + 0.5f));
    }

    static ImU32 kind_accent(const LGRAPH::GraphNode& node)
    {
        return node.color;
    }

    static ImVec4 link_color_for(const LGRAPH::Link& link)
    {
        switch (visual_style) {
        case VisualStyle::Legacy:
            return ImColor(255, 186, 120, 255);
        case VisualStyle::Prism: {
            const float hue = std::fmod(static_cast<float>(ImGui::GetTime()) * 0.12f + link.id.Get() * 0.031f, 1.0f);
            return ImColor::HSV(hue, 0.72f, 1.0f, 0.95f);
        }
        case VisualStyle::Minimal:
            return ImColor(166, 176, 192, 210);
        case VisualStyle::Modern:
        default:
            return ImColor(194, 214, 255, 235);
        }
    }

    static float link_thickness()
    {
        switch (visual_style) {
        case VisualStyle::Legacy:
            return 4.5f;
        case VisualStyle::Prism:
            return 4.0f;
        case VisualStyle::Minimal:
            return 2.0f;
        case VisualStyle::Modern:
        default:
            return 3.0f;
        }
    }

    static bool link_flow_enabled()
    {
        return visual_style != VisualStyle::Minimal;
    }

    static ImVec2 calc_text_size(const char* text, ImFont* font = nullptr, float font_size = 0.0f)
    {
        if (font != nullptr) {
            const float size = font_size > 0.0f ? font_size : font->LegacySize;
            return font->CalcTextSizeA(size, FLT_MAX, 0.0f, text);
        }
        return ImGui::CalcTextSize(text);
    }

    static void draw_text_glow(
        ImDrawList* draw_list,
        const ImVec2& pos,
        ImU32 text_color,
        ImU32 glow_color,
        const char* text,
        ImFont* font = nullptr,
        float font_size = 0.0f)
    {
        if (font == nullptr) {
            font = ImGui::GetFont();
        }
        const float size = font_size > 0.0f ? font_size : font->LegacySize;
        for (int dx = -2; dx <= 2; ++dx) {
            for (int dy = -2; dy <= 2; ++dy) {
                const int d2 = dx * dx + dy * dy;
                if (d2 > 0 && d2 <= 4) {
                    draw_list->AddText(font, size, pos + ImVec2(static_cast<float>(dx), static_cast<float>(dy)), glow_color, text);
                }
            }
        }
        draw_list->AddText(font, size, pos + ImVec2(1.5f, 2.0f), IM_COL32(0, 0, 0, 180), text);
        draw_list->AddText(font, size, pos, text_color, text);
    }

    static void draw_rainbow_border(ImDrawList* draw_list, const ImRect& rect)
    {
        const float t = static_cast<float>(ImGui::GetTime()) * 0.18f;
        for (int i = 0; i < 10; ++i) {
            const float hue = std::fmod(t + i * 0.08f, 1.0f);
            const ImColor col = ImColor::HSV(hue, 0.75f, 1.0f, 0.28f - i * 0.018f);
            const float expand = static_cast<float>(i) * 1.5f;
            draw_list->AddRect(
                rect.Min - ImVec2(expand, expand),
                rect.Max + ImVec2(expand, expand),
                col,
                16.0f,
                0,
                2.0f
            );
        }
    }

    static void draw_pin(const LGRAPH::Socket& socket, const ImRect& node_rect, const ImVec2& center)
    {
        const bool is_input = socket.kind == LGRAPH::SocketKind::Input;
        const ImU32 pin_color = is_input ? IM_COL32(132, 197, 255, 255) : IM_COL32(126, 235, 143, 255);
        const ImU32 glow_color = is_input ? IM_COL32(77, 140, 255, 120) : IM_COL32(76, 196, 108, 120);
        ImFont* label_font = ImGui::GetFont();
        const float label_size_px = 16.0f;
        const float hit_radius = 12.0f;
        const ImVec2 hit_min = center - ImVec2(hit_radius, hit_radius);
        const ImVec2 hit_max = center + ImVec2(hit_radius, hit_radius);

        ImGui::SetCursorScreenPos(hit_min);
        ed::BeginPin(socket.id, is_input ? ed::PinKind::Input : ed::PinKind::Output);
        ImGui::Dummy(hit_max - hit_min);
        ed::PinRect(hit_min, hit_max);
        ed::PinPivotRect(hit_min, hit_max);
        ed::PinPivotAlignment(is_input ? ImVec2(0.0f, 0.5f) : ImVec2(1.0f, 0.5f));
        ed::EndPin();

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        draw_list->AddCircleFilled(center, 5.5f, pin_color, 18);
        draw_list->AddCircle(center, 9.0f, glow_color, 18, 2.0f);

        const ImVec2 label_size = calc_text_size(socket.name.c_str(), label_font, label_size_px);
        const ImVec2 label_pos = is_input
            ? ImVec2(node_rect.Min.x + 18.0f, center.y - label_size.y * 0.5f)
            : ImVec2(node_rect.Max.x - 18.0f - label_size.x, center.y - label_size.y * 0.5f);
        draw_text_glow(draw_list, label_pos, IM_COL32(244, 248, 255, 240), glow_color, socket.name.c_str(), label_font, label_size_px);
    }

    static GLuint node_preview_texture(LGRAPH::GraphNode& node)
    {
        return LGRAPH::current_node_output_texture(node);
    }

    static ImRect node_screen_rect(const LGRAPH::GraphNode& node)
    {
        const ImVec2 min = ed::CanvasToScreen(ed::GetNodePosition(node.id));
        const ImVec2 size = ed::GetNodeSize(node.id);
        return ImRect(min, min + size);
    }

    static ImVec2 socket_screen_pos(const LGRAPH::Socket& socket)
    {
        auto* node = LGRAPH::find_node(socket.node_id);
        if (node == nullptr) {
            return ImVec2(0.0f, 0.0f);
        }

        const ImRect rect = node_screen_rect(*node);
        if (socket.kind == LGRAPH::SocketKind::Input) {
            for (int i = 0; i < static_cast<int>(node->inputs.size()); ++i) {
                if (node->inputs[i].id == socket.id) {
                    return pin_center_left(rect, i, static_cast<int>(node->inputs.size()));
                }
            }
        }
        else {
            for (int i = 0; i < static_cast<int>(node->outputs.size()); ++i) {
                if (node->outputs[i].id == socket.id) {
                    return pin_center_right(rect, i, static_cast<int>(node->outputs.size()));
                }
            }
        }

        return rect.GetCenter();
    }

    static bool is_self_link(const LGRAPH::Link& link)
    {
        auto* a = LGRAPH::find_socket(link.start_pin_id);
        auto* b = LGRAPH::find_socket(link.end_pin_id);
        return a != nullptr && b != nullptr && a->node_id == b->node_id;
    }

    static int self_link_rank(const LGRAPH::Link& link)
    {
        auto* a = LGRAPH::find_socket(link.start_pin_id);
        auto* b = LGRAPH::find_socket(link.end_pin_id);
        if (a == nullptr || b == nullptr || a->node_id != b->node_id) {
            return 0;
        }

        int rank = 0;
        for (const auto& other : LGRAPH::links) {
            auto* oa = LGRAPH::find_socket(other.start_pin_id);
            auto* ob = LGRAPH::find_socket(other.end_pin_id);
            if (oa == nullptr || ob == nullptr) {
                continue;
            }
            if (oa->node_id == a->node_id && ob->node_id == a->node_id) {
                if (other.id == link.id) {
                    return rank;
                }
                rank += 1;
            }
        }
        return rank;
    }

    static void draw_ribbon_link(
        ImDrawList* draw_list,
        const LGRAPH::Link& link,
        const ImVec2& p0,
        const ImVec2& c0,
        const ImVec2& c1,
        const ImVec2& p1,
        const ImColor& base_color,
        float thickness,
        bool animated)
    {
        const ImVec4 rgba = base_color.Value;
        const int layers = visual_style == VisualStyle::Legacy ? 5 : 3;
        for (int i = layers; i >= 1; --i) {
            const float t = static_cast<float>(i) / static_cast<float>(layers);
            const float width = thickness + t * (visual_style == VisualStyle::Legacy ? 9.0f : 5.0f);
            const int alpha = static_cast<int>((visual_style == VisualStyle::Legacy ? 30.0f : 18.0f) * t);
            draw_list->AddBezierCubic(
                p0,
                c0,
                c1,
                p1,
                IM_COL32(
                    static_cast<int>(rgba.x * 255.0f),
                    static_cast<int>(rgba.y * 255.0f),
                    static_cast<int>(rgba.z * 255.0f),
                    alpha),
                width,
                42
            );
        }

        draw_list->AddBezierCubic(p0, c0, c1, p1, base_color, thickness, 48);

        if (!animated) {
            return;
        }

        const float time = static_cast<float>(ImGui::GetTime()) * 0.35f + link.id.Get() * 0.031f;
        for (int i = 0; i < 6; ++i) {
            const float t = std::fmod(time + i * 0.13f, 1.0f);
            const ImVec2 pos = ImBezierCubicCalc(p0, c0, c1, p1, t);
            const float radius = std::max(2.5f, thickness * 0.42f - i * 0.15f);
            draw_list->AddCircleFilled(pos, radius, IM_COL32(255, 242, 208, 155 - i * 18), 14);
        }
    }

    static void draw_custom_links()
    {
        if (visual_style != VisualStyle::Legacy
            && visual_style != VisualStyle::Prism
            && std::none_of(LGRAPH::links.begin(), LGRAPH::links.end(), [](const LGRAPH::Link& link) { return is_self_link(link); }))
        {
            return;
        }

        ed::Suspend();
        ImDrawList* draw_list = ed::GetHintForegroundDrawList();
        for (const auto& link : LGRAPH::links) {
            auto* start = LGRAPH::find_socket(link.start_pin_id);
            auto* end = LGRAPH::find_socket(link.end_pin_id);
            if (start == nullptr || end == nullptr) {
                continue;
            }

            if (start->kind == LGRAPH::SocketKind::Input) {
                std::swap(start, end);
            }

            const ImVec2 p0 = socket_screen_pos(*start);
            const ImVec2 p1 = socket_screen_pos(*end);
            const ImColor base = link_color_for(link);
            const bool self = start->node_id == end->node_id;

            ImVec2 c0;
            ImVec2 c1;
            if (self) {
                const int rank = self_link_rank(link);
                const float lift = 82.0f + rank * 34.0f;
                const float spread = 94.0f + rank * 18.0f;
                c0 = p0 + ImVec2(spread, -lift);
                c1 = p1 + ImVec2(-spread, -lift - 12.0f);
            }
            else {
                const float dx = std::max(110.0f, std::abs(p1.x - p0.x) * 0.42f);
                const float sway = visual_style == VisualStyle::Legacy
                    ? std::sin(static_cast<float>(ImGui::GetTime()) * 0.75f + link.id.Get() * 0.07f) * 10.0f
                    : 0.0f;
                c0 = p0 + ImVec2(dx, sway);
                c1 = p1 - ImVec2(dx, sway);
            }

            if (visual_style == VisualStyle::Legacy || self) {
                draw_ribbon_link(draw_list, link, p0, c0, c1, p1, base, self ? 4.2f : 4.8f, true);
            }
            else if (visual_style == VisualStyle::Prism) {
                draw_ribbon_link(draw_list, link, p0, c0, c1, p1, base, 3.6f, true);
            }
        }
        ed::Resume();
    }

    static const char* mode_label_for(const LGRAPH::GraphNode& node)
    {
        if (node.kind == LGRAPH::NodeKind::CUDA) {
            return node.use_pingpong ? "T" : "S";
        }
        if (node.kind == LGRAPH::NodeKind::Shader || node.kind == LGRAPH::NodeKind::Image || node.kind == LGRAPH::NodeKind::TextEditor) {
            return node.use_double_buffer ? "D" : "S";
        }
        return "S";
    }

    static ImVec4 mode_color_for(const LGRAPH::GraphNode& node)
    {
        if (node.kind == LGRAPH::NodeKind::CUDA && node.use_pingpong) {
            return ImVec4(0.76f, 0.46f, 0.15f, 0.95f);
        }
        if ((node.kind == LGRAPH::NodeKind::Shader || node.kind == LGRAPH::NodeKind::Image || node.kind == LGRAPH::NodeKind::TextEditor) && node.use_double_buffer) {
            return ImVec4(0.18f, 0.44f, 0.82f, 0.95f);
        }
        return ImVec4(0.18f, 0.20f, 0.28f, 0.92f);
    }

    static void toggle_mode(LGRAPH::GraphNode& node)
    {
        if (node.kind == LGRAPH::NodeKind::CUDA) {
            node.use_pingpong = !node.use_pingpong;
            return;
        }
        if (node.kind == LGRAPH::NodeKind::Shader || node.kind == LGRAPH::NodeKind::Image || node.kind == LGRAPH::NodeKind::TextEditor) {
            node.use_double_buffer = !node.use_double_buffer;
        }
    }

    static void draw_node_controls(LGRAPH::GraphNode& node, const ImRect& ribbon_rect)
    {
        const float button_h = 22.0f;
        const float on_w = 44.0f;
        const float mode_w = 28.0f;
        const float gap = 6.0f;
        const ImVec2 base_pos(ribbon_rect.Max.x - on_w - mode_w - gap - 10.0f, ribbon_rect.Min.y + 6.0f);

        ImGui::PushID(node.id.Get());

        ImGui::SetCursorScreenPos(base_pos);
        ImGui::PushStyleColor(ImGuiCol_Button, node.enabled ? ImVec4(0.16f, 0.56f, 0.22f, 0.92f) : ImVec4(0.54f, 0.18f, 0.16f, 0.92f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.96f, 0.86f, 0.42f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.20f, 0.82f, 0.28f, 1.0f));
        if (ImGui::Button(node.enabled ? "ON" : "OFF", ImVec2(on_w, button_h))) {
            node.enabled = !node.enabled;
        }
        ImGui::PopStyleColor(3);

        ImGui::SetCursorScreenPos(base_pos + ImVec2(on_w + gap, 0.0f));
        const ImVec4 mode_color = mode_color_for(node);
        ImGui::PushStyleColor(ImGuiCol_Button, mode_color);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(mode_color.x + 0.08f, mode_color.y + 0.08f, mode_color.z + 0.08f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(mode_color.x, mode_color.y, mode_color.z, 1.0f));
        if (ImGui::Button(mode_label_for(node), ImVec2(mode_w, button_h))) {
            toggle_mode(node);
        }
        ImGui::PopStyleColor(3);

        ImGui::PopID();
    }

    static void draw_node(LGRAPH::GraphNode& node)
    {
        const ImVec2 node_size(node_width_for(node), node_height_for(node));
        const GLuint preview_texture = node_preview_texture(node);
        ImFont* title_font = ImGui::GetFont();
        ImFont* body_font = ImGui::GetFont();
        const float title_size_px = 22.0f;
        const float body_size_px = 16.5f;

        ed::PushStyleVar(ed::StyleVar_NodePadding, ImVec4(0.0f, 0.0f, 0.0f, 0.0f));
        ed::PushStyleColor(ed::StyleColor_NodeBg, ImVec4(0, 0, 0, 0));
        ed::PushStyleColor(ed::StyleColor_NodeBorder, ImVec4(0, 0, 0, 0));
        ed::PushStyleColor(ed::StyleColor_HovNodeBorder, ImVec4(0, 0, 0, 0));
        ed::PushStyleColor(ed::StyleColor_SelNodeBorder, ImVec4(0, 0, 0, 0));

        ed::BeginNode(node.id);
        ImGui::PushID(node.id.Get());
        ImGui::BeginGroup();
        ImGui::Dummy(node_size);

        const ImRect node_rect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        const bool selected = ed::IsNodeSelected(node.id) || LGRAPH::selected_node_id == node.id;
        const ImRect ribbon_rect(ImVec2(node_rect.Min.x + 6.0f, node_rect.Max.y - 31.0f), ImVec2(node_rect.Max.x - 6.0f, node_rect.Max.y - 6.0f));

        const ImVec2 title_size = calc_text_size(node.label.c_str(), title_font, title_size_px);
        const ImVec2 title_pos(node_rect.Min.x + 10.0f, node_rect.Min.y - title_size.y - 8.0f);

        draw_list->AddRectFilled(node_rect.Min, node_rect.Max, IM_COL32(20, 24, 31, 246), 16.0f);
        if (preview_texture != 0) {
            draw_list->AddImageRounded(
                (ImTextureID)(intptr_t)preview_texture,
                node_rect.Min,
                node_rect.Max,
                ImVec2(0.0f, 1.0f),
                ImVec2(1.0f, 0.0f),
                IM_COL32(255, 255, 255, node.enabled ? 255 : 120),
                16.0f
            );
        }
        else {
            const ImColor accent = node.color;
            draw_list->AddRectFilledMultiColor(
                node_rect.Min,
                node_rect.Max,
                IM_COL32(12, 16, 22, 250),
                IM_COL32(accent.Value.x * 255.0f, accent.Value.y * 255.0f, accent.Value.z * 255.0f, 120),
                IM_COL32(accent.Value.x * 255.0f, accent.Value.y * 255.0f, accent.Value.z * 255.0f, 70),
                IM_COL32(12, 16, 22, 250)
            );
        }

        draw_list->AddRectFilled(
            ribbon_rect.Min,
            ribbon_rect.Max,
            IM_COL32(12, 16, 22, 112),
            12.0f
        );

        if (!node.enabled) {
            draw_list->AddRectFilled(node_rect.Min, node_rect.Max, IM_COL32(128, 128, 128, 76), 16.0f);
        }

        const ImU32 accent = kind_accent(node);
        const int glow_layers = visual_style == VisualStyle::Legacy ? 7 : (visual_style == VisualStyle::Minimal ? 2 : 4);
        const float glow_step = visual_style == VisualStyle::Legacy ? 2.0f : 1.5f;
        const int glow_alpha = visual_style == VisualStyle::Legacy ? 26 : 18;
        for (int i = 0; i < glow_layers; ++i) {
            const float expand = static_cast<float>(glow_layers - 1 - i) * glow_step;
            const int alpha = glow_alpha + (glow_layers - 1 - i) * 12;
            draw_list->AddRect(
                node_rect.Min - ImVec2(expand, expand),
                node_rect.Max + ImVec2(expand, expand),
                IM_COL32((accent >> IM_COL32_R_SHIFT) & 0xFF, (accent >> IM_COL32_G_SHIFT) & 0xFF, (accent >> IM_COL32_B_SHIFT) & 0xFF, alpha),
                16.0f,
                0,
                2.0f
            );
        }
        draw_list->AddRect(node_rect.Min, node_rect.Max, selected ? IM_COL32(255, 221, 118, 255) : IM_COL32(96, 110, 130, 220), 16.0f, 0, selected ? 3.5f : 2.0f);

        if (node.kind == LGRAPH::NodeKind::TextEditor || node.kind == LGRAPH::NodeKind::CUDA || node.kind == LGRAPH::NodeKind::OptiX) {
            draw_rainbow_border(draw_list, node_rect);
        }

        draw_node_controls(node, ribbon_rect);

        draw_text_glow(
            draw_list,
            title_pos,
            IM_COL32(255, 255, 255, 245),
            IM_COL32(30, 60, 180, 54),
            node.label.c_str(),
            title_font,
            title_size_px
        );
        draw_text_glow(
            draw_list,
            ImVec2(ribbon_rect.Min.x + 10.0f, ribbon_rect.Min.y + 4.0f),
            IM_COL32(232, 236, 242, 235),
            IM_COL32(180, 60, 30, 42),
            LGRAPH::kind_name(node.kind).c_str(),
            body_font,
            body_size_px
        );

        if (LGRAPH::projected_node_id == node.id) {
            const char* projected = "PROJECTED";
            const ImVec2 label_size = calc_text_size(projected, body_font, body_size_px);
            const ImVec2 badge_min(node_rect.Max.x - label_size.x - 18.0f, node_rect.Min.y + 10.0f);
            const ImVec2 badge_max(node_rect.Max.x - 10.0f, node_rect.Min.y + 29.0f);
            draw_list->AddRectFilled(badge_min, badge_max, IM_COL32(255, 210, 74, 212), 8.0f);
            draw_list->AddText(body_font, body_size_px, ImVec2(badge_min.x + 7.0f, badge_min.y + 2.0f), IM_COL32(20, 16, 8, 255), projected);
        }

        if (node.kind == LGRAPH::NodeKind::Image) {
            const str dims = fstr("{}x{}", node.resource_width, node.resource_height);
            const ImVec2 dims_size = calc_text_size(dims.c_str(), body_font, body_size_px);
            draw_list->AddText(
                ImVec2(node_rect.Max.x - dims_size.x - 12.0f, node_rect.Max.y - 24.0f),
                IM_COL32(220, 226, 232, 226),
                dims.c_str()
            );
        }

        if (!node.last_error.empty()) {
            draw_list->AddRectFilled(
                ImVec2(node_rect.Min.x + 10.0f, node_rect.Max.y - 28.0f),
                ImVec2(node_rect.Min.x + 72.0f, node_rect.Max.y - 10.0f),
                IM_COL32(164, 48, 48, 204),
                8.0f
            );
            draw_list->AddText(ImVec2(node_rect.Min.x + 20.0f, node_rect.Max.y - 25.0f), IM_COL32(255, 244, 244, 255), "Issue");

            if (ImGui::IsMouseHoveringRect(node_rect.Min, node_rect.Max)) {
                ImGui::SetTooltip("%s", node.last_error.c_str());
            }
        }

        for (int i = 0; i < static_cast<int>(node.inputs.size()); ++i) {
            draw_pin(node.inputs[i], node_rect, pin_center_left(node_rect, i, static_cast<int>(node.inputs.size())));
        }

        for (int i = 0; i < static_cast<int>(node.outputs.size()); ++i) {
            draw_pin(node.outputs[i], node_rect, pin_center_right(node_rect, i, static_cast<int>(node.outputs.size())));
        }

        ImGui::EndGroup();
        ImGui::PopID();
        ed::EndNode();

        ed::PopStyleColor(4);
        ed::PopStyleVar();
    }

    void draw_menu_bar()
    {
        if (!ImGui::BeginMenuBar()) {
            return;
        }

        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Clear Graph")) {
                LGRAPH::clear_graph();
            }

            if (ImGui::MenuItem("Reload Sources")) {
                G::read_shaders_from_folder();
                CUDA_RT::read_cuda_from_folder();
                OPTIX_RT::load_optix_programs("optix_programs");
            }

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Shader")) {
            for (const auto& [name, code] : G::named_code) {
                if (name == "vertex_shader") {
                    continue;
                }

                if (ImGui::MenuItem(name.c_str())) {
                    auto& node = LGRAPH::spawn_code_node(LGRAPH::NodeKind::Shader, name, code);
                    ed::SetNodePosition(node.id, LGRAPH::consume_spawn_pos());
                    LGRAPH::select_node(node.id);
                }
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("CUDA")) {
            for (const auto& [name, code] : CUDA_RT::named_cuda_code) {
                if (ImGui::MenuItem(name.c_str())) {
                    auto& node = LGRAPH::spawn_code_node(LGRAPH::NodeKind::CUDA, name, code);
                    ed::SetNodePosition(node.id, LGRAPH::consume_spawn_pos());
                    LGRAPH::select_node(node.id);
                }
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("OptiX")) {
            for (const auto& [name, code] : OPTIX_RT::named_optix_code) {
                if (ImGui::MenuItem(name.c_str())) {
                    auto& node = LGRAPH::spawn_code_node(LGRAPH::NodeKind::OptiX, name, code);
                    ed::SetNodePosition(node.id, LGRAPH::consume_spawn_pos());
                    LGRAPH::select_node(node.id);
                }
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Resource")) {
            ImGui::MenuItem("Drop image files in this window", nullptr, false, false);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Style")) {
            for (int i = 0; i < 4; ++i) {
                const auto style = static_cast<VisualStyle>(i);
                if (ImGui::MenuItem(visual_style_name(style), nullptr, visual_style == style)) {
                    visual_style = style;
                }
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Special")) {
            if (ImGui::MenuItem("Text Editor")) {
                auto& node = LGRAPH::spawn_code_node(
                    LGRAPH::NodeKind::TextEditor,
                    "Text Editor",
                    LGRAPH::text_editor_shader_source()
                );
                ed::SetNodePosition(node.id, LGRAPH::consume_spawn_pos());
                LGRAPH::select_node(node.id);
            }
            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }

    void handle_selection()
    {
        if (ed::HasSelectionChanged()) {
            ed::NodeId selected_nodes[1];
            if (ed::GetSelectedNodes(selected_nodes, 1) > 0) {
                LGRAPH::select_node(selected_nodes[0]);
            }
        }

        if (ImGui::IsMouseClicked(0)) {
            const ed::NodeId hovered = ed::GetHoveredNode();
            if (hovered) {
                LGRAPH::select_node(hovered);
            }
        }
    }

    void handle_create_delete()
    {
        if (ed::BeginCreate()) {
            ed::PinId start_pin_id;
            ed::PinId end_pin_id;
            if (ed::QueryNewLink(&start_pin_id, &end_pin_id)) {
                auto* a = LGRAPH::find_socket(start_pin_id);
                auto* b = LGRAPH::find_socket(end_pin_id);
                str reason;
                if (LGRAPH::can_create_link(a, b, &reason)) {
                    if (ed::AcceptNewItem(ImColor(85, 220, 130), 2.0f)) {
                        LGRAPH::connect(start_pin_id, end_pin_id);
                    }
                }
                else {
                    if (!reason.empty()) {
                        ImGui::SetTooltip("%s", reason.c_str());
                    }
                    ed::RejectNewItem(ImColor(220, 92, 92), 2.0f);
                }
            }
        }
        ed::EndCreate();

        if (ed::BeginDelete()) {
            ed::LinkId deleted_link;
            while (ed::QueryDeletedLink(&deleted_link)) {
                if (ed::AcceptDeletedItem()) {
                    queue_link_delete(deleted_link);
                }
            }

            ed::NodeId deleted_node;
            while (ed::QueryDeletedNode(&deleted_node)) {
                if (ed::AcceptDeletedItem()) {
                    queue_node_delete(deleted_node);
                }
            }
        }
        ed::EndDelete();
    }

    void setup_window_callbacks()
    {
        glfwSetDropCallback(my_handle(), [](GLFWwindow* window, int count, const char** paths) {
            auto* self = Window::get_userptr<Node_Window>(window);
            if (self == nullptr) {
                return;
            }

            double mouse_x = 0.0;
            double mouse_y = 0.0;
            glfwGetCursorPos(window, &mouse_x, &mouse_y);
            for (int i = 0; i < count; ++i) {
                PendingDrop drop;
                drop.path = paths[i];
                drop.mouse_pos = ImVec2(static_cast<float>(mouse_x), static_cast<float>(mouse_y));
                self->pending_drops.push_back(drop);
            }
        });
    }

    void process_pending_drops()
    {
        if (pending_drops.empty()) {
            return;
        }

        for (const auto& drop : pending_drops) {
            if (!LIMG::is_supported_image_file(drop.path)) {
                continue;
            }

            auto* node = LGRAPH::spawn_image_node_from_file(drop.path);
            if (node == nullptr) {
                continue;
            }

            ed::SetNodePosition(node->id, ed::ScreenToCanvas(drop.mouse_pos));
            LGRAPH::select_node(node->id);
        }

        pending_drops.clear();
    }

    void render_node_editor()
    {
        use_context_glfw();
        use_context_imgui();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(my_handle(), &width, &height);
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(static_cast<float>(width), static_cast<float>(height)));

        ImGui::Begin(
            my_title.c_str(),
            nullptr,
            ImGuiWindowFlags_NoResize
                | ImGuiWindowFlags_NoMove
                | ImGuiWindowFlags_NoCollapse
                | ImGuiWindowFlags_MenuBar
        );

        draw_menu_bar();

        ed::SetCurrentEditor(LGRAPH::editor_context);
        ed::Begin("LumenFluxGraph");

        process_pending_drops();
        handle_selection();

        for (auto& node : LGRAPH::nodes) {
            if (LGRAPH::can_emit_texture(node->kind)) {
                std::set<int> visiting;
                LGRAPH::render_node_output_texture(*node, visiting, true);
            }
        }

        for (auto& node : LGRAPH::nodes) {
            draw_node(*node);
        }

        for (const auto& link : LGRAPH::links) {
            ImVec4 base_color = link_color_for(link);
            const bool self_link = is_self_link(link);
            const bool custom_link = visual_style == VisualStyle::Legacy || visual_style == VisualStyle::Prism || self_link;
            if (!self_link) {
            if (custom_link) {
                base_color.w = std::min(base_color.w, 0.05f);
                ed::Link(link.id, link.start_pin_id, link.end_pin_id, base_color, std::max(1.0f, link_thickness() * 0.3f));
            }
            else {
                ed::Link(link.id, link.start_pin_id, link.end_pin_id, base_color, link_thickness());
            }
            }

            if (link_flow_enabled() && !custom_link) {
                ed::Flow(link.id);
            }
        }

        draw_custom_links();

        handle_create_delete();

        ed::End();
        ImGui::End();

        ImGui::Render();
        ImDrawData* draw_data = ImGui::GetDrawData();
        glViewport(0, 0, width, height);
        glClearColor(0.08f, 0.10f, 0.13f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(draw_data);
        process_pending_graph_mutations();
    }

    Node_Window(int w, int h, const str& title, GLFWmonitor* monitor = nullptr, GLFWwindow* share = nullptr)
        : ImGui_Window(w, h, title, monitor, share)
    {
        init_node_editor();
        this_2_userptr();
        setup_window_callbacks();
        set_render_callback([this]() { render_node_editor(); });
    }
};
