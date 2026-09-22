
#pragma once
#include <vector>
#include <cmath>

#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GL/glew.h>

#include "imgui_internal.h"

static inline float len2(const ImVec2& a, const ImVec2& b) {
    float dx = b.x - a.x, dy = b.y - a.y; return sqrtf(dx * dx + dy * dy);
}
static inline ImU32 HSV(float h, float s, float v, float a) {
    float r, g, b; ImGui::ColorConvertHSVtoRGB(h, s, v, r, g, b);
    return ImGui::GetColorU32(ImVec4(r, g, b, a));
}

static void AlignForHeight(float item_height) {
    float frame_h = ImGui::GetFrameHeight();                 // row height decided by style/buttons
    float y_offset = (frame_h - item_height) * 0.5f;
    if (y_offset > 0.0f) ImGui::SetCursorPosY(ImGui::GetCursorPosY() + y_offset);
}


#define FANCY_TOGGLE_USE_GL
#ifdef FANCY_TOGGLE_USE_GL
static void FancySetAdditive(const ImDrawList*, const ImDrawCmd*) {
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE); // additive
}
#endif

static inline float Smoothstep(float a, float b, float x) {
    x = (x - a) / (b - a);
    x = ImClamp(x, 0.0f, 1.0f);
    return x * x * (3.0f - 2.0f * x);
}
static inline ImVec4 Mix(ImVec4 A, ImVec4 B, float t) {
    return ImVec4(A.x + (B.x - A.x) * t, A.y + (B.y - A.y) * t, A.z + (B.z - A.z) * t, A.w + (B.w - A.w) * t);
}
static inline ImU32 U32(ImVec4 c) { return ImGui::GetColorU32(c); }


static inline float SegLen(const ImVec2& a, const ImVec2& b) {
    float dx = b.x - a.x, dy = b.y - a.y;
    return sqrtf(dx * dx + dy * dy);
}

static inline ImVec2 SnapHalf(ImVec2 p) { return ImVec2(floorf(p.x) + 0.5f, floorf(p.y) + 0.5f); }

// Equal-arc-length rounded-rect polyline (≈1 px per step @ current scale)
static void BuildRoundedRectPolylineEqual(std::vector<ImVec2>& out,
    const ImRect& rect, float radius,
    float px_step = 1.0f) // desired step in pixels
{
    out.clear();
    float w = rect.GetWidth(), h = rect.GetHeight();
    radius = ImClamp(radius, 0.0f, ImMin(w, h) * 0.5f);

    auto snap = [](ImVec2 p) { return ImVec2(floorf(p.x) + 0.5f, floorf(p.y) + 0.5f); };

    // corners
    ImVec2 cTL(rect.Min.x + radius, rect.Min.y + radius);
    ImVec2 cTR(rect.Max.x - radius, rect.Min.y + radius);
    ImVec2 cBR(rect.Max.x - radius, rect.Max.y - radius);
    ImVec2 cBL(rect.Min.x + radius, rect.Max.y - radius);

    auto arc = [&](const ImVec2& c, float r, float a0, float a1) {
        float ang = fabsf(a1 - a0);
        int segs = ImMax(8, (int)ceilf(ang * r / ImMax(0.5f, px_step)));
        for (int i = 0; i <= segs; ++i) {
            float u = (float)i / (float)segs;
            float a = a0 + (a1 - a0) * u;
            out.push_back(snap(ImVec2(c.x + cosf(a) * r, c.y + sinf(a) * r)));
        }
    };
    auto edge = [&](ImVec2 a, ImVec2 b) {
        float L = sqrtf((b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y));
        int segs = ImMax(1, (int)ceilf(L / ImMax(0.5f, px_step)));
        for (int i = 1; i <= segs; ++i) {
            float u = (float)i / (float)segs;
            out.push_back(snap(ImVec2(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u)));
        }
    };

    // Start at bottom-center (seam hidden), go clockwise
    arc(cBL, radius, IM_PI * 0.5f, IM_PI);                          // bottom-left corner
    edge(ImVec2(cBL.x, rect.Max.y), ImVec2(cBR.x, rect.Max.y));   // bottom edge
    arc(cBR, radius, 0.0f, IM_PI * 0.5f);                           // bottom-right
    edge(ImVec2(rect.Max.x, cBR.y), ImVec2(rect.Max.x, cTR.y));   // right edge
    arc(cTR, radius, -IM_PI * 0.5f, 0.0f);                          // top-right
    edge(ImVec2(cTR.x, rect.Min.y), ImVec2(cTL.x, rect.Min.y));   // top edge
    arc(cTL, radius, IM_PI, IM_PI * 1.5f);                          // top-left
    edge(ImVec2(rect.Min.x, cTL.y), ImVec2(rect.Min.x, cBL.y));   // left edge
}


// Build a clockwise rounded-rect polyline that follows the exact silhouette.
// edge_steps controls straight-edge subdivision; arc_steps controls corner smoothness.
static void BuildRoundedRectPolyline(std::vector<ImVec2>& out,
    const ImRect& rect, float radius,
    int edge_steps = 16, int arc_steps = 12)
{
    out.clear();
    ImVec2 tl(rect.Min.x + radius, rect.Min.y + radius);
    ImVec2 tr(rect.Max.x - radius, rect.Min.y + radius);
    ImVec2 br(rect.Max.x - radius, rect.Max.y - radius);
    ImVec2 bl(rect.Min.x + radius, rect.Max.y - radius);

    auto add_arc = [&](const ImVec2& c, float r, float a0, float a1) {
        for (int i = 0; i <= arc_steps; ++i) {
            float u = (float)i / (float)arc_steps;
            float a = a0 + (a1 - a0) * u;
            out.push_back(ImVec2(c.x + cosf(a) * r, c.y + sinf(a) * r));
        }
    };
    auto add_edge = [&](const ImVec2& a, const ImVec2& b) {
        for (int i = 1; i <= edge_steps; ++i) {
            float u = (float)i / (float)edge_steps;
            out.push_back(ImVec2(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u));
        }
    };

    // Start at top-left corner (end of arc), go clockwise
    add_arc(tl, radius, IM_PI, IM_PI * 1.5f);           // top-left
    add_edge(ImVec2(tl.x, rect.Min.y), ImVec2(tr.x, rect.Min.y)); // top edge
    add_arc(tr, radius, -IM_PI * 0.5f, 0.0f);           // top-right
    add_edge(ImVec2(rect.Max.x, tr.y), ImVec2(rect.Max.x, br.y)); // right edge
    add_arc(br, radius, 0.0f, IM_PI * 0.5f);            // bottom-right
    add_edge(ImVec2(br.x, rect.Max.y), ImVec2(bl.x, rect.Max.y)); // bottom edge
    add_arc(bl, radius, IM_PI * 0.5f, IM_PI);           // bottom-left
    add_edge(ImVec2(rect.Min.x, bl.y), ImVec2(rect.Min.x, tl.y)); // left edge
}


namespace MyWidgets
{
    bool FancyToggle(const char* label, bool* v,
        ImVec2 size = ImVec2(56, 30),
        int glow_rings = 18, float glow_outer_scale = 2.2f, float glow_strength = 1.0f)
    {
        ImGuiWindow* window = ImGui::GetCurrentWindow();
        if (window->SkipItems) return false;

        ImGuiContext& g = *GImGui;
        const ImGuiStyle& style = g.Style;

        // Rect & ID
        ImVec2 pos = window->DC.CursorPos;
        ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
        ImGuiID id = window->GetID(label);
        ImGui::ItemSize(bb, style.FramePadding.y);
        if (!ImGui::ItemAdd(bb, id)) return false;

        // Interaction
        bool hovered, held;
        bool pressed = ImGui::ButtonBehavior(bb, id, &hovered, &held, ImGuiButtonFlags_PressedOnClick);
        if (pressed) *v = !*v;

        // Anim state 0..1
        ImGuiStorage* storage = &window->StateStorage;   // StateStorage is a value member; take its address
        float t = storage->GetFloat(id, *v ? 1.0f : 0.0f);
        float target = *v ? 1.0f : 0.0f;
        float k = ImGui::GetIO().DeltaTime * 14.0f;
        t = t + (target - t) * ImClamp(k, 0.0f, 1.0f);
        storage->SetFloat(id, t);

        auto Mix = [](ImVec4 a, ImVec4 b, float u) { return ImVec4(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u, a.z + (b.z - a.z) * u, a.w + (b.w - a.w) * u); };
        auto U32 = [](ImVec4 c) { return ImGui::GetColorU32(c); };
        auto Smooth = [](float e0, float e1, float x) {
            x = ImClamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
            return x * x * (3.0f - 2.0f * x);
        };

        // Palette
        const ImVec4 ACCENT_ON = ImVec4(0.22f, 0.77f, 0.41f, 1.00f); // emerald
        const ImVec4 ACCENT_OFF = ImVec4(0.95f, 0.40f, 0.20f, 1.00f); // orange-red
        const ImVec4 TRACK_DARK = ImVec4(0.18f, 0.18f, 0.18f, 0.92f);

        const ImVec4 KNOB_ON_BASE = ImVec4(0.98f, 0.995f, 0.985f, 1.0f); // near-white with hint of mint
        const ImVec4 KNOB_OFF_BASE = ImVec4(1.00f, 0.78f, 0.55f, 1.0f);   // warm OFF
        const ImVec4 KNOB_RING_ON = Mix(ACCENT_ON, ImVec4(0, 0, 0, 1), 0.35f);
        const ImVec4 KNOB_RING_OFF = Mix(ACCENT_OFF, ImVec4(0, 0, 0, 1), 0.35f);

        // Track
        ImVec4 track_col = Mix(TRACK_DARK, Mix(TRACK_DARK, ACCENT_ON, 0.35f), t);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        float r = bb.GetHeight() * 0.5f;
        dl->AddRectFilled(bb.Min, bb.Max, U32(track_col), r);

        // Knob position
        float knob_x = ImLerp(bb.Min.x + r, bb.Max.x - r, t);
        float press_y = held ? 1.0f : 0.0f;
        ImVec2 knob_center(knob_x, bb.Min.y + r + press_y);

        // ===== Natural GREEN GLOW when ON =====
        if (t > 0.001f && glow_rings > 0)
        {
            float pulse = 0.95f + 0.05f * sinf(6.0f * (float)ImGui::GetTime());
            float intensity = glow_strength * t * pulse * (hovered ? 1.15f : 1.0f);
            float inner = r * 0.70f;
            float outer = r * glow_outer_scale;
            for (int i = 0; i < glow_rings; ++i)
            {
                float u = (float)i / (float)(glow_rings - 1);
                float rad = ImLerp(inner, outer, u);
                float fall = (1.0f - u); fall *= fall;                // quadratic falloff
                float a = intensity * (0.28f * fall);
                ImVec4 gcol = ImVec4(ACCENT_ON.x, ACCENT_ON.y, ACCENT_ON.z, a);
                dl->AddCircleFilled(knob_center, rad, U32(gcol));
            }
        }

        // ===== Knob (layered/3D) =====
        float knob_rad = r * (0.82f - (held ? 0.02f : 0.0f) + (hovered ? 0.01f : 0.0f));

        // Drop shadow
        {
            ImVec2 sh_center = ImVec2(knob_center.x, knob_center.y + r * 0.20f);
            for (int i = 0; i < 6; ++i) {
                float u = (float)i / 5.0f;
                float rad = knob_rad * (1.00f + u * 0.45f);
                float a = (0.22f * (1.0f - u)) * 0.65f;
                dl->AddCircleFilled(sh_center, rad, IM_COL32(0, 0, 0, (int)(255 * a)));
            }
        }

        // Base fill (blend OFF→ON)
        ImVec4 knob_base = Mix(KNOB_OFF_BASE, KNOB_ON_BASE, t);
        dl->AddCircleFilled(knob_center, knob_rad, U32(knob_base));

        // Rim (accented; blends OFF→ON)
        ImVec4 rim_col = Mix(KNOB_RING_OFF, KNOB_RING_ON, t);
        dl->AddCircle(knob_center, knob_rad, U32(rim_col), 0, 1.6f);

        // === ON-FANCY: mint→emerald edge tint + fresnel + sparkle (fades in with t) ===
        float fancy = Smooth(0.55f, 1.0f, t);           // only when mostly ON

        if (fancy > 0.0f)
        {
            // Subtle mint-to-emerald tint toward the rim (radial layers)
            ImVec4 MINT = ImVec4(0.88f, 1.00f, 0.92f, 1.0f);
            ImVec4 EMERALD = ACCENT_ON;
            int layers = 10;
            for (int i = 0; i < layers; ++i) {
                float u = (float)i / (layers - 1);                  // 0..1 center->edge
                float edge = Smooth(0.35f, 1.0f, u);                // emphasize rim
                ImVec4 tint = Mix(MINT, EMERALD, edge);
                float alpha = 0.10f * (1.0f - u) * fancy;           // very soft
                float rad = knob_rad * (1.0f - u * 0.90f);
                dl->AddCircleFilled(knob_center, rad, U32(ImVec4(tint.x, tint.y, tint.z, alpha)));
            }

            // Fresnel rim: multiple thin strokes, greener toward edge
            int rim_layers = 5;
            for (int i = 0; i < rim_layers; ++i) {
                float u = (float)i / (rim_layers - 1);
                float rad = knob_rad * (0.90f + 0.06f * u);
                float a = (0.20f * (1.0f - u)) * fancy;
                ImVec4 c = Mix(EMERALD, ImVec4(1, 1, 1, 1), 0.35f);   // greenish-white
                dl->AddCircle(knob_center, rad, U32(ImVec4(c.x, c.y, c.z, a)), 0, 1.2f);
            }

            // Animated sparkle/glint across the dome (subtle)
            float s = fmodf((float)ImGui::GetTime() * 0.8f, 1.0f);  // 0..1 sweep
            float sweep = -0.25f + s * 0.50f;                       // left->right
            ImVec2 gl = ImVec2(knob_center.x + sweep * knob_rad, knob_center.y - knob_rad * 0.35f);
            dl->AddCircleFilled(gl, knob_rad * 0.18f, IM_COL32(255, 255, 255, (int)(70 * fancy)));
            dl->AddCircleFilled(gl, knob_rad * 0.10f, IM_COL32(255, 255, 255, (int)(110 * fancy)));
        }

        // Inner soft radial shading (global curvature)
        {
            int layers = 10;
            for (int i = 0; i < layers; ++i) {
                float u = (float)i / (layers - 1);
                float rad = knob_rad * (1.0f - u * 0.92f);
                float dark = 0.10f * (1.0f - u);
                ImVec4 col = Mix(knob_base, ImVec4(0, 0, 0, 1), dark);
                col.w = 0.10f;
                dl->AddCircleFilled(knob_center, rad, U32(col));
            }
        }

        // Bottom ambient occlusion arc (seated look)
        {
            float thick = knob_rad * 0.22f;
            dl->PathClear();
            dl->PathArcTo(knob_center, knob_rad * 0.92f, IM_PI * 0.15f, IM_PI * 0.85f, 32);
            dl->PathStroke(IM_COL32(0, 0, 0, 40), 0, thick);
        }

        // Specular highlight (top-left, base)
        {
            ImVec2 h = ImVec2(knob_center.x - knob_rad * 0.30f, knob_center.y - knob_rad * 0.30f);
            dl->AddCircleFilled(h, knob_rad * 0.38f, IM_COL32(255, 255, 255, 76));
            dl->AddCircleFilled(h, knob_rad * 0.22f, IM_COL32(255, 255, 255, 120));
        }

        // Subtle inner ring highlight
        dl->AddCircle(knob_center, knob_rad * 0.70f, IM_COL32(255, 255, 255, 45), 0, 1.2f);

        return pressed; // true on state flip
    }


    bool FancyTogglePlus(const char* label, bool* v,
        ImVec2 size = ImVec2(56, 30),
        int glow_rings = 18, float glow_outer_scale = 2.2f, float glow_strength = 1.0f,
        bool use_additive_glow = false, bool embossed_checkmark = true)
    {
        ImGuiWindow* window = ImGui::GetCurrentWindow();
        if (window->SkipItems) return false;

        ImGuiContext& g = *GImGui;
        const ImGuiStyle& style = g.Style;

        // Rect & ID
        ImVec2 pos = window->DC.CursorPos;
        ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
        ImGuiID id = window->GetID(label);
        ImGui::ItemSize(bb, style.FramePadding.y);
        if (!ImGui::ItemAdd(bb, id)) return false;

        // Interaction
        bool hovered, held;
        bool pressed = ImGui::ButtonBehavior(bb, id, &hovered, &held, ImGuiButtonFlags_PressedOnClick);
        if (pressed) *v = !*v;

        // Anim 0..1
        ImGuiStorage* storage = &window->StateStorage;
        float t = storage->GetFloat(id, *v ? 1.0f : 0.0f);
        float target = *v ? 1.0f : 0.0f;
        float k = ImGui::GetIO().DeltaTime * 14.0f;
        t = t + (target - t) * ImClamp(k, 0.0f, 1.0f);
        storage->SetFloat(id, t);

        // Palette
        const ImVec4 ACCENT_ON = ImVec4(0.18f, 0.82f, 0.42f, 1.00f); // emerald
        const ImVec4 ACCENT_OFF = ImVec4(0.95f, 0.40f, 0.20f, 1.00f); // orange/red
        const ImVec4 TRACK_DARK = ImVec4(0.18f, 0.18f, 0.18f, 0.92f);

        // OFF knob palette
        const ImVec4 KNOB_OFF_BASE = ImVec4(1.00f, 0.78f, 0.55f, 1.0f);
        const ImVec4 KNOB_RING_OFF = Mix(ACCENT_OFF, ImVec4(0, 0, 0, 1), 0.35f);

        // ON knob as green sphere (center mint → rim emerald)
        const ImVec4 KNOB_ON_CENTER = ImVec4(0.86f, 0.98f, 0.90f, 1.0f); // light mint center
        const ImVec4 KNOB_ON_RIM = ACCENT_ON;                          // saturated emerald rim
        const ImVec4 KNOB_RING_ON = Mix(ACCENT_ON, ImVec4(0, 0, 0, 1), 0.35f);

        // Track
        ImVec4 track_col = Mix(TRACK_DARK, Mix(TRACK_DARK, ACCENT_ON, 0.35f), t);
        ImDrawList* dl = ImGui::GetWindowDrawList();
        float r = bb.GetHeight() * 0.5f;
        dl->AddRectFilled(bb.Min, bb.Max, U32(track_col), r);

        // Knob position
        float knob_x = ImLerp(bb.Min.x + r, bb.Max.x - r, t);
        float press_y = held ? 1.0f : 0.0f;
        ImVec2 knob_center(knob_x, bb.Min.y + r + press_y);

        // ===== Glow (rings; optional additive for softer bloom) =====
        if (t > 0.001f && glow_rings > 0) {
            float pulse = 0.95f + 0.05f * sinf(6.0f * (float)ImGui::GetTime());
            float intensity = glow_strength * t * pulse * (hovered ? 1.15f : 1.0f);
            float inner = r * 0.70f;
            float outer = r * glow_outer_scale;

#ifdef FANCY_TOGGLE_USE_GL
            if (use_additive_glow) dl->AddCallback(FancySetAdditive, nullptr);
#endif

            for (int i = 0; i < glow_rings; ++i) {
                float u = (float)i / (float)(glow_rings - 1);
                float rad = ImLerp(inner, outer, u);
                float fall = (1.0f - u); fall *= fall;                // quadratic
                float a = intensity * (use_additive_glow ? 0.14f : 0.28f) * fall;
                dl->AddCircleFilled(knob_center, rad, U32(ImVec4(ACCENT_ON.x, ACCENT_ON.y, ACCENT_ON.z, a)));
            }

#ifdef FANCY_TOGGLE_USE_GL
            if (use_additive_glow) dl->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
#endif
        }

        // ===== Knob (layered/3D) =====
        float knob_rad = r * (0.82f - (held ? 0.02f : 0.0f) + (hovered ? 0.01f : 0.0f));

        // Drop shadow
        {
            ImVec2 sh_center = ImVec2(knob_center.x, knob_center.y + r * 0.20f);
            for (int i = 0; i < 6; ++i) {
                float u = (float)i / 5.0f;
                float rad = knob_rad * (1.00f + u * 0.45f);
                float a = (0.22f * (1.0f - u)) * 0.65f;
                dl->AddCircleFilled(sh_center, rad, IM_COL32(0, 0, 0, (int)(255 * a)));
            }
        }

        // --- OFF look (warm) blended in early; fades out as t→1
        float off_w = (1.0f - Smoothstep(0.15f, 0.45f, t));
        if (off_w > 0.0f) {
            ImVec4 base_off = KNOB_OFF_BASE; base_off.w *= off_w;
            dl->AddCircleFilled(knob_center, knob_rad, U32(base_off));
            ImVec4 rim_off = KNOB_RING_OFF; rim_off.w *= off_w;
            dl->AddCircle(knob_center, knob_rad, U32(rim_off), 0, 1.6f);
        }

        // --- ON look (GREEN SPHERE) fades in as t→1
        float on_w = Smoothstep(0.35f, 1.0f, t);
        if (on_w > 0.0f) {
            // Base rim layer (emerald)
            ImVec4 rim = KNOB_ON_RIM; rim.w *= on_w;
            dl->AddCircleFilled(knob_center, knob_rad, U32(rim));

            // Radial core layers (mint center → emerald rim)
            int layers = 12;
            for (int i = 0; i < layers; ++i) {
                // v=0 at center, 1 at edge; bias toward center to keep rim saturated
                float v = (float)i / (layers - 1);
                float bias = powf(v, 1.6f);
                ImVec4 col = Mix(KNOB_ON_CENTER, KNOB_ON_RIM, bias);
                col.w = 0.75f * (1.0f - v) * on_w;        // stronger near center
                float rad = knob_rad * (1.0f - v * 0.92f);
                dl->AddCircleFilled(knob_center, rad, U32(col));
            }

            // Green rim line for crisp edge
            ImVec4 rimline = Mix(KNOB_ON_RIM, ImVec4(1, 1, 1, 1), 0.10f); rimline.w *= on_w;
            dl->AddCircle(knob_center, knob_rad, U32(rimline), 0, 1.6f);
        }

        // Global curvature shading (very subtle; works for both states)
        {
            int layers = 8;
            for (int i = 0; i < layers; ++i) {
                float u = (float)i / (layers - 1);
                float rad = knob_rad * (1.0f - u * 0.92f);
                float dark = 0.08f * (1.0f - u);
                ImVec4 col = ImVec4(0, 0, 0, dark * 0.8f);
                dl->AddCircleFilled(knob_center, rad, U32(col));
            }
        }

        // Seated AO arc
        {
            float thick = knob_rad * 0.22f;
            dl->PathClear();
            dl->PathArcTo(knob_center, knob_rad * 0.92f, IM_PI * 0.15f, IM_PI * 0.85f, 32);
            dl->PathStroke(IM_COL32(0, 0, 0, 40), 0, thick);
        }

        // Specular highlight (top-left)
        {
            ImVec2 h = ImVec2(knob_center.x - knob_rad * 0.30f, knob_center.y - knob_rad * 0.30f);
            dl->AddCircleFilled(h, knob_rad * 0.36f, IM_COL32(255, 255, 255, 70));
            dl->AddCircleFilled(h, knob_rad * 0.20f, IM_COL32(255, 255, 255, 110));
        }

        // ===== Embossed checkmark (ON) =====
        if (embossed_checkmark) {
            float ck = Smoothstep(0.60f, 1.0f, t);
            if (ck > 0.0f) {
                ImVec2 p0 = ImVec2(knob_center.x - knob_rad * 0.32f, knob_center.y + knob_rad * 0.00f);
                ImVec2 p1 = ImVec2(knob_center.x - knob_rad * 0.08f, knob_center.y + knob_rad * 0.24f);
                ImVec2 p2 = ImVec2(knob_center.x + knob_rad * 0.36f, knob_center.y - knob_rad * 0.22f);
                float th = knob_rad * 0.17f;

                // Shadow
                ImU32 sh = IM_COL32(0, 0, 0, (int)(90 * ck));
                dl->PathClear(); dl->PathLineTo(p0 + ImVec2(1, 1)); dl->PathLineTo(p1 + ImVec2(1, 1)); dl->PathLineTo(p2 + ImVec2(1, 1));
                dl->PathStroke(sh, 0, th);

                // Highlight
                ImU32 hi = IM_COL32(255, 255, 255, (int)(180 * ck));
                dl->PathClear(); dl->PathLineTo(p0 + ImVec2(-1, -1)); dl->PathLineTo(p1 + ImVec2(-1, -1)); dl->PathLineTo(p2 + ImVec2(-1, -1));
                dl->PathStroke(hi, 0, th * 0.85f);

                // Core (near-white so it stays readable on green)
                ImU32 core = IM_COL32(245, 255, 245, (int)(230 * ck));
                dl->PathClear(); dl->PathLineTo(p0); dl->PathLineTo(p1); dl->PathLineTo(p2);
                dl->PathStroke(core, 0, th * 0.92f);
            }
        }

        // Inner ring highlight
        dl->AddCircle(knob_center, knob_rad * 0.70f, IM_COL32(255, 255, 255, 45), 0, 1.2f);

        return pressed;
    }


    bool FancyKnob(const char* label, bool* v,
        ImVec2 size = ImVec2(56, 30),
        // knob glow params
        int glow_rings = 18, float glow_outer_scale = 2.2f, float glow_strength = 1.0f,
        // slot styling toggles
        bool use_additive_glow = false,
        bool embossed_checkmark = true,
        bool colorful_slot_rim = true, int rim_steps_hint = 36, float rim_alpha = 0.18f,
        bool slot_off_glow = true, int slot_glow_layers = 8, float slot_glow_spread = 1.2f,
        bool slot_drop_shadow = true, int shadow_layers = 6, float shadow_spread = 1.4f, float shadow_offset_y = 2.0f)
    {
        ImGui::GetStyle().AntiAliasedLines = true;
        ImGui::GetStyle().AntiAliasedFill = true;

        ImGuiWindow* window = ImGui::GetCurrentWindow();
        if (window->SkipItems) return false;

        ImGuiContext& g = *GImGui;
        const ImGuiStyle& style = g.Style;

        // Layout & ID
        ImVec2 pos = window->DC.CursorPos;
        ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
        ImGuiID id = window->GetID(label);
        ImGui::ItemSize(bb, style.FramePadding.y);
        if (!ImGui::ItemAdd(bb, id)) return false;

        // Interaction
        bool hovered, held;
        bool pressed = ImGui::ButtonBehavior(bb, id, &hovered, &held, ImGuiButtonFlags_PressedOnClick);
        if (pressed) *v = !*v;

        // Animation 0..1
        ImGuiStorage* storage = &window->StateStorage; // note: address of value member
        float t = storage->GetFloat(id, *v ? 1.0f : 0.0f);
        float target = *v ? 1.0f : 0.0f;
        float k = ImGui::GetIO().DeltaTime * 14.0f;
        t = t + (target - t) * ImClamp(k, 0.0f, 1.0f);
        storage->SetFloat(id, t);

        // Palette
        const ImVec4 ACCENT_ON = ImVec4(0.18f, 0.82f, 0.42f, 1.00f); // emerald
        const ImVec4 ACCENT_OFF = ImVec4(0.95f, 0.40f, 0.20f, 1.00f); // orange/red
        const ImVec4 TRACK_BASE = ImVec4(0.0f, 0.0f, 0.1f, 0.95f);
        const ImVec4 TRACK_TINT = ImVec4(0.12f, 0.30f, 0.18f, 0.25f); // subtle green tint when ON

        // Knob palettes
        const ImVec4 KNOB_OFF_BASE = ImVec4(1.00f, 0.78f, 0.1f, 1.0f);
        const ImVec4 KNOB_RING_OFF = Mix(ACCENT_OFF, ImVec4(0, 0, 0, 1), 0.35f);

        const ImVec4 KNOB_ON_CENTER = ImVec4(0.86f, 0.98f, 0.90f, 1.0f); // minty center
        const ImVec4 KNOB_ON_RIM = ACCENT_ON;                          // emerald rim
        const ImVec4 KNOB_RING_ON = Mix(ACCENT_ON, ImVec4(0, 0, 0, 1), 0.35f);

        ImDrawList* dl = ImGui::GetWindowDrawList();
        float r = bb.GetHeight() * 0.5f;

        // ---------- SLOT BACKDROP: drop shadow ----------
        if (slot_drop_shadow) {
            for (int i = 0; i < shadow_layers; ++i) {
                float u = (float)i / ImMax(1, shadow_layers - 1);
                ImRect ex = bb; ex.Expand(u * shadow_spread);
                ImVec2 off(0.0f, shadow_offset_y);
                float alpha = (0.22f * (1.0f - u));
                dl->AddRectFilled(ex.Min + off, ex.Max + off, IM_COL32(0, 0, 0, (int)(255 * alpha)), r + u * shadow_spread);
            }
        }

        // ---------- SLOT BACKDROP: red OFF halo ----------
        if (slot_off_glow) {
            float strength = (1.0f - t) * (hovered ? 1.10f : 1.0f);
            for (int i = 0; i < slot_glow_layers; ++i) {
                float u = (float)i / ImMax(1, slot_glow_layers - 1);
                ImRect ex = bb; ex.Expand(u * slot_glow_spread);
                float fall = (1.0f - u); fall *= fall;
                ImVec4 col = ImVec4(ACCENT_OFF.x, ACCENT_OFF.y, ACCENT_OFF.z, 0.22f * fall * strength);
                dl->AddRectFilled(ex.Min, ex.Max, U32(col), r + u * slot_glow_spread);
            }
        }

        // ---------- SLOT (base) ----------
        // Compute knob color at current animation state
        ImVec4 knob_col = Mix(ACCENT_OFF, ACCENT_ON, t);

        // Blend it with track base for a softer look
        ImVec4 track_col = Mix(TRACK_BASE, knob_col, 0.7f);  // how strongly the slot takes the knob color

        dl->AddRectFilled(bb.Min, bb.Max, U32(track_col), r);

        // Faint inner tint in the knob color
        if (t > 0.0f) {
            ImRect inner = bb; inner.Expand(-1.0f);
            ImVec4 tint = knob_col;
            tint.w = 0.25f * t;  // make it faint
            dl->AddRectFilled(inner.Min, inner.Max, U32(tint), r - 1.0f);
        }

        // Subtle green inner tint when ON
        if (t > 0.0f) {
            ImRect inner = bb; inner.Expand(-1.0f);
            ImVec4 tint = TRACK_TINT; tint.w *= 0.6f * t;
            dl->AddRectFilled(inner.Min, inner.Max, U32(tint), r - 1.0f);
        }


        // ---------- SLOT RIM (rainbow) aligned to silhouette ----------
        if (colorful_slot_rim) {
            const float t_hover = hovered ? 1.2f : 1.0f;
            const float alpha = rim_alpha * t_hover;

            const float inner_offset = -0.75f;
            const float outer_offset = +0.25f;

            ImRect rin = bb;  rin.Expand(inner_offset);
            ImRect rout = bb; rout.Expand(outer_offset);

            // Slightly higher sampling = smoother curves, fewer joints
            int EDGE_STEPS = ImClamp(rim_steps_hint, 18, 64);
            int ARC_STEPS = ImClamp(rim_steps_hint * 3 / 4, 12, 48);


            EDGE_STEPS = 64;
            ARC_STEPS = 64;

            std::vector<ImVec2> path_in, path_out;
            BuildRoundedRectPolyline(path_in, rin, r + inner_offset, EDGE_STEPS, ARC_STEPS);
            BuildRoundedRectPolyline(path_out, rout, r + outer_offset, EDGE_STEPS, ARC_STEPS);

            auto SnapHalf = [](ImVec2 p) { return ImVec2(floorf(p.x) + 0.5f, floorf(p.y) + 0.5f); };
            auto DrawColoredPath = [&](const std::vector<ImVec2>& pts, float thick, float base_alpha) {
                ImDrawList* dl2 = ImGui::GetWindowDrawList();

                // 1) Base stroke underlay to hide pinholes (neutral, low alpha)
                dl2->PathClear();
                for (const ImVec2& p : pts) dl2->PathLineTo(SnapHalf(p));
                dl2->PathStroke(IM_COL32(255, 255, 255, (int)(255 * base_alpha)), true, thick * 0.9f);

                // 2) Rainbow segments with tangent overlap + pixel snapping
                const float overlap = thick * 0.75f;        // extend ends by ~3/4 thickness
                // Perimeter length for hue ramp
                float total = 0.0f;
                for (int i = 0; i < (int)pts.size(); ++i) total += SegLen(pts[i], pts[(i + 1) % pts.size()]);
                float acc = 0.0f;

                for (int i = 0; i < (int)pts.size(); ++i) {
                    ImVec2 a = pts[i];
                    ImVec2 b = pts[(i + 1) % pts.size()];
                    ImVec2 d = ImVec2(b.x - a.x, b.y - a.y);
                    float L = sqrtf(d.x * d.x + d.y * d.y);
                    if (L <= 1e-4f) continue;
                    ImVec2 n = ImVec2(d.x / L, d.y / L);

                    // Extend to overlap neighbors and snap to pixel grid
                    ImVec2 a2 = SnapHalf(ImVec2(a.x - n.x * overlap, a.y - n.y * overlap));
                    ImVec2 b2 = SnapHalf(ImVec2(b.x + n.x * overlap, b.y + n.y * overlap));

                    float u_mid = (acc + L * 0.5f) / ImMax(total, 1e-4f);   // 0..1 along loop
                    float h = 0.02f + 0.62f * u_mid;                        // warm->cool->warm sweep
                    ImU32 col = HSV(h, 0.60f, 1.00f, alpha);

                    // Use slightly *thicker* stroke to guarantee coverage over base
                    dl2->AddLine(a2, b2, col, thick);
                    acc += L;
                }
            };

            // Twin rims hugging the silhouette
            const float THICK = 2.0f;           // try 2.0–2.5 for hi-dpi
            const float BASE = alpha * 0.25f;  // base underlay alpha
            DrawColoredPath(path_in, THICK, BASE);
            DrawColoredPath(path_out, THICK, BASE);
        }

        // ---------- KNOB ----------
        float knob_x = ImLerp(bb.Min.x + r, bb.Max.x - r, t);
        float press_y = held ? 1.0f : 0.0f;
        ImVec2 knob_center(knob_x, bb.Min.y + r + press_y);

        // Knob glow (emerald)
        if (t > 0.001f && glow_rings > 0) {
            float pulse = 0.95f + 0.05f * sinf(6.0f * (float)ImGui::GetTime());
            float intensity = glow_strength * t * pulse * (hovered ? 1.15f : 1.0f);
            float inner = r * 0.70f;
            float outer = r * glow_outer_scale;

#ifdef FANCY_TOGGLE_USE_GL
            if (use_additive_glow) dl->AddCallback(FancySetAdditive, nullptr);
#endif
            for (int i = 0; i < glow_rings; ++i) {
                float uu = (float)i / (glow_rings - 1);
                float rad = ImLerp(inner, outer, uu);
                float fall = (1.0f - uu); fall *= fall;
                float a = intensity * (use_additive_glow ? 0.14f : 0.28f) * fall;
                dl->AddCircleFilled(knob_center, rad, U32(ImVec4(ACCENT_ON.x, ACCENT_ON.y, ACCENT_ON.z, a)), 64);
            }
#ifdef FANCY_TOGGLE_USE_GL
            if (use_additive_glow) dl->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
#endif
        }

        float knob_rad = r * (0.82f - (held ? 0.02f : 0.0f) + (hovered ? 0.01f : 0.0f));

        // Knob shadow
        {
            ImVec2 sh_center = ImVec2(knob_center.x, knob_center.y + r * 0.20f);
            for (int i = 0; i < 6; ++i) {
                float u = (float)i / 5.0f;
                float rad = knob_rad * (1.00f + u * 0.45f);
                float a = (0.22f * (1.0f - u)) * 0.65f;
                dl->AddCircleFilled(sh_center, rad, IM_COL32(0, 0, 0, (int)(255 * a)), 64);
            }
        }

        // OFF look blended early; fades as t→1
        float off_w = (1.0f - Smoothstep(0.15f, 0.45f, t));
        if (off_w > 0.0f) {
            ImVec4 base_off = KNOB_OFF_BASE; base_off.w *= off_w;
            dl->AddCircleFilled(knob_center, knob_rad, U32(base_off), 64);
            ImVec4 rim_off = KNOB_RING_OFF; rim_off.w *= off_w;
            dl->AddCircle(knob_center, knob_rad, U32(rim_off), 64, 1.6f);
        }

        // ON look (green sphere) fades in as t→1
        float on_w = Smoothstep(0.35f, 1.0f, t);
        if (on_w > 0.0f) {
            // Emerald base at rim
            ImVec4 rim = KNOB_ON_RIM; rim.w *= on_w;
            dl->AddCircleFilled(knob_center, knob_rad, U32(rim), 64);

            // Mint center → emerald rim layers
            int layers = 12;
            for (int i = 0; i < layers; ++i) {
                float v2 = (float)i / (layers - 1);
                float bias = powf(v2, 1.6f);
                ImVec4 col = Mix(KNOB_ON_CENTER, KNOB_ON_RIM, bias);
                col.w = 0.75f * (1.0f - v2) * on_w;
                float rad = knob_rad * (1.0f - v2 * 0.92f);
                dl->AddCircleFilled(knob_center, rad, U32(col), 64);
            }
            // Crisp green rimline
            ImVec4 rimline = Mix(KNOB_ON_RIM, ImVec4(1, 1, 1, 1), 0.10f); rimline.w *= on_w;
            dl->AddCircle(knob_center, knob_rad, U32(rimline), 64, 1.6f);
        }

        // Global curvature shading on knob
        {
            int layers = 8;
            for (int i = 0; i < layers; ++i) {
                float u = (float)i / (layers - 1);
                float rad = knob_rad * (1.0f - u * 0.92f);
                float dark = 0.08f * (1.0f - u);
                dl->AddCircleFilled(knob_center, rad, IM_COL32(0, 0, 0, (int)(255 * dark * 0.8f)), 64);
            }
        }

        // Seated AO arc
        {
            float thick = knob_rad * 0.22f;
            dl->PathClear();
            dl->PathArcTo(knob_center, knob_rad * 0.92f, IM_PI * 0.15f, IM_PI * 0.85f, 64);
            dl->PathStroke(IM_COL32(0, 0, 0, 40), 0, thick);
        }

        // Specular highlight
        {
            ImVec2 h = ImVec2(knob_center.x - knob_rad * 0.30f, knob_center.y - knob_rad * 0.30f);
            dl->AddCircleFilled(h, knob_rad * 0.36f, IM_COL32(255, 255, 255, 70), 64);
            dl->AddCircleFilled(h, knob_rad * 0.20f, IM_COL32(255, 255, 255, 110), 64);
        }

        // Embossed checkmark when ON
        if (embossed_checkmark) {
            float ck = Smoothstep(0.60f, 1.0f, t);
            if (ck > 0.0f) {
                ImVec2 p0 = ImVec2(knob_center.x - knob_rad * 0.32f, knob_center.y + knob_rad * 0.00f);
                ImVec2 p1 = ImVec2(knob_center.x - knob_rad * 0.08f, knob_center.y + knob_rad * 0.24f);
                ImVec2 p2 = ImVec2(knob_center.x + knob_rad * 0.36f, knob_center.y - knob_rad * 0.22f);
                float th = knob_rad * 0.17f;

                // Shadow
                ImU32 sh = IM_COL32(0, 0, 0, (int)(90 * ck));
                dl->PathClear(); dl->PathLineTo(p0 + ImVec2(1, 1)); dl->PathLineTo(p1 + ImVec2(1, 1)); dl->PathLineTo(p2 + ImVec2(1, 1));
                dl->PathStroke(sh, 0, th);

                // Highlight
                ImU32 hi = IM_COL32(255, 255, 255, (int)(180 * ck));
                dl->PathClear(); dl->PathLineTo(p0 + ImVec2(-1, -1)); dl->PathLineTo(p1 + ImVec2(-1, -1)); dl->PathLineTo(p2 + ImVec2(-1, -1));
                dl->PathStroke(hi, 0, th * 0.85f);

                // Core
                ImU32 core = IM_COL32(245, 255, 245, (int)(230 * ck));
                dl->PathClear(); dl->PathLineTo(p0); dl->PathLineTo(p1); dl->PathLineTo(p2);
                dl->PathStroke(core, 0, th * 0.92f);
            }
        }

        // Inner ring highlight
        dl->AddCircle(knob_center, knob_rad * 0.70f, IM_COL32(255, 255, 255, 45), 64, 1.2f);

        return pressed; // true on state flip
    }


    struct BevelPanelResult { bool hovered = false, held = false, clicked = false; ImRect inner; };

    static ImU32 Mul(ImU32 c, float k) {
        ImVec4 v = ImGui::ColorConvertU32ToFloat4(c);
        v.x = ImClamp(v.x * k, 0.0f, 1.0f);
        v.y = ImClamp(v.y * k, 0.0f, 1.0f);
        v.z = ImClamp(v.z * k, 0.0f, 1.0f);
        return ImGui::GetColorU32(v);
    }


}
