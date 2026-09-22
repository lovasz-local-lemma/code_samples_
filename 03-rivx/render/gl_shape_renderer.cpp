// backend/src/gl_shape_renderer.cpp
// GL stencil-then-cover shape renderer: GlShapePipeline with an offscreen
// MSAA target and a solid convex-fan cover, resolved + composited into the
// destination framebuffer.
#include "render/gl_shape_renderer.hpp"
#include "render/gl_util.hpp"
#include "rive/renderer/gl/gles3.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace rive_backend
{
namespace
{
#include "render/gl_shaders.inl" // GLSL sources extracted (still in this anon namespace)
} // namespace

struct GlShapePipeline::Impl
{
    int w = 0, h = 0, samples = 4, wantSamples = 4;
    // The COLOUR-CHAIN format, per instance (the setSampleCount idiom). Three
    // allocations are welded to it and move TOGETHER: msColor (the accumulation target),
    // resolveTex (the MSAA-resolve blit requires read/draw format identity), and bgTex
    // (the beginScene background snapshot -- an 8-bit bgTex under a 16F destination would
    // re-quantize the whole accumulated scene every beginScene/endScene round-trip).
    // Coverage scratch (covTex/covTmpTex/atlas) stays RGBA8 DELIBERATELY: the composite
    // shaders read only .a, and the feather cache's glCopyTexImage2D tiles hard-require
    // the match. Default RGBA8 keeps every existing instance byte-identical.
    GLenum colorFormat = GL_RGBA8, wantColorFormat = GL_RGBA8;
    GLuint msFbo = 0, msColor = 0, msStencil = 0;
    GLuint resolveFbo = 0, resolveTex = 0;
    GLuint solidProg = 0, blitProg = 0, gradProg = 0, vao = 0, vbo = 0, emptyVao = 0;
    GLint uViewport = -1, uColor = -1, uTex = -1;
    // Textured-quad (image) program + its own interleaved pos+uv VAO/VBO, since
    // the shared vao/vbo only feeds position at location 0.
    GLuint imageProg = 0, imageVao = 0, imageVbo = 0;
    GLint uiViewport = -1, uiImage = -1, uiOpacity = -1;
    GLint ugViewport = -1, ugKind = -1, ugP0 = -1, ugP1 = -1, ugRadius = -1,
          ugOpacity = -1, ugStopCount = -1, ugStopOffset = -1, ugStopColor = -1,
          ugInvLin = -1, ugInvTrans = -1;
    GLuint covFbo = 0, covTex = 0, covStencil = 0, covTmpFbo = 0, covTmpTex = 0;
    GLuint bgTex = 0; // snapshot of the destination FBO so shapes can blend against the background
    // Advanced ("atlas") feather: downscaled blur targets + the active coverage
    // texture the composite samples (covTex full-res, or covAtlas).
    GLuint covAtlas = 0, covAtlasTmp = 0, covAtlasFbo = 0, covAtlasTmpFbo = 0;
    int atlasW = 0, atlasH = 0;
    int featherMode = 0; // 0 = gaussian, 1 = atlas, 2 = analytic (erf/SDF)
    bool boundedFeatherFlood = false; // opt-in bounded JFA start (not byte-exact)
    // Analytic feather: jump-flood SDF ping-pong targets (RGBA32F: seed coords +
    // valid) + the erf composite.
    GLuint jfaA = 0, jfaB = 0, jfaAFbo = 0, jfaBFbo = 0;
    GLuint seedProg = 0, jfaProg = 0, erfProg = 0;
    GLint uSeedCov = -1, uSeedInv = -1;
    GLint uJfaSeed = -1, uJfaInv = -1, uJfaStep = -1;
    GLint uErfSeed = -1, uErfCov = -1, uErfInv = -1, uErfRadius = -1,
          uErfHalfWidth = -1;
    GLuint activeCovTex = 0;
    rive::BlendMode blend = rive::BlendMode::srcOver;

    // --- Static-feather coverage cache -------------------------------------
    // The feather COVERAGE texture (activeCovTex) is paint-independent: it is a
    // pure function of {device geometry, fill rule, feather params, clip, feather
    // mode + bounded flag, viewport w/h}. Geometrically-static bakes (beam / box /
    // sweep) re-issue byte-identical device geometry every frame and only animate
    // paint (gradient stops / opacity) in the LATER composite, so their coverage
    // is identical every frame. Cache the box crop of covTmpTex (analytic) /
    // covTex (gaussian) keyed by a 64-bit content hash; on a later frame skip the
    // cover pass + JFA/erf storm and blit the cached texels back. Byte-identical
    // by construction (raw RGBA8->RGBA8 copy + the unchanged composite).
    struct Tiebreak
    {
        uint32_t totalGeomPointCount = 0;
        int x0 = 0, y0 = 0, bw = 0, bh = 0;
        float featherPx = 0.0f, halfWidth = 0.0f;
        uint8_t kind = 0, eo = 0, featherMode = 0, bounded = 0;
        uint32_t clipPointCount = 0;
        bool operator==(const Tiebreak& o) const
        {
            return totalGeomPointCount == o.totalGeomPointCount && x0 == o.x0 &&
                   y0 == o.y0 && bw == o.bw && bh == o.bh &&
                   featherPx == o.featherPx && halfWidth == o.halfWidth &&
                   kind == o.kind && eo == o.eo && featherMode == o.featherMode &&
                   bounded == o.bounded && clipPointCount == o.clipPointCount;
        }
    };
    struct Entry
    {
        GLuint tex = 0;
        int x0 = 0, y0 = 0, bw = 0, bh = 0;
        size_t bytes = 0;
        uint64_t lastFrame = 0;
        Tiebreak tb;
    };
    std::unordered_map<uint64_t, Entry> cache;
    size_t cacheBytes = 0;
    size_t budget = size_t(512) << 20;    // 512 MB default (configurable)
    const size_t skipCapPx = 2u * 1024 * 1024; // huge-leg skip cap (~2 Mpx)
    uint64_t currentFrame = 0;
    bool cacheEnabled = true;
    GLuint cacheReadFbo = 0;              // persistent FBO the tile attaches to
    // Per-frame stats (env-gated stderr report; never affects pixels/stdout).
    uint64_t frameHits = 0, frameMisses = 0, frameSkips = 0;
    // The PINNED-OUT counter: legs featherStore could not store because nothing
    // was evictable. Without it, a cache that has given up storing entirely still
    // reports a healthy-looking hit rate over the legs it does serve; a cache
    // that cannot store is the single most useful thing to know about a cache.
    uint64_t framePinnedOut = 0;
    // Saturation latch: has the pool ever been unable to store because the whole
    // working set was pinned? One-shot warning, so a saturated run says so once on
    // stderr instead of only under an env var nobody knows to set.
    bool saturatedWarned = false;
    bool budgetFromEnv = false; // an explicit budget is never second-guessed
    // Where the budget came from. Reported in the stats line because a
    // silently-inert autosize is exactly as invisible as the saturation it exists
    // to prevent -- "budget=512MB" alone cannot tell you whether the GPU was asked
    // and said 512, or was never asked at all.
    const char* budgetSource = "default";
    bool cacheEnvRead = false, statsEnv = false;
    void readCacheEnv()
    {
        if (cacheEnvRead) return;
        cacheEnvRead = true;
        if (const char* s = std::getenv("RIVE_FEATHER_CACHE_STATS"))
            statsEnv = (s[0] != '0' && s[0] != '\0');
        if (std::getenv("RIVE_FEATHER_CACHE_OFF")) cacheEnabled = false;
        if (const char* b = std::getenv("RIVE_FEATHER_CACHE_BUDGET_MB"))
        {
            const long mb = std::atol(b);
            if (mb > 0)
            {
                budget = size_t(mb) << 20;
                budgetFromEnv = true;
                budgetSource = "env";
            }
        }
        if (!budgetFromEnv) sizeBudgetToGpu();
    }

    // Size the pool to the GPU instead of to a guess.
    //
    // A fixed default fails as a cliff, not a gradient: a working set slightly
    // over budget (e.g. 606 MB against a 512 MB pool) evicts and recomputes
    // hundreds of tiles EVERY frame, each paying the full JFA storm the cache
    // exists to avoid -- measured as 85.3% hit / 285 ms at 512 MB versus
    // 100% hit / 200 ms at 1024 MB, a 1.42x lost to a default.
    //
    // THE SAFETY RULE, and it is the whole design: THIS CAN ONLY EVER RAISE THE
    // BUDGET, NEVER LOWER IT. A GPU whose free memory cannot be queried, or which
    // reports something small or absurd, lands on `budget` unchanged -- i.e.
    // the default behaviour. So the worst case on an unfamiliar GPU is the
    // status quo, and the query can be wrong without being harmful. That matters
    // because the failure this guards against (over-committing VRAM on a small
    // card, provoking driver-side thrash far worse than a cache miss) is invisible
    // on a development machine with plenty of VRAM.
    //
    // A QUARTER of FREE memory, hard-capped at 2 GB. A quarter because this pool is
    // one consumer among many -- the scene's own targets, the aux buffers, the
    // official runtime's atlases -- and a cache that wins its scene by starving the
    // renderer that draws it has not won anything.
    void sizeBudgetToGpu()
    {
        GLint freeKb = 0;
        const char* src = nullptr;
        // Vendor queries, both plain integer reads with no extension entry points.
        // Guarded by clearing the error state: a driver that does not know the enum
        // raises GL_INVALID_ENUM and leaves the destination untouched, which would
        // otherwise leak a spurious error into the next real call's glGetError.
        while (glGetError() != GL_NO_ERROR) {} // drain pre-existing errors
        constexpr GLenum kNvxCurrentAvailVidmem = 0x9049; // GL_NVX_gpu_memory_info
        constexpr GLenum kAtiTextureFreeMemory = 0x87FC;  // GL_ATI_meminfo (vec4, kb)
        glGetIntegerv(kNvxCurrentAvailVidmem, &freeKb);
        if (glGetError() == GL_NO_ERROR && freeKb > 0)
        {
            src = "gpu/nvx";
        }
        else
        {
            GLint ati[4] = {0, 0, 0, 0};
            freeKb = 0;
            glGetIntegerv(kAtiTextureFreeMemory, ati);
            if (glGetError() == GL_NO_ERROR && ati[0] > 0)
            {
                freeKb = ati[0]; // [0] = total free
                src = "gpu/ati";
            }
        }
        while (glGetError() != GL_NO_ERROR) {} // leave the error state as we found it
        if (src == nullptr || freeKb <= 0)
        {
            budgetSource = "default(no-gpu-query)";
            return; // unknown GPU -> keep the default
        }
        const size_t freeBytes = size_t(freeKb) * 1024u;
        size_t want = freeBytes / 4u;
        constexpr size_t kCap = size_t(2048) << 20;
        if (want > kCap) want = kCap;
        if (want > budget)
        {
            budget = want; // RAISE ONLY -- never below the default
            budgetSource = src;
        }
        else
        {
            // The GPU is small enough that a quarter of it is under our default.
            // Keep the default (raise-only), and SAY the query worked -- otherwise
            // this reads identically to a failed query.
            budgetSource = "default(gpu-smaller)";
        }
    }
    static void fnv(uint64_t& h, const void* data, size_t n)
    {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < n; ++i)
        {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    }
    void flushCache()
    {
        for (auto& kv : cache)
            if (kv.second.tex) glDeleteTextures(1, &kv.second.tex);
        cache.clear();
        cacheBytes = 0;
    }

    // Configure GL blending for the current blend mode (premultiplied src over
    // an opaque backdrop).
    void applyBlend()
    {
        glBlendEquation(GL_FUNC_ADD);
        switch (blend)
        {
            case rive::BlendMode::multiply:
                glBlendFunc(GL_DST_COLOR, GL_ONE_MINUS_SRC_ALPHA);
                break;
            case rive::BlendMode::screen:
                glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR);
                break;
            case rive::BlendMode::darken:
                glBlendEquation(GL_MIN);
                glBlendFunc(GL_ONE, GL_ONE);
                break;
            case rive::BlendMode::lighten:
                glBlendEquation(GL_MAX);
                glBlendFunc(GL_ONE, GL_ONE);
                break;
            default: // srcOver + unsupported modes
                glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
                break;
        }
    }

    std::vector<ClipLayer> clip; // active clip layers (intersection)
    // Blur-clip: the POST-FEATHER intersective window. `clip` above gates the COVERAGE
    // (clip-then-feather -- the blur escapes past the cut); this gates the COMPOSITE of the
    // already-blurred coverage (feather-then-clip -- the cut survives as a sharp edge). See
    // GlShapePipeline::setBlurClip. Empty (the default) = every composite runs with ZERO
    // added GL calls, which keeps every other consumer of this pipeline bit-identical.
    std::vector<Contour> blurClip;
    bool blurClipEvenOdd = false;

    void scissorBbox(Pt mn, Pt mx)
    {
        glScissor(static_cast<int>(std::floor(mn.x)),
                  static_cast<int>(std::floor(h - mx.y)),
                  static_cast<int>(std::ceil(mx.x - mn.x)) + 1,
                  static_cast<int>(std::ceil(mx.y - mn.y)) + 1);
    }

    // Padded device-space bbox for the per-shape feather passes. The feather
    // pipeline (coverage clear, JFA/blur, composite) otherwise runs full-viewport
    // for every shape; scissoring to this box makes the cost scale with the shape
    // size instead of the screen size (the big gsplat win). Pad covers the soft
    // skirt so the edge falloff isn't clipped.
    Pt fbMn{0.0f, 0.0f}, fbMx{0.0f, 0.0f};
    void setFeatherBox(Pt mn, Pt mx, float featherPx)
    {
        const float pad = std::max(featherPx * 1.5f, 0.5f) * 1.5f + 4.0f;
        fbMn = {std::clamp(mn.x - pad, 0.0f, float(w)),
                std::clamp(mn.y - pad, 0.0f, float(h))};
        fbMx = {std::clamp(mx.x + pad, 0.0f, float(w)),
                std::clamp(mx.y + pad, 0.0f, float(h))};
        if (fbMx.x < fbMn.x) fbMx.x = fbMn.x; // never a negative-width scissor
        if (fbMx.y < fbMn.y) fbMx.y = fbMn.y;
    }
    void featherScissorOn() { glEnable(GL_SCISSOR_TEST); scissorBbox(fbMn, fbMx); }

    // Restore the invariant "covTex is all zeros outside any active feather box".
    // The feather passes READ covTex beyond the write scissor (the JFA seed taps
    // +-1 texel past it, the atlas path mip-downsamples the WHOLE texture, the
    // gaussian taps ~3 sigma out), so stale white coverage left behind by a
    // previous feathered draw would manufacture phantom edges along the next
    // shape's box rim -- faint rectangular borders on feather+texture scenes, in
    // every feather mode. Clearing our own box after compositing keeps the
    // texture all-zero by induction.
    void clearFeatherScratch()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, covFbo);
        glDisable(GL_BLEND);
        featherScissorOn();
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        glBindFramebuffer(GL_FRAMEBUFFER, msFbo); // back to post-composite state
        glEnable(GL_BLEND);
    }

    // Build the clip mask into stencil bit 7 over the bbox: bit7 = 1 where inside
    // ALL active clip layers (intersection; non-convex via winding). Assumes
    // solidProg active, viewport set, vao/vbo bound. Leaves bits 0-6 = 0.
    void buildClipMask(Pt mn, Pt mx)
    {
        // Empty-clip fast path: with no active clip layer, setting bit7=1 across
        // the ENTIRE bbox would accomplish nothing but a scissored stencil clear +
        // a STREAM quad upload + a full-bbox draw, per shape. Skip it; the
        // winding/union/image cover gates fall back to clipGate()'s GL_ALWAYS,
        // which passes for exactly the same fragments (a subset of that bbox), and
        // the winding-only cover tests never read bit7. Relies on the between-
        // draws all-zero-stencil invariant (beginScene clears msFbo stencil; every
        // draw resets its own bbox with mask 0xFF; every write is bbox-confined;
        // feather passes pre-clear covFbo over the feather box).
        if (clip.empty()) return;
        const Pt quad[6] = {{mn.x, mn.y}, {mx.x, mn.y}, {mx.x, mx.y},
                            {mn.x, mn.y}, {mx.x, mx.y}, {mn.x, mx.y}};
        glEnable(GL_STENCIL_TEST);
        glDisable(GL_CULL_FACE);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        // Clear the bbox stencil, then set bit7 = 1 across the whole bbox.
        glEnable(GL_SCISSOR_TEST);
        scissorBbox(mn, mx);
        glStencilMask(0xFF);
        glClear(GL_STENCIL_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        glStencilMask(0x80);
        glStencilFunc(GL_ALWAYS, 0x80, 0x80);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        // Intersect: clear bit7 wherever a layer does NOT cover.
        for (const ClipLayer& layer : clip)
        {
            glStencilMask(0x7F);
            glStencilFunc(GL_ALWAYS, 0, 0x7F);
            if (layer.evenOdd)
                glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);
            else
            {
                glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_INCR_WRAP);
                glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_DECR_WRAP);
            }
            // SCISSOR THE WINDING FANS TO THE BBOX. A clip contour can be unbounded
            // geometry -- a caller realising "paint outside the occluder union"
            // with an ENCLOSING RECTANGLE of +-1e4 margin has a fan covering the
            // WHOLE framebuffer. Unscissored, those winding bits would land
            // everywhere while the per-layer cleanup below only clears the bbox --
            // leaving winding bits = 1 across the rest of the screen and BREAKING
            // the "all-zero stencil between draws" invariant that the empty-clip
            // fast path above relies on. The very next UNCLIPPED fill would take
            // that fast path (no bbox clear) and its cover pass GL_NOTEQUAL
            // 0x00,0x7F would pass over its ENTIRE bounding quad -- the shape
            // painted as an axis-aligned RECTANGLE instead of its silhouette.
            // Semantically inert INSIDE the bbox: bit7 is written only by the two
            // bbox-sized quads above/below, so the clip intersection over every
            // fragment any later pass can rasterize is bit-identical.
            glEnable(GL_SCISSOR_TEST);
            scissorBbox(mn, mx);
            for (const Contour& c : layer.contours)
            {
                if (c.points.size() < 3) continue;
                glBufferData(GL_ARRAY_BUFFER,
                             GLsizeiptr(c.points.size() * sizeof(Pt)),
                             c.points.data(), GL_STREAM_DRAW);
                glDrawArrays(GL_TRIANGLE_FAN, 0, GLsizei(c.points.size()));
            }
            glDisable(GL_SCISSOR_TEST);
            glStencilMask(0x80);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
            if (layer.evenOdd)
                glStencilFunc(GL_EQUAL, 0x00, 0x01); // outside = even
            else
                glStencilFunc(GL_EQUAL, 0x00, 0x7F); // outside = winding 0
            glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STREAM_DRAW);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            // Clear winding bits before the next layer.
            glEnable(GL_SCISSOR_TEST);
            scissorBbox(mn, mx);
            glStencilMask(0x7F);
            glClear(GL_STENCIL_BUFFER_BIT);
            glDisable(GL_SCISSOR_TEST);
        }
    }

    // Stencil gate for the winding / stroke-union / image cover passes: EQUAL on
    // the clip bit when buildClipMask built one, else ALWAYS (the empty-clip fast
    // path skips buildClipMask, so bit7 is never written). Equivalent to a
    // full-bbox bit7 because that bbox is a superset of every fragment these
    // passes rasterize -- GL_EQUAL,0x80,0x80 and GL_ALWAYS pass for exactly the
    // same fragments; stencilMask/stencilOp are unchanged so no sfail op fires in
    // either case. Also disables face culling for the winding fans (GL_CULL_FACE
    // is only ever DISABLED anywhere in the backend, so this is a no-op that
    // removes a hidden dependency on buildClipMask having done it).
    void clipGate()
    {
        if (clip.empty())
        {
            glDisable(GL_CULL_FACE);
            glStencilFunc(GL_ALWAYS, 0x80, 0x80);
        }
        else
            glStencilFunc(GL_EQUAL, 0x80, 0x80);
    }

    // Accumulate shape winding into bits 0-6, gated to the clip mask (bit7).
    void windingPass(const std::vector<Contour>& contours, bool evenOdd)
    {
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glEnable(GL_STENCIL_TEST);
        glStencilMask(0x7F);
        clipGate(); // gate winding to the clip mask (ALWAYS when unclipped)
        if (evenOdd)
            glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);
        else
        {
            glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_INCR_WRAP);
            glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_DECR_WRAP);
        }
        for (const Contour& c : contours)
        {
            if (c.points.size() < 3) continue;
            glBufferData(GL_ARRAY_BUFFER,
                         GLsizeiptr(c.points.size() * sizeof(Pt)),
                         c.points.data(), GL_STREAM_DRAW);
            glDrawArrays(GL_TRIANGLE_FAN, 0, GLsizei(c.points.size()));
        }
    }

    // Configure the cover-pass stencil test (winding bits only).
    void coverStencil(bool evenOdd)
    {
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glStencilMask(0x00);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        if (evenOdd)
            glStencilFunc(GL_EQUAL, 0x01, 0x01);
        else
            glStencilFunc(GL_NOTEQUAL, 0x00, 0x7F);
    }

    void resetStencil(Pt mn, Pt mx)
    {
        glEnable(GL_SCISSOR_TEST);
        scissorBbox(mn, mx);
        glStencilMask(0xFF);
        glClear(GL_STENCIL_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_STENCIL_TEST);
    }

    // --- Feather coverage helpers (shared by feathered fills + strokes) -------
    // Render a fill's white coverage into covFbo via clip-gated stencil-then-
    // cover.
    void coverFill(const std::vector<Contour>& contours, bool eo, Pt mn, Pt mx)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, covFbo);
        glViewport(0, 0, w, h);
        glDisable(GL_BLEND);
        glClearColor(0, 0, 0, 0);
        featherScissorOn(); // clear only the padded feather box, not the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        glUseProgram(solidProg);
        glUniform2f(uViewport, float(w), float(h));
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Pt), (void*)0);
        buildClipMask(mn, mx);
        windingPass(contours, eo);
        glDisable(GL_BLEND);
        glUniform4f(uColor, 1.0f, 1.0f, 1.0f, 1.0f);
        coverStencil(eo);
        const Pt quad[6] = {{mn.x, mn.y}, {mx.x, mn.y}, {mx.x, mx.y},
                            {mn.x, mn.y}, {mx.x, mx.y}, {mn.x, mx.y}};
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        resetStencil(mn, mx);
        glBindVertexArray(0);
    }

    // Render a stroke's white coverage into covFbo: union the ribbon triangles
    // into the stencil (gated to the clip bit), then cover once -- no double
    // counting on self-overlapping strokes.
    void coverStroke(const std::vector<Pt>& tris, Pt mn, Pt mx)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, covFbo);
        glViewport(0, 0, w, h);
        glDisable(GL_BLEND);
        glClearColor(0, 0, 0, 0);
        featherScissorOn(); // clear only the padded feather box
        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        glUseProgram(solidProg);
        glUniform2f(uViewport, float(w), float(h));
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Pt), (void*)0);
        buildClipMask(mn, mx);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glEnable(GL_STENCIL_TEST);
        glStencilMask(0x7F);
        clipGate(); // gate the stroke union to the clip mask (ALWAYS unclipped)
        glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);
        glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(tris.size() * sizeof(Pt)),
                     tris.data(), GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, GLsizei(tris.size()));
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glDisable(GL_BLEND);
        glUniform4f(uColor, 1.0f, 1.0f, 1.0f, 1.0f);
        glStencilMask(0x00);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        glStencilFunc(GL_NOTEQUAL, 0x00, 0x7F);
        const Pt quad[6] = {{mn.x, mn.y}, {mx.x, mn.y}, {mx.x, mx.y},
                            {mn.x, mn.y}, {mx.x, mx.y}, {mn.x, mx.y}};
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STREAM_DRAW);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        resetStencil(mn, mx);
        glBindVertexArray(0);
    }

    // Separable Gaussian blur of the coverage: covTex -> covTmpTex -> covTex.
    void blurCov(float sigma)
    {
        glDisable(GL_BLEND);
        // Stale-scratch ring: the V pass below samples covTmpTex up to the kernel reach
        // (3*sigma) BEYOND the feather box, but the H pass only WRITES covTmpTex inside
        // the box (featherScissorOn) -- so those out-of-box taps would read whatever the
        // PREVIOUS feathered draw's H pass left there (or undefined alloc contents on the
        // first draw). Stale white rows just past the box rim smear back in and then CUT
        // at the write scissor: a straight bright line hugging the feather-box top/bottom
        // rim, inside flat faces. This is the same phantom-edge class clearFeatherScratch's
        // invariant guards against, but that invariant covers covTex ONLY; covTmpTex
        // needs its own ring guarantee in gaussian mode. featherAnalytic already
        // ring-clears its own scratch ("box + maxStep margin"); this is the gaussian twin.
        // Clear the box + kernel-reach ring so every out-of-box tap reads a genuine zero
        // (the coverage out there IS zero: covTex's own invariant). Byte-identical
        // whenever the ring was already zero -- i.e. whenever the artifact was absent.
        {
            const float reach = 3.0f * std::max(sigma, 0.5f) + 2.0f; // kernel + linear-tap slack
            const Pt rMn{std::clamp(fbMn.x - reach, 0.0f, float(w)),
                         std::clamp(fbMn.y - reach, 0.0f, float(h))};
            const Pt rMx{std::clamp(fbMx.x + reach, 0.0f, float(w)),
                         std::clamp(fbMx.y + reach, 0.0f, float(h))};
            glBindFramebuffer(GL_FRAMEBUFFER, covTmpFbo);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glEnable(GL_SCISSOR_TEST);
            scissorBbox(rMn, rMx);
            glClear(GL_COLOR_BUFFER_BIT);
            glDisable(GL_SCISSOR_TEST);
        }
        glUseProgram(featherBlurProg);
        glUniform1i(ufbTex, 0);
        glUniform2f(ufbInvSize, 1.0f / w, 1.0f / h);
        glUniform1f(ufbSigma, sigma);
        glActiveTexture(GL_TEXTURE0);
        glBindVertexArray(emptyVao);
        glViewport(0, 0, w, h);
        featherScissorOn(); // only blur the padded feather box
        glBindFramebuffer(GL_FRAMEBUFFER, covTmpFbo);
        glBindTexture(GL_TEXTURE_2D, covTex);
        glUniform2f(ufbDir, 1.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindFramebuffer(GL_FRAMEBUFFER, covFbo);
        glBindTexture(GL_TEXTURE_2D, covTmpTex);
        glUniform2f(ufbDir, 0.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisable(GL_SCISSOR_TEST);
        glBindVertexArray(0);
        activeCovTex = covTex;
    }

    // Advanced feather: downscale the coverage into a small atlas (scale =
    // 16/max(featherRadius,16)), Gaussian-blur there with a bounded radius so it
    // stays well-sampled for any feather size (like Rive's atlas), then let the
    // composite bilinearly upsample it -- "gaussian -> linear -> gaussian".
    void blurCovAtlas(float featherPx)
    {
        const float radius = std::max(featherPx * 1.5f, 1.0f);
        float s = 16.0f / std::max(radius, 16.0f);
        if (s > 1.0f) s = 1.0f;
        if (s < 0.0625f) s = 0.0625f;
        const int aw = std::max(16, static_cast<int>(float(w) * s + 0.5f));
        const int ah = std::max(16, static_cast<int>(float(h) * s + 0.5f));
        if (aw != atlasW || ah != atlasH)
        {
            atlasW = aw;
            atlasH = ah;
            const GLuint texs[2] = {covAtlas, covAtlasTmp};
            for (GLuint t : texs)
            {
                glBindTexture(GL_TEXTURE_2D, t);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, aw, ah, 0, GL_RGBA,
                             GL_UNSIGNED_BYTE, nullptr);
            }
        }
        // 1. Downsample covTex into covAtlas. Refresh mips first so the blit
        // area-averages (full-screen tri into a small viewport auto-selects the
        // matching LOD) instead of point-sampling.
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, covTex);
        // Enable mip filtering + refresh mips (only the atlas path needs them;
        // the full-res path samples LOD 0 regardless).
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_LINEAR);
        glGenerateMipmap(GL_TEXTURE_2D);
        glDisable(GL_BLEND);
        glBindVertexArray(emptyVao);
        glBindFramebuffer(GL_FRAMEBUFFER, covAtlasFbo);
        glViewport(0, 0, aw, ah);
        glUseProgram(blitProg);
        glUniform1i(uTex, 0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        // 2. Separable Gaussian at atlas resolution (bounded sigma -> exact).
        const float sigmaA = std::max(featherPx * 0.5f * s, 0.5f);
        glUseProgram(featherBlurProg);
        glUniform1i(ufbTex, 0);
        glUniform2f(ufbInvSize, 1.0f / float(aw), 1.0f / float(ah));
        glUniform1f(ufbSigma, sigmaA);
        glBindFramebuffer(GL_FRAMEBUFFER, covAtlasTmpFbo);
        glViewport(0, 0, aw, ah);
        glBindTexture(GL_TEXTURE_2D, covAtlas);
        glUniform2f(ufbDir, 1.0f, 0.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindFramebuffer(GL_FRAMEBUFFER, covAtlasFbo);
        glBindTexture(GL_TEXTURE_2D, covAtlasTmp);
        glUniform2f(ufbDir, 0.0f, 1.0f);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);
        activeCovTex = covAtlas; // composite upsamples this (GL_LINEAR)
    }

    // Analytic feather: jump-flood the coverage's 0.5 isoline into a signed
    // distance field, then map distance through erf -> coverage in covTmpTex.
    // This is the genuine Rive feather (coverage = erf of edge distance), not a
    // blur of the mask, so it stays crisp + artifact-free at any radius.
    void featherAnalytic(float featherPx, float halfWidth)
    {
        const float featherRadius = std::max(featherPx * 1.5f, 0.5f);
        glDisable(GL_BLEND);
        glDisable(GL_STENCIL_TEST);
        glBindVertexArray(emptyVao);
        glViewport(0, 0, w, h);
        // The JFA step samples up to maxStep texels outside the feather box; if
        // that lands on a PREVIOUS shape's stale seeds it bleeds a faint
        // rectangular border in. Clear both ping-pong targets to "invalid" (z=0)
        // over a box + maxStep margin so those out-of-box reads return no-seed.
        const int boxW = std::max(1, int(std::ceil(fbMx.x - fbMn.x)));
        const int boxH = std::max(1, int(std::ceil(fbMx.y - fbMn.y)));
        const float margin = float(std::max(boxW, boxH) / 2 + 2);
        const Pt mMn{std::clamp(fbMn.x - margin, 0.0f, float(w)),
                     std::clamp(fbMn.y - margin, 0.0f, float(h))};
        const Pt mMx{std::clamp(fbMx.x + margin, 0.0f, float(w)),
                     std::clamp(fbMx.y + margin, 0.0f, float(h))};
        glDisable(GL_BLEND);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glEnable(GL_SCISSOR_TEST);
        scissorBbox(mMn, mMx);
        glBindFramebuffer(GL_FRAMEBUFFER, jfaAFbo);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, jfaBFbo);
        glClear(GL_COLOR_BUFFER_BIT);
        // Also clear the erf-output texture (what compositeSolid samples with
        // GL_LINEAR): otherwise, at the feather-box edge the composite reads a
        // PREVIOUS shape's stale coverage just outside the box -> a faint
        // rectangular outline, visible over a bright (textured) backdrop. The erf
        // below overwrites the tight box; this leaves a clean 0 ring around it.
        glBindFramebuffer(GL_FRAMEBUFFER, covTmpFbo);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);

        featherScissorOn(); // seed/JFA/erf passes only touch the feather box
        glActiveTexture(GL_TEXTURE0);
        // Sample covTex at LOD 0 (a prior atlas feather may have left a mipmap
        // min filter on it).
        glBindTexture(GL_TEXTURE_2D, covTex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        // 1. Seed boundary crossings from covTex into jfaA.
        glUseProgram(seedProg);
        glBindFramebuffer(GL_FRAMEBUFFER, jfaAFbo);
        glUniform1i(uSeedCov, 0);
        glUniform2f(uSeedInv, 1.0f / float(w), 1.0f / float(h));
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // 2. Jump-flood: ping-pong with a halving step.
        glUseProgram(jfaProg);
        glUniform1i(uJfaSeed, 0);
        glUniform2f(uJfaInv, 1.0f / float(w), 1.0f / float(h));
        GLuint readTex = jfaA, writeTex = jfaB, writeFbo = jfaBFbo;
        // Flood only as far as the feather box, not the whole screen: a small
        // splat needs ~log2(box) steps, not ~log2(screen).
        //
        // Bounded flood (opt-in via setBoundedFeatherFlood; DEFAULT OFF so the
        // shipped render is byte-for-byte the full flood). The erf output is
        // already saturated (exactly 0 exterior / 1 interior) beyond
        // ~featherRadius from the edge, so a pixel's coverage byte can only depend
        // on its nearest-seed distance when that distance is in-band. Halving the
        // full start (max(boxW,boxH)/2 >> k) down to the smallest value still
        // above 3*featherRadius yields a step sequence that is a bit-identical
        // SUFFIX of the baseline schedule, converging most in-band pixels to the
        // same seed and dropping the coarse (erf-saturated) passes -- a large win
        // on big-box / small-feather shapes (thin feathered beams). It is NOT
        // fully byte-exact: at medial-axis pixels of thin features two seeds tie
        // and the shortened schedule can pick the other one, flipping a few erf
        // bytes at the steep midpoint (on the order of a few pixels per shape).
        // Hence off by default; when off, floodStart == fullStart == the full
        // schedule exactly.
        int floodStart = std::max(boxW, boxH) / 2;
        if (boundedFeatherFlood)
        {
            const float bandPx = 3.0f * featherRadius;
            while ((floodStart >> 1) >= 1 && float(floodStart >> 1) >= bandPx)
                floodStart >>= 1;
        }
        for (int step = floodStart; step >= 1; step >>= 1)
        {
            glUniform1f(uJfaStep, static_cast<float>(step));
            glBindFramebuffer(GL_FRAMEBUFFER, writeFbo);
            glBindTexture(GL_TEXTURE_2D, readTex);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            std::swap(readTex, writeTex);
            writeFbo = (writeTex == jfaA) ? jfaAFbo : jfaBFbo;
        }
        // readTex holds the final JFA result.

        // 3. Erf composite: signed distance -> coverage in covTmpTex.
        glUseProgram(erfProg);
        glBindFramebuffer(GL_FRAMEBUFFER, covTmpFbo);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, readTex);
        glUniform1i(uErfSeed, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, covTex);
        glUniform1i(uErfCov, 1);
        glUniform2f(uErfInv, 1.0f / float(w), 1.0f / float(h));
        glUniform1f(uErfRadius, featherRadius);
        glUniform1f(uErfHalfWidth, halfWidth);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glDisable(GL_SCISSOR_TEST);
        glActiveTexture(GL_TEXTURE0);
        glBindVertexArray(0);
        glUseProgram(0);
        activeCovTex = covTmpTex;
    }

    // Apply the selected feather mode to the coverage currently in covTex.
    // halfWidth is the band half-width for analytic strokes (huge for fills).
    void applyFeather(float featherPx, float halfWidth)
    {
        switch (featherMode)
        {
            case 2: featherAnalytic(featherPx, halfWidth); break;
            case 1: blurCovAtlas(featherPx); break;
            default: blurCov(std::max(featherPx * 0.5f, 0.5f)); break;
        }
    }

    // The tight integer feather-box rect (GL bottom-left origin). Matches
    // scissorBbox's floor/ceil+1 EXACTLY (so cached texels land at identical
    // device coords) then clamps to [0,w]x[0,h] so the GPU copy/blit stays in
    // range. Computed identically on store and hit for a given shape.
    void tightBox(int& x0, int& y0, int& bw, int& bh) const
    {
        x0 = static_cast<int>(std::floor(fbMn.x));
        y0 = static_cast<int>(std::floor(h - fbMx.y));
        bw = static_cast<int>(std::ceil(fbMx.x - fbMn.x)) + 1;
        bh = static_cast<int>(std::ceil(fbMx.y - fbMn.y)) + 1;
        if (x0 < 0) x0 = 0;
        if (y0 < 0) y0 = 0;
        if (x0 > w) x0 = w;
        if (y0 > h) y0 = h;
        if (bw < 0) bw = 0;
        if (bh < 0) bh = 0;
        if (x0 + bw > w) bw = w - x0;
        if (y0 + bh > h) bh = h - y0;
    }

    // Produce activeCovTex for a feathered fill/stroke, serving a cached tile on
    // a content-hash hit (skip the cover pass + JFA/erf storm) or computing +
    // storing on a miss. Sets the feather box from the raw device bounds mn,mx.
    // kind: 0 fill (contours+eo), 1 stroke (tris). Paint is applied LATER by the
    // caller's compositeSolid/compositeGradient, so the same tile serves solid
    // and gradient variants of identical geometry.
    void ensureFeatherCoverage(int kind, const std::vector<Contour>* contours,
                               bool eo, const std::vector<Pt>* tris,
                               float featherPx, float halfWidth, Pt mn, Pt mx)
    {
        readCacheEnv();
        setFeatherBox(mn, mx, featherPx); // 0. clip all passes to the shape
        int x0, y0, bw, bh;
        tightBox(x0, y0, bw, bh);
        const size_t area = size_t(bw) * size_t(bh);
        // GATE: atlas mode (downsampled activeCovTex -> redirect breaks
        // byte-identity), caching disabled, degenerate/oversized boxes -> run the
        // plain per-frame path with no cache.
        if (featherMode == 1 || !cacheEnabled || bw < 1 || bh < 1 ||
            area > skipCapPx)
        {
            if (kind == 0) coverFill(*contours, eo, mn, mx);
            else coverStroke(*tris, mn, mx);
            applyFeather(featherPx, halfWidth);
            ++frameSkips;
            return;
        }

        // --- Content hash (paint EXCLUDED) -------------------------------------
        uint64_t key = 1469598103934665603ULL;
        const uint8_t hk = static_cast<uint8_t>(kind);
        const uint8_t heo = static_cast<uint8_t>(kind == 0 ? (eo ? 1 : 0) : 0);
        const uint8_t hfm = static_cast<uint8_t>(featherMode);
        const uint8_t hbf = static_cast<uint8_t>(boundedFeatherFlood ? 1 : 0);
        fnv(key, &hk, 1);
        fnv(key, &heo, 1);
        fnv(key, &hfm, 1);
        fnv(key, &hbf, 1);
        fnv(key, &w, sizeof(int));
        fnv(key, &h, sizeof(int));
        fnv(key, &featherPx, sizeof(float));
        fnv(key, &halfWidth, sizeof(float));
        uint32_t totalGeomPoints = 0;
        if (kind == 0)
        {
            for (const Contour& c : *contours)
            {
                const uint32_t n = static_cast<uint32_t>(c.points.size());
                fnv(key, &n, sizeof(uint32_t));
                if (n) fnv(key, c.points.data(), size_t(n) * sizeof(Pt));
                totalGeomPoints += n;
            }
        }
        else
        {
            const uint32_t n = static_cast<uint32_t>(tris->size());
            fnv(key, &n, sizeof(uint32_t));
            if (n) fnv(key, tris->data(), size_t(n) * sizeof(Pt));
            totalGeomPoints = n;
        }
        uint32_t clipPoints = 0;
        const uint32_t nlayers = static_cast<uint32_t>(clip.size());
        fnv(key, &nlayers, sizeof(uint32_t));
        for (const ClipLayer& layer : clip)
        {
            const uint8_t leo = static_cast<uint8_t>(layer.evenOdd ? 1 : 0);
            fnv(key, &leo, 1);
            const uint32_t nc = static_cast<uint32_t>(layer.contours.size());
            fnv(key, &nc, sizeof(uint32_t));
            for (const Contour& c : layer.contours)
            {
                const uint32_t n = static_cast<uint32_t>(c.points.size());
                fnv(key, &n, sizeof(uint32_t));
                if (n) fnv(key, c.points.data(), size_t(n) * sizeof(Pt));
                clipPoints += n;
            }
        }

        Tiebreak tb;
        tb.totalGeomPointCount = totalGeomPoints;
        tb.x0 = x0; tb.y0 = y0; tb.bw = bw; tb.bh = bh;
        tb.featherPx = featherPx; tb.halfWidth = halfWidth;
        tb.kind = hk; tb.eo = heo; tb.featherMode = hfm; tb.bounded = hbf;
        tb.clipPointCount = clipPoints;

        auto it = cache.find(key);
        if (it != cache.end() && it->second.tb == tb)
        {
            // HIT: reproduce covTmpTex's clean 0-ring + blit the cached tile back
            // into the tight box, skipping the entire cover + feather compute.
            featherHitReplay(it->second);
            it->second.lastFrame = currentFrame;
            ++frameHits;
            return;
        }
        if (it != cache.end())
        {
            // Key collision with a structurally different leg (astronomically
            // rare): drop the stale tile and recompute -- never serve mismatched
            // texels.
            if (it->second.tex) glDeleteTextures(1, &it->second.tex);
            cacheBytes -= it->second.bytes;
            cache.erase(it);
        }

        // MISS: compute the coverage the normal way, then store its box crop.
        if (kind == 0) coverFill(*contours, eo, mn, mx);
        else coverStroke(*tris, mn, mx);
        applyFeather(featherPx, halfWidth);
        ++frameMisses;
        featherStore(key, tb, x0, y0, bw, bh);
    }

    // Copy the freshly-computed feather box out of the active coverage FBO into a
    // fresh pool tile, under the LRU budget + in-frame pin.
    void featherStore(uint64_t key, const Tiebreak& tb, int x0, int y0, int bw,
                      int bh)
    {
        const size_t need = size_t(bw) * size_t(bh) * 4;
        // Evict least-recently-used entries not touched THIS frame (in-frame pin
        // stops a full-scene redraw from self-thrashing). If nothing is evictable
        // the whole working set is pinned -> skip caching this leg (compute-only).
        while (cacheBytes + need > budget)
        {
            uint64_t victimKey = 0;
            uint64_t victimFrame = UINT64_MAX;
            bool found = false;
            for (const auto& kv : cache)
            {
                if (kv.second.lastFrame < currentFrame &&
                    kv.second.lastFrame < victimFrame)
                {
                    victimFrame = kv.second.lastFrame;
                    victimKey = kv.first;
                    found = true;
                }
            }
            if (!found)
            {
                // Fully pinned -- the whole working set is in use this frame and
                // nothing can be evicted, so this leg goes uncached and will
                // recompute next frame too. THIS IS THE SATURATION SIGNAL and must
                // be counted explicitly: the hit rate alone stays plausible because
                // it is computed only over legs the cache still serves.
                ++framePinnedOut;
                if (!saturatedWarned)
                {
                    saturatedWarned = true;
                    std::fprintf(
                        stderr,
                        "[feather-cache] SATURATED: working set exceeds the %zu MB "
                        "budget -- tiles are being recomputed every frame. Raise it "
                        "with RIVE_FEATHER_CACHE_BUDGET_MB=<mb> (set "
                        "RIVE_FEATHER_CACHE_STATS=1 for per-frame detail).\n",
                        budget >> 20);
                }
                return; // fully pinned; do not store this leg
            }
            auto vit = cache.find(victimKey);
            if (vit->second.tex) glDeleteTextures(1, &vit->second.tex);
            cacheBytes -= vit->second.bytes;
            cache.erase(vit);
        }
        // srcFbo backs activeCovTex: gaussian sets activeCovTex=covTex (covFbo),
        // analytic sets activeCovTex=covTmpTex (covTmpFbo).
        const GLuint srcFbo = (featherMode == 0) ? covFbo : covTmpFbo;
        GLuint tile = 0;
        glGenTextures(1, &tile);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tile);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, srcFbo);
        // Raw RGBA8->RGBA8 texel copy of the tight box (same primitive as the
        // beginScene bgTex snapshot) -- no filtering, no conversion.
        glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, x0, y0, bw, bh, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_2D, 0);
        Entry e;
        e.tex = tile;
        e.x0 = x0; e.y0 = y0; e.bw = bw; e.bh = bh;
        e.bytes = need;
        e.lastFrame = currentFrame;
        e.tb = tb;
        cache.emplace(key, e);
        cacheBytes += need;
    }

    // Reconstruct covTmpTex over the shape's box+margin exactly as featherAnalytic
    // leaves it, from a cached tile: clear the box+margin to 0 (the clean ring the
    // composite's rim taps) then 1:1 nearest-blit the tile into the tight box.
    void featherHitReplay(const Entry& e)
    {
        glDisable(GL_BLEND);
        glDisable(GL_STENCIL_TEST);
        // 0-ring: mirror featherAnalytic's covTmpFbo clear over box + margin
        // (the same formula, so the cleared region is identical).
        const int boxW = std::max(1, int(std::ceil(fbMx.x - fbMn.x)));
        const int boxH = std::max(1, int(std::ceil(fbMx.y - fbMn.y)));
        const float margin = float(std::max(boxW, boxH) / 2 + 2);
        const Pt mMn{std::clamp(fbMn.x - margin, 0.0f, float(w)),
                     std::clamp(fbMn.y - margin, 0.0f, float(h))};
        const Pt mMx{std::clamp(fbMx.x + margin, 0.0f, float(w)),
                     std::clamp(fbMx.y + margin, 0.0f, float(h))};
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glEnable(GL_SCISSOR_TEST);
        scissorBbox(mMn, mMx);
        glBindFramebuffer(GL_FRAMEBUFFER, covTmpFbo);
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        // Blit the tile back into the tight box (1:1, GL_NEAREST, RGBA8->RGBA8 ->
        // bit-identical). cacheReadFbo is a persistent read-source FBO.
        if (!cacheReadFbo) glGenFramebuffers(1, &cacheReadFbo);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, cacheReadFbo);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, e.tex, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, covTmpFbo);
        glBlitFramebuffer(0, 0, e.bw, e.bh, e.x0, e.y0, e.x0 + e.bw, e.y0 + e.bh,
                          GL_COLOR_BUFFER_BIT, GL_NEAREST);
        // Leave exactly the state featherAnalytic leaves so the caller's
        // compositeSolid/compositeGradient + clearFeatherScratch run identically.
        glBindFramebuffer(GL_FRAMEBUFFER, covTmpFbo);
        glActiveTexture(GL_TEXTURE0);
        glBindVertexArray(0);
        glUseProgram(0);
        activeCovTex = covTmpTex;
    }

    // --- Blur-clip: the POST-FEATHER intersective gate ---------------------------
    // Install a stencil gate in msFbo so the composite of the ALREADY-BLURRED coverage paints
    // ONLY inside `blurClip`. This is the feather-then-clip ordering: coverFill + blurCov
    // have already run, so the blur is complete; cutting it here leaves a geometrically
    // sharp window edge with the softening confined inside.
    //
    // Deliberately SELF-CONTAINED rather than reusing buildClipMask: that one is written for the
    // `clip` layers and leaves winding bits DIRTY OUTSIDE its bbox (its end-of-layer clear is
    // scissored to the bbox while its contour fans are not). That is survivable in covFbo, whose
    // stencil is re-cleared per coverFill, but msFbo's stencil is scene-wide and only cleared at
    // beginScene -- dirty bits there would corrupt every LATER draw's winding. So EVERY write
    // here is scissored to the feather box, and endBlurClipGate clears exactly that box, keeping
    // the pipeline's all-zero-between-draws invariant intact.
    //
    // Caller contract: msFbo bound, viewport set, the feather box (fbMn/fbMx) current.
    // Returns true when a gate was installed (the caller MUST then call endBlurClipGate).
    // Leaves: GL_STENCIL_TEST on + gated to inside the window, stencil writes masked off,
    // colour mask restored, scissor ON at the feather box (exactly what featherScissorOn would
    // set), blend untouched by us (the caller sets it after).
    bool beginBlurClipGate()
    {
        if (blurClip.empty()) return false; // no clip -> not one added GL call (byte-identical)
        glEnable(GL_SCISSOR_TEST); // confine EVERY stencil write below to the feather box
        scissorBbox(fbMn, fbMx);
        glDisable(GL_BLEND);
        glUseProgram(solidProg);
        glUniform2f(uViewport, float(w), float(h));
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Pt), (void*)0);
        glEnable(GL_STENCIL_TEST);
        glDisable(GL_CULL_FACE); // the winding fans need both faces (INCR front / DECR back)
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glStencilMask(0xFF);
        glClear(GL_STENCIL_BUFFER_BIT); // scissored: the box only
        // Accumulate the window's winding into bits 0-6. One shape (a quad / an ellipse N-gon),
        // so there is no multi-layer intersection to stage through bit 7 -- the cover test below
        // reads the winding bits directly. Scissoring does not perturb winding: a pixel's winding
        // depends only on the fragments AT that pixel.
        glStencilMask(0x7F);
        glStencilFunc(GL_ALWAYS, 0, 0x7F);
        if (blurClipEvenOdd)
            glStencilOp(GL_KEEP, GL_KEEP, GL_INVERT);
        else
        {
            glStencilOpSeparate(GL_FRONT, GL_KEEP, GL_KEEP, GL_INCR_WRAP);
            glStencilOpSeparate(GL_BACK, GL_KEEP, GL_KEEP, GL_DECR_WRAP);
        }
        for (const Contour& c : blurClip)
        {
            if (c.points.size() < 3) continue;
            glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(c.points.size() * sizeof(Pt)),
                         c.points.data(), GL_STREAM_DRAW);
            glDrawArrays(GL_TRIANGLE_FAN, 0, GLsizei(c.points.size()));
        }
        glBindVertexArray(0);
        // Gate the composite to INSIDE the window (coverStencil restores the colour mask, masks
        // stencil writes off and sets EQUAL-1/NOTEQUAL-0 on the winding bits).
        coverStencil(blurClipEvenOdd);
        return true;
    }

    // Tear the gate down: restore the all-zero stencil over the feather box and turn the test
    // off, so the next draw sees exactly the state it would have seen with no blur-clip at all.
    void endBlurClipGate()
    {
        glEnable(GL_SCISSOR_TEST);
        scissorBbox(fbMn, fbMx);
        glStencilMask(0xFF);
        glClear(GL_STENCIL_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_STENCIL_TEST);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        glStencilFunc(GL_ALWAYS, 0, 0xFF); // the pipeline's between-draws default
    }

    // Composite blurred coverage (covTex.a) * solid color into the MSAA scene.
    void compositeSolid(const float color[4], float opacity)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, msFbo);
        glViewport(0, 0, w, h);
        // Cut the finished blur at the window (no-op + zero GL calls when unset).
        const bool gated = beginBlurClipGate();
        featherScissorOn(); // composite only the feather box into the scene
        glEnable(GL_BLEND);
        applyBlend();
        glUseProgram(featherCompProg);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, activeCovTex);
        glUniform1i(ufcCov, 0);
        glUniform4f(ufcColor, color[0], color[1], color[2], color[3]);
        glUniform1f(ufcOpacity, opacity);
        glBindVertexArray(emptyVao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (gated) endBlurClipGate(); // restore the all-zero stencil + test off
        glDisable(GL_SCISSOR_TEST);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);
    }

    // Composite blurred coverage * per-pixel gradient into the MSAA scene.
    void compositeGradient(bool radial, Pt p0, Pt p1, float radius,
                           const std::vector<GradientStop>& stops, float opacity,
                           const float invLin[4], const float invTrans[2])
    {
        glBindFramebuffer(GL_FRAMEBUFFER, msFbo);
        glViewport(0, 0, w, h);
        // Cut the finished blur at the window (no-op + zero GL calls when unset).
        const bool gated = beginBlurClipGate();
        featherScissorOn(); // composite only the feather box into the scene
        glEnable(GL_BLEND);
        applyBlend();
        glUseProgram(featherGradCompProg);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, activeCovTex);
        glUniform1i(ufgCov, 0);
        glUniform2f(ufgViewport, float(w), float(h));
        glUniform1i(ufgKind, radial ? 2 : 1);
        glUniform2f(ufgP0, p0.x, p0.y);
        glUniform2f(ufgP1, p1.x, p1.y);
        glUniform1f(ufgRadius, radius);
        glUniform1f(ufgOpacity, opacity);
        glUniform4f(ufgInvLin, invLin[0], invLin[1], invLin[2], invLin[3]);
        glUniform2f(ufgInvTrans, invTrans[0], invTrans[1]);
        const int n = std::min<int>(static_cast<int>(stops.size()), 16);
        glUniform1i(ufgStopCount, n);
        float offs[16] = {};
        float cols[16 * 4] = {};
        for (int i = 0; i < n; ++i)
        {
            offs[i] = stops[i].offset;
            cols[i * 4 + 0] = stops[i].r;
            cols[i * 4 + 1] = stops[i].g;
            cols[i * 4 + 2] = stops[i].b;
            cols[i * 4 + 3] = stops[i].a;
        }
        if (n > 0)
        {
            glUniform1fv(ufgStopOffset, n, offs);
            glUniform4fv(ufgStopColor, n, cols);
        }
        glBindVertexArray(emptyVao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (gated) endBlurClipGate(); // restore the all-zero stencil + test off
        glDisable(GL_SCISSOR_TEST);
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glUseProgram(0);
    }

    // Bind gradProg and upload the full gradient-cover uniform set (viewport,
    // kind, object-space line/radius, device->object inverse, stops). Shared by
    // the fill cover pass and the stroke-triangle cover pass.
    void useGradientCover(bool radial, Pt p0, Pt p1, float radius,
                          const std::vector<GradientStop>& stops, float opacity,
                          const float invLin[4], const float invTrans[2])
    {
        glUseProgram(gradProg);
        glUniform2f(ugViewport, float(w), float(h));
        glUniform1i(ugKind, radial ? 2 : 1);
        glUniform2f(ugP0, p0.x, p0.y);
        glUniform2f(ugP1, p1.x, p1.y);
        glUniform1f(ugRadius, radius);
        glUniform1f(ugOpacity, opacity);
        glUniform4f(ugInvLin, invLin[0], invLin[1], invLin[2], invLin[3]);
        glUniform2f(ugInvTrans, invTrans[0], invTrans[1]);
        const int n = std::min<int>(static_cast<int>(stops.size()), 16);
        glUniform1i(ugStopCount, n);
        float offs[16] = {};
        float cols[16 * 4] = {};
        for (int i = 0; i < n; ++i)
        {
            offs[i] = stops[i].offset;
            cols[i * 4 + 0] = stops[i].r;
            cols[i * 4 + 1] = stops[i].g;
            cols[i * 4 + 2] = stops[i].b;
            cols[i * 4 + 3] = stops[i].a;
        }
        if (n > 0)
        {
            glUniform1fv(ugStopOffset, n, offs);
            glUniform4fv(ugStopColor, n, cols);
        }
    }

    GLuint featherBlurProg = 0, featherCompProg = 0;
    GLint ufbTex = -1, ufbDir = -1, ufbInvSize = -1, ufbSigma = -1;
    GLint ufcCov = -1, ufcColor = -1, ufcOpacity = -1;
    GLuint featherGradCompProg = 0; // gradient feather composite
    GLint ufgCov = -1, ufgViewport = -1, ufgKind = -1, ufgP0 = -1, ufgP1 = -1,
          ufgRadius = -1, ufgOpacity = -1, ufgInvLin = -1, ufgInvTrans = -1,
          ufgStopCount = -1, ufgStopOffset = -1, ufgStopColor = -1;
    GLuint velProg = 0; // velocity cover (motion vectors, exact)
    GLint uvViewport = -1, uvM = -1, uvMt = -1, uvScale = -1;

    void ensurePrograms()
    {
        if (solidProg) return;
        solidProg = linkGlProgram(kCoverVS, kSolidFS);
        blitProg = linkGlProgram(kBlitVS, kBlitFS);
        gradProg = linkGlProgram(kGradVS, kGradFS);
        velProg = linkGlProgram(kVelCoverVS, kVelCoverFS);
        uvViewport = glGetUniformLocation(velProg, "uViewport");
        uvM        = glGetUniformLocation(velProg, "uM");
        uvMt       = glGetUniformLocation(velProg, "uMt");
        uvScale    = glGetUniformLocation(velProg, "uVelScale");
        uViewport = glGetUniformLocation(solidProg, "uViewport");
        uColor = glGetUniformLocation(solidProg, "uColor");
        uTex = glGetUniformLocation(blitProg, "uTex");
        ugViewport = glGetUniformLocation(gradProg, "uViewport");
        ugKind = glGetUniformLocation(gradProg, "uKind");
        ugP0 = glGetUniformLocation(gradProg, "uP0");
        ugP1 = glGetUniformLocation(gradProg, "uP1");
        ugRadius = glGetUniformLocation(gradProg, "uRadius");
        ugOpacity = glGetUniformLocation(gradProg, "uOpacity");
        ugStopCount = glGetUniformLocation(gradProg, "uStopCount");
        ugStopOffset = glGetUniformLocation(gradProg, "uStopOffset");
        ugStopColor = glGetUniformLocation(gradProg, "uStopColor");
        ugInvLin = glGetUniformLocation(gradProg, "uInvLin");
        ugInvTrans = glGetUniformLocation(gradProg, "uInvTrans");
        // Textured-quad (image) program.
        imageProg = linkGlProgram(kImageVS, kImageFS);
        uiViewport = glGetUniformLocation(imageProg, "uViewport");
        uiImage = glGetUniformLocation(imageProg, "uImage");
        uiOpacity = glGetUniformLocation(imageProg, "uOpacity");
        featherBlurProg = linkGlProgram(kBlitVS, kFeatherBlurFS);
        featherCompProg = linkGlProgram(kBlitVS, kFeatherCompFS);
        ufbTex = glGetUniformLocation(featherBlurProg, "uTex");
        ufbDir = glGetUniformLocation(featherBlurProg, "uDir");
        ufbInvSize = glGetUniformLocation(featherBlurProg, "uInvSize");
        ufbSigma = glGetUniformLocation(featherBlurProg, "uSigma");
        ufcCov = glGetUniformLocation(featherCompProg, "uCov");
        ufcColor = glGetUniformLocation(featherCompProg, "uColor");
        ufcOpacity = glGetUniformLocation(featherCompProg, "uOpacity");
        featherGradCompProg = linkGlProgram(kBlitVS, kFeatherGradCompFS);
        ufgCov = glGetUniformLocation(featherGradCompProg, "uCov");
        ufgViewport = glGetUniformLocation(featherGradCompProg, "uViewport");
        ufgKind = glGetUniformLocation(featherGradCompProg, "uKind");
        ufgP0 = glGetUniformLocation(featherGradCompProg, "uP0");
        ufgP1 = glGetUniformLocation(featherGradCompProg, "uP1");
        ufgRadius = glGetUniformLocation(featherGradCompProg, "uRadius");
        ufgOpacity = glGetUniformLocation(featherGradCompProg, "uOpacity");
        ufgInvLin = glGetUniformLocation(featherGradCompProg, "uInvLin");
        ufgInvTrans = glGetUniformLocation(featherGradCompProg, "uInvTrans");
        ufgStopCount = glGetUniformLocation(featherGradCompProg, "uStopCount");
        ufgStopOffset = glGetUniformLocation(featherGradCompProg, "uStopOffset");
        ufgStopColor = glGetUniformLocation(featherGradCompProg, "uStopColor");
        // Analytic-feather programs (jump-flood SDF + erf).
        seedProg = linkGlProgram(kBlitVS, kJfaSeedFS);
        jfaProg = linkGlProgram(kBlitVS, kJfaStepFS);
        erfProg = linkGlProgram(kBlitVS, kFeatherErfFS);
        uSeedCov = glGetUniformLocation(seedProg, "uCov");
        uSeedInv = glGetUniformLocation(seedProg, "uInvSize");
        uJfaSeed = glGetUniformLocation(jfaProg, "uSeed");
        uJfaInv = glGetUniformLocation(jfaProg, "uInvSize");
        uJfaStep = glGetUniformLocation(jfaProg, "uStep");
        uErfSeed = glGetUniformLocation(erfProg, "uSeed");
        uErfCov = glGetUniformLocation(erfProg, "uCov");
        uErfInv = glGetUniformLocation(erfProg, "uInvSize");
        uErfRadius = glGetUniformLocation(erfProg, "uFeatherRadius");
        uErfHalfWidth = glGetUniformLocation(erfProg, "uHalfWidth");
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenVertexArrays(1, &emptyVao);
        // Dedicated interleaved (pos.xy, uv.xy) VAO/VBO for textured draws.
        glGenVertexArrays(1, &imageVao);
        glGenBuffers(1, &imageVbo);
        glBindVertexArray(imageVao);
        glBindBuffer(GL_ARRAY_BUFFER, imageVbo);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                              (void*)(2 * sizeof(float)));
        glBindVertexArray(0);
    }

    void ensureTargets(int width, int height)
    {
        // The format term is REQUIRED in the early-out -- without it the
        // first-allocated format sticks forever and setColorFormat is a silent no-op.
        if (msFbo && w == width && h == height && samples == wantSamples &&
            colorFormat == wantColorFormat)
            return;
        w = width; h = height; samples = wantSamples; colorFormat = wantColorFormat;
        // covTmpTex/covTex and the viewport size change here, invalidating every
        // absolute-device-coord feather tile (content-addressing already blocks a
        // wrong hit; the flush also frees the now-dead VRAM).
        flushCache();
        if (!msFbo) glGenFramebuffers(1, &msFbo);
        if (!msColor) glGenRenderbuffers(1, &msColor);
        if (!msStencil) glGenRenderbuffers(1, &msStencil);
        glBindRenderbuffer(GL_RENDERBUFFER, msColor);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, colorFormat, w, h);
        glBindRenderbuffer(GL_RENDERBUFFER, msStencil);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples,
                                         GL_DEPTH24_STENCIL8, w, h);
        glBindFramebuffer(GL_FRAMEBUFFER, msFbo);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_RENDERBUFFER, msColor);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                  GL_RENDERBUFFER, msStencil);
        if (!resolveFbo) glGenFramebuffers(1, &resolveFbo);
        if (!resolveTex) glGenTextures(1, &resolveTex);
        glBindTexture(GL_TEXTURE_2D, resolveTex);
        // resolveTex MUST match msColor (the MSAA-resolve glBlitFramebuffer
        // requires identical read/draw formats, else GL_INVALID_OPERATION).
        glTexImage2D(GL_TEXTURE_2D, 0, colorFormat, w, h, 0, GL_RGBA,
                     colorFormat == GL_RGBA8 ? GL_UNSIGNED_BYTE : GL_HALF_FLOAT,
                     nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindFramebuffer(GL_FRAMEBUFFER, resolveFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, resolveTex, 0);

        // Feather coverage scratch: single-sample color+stencil, plus a temp
        // for the separable blur.
        auto makeColorTex = [&](GLuint& tex) {
            if (!tex) glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        };
        if (!covStencil) glGenRenderbuffers(1, &covStencil);
        glBindRenderbuffer(GL_RENDERBUFFER, covStencil);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
        makeColorTex(covTex); // GL_LINEAR; the atlas path enables mips on demand
        if (!covFbo) glGenFramebuffers(1, &covFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, covFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, covTex, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                  GL_RENDERBUFFER, covStencil);
        makeColorTex(covTmpTex);
        // bgTex rides the COLOUR chain, not the coverage chain -- it holds the
        // destination's accumulated content between beginScene/endScene round-trips.
        {
            if (!bgTex) glGenTextures(1, &bgTex);
            glBindTexture(GL_TEXTURE_2D, bgTex);
            glTexImage2D(GL_TEXTURE_2D, 0, colorFormat, w, h, 0, GL_RGBA,
                         colorFormat == GL_RGBA8 ? GL_UNSIGNED_BYTE : GL_HALF_FLOAT,
                         nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        if (!covTmpFbo) glGenFramebuffers(1, &covTmpFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, covTmpFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, covTmpTex, 0);
        // Newly (re)allocated scratch is UNDEFINED, and the feather passes read
        // outside their write scissor -- clear both fully once so the very first
        // feathered draw's rim reads see zeros, not garbage.
        glDisable(GL_SCISSOR_TEST);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glBindFramebuffer(GL_FRAMEBUFFER, covFbo);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, covTmpFbo);
        glClear(GL_COLOR_BUFFER_BIT);

        // Advanced-feather atlas targets (color-only; resized per feather in
        // blurCovAtlas).
        makeColorTex(covAtlas);
        makeColorTex(covAtlasTmp);
        if (!covAtlasFbo) glGenFramebuffers(1, &covAtlasFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, covAtlasFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, covAtlas, 0);
        if (!covAtlasTmpFbo) glGenFramebuffers(1, &covAtlasTmpFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, covAtlasTmpFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, covAtlasTmp, 0);
        atlasW = atlasH = 0;     // force atlas resize on next feather
        activeCovTex = covTex;   // default coverage source

        // Analytic-feather JFA targets: RGBA32F (exact seed coords), NEAREST so
        // seeds are never interpolated.
        auto makeJfaTex = [&](GLuint& tex) {
            if (!tex) glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT,
                         nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        };
        makeJfaTex(jfaA);
        makeJfaTex(jfaB);
        if (!jfaAFbo) glGenFramebuffers(1, &jfaAFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, jfaAFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               jfaA, 0);
        if (!jfaBFbo) glGenFramebuffers(1, &jfaBFbo);
        glBindFramebuffer(GL_FRAMEBUFFER, jfaBFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                               jfaB, 0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

GlShapePipeline::GlShapePipeline() : m_impl(std::make_unique<Impl>()) {}
GlShapePipeline::~GlShapePipeline() = default;

void GlShapePipeline::beginScene(int width, int height, unsigned int bgFbo)
{
    if (width < 1) width = 1;
    if (height < 1) height = 1;
    m_impl->ensurePrograms();
    m_impl->ensureTargets(width, height);

    // Static-feather cache LRU clock + per-frame stats (once per main frame).
    m_impl->readCacheEnv();
    ++m_impl->currentFrame;
    m_impl->frameHits = m_impl->frameMisses = m_impl->frameSkips = 0;
    m_impl->framePinnedOut = 0; // per-frame like the rest (the latch is not)

    // Snapshot the destination framebuffer (the background) so shapes can blend
    // against it -- required for multiply/screen. This MUST be the same target
    // endScene() composites into (the viewport texture), not FBO 0; reading FBO
    // 0 here would pull the whole previous-frame screen (UI panels included) into
    // the viewport and feed back.
    glBindFramebuffer(GL_READ_FRAMEBUFFER, bgFbo);
    glBindTexture(GL_TEXTURE_2D, m_impl->bgTex);
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, width, height);

    // Seed the MSAA buffer with the background (opaque), then clear stencil.
    glBindFramebuffer(GL_FRAMEBUFFER, m_impl->msFbo);
    glViewport(0, 0, width, height);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);
    glClear(GL_STENCIL_BUFFER_BIT);
    glUseProgram(m_impl->blitProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_impl->bgTex);
    glUniform1i(m_impl->uTex, 0);
    glBindVertexArray(m_impl->emptyVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

void GlShapePipeline::setBlendMode(rive::BlendMode mode)
{
    m_impl->blend = mode;
}

void GlShapePipeline::setAdvancedFeather(bool on)
{
    // Legacy two-state setter: atlas when on, gaussian when off (leaves analytic
    // alone). Prefer setFeatherMode.
    if (m_impl->featherMode != 2)
        m_impl->featherMode = on ? 1 : 0;
}

void GlShapePipeline::setFeatherMode(int mode)
{
    m_impl->featherMode = mode; // 0 gaussian, 1 atlas, 2 analytic
}

void GlShapePipeline::setBoundedFeatherFlood(bool on)
{
    m_impl->boundedFeatherFlood = on; // opt-in bounded analytic JFA (see header)
}

void GlShapePipeline::setFeatherCacheEnabled(bool on)
{
    m_impl->cacheEnabled = on; // A/B toggle for the static-feather coverage cache
    if (!on) m_impl->flushCache();
}

void GlShapePipeline::setFeatherCacheBudget(size_t bytes)
{
    m_impl->budget = bytes; // tile-pool VRAM budget (LRU-evicted)
}

void GlShapePipeline::setSampleCount(int samples)
{
    // Clamp to [1, GL_MAX_SAMPLES]; ensureTargets rebuilds the multisample
    // renderbuffers when the requested count differs from the live one.
    GLint maxS = 4;
    glGetIntegerv(GL_MAX_SAMPLES, &maxS);
    if (samples < 1) samples = 1;
    if (samples > maxS) samples = maxS;
    m_impl->wantSamples = samples;
}

void GlShapePipeline::setColorFormat(unsigned int internalFormat)
{
    // Per-INSTANCE colour-chain format (the setSampleCount idiom). Whitelist --
    // only the two formats the chain supports; anything else keeps RGBA8 (a crafted or
    // future value must not allocate an untested chain). ensureTargets rebuilds when the
    // live format differs (its early-out carries the format term).
    m_impl->wantColorFormat =
        internalFormat == GL_RGBA16F ? GL_RGBA16F : GL_RGBA8;
}

void GlShapePipeline::setClip(const std::vector<ClipLayer>& clips)
{
    m_impl->clip = clips;
}

// The POST-FEATHER intersective window (see the header for the setClip contrast -- setClip
// cuts the coverage BEFORE the blur, this cuts the composite AFTER it). Empty = off.
void GlShapePipeline::setBlurClip(const std::vector<Contour>& window, bool evenOdd)
{
    m_impl->blurClip = window;
    m_impl->blurClipEvenOdd = evenOdd;
}

void GlShapePipeline::fillSolid(const std::vector<Contour>& contours,
                                rive::FillRule rule, const float color[4])
{
    if (contours.empty()) return;
    Pt mn, mx;
    if (!contoursBounds(contours, mn, mx)) return;

    const bool eo = rule == rive::FillRule::evenOdd;
    glBindFramebuffer(GL_FRAMEBUFFER, m_impl->msFbo);
    glUseProgram(m_impl->solidProg);
    glUniform2f(m_impl->uViewport, float(m_impl->w), float(m_impl->h));
    glBindVertexArray(m_impl->vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Pt), (void*)0);

    m_impl->buildClipMask(mn, mx);
    m_impl->windingPass(contours, eo);

    glEnable(GL_BLEND);
    m_impl->applyBlend();
    glUniform4f(m_impl->uColor, color[0], color[1], color[2], color[3]);
    m_impl->coverStencil(eo);
    const Pt quad[6] = {{mn.x, mn.y}, {mx.x, mn.y}, {mx.x, mx.y},
                        {mn.x, mn.y}, {mx.x, mx.y}, {mn.x, mx.y}};
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    m_impl->resetStencil(mn, mx);
    glBindVertexArray(0);
}

void GlShapePipeline::fillVelocity(const std::vector<Contour>& contours,
                                   rive::FillRule rule, const float linM[4],
                                   const float transM[2], float velScale)
{
    if (contours.empty()) return;
    Pt mn, mx;
    if (!contoursBounds(contours, mn, mx)) return;

    const bool eo = rule == rive::FillRule::evenOdd;
    glBindFramebuffer(GL_FRAMEBUFFER, m_impl->msFbo);
    glUseProgram(m_impl->solidProg);
    glUniform2f(m_impl->uViewport, float(m_impl->w), float(m_impl->h));
    glBindVertexArray(m_impl->vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Pt), (void*)0);

    m_impl->buildClipMask(mn, mx);
    m_impl->windingPass(contours, eo);

    // Cover pass with the velocity program. mat2 is column-major:
    // [xx, yx, xy, yy] from rive Mat2D (linM = {xx, xy, yx, yy}).
    glUseProgram(m_impl->velProg);
    glUniform2f(m_impl->uvViewport, float(m_impl->w), float(m_impl->h));
    const float m2[4] = {linM[0], linM[2], linM[1], linM[3]};
    glUniformMatrix2fv(m_impl->uvM, 1, GL_FALSE, m2);
    glUniform2f(m_impl->uvMt, transM[0], transM[1]);
    glUniform1f(m_impl->uvScale, velScale);
    glEnable(GL_BLEND);
    m_impl->applyBlend();
    m_impl->coverStencil(eo);
    const Pt quad[6] = {{mn.x, mn.y}, {mx.x, mn.y}, {mx.x, mx.y},
                        {mn.x, mn.y}, {mx.x, mx.y}, {mn.x, mx.y}};
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    m_impl->resetStencil(mn, mx);
    glBindVertexArray(0);
}

void GlShapePipeline::fillGradient(const std::vector<Contour>& contours,
                                   rive::FillRule rule, bool radial, Pt p0,
                                   Pt p1, float radius,
                                   const std::vector<GradientStop>& stops,
                                   float opacity, const float invLin[4],
                                   const float invTrans[2])
{
    if (contours.empty()) return;
    Pt mn, mx;
    if (!contoursBounds(contours, mn, mx)) return;

    const bool eo = rule == rive::FillRule::evenOdd;
    glBindFramebuffer(GL_FRAMEBUFFER, m_impl->msFbo);
    glUseProgram(m_impl->solidProg);
    glUniform2f(m_impl->uViewport, float(m_impl->w), float(m_impl->h));
    glBindVertexArray(m_impl->vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Pt), (void*)0);

    m_impl->buildClipMask(mn, mx);
    m_impl->windingPass(contours, eo);

    // --- Cover pass with the gradient program. ---
    m_impl->useGradientCover(radial, p0, p1, radius, stops, opacity, invLin,
                             invTrans);
    glEnable(GL_BLEND);
    m_impl->applyBlend();
    m_impl->coverStencil(eo);
    const Pt quad[6] = {{mn.x, mn.y}, {mx.x, mn.y}, {mx.x, mx.y},
                        {mn.x, mn.y}, {mx.x, mx.y}, {mn.x, mx.y}};
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    m_impl->resetStencil(mn, mx);
    glBindVertexArray(0);
}

void GlShapePipeline::fillFeathered(const std::vector<Contour>& contours,
                                    rive::FillRule rule, const float color[4],
                                    float featherPx, float opacity)
{
    if (contours.empty()) return;
    Pt mn, mx;
    if (!contoursBounds(contours, mn, mx)) return;
    const bool eo = rule == rive::FillRule::evenOdd;
    // 0-2. static-feather cache: sets the feather box, then serves a cached
    // coverage tile (hit) or computes + stores it (miss). fill: hw = inf.
    m_impl->ensureFeatherCoverage(0, &contours, eo, nullptr, featherPx, 1.0e6f,
                                  mn, mx);
    m_impl->compositeSolid(color, opacity);    // 3. coverage * paint -> scene
    m_impl->clearFeatherScratch();             // 4. zero our box (no stale bleed)
}

void GlShapePipeline::strokeFeathered(const std::vector<Pt>& tris,
                                      const float color[4], float featherPx,
                                      float opacity, float strokeHalfWidth)
{
    // Feathered stroke: feather the stroke's own coverage instead of drawing a
    // hard ribbon. The analytic feather needs the band half-width so the two
    // edges' erfs combine correctly (a thin stroke is dim, a thick one keeps a
    // bright core) -- the gaussian/atlas modes ignore it.
    if (tris.size() < 3) return;
    Pt mn{1e30f, 1e30f}, mx{-1e30f, -1e30f};
    for (const Pt& p : tris)
    {
        mn.x = std::min(mn.x, p.x);
        mn.y = std::min(mn.y, p.y);
        mx.x = std::max(mx.x, p.x);
        mx.y = std::max(mx.y, p.y);
    }
    m_impl->ensureFeatherCoverage(1, nullptr, false, &tris, featherPx,
                                  std::max(strokeHalfWidth, 0.5f), mn, mx);
    m_impl->compositeSolid(color, opacity);
    m_impl->clearFeatherScratch();            // zero our box (no stale bleed)
}

void GlShapePipeline::fillFeatheredGradient(
    const std::vector<Contour>& contours, rive::FillRule rule, bool radial,
    Pt p0, Pt p1, float radius, const std::vector<GradientStop>& stops,
    float featherPx, float opacity, const float invLin[4],
    const float invTrans[2])
{
    // Feathered gradient fill: blur the fill coverage, then composite the actual
    // gradient per pixel (not a flat averaged color).
    if (contours.empty()) return;
    Pt mn, mx;
    if (!contoursBounds(contours, mn, mx)) return;
    const bool eo = rule == rive::FillRule::evenOdd;
    m_impl->ensureFeatherCoverage(0, &contours, eo, nullptr, featherPx, 1.0e6f,
                                  mn, mx); // fill: hw = inf
    m_impl->compositeGradient(radial, p0, p1, radius, stops, opacity, invLin,
                              invTrans);
    m_impl->clearFeatherScratch();           // zero our box (no stale bleed)
}

void GlShapePipeline::fillTriangles(const std::vector<Pt>& tris,
                                    const float color[4])
{
    if (tris.size() < 3) return;
    Pt mn{1e30f, 1e30f}, mx{-1e30f, -1e30f};
    for (const Pt& p : tris)
    {
        mn.x = std::min(mn.x, p.x);
        mn.y = std::min(mn.y, p.y);
        mx.x = std::max(mx.x, p.x);
        mx.y = std::max(mx.y, p.y);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, m_impl->msFbo);
    glUseProgram(m_impl->solidProg);
    glUniform2f(m_impl->uViewport, float(m_impl->w), float(m_impl->h));
    glBindVertexArray(m_impl->vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Pt), (void*)0);

    m_impl->buildClipMask(mn, mx);

    // Union the (possibly self-overlapping) ribbon triangles into the stencil,
    // gated to the clip bit, so overlaps are drawn ONCE in the cover pass -- no
    // double-blend seams on self-intersecting strokes.
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glEnable(GL_STENCIL_TEST);
    glStencilMask(0x7F);
    m_impl->clipGate(); // gate the ribbon union to the clip mask (ALWAYS unclipped)
    glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);
    glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(tris.size() * sizeof(Pt)),
                 tris.data(), GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, GLsizei(tris.size()));

    // Cover the union once.
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_BLEND);
    m_impl->applyBlend();
    glUniform4f(m_impl->uColor, color[0], color[1], color[2], color[3]);
    glStencilMask(0x00);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glStencilFunc(GL_NOTEQUAL, 0x00, 0x7F);
    const Pt quad[6] = {{mn.x, mn.y}, {mx.x, mn.y}, {mx.x, mx.y},
                        {mn.x, mn.y}, {mx.x, mx.y}, {mn.x, mx.y}};
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    m_impl->resetStencil(mn, mx);
    glBindVertexArray(0);
}

void GlShapePipeline::fillTrianglesGradient(const std::vector<Pt>& tris,
                                            bool radial, Pt p0, Pt p1,
                                            float radius,
                                            const std::vector<GradientStop>& stops,
                                            float opacity, const float invLin[4],
                                            const float invTrans[2])
{
    if (tris.size() < 3) return;
    Pt mn{1e30f, 1e30f}, mx{-1e30f, -1e30f};
    for (const Pt& p : tris)
    {
        mn.x = std::min(mn.x, p.x);
        mn.y = std::min(mn.y, p.y);
        mx.x = std::max(mx.x, p.x);
        mx.y = std::max(mx.y, p.y);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, m_impl->msFbo);
    glUseProgram(m_impl->solidProg);
    glUniform2f(m_impl->uViewport, float(m_impl->w), float(m_impl->h));
    glBindVertexArray(m_impl->vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Pt), (void*)0);

    m_impl->buildClipMask(mn, mx);

    // Union the ribbon triangles into the stencil (identical to
    // fillTriangles): overlaps cover ONCE, no double-blend seams.
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glEnable(GL_STENCIL_TEST);
    glStencilMask(0x7F);
    m_impl->clipGate(); // gate the ribbon union to the clip mask (ALWAYS unclipped)
    glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);
    glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(tris.size() * sizeof(Pt)),
                 tris.data(), GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, GLsizei(tris.size()));

    // Cover the union once with the gradient program.
    m_impl->useGradientCover(radial, p0, p1, radius, stops, opacity, invLin,
                             invTrans);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_BLEND);
    m_impl->applyBlend();
    glStencilMask(0x00);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    glStencilFunc(GL_NOTEQUAL, 0x00, 0x7F);
    const Pt quad[6] = {{mn.x, mn.y}, {mx.x, mn.y}, {mx.x, mx.y},
                        {mn.x, mn.y}, {mx.x, mx.y}, {mn.x, mx.y}};
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    m_impl->resetStencil(mn, mx);
    glBindVertexArray(0);
}

void GlShapePipeline::strokeFeatheredGradient(
    const std::vector<Pt>& tris, bool radial, Pt p0, Pt p1, float radius,
    const std::vector<GradientStop>& stops, float featherPx, float opacity,
    float strokeHalfWidth, const float invLin[4], const float invTrans[2])
{
    // Feathered gradient stroke: strokeFeathered's coverage pipeline with
    // fillFeatheredGradient's per-pixel gradient composite.
    if (tris.size() < 3) return;
    Pt mn{1e30f, 1e30f}, mx{-1e30f, -1e30f};
    for (const Pt& p : tris)
    {
        mn.x = std::min(mn.x, p.x);
        mn.y = std::min(mn.y, p.y);
        mx.x = std::max(mx.x, p.x);
        mx.y = std::max(mx.y, p.y);
    }
    m_impl->ensureFeatherCoverage(1, nullptr, false, &tris, featherPx,
                                  std::max(strokeHalfWidth, 0.5f), mn, mx);
    m_impl->compositeGradient(radial, p0, p1, radius, stops, opacity, invLin,
                              invTrans);
    m_impl->clearFeatherScratch();            // zero our box (no stale bleed)
}

void GlShapePipeline::drawTexturedQuad(const Pt corners[4], const float uvs[8],
                                       unsigned int glTex, float opacity)
{
    if (glTex == 0) return;
    // Quad bbox (device space) for the clip mask + stencil scissor.
    Pt mn{1e30f, 1e30f}, mx{-1e30f, -1e30f};
    for (int i = 0; i < 4; ++i)
    {
        mn.x = std::min(mn.x, corners[i].x);
        mn.y = std::min(mn.y, corners[i].y);
        mx.x = std::max(mx.x, corners[i].x);
        mx.y = std::max(mx.y, corners[i].y);
    }
    if (mx.x <= mn.x || mx.y <= mn.y) return;

    glBindFramebuffer(GL_FRAMEBUFFER, m_impl->msFbo);
    glViewport(0, 0, m_impl->w, m_impl->h);

    // Build the clip mask into stencil bit 7 over the bbox using the shared
    // solid program + position-only vao/vbo (same as the fill paths). With an
    // active clip this sets bit7=1 only inside the intersection; with no clip it
    // is skipped and clipGate() falls back to GL_ALWAYS, so the cover gate below
    // passes everywhere inside the quad either way.
    glUseProgram(m_impl->solidProg);
    glUniform2f(m_impl->uViewport, float(m_impl->w), float(m_impl->h));
    glBindVertexArray(m_impl->vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Pt), (void*)0);
    m_impl->buildClipMask(mn, mx);

    // Cover pass: draw the two textured triangles, gated to the clip bit.
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_STENCIL_TEST);
    glStencilMask(0x00);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    m_impl->clipGate(); // gate the textured quad to the clip mask (ALWAYS unclipped)
    glEnable(GL_BLEND);
    m_impl->applyBlend(); // honor the current blend mode

    glUseProgram(m_impl->imageProg);
    glUniform2f(m_impl->uiViewport, float(m_impl->w), float(m_impl->h));
    glUniform1f(m_impl->uiOpacity, opacity);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, glTex);
    glUniform1i(m_impl->uiImage, 0);

    // Two triangles (0,1,2)+(0,2,3), interleaved pos.xy + uv.xy.
    const int idx[6] = {0, 1, 2, 0, 2, 3};
    float verts[6 * 4];
    for (int i = 0; i < 6; ++i)
    {
        const int c = idx[i];
        verts[i * 4 + 0] = corners[c].x;
        verts[i * 4 + 1] = corners[c].y;
        verts[i * 4 + 2] = uvs[c * 2 + 0];
        verts[i * 4 + 3] = uvs[c * 2 + 1];
    }
    glBindVertexArray(m_impl->imageVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->imageVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    m_impl->resetStencil(mn, mx);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

void GlShapePipeline::drawTexturedMesh(const float* posUv, int vertCount,
                                       unsigned int glTex, float opacity)
{
    if (glTex == 0 || posUv == nullptr || vertCount < 3) return;
    // Mesh bbox (device space) for the clip mask + stencil scissor.
    Pt mn{1e30f, 1e30f}, mx{-1e30f, -1e30f};
    for (int i = 0; i < vertCount; ++i)
    {
        const float x = posUv[i * 4 + 0], y = posUv[i * 4 + 1];
        mn.x = std::min(mn.x, x);
        mn.y = std::min(mn.y, y);
        mx.x = std::max(mx.x, x);
        mx.y = std::max(mx.y, y);
    }
    if (mx.x <= mn.x || mx.y <= mn.y) return;

    glBindFramebuffer(GL_FRAMEBUFFER, m_impl->msFbo);
    glViewport(0, 0, m_impl->w, m_impl->h);

    // Clip mask into stencil bit 7 over the bbox (shared solid prog + pos vao),
    // same as drawTexturedQuad / the fill paths.
    glUseProgram(m_impl->solidProg);
    glUniform2f(m_impl->uViewport, float(m_impl->w), float(m_impl->h));
    glBindVertexArray(m_impl->vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Pt), (void*)0);
    m_impl->buildClipMask(mn, mx);

    // Cover: the textured triangles, gated to the clip bit.
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glEnable(GL_STENCIL_TEST);
    glStencilMask(0x00);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
    m_impl->clipGate(); // gate the textured mesh to the clip mask (ALWAYS unclipped)
    glEnable(GL_BLEND);
    m_impl->applyBlend();

    glUseProgram(m_impl->imageProg);
    glUniform2f(m_impl->uiViewport, float(m_impl->w), float(m_impl->h));
    glUniform1f(m_impl->uiOpacity, opacity);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, glTex);
    glUniform1i(m_impl->uiImage, 0);

    glBindVertexArray(m_impl->imageVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_impl->imageVbo);
    glBufferData(GL_ARRAY_BUFFER,
                 GLsizeiptr(size_t(vertCount) * 4 * sizeof(float)), posUv,
                 GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, vertCount);

    m_impl->resetStencil(mn, mx);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

void GlShapePipeline::endScene(unsigned int dstFbo)
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_impl->msFbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_impl->resolveFbo);
    glBlitFramebuffer(0, 0, m_impl->w, m_impl->h, 0, 0, m_impl->w, m_impl->h,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, dstFbo);
    glViewport(0, 0, m_impl->w, m_impl->h);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_SCISSOR_TEST);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(m_impl->blitProg);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_impl->resolveTex);
    glUniform1i(m_impl->uTex, 0);
    glBindVertexArray(m_impl->emptyVao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    // Env-gated feather-cache report (RIVE_FEATHER_CACHE_STATS=1). Off by default
    // so it never touches pixels or the default stdout.
    if (m_impl->statsEnv)
    {
        const uint64_t tot =
            m_impl->frameHits + m_impl->frameMisses + m_impl->frameSkips;
        const double hitPct =
            tot ? 100.0 * double(m_impl->frameHits) / double(tot) : 0.0;
        std::fprintf(stderr,
                     "[feather-cache] frame=%llu hits=%llu misses=%llu "
                     "skips=%llu pinnedOut=%llu hit%%=%.1f tiles=%zu vram=%.2fMB "
                     "budget=%zuMB(%s)%s\n",
                     (unsigned long long)m_impl->currentFrame,
                     (unsigned long long)m_impl->frameHits,
                     (unsigned long long)m_impl->frameMisses,
                     (unsigned long long)m_impl->frameSkips,
                     (unsigned long long)m_impl->framePinnedOut, hitPct,
                     m_impl->cache.size(),
                     double(m_impl->cacheBytes) / (1024.0 * 1024.0),
                     m_impl->budget >> 20, m_impl->budgetSource,
                     m_impl->framePinnedOut ? " SATURATED" : "");
    }
}
} // namespace rive_backend
