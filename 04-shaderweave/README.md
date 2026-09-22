# ShaderWeave — node-based GPU programming environment

GLSL, CUDA-JIT and OptiX kernels share one feedback graph; every intermediate is live and inspectable.
This is the most hand-written project in the collection, and the one to read for software engineering:
runtimes, resource ownership, scheduling, reflection. Portfolio:
https://lovasz-local-lemma.github.io/projects/shaderweave/index.html

## Runtimes and graph

| Read | What it shows |
| --- | --- |
| [`CodeNodeRuntime.cpp`](CodeNodeRuntime.cpp) / [`.hpp`](CodeNodeRuntime.hpp) | PImpl-isolated CUDA and OptiX node runtimes. NVRTC compiles user source (wrapped with an injected `Uniforms` header and a `mainImageKernel` footer) to PTX, loaded with `cuModuleLoadDataEx`, writing into a GL PBO registered through `cudaGraphicsGLRegisterBuffer`; a two-tier saved/project kernel override lets a live-edited kernel shadow the committed one. Every driver call is checked with if-init statements and surfaces a readable `last_error`; the OptiX side maps an input GL texture into a `cudaTextureObject_t` per frame and unwinds on every failure path. |
| [`NodeGraph.hpp`](NodeGraph.hpp) | Pull-based evaluation: `render_dependency_texture` renders upstream nodes recursively with a visiting set for cycle detection, ping-pongs target/feedback targets when a node links to itself, and gates recompilation on a per-node revision counter. Sockets are derived from the shader source (`iChannelN`, `sampler2D/3D/Cube`, `image2D/3D`, `writeonly` → output) and rebuilt while preserving pin ids so links survive an edit; three backend preview runtimes hang off one `GraphNode`. |
| [`LegacyNodeGraphActivity.hpp`](LegacyNodeGraphActivity.hpp) | Per-frame render-plan construction: a path-compressed, union-by-rank disjoint set groups nodes into components, the component of the projected node becomes the active set, and a post-order DFS over input links yields a dependency-ordered render list that terminates on feedback cycles. Memoised behind a `(graph_revision, projected_node, preview_revision)` key so the O(N+E) rebuild runs only on topology change. |
| [`optix_related.hpp`](optix_related.hpp) | Complete OptiX 7+ host bring-up: `OptixModuleManager` (NVRTC → PTX → `optixModuleCreate` with a per-name cache), `OptixPipelineBuilder` (program groups, pipeline link, per-group stack-size aggregation, aligned SBT records, launch with a persistent params buffer), `OptixContextManager` (driver-context discovery with runtime fallback), and the device-side prelude injected before user code. |
| [`cuda_related.hpp`](cuda_related.hpp) | The runtime-API counterpart: `CUDAKernel`, `CUDAGLTexture` interop, a `DisplayShader`, and an `FFTConvolver`. |
| [`shader_related.hpp`](shader_related.hpp) | Reflection-driven uniforms: `GL_ACTIVE_UNIFORMS` enumerated into a name → meta map, values in a `std::variant`, `set_uniform` dispatched by `std::visit` to a static-overload `UniformDispatcher` with `if constexpr` gating and a `static_assert` on unsupported types; `gen_missing_uniforms` scans Shadertoy-style source and injects omitted declarations. |
| [`progressive_pt.cu`](progressive_pt.cu) | A compact SDF sphere-traced path tracer in one raygen program: TEA-hash RNG, cosine hemisphere sampling, diffuse / rough-metal / dielectric lobes with Schlick and TIR, Russian roulette, thin-lens depth of field, motion blur, running-mean accumulation, ACES. |

## ImGui layer

| Read | What it shows |
| --- | --- |
| [`FancyKnob.hpp`](FancyKnob.hpp) | `FancyKnob`: an immediate-mode control built from ImGui's lower-level APIs — `ButtonBehavior` for interaction, `ImGuiStorage` for persistent animation state, draw-list geometry for rounded paths, layered shading, shadows, animated highlights and optional additive glow. Appearance is produced from geometry each frame, never from an image. |
| [`Window_Editor.hpp`](Window_Editor.hpp), [`Window_Node.hpp`](Window_Node.hpp) | The live-edit window that uses the widgets, and the node window (drag-and-drop, pending drops, per-node preview). |
| [`node-editor-fork.diff`](node-editor-fork.diff) | My fork of Michal Cichon's `imgui-node-editor` as a diff against upstream: `BeginNode(id, traits)` at the public seam, trait storage on each editor node, propagation after layout, and a custom border path where the `Ribbon` trait drives animated perimeter geometry and every node receives a layered glow (784 added / 239 removed lines in `imgui_node_editor.cpp`). Only the diff is included; the library keeps its MIT notice (`LICENSE-imgui-node-editor.txt`). |
