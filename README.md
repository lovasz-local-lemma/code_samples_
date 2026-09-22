# Code samples — Shaojie Jiao

**Reading guide (start here): https://lovasz-local-lemma.github.io/code-samples/** — every file introduced
in a sentence, entry points linked, the passages worth a reviewer's minute shown inline. This README is the
table of contents behind it.

Selected source from my rendering research, graphics tools, simulations and geometry projects, chosen
to be *read*. To make inspection easy I have lifted these files out of their projects rather than pointing
at whole repositories. Everything shown is my own work; external code appears only where it is the interface
my code plugs into (Tungsten's renderer framework, the Rive runtime's path types, ImGui, CUDA/OptiX), and it
is named as such. Nothing here needs to be built; the systems themselves, with recordings and live demos, are
on the portfolio: https://lovasz-local-lemma.github.io/ — contact: shaojie.jiao.gr@dartmouth.edu

Three tiers: **research** (01–02: photon primitives in Tungsten, RadianceLab), **personal projects**
(03–05, 07: RIVX, ShaderWeave, Crucible2.5D and Modal2D, Fortune's Loom), and **breadth** (06, 08: a compact
volumetric path tracer with its verification harnesses, a multi-backend FFT with a measured planner, a
red-black ordered set, render-graph scheduling, a bilateral grid).

## Start here (an afternoon's reading, in this order)

1. [`02-radiancelab/pp_hourglass_sheet_solver.cpp`](02-radiancelab/pp_hourglass_sheet_solver.cpp) — the photon hourglass: refraction chain, a double-precision sheet solver whose discriminant is the estimator's Jacobian, branch recovery.
2. [`01-tungsten-photon-primitives/CONTRIBUTION.md`](01-tungsten-photon-primitives/CONTRIBUTION.md) — the SIGGRAPH 2019 research branch of Tungsten, with my contribution mapped function by function and diffed against upstream.
3. [`03-rivx/render/gl_shape_renderer.cpp`](03-rivx/render/gl_shape_renderer.cpp) — a production stencil-and-cover vector rasterizer with jump-flood feathering and a content-hashed tile cache.
4. [`05-simulation/crucible/mpm_p2g.comp`](05-simulation/crucible/mpm_p2g.comp) — MLS-MPM particle-to-grid with snow, sand, anisotropic damage and phase-field fracture in one compute kernel.
5. [`04-shaderweave/CodeNodeRuntime.cpp`](04-shaderweave/CodeNodeRuntime.cpp) — NVRTC-compiled CUDA and OptiX node runtimes behind a PImpl boundary, every failure path unwound.

## Sections

| Folder | What is in it |
| --- | --- |
| [`01-tungsten-photon-primitives/`](01-tungsten-photon-primitives/CONTRIBUTION.md) | Photon primitives in Tungsten: analytic ray–primitive intersection, closed-form Jacobians, primitive construction, grid acceleration — ~6,200 lines added over 1,270 upstream, diffs included |
| [`02-radiancelab/`](02-radiancelab/README.md) | The hourglass solver, the rasterized correlated gather, bidirectional MIS on the GPU, an FFT-preconditioned PCG phase unwrapper, the swept-shape classifier, a boundary-integral derivative study |
| [`03-rivx/`](03-rivx/README.md) | GL-free geometry core (contour tracing, curves, flattening/stroking) with tests; stencil-and-cover renderer; caustic optics; conic projection; order-independent occlusion for vector export |
| [`04-shaderweave/`](04-shaderweave/README.md) | CUDA-JIT / OptiX node runtimes, pull-based graph evaluation, render-plan scheduling, reflection-driven uniforms, ImGui widgets and a node-editor fork as a diff |
| [`05-simulation/`](05-simulation/README.md) | Crucible2.5D: MPM P2G / grid / damage kernels, GPU counting-sort neighbour search, SPH forces, codimensional detection, blast fronts. Modal2D: shell and plate finite elements, shift-invert Lanczos, real-time friction-driven modal synthesis |
| [`06-light-transport-compact/`](06-light-transport-compact/README.md) | A compact volumetric path tracer (VNDF GGX, delta/ratio tracking, MIS) and the brute-force verification harnesses that keep it honest |
| [`07-geometry/`](07-geometry/README.md) | Fortune's sweepline with five balanced-tree beachline policies, robust predicates, lazy-exact scalars, Apollonius diagrams; DFA → regular expression |
| [`08-numerics-and-engineering/`](08-numerics-and-engineering/README.md) | A radix-2 FFT with compile-time backend policies and a measured planner; a red-black ordered set with executable invariants; render-graph scheduling; a bilateral grid — each with tests |

## Provenance

Every file here is code I wrote or directed. Third-party code is not redistributed except as diffs against it
(Tungsten, `imgui-node-editor`), and every notice is kept.

## License

My code is MIT (`LICENSE`). The Tungsten excerpts keep Tungsten's license
(`01-tungsten-photon-primitives/LICENSE.txt`); my additions to those files are offered under the same terms.
`imgui-node-editor` is MIT (`04-shaderweave/LICENSE-imgui-node-editor.txt`).
