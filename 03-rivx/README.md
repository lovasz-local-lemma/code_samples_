# RIVX — Rive rendering and authoring studio

RIVX turns 3D scenes into Rive artwork and animation: a custom stencil-and-cover backend beside the
official one, scene inspection, content-aware effects, and export to `.riv` checked in the official
runtime. Portfolio: https://lovasz-local-lemma.github.io/projects/rivex/index.html

## Geometry core (GL-free, unit-tested)

| Read | What it shows |
| --- | --- |
| [`curve/contour_trace.cpp`](curve/contour_trace.cpp) | Raster field → binarize → largest 8-connected component → Moore-neighbour boundary trace (Jacob's stopping criterion) → Douglas–Peucker → resample to N points evenly spaced by arc length. Pure `std`, deterministic, self-test built in. |
| [`curve/curve_eval.cpp`](curve/curve_eval.cpp) | Bezier of any degree (de Casteljau), clamped and periodic uniform B-splines, Yuksel C² interpolating splines → sampled polylines; stable at degenerate inputs, no exceptions in hot paths. |
| [`render/shape_geometry.cpp`](render/shape_geometry.cpp) | Path flattening, fill-rule classification (non-zero / even-odd), stroke tessellation with joins and caps, gradient sampling, feather sigma. Uses the Rive runtime's path and matrix types. |
| [`tests/`](tests/) | The three test programs (`curve_eval_tests`, `shape_geometry_tests`, the contour self-test). Only the test programs were written with AI assistance and keep their inline markers; the modules they test are mine. |

## Rendering and 3D-to-vector

| Read | What it shows |
| --- | --- |
| [`render/gl_shape_renderer.cpp`](render/gl_shape_renderer.cpp) | The stencil-then-cover vector rasterizer: stencil bit-7 clip-mask intersection with per-layer winding (`INCR_WRAP`/`DECR_WRAP` for non-zero, `INVERT` for even-odd), a scissored all-zero-stencil invariant, analytic feather as seed → jump-flood SDF → erf coverage with a bounded flood schedule, and a content-hashed (FNV-1a over contours, clips and parameters) LRU tile cache for static feather coverage with in-frame pinning and byte-identical replay via `glBlitFramebuffer`. `buildClipMask`, `windingPass`, `coverStencil`, `featherAnalytic`, `ensureFeatherCoverage`. |
| [`lab3d/caustic_solver.hpp`](lab3d/caustic_solver.hpp) | Exact meridian ray-fan optics for a point light through a glass sphere: Snell with TIR/graze classification, optical-vs-geometric clocks, wavefronts split at folds (sign change of the turning cross product), the caustic as the envelope of adjacent-ray crossings plus the axial focus segment, sphere/sphere and sphere/plane intersection circles, revolve-to-3D, clipping by box and sphere shadow. Double inside, float at the API edge, scale-relative thresholds. |
| [`lab3d/primitive_project.hpp`](lab3d/primitive_project.hpp) | Closed-form perspective projection of primitives to screen conics, with the derivation of the sphere silhouette ellipse from the tangent cone in the comments (`ξ₀ = sin b cos b / A`, semi-axes `sin a cos a / A` and `sin a / √A`, `A = cos²α − sin²β`), and a moment-fit ellipse for perspective disks via centroid and 2×2 covariance eigen-decomposition. |
| [`scene3d/scene3d_occlusion.cpp`](scene3d/scene3d_occlusion.cpp) | A pluggable occlusion stage (none → clip path → z-buffer → split → analytic) that makes painter-order export order-independent: Sutherland–Hodgman clipping to a convex window, union of convex polygons by edge classification with loop chaining through a quantized endpoint map, true ray depth per pixel, and a plane–plane split line for mutually overlapping faces, correct under both fill rules. |
