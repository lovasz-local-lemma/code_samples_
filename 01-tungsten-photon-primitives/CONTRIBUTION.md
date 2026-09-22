# Photon primitives in Tungsten — what is mine

These eight files come from my research branch of [Tungsten](https://github.com/tunabrain/tungsten),
Benedikt Bitterli's physically based renderer. The branch implements *photon primitives*: photon paths
turned into geometric primitives (beams, planes, spheres, disks, cones, cylinders, hyperboloids) that a
camera ray intersects analytically, giving unbiased volumetric density estimators combined under MIS
(SIGGRAPH 2019, "Photon Surfaces for Robust, Unbiased Volumetric Density Estimation", joint first author;
"Temporally Sliced Photon Primitives for Time-of-Flight Rendering", EGSR 2022, co-author). The work
continues today in RadianceLab, my own renderer.

The renderer framework — scene, materials, media, sampling, the original photon map — is Bitterli's and
keeps his license (`LICENSE.txt`). **My contribution is the diff against upstream**, reproduced in
`upstream-diff/` (unified diffs against upstream `master`, both sides formatted with the same
`.clang-format` so only real differences show; the base revision I branched from is older, so a handful
of the "removed" lines are upstream's later edits rather than mine).

The files were cleaned for publication: commented-out old code and scratch notes were removed, and
nothing else. `../.tools/verify_comment_only.py` strips comments from the archived originals and from these
copies, tokenizes both, and confirms the code token streams are identical; it passes on all eight files.
That cleanup removed about 3,300 lines of dead comments, most of them from the experimental path tracer.

## Size of the contribution

| File | Upstream lines | This branch | Upstream lines kept | Added |
| --- | ---: | ---: | ---: | ---: |
| `photon-map/Photon.hpp` | 133 | 3,221 | 122 | **3,099** |
| `photon-map/PhotonMapIntegrator.cpp` | 589 | 1,453 | 568 | **885** |
| `photon-map/PhotonMapIntegrator.hpp` | 112 | 150 | 111 | 39 |
| `photon-map/GridAccel.hpp` | 296 | 536 | 295 | **241** |
| `photon-map/photon.cpp` | — (new file) | 144 | 0 | **144** |
| `experimental-path-tracer/PathTracer.cpp` | 169 | 1,796 | 98 | **1,698** |
| `experimental-path-tracer/PathTracer.hpp` | 24 | 45 | 23 | 22 |
| `experimental-path-tracer/PathTracerSettings.hpp` | 52 | 106 | 52 | 54 |

About 6,200 added lines of live code over 1,270 retained upstream lines (line counts after formatting
both sides identically; `difflib`, whole lines).

## Where to read — in this order

1. **`photon-map/photon.cpp`** — `PhotonPrimitive::precompute` (line 54): how a stored photon path
   segment becomes a primitive: choose the segment, derive the cached geometry, compute bounds. The
   shortest complete entry point. *All mine.*
2. **`photon-map/Photon.hpp` → `class PP_PRIM`** (line 1046, to ~2500). The primitive itself. Three
   routines carry the mathematics:
   - `intersect` (line 1275): analytic ray–primitive intersection dispatched by primitive type, returning
     an `intersection_solution` object whose `intersection_root` records carry typed hit coordinates (UV,
     UVW, T) — the object-oriented form. The later solver in RadianceLab writes roots and Jacobians into flat
     fixed arrays instead: the same mathematics, with elegance traded for throughput.
   - `_update_constJAC` (line 1689): the closed-form Jacobians that turn a hit into an unbiased density
     estimate — the part that makes the estimators unbiased rather than kernel-blurred.
   - `precompute` (line 1949): the per-primitive cache (frames, bounding boxes, strategy selection).
   `class PP_subpath` (line 897) holds the path-side quantities the primitive is built from. *All mine.*
3. **`photon-map/Photon.hpp` → `class PhotonPrimitive`** (line 2509): the owner of a photon's primitives
   across strategies; `precompute_as_beam` (line 2728) is the beam special case. *Mine.*
4. **`photon-map/PhotonMapIntegrator.cpp`** — `makePrimitiveBVH` (line 393), `buildPrimitiveAccStructure`
   (541), `Add_Dangling_Segments` (564), `Add_marg_segments` (737), `PostProcess_PathPhotons` (772),
   `prepare_PhotonPrimitives` (1029): connecting traced photon paths, estimator settings, and primitive
   construction into the integrator. *Mine; the surrounding integrator skeleton is upstream.*
5. **`photon-map/GridAccel.hpp`** — `iteratePrimitive` (line 174) and `buildAccel_primitives` (353): the
   grid acceleration structure extended from points to extended primitives. *Mine; the point-grid is upstream.*
6. **`experimental-path-tracer/PathTracer.cpp`** — the same idea in the other framework. Upstream's path
   tracer is a few dozen lines; mine turns each camera sample into an *uncorrelated* photon-primitive
   estimate: draw a light position, direction and free-flight distance, build the sheet those sampled
   dimensions sweep (`iz*` / `surf_type`, line 49), intersect the camera ray with it analytically (`inter`,
   line 156: plane, quad, cone, sphere, cylinder), and divide by the same Jacobian (`routine`, line 691;
   `TP` / `PDF_t`, line 505). No stored photon map, so no correlation between pixels and nothing to cache,
   where the photon-map side freezes primitives from stored paths and reuses them. `inter` is the flat
   counterpart of `PP_PRIM::intersect`, and the closing Jacobian factor of `_update_constJAC`. *Mine; the
   camera/light sampling calls are upstream's interfaces.*

## Honest notes on the snapshot

This is research code as it was when the experiments ran, with dead comments removed and nothing else
changed. Three things a careful reader will find, all mine to own: `PP_PRIM::am_I` has no return on the invalid-primitive path;
`COMB_ALL` in `prepare_PhotonPrimitives` is an unimplemented branch marked "not handled yet"; and
`traceSample` in the experimental path tracer returns after one selected experiment, leaving its later
combination block inactive. One helper in `Photon.hpp` is explicitly credited to a colleague ("YANG's
code") and is not mine. The excerpts depend on the surrounding renderer and are not a standalone build.
