# Simulation — Crucible2.5D and Modal2D

## Crucible2.5D — multiphysics laboratory

MPM solids and jellies, SPH liquids and Eulerian air coupled with heat, fracture, magnetism and growing
materials; 2D in principle, 2.5D where extra flexibility helps. Portfolio:
https://lovasz-local-lemma.github.io/projects/physics-2d/index.html

| Read | What it shows |
| --- | --- |
| [`crucible/mpm_p2g.comp`](crucible/mpm_p2g.comp) | The MLS-MPM particle-to-grid kernel: APIC affine update `F = (I + dt·C)F`, closed-form 2×2 SVD with singular-value clamping, fixed-corotated Neo-Hookean stress, Stomakhin snow plasticity, Drucker–Prager sand return mapping in log-strain space, AnisoMPM (Wolper 2020) anisotropic damage with a spectral positive part and non-local Laplacian regularisation, a phase-field brittle branch, a compression barrier, and a 3×3 B-spline fixed-point atomic scatter of mass, momentum and heat. Also engineering under GL constraints: texture-buffer LUTs and `imageBuffer` state to sidestep the 16-SSBO driver cap, and z-occupancy bit-packed into material ids for the 2.5D model. |
| [`crucible/mpm_grid_op.comp`](crucible/mpm_grid_op.comp) | The grid stage between P2G and G2P: fixed-point decode, once-only read-and-clear of the SPH→MPM reaction impulse (with the int32-wrap rationale), 2.5D overhang support, SDF normal-projection boundary conditions and velocity clamp. |
| [`crucible/mpm_damage_gather.comp`](crucible/mpm_damage_gather.comp) | Non-local damage regularisation as a two-pass particle–grid transfer: a weight-normalised damage field scattered with fixed-point atomics, then `∇²d` at each particle from the quadratic B-spline's own second derivative `(1, −2, 1)`, with a free-surface fallback. The comments record the bug hunt (stencil centred on `floor(cell_pos)` vs the scatter's `round`, intact particles must scatter so the denominator is a field). |
| [`crucible/spatial_hash.cpp`](crucible/spatial_hash.cpp) / [`.h`](crucible/spatial_hash.h) | GPU neighbour search as a seven-step counting sort: multiplicative XOR hash into a power-of-two table, atomic count, three-pass work-efficient Blelloch exclusive scan (shared-memory up/down sweep, block sums, add-back), cell ends, atomic-offset scatter. |
| [`crucible/sph_force.comp`](crucible/sph_force.comp) | SPH force/integrate over the hash: symmetric pressure gradient `m_j (p_i/ρ_i² + p_j/ρ_j²)∇W`, Monaghan artificial viscosity, XSPH anti-clustering, CSF surface tension from colour-field gradient and Laplacian with a separate immiscible-interface channel, curl/divergence diagnostics, reciprocal thermal mixing, then an SDF sphere-tracing continuous-collision step with skin, restitution, friction and a re-aimed remainder. |
| [`crucible/sph_codim_detect.comp`](crucible/sph_codim_detect.comp) | Codimensional detection: kernel-weighted covariance of neighbour offsets, closed-form 2×2 eigen-decomposition, eigenvalue ratio as a thread-vs-bulk classifier, principal eigenvector as the local tangent. A small PCA on the GPU. |
| [`crucible/shock_front.cpp`](crucible/shock_front.cpp) / [`.h`](crucible/shock_front.h) | Blast fronts with a real model: analytic 2D Sedov–Taylor radius `r(t) = ξ (E t² / ρ₀)^{1/4}` with a derived transition to constant acoustic speed, retirement by overpressure decay, flood-fill detection of newly opened vents with hysteresis, and a persistent per-particle guard buffer that guarantees one impulse per front. |

## Modal2D — virtual instruments from finite-element vibration modes

Portfolio: https://lovasz-local-lemma.github.io/projects/modal2d/index.html

| Read | What it shows |
| --- | --- |
| [`modal2d/element_shell.cpp`](modal2d/element_shell.cpp) | A Marguerre shallow-shell triangle: CST membrane and DKT/Mindlin bending blocks composed into a 15-DOF element through DOF scatter maps, plus the bending–stretching coupling `K_mb = Bᵐᵀ Dₘ S_z G_w` and the `K_bb_extra` term with its physical justification (without it, curved modes fall below flat-plate modes). Verified: block-diagonal at zero curvature, linear in curvature, domes stiffen. |
| [`modal2d/element_mindlin.cpp`](modal2d/element_mindlin.cpp) | A Mindlin–Reissner thick-plate triangle with 3-point Gauss transverse shear and Lyly–Stenberg–Vihinen stabilisation `h²/(h² + α h_K²)` against shear locking, with the rank argument (pure DSG shear is rank 2, leaving a spurious zero-energy mode) in 100 lines. |
| [`modal2d/eigensolve_sparse.cpp`](modal2d/eigensolve_sparse.cpp) | Shift-invert Lanczos for `Kφ = λMφ` written from scratch: `SimplicialLDLT` on `K − σM`, M-orthonormal Krylov basis with full reorthogonalisation, lucky-breakdown restart with a random M-orthogonal vector, Ritz back-transform `λ = 1/μ + σ`, and regularisation of zero-stiffness/zero-mass DOFs for free-free structures. |
| [`modal2d/sustained_modal_voice.h`](modal2d/sustained_modal_voice.h) | Bowed, rubbed and blown modal synthesis in real time: implicit-midpoint integration of mass-normalised modal ODEs with prewarped decay, a projected Gauss–Seidel sweep over up to 13 friction contacts sharing one body velocity, a bracketed bisection of a regularised Stribeck stick/slip law cached in a 4097-entry LUT, a Chamberlin SVF for noise-band and lip drive, and an explicit passivity bound so stationary contacts cannot pump energy. No allocation in `render_add`. |
