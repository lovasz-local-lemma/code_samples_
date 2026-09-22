# Light transport, compact

Small enough to read in one sitting, complete enough to be correct: a unidirectional volumetric path
tracer from the light-field project, with the verification harnesses that keep it honest.

| Read | What it shows |
| --- | --- |
| [`bsdf.cpp`](bsdf.cpp) | Visible-normal (Heitz 2018) microfacet sampling shared between isotropic and anisotropic GGX, separable Smith `G1`/`Λ`, an exact standalone `pdf()` kept in lockstep with `sample()` for MIS, Schlick glass with TIR, and `std::variant` + `if constexpr` dispatch in the header. The comments cite measured evidence (21×/29.5× albedo weights at grazing with D-sampling versus the `albedo·G1(ωi) ≤ albedo` bound with VNDF) — fireflies debugged with numbers. |
| [`integrator.h`](integrator.h) | Path tracing with next-event estimation, power-heuristic MIS on both the light-sample and emitter-hit sides (distance accumulated correctly across null-BSDF pass-throughs), `std::optional<Medium>` tracking across boundaries, delta-tracking medium events folded into throughput, and a transmittance-aware shadow-ray walker through null-surface shells. Three real estimator bugs are documented with the symptom each produced. |
| [`medium.cpp`](medium.cpp) | Homogeneous free-flight sampling from the luminance pdf with spectral weight correction (chromatic media), delta/Woodcock tracking for a heterogeneous Perlin-fBm density with an exact majorant from the clamped density range, ratio tracking for unbiased shadow transmittance, Henyey–Greenstein phase sampling. |
| [`verify_bsdfs.cpp`](verify_bsdfs.cpp) | Directional albedo by brute-force quadrature in half-vector *slope* space on a log-polar grid, with the full measure chain written out (`dωi = 4(ωo·h) dωh`, `dωh = cos³ dxs dys`, `dxs dys = αx αy du dv`), so razor-thin anisotropic lobes (`αy = 0.02`) integrate as reliably as wide ones; Monte-Carlo mean / standard error / max weight against it at 4σ; a pdf-lockstep test; an isotropic-reduction cross-check between two independent GGX implementations. |
| [`verify_medium.cpp`](verify_medium.cpp) | Optical depth by a 20k-step reference sum over the exposed density field; asserts ratio-tracking `E[Tr]` and delta-tracking escape frequency reproduce `exp(−τ)`, plus a majorant-domination sweep over 200k random points. |
