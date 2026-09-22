// backend/src/contour_trace.hpp
// Grayscale-field -> single ordered, arc-length-resampled polyline. GL-free,
// pure std/cmath, deterministic. Turns an edge / occupancy / mask buffer into
// ONE closed contour for the vector toys (beam, spacetime, .rivx import).
// Self-contained: no project deps beyond <vector>.
#pragma once
#include <vector>

namespace rive_backend
{
// Trace the dominant contour of a grayscale field and resample it to a fixed
// point count. `field` is row-major, size w*h, values ~0..1 (e.g. an edge or
// occupancy buffer). Steps: (1) binarize at `threshold`; (2) find the LARGEST
// connected foreground component; (3) trace its outer boundary as ONE ordered,
// closed loop (Moore-neighbour / Theo Pavlidis boundary following); (4) simplify
// with Douglas-Peucker (epsilon in pixels); (5) resample the simplified loop to
// EXACTLY `outN` points evenly spaced by ARC LENGTH. Returns a flat array
// [x0,y0,x1,y1,...] of length 2*outN in NORMALISED image coords (x = px/w in
// [0,1], y = py/h in [0,1]). Returns an empty vector if no component is found.
std::vector<float> traceDominantContour(const std::vector<float>& field,
                                        int w, int h, float threshold,
                                        float simplifyEpsPx, int outN);

// Self-test: builds a synthetic filled disk in a 64x64 field, runs
// traceDominantContour(..., outN=128) and checks the result is a flat array of
// 256 floats, all in [0,1], with every resampled point lying ~ on the disk's
// circle. Returns 0 on pass, non-zero on the first failing check.
int contourTraceSelfTest();
} // namespace rive_backend
