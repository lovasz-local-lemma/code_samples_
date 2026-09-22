// backend/src/contour_trace.cpp
// Implementation of traceDominantContour: binarize -> largest connected
// component (iterative flood fill, 8-connectivity) -> Moore-neighbour boundary
// trace with Jacob's stopping criterion -> Douglas-Peucker -> arc-length
// resample. GL-free, pure std.
#include "curve/contour_trace.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace rive_backend
{
namespace
{
struct IPt { int x, y; };

// Bounds-checked foreground test. Out-of-bounds is always background, so a
// component touching the image border traces cleanly (no special-casing).
inline bool fg(const std::vector<uint8_t>& bin, int w, int h, int x, int y)
{
    if (x < 0 || y < 0 || x >= w || y >= h) return false;
    return bin[(size_t)y * w + x] != 0;
}

// ---- Step 2: largest connected component (iterative flood fill, 8-conn) -----
// Labels every foreground pixel with its component id, returns the id (and via
// out-params the pixel count + a seed pixel) of the largest component. No
// recursion: an explicit stack of pixel indices, so a huge solid region cannot
// blow the call stack. Returns -1 if there is no foreground at all.
int largestComponent(const std::vector<uint8_t>& bin, int w, int h,
                     std::vector<int>& label, IPt& seedOut, int& sizeOut)
{
    label.assign((size_t)w * h, -1);
    std::vector<int> stack;
    stack.reserve(256);

    static const int dx8[8] = {1, -1, 0, 0, 1, 1, -1, -1};
    static const int dy8[8] = {0, 0, 1, -1, 1, -1, 1, -1};

    int bestId = -1, bestSize = 0;          // largest overall (fallback)
    IPt bestSeed{0, 0};
    int bestIntId = -1, bestIntSize = 0;    // largest NOT touching the image border
    IPt bestIntSeed{0, 0};
    int nextId = 0;

    for (int sy = 0; sy < h; ++sy)
    {
        for (int sx = 0; sx < w; ++sx)
        {
            size_t si = (size_t)sy * w + sx;
            if (bin[si] == 0 || label[si] != -1) continue;

            // Flood this component.
            int id = nextId++;
            int count = 0;
            bool touchesBorder = false;
            stack.clear();
            stack.push_back((int)si);
            label[si] = id;
            while (!stack.empty())
            {
                int idx = stack.back();
                stack.pop_back();
                ++count;
                int px = idx % w;
                int py = idx / w;
                if (px == 0 || py == 0 || px == w - 1 || py == h - 1)
                    touchesBorder = true;   // background/frame components touch a border
                for (int k = 0; k < 8; ++k)
                {
                    int nx = px + dx8[k];
                    int ny = py + dy8[k];
                    if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                    size_t ni = (size_t)ny * w + nx;
                    if (bin[ni] == 0 || label[ni] != -1) continue;
                    label[ni] = id;
                    stack.push_back((int)ni);
                }
            }

            if (count > bestSize) { bestSize = count; bestId = id; bestSeed = IPt{sx, sy}; }
            if (!touchesBorder && count > bestIntSize)
            { bestIntSize = count; bestIntId = id; bestIntSeed = IPt{sx, sy}; }
        }
    }

    // Prefer the largest INTERIOR component (so a full-frame background never wins);
    // fall back to the largest overall when everything touches the border.
    if (bestIntId != -1) { seedOut = bestIntSeed; sizeOut = bestIntSize; return bestIntId; }
    seedOut = bestSeed; sizeOut = bestSize; return bestId;
}

// ---- Step 3: Moore-neighbour boundary trace (Jacob's stopping criterion) ----
// Walk the OUTER contour of the component whose pixels are exactly those of
// `bin` masked to the chosen label. We mask once into a fresh buffer so other
// components (and any holes' separate components) can't be stepped onto.
//
// Algorithm: find the start pixel (top-most, then left-most foreground pixel of
// the component) -- it is guaranteed on the outer boundary. From each boundary
// pixel, scan the 8 Moore neighbours clockwise starting just past the cell we
// arrived from (the "backtrack" cell), pick the first foreground neighbour as
// the next boundary pixel. Stop when we re-enter the start pixel FROM the same
// neighbour we first left it from (Jacob's criterion) -- this terminates
// correctly even for one-pixel-wide spurs where the start is revisited from a
// different direction mid-walk. Returns the ordered closed loop (start pixel
// appears once at the front; the loop is implicitly closed by the caller). For
// a single isolated pixel the loop is just that one pixel.
std::vector<IPt> traceBoundary(const std::vector<uint8_t>& mask, int w, int h,
                               IPt start)
{
    std::vector<IPt> contour;

    // Single pixel with no foreground neighbour: degenerate, return it alone.
    auto isFg = [&](int x, int y) { return fg(mask, w, h, x, y); };

    // Clockwise Moore neighbourhood, starting east, going CW:
    //   index: 0:E 1:SE 2:S 3:SW 4:W 5:NW 6:N 7:NE
    static const int mdx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    static const int mdy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    // Given the direction index we ENTERED the current pixel from (i.e. the
    // index, in the CURRENT pixel's neighbourhood, pointing back to where we
    // came from), the clockwise scan should begin at the cell just CW of that
    // backtrack cell. Helper to advance an index clockwise.
    auto cw = [](int i) { return (i + 1) & 7; };

    // Direction from pixel a to neighbour b expressed as a Moore index, or -1.
    auto dirIndex = [&](IPt a, IPt b) -> int {
        int ddx = b.x - a.x, ddy = b.y - a.y;
        for (int k = 0; k < 8; ++k)
            if (mdx[k] == ddx && mdy[k] == ddy) return k;
        return -1;
    };

    // Establish the first move out of `start`. We arrived at `start` from the
    // west (outside the component, since start is the left-most of the top row),
    // so begin the clockwise scan from index pointing west's CW successor. Using
    // the canonical convention: pretend we entered start from the WEST neighbour
    // (index 4), so begin scanning at cw(4)=5 (NW) going clockwise. The first
    // foreground neighbour found is the next contour pixel.
    IPt cur = start;
    contour.push_back(cur);

    // Find the first foreground neighbour to bootstrap direction.
    int firstDir = -1;
    {
        int begin = cw(4); // start scanning just CW of the west backtrack cell
        for (int s = 0; s < 8; ++s)
        {
            int k = (begin + s) & 7;
            int nx = cur.x + mdx[k], ny = cur.y + mdy[k];
            if (isFg(nx, ny)) { firstDir = k; break; }
        }
    }
    if (firstDir < 0)
    {
        // Isolated single pixel: the contour is just the start pixel.
        return contour;
    }

    // Step to the first boundary neighbour.
    IPt prev = cur;
    cur = IPt{cur.x + mdx[firstDir], cur.y + mdy[firstDir]};

    // Remember how we first left the start, for Jacob's stopping test.
    const IPt startPixel = start;
    const int startFirstDir = firstDir;

    // Guard against pathological non-termination (shouldn't happen, but a bad
    // mask must never hang the app). Upper bound: each boundary pixel can be
    // visited at most a constant number of times; 8*area is generous.
    const size_t maxSteps = (size_t)8 * (size_t)w * (size_t)h + 16;
    size_t steps = 0;

    while (steps++ < maxSteps)
    {
        contour.push_back(cur);

        // The backtrack cell in cur's neighbourhood points at `prev`.
        int back = dirIndex(cur, prev);
        if (back < 0) back = 0; // defensive; geometrically always valid
        int begin = cw(back);   // begin clockwise scan just past backtrack

        int nextDir = -1;
        for (int s = 0; s < 8; ++s)
        {
            int k = (begin + s) & 7;
            int nx = cur.x + mdx[k], ny = cur.y + mdy[k];
            if (isFg(nx, ny)) { nextDir = k; break; }
        }

        if (nextDir < 0)
        {
            // cur has no foreground neighbour (isolated after masking) -- stop.
            break;
        }

        IPt nxt{cur.x + mdx[nextDir], cur.y + mdy[nextDir]};

        // Jacob's stopping criterion: we are done when we are about to re-enter
        // the start pixel having left it the same way we originally did. We test
        // it as: current pixel == start AND the move we're about to make equals
        // the first move out of start. (Equivalent classic form: stop when the
        // NEXT pixel is start and we ENTER it identically; we check the move out
        // of start to be robust to thin spurs.)
        if (cur.x == startPixel.x && cur.y == startPixel.y &&
            nextDir == startFirstDir)
        {
            // We've come full circle. Drop the duplicated start we just pushed.
            contour.pop_back();
            break;
        }

        prev = cur;
        cur = nxt;
    }

    return contour;
}

// ---- Step 4: Douglas-Peucker on a closed loop ------------------------------
// Run DP on the loop split at its two farthest-apart vertices so the closed
// contour is simplified as two open polylines (avoids the classic "collapse the
// whole loop" failure when start/end are adjacent). Indices are kept; the
// caller rebuilds the simplified loop in order. eps in pixels.
void dpSegment(const std::vector<IPt>& pts, int lo, int hi, double eps2,
               std::vector<char>& keep)
{
    // Iterative (stack-based) DP to avoid deep recursion on long contours.
    std::vector<std::pair<int, int>> work;
    work.reserve(64);
    work.emplace_back(lo, hi);
    while (!work.empty())
    {
        auto [a, b] = work.back();
        work.pop_back();
        if (b <= a + 1) continue; // nothing between

        double ax = pts[a].x, ay = pts[a].y;
        double bx = pts[b].x, by = pts[b].y;
        double dx = bx - ax, dy = by - ay;
        double seg2 = dx * dx + dy * dy;

        int idxMax = -1;
        double distMax = -1.0;
        for (int i = a + 1; i < b; ++i)
        {
            double px = pts[i].x - ax, py = pts[i].y - ay;
            double d2;
            if (seg2 <= 1e-12)
            {
                // Degenerate segment (a == b in space): point-to-point distance.
                d2 = px * px + py * py;
            }
            else
            {
                double cross = px * dy - py * dx;
                d2 = (cross * cross) / seg2;
            }
            if (d2 > distMax)
            {
                distMax = d2;
                idxMax = i;
            }
        }

        if (idxMax >= 0 && distMax > eps2)
        {
            keep[idxMax] = 1;
            work.emplace_back(a, idxMax);
            work.emplace_back(idxMax, b);
        }
    }
}

std::vector<IPt> douglasPeuckerClosed(const std::vector<IPt>& loop, double epsPx)
{
    int n = (int)loop.size();
    if (n <= 2) return loop;

    // Find the vertex farthest from loop[0] -> gives a stable second split point.
    int far = 0;
    double bestD = -1.0;
    for (int i = 1; i < n; ++i)
    {
        double dx = loop[i].x - loop[0].x;
        double dy = loop[i].y - loop[0].y;
        double d = dx * dx + dy * dy;
        if (d > bestD) { bestD = d; far = i; }
    }
    if (far == 0) far = n / 2; // all coincident-ish; pick the antipode

    double eps2 = (double)epsPx * (double)epsPx;
    std::vector<char> keep(n, 0);
    keep[0] = 1;
    keep[far] = 1;

    // Two open arcs: [0..far] and [far..n-1] + wrap to 0. We handle the wrap by
    // appending index 0 as a virtual endpoint for the second arc.
    dpSegment(loop, 0, far, eps2, keep);

    // Second arc loop[far..n-1, 0]: build a temporary index-mapped run.
    {
        std::vector<IPt> arc;
        arc.reserve(n - far + 1);
        std::vector<int> srcIdx;
        srcIdx.reserve(n - far + 1);
        for (int i = far; i < n; ++i) { arc.push_back(loop[i]); srcIdx.push_back(i); }
        arc.push_back(loop[0]); srcIdx.push_back(0); // wrap endpoint
        std::vector<char> keep2(arc.size(), 0);
        keep2.front() = 1;
        keep2.back() = 1;
        dpSegment(arc, 0, (int)arc.size() - 1, eps2, keep2);
        for (size_t i = 0; i < arc.size(); ++i)
            if (keep2[i]) keep[srcIdx[i]] = 1;
    }

    std::vector<IPt> out;
    out.reserve(n);
    for (int i = 0; i < n; ++i)
        if (keep[i]) out.push_back(loop[i]);
    return out;
}

// ---- Step 5: arc-length resample of a closed loop --------------------------
// `loop` is the simplified closed polygon (last vertex implicitly connects to
// first). Place outN points at equal cumulative-arc-length intervals around the
// total perimeter and return them as normalised coords.
std::vector<float> resampleClosed(const std::vector<IPt>& loop, int w, int h,
                                  int outN)
{
    std::vector<float> out;
    int n = (int)loop.size();
    if (outN <= 0 || n == 0) return out;

    out.reserve((size_t)outN * 2);

    auto emit = [&](double px, double py) {
        out.push_back((float)(px / (double)w));
        out.push_back((float)(py / (double)h));
    };

    if (n == 1)
    {
        // Degenerate: a single point. Emit it outN times so the caller still
        // gets a well-formed 2*outN array.
        for (int i = 0; i < outN; ++i) emit(loop[0].x, loop[0].y);
        return out;
    }

    // Cumulative chord length around the closed loop (segment i: loop[i]->loop[i+1],
    // with the final segment wrapping loop[n-1]->loop[0]).
    std::vector<double> cum(n + 1, 0.0);
    for (int i = 0; i < n; ++i)
    {
        int j = (i + 1) % n;
        double dx = loop[j].x - loop[i].x;
        double dy = loop[j].y - loop[i].y;
        cum[i + 1] = cum[i] + std::sqrt(dx * dx + dy * dy);
    }
    double total = cum[n];

    if (total <= 1e-9)
    {
        // All vertices coincide: emit the single location outN times.
        for (int i = 0; i < outN; ++i) emit(loop[0].x, loop[0].y);
        return out;
    }

    int seg = 0;
    for (int i = 0; i < outN; ++i)
    {
        double target = (total * i) / (double)outN; // [0, total)
        // Advance the segment cursor so cum[seg] <= target < cum[seg+1].
        while (seg < n - 1 && cum[seg + 1] <= target) ++seg;
        double segLen = cum[seg + 1] - cum[seg];
        double t = (segLen > 1e-12) ? (target - cum[seg]) / segLen : 0.0;
        int a = seg, b = (seg + 1) % n;
        double px = loop[a].x + t * (loop[b].x - loop[a].x);
        double py = loop[a].y + t * (loop[b].y - loop[a].y);
        emit(px, py);
    }
    return out;
}
} // namespace

std::vector<float> traceDominantContour(const std::vector<float>& field,
                                        int w, int h, float threshold,
                                        float simplifyEpsPx, int outN)
{
    std::vector<float> empty;
    if (w <= 0 || h <= 0 || outN <= 0) return empty;
    if (field.size() < (size_t)w * (size_t)h) return empty;

    // Step 1: binarize.
    std::vector<uint8_t> bin((size_t)w * h, 0);
    bool anyFg = false;
    for (size_t i = 0; i < (size_t)w * h; ++i)
    {
        if (field[i] >= threshold) { bin[i] = 1; anyFg = true; }
    }
    if (!anyFg) return empty;

    // Step 2: largest connected component.
    std::vector<int> label;
    IPt seed{0, 0};
    int compSize = 0;
    int bestId = largestComponent(bin, w, h, label, seed, compSize);
    if (bestId < 0 || compSize <= 0) return empty;

    // Mask: keep only the chosen component's pixels.
    std::vector<uint8_t> mask((size_t)w * h, 0);
    for (size_t i = 0; i < (size_t)w * h; ++i)
        if (label[i] == bestId) mask[i] = 1;

    // Find the canonical start pixel: top-most row, then left-most column of the
    // masked component. Guaranteed on the outer boundary.
    IPt start{-1, -1};
    for (int y = 0; y < h && start.x < 0; ++y)
        for (int x = 0; x < w; ++x)
            if (mask[(size_t)y * w + x]) { start = IPt{x, y}; break; }
    if (start.x < 0) return empty; // shouldn't happen given compSize > 0

    // Step 3: boundary trace.
    std::vector<IPt> contour = traceBoundary(mask, w, h, start);
    if (contour.empty()) return empty;

    // Step 4: Douglas-Peucker simplification (skip if eps <= 0 or tiny loop).
    std::vector<IPt> simplified;
    if (simplifyEpsPx > 0.0f && contour.size() > 2)
        simplified = douglasPeuckerClosed(contour, simplifyEpsPx);
    else
        simplified = contour;
    if (simplified.empty()) simplified = contour;

    // Step 5: arc-length resample to exactly outN points, normalised.
    return resampleClosed(simplified, w, h, outN);
}

// ---------------------------------------------------------------------------
// Self-test
// ---------------------------------------------------------------------------
int contourTraceSelfTest()
{
    const int W = 64, H = 64;
    const double cx = 31.5, cy = 31.5; // centre between pixels
    const double R = 20.0;             // disk radius in pixels

    std::vector<float> field((size_t)W * H, 0.0f);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
        {
            double dx = (double)x - cx;
            double dy = (double)y - cy;
            if (dx * dx + dy * dy <= R * R)
                field[(size_t)y * W + x] = 1.0f;
        }

    const int outN = 128;
    std::vector<float> poly =
        traceDominantContour(field, W, H, 0.5f, 1.0f, outN);

    if ((int)poly.size() != 2 * outN)
    {
        std::printf("contourTraceSelfTest FAIL: size %d != %d\n",
                    (int)poly.size(), 2 * outN);
        return 1;
    }

    // All coords in [0,1].
    for (size_t i = 0; i < poly.size(); ++i)
    {
        if (!(poly[i] >= 0.0f && poly[i] <= 1.0f))
        {
            std::printf("contourTraceSelfTest FAIL: coord %zu = %f out of [0,1]\n",
                        i, poly[i]);
            return 2;
        }
    }

    // Each resampled point should sit ~ on the circle of radius R/W (x) and R/H
    // (y) about the normalised centre. The traced boundary is the outermost
    // foreground pixel ring, so its radius is ~ R (a fraction of a pixel under
    // R due to the discrete fill); allow a generous tolerance of ~2 px.
    const double ncx = cx / (double)W; // normalised centre
    const double ncy = cy / (double)H;
    const double expectR = R / (double)W; // square field, W==H, so same for y
    const double tolNorm = 2.5 / (double)W; // ~2.5 px tolerance, normalised

    double maxErr = 0.0;
    for (int i = 0; i < outN; ++i)
    {
        double x = poly[2 * i + 0];
        double y = poly[2 * i + 1];
        double dx = x - ncx;
        double dy = y - ncy;
        double r = std::sqrt(dx * dx + dy * dy);
        double err = std::fabs(r - expectR);
        if (err > maxErr) maxErr = err;
    }

    if (maxErr > tolNorm)
    {
        std::printf("contourTraceSelfTest FAIL: max radial err %f > tol %f "
                    "(normalised)\n",
                    maxErr, tolNorm);
        return 3;
    }

    std::printf("contourTraceSelfTest PASS: %d pts, max radial err %.4f "
                "(tol %.4f, normalised)\n",
                outN, maxErr, tolNorm);
    return 0;
}
} // namespace rive_backend
